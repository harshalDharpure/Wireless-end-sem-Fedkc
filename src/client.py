from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import DatasetName, create_model
from src.utils import kl_divergence_kd_loss


MethodName = Literal["fedavg", "flwf2"]


def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: {len(keys)} != {len(parameters)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


@dataclass(frozen=True)
class ClientHyperParams:
    batch_size: int = 32
    local_epochs: int = 5
    lr: float = 0.01
    momentum: float = 0.9


@dataclass(frozen=True)
class Flwf2Params:
    T: float = 2.0
    alpha_ce: float = 0.001
    beta_kd_client: float = 0.7
    # Remaining weight goes to KD_server: (1 - alpha - beta)


class ContinualFlowerClient(fl.client.NumPyClient):
    """
    Flower client for continual federated learning with two tasks.

    Server contract (via `fit` config dict):
    - method: "fedavg" or "flwf2"
    - task_id: int (1 or 2)
    - task_start: int (0/1) marks first round of that task
    - seed: int (optional) overrides local seed
    """

    def __init__(
        self,
        *,
        cid: int,
        dataset: DatasetName,
        train_loaders: Dict[int, DataLoader],  # task_id -> loader
        device: Optional[torch.device] = None,
        hp: ClientHyperParams = ClientHyperParams(),
        flwf2: Flwf2Params = Flwf2Params(),
        seed: int = 42,
    ) -> None:
        self.cid = int(cid)
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.device = device or get_device()
        self.hp = hp
        self.flwf2 = flwf2
        self.seed = int(seed)

        set_global_seeds(self.seed)

        self.model: nn.Module = create_model(dataset).to(self.device)
        self.prev_model: Optional[nn.Module] = None  # Teacher 1 (client previous task model)

        # Tracks last seen task_id to detect boundaries even if server forgets task_start.
        self._current_task_id: Optional[int] = None

    # -------- Flower hooks --------

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict) -> Tuple[list[np.ndarray], int, dict]:
        # Apply received global parameters
        set_parameters(self.model, parameters)

        method: MethodName = str(config.get("method", "fedavg")).lower()  # type: ignore[assignment]
        task_id = int(config.get("task_id", 1))
        task_start = int(config.get("task_start", 0)) == 1
        task_end = int(config.get("task_end", 0)) == 1
        seed = int(config.get("seed", self.seed))
        set_global_seeds(seed + self.cid)  # slight per-client offset

        if task_id not in self.train_loaders:
            raise ValueError(f"Client {self.cid} missing loader for task_id={task_id}")

        # Detect task boundary (preferred: server sets task_start)
        boundary = task_start or (self._current_task_id is not None and task_id != self._current_task_id)

        # Teacher 2 is always the pre-training server model snapshot for this round
        server_model = copy.deepcopy(self.model).to(self.device)
        for p in server_model.parameters():
            p.requires_grad = False
        server_model.eval()

        # At task boundary: do NOT overwrite prev_model if already set by task_end snapshot.
        # Fallback: if prev_model missing and we move to a new task, snapshot current model.
        if boundary and self._current_task_id is not None and self.prev_model is None:
            self.prev_model = copy.deepcopy(self.model).to(self.device)
            for p in self.prev_model.parameters():
                p.requires_grad = False
            self.prev_model.eval()

        self._current_task_id = task_id

        num_examples = self._train_one_task(
            task_id=task_id,
            method=method,
            server_model=server_model,
        )

        # Preferred: snapshot teacher after completing Task 1 (or end of any task)
        if task_end:
            self.prev_model = copy.deepcopy(self.model).to(self.device)
            for p in self.prev_model.parameters():
                p.requires_grad = False
            self.prev_model.eval()

        return get_parameters(self.model), num_examples, {"cid": self.cid, "task_id": task_id}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        # Evaluation is done on the server with the shared global test set.
        # We still implement this to satisfy Flower interface; returns dummy values.
        set_parameters(self.model, parameters)
        return 0.0, 0, {}

    # -------- Training --------

    def _train_one_task(
        self,
        *,
        task_id: int,
        method: MethodName,
        server_model: nn.Module,
    ) -> int:
        self.model.train()

        loader = self.train_loaders[task_id]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hp.lr,
            momentum=self.hp.momentum,
        )

        total_examples = 0
        for _ in range(self.hp.local_epochs):
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                student_logits = self.model(x)
                ce = criterion(student_logits, y)

                if method == "fedavg" or task_id == 1:
                    loss = ce
                else:
                    # FLwF-2 applies on Task 2: dual-teacher KD.
                    # Teacher 1 (client previous model) might be None if boundary not triggered;
                    # if missing, fall back to using server teacher only.
                    with torch.no_grad():
                        t_server = server_model(x)
                        t_client = self.prev_model(x) if self.prev_model is not None else None

                    kd_server = kl_divergence_kd_loss(student_logits, t_server, T=self.flwf2.T)
                    if t_client is not None:
                        kd_client = kl_divergence_kd_loss(student_logits, t_client, T=self.flwf2.T)
                    else:
                        kd_client = torch.zeros_like(kd_server)

                    alpha = self.flwf2.alpha_ce
                    beta = self.flwf2.beta_kd_client
                    gamma = 1.0 - alpha - beta
                    if gamma < 0:
                        raise ValueError("Invalid FLwF-2 weights: 1 - alpha - beta must be >= 0.")

                    loss = alpha * ce + beta * kd_client + gamma * kd_server

                loss.backward()
                optimizer.step()
                total_examples += int(x.shape[0])

        return total_examples


def create_flower_client_fn(
    *,
    dataset: DatasetName,
    client_loaders: Dict[int, Dict[int, DataLoader]],  # cid -> task_id -> loader
    hp: ClientHyperParams = ClientHyperParams(),
    flwf2: Flwf2Params = Flwf2Params(),
    seed: int = 42,
):
    """
    Factory for Flower simulation:
      flwr.simulation.start_simulation(client_fn=..., num_clients=...)
    """

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        cid = int(context.node_config["partition-id"]) if "partition-id" in context.node_config else int(context.node_id)
        return ContinualFlowerClient(
            cid=cid,
            dataset=dataset,
            train_loaders=client_loaders[cid],
            hp=hp,
            flwf2=flwf2,
            seed=seed,
        ).to_client()

    return client_fn


