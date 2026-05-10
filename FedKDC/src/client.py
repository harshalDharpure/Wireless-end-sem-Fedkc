from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import get_client_loaders_for_task, make_federated_dataset
from .model import create_model, set_parameters


def _parameters_to_tensors(parameters: list[np.ndarray]) -> list[torch.Tensor]:
    return [torch.tensor(p) for p in parameters]


def _tensors_to_parameters(state_dict_items) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in state_dict_items]


# Similarity embedding size used by FedKDC-CL clustering.
_SIM_DIM = 256


def _fixed_dim_sim(vec: np.ndarray, dim: int = _SIM_DIM) -> list[float]:
    """Return a fixed-dim float list (pad/truncate) for Flower metrics transport."""
    x = vec.astype(np.float32).ravel()
    if x.shape[0] >= dim:
        x = x[:dim]
    else:
        x = np.pad(x, (0, dim - x.shape[0]), mode="constant", constant_values=0.0)
    return [float(v) for v in x.tolist()]


def _adaptive_temperature_from_entropy(
    entropy: float,
    base_t: float,
    target_entropy: float,
    t_min: float,
    t_max: float,
) -> float:
    # Increase temperature when entropy is too low (overconfident),
    # decrease when entropy is too high (underconfident).
    # Deterministic and lightweight; keeps behavior reproducible.
    if target_entropy <= 0:
        return float(np.clip(base_t, t_min, t_max))
    ratio = entropy / target_entropy
    t = base_t * (1.0 / max(1e-6, ratio))
    return float(np.clip(t, t_min, t_max))


def _batch_entropy_from_logits(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=1).clamp_min(1e-8)
    ent = (-p * p.log()).sum(dim=1).mean()
    return float(ent.detach().cpu().item())


@dataclass
class ClientConfig:
    batch_size: int
    local_epochs: int
    lr: float
    momentum: float
    weight_decay: float
    max_grad_norm: float

    # KD
    kd_enabled: bool
    kd_epochs: int
    kd_weight: float
    ce_weight: float
    base_temperature: float
    target_entropy: float
    min_temperature: float
    max_temperature: float


class FedKDCClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        dataset: str,
        task_name: str,
        class_subset,
        num_clients: int,
        iid: bool,
        dirichlet_alpha: float,
        val_ratio: float,
        cfg: ClientConfig,
        device: torch.device,
        num_workers: int,
    ) -> None:
        self.cid = int(cid)
        self.dataset = dataset
        self.task_name = task_name
        self.class_subset = class_subset
        self.cfg = cfg
        self.device = device

        self.model = create_model(dataset).to(device)
        self.teacher = create_model(dataset).to(device)

        self.fds = make_federated_dataset(
            dataset=dataset,
            num_partitions=num_clients,
            iid=iid,
            dirichlet_alpha=dirichlet_alpha,
        )
        self.train_loader, self.val_loader = get_client_loaders_for_task(
            fds=self.fds,
            client_id=self.cid,
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            val_ratio=val_ratio,
            class_subset=class_subset,
        )

    def get_parameters(self, config: Dict[str, Any]):
        return _tensors_to_parameters(self.model.state_dict().items())

    def fit(self, parameters, config: Dict[str, Any]):
        try:
            set_parameters(self.model, _parameters_to_tensors(parameters))
        except Exception:
            # Flower may send `None` initial parameters; fall back to local init.
            pass

        mode = config.get("mode", "self")  # self|kd
        teacher_params = config.get("teacher_params", None)
        if teacher_params is None and config.get("teacher_params_pickle") is not None:
            teacher_params = pickle.loads(config["teacher_params_pickle"])
        # Some clients may have zero samples for a given continual task (class-subset filtering).
        # In that case, we skip local training and return current weights with num_examples=0.
        if len(self.train_loader.dataset) == 0:
            metrics = {
                "cid": self.cid,
                "task": self.task_name,
                "sim_vec_json": json.dumps([0.0] * _SIM_DIM),
                "skipped": True,
            }
            return self.get_parameters({}), 0, metrics

        if mode == "kd" and teacher_params is not None and self.cfg.kd_enabled:
            # teacher_params is a flat list of numpy arrays (same structure as model)
            set_parameters(self.teacher, _parameters_to_tensors(teacher_params))
            self._train_kd()
        else:
            self._train_self()

        # Similarity signal for clustering: last-layer weight vector (lightweight and deterministic)
        last = None
        for name, p in self.model.named_parameters():
            if "fc.weight" in name or "classifier.3.weight" in name:
                last = p.detach().flatten().cpu().numpy()
        if last is None:
            # fallback: mean of all params
            vec = np.concatenate([p.detach().flatten().cpu().numpy() for p in self.model.parameters()])
            last = vec[: min(4096, vec.shape[0])]
        metrics = {
            "cid": self.cid,
            "task": self.task_name,
            # Flower RecordDict metrics accept only Scalar types (no lists); embed vector as JSON.
            "sim_vec_json": json.dumps(_fixed_dim_sim(last, _SIM_DIM)),
        }
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config: Dict[str, Any]):
        set_parameters(self.model, _parameters_to_tensors(parameters))
        if len(self.val_loader.dataset) == 0:
            return 0.0, 0, {"accuracy": float("nan")}
        loss, acc = self._eval()
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}

    def _train_self(self) -> None:
        self.model.train()
        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.cfg.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                opt.step()

    def _train_kd(self) -> None:
        self.model.train()
        self.teacher.eval()

        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )
        ce = nn.CrossEntropyLoss()
        kld = nn.KLDivLoss(reduction="batchmean")

        for _ in range(self.cfg.kd_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    t_logits = self.teacher(xb)
                    ent = _batch_entropy_from_logits(t_logits)
                    T = _adaptive_temperature_from_entropy(
                        entropy=ent,
                        base_t=self.cfg.base_temperature,
                        target_entropy=self.cfg.target_entropy,
                        t_min=self.cfg.min_temperature,
                        t_max=self.cfg.max_temperature,
                    )
                    t_probs = F.softmax(t_logits / T, dim=1)

                s_logits = self.model(xb)
                ce_loss = ce(s_logits, yb)
                kd_loss = kld(F.log_softmax(s_logits / T, dim=1), t_probs) * (T * T)

                loss = self.cfg.ce_weight * ce_loss + self.cfg.kd_weight * kd_loss
                loss.backward()
                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                opt.step()

    @torch.no_grad()
    def _eval(self) -> Tuple[float, float]:
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in self.val_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))
        return total_loss / max(1, total), correct / max(1, total)


def client_fn_builder(
    *,
    dataset: str,
    task_name: str,
    class_subset,
    num_clients: int,
    iid: bool,
    dirichlet_alpha: float,
    val_ratio: float,
    client_cfg: ClientConfig,
    device: torch.device,
    num_workers: int,
):
    def client_fn(cid: str) -> fl.client.Client:
        return FedKDCClient(
            cid=cid,
            dataset=dataset,
            task_name=task_name,
            class_subset=class_subset,
            num_clients=num_clients,
            iid=iid,
            dirichlet_alpha=dirichlet_alpha,
            val_ratio=val_ratio,
            cfg=client_cfg,
            device=device,
            num_workers=num_workers,
        ).to_client()

    return client_fn

