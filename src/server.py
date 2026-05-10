from __future__ import annotations

import csv
import os
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
import yaml

from src.client import ClientHyperParams, Flwf2Params, create_flower_client_fn
from src.data import DatasetName, build_federated_continual_dataloaders
from src.model import create_model
from src.utils import MetricsTracker, RoundMetrics, communication_cost_bytes, model_num_bytes


MethodName = Literal["fedavg", "flwf2"]


def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class ServerConfig:
    # Core
    method: MethodName
    dataset: DatasetName
    num_clients: int
    fraction_fit: float
    local_epochs: int
    batch_size: int
    lr: float
    momentum: float
    # Continual learning
    rounds_task1: int
    rounds_task2: int
    # Partitioning
    partition: Literal["iid", "dirichlet"]
    dirichlet_alpha: Optional[float]
    # Misc
    seed: int = 42
    data_dir: str = "./data"
    results_dir: str = "./results"
    client_num_cpus: int = 1
    client_num_gpus: float = 0.0

    @property
    def num_rounds(self) -> int:
        return int(self.rounds_task1 + self.rounds_task2)


def _round_to_task(round_num: int, cfg: ServerConfig) -> Tuple[int, bool, bool]:
    """
    Map global round number (1-indexed) to:
    - task_id (1 or 2)
    - task_start flag (True for first round of task)
    - task_end flag (True for last round of task)
    """
    if round_num <= cfg.rounds_task1:
        task_id = 1
        task_start = round_num == 1
        task_end = round_num == cfg.rounds_task1
        return task_id, task_start, task_end
    # task 2
    r2 = round_num - cfg.rounds_task1
    task_id = 2
    task_start = r2 == 1
    task_end = r2 == cfg.rounds_task2
    return task_id, task_start, task_end


def _evaluate_model(
    model: nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],  # task_id -> loader
    device: torch.device,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    task_acc: Dict[int, float] = {}

    with torch.no_grad():
        for task_id, loader in test_loaders.items():
            task_correct = 0
            task_n = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y).item()
                total_loss += float(loss)
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == y).sum().item())
                n = int(y.shape[0])
                total_n += n
                task_correct += int((pred == y).sum().item())
                task_n += n
            task_acc[int(task_id)] = float(task_correct / max(task_n, 1))

    avg_loss = float(total_loss / max(total_n, 1))
    acc = float(total_correct / max(total_n, 1))
    return avg_loss, acc, task_acc


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_federated_continual_learning(cfg: ServerConfig) -> Path:
    set_global_seeds(cfg.seed)

    # Build client loaders (per-client per-task) and global test loaders (per-task).
    tasks, client_loaders, test_loaders, _, _ = build_federated_continual_dataloaders(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        num_clients=cfg.num_clients,
        partition=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        batch_size=cfg.batch_size,
        local_seed=cfg.seed,
    )
    assert len(tasks) == 2, "Assignment assumes exactly two tasks."

    # Client creation fn
    hp = ClientHyperParams(
        batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs,
        lr=cfg.lr,
        momentum=cfg.momentum,
    )
    client_fn = create_flower_client_fn(
        dataset=cfg.dataset,
        client_loaders=client_loaders,
        hp=hp,
        flwf2=Flwf2Params(),
        seed=cfg.seed,
    )

    # Evaluation function on server (shared test set for all clients)
    #
    # IMPORTANT (simulation stability):
    # When Ray clients reserve GPU memory (`client_num_gpus > 0`), running centralized
    # evaluation on the same default CUDA device can OOM (server + Ray actors contend).
    # For Flower simulation workloads, CPU evaluation is typically safer and still correct.
    if cfg.client_num_gpus > 0 and torch.cuda.is_available():
        eval_device = torch.device("cpu")
    else:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_model = create_model(cfg.dataset).to(eval_device)

    # Rebuild test loaders for eval device (avoid pin_memory on CPU).
    test_loaders_cpu: Dict[int, torch.utils.data.DataLoader] = {}
    for task_id, loader in test_loaders.items():
        ds = loader.dataset
        test_loaders_cpu[int(task_id)] = torch.utils.data.DataLoader(
            ds,
            batch_size=int(loader.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    tracker = MetricsTracker()
    metrics_rows: list[dict] = []

    # Precompute model bytes for communication cost.
    model_bytes = model_num_bytes(eval_model.state_dict())
    sampled_clients = int(np.ceil(cfg.fraction_fit * cfg.num_clients))

    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, _config: dict):
        # Load weights into eval model
        from src.client import set_parameters as _set_params  # avoid circular import at module import

        _set_params(eval_model, list(parameters))
        loss, acc, task_acc = _evaluate_model(eval_model, test_loaders_cpu, eval_device)

        m = RoundMetrics(
            round=int(server_round),
            global_test_loss=float(loss),
            global_test_acc=float(acc),
            task_test_acc={int(k): float(v) for k, v in task_acc.items()},
        )
        tracker.add(m)

        task_id, task_start, task_end = _round_to_task(server_round, cfg)
        # Task-1 forgetting is most meaningful after Task 2 begins, but we log it every round.
        forgetting_t1 = tracker.get_task_forgetting_latest(1)
        forgetting_t2 = tracker.get_task_forgetting_latest(2)
        bwt = tracker.get_bwt_task1()
        conv = tracker.get_convergence_round(threshold=0.80)

        comm = communication_cost_bytes(
            model_bytes=model_bytes,
            num_clients_sampled=sampled_clients,
            rounds=server_round,
        )

        row = {
            "round": int(server_round),
            "method": cfg.method,
            "dataset": cfg.dataset,
            "num_clients": int(cfg.num_clients),
            "partition": cfg.partition,
            "dirichlet_alpha": "" if cfg.dirichlet_alpha is None else float(cfg.dirichlet_alpha),
            "task_id": int(task_id),
            "task_start": int(task_start),
            "task_end": int(task_end),
            "global_test_loss": float(loss),
            "global_test_acc": float(acc),
            "task1_test_acc": float(task_acc.get(1, 0.0)),
            "task2_test_acc": float(task_acc.get(2, 0.0)),
            "forgetting_task1": float(forgetting_t1),
            "forgetting_task2": float(forgetting_t2),
            "bwt_task1": "" if bwt is None else float(bwt),
            "convergence_round_80": "" if conv is None else int(conv),
            "communication_cost_bytes_cum": int(comm),
            "model_bytes": int(model_bytes),
            "sampled_clients_per_round": int(sampled_clients),
        }
        metrics_rows.append(row)

        # Return to Flower (loss, metrics)
        return float(loss), {"accuracy": float(acc), "task1_acc": float(task_acc.get(1, 0.0)), "task2_acc": float(task_acc.get(2, 0.0))}

    def on_fit_config_fn(server_round: int) -> dict:
        task_id, task_start, task_end = _round_to_task(server_round, cfg)
        return {
            "method": cfg.method,
            "task_id": int(task_id),
            "task_start": int(task_start),
            "task_end": int(task_end),
            "seed": int(cfg.seed),
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=float(cfg.fraction_fit),
        fraction_evaluate=0.0,
        min_fit_clients=int(sampled_clients),
        min_evaluate_clients=0,
        min_available_clients=int(cfg.num_clients),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(cfg.num_clients),
        config=fl.server.ServerConfig(num_rounds=int(cfg.num_rounds)),
        strategy=strategy,
        client_resources={"num_cpus": int(cfg.client_num_cpus), "num_gpus": float(cfg.client_num_gpus)},
        ray_init_args={
            "include_dashboard": False,
            "ignore_reinit_error": True,
            "_temp_dir": "/tmp/ray_fedlearn",
            "log_to_driver": False,
        },
    )
    _ = history  # metrics are recorded via evaluate_fn

    out_path = Path(cfg.results_dir) / "metrics.csv"
    _write_metrics_csv(out_path, metrics_rows)
    return out_path


def _parse_env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_config_yaml(path: Path) -> ServerConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    # Minimal validation + mapping to our strongly-typed config.
    method = str(raw["method"]).lower()
    dataset = str(raw["dataset"]).lower()

    fed = raw.get("federated", {})
    training = raw.get("training", {})
    opt = (training.get("optimizer") or {}) if isinstance(training, dict) else {}
    cont = raw.get("continual", {})
    data = raw.get("data", {})
    out = raw.get("output", {})
    resources = raw.get("resources", {}) or {}
    client_res = resources.get("client", {}) if isinstance(resources, dict) else {}

    return ServerConfig(
        method=method,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        num_clients=int(fed.get("num_clients", 10)),
        fraction_fit=float(fed.get("fraction_fit", 0.5)),
        local_epochs=int(training.get("local_epochs", 5)),
        batch_size=int(training.get("batch_size", 32)),
        lr=float(opt.get("lr", 0.01)),
        momentum=float(opt.get("momentum", 0.9)),
        rounds_task1=int(cont.get("rounds_task1", 5)),
        rounds_task2=int(cont.get("rounds_task2", 5)),
        partition=str(data.get("partition", "iid")).lower(),  # type: ignore[arg-type]
        dirichlet_alpha=(None if data.get("dirichlet_alpha", None) is None else float(data["dirichlet_alpha"])),
        seed=int(raw.get("seed", 42)),
        data_dir=str(data.get("data_dir", "./data")),
        results_dir=str(out.get("results_dir", "./results")),
        client_num_cpus=int(client_res.get("num_cpus", 1)),
        client_num_gpus=float(client_res.get("num_gpus", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Continual Learning (Flower + PyTorch)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config_yaml(Path(args.config))
    # Allow env overrides for convenience.
    cfg = ServerConfig(
        method=cfg.method,
        dataset=cfg.dataset,
        num_clients=cfg.num_clients,
        fraction_fit=cfg.fraction_fit,
        local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        momentum=cfg.momentum,
        rounds_task1=cfg.rounds_task1,
        rounds_task2=cfg.rounds_task2,
        partition=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        data_dir=_parse_env_or_default("FEDLEARN_DATA_DIR", cfg.data_dir),
        results_dir=_parse_env_or_default("FEDLEARN_RESULTS_DIR", cfg.results_dir),
        client_num_cpus=cfg.client_num_cpus,
        client_num_gpus=cfg.client_num_gpus,
    )

    out = run_federated_continual_learning(cfg)
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()


