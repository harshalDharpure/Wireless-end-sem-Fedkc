from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_client_train_device(cfg: Dict[str, Any], server_device: torch.device) -> torch.device:
    """Torch device for Flower virtual clients (often CPU is more stable under Ray VCE)."""
    mode = str(cfg.get("runtime", {}).get("client_torch_device", "same")).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return server_device


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


@dataclass
class ContinualHistory:
    task_names: List[str]
    # acc[task_i][eval_on_task_j] = accuracy (0..1)
    acc_matrix: List[List[float]]
    # max accuracy achieved on each task so far
    max_acc_per_task: List[float]
    # final acc after last task (filled at end)
    final_acc_per_task: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_names": self.task_names,
            "acc_matrix": self.acc_matrix,
            "max_acc_per_task": self.max_acc_per_task,
            "final_acc_per_task": self.final_acc_per_task,
        }


def compute_cat10_metrics(history: ContinualHistory) -> Dict[str, Any]:
    # Task-wise accuracy is the final row of acc_matrix (after last task), for each task
    if not history.acc_matrix:
        raise ValueError("Empty acc_matrix")

    final_row = history.acc_matrix[-1]
    final_avg_acc = float(np.mean(final_row))

    # Catastrophic forgetting: F_k = max_acc(task_k) - final_acc(task_k)
    final_acc = final_row
    forgetting = [float(m - f) for m, f in zip(history.max_acc_per_task, final_acc)]
    avg_forgetting = float(np.mean(forgetting))

    # Backward transfer (simple): mean over k<last of (final_acc_k - acc_at_end_of_task_k_k)
    # (positive means later tasks helped earlier ones)
    bwt_terms = []
    for k in range(len(history.task_names) - 1):
        acc_end_of_task_k = history.acc_matrix[k][k]
        bwt_terms.append(float(final_acc[k] - acc_end_of_task_k))
    bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    return {
        "task_wise_accuracy_final": {name: float(a) for name, a in zip(history.task_names, final_acc)},
        "final_average_accuracy": final_avg_acc,
        "forgetting_per_task": {name: float(f) for name, f in zip(history.task_names, forgetting)},
        "average_forgetting": avg_forgetting,
        "backward_transfer": bwt,
    }


def plot_accuracy_over_rounds(
    rounds: List[int],
    accuracies: List[float],
    out_path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(rounds, accuracies, linewidth=2)
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_accuracy_over_tasks(
    task_names: List[str],
    final_task_wise_acc: List[float],
    out_path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(task_names))
    plt.bar(xs, final_task_wise_acc)
    plt.xticks(xs, task_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy (final)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_forgetting_curve(
    task_names: List[str],
    forgetting: List[float],
    out_path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(task_names))
    plt.bar(xs, forgetting, color="tab:red")
    plt.xticks(xs, task_names, rotation=30, ha="right")
    plt.ylabel("Forgetting (max - final)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_method_comparison(
    task_names: List[str],
    fedavg_final: List[float],
    fedkdc_final: List[float],
    out_path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(9, 4))
    xs = np.arange(len(task_names))
    w = 0.38
    plt.bar(xs - w / 2, fedavg_final, width=w, label="FedAvg")
    plt.bar(xs + w / 2, fedkdc_final, width=w, label="FedKDC-CL")
    plt.xticks(xs, task_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy (final)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
