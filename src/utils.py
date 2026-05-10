from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Distillation / KL utilities
# -----------------------------


def temperature_log_softmax(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Compute log_softmax(logits / T)."""
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    return F.log_softmax(logits / T, dim=1)


def temperature_softmax(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Compute softmax(logits / T)."""
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    return F.softmax(logits / T, dim=1)


def kl_divergence_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    T: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Knowledge distillation loss using KL divergence with temperature scaling.

    CRITICAL (assignment requirement):
    - Use log_softmax(student/T)
    - Use softmax(teacher/T)
    - Use KLDivLoss correctly

    We also apply the standard (T^2) scaling (Hinton et al.) so gradients are comparable
    across temperatures.
    """
    log_p = temperature_log_softmax(student_logits, T)
    q = temperature_softmax(teacher_logits, T)
    kld = F.kl_div(log_p, q, reduction=reduction)
    return kld * (T * T)


# -----------------------------
# Continual-learning metrics
# -----------------------------


def catastrophic_forgetting(prev_best_acc: float, current_acc: float) -> float:
    """
    Catastrophic forgetting for a task:
      F = max(previous_accuracy_task) - current_accuracy_task
    """
    return float(prev_best_acc - current_acc)


def compute_task_forgetting_from_history(task_acc_history: Sequence[float]) -> float:
    """
    Compute forgetting given a per-round accuracy history for one task.
    Uses the last value as current accuracy and the max before last as prior best.
    Returns 0 if there is insufficient history.
    """
    if len(task_acc_history) < 2:
        return 0.0
    prev_best = float(np.max(task_acc_history[:-1]))
    current = float(task_acc_history[-1])
    return catastrophic_forgetting(prev_best, current)


def backward_transfer(
    acc_task1_after_task1: float,
    acc_task1_after_task2: float,
) -> float:
    """
    Backward Transfer (BWT) for Task 1 after learning Task 2.
    Common definition:
      BWT = A_{1,2} - A_{1,1}
    (negative indicates forgetting).
    """
    return float(acc_task1_after_task2 - acc_task1_after_task1)


def convergence_round(
    acc_per_round: Sequence[float],
    *,
    threshold: float = 0.80,
) -> Optional[int]:
    """
    Return the first (1-indexed) round where accuracy >= threshold, else None.
    """
    for i, acc in enumerate(acc_per_round, start=1):
        if float(acc) >= threshold:
            return i
    return None


# -----------------------------
# Communication cost
# -----------------------------


def model_num_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Compute model size in bytes from a state_dict."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return int(total)


def communication_cost_bytes(
    *,
    model_bytes: int,
    num_clients_sampled: int,
    rounds: int,
    include_server_to_client: bool = True,
    include_client_to_server: bool = True,
) -> int:
    """
    Rough communication cost in bytes across training.

    Assumptions (standard FedAvg accounting):
    - Each sampled client downloads the global model once per round (server->client)
    - Each sampled client uploads updated weights once per round (client->server)
    """
    per_round = 0
    if include_server_to_client:
        per_round += num_clients_sampled * model_bytes
    if include_client_to_server:
        per_round += num_clients_sampled * model_bytes
    return int(per_round * rounds)


# -----------------------------
# Metric containers
# -----------------------------


@dataclass
class RoundMetrics:
    round: int
    global_test_loss: float
    global_test_acc: float
    task_test_acc: Dict[int, float]  # task_id -> acc


class MetricsTracker:
    """
    Small helper to accumulate per-round metrics and derive forgetting/BWT summaries.
    Designed to be used by the server loop.
    """

    def __init__(self) -> None:
        self.rounds: List[int] = []
        self.global_test_loss: List[float] = []
        self.global_test_acc: List[float] = []
        self.task_acc: Dict[int, List[float]] = {}

    def add(self, m: RoundMetrics) -> None:
        self.rounds.append(int(m.round))
        self.global_test_loss.append(float(m.global_test_loss))
        self.global_test_acc.append(float(m.global_test_acc))
        for task_id, acc in m.task_test_acc.items():
            self.task_acc.setdefault(int(task_id), []).append(float(acc))

    def get_task_forgetting_latest(self, task_id: int) -> float:
        hist = self.task_acc.get(task_id, [])
        return compute_task_forgetting_from_history(hist)

    def get_convergence_round(self, threshold: float = 0.80) -> Optional[int]:
        return convergence_round(self.global_test_acc, threshold=threshold)

    def get_bwt_task1(self) -> Optional[float]:
        """
        If at least two task-1 accuracies are present:
        - Treat first recorded task-1 acc as after Task 1
        - Treat last recorded task-1 acc as after Task 2
        """
        hist = self.task_acc.get(1, [])
        if len(hist) < 2:
            return None
        return backward_transfer(hist[0], hist[-1])


