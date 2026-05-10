from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from .data import get_global_test_loader_for_task
from .model import create_model, set_parameters


class FedAvgStoreParams(fl.server.strategy.FedAvg):
    """FedAvg which stores the latest aggregated parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latest_parameters: fl.common.Parameters | None = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        params = super().initialize_parameters(client_manager)
        self.latest_parameters = params
        return params

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None
        params, metrics = aggregated
        self.latest_parameters = params
        return params, metrics


def make_fedavg_strategy(
    *,
    dataset: str,
    class_subset,
    device: torch.device,
    eval_device: torch.device,
    batch_size: int,
    num_workers: int,
    fraction_fit: float,
    eval_every: int,
    results_hook: Optional[Callable[[int, float], None]] = None,
) -> fl.server.strategy.Strategy:
    # Centralized evaluation to log accuracy over communication rounds
    test_loader = get_global_test_loader_for_task(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        class_subset=class_subset,
    )
    model = create_model(dataset).to(eval_device)

    # Flower FedAvg will pass `parameters_ndarrays` (List[np.ndarray]) to evaluate_fn.
    def evaluate_fn(server_round: int, parameters_ndarrays: List[np.ndarray], config: Dict[str, Any]):
        if eval_every > 0 and (server_round % eval_every != 0):
            return None
        set_parameters(model, [torch.tensor(a) for a in parameters_ndarrays])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(eval_device), yb.to(eval_device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(xb.size(0))
        acc = correct / max(1, total)
        if results_hook is not None:
            results_hook(server_round, float(acc))
        # Flower expects (loss, metrics)
        return 0.0, {"accuracy": float(acc)}

    # Enforce course FedAvg settings at the strategy layer
    init_model = create_model(dataset)
    init_params = fl.common.ndarrays_to_parameters(
        [v.detach().cpu().numpy() for _, v in init_model.state_dict().items()]
    )

    return FedAvgStoreParams(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
        accept_failures=True,
        initial_parameters=init_params,
        on_fit_config_fn=lambda r: {"mode": "self"},
        evaluate_fn=evaluate_fn,
    )

