from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from .client import ClientConfig, client_fn_builder
from .data import TaskSpec
from .fedavg import make_fedavg_strategy
from .fedkdc_cl import make_fedkdc_cl_strategy
from .model import create_model, set_parameters
from .utils import (
    ContinualHistory,
    compute_cat10_metrics,
    ensure_dir,
    load_config,
    now_ts,
    resolve_client_train_device,
    resolve_device,
    save_json,
    set_global_seed,
    plot_accuracy_over_rounds,
    plot_accuracy_over_tasks,
    plot_forgetting_curve,
    plot_method_comparison,
)
from .data import get_global_test_loader_for_task


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedKDC-CL Continual Federated Learning (Flower)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--method", type=str, default="both", choices=["fedavg", "fedkdc-cl", "both"])
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--iid", action="store_true")
    p.add_argument("--alpha", type=float, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate_global_on_task(
    *,
    dataset: str,
    class_subset,
    parameters: fl.common.Parameters,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    model = create_model(dataset).to(device)
    ndarrays = fl.common.parameters_to_ndarrays(parameters)
    set_parameters(model, [torch.tensor(a) for a in ndarrays])
    loader = get_global_test_loader_for_task(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        class_subset=class_subset,
    )
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(xb.size(0))
    return correct / max(1, total)


def run_task(
    *,
    method: str,
    task: TaskSpec,
    cfg: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> Tuple[fl.common.Parameters, List[int], List[float]]:
    num_clients = int(cfg["fl"]["num_clients"])
    fraction_fit = float(cfg["fl"]["client_fraction"])
    eval_every = int(cfg["fl"]["eval_every"])
    num_workers = int(cfg["project"]["num_workers"])

    client_cfg = ClientConfig(
        batch_size=int(cfg["client"]["batch_size"]),
        local_epochs=int(cfg["client"]["local_epochs"]),
        lr=float(cfg["client"]["lr"]),
        momentum=float(cfg["client"]["momentum"]),
        weight_decay=float(cfg["client"]["weight_decay"]),
        max_grad_norm=float(cfg["client"]["max_grad_norm"]),
        kd_enabled=bool(cfg["fedkdc_cl"]["kd"]["enabled"]),
        kd_epochs=int(cfg["fedkdc_cl"]["kd"]["kd_epochs"]),
        kd_weight=float(cfg["fedkdc_cl"]["kd"]["kd_weight"]),
        ce_weight=float(cfg["fedkdc_cl"]["kd"]["ce_weight"]),
        base_temperature=float(cfg["fedkdc_cl"]["kd"]["base_temperature"]),
        target_entropy=float(cfg["fedkdc_cl"]["kd"]["target_entropy"]),
        min_temperature=float(cfg["fedkdc_cl"]["kd"]["min_temperature"]),
        max_temperature=float(cfg["fedkdc_cl"]["kd"]["max_temperature"]),
    )

    round_ids: List[int] = []
    round_accs: List[float] = []

    def hook(r: int, a: float) -> None:
        round_ids.append(int(r))
        round_accs.append(float(a))

    # Run evaluation on CPU to avoid GPU OOM in large sweeps
    eval_device = torch.device("cpu")

    if method == "fedavg":
        strategy = make_fedavg_strategy(
            dataset=task.dataset,
            class_subset=task.class_subset,
            device=device,
            eval_device=eval_device,
            batch_size=client_cfg.batch_size,
            num_workers=num_workers,
            fraction_fit=fraction_fit,
            eval_every=eval_every,
            results_hook=hook,
        )
    elif method == "fedkdc-cl":
        strategy = make_fedkdc_cl_strategy(
            dataset=task.dataset,
            class_subset=task.class_subset,
            device=device,
            eval_device=eval_device,
            batch_size=client_cfg.batch_size,
            num_workers=num_workers,
            fraction_fit=fraction_fit,
            eval_every=eval_every,
            num_clusters=int(cfg["fedkdc_cl"]["num_clusters"]),
            promote_every=int(cfg["fedkdc_cl"]["drift_guard"]["promote_every"]),
            kd_enabled=bool(cfg["fedkdc_cl"]["kd"]["enabled"]),
            results_hook=hook,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    client_train_device = resolve_client_train_device(cfg, device)
    client_fn = client_fn_builder(
        dataset=task.dataset,
        task_name=task.name,
        class_subset=task.class_subset,
        num_clients=num_clients,
        iid=bool(cfg["data"]["iid"]),
        dirichlet_alpha=float(cfg["data"]["dirichlet_alpha"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        client_cfg=client_cfg,
        device=client_train_device,
        num_workers=num_workers,
    )

    # Provide initial parameters to avoid client init round failures
    init_model = create_model(task.dataset)
    initial_parameters = fl.common.ndarrays_to_parameters(
        [v.detach().cpu().numpy() for _, v in init_model.state_dict().items()]
    )

    # Simulation
    runtime = cfg.get("runtime", {})
    num_cpus = float(runtime.get("client_num_cpus", 2))
    num_gpus = float(runtime.get("client_num_gpus", 0.25 if device.type == "cuda" else 0))
    ray_init_args = dict(runtime.get("ray_init_args", {"include_dashboard": False, "log_to_driver": False}))

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=int(task.rounds)),
        strategy=strategy,
        # Limit actor pool size to keep simulation stable.
        client_resources={
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        },
        ray_init_args=ray_init_args,
    )

    # Save per-task per-method curve
    save_json(
        {"rounds": round_ids, "accuracy": round_accs},
        run_dir / "curves" / f"{method}__{task.name}__round_accuracy.json",
    )
    plot_accuracy_over_rounds(
        rounds=round_ids,
        accuracies=round_accs,
        out_path=run_dir / "plots" / f"{method}__{task.name}__acc_vs_rounds.png",
        title=f"{method} | {task.name} | Accuracy vs rounds",
    )

    final_params = getattr(strategy, "latest_parameters", None)
    if final_params is None:
        # If all clients failed (e.g., transient Ray issues), fall back to a freshly
        # initialized model so the continual loop can continue and still produce plots/metrics.
        model = create_model(task.dataset)
        final_params = fl.common.ndarrays_to_parameters(
            [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
        )
    return final_params, round_ids, round_accs


def run_continual(
    *,
    method: str,
    cfg: Dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> Tuple[ContinualHistory, Dict[str, Any]]:
    tasks = [
        TaskSpec(
            name=t["name"],
            dataset=t["dataset"],
            class_subset=t.get("class_subset", None),
            rounds=int(t["rounds"]),
        )
        for t in cfg["continual"]["tasks"]
    ]

    # Track acc matrix incrementally: after finishing task i, evaluate on tasks 0..i
    acc_matrix: List[List[float]] = []
    max_acc: List[float] = [0.0 for _ in tasks]
    final_acc: List[float] = [0.0 for _ in tasks]

    params: Optional[fl.common.Parameters] = None

    for i, task in enumerate(tasks):
        params, _, _ = run_task(method=method, task=task, cfg=cfg, run_dir=run_dir, device=device)

        # Evaluate on all seen tasks (0..i) using centralized test split per task
        row = []
        for j in range(i + 1):
            tj = tasks[j]
            acc = evaluate_global_on_task(
                dataset=tj.dataset,
                class_subset=tj.class_subset,
                parameters=params,
                device=torch.device("cpu"),
                batch_size=int(cfg["client"]["batch_size"]),
                num_workers=int(cfg["project"]["num_workers"]),
            )
            row.append(float(acc))
            max_acc[j] = max(max_acc[j], float(acc))
        # pad for readability (unseen tasks)
        while len(row) < len(tasks):
            row.append(float("nan"))
        acc_matrix.append(row)

    # Final task-wise acc is last row (first len(tasks) entries)
    last = acc_matrix[-1]
    for k in range(len(tasks)):
        final_acc[k] = float(last[k])

    history = ContinualHistory(
        task_names=[t.name for t in tasks],
        acc_matrix=acc_matrix,
        max_acc_per_task=max_acc,
        final_acc_per_task=final_acc,
    )
    metrics = compute_cat10_metrics(history)
    save_json(history.to_dict(), run_dir / "metrics" / f"{method}__continual_history.json")
    save_json(metrics, run_dir / "metrics" / f"{method}__cat10_metrics.json")

    # Plots required by prompt
    plot_accuracy_over_tasks(
        task_names=history.task_names,
        final_task_wise_acc=[float(a) for a in history.final_acc_per_task],
        out_path=run_dir / "plots" / f"{method}__acc_vs_tasks.png",
        title=f"{method} | Accuracy vs tasks (final)",
    )
    forgetting = [
        float(m - f)
        for m, f in zip(history.max_acc_per_task, history.final_acc_per_task)
    ]
    plot_forgetting_curve(
        task_names=history.task_names,
        forgetting=forgetting,
        out_path=run_dir / "plots" / f"{method}__forgetting_curve.png",
        title=f"{method} | Forgetting curve",
    )

    return history, metrics


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.num_clients is not None:
        cfg["fl"]["num_clients"] = int(args.num_clients)
    if args.alpha is not None:
        cfg["data"]["dirichlet_alpha"] = float(args.alpha)
        cfg["data"]["iid"] = False
    if args.iid:
        cfg["data"]["iid"] = True

    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    device = resolve_device(str(cfg["project"]["device"]))

    run_dir = ensure_dir(Path(cfg["project"]["results_dir"]) / f"{now_ts()}__seed{seed}")

    methods: List[str]
    if args.method == "both":
        methods = ["fedavg", "fedkdc-cl"]
    else:
        methods = [args.method]

    all_hist: Dict[str, ContinualHistory] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        h, met = run_continual(method=m, cfg=cfg, run_dir=run_dir, device=device)
        all_hist[m] = h
        all_metrics[m] = met

    # Comparison plots (FedAvg vs FedKDC-CL)
    if "fedavg" in all_hist and "fedkdc-cl" in all_hist:
        task_names = all_hist["fedavg"].task_names
        plot_method_comparison(
            task_names=task_names,
            fedavg_final=[float(a) for a in all_hist["fedavg"].final_acc_per_task],
            fedkdc_final=[float(a) for a in all_hist["fedkdc-cl"].final_acc_per_task],
            out_path=run_dir / "plots" / "comparison__fedavg_vs_fedkdccl__acc_vs_tasks.png",
            title="FedAvg vs FedKDC-CL | Accuracy vs tasks (final)",
        )

    save_json({k: v.to_dict() for k, v in all_hist.items()}, run_dir / "metrics" / "all_histories.json")
    save_json(all_metrics, run_dir / "metrics" / "all_cat10_metrics.json")

    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

