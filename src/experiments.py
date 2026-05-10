from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import yaml

from src.server import ServerConfig, run_federated_continual_learning


_RESOURCE_OR_PATH_KEYS = {
    "client_num_cpus",
    "client_num_gpus",
    "data_dir",
    "results_dir",
    "runs_subdir",
}


def _stable_run_id(payload: Dict[str, Any]) -> str:
    """
    Stable identifier for a run based on config content.

    Excludes runtime resource and path keys (CPU/GPU allocation, data/result
    directories) so that changing them does not invalidate already-completed
    runs.
    """
    filtered = {k: v for k, v in payload.items() if k not in _RESOURCE_OR_PATH_KEYS}
    b = json.dumps(filtered, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:12]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _iter_runs(sweep: dict) -> Iterable[Dict[str, Any]]:
    seed = int(sweep.get("seed", 42))
    methods = list(sweep["methods"])
    datasets = list(sweep["datasets"])
    num_clients_list = list(sweep["num_clients"])
    partitions = list(sweep["partitions"])

    fed = sweep.get("federated", {}) or {}
    training = sweep.get("training", {}) or {}
    opt = (training.get("optimizer") or {}) if isinstance(training, dict) else {}
    cont = sweep.get("continual", {}) or {}
    data = sweep.get("data", {}) or {}
    resources = sweep.get("resources", {}) or {}
    client_res = resources.get("client", {}) if isinstance(resources, dict) else {}
    out = sweep.get("output", {}) or {}

    for method in methods:
        for dataset in datasets:
            for num_clients in num_clients_list:
                for part in partitions:
                    partition = str(part["name"]).lower()
                    alpha = part.get("dirichlet_alpha", None)
                    yield {
                        "seed": seed,
                        "method": str(method).lower(),
                        "dataset": str(dataset).lower(),
                        "num_clients": int(num_clients),
                        "fraction_fit": float(fed.get("fraction_fit", 0.5)),
                        "local_epochs": int(training.get("local_epochs", 5)),
                        "batch_size": int(training.get("batch_size", 32)),
                        "lr": float(opt.get("lr", 0.01)),
                        "momentum": float(opt.get("momentum", 0.9)),
                        "rounds_task1": int(cont.get("rounds_task1", 10)),
                        "rounds_task2": int(cont.get("rounds_task2", 10)),
                        "partition": partition,
                        "dirichlet_alpha": None if alpha is None else float(alpha),
                        "data_dir": str(data.get("data_dir", "./data")),
                        "results_dir": str(out.get("results_dir", "./results")),
                        "runs_subdir": str(out.get("runs_subdir", "runs")),
                        "client_num_cpus": int(client_res.get("num_cpus", 1)),
                        "client_num_gpus": float(client_res.get("num_gpus", 0.0)),
                    }


def _cfg_from_run(run: Dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        method=run["method"],
        dataset=run["dataset"],
        num_clients=run["num_clients"],
        fraction_fit=run["fraction_fit"],
        local_epochs=run["local_epochs"],
        batch_size=run["batch_size"],
        lr=run["lr"],
        momentum=run["momentum"],
        rounds_task1=run["rounds_task1"],
        rounds_task2=run["rounds_task2"],
        partition=run["partition"],
        dirichlet_alpha=run["dirichlet_alpha"],
        seed=run["seed"],
        data_dir=run["data_dir"],
        results_dir=run["results_dir"],
        client_num_cpus=run["client_num_cpus"],
        client_num_gpus=run["client_num_gpus"],
    )


def _copy_metrics_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def _append_run_index_row(run_index_path: Path, row: dict) -> None:
    run_index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not run_index_path.exists()
    with run_index_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _load_existing_run_ids_from_runs_dir(runs_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    if not runs_dir.exists():
        return ids
    for p in runs_dir.glob("*.csv"):
        if p.name == "runs_index.csv":
            continue
        stem = p.stem
        if "_" not in stem:
            continue
        rid = stem.rsplit("_", 1)[-1]
        if len(rid) == 12 and all(c in "0123456789abcdef" for c in rid):
            ids.add(rid)
    return ids


def _load_existing_tags_from_runs_dir(runs_dir: Path) -> Set[str]:
    """
    Resume helper: identify completed runs by their *combo tag*
    (method_dataset_C{n}_partition[_a{alpha}]) so that pure runtime knobs
    (GPU/CPU allocation, paths) do not force re-execution.
    """
    tags: Set[str] = set()
    if not runs_dir.exists():
        return tags
    for p in runs_dir.glob("*.csv"):
        if p.name == "runs_index.csv":
            continue
        stem = p.stem
        if "_" not in stem:
            continue
        head, rid = stem.rsplit("_", 1)
        if len(rid) == 12 and all(c in "0123456789abcdef" for c in rid):
            tags.add(head)
    return tags


def _dedupe_agg(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return agg_df
    keys = [c for c in ["run_id", "round"] if c in agg_df.columns]
    if keys:
        agg_df = agg_df.drop_duplicates(subset=keys, keep="last")
        agg_df = agg_df.sort_values(by=keys).reset_index(drop=True)
    return agg_df


def _ensure_run_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward compatibility: older metrics.csv files may not include run_id/run_tag.
    """
    if df.empty:
        return df
    df = df.copy()
    if "run_id" not in df.columns:
        for c in ["method", "dataset", "num_clients", "partition", "dirichlet_alpha"]:
            if c not in df.columns:
                df[c] = ""
        df["run_id"] = (
            df["method"].astype(str)
            + "_"
            + df["dataset"].astype(str)
            + "_C"
            + df["num_clients"].astype(str)
            + "_"
            + df["partition"].astype(str)
            + "_a"
            + df["dirichlet_alpha"].astype(str)
        )
    if "run_tag" not in df.columns:
        df["run_tag"] = df["run_id"]
    return df


def _run_single_in_subprocess(run: Dict[str, Any], log_path: Path) -> int:
    """
    Execute exactly one (method, dataset, clients, partition[, alpha]) run in a
    fresh Python subprocess so that Ray, CUDA caches, and dataloader workers
    are released between runs. Returns the subprocess exit code.
    """
    payload = json.dumps(run)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.experiments",
        "--single-run",
        "--run-json",
        payload,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        proc.wait()
        return int(proc.returncode)


def _execute_one_run(run: Dict[str, Any]) -> Path:
    """
    Body of a `--single-run` invocation: run one experiment and return its
    metrics CSV path. Designed to be the only work performed in a subprocess.
    """
    cfg = _cfg_from_run(run)
    _ensure_dir(Path(cfg.results_dir))
    metrics_csv_path = run_federated_continual_learning(cfg)
    return metrics_csv_path


def run_sweep(sweep_yaml: Path, *, resume: bool = True, force: bool = False) -> Path:
    sweep = _load_yaml(sweep_yaml)

    base_results_dir = Path(str(sweep.get("output", {}).get("results_dir", "./results")))
    runs_subdir = str(sweep.get("output", {}).get("runs_subdir", "runs"))
    runs_dir = base_results_dir / runs_subdir
    _ensure_dir(runs_dir)

    run_index_path = runs_dir / "runs_index.csv"
    agg_path = base_results_dir / "metrics.csv"

    existing_run_ids = _load_existing_run_ids_from_runs_dir(runs_dir) if resume else set()
    existing_tags = _load_existing_tags_from_runs_dir(runs_dir) if resume else set()

    if resume and agg_path.exists():
        agg_df = _ensure_run_columns(pd.read_csv(agg_path))
    else:
        agg_df = pd.DataFrame()

    skipped = 0
    ran = 0
    failed = 0

    planned_runs = list(_iter_runs(sweep))
    total = len(planned_runs)

    per_run_log = base_results_dir / "logs" / "per_run.log"

    for run in planned_runs:
        run_id = _stable_run_id({k: v for k, v in run.items() if k not in ("results_dir",)})
        cfg = _cfg_from_run(run)

        tag = f"{cfg.method}_{cfg.dataset}_C{cfg.num_clients}_{cfg.partition}"
        if cfg.partition == "dirichlet":
            tag += f"_a{cfg.dirichlet_alpha}"
        out_metrics = runs_dir / f"{tag}_{run_id}.csv"

        already_done = (
            (run_id in existing_run_ids and out_metrics.exists())
            or (tag in existing_tags)
        )
        if resume and (not force) and already_done:
            skipped += 1
            print(f"[skip {skipped}/{total}] tag={tag}", flush=True)
            continue

        if force and out_metrics.exists():
            out_metrics.unlink()

        _ensure_dir(Path(cfg.results_dir))

        print(f"[run {skipped + ran + failed + 1}/{total}] {tag} ({run_id})", flush=True)

        with per_run_log.open("ab") as logf:
            logf.write(f"\n\n========== {tag} ({run_id}) ==========\n".encode())

        # Retry up to 2 times in case of transient Ray/raylet flakes.
        rc = -1
        for attempt in range(1, 3):
            rc = _run_single_in_subprocess(run, per_run_log)
            if rc == 0 and (
                Path(cfg.results_dir) / "metrics.csv"
            ).exists():
                break
            print(
                f"  [warn] {tag} subprocess exited rc={rc} (attempt {attempt}); "
                f"retrying after cleanup" if attempt == 1 else
                f"  [error] {tag} failed twice (rc={rc}); skipping",
                flush=True,
            )

        produced_metrics = Path(cfg.results_dir) / "metrics.csv"
        if rc != 0 or not produced_metrics.exists():
            failed += 1
            with per_run_log.open("ab") as logf:
                logf.write(f"FAILED rc={rc} tag={tag}\n".encode())
            continue

        _copy_metrics_csv(produced_metrics, out_metrics)

        df = pd.read_csv(out_metrics)
        df["run_id"] = run_id
        df["run_tag"] = tag

        agg_df = pd.concat([agg_df, df], axis=0, ignore_index=True)
        agg_df = _ensure_run_columns(agg_df)
        agg_df = _dedupe_agg(agg_df)
        agg_df.to_csv(agg_path, index=False)

        index_row = {
            "run_id": run_id,
            "run_tag": tag,
            "method": cfg.method,
            "dataset": cfg.dataset,
            "num_clients": cfg.num_clients,
            "partition": cfg.partition,
            "dirichlet_alpha": "" if cfg.dirichlet_alpha is None else cfg.dirichlet_alpha,
            "rounds": cfg.num_rounds,
            "client_num_gpus": cfg.client_num_gpus,
            "metrics_csv": str(out_metrics),
        }
        _append_run_index_row(run_index_path, index_row)

        existing_run_ids.add(run_id)
        existing_tags.add(tag)
        ran += 1

    print(
        f"Sweep finished: planned={total}, ran={ran}, skipped={skipped}, failed={failed}",
        flush=True,
    )
    print(f"Aggregated metrics: {agg_path}")
    print(f"Run index: {run_index_path}")

    return agg_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full experiment sweep (writes results/metrics.csv)")
    ap.add_argument("--sweep", type=str, default="configs/sweep.yaml", help="Sweep YAML path")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip runs whose per-run CSV already exists")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip existing runs")
    ap.add_argument("--force", action="store_true", help="Re-run even if per-run CSV exists (deletes that CSV first)")
    ap.add_argument(
        "--single-run",
        action="store_true",
        help="Internal: execute one (method,dataset,...) run from --run-json and exit",
    )
    ap.add_argument(
        "--run-json",
        type=str,
        default=None,
        help="Internal: JSON-encoded run dict consumed by --single-run",
    )
    args = ap.parse_args()

    if args.single_run:
        if not args.run_json:
            raise SystemExit("--single-run requires --run-json")
        run = json.loads(args.run_json)
        path = _execute_one_run(run)
        print(f"[single-run] wrote {path}", flush=True)
        return

    out = run_sweep(Path(args.sweep), resume=bool(args.resume), force=bool(args.force))
    print(f"Wrote aggregated metrics to {out}")


if __name__ == "__main__":
    main()

