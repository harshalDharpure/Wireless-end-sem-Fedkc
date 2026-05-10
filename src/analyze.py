from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _last_round_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Last row per run_id (or synthesize an id if not present)
    if "run_id" not in df.columns:
        # Backward compatibility: server-only metrics.csv has no run_id.
        cols = ["method", "dataset", "num_clients", "partition", "dirichlet_alpha"]
        for c in cols:
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

    idx = df.groupby("run_id")["round"].idxmax()
    out = df.loc[idx].copy()
    return out


def build_summary_table(metrics_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)

    required_cols = {"method", "dataset", "num_clients", "round", "global_test_acc", "communication_cost_bytes_cum", "forgetting_task1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metrics.csv missing columns: {sorted(missing)}")

    last = _last_round_summary(df)

    # Prefer convergence_round_80 from the final row (it’s constant per run once achieved)
    conv = last.get("convergence_round_80", pd.Series([None] * len(last)))

    summary = pd.DataFrame(
        {
            "Method": last["method"],
            "Dataset": last["dataset"],
            "Clients": last["num_clients"],
            "Rounds": last["round"],
            "Accuracy": last["global_test_acc"],
            "Convergence Round": conv,
            "Communication Cost": last["communication_cost_bytes_cum"],
            "Forgetting": last["forgetting_task1"],
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def plot_accuracy_loss(metrics_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    _ensure_dir(out_dir)

    sns.set_theme(style="whitegrid")

    # Accuracy vs rounds
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df,
        x="round",
        y="global_test_acc",
        hue="method",
        style="partition",
        errorbar=None,
    )
    plt.title("Global Test Accuracy vs Rounds")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_rounds.png", dpi=200)
    plt.close()

    # Loss vs rounds
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df,
        x="round",
        y="global_test_loss",
        hue="method",
        style="partition",
        errorbar=None,
    )
    plt.title("Global Test Loss vs Rounds")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_vs_rounds.png", dpi=200)
    plt.close()


def plot_forgetting(metrics_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    _ensure_dir(out_dir)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df,
        x="round",
        y="forgetting_task1",
        hue="method",
        style="partition",
        errorbar=None,
    )
    plt.title("Task 1 Forgetting vs Rounds")
    plt.tight_layout()
    plt.savefig(out_dir / "forgetting_vs_rounds.png", dpi=200)
    plt.close()


def plot_iid_vs_noniid(metrics_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    _ensure_dir(out_dir)
    sns.set_theme(style="whitegrid")

    last = _last_round_summary(df)
    last["partition_label"] = last.apply(
        lambda r: f"dirichlet_a{r['dirichlet_alpha']}" if r["partition"] == "dirichlet" else "iid",
        axis=1,
    )

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=last,
        x="partition_label",
        y="global_test_acc",
        hue="method",
        errorbar=None,
    )
    plt.xticks(rotation=25, ha="right")
    plt.title("IID vs Non-IID (Dirichlet α) — Final Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "iid_vs_noniid_final_accuracy.png", dpi=200)
    plt.close()


def plot_fedavg_vs_flwf2(metrics_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    _ensure_dir(out_dir)
    sns.set_theme(style="whitegrid")

    last = _last_round_summary(df)

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=last,
        x="method",
        y="global_test_acc",
    )
    plt.title("FedAvg vs FLwF-2 — Final Accuracy (all runs)")
    plt.tight_layout()
    plt.savefig(out_dir / "fedavg_vs_flwf2_final_accuracy.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze results/metrics.csv and generate plots + summary table")
    ap.add_argument("--metrics", type=str, default="results/metrics.csv", help="Aggregated metrics CSV")
    ap.add_argument("--out_dir", type=str, default="results/plots", help="Directory to write plots")
    ap.add_argument("--summary_csv", type=str, default="results/summary_table.csv", help="Summary table output path")
    args = ap.parse_args()

    metrics_csv = Path(args.metrics)
    out_dir = Path(args.out_dir)
    summary_csv = Path(args.summary_csv)

    build_summary_table(metrics_csv, summary_csv)
    plot_accuracy_loss(metrics_csv, out_dir)
    plot_forgetting(metrics_csv, out_dir)
    plot_iid_vs_noniid(metrics_csv, out_dir)
    plot_fedavg_vs_flwf2(metrics_csv, out_dir)
    print(f"Wrote plots to {out_dir} and summary to {summary_csv}")


if __name__ == "__main__":
    main()

