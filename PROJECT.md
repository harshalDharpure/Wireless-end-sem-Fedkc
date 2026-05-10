# Federated Continual Learning Project — Full Guide

This document explains **what this repository does** and **how to run everything end-to-end**, in simple steps.

---

## 1. What is this project?

This is a **research codebase** for **Federated Continual Learning (FCL)**:

- Many **clients** train the same neural network **without sharing raw data**.
- A **server** collects client updates and averages them (**FedAvg**).
- Learning happens in **two sequential tasks** (continual learning): first half of the dataset classes, then the second half — so the model can **forget** the first task when it learns the second.
- We compare:
  - **FedAvg** — standard federated averaging with cross-entropy only.
  - **FLwF-2** — **dual-teacher knowledge distillation**: each client keeps a copy of its previous model and uses the server model as a second teacher, with extra KL-distillation losses (as in the FLwF paper).

**Framework:** [Flower](https://flower.ai/) (`flwr`) + **PyTorch** ≥ 2.0, Python ≥ 3.9.

---

## 2. Folder map (what lives where)

| Path | Purpose |
|------|---------|
| `configs/default.yaml` | Settings for **one** experiment (method, dataset, clients, partition, rounds, GPU). |
| `configs/sweep.yaml` | Settings for the **full grid** of experiments (all methods × datasets × clients × partitions). |
| `src/server.py` | Starts the Flower simulation, FedAvg strategy, server-side evaluation, writes `results/metrics.csv` per run. |
| `src/client.py` | Local training: FedAvg vs FLwF-2 loss, `prev_model` / server teacher handling. |
| `src/model.py` | Small CNN shared by all runs. |
| `src/data.py` | Loads MNIST / FMNIST / CIFAR-10, splits tasks, IID vs Dirichlet partitioning. |
| `src/utils.py` | Distillation math, forgetting, communication cost, metrics helpers. |
| `src/experiments.py` | Runs the **full sweep**, can **resume** skipped runs, optional subprocess isolation. |
| `src/analyze.py` | Builds plots + `summary_table.csv` from aggregated metrics. |
| `scripts/run_sweep_supervised.sh` | Optional: restarts the sweep if Ray crashes (long unattended runs). |
| `results/metrics.csv` | **Main result file**: all rounds × all completed runs. |
| `results/runs/*.csv` | One CSV per experiment combo (plus `runs_index.csv`). |
| `results/plots/*.png` | Figures from `analyze.py`. |
| `main.tex`, `tables/`, `images/` | IEEE-style paper sources (figures copied into `images/`). |

---

## 3. One-time setup (virtual environment)

From the project root (`fed_learn/`):

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Always use `.venv/bin/python` (or `source .venv/bin/activate`) so you get **torch**, **flwr[simulation]** (includes Ray), etc.

---

## 4. Run a **single** experiment (quick test)

1. Edit `configs/default.yaml`:
   - `method`: `fedavg` or `flwf2`
   - `dataset`: `mnist`, `fmnist`, or `cifar10`
   - Under `federated`, set `num_clients` (e.g. `10`)
   - Under `data`, set `partition`: `iid` or `dirichlet`, and `dirichlet_alpha` if needed
   - Under `resources.client`, set `num_cpus` / `num_gpus` (see §7)

2. Run:

```bash
cd /path/to/fed_learn
.venv/bin/python -m src.server --config configs/default.yaml
```

3. Output:
   - `results/metrics.csv` for **that run** (overwrites each time unless you change output paths in YAML).

---

## 5. Run the **complete experiment grid** (90 runs)

The assignment-style grid is:

- **Methods:** FedAvg, FLwF-2  
- **Datasets:** MNIST, FMNIST, CIFAR-10  
- **Clients:** 10, 50, 100  
- **Partitions:** IID + Dirichlet α ∈ {0.01, 0.1, 0.5, 1.0}  

→ **2 × 3 × 3 × 5 = 90** configurations.

### 5.1 Basic command

```bash
cd /path/to/fed_learn
.venv/bin/python -m src.experiments --sweep configs/sweep.yaml
```

- By default **`--resume`** is on: runs whose per-run CSV already exists are **skipped**.
- To force rerunning everything: `--force` (use carefully; deletes that run’s CSV first).

### 5.2 Where results go

| Output | Meaning |
|--------|---------|
| `results/runs/<tag>_<hash>.csv` | Metrics for one combo. |
| `results/runs/runs_index.csv` | Index of completed runs. |
| `results/metrics.csv` | **All runs appended** (deduplicated by `run_id` + `round`). |

### 5.3 GPU / faster or safer runs

Edit `configs/sweep.yaml` → `resources.client`:

- `num_gpus: 1.0` → roughly one full GPU per Ray virtual client (fewer parallel clients, stable).
- `num_gpus: 0.5` → two clients can share one GPU (more parallel, risk of OOM on small GPUs).

Pin visible GPUs so you don’t steal busy devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 .venv/bin/python -u -m src.experiments --sweep configs/sweep.yaml
```

Use `-u` for unbuffered logs if you redirect to a file.

### 5.4 Long unattended sweep (optional supervisor)

If the machine often kills Ray or the process dies overnight:

```bash
chmod +x scripts/run_sweep_supervised.sh
CUDA_VISIBLE_DEVICES=0,1,3,4 nohup bash scripts/run_sweep_supervised.sh > results/logs/supervisor.out 2>&1 &
echo $! > results/supervisor.pid
```

It restarts the sweep script until it prints `Sweep finished:` or hits max attempts. Progress still relies on **resume** (completed combos are not recomputed).

---

## 6. After experiments: plots and summary table

When `results/metrics.csv` contains all runs you care about:

```bash
.venv/bin/python -m src.analyze \
  --metrics results/metrics.csv \
  --out_dir results/plots \
  --summary_csv results/summary_table.csv
```

This writes:

- `results/plots/accuracy_vs_rounds.png`
- `results/plots/loss_vs_rounds.png`
- `results/plots/forgetting_vs_rounds.png`
- `results/plots/iid_vs_noniid_final_accuracy.png`
- `results/plots/fedavg_vs_flwf2_final_accuracy.png`
- `results/summary_table.csv` — one row per run with Method, Dataset, Clients, Rounds, Accuracy, Communication Cost, Forgetting, etc.

For the LaTeX paper, copy plots into `images/`:

```bash
cp results/plots/*.png images/
```

---

## 7. Rebuilding `metrics.csv` from disk (optional)

If old runs exist only as `results/runs/*.csv` but were never merged into `results/metrics.csv`, you can concatenate them with a small Python one-liner or re-run the sweep with `--resume` (it will skip finished tags). A full rebuild-from-runs script can be added if you need it regularly.

---

## 8. Default training settings (assignment-aligned)

From `configs/sweep.yaml` / `default.yaml` (check files for exact values):

| Setting | Typical value |
|---------|----------------|
| Client fraction per round | 0.5 |
| Local epochs | 5 |
| Batch size | 32 |
| Optimizer | SGD, lr 0.01, momentum 0.9 |
| Continual rounds | 10 + 10 (Task 1 + Task 2) |
| FLwF-2 | α_CE 0.001, β_KD 0.7, temperature T = 2 |
| Seeds | 42 (random, numpy, torch) |

---

## 9. Troubleshooting (short)

| Problem | What to try |
|---------|-------------|
| `ModuleNotFoundError` (torch, flwr, ray) | Use `.venv/bin/python`; reinstall `requirements.txt`. |
| CUDA out of memory | Increase `num_gpus` per client in YAML (fewer parallel actors), or reduce concurrent GPUs via `CUDA_VISIBLE_DEVICES`. |
| Sweep stops / Ray errors | Use `scripts/run_sweep_supervised.sh`; check `results/sweep.log` or `results/logs/per_run.log`. |
| Want only partial grid | Temporarily edit lists in `configs/sweep.yaml` (`methods`, `datasets`, `num_clients`, `partitions`). |

---

## 10. Paper (`main.tex`)

- Narrative + `\input{tables/*.tex}` + `\includegraphics{images/*.png}`.
- After updating metrics and plots, regenerate tables if you maintain auto-generated `tables/headline.tex` and `tables/full_results.tex` from the final `metrics.csv`.

---

## Quick reference — minimal “run everything” checklist

```bash
# 1) Setup once
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2) Full sweep (many hours; use GPU env + optional nohup)
CUDA_VISIBLE_DEVICES=0,1,3,4 .venv/bin/python -u -m src.experiments --sweep configs/sweep.yaml

# 3) Analyze
.venv/bin/python -m src.analyze --metrics results/metrics.csv --out_dir results/plots --summary_csv results/summary_table.csv
cp results/plots/*.png images/

# 4) Build PDF (if you have LaTeX installed)
pdflatex main.tex && pdflatex main.tex
```

That’s the complete project flow from code to figures and paper assets.
