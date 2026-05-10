# FedKDC / FedKDC-CL — Project Guide

Simple overview of what this codebase does and how to run experiments end-to-end.

---

## 1. What this project is

This is a **research codebase** for **federated continual learning**:

- **Setting:** Many clients hold local data. The server coordinates training **without** seeing raw data (standard federated learning).
- **Continual twist:** Training happens in **tasks** (e.g. first learn digits 0–4 on MNIST, then digits 5–9). After each task, the code evaluates how well the global model remembers earlier tasks (**forgetting**, **backward transfer**) as well as overall accuracy-style summaries.

Two algorithms are implemented:

| Method | Idea (short) |
|--------|----------------|
| **FedAvg** | Clients train with cross-entropy; server averages their weights. Baseline. |
| **FedKDC-CL** | Adds **clustered knowledge distillation**: clients are grouped by similarity; the server builds cluster-level “teachers” and runs KD phases, plus optional drift-aware scheduling and adaptive KD temperature. |

**Stack:** [Flower](https://flower.ai/) simulation (Ray virtual clients) + PyTorch + YAML configs.

The LaTeX paper in the repo (`main.tex`) describes a related **FLwF-2 vs FedAvg** study and a separate **FedKDC-CL vs FedAvg** MNIST grid; the **code** here is what produces FedKDC-style runs and metrics under `results/`.

---

## 2. Repository layout (what matters)

| Path | Role |
|------|------|
| `configs/` | YAML experiments: `default.yaml` (longer runs), `grid_fast.yaml` (short MNIST grid for sweeps), `smoke.yaml` (tiny sanity check). |
| `src/server.py` | Entry point: reads config, starts simulation, runs continual tasks, saves metrics and plots. |
| `src/client.py` | Flower client: local training loops for FedAvg and FedKDC-CL. |
| `src/fedkdc_cl.py` | FedKDC-CL strategy logic (clustering, KD rounds, drift guard hooks). |
| `src/fedavg.py` | FedAvg strategy glue. |
| `src/model.py` | CNN used for image datasets. |
| `src/data.py` | Datasets + IID / Dirichlet partitioning. |
| `src/utils.py` | Metrics (accuracy per task, forgetting, BWT), plotting, JSON saves. |
| `src/launcher.py` | Optional **grid runner**: loops over client counts and Dirichlet α (and optionally IID). |
| `results/` | **Created automatically**: each run → `results/<timestamp>__seed42/` with `metrics/`, `plots/`, `curves/`. |
| `report/` | Human-written summaries (`EXPERIMENT_RESULTS.md`) and this guide. |
| `main.tex` | Paper source (figures may point under `images/`). |

---

## 3. One-time setup

From the project root (`FedKDC/`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need a machine with **PyTorch** working (CPU is OK for small configs; GPU speeds up server-side evaluation when `device: cuda` in YAML).

---

## 4. Sanity check (fastest run)

Uses tiny MNIST, 2 clients, 1 round — verifies imports, Ray, and data loaders:

```bash
python -m src.server --config configs/smoke.yaml --method both
```

If this finishes and creates a folder under `results/`, your environment is basically OK.

---

## 5. Single experiment (one configuration)

The main command shape is:

```bash
python -m src.server --config <YAML> --method <fedavg|fedkdc-cl|both> [overrides...]
```

**Examples:**

```bash
# Default config: reads configs/default.yaml (many rounds / multi-task stream — heavier)
python -m src.server --config configs/default.yaml --method both

# Same file but only FedAvg
python -m src.server --config configs/default.yaml --method fedavg

# Override client count and non-IID strength (Dirichlet α); FedAvg + FedKDC-CL
python -m src.server --config configs/default.yaml --method both --num-clients 50 --alpha 0.1

# IID partitioning instead of Dirichlet
python -m src.server --config configs/default.yaml --method both --num-clients 100 --iid
```

**What `--method` does:**

- `fedavg` — runs the FedAvg path only.
- `fedkdc-cl` — runs FedKDC-CL only.
- `both` — runs **both** methods in sequence inside the same driver (two result bundles under the same timestamped run folder, as implemented in `server.py`).

**Where outputs go:**  
`results/<YYYYMMDD-HHMMSS>__seed42/` (seed comes from YAML `project.seed`, usually `42`).

Inside each run you typically find:

- **Metrics** (numbers per method / task),
- **Plots** (PNG comparison charts where enabled),
- **Curves** (per-round traces).

---

## 6. Full grid sweep (many experiments in one go)

For systematic comparisons (e.g. paper-style MNIST grid), use **`src/launcher.py`**. It calls `src.server` repeatedly with different `--num-clients`, `--alpha`, and optionally `--iid`.

**Example — FedKDC “fast grid” config** (2 rounds per task, MNIST 0–4 then 5–9), matching the spirit of `configs/grid_fast.yaml`:

```bash
python -m src.launcher \
  --config configs/grid_fast.yaml \
  --method both \
  --clients 10,50,100 \
  --alphas 0.01,0.1,0.5,1.0 \
  --run-iid \
  --continue-on-error
```

This runs:

- For each of `10, 50, 100` clients: every Dirichlet α in `0.01, 0.1, 0.5, 1.0`,
- Then the same client counts with **`--iid`**.

Logs are written under:

`results/grid_launcher__<timestamp>/launcher.log`

**Note:** `launcher.py` sets `CUDA_VISIBLE_DEVICES` in code (default intent: pin to one GPU). If you have no GPU or a different GPU index, edit `src/launcher.py` or set the variable in your shell **before** running so it matches your machine.

---

## 7. How to “run experiments completely” (checklist)

1. **Activate venv** and install deps (`requirements.txt`).
2. Run **`smoke`** once → confirm `results/` appears.
3. Pick a **config**:
   - **`grid_fast.yaml`** — quick MNIST continual grid-friendly runs.
   - **`default.yaml`** — longer / richer continual stream (more time).
4. Either:
   - **One shot:** `python -m src.server --config ... --method both` with optional `--num-clients`, `--alpha`, `--iid`, or  
   - **Full sweep:** `python -m src.launcher --config ... --run-iid ...`.
5. After runs, open **`results/<run>/plots/`** and **`metrics/`**; optionally align numbers with **`report/EXPERIMENT_RESULTS.md`** for the May 2026 grid summary.

---

## 8. Changing behaviour without editing Python

Edit the YAML:

- **`fl.num_clients`** — default client count (CLI `--num-clients` overrides).
- **`fl.num_rounds`** — rounds **per simulation segment** (paired with continual task definitions).
- **`continual.tasks`** — task names, class subsets, rounds per task.
- **`data.iid` / `data.dirichlet_alpha`** — baseline partitioning (CLI `--iid` / `--alpha` override).
- **`fedkdc_cl.*`** — clusters, KD weights, temperature bounds, drift guard.

Always keep **`project.seed`** fixed if you want reproducible comparisons.

---

## 9. Paper vs code

- **`main.tex`** — manuscript (may cite aggregate tables under `tables/` and figures under `images/`).
- **`report/EXPERIMENT_RESULTS.md`** — concise numeric summary for one completed grid campaign.
- **This guide** — how to **produce** those runs; it does not replace reading `configs/*.yaml` for exact hyperparameters.

---

## 10. Common issues

| Symptom | What to try |
|--------|--------------|
| Ray / Flower startup errors | Ensure `flwr[simulation]` installed; close stray Ray processes; use `smoke.yaml` first. |
| CUDA out of memory | Use `grid_fast.yaml` + `runtime.client_torch_device: cpu`, or lower `num_clients` / rounds. |
| Wrong GPU | Set `CUDA_VISIBLE_DEVICES` before launch; adjust `launcher.py` if you use the grid script. |
| Empty or missing plots | Check run finished without traceback; inspect `results/<run>/plots/`. |

---

*Generated as a plain-language companion to `README.md` and the LaTeX/paper materials.*
