# FedKDC-CL (Flower) — Continual Federated Learning (Cat 10)

This repository adds a **reproducible, modular, research-grade Flower (flwr) codebase** implementing:

- **FedAvg** baseline (strict course settings)
- **FedKDC-CL**: *Federated Knowledge Distillation via Consensus* extended to **continual / task-sequential federated learning**

The implementation is under:

- `configs/`
- `src/`
- `results/` (auto-generated)
- `report/` (place for your writeup)

## Key features (mapped to the prompt)

- **Framework**: Flower simulation API (`flwr.simulation.start_simulation`)
- **Reproducibility**: fixed seeds (random/numpy/torch) = `42`
- **FedAvg baseline (strict)**:
  - client fraction \(=0.5\)
  - local epochs \(=5\)
  - batch size \(=32\)
  - optimizer = SGD(momentum=0.9), lr=0.01
  - loss = CrossEntropy
  - supports 10/50/100 clients
- **Partitioning**: IID or Dirichlet non-IID with \(\alpha \in \{0.01, 0.1, 0.5, 1.0\}\)
- **Datasets**: MNIST, FMNIST, CIFAR10, CIFAR100, Shakespeare, Sent140 via `flwr-datasets`
- **Continual task stream** (task-sequential FL):
  - configured in `configs/default.yaml` (`continual.tasks`)
  - after each task, evaluates on all seen tasks and stores history
- **FedKDC-CL components**:
  - **Self-learning stage**: standard local training (FedAvg-style)
  - **Leader–follower election**: server clusters clients using similarity vectors returned by clients
  - **CKD**: server builds **cluster teachers** (cluster-averaged models) and runs **KD rounds**
  - **DriftGuard**: periodically promotes low-activity clients into leadership
  - **Entropy reducer**: adaptive temperature scaling (client-side) based on teacher prediction entropy
- **Cat 10 metrics**:
  - task-wise accuracy
  - final average accuracy
  - catastrophic forgetting \(F_k = \max acc_k - final\_acc_k\)
  - backward transfer (BWT)
  - plots saved as PNG (300 DPI)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (simulation)

Run both methods (FedAvg + FedKDC-CL) on the continual stream from the config:

```bash
python -m src.server --config configs/default.yaml --method both
```

Override clients and partitioning:

```bash
python -m src.server --config configs/default.yaml --method both --num-clients 50 --alpha 0.1
python -m src.server --config configs/default.yaml --method fedavg --num-clients 100 --iid
```

## Where metrics and plots are produced

- **Cat 10 metrics**: `src/utils.py` (`compute_cat10_metrics`)
- **Continual evaluation loop**: `src/server.py` (`run_continual`)
- **Plots (PNG, 300 DPI)**: `src/utils.py` plotting helpers
- **Output folder**: `results/<timestamp>__seed42/`
  - `metrics/` (JSON)
  - `plots/` (PNG)
  - `curves/` (round-wise JSON)
