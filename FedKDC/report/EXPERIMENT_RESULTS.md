# FedKDC-CL experiment results summary

Auto-generated summary of completed runs under `results/`. 

## Grid sweep (May 9, 2026)

Config: `configs/grid_fast.yaml` — MNIST continual (2 tasks), 2 rounds/task, fast settings. 
Launcher: `results/grid_launcher__20260509-125559/launcher.log` (15 combos, exit 0).

| # | Clients | Setting | FedAvg / FedKDC-CL (summary metrics) |
|---|---------|---------|----------------------------------------|
| 1 | 10 | Dirichlet α=0.01 | **fedavg**: final_avg_acc=0.1706, avg_forget=0.1993, BWT=-0.3985 · **fedkdc-cl**: final_avg_acc=0.1897, avg_forget=0.0955, BWT=-0.1911 |
| 2 | 10 | Dirichlet α=0.1 | **fedavg**: final_avg_acc=0.2408, avg_forget=0.4774, BWT=-0.9549 · **fedkdc-cl**: final_avg_acc=0.3414, avg_forget=0.4866, BWT=-0.9731 |
| 3 | 10 | Dirichlet α=0.5 | **fedavg**: final_avg_acc=0.4200, avg_forget=0.4805, BWT=-0.9611 · **fedkdc-cl**: final_avg_acc=0.4278, avg_forget=0.4869, BWT=-0.9737 |
| 4 | 10 | Dirichlet α=1.0 | **fedavg**: final_avg_acc=0.4742, avg_forget=0.4927, BWT=-0.9854 · **fedkdc-cl**: final_avg_acc=0.4638, avg_forget=0.4910, BWT=-0.9819 |
| 5 | 50 | Dirichlet α=0.01 | **fedavg**: final_avg_acc=0.0964, avg_forget=0.0392, BWT=-0.0784 · **fedkdc-cl**: final_avg_acc=0.1004, avg_forget=0.0000, BWT=0.1868 |
| 6 | 50 | Dirichlet α=0.1 | **fedavg**: final_avg_acc=0.2330, avg_forget=0.2435, BWT=-0.4871 · **fedkdc-cl**: final_avg_acc=0.2363, avg_forget=0.2972, BWT=-0.5945 |
| 7 | 50 | Dirichlet α=0.5 | **fedavg**: final_avg_acc=0.4416, avg_forget=0.4606, BWT=-0.9212 · **fedkdc-cl**: final_avg_acc=0.4200, avg_forget=0.4648, BWT=-0.9296 |
| 8 | 50 | Dirichlet α=1.0 | **fedavg**: final_avg_acc=0.4356, avg_forget=0.4784, BWT=-0.9568 · **fedkdc-cl**: final_avg_acc=0.4284, avg_forget=0.4560, BWT=-0.9120 |
| 9 | 100 | Dirichlet α=0.01 | **fedavg**: final_avg_acc=0.0964, avg_forget=0.0392, BWT=-0.0784 · **fedkdc-cl**: final_avg_acc=0.1004, avg_forget=0.0000, BWT=0.1868 |
| 10 | 100 | Dirichlet α=0.1 | **fedavg**: final_avg_acc=0.2696, avg_forget=0.1242, BWT=-0.2485 · **fedkdc-cl**: final_avg_acc=0.1324, avg_forget=0.2729, BWT=-0.5458 |
| 11 | 100 | Dirichlet α=0.5 | **fedavg**: final_avg_acc=0.2637, avg_forget=0.4419, BWT=-0.8838 · **fedkdc-cl**: final_avg_acc=0.3110, avg_forget=0.3793, BWT=-0.7585 |
| 12 | 100 | Dirichlet α=1.0 | **fedavg**: final_avg_acc=0.3261, avg_forget=0.4313, BWT=-0.8626 · **fedkdc-cl**: final_avg_acc=0.3034, avg_forget=0.3709, BWT=-0.7418 |
| 13 | 10 | IID | **fedavg**: final_avg_acc=0.4853, avg_forget=0.4916, BWT=-0.9833 · **fedkdc-cl**: final_avg_acc=0.4770, avg_forget=0.4891, BWT=-0.9782 |
| 14 | 50 | IID | **fedavg**: final_avg_acc=0.4342, avg_forget=0.4781, BWT=-0.9562 · **fedkdc-cl**: final_avg_acc=0.3912, avg_forget=0.4674, BWT=-0.9348 |
| 15 | 100 | IID | **fedavg**: final_avg_acc=0.3548, avg_forget=0.4466, BWT=-0.8932 · **fedkdc-cl**: final_avg_acc=0.3038, avg_forget=0.3871, BWT=-0.7743 |

## Other pilot runs

(Not part of the May 9 grid sweep log mapping.)

| Pilot | Notes | Metrics snippet |
|-------|-------|-----------------|
| P1 | Earlier sanity / submit-fast style run | **fedavg**: final_avg_acc=0.0964, avg_forget=0.0392, BWT=-0.0784 · **fedkdc-cl**: final_avg_acc=0.1004, avg_forget=0.0000, BWT=0.1868 |
| P2 | Earlier sanity / submit-fast style run | **fedavg**: final_avg_acc=0.1988, avg_forget=0.4896, BWT=-0.9792 · **fedkdc-cl**: final_avg_acc=0.1800, avg_forget=0.4929, BWT=-0.9858 |
| P3 | Earlier sanity / submit-fast style run | **fedkdc-cl**: final_avg_acc=0.1654, avg_forget=0.1004, BWT=-0.2008 |

## Artefacts per complete run

Each run folder typically stores aggregated metric summaries, round histories,
comparison plots, and per-task learning curves.

— Generated from `/DATA/vaneet_2221cs15/fkdc/FedKDC`
