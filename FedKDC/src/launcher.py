from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full FedAvg vs FedKDC-CL grid (background-friendly).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to use (default: current).")
    p.add_argument("--clients", type=str, default="10,50,100", help="Comma-separated client counts.")
    p.add_argument("--alphas", type=str, default="0.01,0.1,0.5,1.0", help="Comma-separated Dirichlet alphas.")
    p.add_argument("--run-iid", action="store_true", help="Also run IID setting.")
    p.add_argument("--method", type=str, default="both", choices=["fedavg", "fedkdc-cl", "both"])
    p.add_argument("--continue-on-error", action="store_true", help="Keep going even if a combo fails.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    clients = [int(x.strip()) for x in args.clients.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    log_dir = Path("results") / f"grid_launcher__{_ts()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = log_dir / "launcher.log"
    failures_log = log_dir / "failures.txt"
    failures: list[str] = []

    def run(cmd: list[str]) -> None:
        with launcher_log.open("a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
            f.flush()
            # Pin to a mostly-free GPU (adjust if needed)
            env = dict(**{k: v for k, v in dict(**os.environ).items()})
            env.setdefault("CUDA_VISIBLE_DEVICES", "4")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            rc = p.wait()
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EXIT: {rc}\n")
            f.flush()
            if rc != 0:
                failures.append(f"rc={rc} cmd={' '.join(cmd)}")
                with failures_log.open("a", encoding="utf-8") as ff:
                    ff.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] rc={rc} cmd={' '.join(cmd)}\n")
                if not args.continue_on_error:
                    raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")

    # Non-IID Dirichlet sweep
    for n in clients:
        for a in alphas:
            cmd = [
                args.python,
                "-m",
                "src.server",
                "--config",
                args.config,
                "--method",
                args.method,
                "--num-clients",
                str(n),
                "--alpha",
                str(a),
            ]
            run(cmd)

    # IID sweep (optional)
    if args.run_iid:
        for n in clients:
            cmd = [
                args.python,
                "-m",
                "src.server",
                "--config",
                args.config,
                "--method",
                args.method,
                "--num-clients",
                str(n),
                "--iid",
            ]
            run(cmd)

    if failures:
        print(f"Grid finished with {len(failures)} failures. See: {failures_log}")
    print(f"Grid complete. Launcher log: {launcher_log}")


if __name__ == "__main__":
    main()

