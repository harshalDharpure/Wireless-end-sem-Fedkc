from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from sklearn.cluster import KMeans

from .data import get_global_test_loader_for_task
from .model import create_model, set_parameters


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


@dataclass
class FedKDCClConfig:
    num_clusters: int
    promote_every: int
    kd_enabled: bool


class FedKDCClStrategy(fl.server.strategy.FedAvg):
    """
    FedKDC-CL (research-grade scaffold):
    - Self-learning: standard local training (FedAvg).
    - Leader-follower election: clusters clients using similarity vectors (from client metrics).
    - Clustered Knowledge Distillation (CKD):
        * intra-cluster KD: leaders distill from cluster teacher (cluster-aggregated model).
        * inter-cluster KD: followers optionally distill from global teacher (global model).
    - DriftGuard: periodically promotes low-activity clients into leadership.
    - Entropy reducer: implemented client-side via adaptive temperature.

    Implementation note:
    Flower strategy API is round-based. We approximate two-stage training by alternating rounds:
    odd rounds: self-learning; even rounds: KD fine-tune (leaders get cluster teachers).
    """

    def __init__(
        self,
        *,
        cfg: FedKDCClConfig,
        dataset: str,
        class_subset,
        device: torch.device,
        eval_device: torch.device,
        batch_size: int,
        num_workers: int,
        fraction_fit: float,
        eval_every: int,
        results_hook: Optional[Callable[[int, float], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=2,
            min_available_clients=2,
            **kwargs,
        )
        self.cfg = cfg
        self.dataset = dataset
        self.class_subset = class_subset
        self.device = device
        self.eval_device = eval_device

        self._latest_sim: Dict[int, np.ndarray] = {}
        self._cluster_of: Dict[int, int] = {}
        self._leaders: Dict[int, int] = {}  # cluster -> cid
        self._leader_counts: Dict[int, int] = {}
        self._cluster_teacher_params: Dict[int, List[np.ndarray]] = {}
        self._global_teacher_params: Optional[List[np.ndarray]] = None

        self._eval_every = eval_every
        self._results_hook = results_hook
        self._test_loader = get_global_test_loader_for_task(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            class_subset=class_subset,
        )
        self._eval_model = create_model(dataset).to(eval_device)
        self.latest_parameters: fl.common.Parameters | None = None

    def _is_kd_round(self, server_round: int) -> bool:
        return self.cfg.kd_enabled and (server_round % 2 == 0)

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        if not fit_ins:
            return fit_ins

        kd_round = self._is_kd_round(server_round)
        for client, ins in fit_ins:
            cid = int(client.cid)
            if not kd_round:
                ins.config["mode"] = "self"
                continue

            # KD round: leaders distill from cluster teacher; others distill from global teacher
            ins.config["mode"] = "kd"
            cluster_id = self._cluster_of.get(cid, -1)
            if cluster_id != -1 and self._leaders.get(cluster_id, None) == cid:
                teacher = self._cluster_teacher_params.get(cluster_id, None)
                if teacher is not None:
                    ins.config["teacher_params_pickle"] = pickle.dumps(
                        teacher, protocol=pickle.HIGHEST_PROTOCOL
                    )
            else:
                if self._global_teacher_params is not None:
                    # FitIns.config must use Flower Scalar types only; ndarray lists are not allowed.
                    ins.config["teacher_params_pickle"] = pickle.dumps(
                        self._global_teacher_params, protocol=pickle.HIGHEST_PROTOCOL
                    )

        return fit_ins

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        params, metrics = aggregated
        self.latest_parameters = params
        # When every fit fails, Flower can return `(None, metrics)`; avoid crashing.
        if params is not None:
            self._global_teacher_params = fl.common.parameters_to_ndarrays(params)

        # Update similarity vectors from client metrics (leader-follower election)
        for client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            cid = int(m.get("cid", client_proxy.cid))
            sj = m.get("sim_vec_json")
            if sj is not None:
                self._latest_sim[cid] = np.asarray(json.loads(str(sj)), dtype=np.float32)

        self._elect_leaders_and_teachers(server_round, results)
        return params, metrics

    def _elect_leaders_and_teachers(self, server_round: int, results) -> None:
        if not self._latest_sim:
            return

        cids = sorted(self._latest_sim.keys())
        X = np.stack([self._latest_sim[c] for c in cids], axis=0)

        k = min(self.cfg.num_clusters, len(cids))
        if k <= 1:
            self._cluster_of = {cid: 0 for cid in cids}
            self._leaders = {0: cids[0]}
            return

        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        self._cluster_of = {cid: int(lbl) for cid, lbl in zip(cids, labels)}

        # Choose leader per cluster:
        # - default: closest to cluster centroid (consensus)
        # - DriftGuard: every promote_every rounds, promote lowest-activity client in that cluster
        centroids = km.cluster_centers_
        leaders: Dict[int, int] = {}
        for cluster_id in range(k):
            members = [cid for cid in cids if self._cluster_of[cid] == cluster_id]
            if not members:
                continue

            if self.cfg.promote_every > 0 and (server_round % self.cfg.promote_every == 0):
                # promote low-activity (few times as leader) to mitigate drift / inactivity
                leader = min(members, key=lambda c: self._leader_counts.get(c, 0))
            else:
                centroid = centroids[cluster_id]
                leader = max(members, key=lambda c: _cosine_sim(self._latest_sim[c], centroid))
            leaders[cluster_id] = leader
            self._leader_counts[leader] = self._leader_counts.get(leader, 0) + 1

        self._leaders = leaders

        # Build cluster teachers (CKD): aggregate client models inside each cluster
        # We reuse the already returned parameters from clients; cluster teacher = weighted average.
        # (This is a faithful "consensus teacher" variant, compatible with Flower.)
        cluster_sums: Dict[int, List[np.ndarray]] = {}
        cluster_counts: Dict[int, int] = {}
        for client_proxy, fit_res in results:
            cid = int(client_proxy.cid)
            cluster_id = self._cluster_of.get(cid, None)
            if cluster_id is None:
                continue
            if fit_res.parameters is None:
                continue
            nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
            n = int(fit_res.num_examples)
            if cluster_id not in cluster_sums:
                cluster_sums[cluster_id] = [arr.astype(np.float64) * n for arr in nds]
                cluster_counts[cluster_id] = n
            else:
                cluster_sums[cluster_id] = [s + arr.astype(np.float64) * n for s, arr in zip(cluster_sums[cluster_id], nds)]
                cluster_counts[cluster_id] += n

        self._cluster_teacher_params = {}
        for cluster_id, sums in cluster_sums.items():
            denom = max(1, cluster_counts[cluster_id])
            self._cluster_teacher_params[cluster_id] = [ (s / denom).astype(np.float32) for s in sums ]

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
        if self._eval_every > 0 and (server_round % self._eval_every != 0):
            return None
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        set_parameters(self._eval_model, [torch.tensor(a) for a in ndarrays])
        self._eval_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in self._test_loader:
                xb, yb = xb.to(self.eval_device), yb.to(self.eval_device)
                logits = self._eval_model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(xb.size(0))
        acc = correct / max(1, total)
        if self._results_hook is not None:
            self._results_hook(server_round, float(acc))
        return 0.0, {"accuracy": float(acc)}


def make_fedkdc_cl_strategy(
    *,
    dataset: str,
    class_subset,
    device: torch.device,
    eval_device: torch.device,
    batch_size: int,
    num_workers: int,
    fraction_fit: float,
    eval_every: int,
    num_clusters: int,
    promote_every: int,
    kd_enabled: bool,
    results_hook: Optional[Callable[[int, float], None]] = None,
) -> fl.server.strategy.Strategy:
    cfg = FedKDCClConfig(
        num_clusters=num_clusters,
        promote_every=promote_every,
        kd_enabled=kd_enabled,
    )
    init_model = create_model(dataset)
    init_params = fl.common.ndarrays_to_parameters(
        [v.detach().cpu().numpy() for _, v in init_model.state_dict().items()]
    )
    return FedKDCClStrategy(
        cfg=cfg,
        dataset=dataset,
        class_subset=class_subset,
        device=device,
        eval_device=eval_device,
        batch_size=batch_size,
        num_workers=num_workers,
        fraction_fit=fraction_fit,
        eval_every=eval_every,
        results_hook=results_hook,
        initial_parameters=init_params,
        accept_failures=True,
    )

