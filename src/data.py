from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DatasetName = Literal["mnist", "fmnist", "cifar10"]


@dataclass(frozen=True)
class TaskSpec:
    """A continual-learning task defined by a set of labels/classes."""

    task_id: int
    classes: Tuple[int, ...]


def set_global_seeds(seed: int = 42) -> None:
    # Reproducibility requirements (also mirrored elsewhere in the project).
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset_num_classes(name: DatasetName) -> int:
    if name in ("mnist", "fmnist", "cifar10"):
        return 10
    raise ValueError(f"Unsupported dataset: {name}")


def get_default_transforms(name: DatasetName) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, test_transform)."""
    if name in ("mnist", "fmnist"):
        train_tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return train_tfms, test_tfms

    if name == "cifar10":
        train_tfms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        return train_tfms, test_tfms

    raise ValueError(f"Unsupported dataset: {name}")


def load_datasets(
    name: DatasetName,
    data_dir: str = "./data",
    *,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load (train_dataset, test_dataset) with standard torchvision datasets."""
    train_tfms, test_tfms = get_default_transforms(name)

    if name == "mnist":
        train_ds = datasets.MNIST(root=data_dir, train=True, download=download, transform=train_tfms)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=download, transform=test_tfms)
        return train_ds, test_ds

    if name == "fmnist":
        train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=train_tfms)
        test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=test_tfms)
        return train_ds, test_ds

    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_tfms)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_tfms)
        return train_ds, test_ds

    raise ValueError(f"Unsupported dataset: {name}")


def _get_targets(dataset: Dataset) -> np.ndarray:
    """Extract labels as a numpy array for common torchvision datasets."""
    # torchvision datasets typically expose targets as list/array-like.
    if hasattr(dataset, "targets"):
        t = getattr(dataset, "targets")
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy().astype(np.int64)
        return np.asarray(t, dtype=np.int64)
    raise ValueError("Dataset does not expose `targets`; unsupported dataset type.")


def make_class_incremental_tasks(num_classes: int) -> List[TaskSpec]:
    """Two tasks: first half classes, then second half classes."""
    if num_classes % 2 != 0:
        raise ValueError("This assignment assumes even number of classes for 2 tasks.")
    half = num_classes // 2
    return [
        TaskSpec(task_id=1, classes=tuple(range(0, half))),
        TaskSpec(task_id=2, classes=tuple(range(half, num_classes))),
    ]


def filter_indices_by_classes(targets: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    classes_arr = np.asarray(classes, dtype=np.int64)
    mask = np.isin(targets, classes_arr)
    return np.nonzero(mask)[0]


def subset_for_task(dataset: Dataset, task: TaskSpec) -> Subset:
    targets = _get_targets(dataset)
    idx = filter_indices_by_classes(targets, task.classes)
    return Subset(dataset, idx.tolist())


def partition_indices_iid(
    indices: np.ndarray,
    num_clients: int,
    *,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """IID partition: shuffle then split as evenly as possible."""
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)

    splits = np.array_split(idx, num_clients)
    return {cid: split.astype(np.int64) for cid, split in enumerate(splits)}


def partition_indices_dirichlet(
    targets: np.ndarray,
    indices: np.ndarray,
    num_clients: int,
    alpha: float,
    *,
    seed: int = 42,
    min_size: int = 10,
) -> Dict[int, np.ndarray]:
    """
    Dirichlet non-IID split (label distribution skew).

    Based on common FL benchmarking practice:
    - For each class k, sample a proportion vector p_k ~ Dir(alpha)
    - Split indices of class k across clients according to p_k
    - Repeat until each client has at least `min_size` samples
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0.")
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0.")

    rng = np.random.default_rng(seed)
    y = targets[indices]
    classes = np.unique(y)

    # Allocate lists then concatenate.
    client_bins: List[List[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        c_idx = indices[y == c]
        c_idx = c_idx.copy()
        rng.shuffle(c_idx)

        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

        # Convert proportions -> split sizes summing to len(c_idx).
        split_points = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        splits = np.split(c_idx, split_points)
        for cid, part in enumerate(splits):
            client_bins[cid].extend(part.tolist())

    # If any client is too small, rebalance by moving samples from large clients.
    # This avoids pathological failures for very small alpha and few classes.
    if min_size > 0:
        sizes = np.array([len(b) for b in client_bins], dtype=np.int64)
        # Fast path: already satisfied
        if sizes.min() < min_size:
            # Work on mutable lists; prefer deterministic moves using rng.
            # Keep transferring until all clients meet min_size or no legal transfer exists.
            safety = 0
            while sizes.min() < min_size and safety < 100000:
                safety += 1
                receiver = int(np.argmin(sizes))
                donor = int(np.argmax(sizes))
                if sizes[donor] <= min_size:
                    break  # no one can donate
                # Move one random sample from donor to receiver
                j = int(rng.integers(low=0, high=len(client_bins[donor])))
                client_bins[receiver].append(client_bins[donor].pop(j))
                sizes[receiver] += 1
                sizes[donor] -= 1

    return {cid: np.asarray(sorted(b), dtype=np.int64) for cid, b in enumerate(client_bins)}


def build_client_task_subsets(
    train_dataset: Dataset,
    *,
    num_clients: int,
    partition: Literal["iid", "dirichlet"],
    dirichlet_alpha: Optional[float] = None,
    seed: int = 42,
    dirichlet_min_size: int = 1,
) -> Tuple[List[TaskSpec], Dict[int, Dict[int, Subset]]]:
    """
    Returns:
    - tasks: [Task 1, Task 2]
    - client_task_subsets: dict[cid][task_id] -> Subset over *that client's* data for that task.

    Notes:
    - Tasks are sequential by construction (caller trains task1 then task2).
    - No mixing: each task partitions only its own class subset.

    Important:
    - We partition *per task* to avoid clients having zero samples for a task under
      highly non-IID settings (which can crash DataLoader(shuffle=True) and break FedAvg
      weighting if sampled clients have zero examples).
    """
    targets = _get_targets(train_dataset)
    tasks = make_class_incremental_tasks(get_dataset_num_classes(_infer_dataset_name(train_dataset)))

    client_task_subsets: Dict[int, Dict[int, Subset]] = {cid: {} for cid in range(num_clients)}

    for task in tasks:
        task_indices = filter_indices_by_classes(targets, task.classes).astype(np.int64)

        if partition == "iid":
            per_task_client_indices = partition_indices_iid(task_indices, num_clients, seed=seed + task.task_id)
        elif partition == "dirichlet":
            if dirichlet_alpha is None:
                raise ValueError("dirichlet_alpha must be provided when partition='dirichlet'.")
            per_task_client_indices = partition_indices_dirichlet(
                targets=targets,
                indices=task_indices,
                num_clients=num_clients,
                alpha=float(dirichlet_alpha),
                seed=seed + task.task_id,
                min_size=int(dirichlet_min_size),
            )
        else:
            raise ValueError(f"Unsupported partition: {partition}")

        for cid, idx in per_task_client_indices.items():
            client_task_subsets[cid][task.task_id] = Subset(train_dataset, idx.tolist())

    return tasks, client_task_subsets


def build_test_task_subsets(test_dataset: Dataset) -> Dict[int, Subset]:
    """Shared global test subsets per task (same for all clients)."""
    tasks = make_class_incremental_tasks(get_dataset_num_classes(_infer_dataset_name(test_dataset)))
    return {task.task_id: subset_for_task(test_dataset, task) for task in tasks}


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    # Deterministic ordering is controlled by seeds + explicit shuffle flags.
    # torch RandomSampler requires num_samples > 0, so disable shuffle for empty datasets.
    if hasattr(dataset, "__len__") and len(dataset) == 0:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _infer_dataset_name(dataset: Dataset) -> DatasetName:
    """
    Infer dataset name from common torchvision dataset classes.
    This avoids threading config everywhere inside this module.
    """
    # Subset wraps the underlying dataset.
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset  # type: ignore[assignment]

    if isinstance(base, datasets.MNIST):
        return "mnist"
    if isinstance(base, datasets.FashionMNIST):
        return "fmnist"
    if isinstance(base, datasets.CIFAR10):
        return "cifar10"
    raise ValueError(f"Unsupported dataset type: {type(base)}")


def build_federated_continual_dataloaders(
    *,
    dataset: DatasetName,
    data_dir: str,
    num_clients: int,
    partition: Literal["iid", "dirichlet"],
    dirichlet_alpha: Optional[float],
    batch_size: int,
    local_seed: int = 42,
) -> Tuple[List[TaskSpec], Dict[int, Dict[int, DataLoader]], Dict[int, DataLoader], Dataset, Dataset]:
    """
    Convenience builder used by client/server orchestration.

    Returns:
    - tasks
    - client_loaders: dict[cid][task_id] -> train DataLoader (client-specific)
    - test_loaders: dict[task_id] -> test DataLoader (global shared)
    - train_dataset, test_dataset (raw for optional extra uses)
    """
    set_global_seeds(local_seed)
    train_ds, test_ds = load_datasets(dataset, data_dir=data_dir, download=True)

    tasks, client_task_subsets = build_client_task_subsets(
        train_ds,
        num_clients=num_clients,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        seed=local_seed,
        dirichlet_min_size=1,
    )

    test_task_subsets = build_test_task_subsets(test_ds)

    client_loaders: Dict[int, Dict[int, DataLoader]] = {}
    for cid, per_task in client_task_subsets.items():
        client_loaders[cid] = {
            task_id: make_dataloader(ds, batch_size=batch_size, shuffle=True)
            for task_id, ds in per_task.items()
        }

    test_loaders: Dict[int, DataLoader] = {
        task_id: make_dataloader(ds, batch_size=batch_size, shuffle=False)
        for task_id, ds in test_task_subsets.items()
    }

    return tasks, client_loaders, test_loaders, train_ds, test_ds


