from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Image as HFImage

try:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
except Exception as e:  # pragma: no cover
    FederatedDataset = None  # type: ignore
    DirichletPartitioner = None  # type: ignore
    IidPartitioner = None  # type: ignore

from torchvision import transforms
import torchvision.transforms.functional as TF


def _hf_dataset_name(dataset: str) -> str:
    """Map project dataset names to HuggingFace dataset IDs used by flwr-datasets."""
    ds = dataset.lower()
    if ds == "mnist":
        return "mnist"
    if ds in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return "fashion_mnist"
    if ds == "cifar10":
        return "cifar10"
    if ds == "cifar100":
        return "cifar100"
    return dataset


def _raw_image_for_torchvision(img: Any) -> Any:
    """
    flwr-datasets / HF datasets sometimes decode MNIST-like images as nested Python lists.
    torchvision expects PIL or ndarray.
    """

    if isinstance(img, list):
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 1:
            s = int(np.sqrt(arr.shape[0]))
            arr = arr.reshape(s, s)
        return arr
    return img


def _unwrap_hf_image_value(img: Any) -> Any:
    """
    HF datasets formatting sometimes wraps decoded images as length-1 lists, e.g. [PIL.Image].
    """

    if isinstance(img, list) and len(img) == 1:
        inner = img[0]
        # Common HF quirk: decoded PIL wrapped as a length-1 list.
        if not isinstance(inner, list):
            return inner
    return img


def _prepare_image_for_torchvision(img: Any) -> Any:
    """
    Normalize HF image values into something `torchvision` transforms accept reliably.
    """

    img = _unwrap_hf_image_value(img)
    return _raw_image_for_torchvision(img)


def _pil_to_model_tensor(img: Any, *, train: bool, dataset: str, sample_id: int) -> torch.Tensor:
    """
    Convert decoded PIL/ndarray/list images into a CHW float tensor in [0,1].

    We avoid HF `set_transform` + `torchvision.Compose` here because some HF formatting paths
    can produce inconsistent tensor shapes for grayscale images.
    """

    pic = _prepare_image_for_torchvision(img)
    ds = dataset.lower()

    if ds in {"cifar10", "cifar100"} and train:
        # Deterministic "random" augmentations derived from sample_id (stable across runs for a given index).
        pic = TF.pad(pic, 4, padding_mode="reflect")
        max_left = pic.size[0] - 32
        max_top = pic.size[1] - 32
        left = int(sample_id % (max_left + 1))
        top = int((sample_id // (max_left + 1)) % (max_top + 1))
        pic = TF.crop(pic, top, left, 32, 32)
        if (sample_id % 2) == 0:
            pic = TF.hflip(pic)

    t = TF.to_tensor(pic)  # CHW float32 [0,1]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t


@dataclass
class TaskSpec:
    name: str
    dataset: str
    class_subset: Optional[List[int]]
    rounds: int


def _vision_transforms(dataset: str) -> Tuple[Any, Any]:
    ds = dataset.lower()
    if ds in {"mnist", "fmnist"}:
        t = transforms.Compose([transforms.ToTensor()])
        return t, t
    if ds in {"cifar10", "cifar100"}:
        train_t = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_t = transforms.Compose([transforms.ToTensor()])
        return train_t, test_t
    return None, None


def _filter_by_class_subset(batch: Dict[str, Any], class_subset: List[int], label_key: str) -> Dict[str, Any]:
    y = np.array(batch[label_key])
    mask = np.isin(y, np.array(class_subset))
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [vv for vv, m in zip(v, mask) if m]
        else:
            out[k] = v[mask]
    return out


def make_federated_dataset(
    dataset: str,
    num_partitions: int,
    iid: bool,
    dirichlet_alpha: float,
) -> FederatedDataset:
    if FederatedDataset is None:
        raise RuntimeError(
            "flwr-datasets is required. Install with `pip install flwr-datasets`."
        )

    if iid:
        partitioner = IidPartitioner(num_partitions=num_partitions)
    else:
        # Dirichlet over label distribution (non-IID)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            alpha=dirichlet_alpha,
            partition_by="label",
        )

    return FederatedDataset(dataset=_hf_dataset_name(dataset), partitioners={"train": partitioner})


def get_client_loaders_for_task(
    fds: FederatedDataset,
    client_id: int,
    dataset: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    class_subset: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader]:
    partition = fds.load_partition(client_id, split="train")

    ds_lower = dataset.lower()
    label_key = "label"
    image_key = "image"

    # Apply continual task filter (vision datasets)
    if class_subset is not None and ds_lower in {"mnist", "fmnist", "cifar10", "cifar100"}:
        partition = partition.map(
            lambda b: _filter_by_class_subset(b, class_subset, label_key),
            batched=True,
        )
        # After row-wise filtering, re-cast image column back to HF `Image` to keep decode semantics stable.
        if image_key in partition.column_names:
            partition = partition.cast_column(image_key, HFImage())

    # Split train/val deterministically
    n = len(partition)
    n_val = int(n * val_ratio)
    idx = np.arange(n)
    # deterministic split based on client_id to avoid leakage differences
    rng = np.random.default_rng(42 + client_id)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_ds = partition.select(train_idx.tolist())
    val_ds = partition.select(val_idx.tolist())

    # Vision transforms: convert PIL to tensor
    train_t, test_t = _vision_transforms(dataset)
    if train_t is not None:
        def _collate_train(batch):
            xs = torch.stack(
                [
                    _pil_to_model_tensor(
                        b[image_key],
                        train=True,
                        dataset=dataset,
                        sample_id=(client_id * 1_000_003 + i),
                    )
                    for i, b in enumerate(batch)
                ]
            )
            ys = torch.tensor([int(b[label_key]) for b in batch], dtype=torch.long)
            return xs, ys

        def _collate_val(batch):
            xs = torch.stack(
                [
                    _pil_to_model_tensor(b[image_key], train=False, dataset=dataset, sample_id=i)
                    for i, b in enumerate(batch)
                ]
            )
            ys = torch.tensor([int(b[label_key]) for b in batch], dtype=torch.long)
            return xs, ys

        collate_fn_train = _collate_train
        collate_fn_val = _collate_val
    else:
        # Text datasets: minimal tokenization placeholder using HF tokenizers can be added later
        text_key = "text" if "text" in train_ds.column_names else train_ds.column_names[0]
        label_key = "label" if "label" in train_ds.column_names else train_ds.column_names[-1]

        def _collate(batch):
            # Naive whitespace token ids (hash trick) to keep code runnable without heavy preprocessing
            max_len = 64
            vocab_mod = 50000
            xs = []
            ys = []
            for b in batch:
                toks = str(b[text_key]).lower().split()[:max_len]
                ids = [(hash(tok) % (vocab_mod - 1)) + 1 for tok in toks]
                if len(ids) < max_len:
                    ids += [0] * (max_len - len(ids))
                xs.append(torch.tensor(ids, dtype=torch.long))
                ys.append(int(b[label_key]))
            return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

        collate_fn_train = _collate
        collate_fn_val = _collate

    # If a client gets zero samples for the current task (possible under class-subset filtering
    # with Dirichlet partitioning), avoid RandomSampler(shuffle=True) which errors on empty sets.
    train_shuffle = len(train_ds) > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_val,
    )
    return train_loader, val_loader


def get_global_test_loader_for_task(
    dataset: str,
    batch_size: int,
    num_workers: int,
    class_subset: Optional[List[int]] = None,
) -> DataLoader:
    # We use FederatedDataset test split without partitioning (centralized eval).
    # flwr-datasets requires a `partitioners` argument even if we only call `load_split`.
    # Provide a trivial 1-partition partitioner for "train" to satisfy the constructor.
    fds = FederatedDataset(dataset=_hf_dataset_name(dataset), partitioners={"train": 1})
    test_ds = fds.load_split("test")

    ds_lower = dataset.lower()
    label_key = "label"
    image_key = "image"

    if class_subset is not None and ds_lower in {"mnist", "fmnist", "cifar10", "cifar100"}:
        test_ds = test_ds.map(lambda b: _filter_by_class_subset(b, class_subset, label_key), batched=True)
        if image_key in test_ds.column_names:
            test_ds = test_ds.cast_column(image_key, HFImage())

    train_t, test_t = _vision_transforms(dataset)
    if test_t is not None:
        def _collate(batch):
            xs = torch.stack(
                [
                    _pil_to_model_tensor(b[image_key], train=False, dataset=dataset, sample_id=i)
                    for i, b in enumerate(batch)
                ]
            )
            ys = torch.tensor([int(b[label_key]) for b in batch], dtype=torch.long)
            return xs, ys
    else:
        text_key = "text" if "text" in test_ds.column_names else test_ds.column_names[0]
        label_key = "label" if "label" in test_ds.column_names else test_ds.column_names[-1]

        def _collate(batch):
            max_len = 64
            vocab_mod = 50000
            xs = []
            ys = []
            for b in batch:
                toks = str(b[text_key]).lower().split()[:max_len]
                ids = [(hash(tok) % (vocab_mod - 1)) + 1 for tok in toks]
                if len(ids) < max_len:
                    ids += [0] * (max_len - len(ids))
                xs.append(torch.tensor(ids, dtype=torch.long))
                ys.append(int(b[label_key]))
            return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)

