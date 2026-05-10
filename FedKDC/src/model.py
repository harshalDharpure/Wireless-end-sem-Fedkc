from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


@dataclass
class ModelSpec:
    dataset: str
    num_classes: int
    input_channels: int


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if in_ch == 1 else 64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class TextBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        # Use last timestep (packed sequences can be added later if needed)
        pooled = out[:, -1, :]
        return self.fc(pooled)


def get_model_spec(dataset: str) -> ModelSpec:
    ds = dataset.lower()
    if ds in {"mnist", "fmnist"}:
        return ModelSpec(dataset=dataset, num_classes=10, input_channels=1)
    if ds == "cifar10":
        return ModelSpec(dataset=dataset, num_classes=10, input_channels=3)
    if ds == "cifar100":
        return ModelSpec(dataset=dataset, num_classes=100, input_channels=3)
    if ds in {"shakespeare", "sent140"}:
        # These are handled with separate NLP model settings; num_classes are defaults
        return ModelSpec(dataset=dataset, num_classes=2, input_channels=0)
    raise ValueError(f"Unknown dataset: {dataset}")


def create_model(dataset: str, nlp_vocab_size: int = 50000) -> nn.Module:
    spec = get_model_spec(dataset)
    ds = dataset.lower()

    if ds in {"mnist", "fmnist"}:
        return SimpleCNN(in_ch=1, num_classes=spec.num_classes)

    if ds in {"cifar10", "cifar100"}:
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, spec.num_classes)
        return m

    if ds in {"sent140"}:
        return TextBiLSTM(vocab_size=nlp_vocab_size, embed_dim=128, hidden_dim=128, num_classes=2)

    if ds in {"shakespeare"}:
        # Character-level next-token prediction; to keep project runnable, we treat it as
        # a classification proxy (2-class) unless user switches task definition.
        return TextBiLSTM(vocab_size=nlp_vocab_size, embed_dim=128, hidden_dim=128, num_classes=2)

    raise ValueError(f"Unknown dataset: {dataset}")


def get_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [val.detach().cpu() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[torch.Tensor]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError("Mismatch in parameter lengths")
    new_state = {k: v.to(dtype=state_dict[k].dtype) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)
