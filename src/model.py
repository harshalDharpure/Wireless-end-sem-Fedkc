from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


DatasetName = Literal["mnist", "fmnist", "cifar10"]


@dataclass(frozen=True)
class ModelSpec:
    in_channels: int
    num_classes: int = 10


class LightweightCNN(nn.Module):
    """
    Lightweight CNN:
      Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax

    Note:
    - We return logits (no softmax) for numerical stability with CrossEntropyLoss.
    - Distillation will apply (log_)softmax externally with temperature scaling.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

        self.conv1 = nn.Conv2d(spec.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened feature size with a dummy forward for robustness across input sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, spec.in_channels, 32, 32)
            feat = self._forward_features(dummy)
            flat_dim = int(feat.view(1, -1).shape[1])

        self.fc = nn.Linear(flat_dim, spec.num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MNIST/FMIST are 28x28; CIFAR-10 is 32x32. Upsample small inputs to keep a single FC size.
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[-1] != 32 or x.shape[-2] != 32:
            x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def get_model_spec(dataset: DatasetName) -> ModelSpec:
    if dataset in ("mnist", "fmnist"):
        return ModelSpec(in_channels=1, num_classes=10)
    if dataset == "cifar10":
        return ModelSpec(in_channels=3, num_classes=10)
    raise ValueError(f"Unsupported dataset: {dataset}")


def create_model(dataset: DatasetName) -> nn.Module:
    """Factory used by client/server."""
    return LightweightCNN(get_model_spec(dataset))


