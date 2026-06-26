from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_gaussian_mi(dim: int, rho: float) -> float:
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return -0.5 * dim * math.log(1.0 - rho**2)


def sample_correlated_gaussian(
    batch_size: int,
    dim: int,
    rho: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, dim, device=device)
    eps = torch.randn(batch_size, dim, device=device)
    y = rho * x + math.sqrt(1.0 - rho**2) * eps
    return x, y


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def cubic(x: torch.Tensor) -> torch.Tensor:
    return x**3


def sinusoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


FUNCTIONS: dict[str, TensorFunction] = {
    "identity": identity,
    "cubic": cubic,
    "sinusoid": sinusoid,
}

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.shape[0], num_classes, device=labels.device)
    out[torch.arange(labels.shape[0], device=labels.device), labels.long()] = 1.0
    return out


def gaussian_grid_centers(
    grid_size: int = 5,
    spacing: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    offset = (grid_size - 1) * spacing / 2.0
    centers = [
        (i * spacing - offset, j * spacing - offset)
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    return torch.tensor(centers, dtype=torch.float32, device=device)


def sample_gaussian_grid(
    batch_size: int,
    grid_size: int = 5,
    std: float = 0.05,
    spacing: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = gaussian_grid_centers(grid_size, spacing, device)
    labels = torch.randint(0, centers.shape[0], (batch_size,), device=device)
    samples = centers[labels] + std * torch.randn(batch_size, 2, device=device)
    labels_one_hot = one_hot(labels, centers.shape[0])
    scale = max((grid_size - 1) * spacing / 2.0, 1.0)
    return samples / scale, labels, labels_one_hot


def sample_noisy_function(
    batch_size: int,
    dim: int,
    sigma: float,
    function: TensorFunction,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = 2.0 * torch.rand(batch_size, dim, device=device) - 1.0
    y = function(x) + sigma * torch.randn_like(x)
    return x, y
