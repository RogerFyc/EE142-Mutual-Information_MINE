from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_gaussian_mi(dim: int, rho: float) -> float:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return -0.5 * dim * math.log(1.0 - rho * rho)


@dataclass(frozen=True)
class CorrelatedGaussianSampler:
    dim: int
    rho: float

    @property
    def x_dim(self) -> int:
        return self.dim

    @property
    def y_dim(self) -> int:
        return self.dim

    @property
    def true_mi(self) -> float:
        return true_gaussian_mi(self.dim, self.rho)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(
            batch_size,
            self.dim,
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            batch_size,
            self.dim,
            device=device,
            generator=generator,
        )
        y = self.rho * x + math.sqrt(1.0 - self.rho**2) * noise
        return x, y


class NonlinearNuisanceSampler:
    """Invertible nonlinear signal with independent nuisance dimensions."""

    def __init__(
        self,
        signal_dim: int = 16,
        observed_dim: int = 128,
        rho: float = 0.6,
        transform_seed: int = 1729,
    ) -> None:
        if observed_dim < signal_dim:
            raise ValueError("observed_dim must be at least signal_dim")
        if abs(rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        self.signal_dim = signal_dim
        self.observed_dim = observed_dim
        self.rho = rho
        generator = torch.Generator().manual_seed(transform_seed)
        a = torch.randn(signal_dim, signal_dim, generator=generator)
        b = torch.randn(signal_dim, signal_dim, generator=generator)
        self._a = torch.linalg.qr(a).Q
        self._b = torch.linalg.qr(b).Q

    @property
    def x_dim(self) -> int:
        return self.observed_dim

    @property
    def y_dim(self) -> int:
        return self.observed_dim

    @property
    def true_mi(self) -> float:
        return true_gaussian_mi(self.signal_dim, self.rho)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.randn(
            batch_size,
            self.signal_dim,
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            batch_size,
            self.signal_dim,
            device=device,
            generator=generator,
        )
        v = self.rho * u + math.sqrt(1.0 - self.rho**2) * noise
        a = self._a.to(device=device, dtype=u.dtype)
        b = self._b.to(device=device, dtype=u.dtype)
        x_signal = torch.tanh(u @ a.T)
        y_signal = torch.tanh(v @ b.T)
        nuisance_dim = self.observed_dim - self.signal_dim
        if nuisance_dim == 0:
            return x_signal, y_signal
        x_nuisance = torch.randn(
            batch_size,
            nuisance_dim,
            device=device,
            generator=generator,
        )
        y_nuisance = torch.randn(
            batch_size,
            nuisance_dim,
            device=device,
            generator=generator,
        )
        return (
            torch.cat((x_signal, x_nuisance), dim=1),
            torch.cat((y_signal, y_nuisance), dim=1),
        )

