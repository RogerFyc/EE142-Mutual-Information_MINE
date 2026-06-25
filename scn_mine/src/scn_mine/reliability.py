from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from .objectives import MIObjective


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float


@torch.no_grad()
def batch_estimates(
    model: nn.Module,
    objective: MIObjective,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[float]:
    model.eval()
    return [objective(model(x, y)).estimate.item() for x, y in batches]


def bootstrap_interval(
    values: list[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> BootstrapInterval:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = min(int(tail * resamples), resamples - 1)
    upper_index = min(int((1.0 - tail) * resamples), resamples - 1)
    return BootstrapInterval(
        mean=float(sum(values) / len(values)),
        lower=float(means[lower_index]),
        upper=float(means[upper_index]),
    )


@torch.no_grad()
def independence_estimate(
    model: nn.Module,
    objective: MIObjective,
    x: torch.Tensor,
    independent_y: torch.Tensor,
) -> float:
    if x.shape[0] != independent_y.shape[0]:
        raise ValueError("independent batches must have the same size")
    return objective(model(x, independent_y)).estimate.item()
