from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .model import MINE


BatchSampler = Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainResult:
    final_mi: float
    steps: list[int]
    estimates: list[float]
    losses: list[float]


def train_mine(
    mine: MINE,
    sampler: BatchSampler,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    eval_every: int = 100,
    eval_batches: int = 4,
    clip_grad_norm: float | None = None,
) -> TrainResult:
    mine.to(device)
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)
    recorded_steps: list[int] = []
    estimates: list[float] = []
    losses: list[float] = []

    for step in range(1, steps + 1):
        x_all, y_all = sampler(2 * batch_size, device)
        x = x_all[:batch_size]
        y = y_all[:batch_size]
        y_marginal = y_all[batch_size:]
        optimizer.zero_grad(set_to_none=True)
        loss = mine(x, y, y_marginal=y_marginal)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(mine.parameters(), clip_grad_norm)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            mi_values = []
            for _ in range(eval_batches):
                x_all, y_all = sampler(2 * batch_size, device)
                x_eval = x_all[:batch_size]
                y_eval = y_all[:batch_size]
                y_marginal = y_all[batch_size:]
                mi_values.append(
                    mine.estimate(
                        x_eval, y_eval, y_marginal=y_marginal
                    ).item()
                )
            recorded_steps.append(step)
            estimates.append(float(sum(mi_values) / len(mi_values)))
            losses.append(float(loss.item()))
            print(
                f"step={step:5d} loss={loss.item(): .4f} "
                f"mi_estimate={estimates[-1]: .4f}"
            )

    return TrainResult(
        final_mi=estimates[-1],
        steps=recorded_steps,
        estimates=estimates,
        losses=losses,
    )


def train_mine_reference(
    mine: MINE,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> TrainResult:
    mine.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)
    recorded_steps: list[int] = []
    estimates: list[float] = []
    losses: list[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            if batch_indices.numel() < 2:
                continue
            x_batch = train_x[batch_indices]
            y_batch = train_y[batch_indices]
            y_marginal = y_batch[
                torch.randperm(y_batch.shape[0], device=device)
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = mine(x_batch, y_batch, y_marginal=y_marginal)
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            epoch_estimates = []
            with torch.no_grad():
                for start in range(0, test_x.shape[0], batch_size):
                    x_batch = test_x[start : start + batch_size]
                    y_batch = test_y[start : start + batch_size]
                    if x_batch.shape[0] < 2:
                        continue
                    y_marginal = y_batch[
                        torch.randperm(y_batch.shape[0], device=device)
                    ]
                    epoch_estimates.append(
                        mine.estimate(
                            x_batch,
                            y_batch,
                            y_marginal=y_marginal,
                        ).item()
                    )
            estimate = float(sum(epoch_estimates) / len(epoch_estimates))
            recorded_steps.append(global_step)
            estimates.append(estimate)
            losses.append(float(loss.item()))
            print(
                f"epoch={epoch:4d} step={global_step:5d} "
                f"loss={loss.item(): .4f} mi_estimate={estimate: .4f}"
            )

    return TrainResult(
        final_mi=estimates[-1],
        steps=recorded_steps,
        estimates=estimates,
        losses=losses,
    )
