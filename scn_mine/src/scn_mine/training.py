from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from .objectives import MIObjective


class Sampler(Protocol):
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    eval_every: int = 50
    eval_batches: int = 8
    patience: int = 12
    lr_decay_fraction: float = 0.6
    lr_decay_factor: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.steps,
            self.batch_size,
            self.eval_every,
            self.eval_batches,
            self.patience,
        ) <= 0:
            raise ValueError("step, batch, evaluation, and patience values must be positive")
        if not 0.0 < self.lr_decay_fraction <= 1.0:
            raise ValueError("lr_decay_fraction must be in (0, 1]")
        if not 0.0 < self.lr_decay_factor <= 1.0:
            raise ValueError("lr_decay_factor must be in (0, 1]")


@dataclass
class HistoryRow:
    step: int
    train_estimate: float
    validation_estimate: float
    training_objective: float
    loss: float
    saturation_ratio: float
    clip: float
    mix: float
    learning_rate: float


@dataclass
class TrainResult:
    best_step: int
    best_validation: float
    train_at_best: float
    saturation_at_best: float
    history: list[HistoryRow] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    objective: MIObjective,
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    model.eval()
    objective.eval()
    estimates: list[float] = []
    saturations: list[float] = []
    for x, y in validation_batches:
        output = objective(model(x, y))
        estimates.append(output.estimate.item())
        saturations.append(output.saturation_ratio.item())
    return (
        float(sum(estimates) / len(estimates)),
        float(sum(saturations) / len(saturations)),
    )


def train_estimator(
    model: nn.Module,
    objective: MIObjective,
    sampler: Sampler,
    config: TrainConfig,
    device: torch.device,
    validation_seed: int = 10_000,
    train_eval_seed: int = 20_000,
) -> TrainResult:
    model.to(device)
    objective.to(device)
    validation_generator = torch.Generator(device=device).manual_seed(
        validation_seed
    )
    validation_batches = [
        sampler.sample(
            config.batch_size,
            device,
            generator=validation_generator,
        )
        for _ in range(config.eval_batches)
    ]
    train_eval_generator = torch.Generator(device=device).manual_seed(
        train_eval_seed
    )
    train_eval_batches = [
        sampler.sample(
            config.batch_size,
            device,
            generator=train_eval_generator,
        )
        for _ in range(config.eval_batches)
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_objective_state = copy.deepcopy(objective.state_dict())
    best_progress = (0, config.steps)
    best_validation = float("-inf")
    best_step = 0
    train_at_best = float("nan")
    saturation_at_best = float("nan")
    stale_evaluations = 0
    history: list[HistoryRow] = []
    decay_step = max(1, int(config.steps * config.lr_decay_fraction))

    for step in range(1, config.steps + 1):
        if step == decay_step + 1:
            for group in optimizer.param_groups:
                group["lr"] *= config.lr_decay_factor
        model.train()
        objective.train()
        objective.set_progress(step, config.steps)
        x, y = sampler.sample(config.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        output = objective(model(x, y))
        if not torch.isfinite(output.loss):
            print(f"step={step:5d} loss is non-finite; restoring best checkpoint")
            break
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % config.eval_every == 0
            or step == config.steps
        )
        if not should_evaluate:
            continue
        train_estimate, _ = evaluate(
            model,
            objective,
            train_eval_batches,
        )
        validation, validation_saturation = evaluate(
            model,
            objective,
            validation_batches,
        )
        row = HistoryRow(
            step=step,
            train_estimate=train_estimate,
            validation_estimate=validation,
            training_objective=output.training_objective.item(),
            loss=output.loss.item(),
            saturation_ratio=validation_saturation,
            clip=output.clip,
            mix=output.mix,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(row)
        if not torch.isfinite(torch.tensor(validation)):
            print(f"step={step:5d} validation is non-finite; restoring best checkpoint")
            break
        print(
            f"step={step:5d} train={row.train_estimate: .4f} "
            f"validation={validation: .4f} objective={row.training_objective: .4f} "
            f"clip={row.clip:.2f} mix={row.mix:.3f} "
            f"saturation={validation_saturation: .3f}"
        )

        if validation > best_validation:
            best_validation = validation
            best_step = step
            train_at_best = train_estimate
            saturation_at_best = validation_saturation
            best_state = copy.deepcopy(model.state_dict())
            best_objective_state = copy.deepcopy(objective.state_dict())
            best_progress = (step, config.steps)
            stale_evaluations = 0
        else:
            stale_evaluations += 1
            if stale_evaluations >= config.patience:
                break

    model.load_state_dict(best_state)
    objective.load_state_dict(best_objective_state)
    objective.set_progress(*best_progress)
    return TrainResult(
        best_step=best_step,
        best_validation=best_validation,
        train_at_best=train_at_best,
        saturation_at_best=saturation_at_best,
        history=history,
    )
