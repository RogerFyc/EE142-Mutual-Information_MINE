from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-8


class EMALogMeanExp(torch.autograd.Function):
    """Original MINE gradient correction with an EMA denominator."""

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        running_exp_mean: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(scores, running_exp_mean)
        return logmeanexp(scores)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scores, running_exp_mean = ctx.saved_tensors
        gradient = (
            scores.exp().detach()
            / (running_exp_mean + EPS)
            / scores.numel()
        )
        return grad_output * gradient, None


def logmeanexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values.reshape(-1), dim=0) - math.log(values.numel())


def smooth_clip(values: torch.Tensor, clip: float) -> torch.Tensor:
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    return clip * torch.tanh(values / clip)


def linear_schedule(
    step: int,
    total_steps: int,
    start: float,
    end: float,
    fraction: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    horizon = max(1, int(total_steps * fraction))
    progress = min(max(step, 0) / horizon, 1.0)
    return start + progress * (end - start)


def off_diagonal(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("all-pairs scores must be a square matrix")
    if values.shape[0] < 2:
        raise ValueError("at least two samples are required")
    mask = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    return values[mask]


@dataclass(frozen=True)
class ObjectiveConfig:
    method: str = "scn"
    clip: float = 10.0
    mix: float = 1.0
    clip_start: float = 5.0
    mix_start: float = 0.7
    clip_anneal_fraction: float = 0.6
    mix_anneal_fraction: float = 0.7
    temperature: float = 0.7
    overflow_weight: float = 1e-4
    ema_alpha: float = 0.01

    def __post_init__(self) -> None:
        valid = {"mine", "infonce", "nwj", "smile", "clip_dv", "scn"}
        if self.method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}")
        if self.clip <= 0.0 or self.clip_start <= 0.0:
            raise ValueError("clip values must be positive")
        if not 0.0 <= self.mix_start <= self.mix <= 1.0:
            raise ValueError("mix values must satisfy 0 <= mix_start <= mix <= 1")
        if not 0.0 < self.clip_anneal_fraction <= 1.0:
            raise ValueError("clip_anneal_fraction must be in (0, 1]")
        if not 0.0 < self.mix_anneal_fraction <= 1.0:
            raise ValueError("mix_anneal_fraction must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.overflow_weight < 0.0:
            raise ValueError("overflow_weight must be non-negative")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")


@dataclass
class ObjectiveOutput:
    estimate: torch.Tensor
    training_objective: torch.Tensor
    loss: torch.Tensor
    dv: torch.Tensor
    infonce: torch.Tensor
    overflow: torch.Tensor
    saturation_ratio: torch.Tensor
    clip: float
    mix: float


class MIObjective(nn.Module):
    """MINE-family objectives evaluated from an all-pairs score matrix."""

    def __init__(self, config: ObjectiveConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("running_exp_mean", torch.tensor(0.0))
        self.register_buffer("clipped_running_exp_mean", torch.tensor(0.0))
        self.current_step = 0
        self.total_steps = 1

    def set_progress(self, step: int, total_steps: int) -> None:
        self.current_step = step
        self.total_steps = total_steps

    def current_clip(self) -> float:
        if self.config.method != "scn":
            return self.config.clip
        return linear_schedule(
            self.current_step,
            self.total_steps,
            self.config.clip_start,
            self.config.clip,
            self.config.clip_anneal_fraction,
        )

    def current_mix(self) -> float:
        if self.config.method != "scn":
            return self.config.mix
        return linear_schedule(
            self.current_step,
            self.total_steps,
            self.config.mix_start,
            self.config.mix,
            self.config.mix_anneal_fraction,
        )

    def _ema_logmeanexp(
        self,
        scores: torch.Tensor,
        running_mean: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_exp_mean = torch.exp(logmeanexp(scores.detach()))
            if running_mean.item() == 0.0:
                running_mean.copy_(batch_exp_mean)
            else:
                running_mean.lerp_(batch_exp_mean, self.config.ema_alpha)
        return EMALogMeanExp.apply(scores, running_mean)

    def forward(self, raw_scores: torch.Tensor) -> ObjectiveOutput:
        clip = self.current_clip()
        mix = self.current_mix()
        diagonal = torch.diagonal(raw_scores)
        marginal = off_diagonal(raw_scores)
        clipped = smooth_clip(raw_scores, clip)
        clipped_diagonal = torch.diagonal(clipped)
        clipped_marginal = off_diagonal(clipped)

        if self.config.method == "mine" and self.training:
            marginal_log_mean = self._ema_logmeanexp(
                marginal,
                self.running_exp_mean,
            )
        else:
            marginal_log_mean = logmeanexp(marginal)
        raw_dv = diagonal.mean() - marginal_log_mean
        if self.config.method in {"clip_dv", "scn"} and self.training:
            clipped_marginal_log_mean = self._ema_logmeanexp(
                clipped_marginal,
                self.clipped_running_exp_mean,
            )
        else:
            clipped_marginal_log_mean = logmeanexp(clipped_marginal)
        clipped_dv = clipped_diagonal.mean() - clipped_marginal_log_mean
        infonce = (
            clipped_diagonal / self.config.temperature
            - torch.logsumexp(clipped / self.config.temperature, dim=1)
            + math.log(raw_scores.shape[0])
        ).mean()
        nwj = diagonal.mean() - torch.exp(marginal - 1.0).mean()
        smile = diagonal.mean() - logmeanexp(
            marginal.clamp(-clip, clip)
        )

        if self.config.method == "mine":
            estimate = raw_dv
            training_objective = raw_dv
        elif self.config.method == "infonce":
            estimate = infonce
            training_objective = infonce
        elif self.config.method == "nwj":
            estimate = nwj
            training_objective = nwj
        elif self.config.method == "smile":
            estimate = smile
            training_objective = -(
                F.softplus(-diagonal).mean() + F.softplus(marginal).mean()
            )
        elif self.config.method == "clip_dv":
            estimate = clipped_dv
            training_objective = clipped_dv
        else:
            estimate = clipped_dv
            training_objective = mix * clipped_dv + (1.0 - mix) * infonce

        overflow = torch.relu(raw_scores.abs() - clip).square().mean()
        saturation_ratio = (raw_scores.abs() > clip).float().mean()
        if self.config.method == "smile":
            # SMILE estimates MI from a learned density ratio. Directly
            # maximizing the clipped estimator is unbounded because its joint
            # term is not clipped. Logistic classification consistently learns
            # the joint/product log density ratio without that failure mode.
            loss = F.softplus(-diagonal).mean() + F.softplus(marginal).mean()
        else:
            regularizer = (
                self.config.overflow_weight * overflow
                if self.config.method == "scn"
                else raw_scores.new_zeros(())
            )
            loss = -training_objective + regularizer
        return ObjectiveOutput(
            estimate=estimate,
            training_objective=training_objective,
            loss=loss,
            dv=clipped_dv if self.config.method in {"clip_dv", "scn"} else raw_dv,
            infonce=infonce,
            overflow=overflow,
            saturation_ratio=saturation_ratio,
            clip=clip,
            mix=mix,
        )
