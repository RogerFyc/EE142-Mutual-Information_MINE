from __future__ import annotations

import math

import torch
from torch import nn


EPS = 1e-8


def logmeanexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


class EMALogMeanExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, running_exp_mean: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scores, running_exp_mean)
        return logmeanexp(scores.view(-1), dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        scores, running_exp_mean = ctx.saved_tensors
        scores = scores.view(-1)
        grad = scores.exp().detach() / (running_exp_mean + EPS) / scores.numel()
        return grad_output * grad.view_as(scores), None


class StatisticsNetwork(nn.Module):

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_size: int = 128,
        hidden_layers: int = 2,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        activations: dict[str, type[nn.Module]] = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of: {sorted(activations)}")
        layers: list[nn.Module] = []
        in_dim = x_dim + y_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), activations[activation]()])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1)).view(-1)


class MINE(nn.Module):

    def __init__(
        self,
        statistics_network: nn.Module,
        loss: str = "mine",
        ema_alpha: float = 0.01,
        smile_clip: float = 5.0,
    ) -> None:
        super().__init__()
        if loss not in {"mine", "biased", "fdiv", "smile"}:
            raise ValueError("loss must be one of: mine, biased, fdiv, smile")
        if smile_clip <= 0.0:
            raise ValueError("smile_clip must be positive")
        self.statistics_network = statistics_network
        self.loss = loss
        self.ema_alpha = ema_alpha
        self.smile_clip = smile_clip
        self.register_buffer("running_exp_mean", torch.tensor(0.0))

    def _marginal_scores(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_marginal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y_marginal is None:
            if y.shape[0] < 2:
                raise ValueError("at least two samples are required to form product marginals")
            shift = torch.randint(1, y.shape[0], (), device=y.device).item()
            y_marginal = torch.roll(y, shifts=shift, dims=0)
        return self.statistics_network(x, y_marginal)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_marginal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint_scores = self.statistics_network(x, y)
        marginal_scores = self._marginal_scores(x, y, y_marginal)
        if self.loss == "smile":
            joint_scores = joint_scores.clamp(
                -self.smile_clip,
                self.smile_clip,
            )
            marginal_scores = marginal_scores.clamp(
                -self.smile_clip,
                self.smile_clip,
            )
        joint_term = joint_scores.mean()

        if self.loss == "mine":
            with torch.no_grad():
                batch_exp_mean = torch.exp(logmeanexp(marginal_scores.detach(), dim=0))
                if self.running_exp_mean.item() == 0.0:
                    self.running_exp_mean.copy_(batch_exp_mean)
                else:
                    self.running_exp_mean.mul_(1.0 - self.ema_alpha)
                    self.running_exp_mean.add_(self.ema_alpha * batch_exp_mean)
            marginal_term = EMALogMeanExp.apply(marginal_scores, self.running_exp_mean)
        elif self.loss == "biased":
            marginal_term = logmeanexp(marginal_scores, dim=0)
        elif self.loss == "fdiv":
            marginal_term = torch.exp(marginal_scores - 1.0).mean()
        else:
            marginal_term = logmeanexp(marginal_scores, dim=0)

        return -(joint_term - marginal_term)

    @torch.no_grad()
    def estimate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_marginal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint_scores = self.statistics_network(x, y)
        marginal_scores = self._marginal_scores(x, y, y_marginal)

        if self.loss == "fdiv":
            return joint_scores.mean() - torch.exp(marginal_scores - 1.0).mean()
        if self.loss == "smile":
            joint_scores = joint_scores.clamp(
                -self.smile_clip,
                self.smile_clip,
            )
            marginal_scores = marginal_scores.clamp(
                -self.smile_clip,
                self.smile_clip,
            )
        return joint_scores.mean() - logmeanexp(marginal_scores, dim=0)
