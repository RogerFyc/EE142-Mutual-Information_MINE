from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from .data import sample_gaussian_grid
from .model import MINE, StatisticsNetwork


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        hidden_layers: int = 2,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.LeakyReLU(0.2)])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ApplicationResult:
    final_loss: float
    final_mi: float
    output_path: Path | None = None


def train_information_bottleneck(
    steps: int,
    batch_size: int,
    latent_dim: int,
    beta: float,
    lr: float,
    device: torch.device,
    outdir: Path,
    hidden_size: int = 128,
    mine_loss: str = "mine",
) -> ApplicationResult:
    num_classes = 25
    encoder = MLP(2, latent_dim, hidden_size).to(device)
    classifier = MLP(latent_dim, num_classes, hidden_size, hidden_layers=1).to(device)
    mine_xz = MINE(StatisticsNetwork(2, latent_dim, hidden_size=hidden_size), loss=mine_loss).to(device)
    opt_main = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    opt_mi = torch.optim.Adam(mine_xz.parameters(), lr=lr)

    last_loss = 0.0
    last_mi = 0.0
    for step in range(1, steps + 1):
        x, labels, _ = sample_gaussian_grid(batch_size, device=device)
        z = encoder(x)
        opt_mi.zero_grad(set_to_none=True)
        mi_loss = mine_xz(x.detach(), z.detach())
        mi_loss.backward()
        opt_mi.step()

        x, labels, _ = sample_gaussian_grid(batch_size, device=device)
        z = encoder(x)
        logits = classifier(z)
        mi_estimate = mine_xz.estimate(x, z)
        loss = F.cross_entropy(logits, labels) + beta * mi_estimate
        opt_main.zero_grad(set_to_none=True)
        loss.backward()
        opt_main.step()
        last_loss = float(loss.item())
        last_mi = float(mi_estimate.item())
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"step={step:5d} loss={last_loss:.4f} i_xz={last_mi:.4f} acc={acc:.3f}")

    with torch.no_grad():
        x, labels, _ = sample_gaussian_grid(2000, device=device)
        z = encoder(x)
        logits = classifier(z)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    out_path = outdir / "information_bottleneck_latent.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z_np = z.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    plt.figure(figsize=(5.5, 5.0))
    if z_np.shape[1] >= 2:
        plt.scatter(z_np[:, 0], z_np[:, 1], s=6, c=labels_np, cmap="tab20")
        plt.xlabel("z0")
        plt.ylabel("z1")
    else:
        plt.scatter(z_np[:, 0], torch.zeros_like(z[:, 0]).cpu().numpy(), s=6, c=labels_np, cmap="tab20")
        plt.xlabel("z0")
    plt.title(f"Information Bottleneck latent space, acc={acc:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return ApplicationResult(last_loss, last_mi, out_path)
