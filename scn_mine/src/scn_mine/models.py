from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "relu": nn.ReLU,
    }
    if name not in activations:
        raise ValueError(f"activation must be one of {sorted(activations)}")
    return activations[name]()


class PairScoreHead(nn.Module):
    """Score all pairs of two equally sized embedding batches."""

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_size: int = 256,
        hidden_layers: int = 2,
        activation: str = "gelu",
        pair_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_size, hidden_layers) <= 0:
            raise ValueError("network dimensions and hidden_layers must be positive")
        layers: list[nn.Module] = []
        in_dim = 3 * feature_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_size),
                    _activation(activation),
                ]
            )
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)
        self.pair_chunk_size = pair_chunk_size

    def forward(self, hx: torch.Tensor, hy: torch.Tensor) -> torch.Tensor:
        if hx.ndim != 2 or hy.ndim != 2:
            raise ValueError("encoded inputs must be rank-2 tensors")
        if hx.shape[1] != hy.shape[1]:
            raise ValueError("the two embedding dimensions must match")
        chunk_size = self.pair_chunk_size or hx.shape[0]
        rows: list[torch.Tensor] = []
        for start in range(0, hx.shape[0], chunk_size):
            hx_chunk = hx[start : start + chunk_size]
            left = hx_chunk[:, None, :].expand(-1, hy.shape[0], -1)
            right = hy[None, :, :].expand(hx_chunk.shape[0], -1, -1)
            pair_features = torch.cat((left, right, left * right), dim=-1)
            rows.append(self.head(pair_features).squeeze(-1))
        return torch.cat(rows, dim=0)


class PairwiseCritic(nn.Module):
    """Joint vector critic producing all scores a_theta(x_i, y_j)."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        feature_dim: int = 128,
        hidden_size: int = 256,
        hidden_layers: int = 2,
        activation: str = "gelu",
        pair_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(x_dim, feature_dim),
            _activation(activation),
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(y_dim, feature_dim),
            _activation(activation),
        )
        self.score_head = PairScoreHead(
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            activation=activation,
            pair_chunk_size=pair_chunk_size,
        )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        return self.x_encoder(x)

    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_encoder(y)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.encode_x(x), self.encode_y(y))


class SmallImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class SharedImagePairCritic(nn.Module):
    """Shared image encoder plus a joint critic head."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        pair_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SmallImageEncoder(embedding_dim)
        self.score_head = PairScoreHead(
            feature_dim=embedding_dim,
            hidden_size=hidden_size,
            hidden_layers=2,
            pair_chunk_size=pair_chunk_size,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.encode(x), self.encode(y))
