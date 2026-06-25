from __future__ import annotations

import struct
from pathlib import Path

import torch
from torch.nn import functional as F


def load_idx_images(path: Path) -> torch.Tensor:
    with path.open("rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 0x00000803:
            raise ValueError(f"unexpected image IDX magic number in {path}")
        data = torch.frombuffer(bytearray(file.read()), dtype=torch.uint8)
    return data.reshape(count, 1, rows, cols).float().div_(255.0)


def load_idx_labels(path: Path) -> torch.Tensor:
    with path.open("rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 0x00000801:
            raise ValueError(f"unexpected label IDX magic number in {path}")
        data = torch.frombuffer(bytearray(file.read()), dtype=torch.uint8)
    return data[:count].long()


def load_mnist_raw(raw_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        load_idx_images(raw_dir / "train-images-idx3-ubyte"),
        load_idx_labels(raw_dir / "train-labels-idx1-ubyte"),
    )


def augment_views(
    images: torch.Tensor,
    generator: torch.Generator | None = None,
    max_rotation_degrees: float = 15.0,
    max_translation: float = 0.12,
    min_scale: float = 0.9,
    max_scale: float = 1.1,
    noise_std: float = 0.05,
) -> torch.Tensor:
    batch_size = images.shape[0]
    device = images.device
    dtype = images.dtype
    uniform = torch.rand(
        batch_size,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    angles = (2.0 * uniform[:, 0] - 1.0) * (
        max_rotation_degrees * torch.pi / 180.0
    )
    scales = min_scale + (max_scale - min_scale) * uniform[:, 1]
    tx = (2.0 * uniform[:, 2] - 1.0) * max_translation
    ty = (2.0 * uniform[:, 3] - 1.0) * max_translation
    cosine = torch.cos(angles) / scales
    sine = torch.sin(angles) / scales
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    augmented = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    noise = torch.randn(
        augmented.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return (augmented + noise_std * noise).clamp_(0.0, 1.0)


class MNISTViewSampler:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            0,
            self.images.shape[0],
            (batch_size,),
            device=device,
            generator=generator,
        )
        images = self.images[indices.cpu()].to(device)
        return (
            augment_views(images, generator=generator),
            augment_views(images, generator=generator),
        )
