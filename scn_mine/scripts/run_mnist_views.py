from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import nn
from torch.nn import functional as F

from scn_mine.data import set_seed
from scn_mine.mnist import MNISTViewSampler, augment_views, load_mnist_raw
from scn_mine.models import SharedImagePairCritic
from scn_mine.objectives import MIObjective, ObjectiveConfig
from scn_mine.training import TrainConfig, train_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST two-view SCN-MINE experiment.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT.parent / "mine_reproduction" / "data" / "MNIST" / "raw",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mine", "infonce", "nwj", "smile", "clip_dv", "scn"],
        default=["mine", "infonce", "nwj", "smile", "scn"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--validation-samples", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--pair-chunk-size", type=int, default=32)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--clip-start", type=float, default=5.0)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--mix-start", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "mnist")
    return parser.parse_args()


@torch.no_grad()
def encode_batches(
    model: SharedImagePairCritic,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        chunks.append(model.encode(images[start : start + batch_size].to(device)).cpu())
    return torch.cat(chunks)


def linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> float:
    classifier = nn.Linear(train_features.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    for _ in range(epochs):
        permutation = torch.randperm(train_features.shape[0], device=device)
        for start in range(0, train_features.shape[0], 256):
            indices = permutation[start : start + 256]
            loss = F.cross_entropy(
                classifier(train_features[indices]),
                train_labels[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        predictions = classifier(validation_features.to(device)).argmax(dim=1).cpu()
    return (predictions == validation_labels).float().mean().item()


@torch.no_grad()
def matching_recall_at_one(
    model: SharedImagePairCritic,
    images: torch.Tensor,
    device: torch.device,
    sample_count: int = 512,
) -> float:
    model.eval()
    selected = images[:sample_count].to(device)
    first = F.normalize(model.encode(augment_views(selected)), dim=1)
    second = F.normalize(model.encode(augment_views(selected)), dim=1)
    matches = (first @ second.T).argmax(dim=1)
    truth = torch.arange(selected.shape[0], device=device)
    return (matches == truth).float().mean().item()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    images, labels = load_mnist_raw(args.data)
    total = args.train_samples + args.validation_samples
    if total > images.shape[0]:
        raise ValueError("requested split is larger than the local MNIST training set")
    images = images[:total]
    labels = labels[:total]
    train_images = images[: args.train_samples]
    train_labels = labels[: args.train_samples]
    validation_images = images[args.train_samples :]
    validation_labels = labels[args.train_samples :]
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for method in args.methods:
        for seed in args.seeds:
            print(f"\n=== MNIST method={method} seed={seed} ===")
            set_seed(seed)
            model = SharedImagePairCritic(
                embedding_dim=args.embedding_dim,
                hidden_size=args.hidden_size,
                pair_chunk_size=args.pair_chunk_size,
            )
            objective = MIObjective(
                ObjectiveConfig(
                    method=method,
                    clip=args.clip,
                    mix=args.mix,
                    clip_start=args.clip_start,
                    mix_start=args.mix_start,
                    temperature=args.temperature,
                )
            )
            result = train_estimator(
                model,
                objective,
                MNISTViewSampler(train_images),
                TrainConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    eval_every=args.eval_every,
                    eval_batches=args.eval_batches,
                    patience=args.patience,
                ),
                device,
                validation_seed=50_000 + seed,
                train_eval_seed=60_000 + seed,
            )
            train_features = encode_batches(model, train_images, device)
            validation_features = encode_batches(model, validation_images, device)
            probe_accuracy = linear_probe(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                args.probe_epochs,
                device,
            )
            recall = matching_recall_at_one(model, validation_images, device)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_step": result.best_step,
                    "heldout_mi": result.best_validation,
                    "train_at_best": result.train_at_best,
                    "linear_probe_accuracy": probe_accuracy,
                    "matching_recall_at_1": recall,
                    "saturation_ratio": result.saturation_at_best,
                }
            )
            print(
                f"probe_accuracy={probe_accuracy:.4f} "
                f"matching_recall@1={recall:.4f}"
            )

    output_path = args.outdir / "mnist_views_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
