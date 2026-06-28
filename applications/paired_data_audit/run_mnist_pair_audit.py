"""Application: paired data quality auditing with MINE and SCN-MINE.

The task is intentionally simple and visual.  We treat MNIST images and labels
as paired data.  Then we corrupt a controlled fraction of labels by replacing
them with random labels.  If an MI estimator is useful as a data-quality
auditor, the estimated I(image; label) should decrease as the corruption rate
increases.  We compare the original MINE estimator with SCN-MINE.

Run from the repository root:

    python applications/paired_data_audit/run_mnist_pair_audit.py \
        --methods mine scn --corruptions 0 0.2 0.4 0.6 0.8 1.0

Fast smoke test:

    python applications/paired_data_audit/run_mnist_pair_audit.py --steps 100 --seeds 0 --corruptions 0 1
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCN_SRC = REPO_ROOT / "scn_mine" / "src"
if str(SCN_SRC) not in sys.path:
    sys.path.insert(0, str(SCN_SRC))

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from scn_mine.data import set_seed
from scn_mine.mnist import load_mnist_raw
from scn_mine.models import PairwiseCritic
from scn_mine.objectives import MIObjective, ObjectiveConfig
from scn_mine.reliability import batch_estimates, bootstrap_interval
from scn_mine.training import TrainConfig, train_estimator


@dataclass
class MNISTPairSampler:
    images: torch.Tensor
    labels: torch.Tensor
    corruption_rate: float

    @property
    def x_dim(self) -> int:
        return 28 * 28

    @property
    def y_dim(self) -> int:
        return 10

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.images.shape[0], (batch_size,), device=device, generator=generator)
        x = self.images[idx.cpu()].to(device).reshape(batch_size, -1)
        labels = self.labels[idx.cpu()].to(device)
        if self.corruption_rate > 0:
            mask = torch.rand(batch_size, device=device, generator=generator) < self.corruption_rate
            random_labels = torch.randint(0, 10, (batch_size,), device=device, generator=generator)
            labels = torch.where(mask, random_labels, labels)
        y = F.one_hot(labels, num_classes=10).float()
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST paired data quality auditing with MINE/SCN-MINE.")
    parser.add_argument("--data", type=Path, default=Path("data/MNIST/raw"))
    parser.add_argument("--methods", nargs="+", choices=["mine", "scn", "clip_dv", "infonce", "smile"], default=["mine", "scn"])
    parser.add_argument("--corruptions", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--heldout-batches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-start", type=float, default=5.0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--mix-start", type=float, default=0.7)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overflow-weight", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/paired_data_audit"))
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def plot_examples(images: torch.Tensor, labels: torch.Tensor, outdir: Path, seed: int = 0) -> None:
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(images.shape[0], generator=rng)[:10]
    true_labels = labels[indices]
    random_labels = torch.randint(0, 10, (10,), generator=rng)

    fig, axes = plt.subplots(2, 10, figsize=(10.5, 2.4))
    for col, idx in enumerate(indices):
        axes[0, col].imshow(images[idx, 0], cmap="gray")
        axes[0, col].set_title(f"y={int(true_labels[col])}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(images[idx, 0], cmap="gray")
        axes[1, col].set_title(f"y={int(random_labels[col])}", fontsize=8)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("clean", fontsize=9)
    axes[1, 0].set_ylabel("corrupt", fontsize=9)
    fig.suptitle("MNIST image-label pairing: clean labels vs. randomly corrupted labels", y=1.08)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "mnist_pair_audit_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    images, labels = load_mnist_raw(args.data)
    images = images[: args.train_samples]
    labels = labels[: args.train_samples]
    plot_examples(images, labels, args.outdir)

    rows: list[dict[str, object]] = []
    for corruption in args.corruptions:
        sampler = MNISTPairSampler(images=images, labels=labels, corruption_rate=corruption)
        for method_index, method in enumerate(args.methods):
            for seed in args.seeds:
                run_seed = int(corruption * 1000) + 10_000 * method_index + seed
                print(f"\n=== method={method} corruption={corruption:.2f} seed={seed} ===")
                set_seed(run_seed)
                model = PairwiseCritic(
                    sampler.x_dim,
                    sampler.y_dim,
                    feature_dim=args.feature_dim,
                    hidden_size=args.hidden_size,
                )
                objective = MIObjective(
                    ObjectiveConfig(
                        method=method,
                        clip=args.clip,
                        clip_start=args.clip_start,
                        mix=args.mix,
                        mix_start=args.mix_start,
                        temperature=args.temperature,
                        overflow_weight=args.overflow_weight,
                    )
                )
                result = train_estimator(
                    model,
                    objective,
                    sampler,
                    TrainConfig(
                        steps=args.steps,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        eval_every=args.eval_every,
                        eval_batches=args.eval_batches,
                        patience=args.patience,
                    ),
                    device,
                    validation_seed=700_000 + run_seed,
                    train_eval_seed=800_000 + run_seed,
                )
                generator = torch.Generator(device=device).manual_seed(900_000 + run_seed)
                heldout = [sampler.sample(args.batch_size, device, generator) for _ in range(args.heldout_batches)]
                interval = bootstrap_interval(batch_estimates(model, objective, heldout), seed=run_seed)
                rows.append(
                    {
                        "method": method,
                        "corruption_rate": corruption,
                        "seed": seed,
                        "estimate": interval.mean,
                        "ci_lower": interval.lower,
                        "ci_upper": interval.upper,
                        "best_step": result.best_step,
                        "train_at_best": result.train_at_best,
                        "saturation_ratio": result.saturation_at_best,
                    }
                )

    result_csv = args.outdir / "mnist_pair_audit_results.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for method in args.methods:
        for corruption in args.corruptions:
            estimates = [float(r["estimate"]) for r in rows if r["method"] == method and float(r["corruption_rate"]) == corruption]
            summary_rows.append(
                {
                    "method": method,
                    "corruption_rate": corruption,
                    "mean_estimate": mean(estimates),
                    "seed_std": std(estimates),
                }
            )

    summary_csv = args.outdir / "mnist_pair_audit_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for method in args.methods:
        method_rows = [r for r in summary_rows if r["method"] == method]
        xs = [float(r["corruption_rate"]) for r in method_rows]
        ys = [float(r["mean_estimate"]) for r in method_rows]
        es = [float(r["seed_std"]) for r in method_rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=method)
    ax.set_xlabel("Label corruption rate")
    ax.set_ylabel(r"Estimated $I(\mathrm{image};\mathrm{label})$ (nats)")
    ax.set_title("Paired data quality auditing on MNIST")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.outdir / "mnist_pair_audit_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for method in args.methods:
        method_rows = [r for r in summary_rows if r["method"] == method]
        xs = [float(r["corruption_rate"]) for r in method_rows]
        ys = [float(r["seed_std"]) for r in method_rows]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("Label corruption rate")
    ax.set_ylabel("Seed standard deviation (nats)")
    ax.set_title("Estimator stability in the MNIST audit task")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(args.outdir / "mnist_pair_audit_variance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {result_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {args.outdir / 'mnist_pair_audit_examples.png'}")
    print(f"Saved: {args.outdir / 'mnist_pair_audit_curve.png'}")
    print(f"Saved: {args.outdir / 'mnist_pair_audit_variance.png'}")


if __name__ == "__main__":
    main()
