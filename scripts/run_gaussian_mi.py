from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import torch

from mine_repro.baselines import kraskov_mi, sample_correlated_gaussian_np
from mine_repro.data import sample_correlated_gaussian, set_seed, true_gaussian_mi
from mine_repro.model import MINE, StatisticsNetwork
from mine_repro.training import train_mine, train_mine_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the MINE correlated Gaussian experiment."
    )
    parser.add_argument(
        "--preset",
        choices=["paper", "reference"],
        default="paper",
        help="'reference' matches mine-pytorch-master's Gaussian notebook.",
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--rhos", type=float, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--kraskov-samples", type=int, default=10000)
    parser.add_argument("--kraskov-k", type=int, default=3)
    parser.add_argument(
        "--include-kraskov",
        action="store_true",
        help="Add the optional KSG/Kraskov nearest-neighbor baseline if scikit-learn is installed.",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["elu", "relu"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.01)
    parser.add_argument(
        "--loss",
        choices=["mine", "biased", "fdiv"],
        nargs="+",
        default=None,
        help="Estimator objective(s) to run. Use '--loss mine fdiv' for a MINE/MINE-f comparison.",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=8,
        help="Number of independent batches averaged for each reported estimate.",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Optionally enable gradient clipping. Disabled by default to match Section 4.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "reference":
        rhos = args.rhos if args.rhos is not None else torch.linspace(-0.99, 0.99, 15).tolist()
        losses = args.loss if args.loss is not None else ["biased"]
        batch_size = args.batch_size if args.batch_size is not None else 500
        hidden_size = args.hidden_size if args.hidden_size is not None else 100
        activation = args.activation if args.activation is not None else "relu"
        lr = args.lr if args.lr is not None else 1e-4
    else:
        rhos = args.rhos if args.rhos is not None else [-0.9, -0.5, 0.0, 0.5, 0.9]
        losses = args.loss if args.loss is not None else ["mine"]
        batch_size = args.batch_size if args.batch_size is not None else 256
        hidden_size = args.hidden_size if args.hidden_size is not None else 256
        activation = args.activation if args.activation is not None else "elu"
        lr = args.lr if args.lr is not None else 1e-4
    steps = args.steps if args.steps is not None else 1000

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.preset == "reference" and args.dim != 1:
        print(
            "Note: mine-pytorch-master's saved mi_estimation.png used --dim 1. "
            "Using the reference preset with another dimension is an extrapolation."
        )
    if args.preset == "paper" and args.dim >= 20 and max(abs(rho) for rho in rhos) >= 0.9 and steps < 10000:
        print(
            "Warning: high-dimensional, high-MI points are strongly underfit with "
            f"--steps {steps}. Use about 20000 steps per rho for the paper-scale curve."
        )

    rows = []
    for loss_name in losses:
        for rho in rhos:
            print(f"\n=== loss={loss_name} rho={rho:+.3f} ===")
            net = StatisticsNetwork(
                args.dim,
                args.dim,
                hidden_size=hidden_size,
                hidden_layers=args.hidden_layers,
                activation=activation,
            )
            mine = MINE(net, loss=loss_name, ema_alpha=args.ema_alpha)

            if args.preset == "reference":
                train_x, train_y = sample_correlated_gaussian(
                    args.samples, args.dim, rho, "cpu"
                )
                test_x, test_y = sample_correlated_gaussian(
                    args.samples, args.dim, rho, "cpu"
                )
                result = train_mine_reference(
                    mine,
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    epochs=args.epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                recorded_steps = args.epochs * (
                    (args.samples + batch_size - 1) // batch_size
                )
            else:
                def sampler(sample_batch_size: int, sample_device: torch.device):
                    return sample_correlated_gaussian(
                        sample_batch_size, args.dim, rho, sample_device
                    )

                result = train_mine(
                    mine,
                    sampler=sampler,
                    steps=steps,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    eval_every=args.eval_every,
                    eval_batches=args.eval_batches,
                    clip_grad_norm=args.clip_grad_norm,
                )
                recorded_steps = steps
            true_mi = true_gaussian_mi(args.dim, rho)
            rows.append(
                {
                    "preset": args.preset,
                    "dim": args.dim,
                    "joint_dim": 2 * args.dim,
                    "steps": recorded_steps,
                    "batch_size": batch_size,
                    "loss": loss_name,
                    "rho": rho,
                    "mine_estimate": result.final_mi,
                    "true_mi": true_mi,
                    "absolute_error": abs(result.final_mi - true_mi),
                }
            )
            print(
                f"loss={loss_name} rho={rho:+.3f} estimate={result.final_mi:.4f} "
                f"true={true_mi:.4f} abs_error={abs(result.final_mi - true_mi):.4f}"
            )
    if args.include_kraskov:
        for rho in rhos:
            true_mi = true_gaussian_mi(args.dim, rho)
            try:
                x_np, y_np = sample_correlated_gaussian_np(
                    args.kraskov_samples,
                    args.dim,
                    rho,
                    seed=args.seed + int((rho + 1.0) * 10_000),
                )
                estimate = kraskov_mi(x_np, y_np, k=args.kraskov_k)
            except ImportError as exc:
                print(f"Skipping Kraskov baseline: {exc}")
                break
            rows.append(
                {
                    "preset": args.preset,
                    "dim": args.dim,
                    "joint_dim": 2 * args.dim,
                    "steps": 0,
                    "batch_size": args.kraskov_samples,
                    "loss": "kraskov",
                    "rho": rho,
                    "mine_estimate": estimate,
                    "true_mi": true_mi,
                    "absolute_error": abs(estimate - true_mi),
                }
            )
            print(
                f"loss=kraskov rho={rho:+.3f} estimate={estimate:.4f} "
                f"true={true_mi:.4f} abs_error={abs(estimate - true_mi):.4f}"
            )

    csv_path = args.outdir / "gaussian_mi_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dim",
                "joint_dim",
                "preset",
                "steps",
                "batch_size",
                "loss",
                "rho",
                "mine_estimate",
                "true_mi",
                "absolute_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(7, 4.5))
    plot_losses = list(losses)
    if any(row["loss"] == "kraskov" for row in rows):
        plot_losses.append("kraskov")
    for loss_name in plot_losses:
        loss_rows = [row for row in rows if row["loss"] == loss_name]
        if not loss_rows:
            continue
        rhos = [row["rho"] for row in loss_rows]
        estimates = [row["mine_estimate"] for row in loss_rows]
        label = {
            "mine": "MINE",
            "fdiv": "MINE-f",
            "biased": "MINE" if args.preset == "reference" else "MINE biased",
            "kraskov": "Kraskov",
        }.get(loss_name, loss_name)
        plt.plot(rhos, estimates, marker="o", label=label)
    truth_rows = [row for row in rows if row["loss"] == losses[0]]
    plt.plot(
        [row["rho"] for row in truth_rows],
        [row["true_mi"] for row in truth_rows],
        linestyle="--",
        marker="x",
        label="True MI",
    )
    plt.xlabel("correlation rho")
    plt.ylabel("mutual information")
    title_dim = 2 if args.dim == 1 else args.dim
    plt.title(f"Mutual Information of {title_dim}-dimensional variables")
    plt.legend()
    plt.tight_layout()
    fig_path = args.outdir / "gaussian_mi.png"
    plt.savefig(fig_path, dpi=160)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
