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

from scn_mine.data import CorrelatedGaussianSampler, set_seed
from scn_mine.models import PairwiseCritic
from scn_mine.objectives import MIObjective, ObjectiveConfig
from scn_mine.reliability import batch_estimates, bootstrap_interval
from scn_mine.training import TrainConfig, train_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MI estimates across rho for 20D correlated Gaussians."
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[
            -0.90,
            -0.70,
            -0.50,
            -0.30,
            -0.10,
            0.00,
            0.10,
            0.30,
            0.50,
            0.70,
            0.90,
        ],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mine", "smile", "scn"],
        default=["mine", "smile", "scn"],
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--heldout-batches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=True,
        help="Skip rows whose training produced non-finite estimates.",
    )
    parser.add_argument("--clip-start", type=float, default=5.0)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--mix-start", type=float, default=0.7)
    parser.add_argument("--mix", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overflow-weight", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "gaussian20_rho_curve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for rho_index, rho in enumerate(args.rhos):
        sampler = CorrelatedGaussianSampler(dim=args.dim, rho=rho)
        for method_index, method in enumerate(args.methods):
            run_seed = args.seed + 1000 * rho_index + 100 * method_index
            print(f"\n=== dim={args.dim} rho={rho:+.2f} method={method} ===")
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
                    mix=args.mix,
                    clip_start=args.clip_start,
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
                validation_seed=200_000 + run_seed,
                train_eval_seed=300_000 + run_seed,
            )
            if not torch.isfinite(torch.tensor(result.best_validation)):
                rows.append(
                    {
                        "dim": args.dim,
                        "rho": rho,
                        "method": method,
                        "seed": run_seed,
                        "true_mi": sampler.true_mi,
                        "heldout_mean": float("nan"),
                        "heldout_ci_lower": float("nan"),
                        "heldout_ci_upper": float("nan"),
                        "absolute_error": float("nan"),
                        "best_step": result.best_step,
                        "saturation_ratio": result.saturation_at_best,
                        "status": "failed",
                    }
                )
                print("Skipping held-out evaluation because training produced NaN/Inf.")
                continue
            generator = torch.Generator(device=device).manual_seed(400_000 + run_seed)
            heldout = [
                sampler.sample(args.batch_size, device, generator)
                for _ in range(args.heldout_batches)
            ]
            interval = bootstrap_interval(
                batch_estimates(model, objective, heldout),
                seed=run_seed,
            )
            rows.append(
                {
                    "dim": args.dim,
                    "rho": rho,
                    "method": method,
                    "seed": run_seed,
                    "true_mi": sampler.true_mi,
                    "heldout_mean": interval.mean,
                    "heldout_ci_lower": interval.lower,
                    "heldout_ci_upper": interval.upper,
                    "absolute_error": abs(interval.mean - sampler.true_mi),
                    "best_step": result.best_step,
                    "saturation_ratio": result.saturation_at_best,
                    "status": "ok",
                }
            )

    csv_path = args.outdir / f"gaussian{args.dim}_rho_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.style.use("seaborn-v0_8")
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    method_labels = {
        "mine": "MINE",
        "smile": "SMILE",
        "scn": "SCN-MINE",
    }
    for method in args.methods:
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row["status"] == "ok"
        ]
        method_rows.sort(key=lambda row: float(row["rho"]))
        if not method_rows:
            continue
        axis.plot(
            [float(row["rho"]) for row in method_rows],
            [float(row["heldout_mean"]) for row in method_rows],
            marker="o",
            linewidth=1.8,
            label=method_labels[method],
        )

    true_rows = []
    for rho in args.rhos:
        sampler = CorrelatedGaussianSampler(dim=args.dim, rho=rho)
        true_rows.append((rho, sampler.true_mi))
    axis.plot(
        [rho for rho, _ in true_rows],
        [value for _, value in true_rows],
        linestyle="--",
        linewidth=1.8,
        color="tab:purple",
        label="True MI",
    )
    axis.set_title(f"Mutual Information of {args.dim}-dimensional variables")
    axis.set_xlabel(r"$\rho$")
    axis.set_ylabel(r"$I(X_a; X_b)$")
    axis.set_xticks(args.rhos)
    axis.tick_params(axis="x", rotation=45)
    axis.legend(framealpha=0.9)
    figure.tight_layout()

    figure_path = args.outdir / f"gaussian{args.dim}_rho_curve.png"
    figure.savefig(figure_path, dpi=180)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
