from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import torch

from scn_mine.data import (
    CorrelatedGaussianSampler,
    NonlinearNuisanceSampler,
    set_seed,
)
from scn_mine.models import PairwiseCritic
from scn_mine.objectives import MIObjective, ObjectiveConfig
from scn_mine.reliability import (
    batch_estimates,
    bootstrap_interval,
    independence_estimate,
)
from scn_mine.training import TrainConfig, train_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SCN-MINE and baselines.")
    parser.add_argument(
        "--scenario",
        choices=["low", "gaussian20", "high"],
        default="low",
        help="'low': 4D Gaussian; 'gaussian20': 20D Gaussian; 'high': nonlinear nuisance.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mine", "infonce", "nwj", "smile", "clip_dv", "scn"],
        default=["mine", "infonce", "nwj", "smile", "scn"],
    )
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=10.0, help="Final clip.")
    parser.add_argument("--clip-start", type=float, default=5.0)
    parser.add_argument("--clip-anneal-fraction", type=float, default=0.6)
    parser.add_argument("--mix", type=float, default=1.0, help="Final DV weight.")
    parser.add_argument("--mix-start", type=float, default=0.7)
    parser.add_argument("--mix-anneal-fraction", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overflow-weight", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "synthetic")
    return parser.parse_args()


def make_sampler(args: argparse.Namespace):
    if args.scenario == "low":
        rho = args.rho if args.rho is not None else 0.95
        return CorrelatedGaussianSampler(dim=4, rho=rho)
    if args.scenario == "gaussian20":
        rho = args.rho if args.rho is not None else 0.6
        return CorrelatedGaussianSampler(dim=20, rho=rho)
    rho = args.rho if args.rho is not None else 0.6
    return NonlinearNuisanceSampler(
        signal_dim=16,
        observed_dim=128,
        rho=rho,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    sampler = make_sampler(args)
    rows: list[dict[str, object]] = []
    histories: dict[tuple[str, int], list[dict[str, object]]] = {}

    for method in args.methods:
        for seed in args.seeds:
            print(f"\n=== scenario={args.scenario} method={method} seed={seed} ===")
            set_seed(seed)
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
                    clip_anneal_fraction=args.clip_anneal_fraction,
                    mix_anneal_fraction=args.mix_anneal_fraction,
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
                validation_seed=100_000 + seed,
                train_eval_seed=150_000 + seed,
            )
            generator = torch.Generator(device=device).manual_seed(200_000 + seed)
            heldout = [
                sampler.sample(args.batch_size, device, generator)
                for _ in range(32)
            ]
            estimates = batch_estimates(model, objective, heldout)
            interval = bootstrap_interval(estimates, seed=seed)
            x_ind = heldout[0][0]
            y_ind = heldout[1][1]
            independence = independence_estimate(model, objective, x_ind, y_ind)
            row = {
                "scenario": args.scenario,
                "method": method,
                "seed": seed,
                "rho": sampler.rho,
                "clip": args.clip,
                "mix": args.mix,
                "clip_start": args.clip_start,
                "mix_start": args.mix_start,
                "temperature": args.temperature,
                "overflow_weight": args.overflow_weight,
                "true_mi": sampler.true_mi,
                "best_step": result.best_step,
                "train_at_best": result.train_at_best,
                "heldout_mean": interval.mean,
                "heldout_ci_lower": interval.lower,
                "heldout_ci_upper": interval.upper,
                "bias": interval.mean - sampler.true_mi,
                "absolute_error": abs(interval.mean - sampler.true_mi),
                "independence_estimate": independence,
                "saturation_ratio": result.saturation_at_best,
            }
            rows.append(row)
            history_rows = [history.__dict__ for history in result.history]
            histories[(method, seed)] = history_rows
            history_path = args.outdir / f"history_{args.scenario}_{method}_{seed}.json"
            history_path.write_text(
                json.dumps(history_rows, indent=2),
                encoding="utf-8",
            )

    csv_path = args.outdir / f"results_{args.scenario}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, (summary_axis, seed_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )
    method_positions = list(range(len(args.methods)))
    all_estimates: list[float] = []
    for position, method in zip(method_positions, args.methods):
        method_rows = [row for row in rows if row["method"] == method]
        estimates = [float(row["heldout_mean"]) for row in method_rows]
        all_estimates.extend(estimates)
        mean = sum(estimates) / len(estimates)
        if len(estimates) > 1:
            variance = sum((value - mean) ** 2 for value in estimates) / (
                len(estimates) - 1
            )
            error = math.sqrt(variance)
        else:
            error = 0.0
        summary_axis.errorbar(
            position,
            mean,
            yerr=error,
            marker="o",
            capsize=5,
            color=f"C{position}",
        )
        jitter = [
            position + 0.06 * (index - (len(estimates) - 1) / 2)
            for index in range(len(estimates))
        ]
        seed_axis.scatter(
            jitter,
            estimates,
            color=f"C{position}",
            alpha=0.8,
            label=method,
        )

    for axis in (summary_axis, seed_axis):
        axis.axhline(
            sampler.true_mi,
            color="black",
            linestyle="--",
            label="true MI",
        )
        axis.set_xticks(method_positions, args.methods, rotation=20)
        axis.set_ylabel("held-out MI estimate (nats)")
        axis.grid(axis="y", alpha=0.25)
    summary_axis.set_title("Mean +/- seed standard deviation")
    seed_axis.set_title("Individual seed estimates")
    seed_axis.legend(ncol=2)
    figure.suptitle(f"{args.scenario} synthetic benchmark")
    figure_path = args.outdir / f"estimates_{args.scenario}.png"
    figure.savefig(figure_path, dpi=160)

    frontier_figure, frontier_axis = plt.subplots(figsize=(6.5, 4.8))
    for position, method in enumerate(args.methods):
        method_rows = [row for row in rows if row["method"] == method]
        estimates = [float(row["heldout_mean"]) for row in method_rows]
        errors = [float(row["absolute_error"]) for row in method_rows]
        mean = sum(estimates) / len(estimates)
        if len(estimates) > 1:
            variance = sum((value - mean) ** 2 for value in estimates) / (
                len(estimates) - 1
            )
            standard_deviation = math.sqrt(variance)
        else:
            standard_deviation = 0.0
        frontier_axis.scatter(
            sum(errors) / len(errors),
            standard_deviation,
            s=70,
            color=f"C{position}",
            label=method,
        )
    frontier_axis.set_xlabel("mean absolute error (nats)")
    frontier_axis.set_ylabel("seed standard deviation (nats)")
    frontier_axis.set_title("Bias-variance frontier (lower left is better)")
    frontier_axis.grid(alpha=0.25)
    frontier_axis.legend()
    frontier_figure.tight_layout()
    frontier_path = args.outdir / f"bias_variance_{args.scenario}.png"
    frontier_figure.savefig(frontier_path, dpi=160)

    for seed in args.seeds:
        curve_figure, axes = plt.subplots(
            3,
            1,
            figsize=(8, 9),
            sharex=True,
            constrained_layout=True,
        )
        for position, method in enumerate(args.methods):
            history = histories[(method, seed)]
            steps = [int(row["step"]) for row in history]
            axes[0].plot(
                steps,
                [float(row["validation_estimate"]) for row in history],
                color=f"C{position}",
                label=method,
            )
            axes[1].plot(
                steps,
                [float(row["training_objective"]) for row in history],
                color=f"C{position}",
                label=method,
            )
            axes[2].plot(
                steps,
                [float(row["saturation_ratio"]) for row in history],
                color=f"C{position}",
                label=method,
            )
        axes[0].axhline(sampler.true_mi, color="black", linestyle="--")
        axes[0].set_ylabel("held-out report")
        axes[1].set_ylabel("training objective")
        axes[2].set_ylabel("saturation ratio")
        axes[2].set_xlabel("training step")
        axes[0].legend(ncol=3)
        axes[0].set_title(f"Convergence diagnostics, seed {seed}")
        for axis in axes:
            axis.grid(alpha=0.25)
        curve_path = args.outdir / f"convergence_{args.scenario}_{seed}.png"
        curve_figure.savefig(curve_path, dpi=160)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {frontier_path}")


if __name__ == "__main__":
    main()
