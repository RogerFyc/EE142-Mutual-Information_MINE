"""Ablation study for SCN-MINE mixture weights and clipping thresholds.

The goal is to turn the SCN-MINE design choices into measurable comparisons:
fixed mixture weights, annealed mixture weights, different clipping thresholds,
and the original MINE/InfoNCE/Clip-DV baselines.

Run from the repository root, for example:

    python experiments/scn_ablation/run_schedule_ablation.py \
        --scenario gaussian20 --rho 0.99 --steps 1200 --seeds 0 1 2

For a faster smoke test:

    python experiments/scn_ablation/run_schedule_ablation.py --steps 100 --seeds 0
"""

from __future__ import annotations

import os

# Avoid duplicated OpenMP runtime error on Windows when PyTorch,
# NumPy, scikit-learn, or matplotlib load different OpenMP backends.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
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

from scn_mine.data import CorrelatedGaussianSampler, NonlinearNuisanceSampler, set_seed
from scn_mine.models import PairwiseCritic
from scn_mine.objectives import MIObjective, ObjectiveConfig
from scn_mine.reliability import batch_estimates, bootstrap_interval
from scn_mine.training import TrainConfig, train_estimator


@dataclass(frozen=True)
class AblationCase:
    name: str
    method: str
    clip_start: float = 5.0
    clip: float = 10.0
    mix_start: float = 0.7
    mix: float = 1.0
    temperature: float = 0.7
    overflow_weight: float = 1e-4
    notes: str = ""


def default_cases() -> list[AblationCase]:
    return [
        AblationCase("MINE-DV", "mine", notes="original DV objective"),
        AblationCase("InfoNCE", "infonce", notes="stable contrastive lower bound"),
        AblationCase("Clip-DV c=5", "clip_dv", clip_start=5.0, clip=5.0, notes="strong clipping, no InfoNCE"),
        AblationCase("Clip-DV c=10", "clip_dv", clip_start=10.0, clip=10.0, notes="weaker clipping, no InfoNCE"),
        AblationCase("SCN fixed 0.5", "scn", clip_start=5.0, clip=5.0, mix_start=0.5, mix=0.5, notes="constant 50% DV + 50% InfoNCE"),
        AblationCase("SCN fixed 0.8", "scn", clip_start=10.0, clip=10.0, mix_start=0.8, mix=0.8, notes="constant 80% DV + 20% InfoNCE"),
        AblationCase("SCN anneal 0.7->1", "scn", clip_start=5.0, clip=10.0, mix_start=0.7, mix=1.0, notes="poster setting"),
        AblationCase("SCN anneal 0.3->1", "scn", clip_start=5.0, clip=10.0, mix_start=0.3, mix=1.0, notes="stronger InfoNCE warm-up"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCN-MINE schedule and loss ablation.")
    parser.add_argument("--scenario", choices=["gaussian20", "high"], default="gaussian20")
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--heldout-batches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/scn_ablation"))
    return parser.parse_args()


def make_sampler(args: argparse.Namespace):
    if args.scenario == "gaussian20":
        return CorrelatedGaussianSampler(dim=20, rho=args.rho)
    return NonlinearNuisanceSampler(signal_dim=16, observed_dim=128, rho=args.rho)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def rmse(values: list[float], target: float) -> float:
    return math.sqrt(mean([(v - target) ** 2 for v in values]))


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    sampler = make_sampler(args)
    cases = default_cases()

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}

    for case_index, case in enumerate(cases):
        case_estimates: list[float] = []
        case_abs_errors: list[float] = []
        case_saturation: list[float] = []
        for seed in args.seeds:
            run_seed = 10_000 * case_index + seed
            print(f"\n=== {case.name} seed={seed} scenario={args.scenario} rho={args.rho} ===")
            set_seed(run_seed)
            model = PairwiseCritic(
                sampler.x_dim,
                sampler.y_dim,
                feature_dim=args.feature_dim,
                hidden_size=args.hidden_size,
            )
            objective = MIObjective(
                ObjectiveConfig(
                    method=case.method,
                    clip=case.clip,
                    clip_start=case.clip_start,
                    mix=case.mix,
                    mix_start=case.mix_start,
                    temperature=case.temperature,
                    overflow_weight=case.overflow_weight,
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
            generator = torch.Generator(device=device).manual_seed(400_000 + run_seed)
            heldout = [sampler.sample(args.batch_size, device, generator) for _ in range(args.heldout_batches)]
            interval = bootstrap_interval(batch_estimates(model, objective, heldout), seed=run_seed)
            estimate = interval.mean
            abs_error = abs(estimate - sampler.true_mi)
            case_estimates.append(estimate)
            case_abs_errors.append(abs_error)
            case_saturation.append(result.saturation_at_best)
            row = {
                "scenario": args.scenario,
                "rho": args.rho,
                "true_mi": sampler.true_mi,
                "case": case.name,
                "method": case.method,
                "seed": seed,
                "estimate": estimate,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "absolute_error": abs_error,
                "bias": estimate - sampler.true_mi,
                "best_step": result.best_step,
                "saturation_ratio": result.saturation_at_best,
                "clip_start": case.clip_start,
                "clip": case.clip,
                "mix_start": case.mix_start,
                "mix": case.mix,
                "temperature": case.temperature,
                "overflow_weight": case.overflow_weight,
                "notes": case.notes,
            }
            rows.append(row)
            histories[f"{case.name}_seed{seed}"] = [h.__dict__ for h in result.history]

        summary_rows.append(
            {
                "case": case.name,
                "method": case.method,
                "mean_estimate": mean(case_estimates),
                "seed_std": std(case_estimates),
                "rmse": rmse(case_estimates, sampler.true_mi),
                "mean_absolute_error": mean(case_abs_errors),
                "mean_saturation_ratio": mean(case_saturation),
                "clip_start": case.clip_start,
                "clip": case.clip,
                "mix_start": case.mix_start,
                "mix": case.mix,
                "notes": case.notes,
            }
        )

    result_csv = args.outdir / "schedule_ablation_results.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = args.outdir / "schedule_ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    history_json = args.outdir / "schedule_ablation_histories.json"
    history_json.write_text(json.dumps(histories, indent=2), encoding="utf-8")

    labels = [row["case"] for row in summary_rows]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    axes[0].bar(x, [float(row["rmse"]) for row in summary_rows])
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE to analytic MI (nats)")
    axes[0].set_title("Accuracy of schedule/loss choices")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, [float(row["seed_std"]) for row in summary_rows])
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Seed standard deviation (nats)")
    axes[1].set_title("Stability across random seeds")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"SCN-MINE ablation at rho={args.rho}; true MI={sampler.true_mi:.2f}")
    fig.savefig(args.outdir / "scn_schedule_ablation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for row in summary_rows:
        ax.scatter(float(row["mean_absolute_error"]), float(row["seed_std"]), s=70)
        ax.annotate(str(row["case"]), (float(row["mean_absolute_error"]), float(row["seed_std"])), fontsize=8)
    ax.set_xlabel("Mean absolute error (nats)")
    ax.set_ylabel("Seed standard deviation (nats)")
    ax.set_title("Bias--variance frontier for SCN design choices")
    ax.grid(alpha=0.25)
    fig.savefig(args.outdir / "scn_schedule_ablation_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {result_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {history_json}")
    print(f"Saved: {args.outdir / 'scn_schedule_ablation_summary.png'}")
    print(f"Saved: {args.outdir / 'scn_schedule_ablation_frontier.png'}")


if __name__ == "__main__":
    main()
