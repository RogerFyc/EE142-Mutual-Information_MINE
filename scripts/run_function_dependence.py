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

from mine_repro.data import FUNCTIONS, sample_noisy_function, set_seed
from mine_repro.model import MINE, StatisticsNetwork
from mine_repro.training import train_mine


DISPLAY_NAMES = {
    "identity": "x",
    "cubic": "x^3",
    "sinusoid": "sin(x)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate MI for Y=f(X)+noise to show nonlinear dependence capture."
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.2, 0.5, 0.9])
    parser.add_argument("--functions", nargs="+", default=["identity", "cubic", "sinusoid"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-alpha", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Optionally enable gradient clipping. Disabled by default to match Section 4.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for function_name in args.functions:
        if function_name not in FUNCTIONS:
            raise ValueError(f"Unknown function {function_name}; choose from {sorted(FUNCTIONS)}")
        function = FUNCTIONS[function_name]
        for sigma in args.sigmas:
            print(f"\n=== function={function_name} sigma={sigma:.3f} ===")
            net = StatisticsNetwork(
                args.dim,
                args.dim,
                hidden_size=args.hidden_size,
                hidden_layers=args.hidden_layers,
            )
            mine = MINE(net, loss="mine", ema_alpha=args.ema_alpha)

            def sampler(batch_size: int, device: torch.device):
                return sample_noisy_function(batch_size, args.dim, sigma, function, device)

            result = train_mine(
                mine,
                sampler=sampler,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                eval_every=args.eval_every,
                clip_grad_norm=args.clip_grad_norm,
            )
            rows.append(
                {
                    "function": function_name,
                    "sigma": sigma,
                    "mine_estimate": result.final_mi,
                }
            )

    csv_path = args.outdir / "function_dependence_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "sigma", "mine_estimate"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(7, 4.5))
    for function_name in args.functions:
        xs = [row["sigma"] for row in rows if row["function"] == function_name]
        ys = [row["mine_estimate"] for row in rows if row["function"] == function_name]
        plt.plot(xs, ys, marker="o", label=DISPLAY_NAMES.get(function_name, function_name))
    plt.xlabel("noise sigma")
    plt.ylabel("estimated mutual information")
    plt.title("MINE captures nonlinear dependence")
    plt.legend()
    plt.tight_layout()
    fig_path = args.outdir / "function_dependence.png"
    plt.savefig(fig_path, dpi=160)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
