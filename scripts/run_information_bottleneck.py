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

from mine_repro.applications import train_information_bottleneck
from mine_repro.data import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a MINE-regularized Information Bottleneck toy experiment."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--mine-loss", choices=["mine", "biased", "fdiv"], default="mine")
    parser.add_argument("--seed", type=int, default=31)
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
    result = train_information_bottleneck(
        steps=args.steps,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        lr=args.lr,
        device=device,
        outdir=args.outdir,
        hidden_size=args.hidden_size,
        mine_loss=args.mine_loss,
    )
    csv_path = args.outdir / "information_bottleneck_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["final_loss", "final_i_xz", "figure"])
        writer.writeheader()
        writer.writerow(
            {
                "final_loss": result.final_loss,
                "final_i_xz": result.final_mi,
                "figure": "" if result.output_path is None else str(result.output_path),
            }
        )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
