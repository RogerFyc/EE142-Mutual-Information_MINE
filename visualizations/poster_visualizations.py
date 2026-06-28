"""Poster helper visualizations for the EE142 MINE project.

This script only creates explanatory figures.  It does not rerun the main MINE
experiments.  The generated images can be used in the poster/report to explain
joint-vs-marginal sampling, SCN-MINE clipping, and application scenarios.

Usage from the repository root:

    python visualizations/poster_visualizations.py --outdir outputs/poster_assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_joint_vs_marginal(outdir: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 500
    rho = 0.82
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
    y_shuffle = rng.permutation(y)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharex=True, sharey=True)
    axes[0].scatter(x, y, s=10, alpha=0.65)
    axes[0].set_title(r"Joint samples $(x_i,y_i) \sim p(x,y)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].scatter(x, y_shuffle, s=10, alpha=0.65)
    axes[1].set_title(r"Shuffled pairs $(x_i,y_{\pi(i)}) \sim p(x)p(y)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("shuffled y")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)

    fig.suptitle("MINE compares dependent pairs with independent pairs", y=1.04)
    _save(fig, outdir / "mine_joint_vs_marginal.png")


def plot_training_pipeline(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.axis("off")

    boxes = [
        (0.04, 0.58, 0.18, 0.25, "Paired batch\n$(x_i,y_i)$"),
        (0.04, 0.15, 0.18, 0.25, "Shuffle within batch\n$(x_i,y_{\\pi(i)})$"),
        (0.34, 0.58, 0.20, 0.25, "Critic score\n$T_\\theta(x_i,y_i)$"),
        (0.34, 0.15, 0.20, 0.25, "Marginal score\n$T_\\theta(x_i,y_{\\pi(i)})$"),
        (0.66, 0.37, 0.26, 0.30, "DV objective\n$E_{joint}[T]-\\log E_{marg}[e^T]$")
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.7, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, transform=ax.transAxes)

    arrows = [
        ((0.22, 0.705), (0.34, 0.705)),
        ((0.22, 0.275), (0.34, 0.275)),
        ((0.54, 0.705), (0.66, 0.52)),
        ((0.54, 0.275), (0.66, 0.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.7))
    ax.text(0.5, 0.93, "MINE training pipeline", ha="center", va="center", fontsize=15, transform=ax.transAxes)
    _save(fig, outdir / "mine_training_pipeline.png")


def plot_scn_clipping(outdir: Path) -> None:
    c = 5.0
    a = np.linspace(-16, 16, 400)
    s = c * np.tanh(a / c)
    clipped_for_plot = np.clip(np.exp(a), 1e-3, 3e3)
    exp_s = np.exp(s)
    steps = np.linspace(0, 10_000, 200)
    lam = 0.7 + 0.3 * steps / steps.max()
    c_sched = 5.0 + 5.0 * steps / steps.max()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))

    axes[0].plot(a, a, linestyle="--", label="raw score A")
    axes[0].plot(a, s, label="clipped score c tanh(A/c)")
    axes[0].axhline(c, linestyle=":", linewidth=1.5)
    axes[0].axhline(-c, linestyle=":", linewidth=1.5)
    axes[0].set_title("Smooth score clipping")
    axes[0].set_xlabel("critic score A")
    axes[0].set_ylabel("score used in loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(a, clipped_for_plot, linestyle="--", label="exp(A), clipped in plot only")
    axes[1].plot(a, exp_s, label="exp(clipped score)")
    axes[1].set_yscale("log")
    axes[1].set_title("Controls exponential term")
    axes[1].set_xlabel("critic score A")
    axes[1].set_ylabel("exponential contribution")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(steps, lam, label=r"$\lambda_t$: DV weight")
    axes[2].plot(steps, c_sched / 10.0, label=r"$c_t/10$: clipping threshold")
    axes[2].set_title("Annealing schedule")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("normalized value")
    axes[2].set_ylim(0.48, 1.03)
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.suptitle("SCN-MINE: stabilize DV with smooth clipping and InfoNCE warm-up", y=1.08)
    _save(fig, outdir / "scn_clipping_and_schedule.png")


def _toy_gaussian_points(rng: np.random.Generator, centers: np.ndarray, n: int = 800, scale: float = 0.045) -> np.ndarray:
    idx = rng.integers(0, len(centers), size=n)
    return centers[idx] + scale * rng.normal(size=(n, 2))


def plot_applications(outdir: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    grid = np.array([(i, j) for i in np.linspace(-1, 1, 5) for j in np.linspace(-1, 1, 5)])
    gan_bad = _toy_gaussian_points(rng, grid[[0, 4, 12, 20, 24]], 500, 0.055)
    gan_good = _toy_gaussian_points(rng, grid, 1000, 0.040)

    theta = rng.uniform(0, 2 * np.pi, size=500)
    radius = 0.55 + 0.06 * rng.normal(size=500)
    ali_x = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    ali_z = np.c_[np.cos(theta), np.sin(theta)] + 0.10 * rng.normal(size=(500, 2))

    beta = np.linspace(0.05, 1.0, 12)
    ixz = 2.8 * np.exp(-1.6 * beta) + 0.15
    izy = 1.9 * (1 - np.exp(-2.3 * ixz / ixz.max()))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    axes[0].scatter(gan_bad[:, 0], gan_bad[:, 1], s=7, alpha=0.35, label="GAN")
    axes[0].scatter(gan_good[:, 0], gan_good[:, 1], s=7, alpha=0.45, label="GAN + MINE")
    axes[0].set_title("GAN + MINE")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].scatter(ali_x[:, 0], ali_x[:, 1], s=8, alpha=0.35, label="data x")
    axes[1].scatter(ali_z[:, 0], ali_z[:, 1], s=8, alpha=0.35, label="latent z")
    axes[1].set_title("ALI + MINE")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(ixz, izy, marker="o")
    for i in [0, 4, 8, 11]:
        axes[2].annotate(rf"$\beta={beta[i]:.2f}$", (ixz[i], izy[i]), fontsize=7)
    axes[2].set_title("Information Bottleneck")
    axes[2].set_xlabel(r"compression $I(X;Z)$")
    axes[2].set_ylabel(r"prediction $I(Z;Y)$")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Applications of differentiable mutual-information objectives", y=1.06)
    _save(fig, outdir / "applications_strip.png")

    # A slightly larger gallery version for posters or slides.
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
    axes = axes.ravel()
    axes[0].scatter(gan_bad[:, 0], gan_bad[:, 1], s=7, alpha=0.35, label="GAN")
    axes[0].scatter(gan_good[:, 0], gan_good[:, 1], s=7, alpha=0.45, label="GAN + MINE")
    axes[0].set_title("Mode coverage in GANs")
    axes[0].legend(fontsize=8)
    axes[1].scatter(ali_x[:, 0], ali_x[:, 1], s=8, alpha=0.35, label="data")
    axes[1].scatter(ali_z[:, 0], ali_z[:, 1], s=8, alpha=0.35, label="latent")
    axes[1].set_title("Data-latent alignment")
    axes[1].legend(fontsize=8)
    axes[2].plot(ixz, izy, marker="o")
    axes[2].set_title("Information plane")
    axes[2].set_xlabel(r"$I(X;Z)$")
    axes[2].set_ylabel(r"$I(Z;Y)$")
    axes[2].grid(alpha=0.25)
    axes[3].axis("off")
    axes[3].text(0.04, 0.75, "MINE as a module", fontsize=13, weight="bold", transform=axes[3].transAxes)
    axes[3].text(
        0.04,
        0.38,
        "Because the estimator is differentiable,\nit can be optimized together with\ngenerative models and representation\nlearning objectives.",
        fontsize=10,
        transform=axes[3].transAxes,
    )
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, outdir / "applications_gallery.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explanatory poster figures for the MINE project.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/poster_assets"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_joint_vs_marginal(args.outdir, args.seed)
    plot_training_pipeline(args.outdir)
    plot_scn_clipping(args.outdir)
    plot_applications(args.outdir, args.seed)


if __name__ == "__main__":
    main()
