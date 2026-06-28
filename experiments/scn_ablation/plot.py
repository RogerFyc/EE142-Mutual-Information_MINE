import os

# Avoid duplicated OpenMP runtime error on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    outdir = Path("outputs/scn_ablation")
    summary_csv = outdir / "schedule_ablation_summary.csv"
    results_csv = outdir / "schedule_ablation_results.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Cannot find {summary_csv}")

    summary = pd.read_csv(summary_csv)

    true_mi = None
    rho = None
    if results_csv.exists():
        results = pd.read_csv(results_csv)
        if "true_mi" in results.columns:
            true_mi = float(results["true_mi"].iloc[0])
        if "rho" in results.columns:
            rho = float(results["rho"].iloc[0])

    labels = summary["case"].astype(str).tolist()
    x = list(range(len(labels)))

    # Figure 1: RMSE and seed-level standard deviation
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].bar(x, summary["rmse"].astype(float))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("RMSE to analytic MI (nats)")
    axes[0].set_title("Accuracy of schedule/loss choices")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary["seed_std"].astype(float))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Seed standard deviation (nats)")
    axes[1].set_title("Stability across random seeds")
    axes[1].grid(axis="y", alpha=0.25)

    title = "SCN-MINE schedule/loss ablation"
    if rho is not None and true_mi is not None:
        title += f" at rho={rho:.2f}; true MI={true_mi:.2f}"
    fig.suptitle(title)

    fig_path = outdir / "scn_schedule_ablation_summary.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: bias--variance frontier
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for _, row in summary.iterrows():
        mae = float(row["mean_absolute_error"])
        std = float(row["seed_std"])
        name = str(row["case"])
        ax.scatter(mae, std, s=70)
        ax.annotate(name, (mae, std), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Mean absolute error (nats)")
    ax.set_ylabel("Seed standard deviation (nats)")
    ax.set_title("Bias--variance frontier for SCN design choices")
    ax.grid(alpha=0.25)

    fig_path2 = outdir / "scn_schedule_ablation_frontier.png"
    fig.savefig(fig_path2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Optional Figure 3: saturation ratio
    if "mean_saturation_ratio" in summary.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x, summary["mean_saturation_ratio"].astype(float))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Mean saturation ratio")
        ax.set_title("How often critic scores exceed the clipping range")
        ax.grid(axis="y", alpha=0.25)

        fig_path3 = outdir / "scn_schedule_ablation_saturation.png"
        fig.savefig(fig_path3, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {outdir / 'scn_schedule_ablation_summary.png'}")
    print(f"Saved {outdir / 'scn_schedule_ablation_frontier.png'}")
    if (outdir / "scn_schedule_ablation_saturation.png").exists():
        print(f"Saved {outdir / 'scn_schedule_ablation_saturation.png'}")


if __name__ == "__main__":
    main()