from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 改成你的输出目录
base = Path("outputs")
files = sorted(base.glob("gaussian_20d_seed*/gaussian_mi_results.csv"))

if not files:
    raise FileNotFoundError("No gaussian_mi_results.csv files found.")

dfs = []
for f in files:
    seed_name = f.parent.name
    df = pd.read_csv(f)
    df["seed_name"] = seed_name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# 按 rho 和 loss 取平均
summary = (
    df.groupby(["loss", "rho"], as_index=False)
      .agg(
          estimate_mean=("mine_estimate", "mean"),
          estimate_std=("mine_estimate", "std"),
          true_mi=("true_mi", "mean"),
      )
)

out_csv = base / "gaussian_20d_3seed_average.csv"
summary.to_csv(out_csv, index=False)

# 画均值曲线
plt.figure(figsize=(7, 4.5))

label_map = {
    "mine": "MINE",
    "fdiv": "MINE-f",
    "biased": "MINE biased",
    "kraskov": "Kraskov",
}

for loss_name in ["mine", "fdiv", "kraskov"]:
    sub = summary[summary["loss"] == loss_name].sort_values("rho")
    if sub.empty:
        continue

    plt.errorbar(
        sub["rho"],
        sub["estimate_mean"],
        yerr=sub["estimate_std"],
        marker="o",
        capsize=3,
        label=label_map.get(loss_name, loss_name),
    )

# True MI 只画一次
truth = summary[summary["loss"] == summary["loss"].iloc[0]].sort_values("rho")
plt.plot(
    truth["rho"],
    truth["true_mi"],
    linestyle="--",
    marker="x",
    label="True MI",
)

plt.xlabel("correlation rho")
plt.ylabel("mutual information")
plt.title("Mutual Information of 20-dimensional variables, averaged over 3 seeds")
plt.legend()
plt.tight_layout()

out_fig = base / "gaussian_20d_3seed_average.png"
plt.savefig(out_fig, dpi=160)

print(f"Saved: {out_csv}")
print(f"Saved: {out_fig}")