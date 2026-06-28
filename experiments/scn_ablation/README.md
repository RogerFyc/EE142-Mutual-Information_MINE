# SCN-MINE schedule/loss ablation

This experiment addresses the question: why use the current SCN-MINE mixture
weight and clipping threshold?

It compares original MINE, InfoNCE, Clip-DV, fixed SCN mixtures, and annealed
SCN mixtures.  The important metrics are RMSE to the analytic Gaussian MI,
seed standard deviation, mean absolute error, and saturation ratio.

Run from the repository root:

```bash
python experiments/scn_ablation/run_schedule_ablation.py \
  --scenario gaussian20 --rho 0.99 --steps 1200 --batch-size 256 --seeds 0 1 2
```

Fast smoke test:

```bash
python experiments/scn_ablation/run_schedule_ablation.py --steps 100 --seeds 0
```

Outputs:

- `outputs/scn_ablation/schedule_ablation_results.csv`
- `outputs/scn_ablation/schedule_ablation_summary.csv`
- `outputs/scn_ablation/scn_schedule_ablation_summary.png`
- `outputs/scn_ablation/scn_schedule_ablation_frontier.png`
