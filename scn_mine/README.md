# SCN-MINE

Independent implementation of the Word report's proposed **Smooth-Clipped
Noise-contrastive MINE**. This directory does not import from or modify
`mine_reproduction`.

SCN-MINE applies

```text
s(x, y) = c * tanh(a(x, y) / c)
```

During training it maximizes an annealed auxiliary objective:

```text
lambda_t * clipped_DV + (1 - lambda_t) * InfoNCE
```

The defaults anneal `clip: 5 -> 10` and `lambda: 0.7 -> 1.0`. InfoNCE is only
an early optimization aid. Checkpoint selection and the final reported MI use
held-out `clipped_DV`, with an EMA-stabilized denominator gradient during
training.

## Included

- annealed SCN-MINE with `c: 5 -> 10`, `lambda: 0.7 -> 1.0`,
  `temperature=0.7`
- pure clipped-DV + EMA baseline (`clip_dv`)
- MINE/DV with the original EMA gradient correction, plus InfoNCE, NWJ and
  SMILE baselines
- shared all-pairs joint critic
- held-out peak early stopping with best-checkpoint restoration
- correlated-Gaussian and nonlinear nuisance synthetic tasks
- analytic Gaussian MI, bootstrap intervals, independence diagnostic, and
  score-saturation reporting
- local-MNIST two-view pretraining, frozen linear probe, and matching recall@1

## Run

Use the existing `mine-reproduction` Conda environment:

```powershell
python `
  scn_mine\scripts\run_synthetic_benchmark.py `
  --scenario low --methods mine smile clip_dv scn `
  --steps 2000 --batch-size 128
```

High-dimensional nonlinear nuisance experiment:

```powershell
python `
  scn_mine\scripts\run_synthetic_benchmark.py `
  --scenario high --methods mine smile clip_dv scn
```

Twenty-dimensional Gaussian experiment:

```powershell
\python `
  scn_mine\scripts\run_synthetic_benchmark.py `
  --scenario gaussian20 --rho 0.6 `
  --methods mine smile clip_dv scn
```

MNIST two-view experiment (reads the existing raw IDX files without modifying
them):

```powershell
python `
  scn_mine\scripts\run_mnist_views.py `
  --methods mine infonce nwj smile scn --steps 1500
```

Outputs are written under `scn_mine/outputs/`, never under
`mine_reproduction/`.

The synthetic CSV includes the analytic truth, held-out clipped-DV estimate, bias,
absolute error, bootstrap interval, shuffled-pair independence estimate, and
score saturation ratio. JSON history files preserve the fixed train-eval
estimate, held-out report estimate, mixed training objective, active clip/mix,
learning rate, and saturation ratio. The script also generates estimate,
bias-variance, and per-seed convergence figures.

## Preliminary evidence

A controlled CPU experiment was run on the report's high-MI Gaussian setting:
4 correlated dimensions, `rho=0.95`, true MI `4.6558` nats, batch size 64,
800 steps, and three seeds. Every method used the same critic size and training
budget.

| Method | Held-out MI (mean +/- seed std) | Mean absolute error |
| --- | ---: | ---: |
| Annealed SCN-MINE | 4.1646 +/- 0.0634 | 0.4912 |
| Pure clipped-DV + EMA | 4.1271 +/- 0.0718 | 0.5287 |
| SMILE (logistic ratio training) | 4.0588 +/- 0.0408 | 0.5970 |
| MINE | 4.0520 +/- 0.0852 | 0.6038 |

Under this limited protocol, annealed SCN-MINE had both lower mean absolute
error and lower seed standard deviation than MINE. This is preliminary support
for the modification plan, not a publication-level claim: the comparison has
only three seeds, 800 steps, and batch size 64.

The original SMILE result was invalid because an unbounded SMILE estimator was
incorrectly optimized directly. The implementation now trains SMILE with
joint-vs-product logistic density-ratio classification and only uses the
clipped expression for evaluation. Old SMILE values under
`outputs/evidence_low_rho095/` and graphs showing thousand-level estimates must
not be cited.
The evidence CSV and curves are under:

```text
outputs/evidence_low_rho095/
outputs/evidence_scn_tuned/
outputs/evidence_scn_dv_heavy/
outputs/corrected_comparison_3seeds/
outputs/annealed_comparison_3seeds/
```

Three seeds on one synthetic setting are not sufficient for a publication
claim. The high-dimensional nuisance and MNIST multi-seed experiments remain
necessary, along with a separately trained independence self-consistency test.
