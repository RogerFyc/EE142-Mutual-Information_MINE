# SCN-MINE

SCN-MINE applies

```text
s(x, y) = c * tanh(a(x, y) / c)
```

During training it maximizes an annealed auxiliary objective:

```text
lambda_t * clipped_DV + (1 - lambda_t) * InfoNCE
```

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
