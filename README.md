# MINE Reproduction

This is a PyTorch reproduction of **Mutual Information Neural
Estimation** ([arXiv:1801.04062v5](https://arxiv.org/pdf/1801.04062)). It also includes our SCN-MINE improvement.

## Scope

Implemented:

- Donsker-Varadhan MINE objective:
  `I(X;Y) >= E_joint[T_theta(X,Y)] - log E_marginal[exp(T_theta(X,Y))]`
- EMA gradient correction for the denominator in Algorithm 1.
- Joint and product-of-marginals terms use independent sample batches. This is
  important at high MI: an in-batch permutation with fixed points can impose an
  artificial estimate near `log(batch_size)`.
- MINE-f objective:
  `E_joint[T_theta(X,Y)] - E_marginal[exp(T_theta(X,Y) - 1)]`
- Section 4.1 style correlated Gaussian MI estimation.
- Optional KSG/Kraskov nearest-neighbor baseline for Figure 1 style plots.
- Section 4.2 style nonlinear dependence experiment with 2-dimensional random
  variables and `identity`, `cubic`, `sin(x)` transforms.

## Environment

```powershell
cd C:\Users\b1565\Desktop\mine_code\mine_reproduction
conda env create -f environment.yml
conda activate mine-reproduction
```

Or with pip:

```powershell
cd C:\Users\b1565\Desktop\mine_code\mine_reproduction
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ".[kraskov]"
```

## Gaussian MI Experiments

Paper Figure 1 includes both bivariate Gaussian variables and 20-dimensional
Gaussian variables. In this repo, `--dim 1` corresponds to a bivariate Gaussian
pair `(X,Y)`, and `--dim 20` corresponds to 20 correlated coordinate pairs.

Here `dim=1` is the dimension of each variable. The plotted joint pair
`(X_a, X_b)` is 2-dimensional, so result CSV files also include
`joint_dim=2`.

### Paper-Scale Runs

For the complete left panel with MINE, MINE-f, Kraskov, and true MI:

Bivariate Gaussian:

```powershell
python .\scripts\run_gaussian_mi.py --preset reference --dim 1 --rhos -0.99 -0.9 -0.7 -0.5 -0.3 -0.1 0 0.1 0.3 0.5 0.7 0.9 0.99 --epochs 200 --samples 2000 --batch-size 500 --hidden-size 100 --activation relu --lr 1e-4 --loss biased fdiv --include-kraskov
```

20-dimensional Gaussian:

```powershell
python .\scripts\run_gaussian_mi.py --preset paper --dim 20 --rhos -0.99 -0.9 -0.7 -0.5 -0.3 -0.1 0 0.1 0.3 0.5 0.7 0.9 0.99 --steps 20000 --batch-size 256 --hidden-size 256 --eval-every 1000 --eval-batches 8 --loss mine fdiv --include-kraskov
```

Add the optional Kraskov baseline if `scikit-learn` is installed:

```powershell
python -m pip install -e ".[kraskov]"
```

If `scikit-learn` is not installed, the script prints a skip message instead
of failing the MINE experiment.

The analytic Gaussian MI is:

```text
I(X;Y) = -0.5 * dim * log(1 - rho^2)
```

For `dim=20` and `rho=0.99`, the analytic value is about `39.17`, while the
paper's finite-training MINE curve is around `23`. A 5000-step CPU run usually
underfits this point. With independent marginal samples and 20000 steps, a
diagnostic run with seed 7 reached `23.02`.

Outputs:

- `outputs/gaussian_mi_results.csv`
- `outputs/gaussian_mi.png`

## Nonlinear Dependence Experiment

The Section 4.2 style experiment uses 2-dimensional random variables by
default:

```powershell
python .\scripts\run_function_dependence.py --dim 2 --steps 1500 --batch-size 256 --sigmas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

It estimates MI for:

- `Y = X + noise`
- `Y = X^3 + noise`
- `Y = sin(X) + noise`

Outputs:

- `outputs/function_dependence_results.csv`
- `outputs/function_dependence.png`

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
python `
  scn_mine\scripts\run_synthetic_benchmark.py `
  --scenario gaussian20 --rho 0.6 `
  --methods mine smile clip_dv scn
```

Plot:

```powershell
python `
  --dim 20 `
  --rhos -0.99 -0.9 -0.7 -0.5 -0.3 -0.1 0 0.1 0.3 0.5 0.7 0.9 0.99 `
  --methods mine smile clip_dv scn `
  --steps 2000 `
  --batch-size 128 `
  --lr 0.0001 `
  --outdir scn_mine\outputs\gaussian20_rho_curve_stable
'''

