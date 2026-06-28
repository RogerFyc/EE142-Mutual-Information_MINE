# Application: paired data quality auditing

This application compares original MINE with SCN-MINE on a more concrete task:
MNIST image-label pairing quality.  We randomly corrupt a known fraction of
labels and estimate `I(image; label)`.  As the corruption rate increases, the
estimated MI should decrease.

Run from the repository root:

```bash
python applications/paired_data_audit/run_mnist_pair_audit.py \
  --methods mine scn \
  --corruptions 0 0.2 0.4 0.6 0.8 1.0 \
  --steps 800 --batch-size 256 --seeds 0 1 2
```

Fast smoke test:

```bash
python applications/paired_data_audit/run_mnist_pair_audit.py --steps 100 --seeds 0 --corruptions 0 1
```

Outputs:

- `outputs/paired_data_audit/mnist_pair_audit_examples.png`
- `outputs/paired_data_audit/mnist_pair_audit_results.csv`
- `outputs/paired_data_audit/mnist_pair_audit_summary.csv`
- `outputs/paired_data_audit/mnist_pair_audit_curve.png`
- `outputs/paired_data_audit/mnist_pair_audit_variance.png`
