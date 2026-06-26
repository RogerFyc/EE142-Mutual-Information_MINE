from __future__ import annotations
import math
import numpy as np


EULER_GAMMA = 0.5772156649015329


def _digamma_positive_integer(n: np.ndarray | int) -> np.ndarray | float:
    """Exact psi(n) for positive integers through harmonic numbers."""
    arr = np.asarray(n, dtype=np.int64)
    if np.any(arr <= 0):
        raise ValueError("digamma approximation expects positive integer inputs")
    max_n = int(arr.max(initial=1))
    harmonic = np.zeros(max_n + 1, dtype=np.float64)
    if max_n > 1:
        harmonic[1:] = np.cumsum(1.0 / np.arange(1, max_n + 1))
    values = harmonic[arr - 1] - EULER_GAMMA
    if np.isscalar(n):
        return float(values)
    return values


def kraskov_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    try:
        from sklearn.neighbors import KDTree, NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn is required for the Kraskov baseline") from exc

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if x.shape[0] <= k:
        raise ValueError("number of samples must be larger than k")

    xy = np.concatenate([x, y], axis=1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn.fit(xy)
    distances, _ = nn.kneighbors(xy)
    eps = np.nextafter(distances[:, k], 0.0)

    x_tree = KDTree(x, metric="chebyshev")
    y_tree = KDTree(y, metric="chebyshev")
    nx = np.array([len(v) - 1 for v in x_tree.query_radius(x, eps)], dtype=np.int64)
    ny = np.array([len(v) - 1 for v in y_tree.query_radius(y, eps)], dtype=np.int64)
    n = x.shape[0]

    estimate = (
        _digamma_positive_integer(k)
        + _digamma_positive_integer(n)
        - np.mean(_digamma_positive_integer(nx + 1) + _digamma_positive_integer(ny + 1))
    )
    return float(max(estimate, 0.0))


def sample_correlated_gaussian_np(
    n: int,
    dim: int,
    rho: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    eps = rng.normal(size=(n, dim))
    y = rho * x + math.sqrt(1.0 - rho**2) * eps
    return x, y
