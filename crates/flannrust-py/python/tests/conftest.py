"""Shared fixtures/helpers for the flannrust-py pytest suite.

Distances are SQUARED for l2/l2_simple (nanoflann/flannrust semantics),
NOT euclidean like scipy.spatial.cKDTree -- every test below computes
brute-force ground truth the same (squared) way.
"""
import numpy as np
import pytest


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


def make_points(n, d, dtype, seed=0, kind="uniform"):
    """Seeded synthetic dataset. `kind`: uniform (effectively tie-free),
    clustered (tight Gaussian blobs -- near-ties likely), duplicates (~30%
    exact repeats -- exact ties guaranteed)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        pts = rng.uniform(-10.0, 10.0, size=(n, d))
    elif kind == "clustered":
        n_centers = max(1, n // 40)
        centers = rng.uniform(-10.0, 10.0, size=(n_centers, d))
        owner = rng.integers(0, n_centers, size=n)
        pts = centers[owner] + rng.normal(0.0, 0.05, size=(n, d))
    elif kind == "duplicates":
        n_base = max(1, int(n * 0.7))
        base = rng.uniform(-10.0, 10.0, size=(n_base, d))
        extra = base[rng.integers(0, n_base, size=n - n_base)]
        pts = np.concatenate([base, extra], axis=0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return np.ascontiguousarray(pts, dtype=dtype)


def bruteforce_dists(pts64, q64, metric):
    """Squared-L2 (l2/l2_simple) or summed-abs (l1) brute-force distances
    from every row of `pts64` to the single query point `q64`."""
    diff = pts64 - q64
    if metric == "l1":
        return np.sum(np.abs(diff), axis=1)
    return np.sum(diff * diff, axis=1)


def ulp_tol(dtype, value):
    """4-ULP bound (numpy's summation order differs from the Rust kernel's)."""
    eps = np.finfo(dtype).eps
    return 4.0 * eps * max(abs(float(value)), 1.0)
