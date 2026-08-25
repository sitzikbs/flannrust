"""Brute-force ground-truth tests for `flannrust.KDTree.query`, across
seeded numpy datasets (uniform / clustered / 30% duplicates), dims
{2, 3, 8, 32}, f32/f64, and all three metrics.

For tie-free data (`uniform`, continuous randoms -> ties vanishingly
unlikely), we additionally require the returned index sequence to equal
numpy's `argpartition`/`argsort` ground truth exactly. For datasets that
can legitimately tie (`clustered`, `duplicates`), we only require the
*distance* multiset to match (within 4 ULP) plus per-result
self-consistency, since insertion-order tie-breaking makes a particular
index sequence non-unique.
"""
import numpy as np
import pytest

import flannrust

from conftest import make_points, bruteforce_dists, ulp_tol

DIMS = [2, 3, 8, 32]
DTYPES = ["float32", "float64"]
METRICS = ["l2", "l1", "l2_simple"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS)
def test_query_matches_bruteforce_uniform_tiefree(dim, dtype, metric):
    n, k, nq = 400, 6, 15
    pts = make_points(n, dim, dtype, seed=1000 + dim, kind="uniform")
    q = make_points(nq, dim, dtype, seed=2000 + dim, kind="uniform")
    tree = flannrust.KDTree(pts, metric=metric)
    d, i = tree.query(q, k=k)

    pts64 = pts.astype(np.float64)
    for row in range(nq):
        bf = bruteforce_dists(pts64, q[row].astype(np.float64), metric)
        sorted_bf = np.sort(bf)
        if np.isclose(sorted_bf[k - 1], sorted_bf[k], rtol=0, atol=1e-9):
            continue  # boundary tie (rare with continuous uniform data): skip
        top_k = np.argsort(bf, kind="stable")[:k]

        np.testing.assert_array_equal(i[row], top_k)
        for j in range(k):
            assert abs(float(d[row, j]) - float(sorted_bf[j])) <= ulp_tol(dtype, sorted_bf[j])


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["clustered", "duplicates"])
def test_query_matches_bruteforce_distance_multiset(dtype, metric, kind):
    dim, n, k, nq = 3, 400, 6, 15
    pts = make_points(n, dim, dtype, seed=42, kind=kind)
    q = make_points(nq, dim, dtype, seed=43, kind="uniform")
    tree = flannrust.KDTree(pts, metric=metric)
    d, i = tree.query(q, k=k)

    pts64 = pts.astype(np.float64)
    for row in range(nq):
        bf = bruteforce_dists(pts64, q[row].astype(np.float64), metric)
        sorted_bf = np.sort(bf)[:k]
        got_sorted = np.sort(d[row].astype(np.float64))
        for j in range(k):
            assert abs(got_sorted[j] - sorted_bf[j]) <= ulp_tol(dtype, sorted_bf[j])

        # self-consistency: each claimed distance matches its claimed index
        for j in range(k):
            idx = int(i[row, j])
            actual = bruteforce_dists(pts64[idx:idx + 1], q[row].astype(np.float64), metric)[0]
            assert abs(actual - float(d[row, j])) <= ulp_tol(dtype, actual)
