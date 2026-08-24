"""Behavior tests for `flannrust.KDTree` -- errors, shapes, padding,
threads/workers determinism, eps slack, the `r=` -> rknn switch, and the
`data` attribute. Distances are SQUARED L2 (nanoflann semantics); radius
arguments are squared too.
"""
import numpy as np
import pytest

import flannrust

from conftest import make_points


# ---------------------------------------------------------------------
# Construction: dtype/shape errors, basic attributes, metric validation
# ---------------------------------------------------------------------

def test_construct_int_dtype_raises_typeerror():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    with pytest.raises(TypeError):
        flannrust.KDTree(pts)


def test_construct_3d_points_raises_valueerror():
    pts = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        flannrust.KDTree(pts)


def test_construct_1d_points_raises_valueerror():
    pts = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError):
        flannrust.KDTree(pts)


def test_basic_attributes():
    pts = make_points(15, 4, "float32", seed=5)
    tree = flannrust.KDTree(pts, leaf_size=3, metric="l1")
    assert tree.n == 15
    assert tree.dim == 4
    assert tree.leaf_size == 3
    assert tree.metric == "l1"
    assert tree.dtype == "float32"


def test_invalid_metric_raises_valueerror():
    pts = make_points(10, 3, "float32")
    with pytest.raises(ValueError):
        flannrust.KDTree(pts, metric="bogus")


@pytest.mark.parametrize("metric", ["l2", "L2", "l1", "L1", "l2_simple", "L2_SIMPLE"])
def test_metric_case_insensitive_accepted(metric):
    pts = make_points(10, 3, "float32")
    tree = flannrust.KDTree(pts, metric=metric)
    assert tree.metric == metric.lower()


def test_leaf_size_zero_raises_valueerror():
    pts = make_points(10, 3, "float32")
    with pytest.raises(ValueError):
        flannrust.KDTree(pts, leaf_size=0)


def test_empty_points_builds_empty_tree(dtype):
    pts = np.zeros((0, 3), dtype=dtype)
    tree = flannrust.KDTree(pts)
    assert tree.n == 0
    assert tree.dim == 3


# ---------------------------------------------------------------------
# query: shapes, padding, dtype/dim/k validation
# ---------------------------------------------------------------------

def test_query_shapes_1d_vs_2d(dtype):
    pts = make_points(50, 3, dtype)
    tree = flannrust.KDTree(pts)

    d1, i1 = tree.query(pts[0], k=5)
    assert d1.shape == (5,)
    assert i1.shape == (5,)

    d2, i2 = tree.query(pts[:4], k=5)
    assert d2.shape == (4, 5)
    assert i2.shape == (4, 5)


def test_query_pads_when_k_exceeds_n(dtype):
    pts = make_points(3, 2, dtype)
    tree = flannrust.KDTree(pts)
    d, i = tree.query(pts[0], k=10)
    assert i[3:].tolist() == [3] * 7  # idx == n
    assert np.all(np.isinf(d[3:]))
    assert not np.any(np.isinf(d[:3]))


def test_empty_tree_query_returns_all_padding(dtype):
    pts = np.zeros((0, 3), dtype=dtype)
    tree = flannrust.KDTree(pts)
    d, i = tree.query(np.zeros(3, dtype=dtype), k=4)
    assert np.all(i == 0)  # idx == n == 0
    assert np.all(np.isinf(d))


def test_k_must_be_positive():
    pts = make_points(10, 3, "float32")
    tree = flannrust.KDTree(pts)
    with pytest.raises(ValueError):
        tree.query(pts[0], k=0)


def test_query_dtype_mismatch_raises_typeerror():
    pts = make_points(10, 3, "float32")
    tree = flannrust.KDTree(pts)
    with pytest.raises(TypeError):
        tree.query(pts[0].astype(np.float64), k=1)


def test_query_dim_mismatch_raises_valueerror():
    pts = make_points(10, 3, "float32")
    tree = flannrust.KDTree(pts)
    with pytest.raises(ValueError):
        tree.query(np.zeros(4, dtype=np.float32), k=1)


def test_query_batched_matches_single(dtype):
    pts = make_points(200, 3, dtype)
    q = make_points(5, 3, dtype, seed=11)
    tree = flannrust.KDTree(pts)
    d_batch, i_batch = tree.query(q, k=4)
    for row in range(5):
        d_one, i_one = tree.query(q[row], k=4)
        np.testing.assert_array_equal(i_batch[row], i_one)
        np.testing.assert_array_equal(d_batch[row], d_one)


# ---------------------------------------------------------------------
# threads (build) / workers (query) determinism
# ---------------------------------------------------------------------

def test_threads_variants_identical_results(dtype):
    pts = make_points(500, 3, dtype)
    q = make_points(50, 3, dtype, seed=1)
    ref = None
    for threads in (None, 1, 4):
        tree = flannrust.KDTree(pts, threads=threads)
        d, i = tree.query(q, k=5)
        if ref is None:
            ref = (d, i)
        else:
            np.testing.assert_array_equal(i, ref[1])
            np.testing.assert_array_equal(d, ref[0])


def test_workers_parallel_matches_sequential(dtype):
    pts = make_points(2000, 3, dtype)
    q = make_points(1000, 3, dtype, seed=2)
    tree = flannrust.KDTree(pts)
    d1, i1 = tree.query(q, k=8, workers=1)
    d2, i2 = tree.query(q, k=8, workers=-1)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(d1, d2)


def test_workers_capped_matches_sequential(dtype):
    pts = make_points(500, 3, dtype)
    q = make_points(200, 3, dtype, seed=4)
    tree = flannrust.KDTree(pts)
    d1, i1 = tree.query(q, k=6, workers=1)
    d2, i2 = tree.query(q, k=6, workers=3)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(d1, d2)


def test_threads_overflow_raises_valueerror():
    # threads=2**32 used to truncate to 0 via `as u32` and panic on
    # `NonZeroU32::new(0).unwrap()` inside the FFI boundary. Must be a
    # clean ValueError instead.
    pts = make_points(10, 3, "float32")
    with pytest.raises(ValueError):
        flannrust.KDTree(pts, threads=2**32)


def test_workers_huge_is_capped_and_matches_sequential(dtype):
    # workers=2**40 used to be passed straight to
    # `rayon::ThreadPoolBuilder::num_threads`, which would spend minutes
    # spawning threads and then panic across the FFI boundary. Must be
    # capped at the CPU count and return promptly with correct results.
    pts = make_points(300, 3, dtype)
    q = make_points(50, 3, dtype, seed=6)
    tree = flannrust.KDTree(pts)
    d1, i1 = tree.query(q, k=5, workers=1)
    d2, i2 = tree.query(q, k=5, workers=2**40)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------
# r= turns query into rknn (M1 test recipe: 10 colinear points, squared
# distances 1..100 from the origin)
# ---------------------------------------------------------------------

def test_query_radius_param_partial_and_full_coverage(dtype):
    pts = np.arange(1, 11, dtype=dtype).reshape(-1, 1)
    tree = flannrust.KDTree(pts)

    # squared radius 10.0 covers only dist^2 in {1, 4, 9} -> 3 points.
    d, i = tree.query(np.zeros(1, dtype=dtype), k=5, r=10.0)
    found = int(np.sum(~np.isinf(d)))
    assert found == 3
    np.testing.assert_array_equal(i[:3], [0, 1, 2])
    assert np.all(np.isinf(d[3:]))

    # squared radius 1000.0 covers all 10 -> closest k=5 returned.
    d2, i2 = tree.query(np.zeros(1, dtype=dtype), k=5, r=1000.0)
    found2 = int(np.sum(~np.isinf(d2)))
    assert found2 == 5
    np.testing.assert_array_equal(i2, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------
# eps: multiplicative slack on the node-bound prune
# ---------------------------------------------------------------------

def test_eps_relaxes_search_bound(dtype):
    pts = make_points(3000, 3, dtype, kind="clustered")
    q = make_points(30, 3, dtype, seed=3)
    tree = flannrust.KDTree(pts)
    d_exact, _ = tree.query(q, k=5, eps=0.0)
    d_eps, _ = tree.query(q, k=5, eps=0.5)

    d_exact64 = d_exact.astype(np.float64)
    d_eps64 = d_eps.astype(np.float64)
    # guaranteed property of eps-approximate NN search: exact <= approx <=
    # (1+eps)*exact (small additive slack for float rounding).
    assert np.all(d_eps64 >= d_exact64 - 1e-6)
    assert np.all(d_eps64 <= 1.5 * d_exact64 + 1e-6)


# ---------------------------------------------------------------------
# data attribute
# ---------------------------------------------------------------------

def test_data_attribute_roundtrips(dtype):
    pts = make_points(20, 3, dtype)
    tree = flannrust.KDTree(pts)
    np.testing.assert_array_equal(tree.data, pts)
    assert tree.data.shape == (20, 3)
    assert tree.data.dtype == pts.dtype


def test_data_attribute_is_readonly(dtype):
    pts = make_points(20, 3, dtype)
    tree = flannrust.KDTree(pts)
    with pytest.raises(ValueError):
        tree.data[0, 0] = 123.0
