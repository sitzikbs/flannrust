"""Behavior tests for `flannrust.KDTree.query_radius` / `.query_box` --
strict `<` at an exact radius boundary, `sorted=False` traversal order,
inclusive box-face boundaries, and box search on an empty tree.

Distances (and the `r=` radius argument) are SQUARED for l2/l2_simple, same
convention as `query` -- see conftest.py / test_behavior.py.
"""
import numpy as np
import pytest

import flannrust

from conftest import make_points


# ---------------------------------------------------------------------
# query_radius: strict `<` at an exact boundary point
# ---------------------------------------------------------------------

def test_query_radius_strict_less_than_at_exact_boundary(dtype):
    # Point (2, 0) is at EXACTLY squared distance 4.0 from the origin.
    pts = np.array([[2.0, 0.0], [0.0, 1.0], [10.0, 10.0]], dtype=dtype)
    tree = flannrust.KDTree(pts)

    idxs, dists = tree.query_radius(np.zeros(2, dtype=dtype), r=4.0)
    assert 0 not in idxs[0].tolist(), "exact-boundary point must be excluded"

    # nextafter in the TREE's own dtype -- `r` is passed through as f64, but
    # gets narrowed to the tree's T before comparison (`T::from_f64`), so a
    # float64 nextafter(4.0) is indistinguishable from 4.0 once narrowed to
    # float32.
    r_next_up = float(np.nextafter(np.array(4.0, dtype=dtype), np.inf))
    idxs2, dists2 = tree.query_radius(np.zeros(2, dtype=dtype), r=r_next_up)
    assert 0 in idxs2[0].tolist(), "nextafter(r, inf) must include the boundary point"


# ---------------------------------------------------------------------
# query_radius: sorted=True gives ascending order, sorted=False gives raw
# traversal order (exact index sequence locked, matching the Rust core's
# `radius_search_sorted_true_gives_ascending_sorted_false_gives_traversal_order`
# unit test construction).
# ---------------------------------------------------------------------

def test_query_radius_sorted_false_gives_traversal_order(dtype):
    pts = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=dtype)
    tree = flannrust.KDTree(pts, leaf_size=2)

    idxs_unsorted, _ = tree.query_radius(np.array([0.4], dtype=dtype), r=200.0, sorted=False)
    np.testing.assert_array_equal(idxs_unsorted[0], [0, 1, 3, 2])

    idxs_sorted, dists_sorted = tree.query_radius(np.array([0.4], dtype=dtype), r=200.0, sorted=True)
    d = dists_sorted[0].astype(np.float64)
    assert np.all(np.diff(d) >= 0), "sorted=True must yield ascending distances"
    # same set either way
    assert set(idxs_unsorted[0].tolist()) == set(idxs_sorted[0].tolist())


# ---------------------------------------------------------------------
# query_radius: (d,) query returns lists of length 1
# ---------------------------------------------------------------------

def test_query_radius_1d_query_returns_length_1_lists(dtype):
    pts = make_points(30, 3, dtype, seed=7)
    tree = flannrust.KDTree(pts)
    idxs, dists = tree.query_radius(pts[0], r=50.0)
    assert len(idxs) == 1
    assert len(dists) == 1
    assert idxs[0].dtype == np.uint32


# ---------------------------------------------------------------------
# query_box: inclusive face boundaries on a 4x4x4 grid (matches the Rust
# core's `find_within_box_face_inclusion` unit test).
# ---------------------------------------------------------------------

def test_query_box_face_inclusion(dtype):
    pts = np.array(
        [[x, y, z] for x in range(4) for y in range(4) for z in range(4)],
        dtype=dtype,
    )
    tree = flannrust.KDTree(pts)

    lo = np.array([1.0, 0.0, 2.0], dtype=dtype)
    hi = np.array([2.0, 3.0, 2.0], dtype=dtype)
    got = set(tree.query_box(lo, hi).tolist())

    for corner in ([1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 2.0]):
        idx = int(np.where((pts == np.array(corner, dtype=dtype)).all(axis=1))[0][0])
        assert idx in got, f"face point {corner} (index {idx}) must be included (inclusive boundaries)"


# ---------------------------------------------------------------------
# query_box: exact traversal-order lock (matches the Rust core's
# `find_within_box_exact_traversal_order` unit test construction).
# ---------------------------------------------------------------------

def test_query_box_exact_traversal_order(dtype):
    pts = np.arange(1, 11, dtype=dtype).reshape(-1, 1)
    tree = flannrust.KDTree(pts, leaf_size=2)

    got = tree.query_box(np.array([2.0], dtype=dtype), np.array([9.0], dtype=dtype))
    np.testing.assert_array_equal(got, [8, 7, 6, 5, 4, 3, 2, 1])
    assert got.dtype == np.uint32


# ---------------------------------------------------------------------
# query_box: empty tree -> empty uint32 array, no crash.
# ---------------------------------------------------------------------

def test_query_box_empty_tree(dtype):
    pts = np.zeros((0, 3), dtype=dtype)
    tree = flannrust.KDTree(pts)
    got = tree.query_box(np.full(3, -1.0, dtype=dtype), np.full(3, 1.0, dtype=dtype))
    assert got.shape == (0,)
    assert got.dtype == np.uint32
