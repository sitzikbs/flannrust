"""Behavior tests for `flannrust.DynamicKDTree` -- add_points/remove_point
churn checked against a fresh static `KDTree` rebuilt over the same live
rows. Distances are SQUARED for l2/l2_simple, same convention as the
static KDTree -- see conftest.py.
"""
import numpy as np
import pytest

import flannrust

from conftest import make_points


def _live_pts_and_idx(all_pts, removed):
    live_idx = [i for i in range(len(all_pts)) if i not in removed]
    return all_pts[live_idx], live_idx


def _assert_matches_static(tree, all_pts, removed, queries, k):
    live_pts, live_idx = _live_pts_and_idx(all_pts, removed)
    if len(live_pts) == 0:
        return
    ref = flannrust.KDTree(live_pts, metric="l2")
    kk = min(k, len(live_pts))
    for q in queries:
        d_dyn, i_dyn = tree.query(q, k=kk)
        d_ref, i_ref = ref.query(q, k=kk)
        mapped = sorted(live_idx[j] for j in i_ref.tolist())
        assert sorted(i_dyn.tolist()) == mapped
        np.testing.assert_allclose(
            np.sort(d_dyn.astype(np.float64)),
            np.sort(d_ref.astype(np.float64)),
            rtol=1e-4,
            atol=1e-4,
        )


# ---------------------------------------------------------------------
# Construction validation (mirrors the static KDTree's TypeError/
# ValueError split).
# ---------------------------------------------------------------------

def test_construct_invalid_params_raise():
    with pytest.raises(ValueError):
        flannrust.DynamicKDTree(dim=0)
    with pytest.raises(ValueError):
        flannrust.DynamicKDTree(dim=2, leaf_size=0)
    with pytest.raises(ValueError):
        flannrust.DynamicKDTree(dim=2, metric="bogus")
    with pytest.raises(ValueError):
        flannrust.DynamicKDTree(dim=2, dtype="bogus")
    with pytest.raises(ValueError):
        flannrust.DynamicKDTree(dim=2, capacity=0)


def test_basic_attributes():
    tree = flannrust.DynamicKDTree(dim=4, dtype="float64", leaf_size=3, metric="l1", capacity=100)
    assert tree.dim == 4
    assert tree.dtype == "float64"
    assert tree.leaf_size == 3
    assert tree.metric == "l1"
    assert tree.capacity == 100
    assert tree.n_total == 0
    assert tree.n_active == 0
    assert tree.removed_len == 0


# ---------------------------------------------------------------------
# add_points in 3 batches (incl. a (d,) single point) -> knn matches a
# fresh static KDTree over the same (all still-live) rows.
# ---------------------------------------------------------------------

def test_add_in_three_batches_matches_static(dtype):
    all_pts = make_points(30, 3, dtype, seed=1)
    tree = flannrust.DynamicKDTree(dim=3, dtype=dtype)

    s, e = tree.add_points(all_pts[0:10])
    assert (s, e) == (0, 10)

    # single point via (d,) shape -- exercises the start==end_inclusive
    # off-by-one at batch size 1.
    s, e = tree.add_points(all_pts[10])
    assert (s, e) == (10, 11)

    s, e = tree.add_points(all_pts[11:30])
    assert (s, e) == (11, 30)

    assert tree.n_total == 30
    assert tree.n_active == 30

    _assert_matches_static(tree, all_pts, removed=set(), queries=all_pts[:5], k=5)


def test_add_empty_batch_is_noop_and_returns_start_start(dtype):
    tree = flannrust.DynamicKDTree(dim=2, dtype=dtype)
    pts = make_points(5, 2, dtype, seed=9)
    tree.add_points(pts)
    assert tree.n_total == 5

    empty = np.zeros((0, 2), dtype=dtype)
    s, e = tree.add_points(empty)
    assert (s, e) == (5, 5)
    assert tree.n_total == 5


# ---------------------------------------------------------------------
# remove_point excludes from results; a second remove of the same index
# returns False.
# ---------------------------------------------------------------------

def test_remove_point_excludes_and_second_remove_returns_false(dtype):
    pts = make_points(20, 2, dtype, seed=2)
    tree = flannrust.DynamicKDTree(dim=2, dtype=dtype)
    tree.add_points(pts)

    assert tree.remove_point(0) is True
    assert tree.n_active == 19
    assert tree.removed_len == 1
    assert tree.remove_point(0) is False

    d, i = tree.query(pts[0], k=1)
    assert 0 not in i.tolist()

    _assert_matches_static(tree, pts, removed={0}, queries=pts[1:6], k=3)


# ---------------------------------------------------------------------
# Interleaved add/remove churn (seeded, 500 ops), checked against a
# static rebuild every 100 ops.
# ---------------------------------------------------------------------

def test_interleaved_churn_matches_static_rebuild_every_100_ops(dtype):
    rng = np.random.default_rng(42)
    dim = 3
    pool = make_points(700, dim, dtype, seed=3)
    tree = flannrust.DynamicKDTree(dim=dim, dtype=dtype)

    all_pts = np.empty((0, dim), dtype=dtype)
    removed = set()
    next_pool_idx = 0

    for op in range(500):
        do_add = rng.random() < 0.6 and next_pool_idx < len(pool)
        if do_add:
            batch = int(min(rng.integers(1, 4), len(pool) - next_pool_idx))
            new_rows = pool[next_pool_idx : next_pool_idx + batch]
            tree.add_points(new_rows)
            all_pts = np.concatenate([all_pts, new_rows], axis=0)
            next_pool_idx += batch
        else:
            candidates = [i for i in range(len(all_pts)) if i not in removed]
            if candidates:
                idx = int(rng.choice(candidates))
                assert tree.remove_point(idx) is True
                removed.add(idx)

        if (op + 1) % 100 == 0:
            live_pts, _ = _live_pts_and_idx(all_pts, removed)
            queries = live_pts[:5] if len(live_pts) else all_pts[:0]
            _assert_matches_static(tree, all_pts, removed, queries, k=3)

    live_pts, _ = _live_pts_and_idx(all_pts, removed)
    _assert_matches_static(tree, all_pts, removed, live_pts[:5], k=3)


# ---------------------------------------------------------------------
# capacity honored: overflow add raises ValueError before touching the
# core (dataset/bookkeeping left untouched).
# ---------------------------------------------------------------------

def test_capacity_honored_overflow_raises_before_touching_core(dtype):
    pts = make_points(5, 2, dtype, seed=4)
    tree = flannrust.DynamicKDTree(dim=2, dtype=dtype, capacity=5)
    tree.add_points(pts)
    assert tree.n_total == 5

    extra = make_points(1, 2, dtype, seed=5)
    with pytest.raises(ValueError):
        tree.add_points(extra)

    assert tree.n_total == 5
    assert tree.n_active == 5


# ---------------------------------------------------------------------
# dtype fixed at construction: mismatched add_points dtype -> TypeError.
# ---------------------------------------------------------------------

def test_add_points_dtype_mismatch_raises_typeerror():
    tree = flannrust.DynamicKDTree(dim=2, dtype="float32")
    bad = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(TypeError):
        tree.add_points(bad)


# ---------------------------------------------------------------------
# Queries on an empty dynamic tree.
# ---------------------------------------------------------------------

def test_query_on_empty_dynamic_tree_returns_padding(dtype):
    tree = flannrust.DynamicKDTree(dim=3, dtype=dtype)
    assert tree.n_total == 0
    assert tree.n_active == 0

    d, i = tree.query(np.zeros(3, dtype=dtype), k=1)
    assert i[0] == 0  # pad_idx == n_total == 0
    assert np.isinf(float(d[0]))

    idxs, dists = tree.query_radius(np.zeros(3, dtype=dtype), r=100.0)
    assert len(idxs[0]) == 0
