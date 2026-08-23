//! TDD tests for the C++ `KDTreeSingleIndexDynamicAdaptor` oracle (M2 Task
//! 1), written against the safe Rust API described in the task-1 brief
//! BEFORE that API exists (RED), then made to pass once `cpp/wrapper.cpp` /
//! `src/lib.rs` implement the `nfrd_*` extern-C surface and
//! `RefDynIndexF32`/`RefDynIndexF64` (GREEN).
//!
//! L2 metric only (M2's dynamic cross-validation scope) -- see
//! `cpp/wrapper.cpp`'s `RowMajorDynAdaptor`/`DynTree` for why the dynamic
//! oracle does not expose the L1/SO2/SO3 switch that the static oracle does.
//!
//! Fixed dim-2 f32 dataset used throughout (row-major, n=8, dim=2), chosen so
//! every pairwise distance from the query is DISTINCT (no ties -- ties would
//! be broken by insertion order per `NANOFLANN_FIRST_MATCH` being undefined,
//! which is deterministic but adds noise to hand-verification):
//!   0: (0,0)   1: (1,0)   2: (2,0)   3: (3,0)
//!   4: (0,1)   5: (1,1)   6: (2,1)   7: (3,1)
//! query: (0.2, 0.3)
//!
//! Hand-computed SQUARED L2 distances from the query:
//!   to 0: 0.2^2 + 0.3^2               = 0.04 + 0.09 = 0.13
//!   to 1: (1-0.2)^2 + 0.3^2           = 0.64 + 0.09 = 0.73
//!   to 2: (2-0.2)^2 + 0.3^2           = 3.24 + 0.09 = 3.33
//!   to 3: (3-0.2)^2 + 0.3^2           = 7.84 + 0.09 = 7.93
//!   to 4: 0.2^2 + (1-0.3)^2           = 0.04 + 0.49 = 0.53
//!   to 5: (1-0.2)^2 + (1-0.3)^2       = 0.64 + 0.49 = 1.13
//!   to 6: (2-0.2)^2 + (1-0.3)^2       = 3.24 + 0.49 = 3.73
//!   to 7: (3-0.2)^2 + (1-0.3)^2       = 7.84 + 0.49 = 8.33
//! Full ascending order (all 8 live): 0, 4, 1, 5, 2, 6, 3, 7
//!   (0.13, 0.53, 0.73, 1.13, 3.33, 3.73, 7.93, 8.33)
//! Restricted to {0,1,2,3} (test 2/4): ascending 0(0.13), 1(0.73), 2(3.33), 3(7.93)
//! Restricted to {0,2,3} (test 3, point 1 removed): ascending 0(0.13), 2(3.33), 3(7.93)

use nanoflann_ref::{RefDynIndexF32, RefDynIndexF64};

const DIM: usize = 2;
const PTS_F32: [f32; 16] = [
    0.0, 0.0, // 0
    1.0, 0.0, // 1
    2.0, 0.0, // 2
    3.0, 0.0, // 3
    0.0, 1.0, // 4
    1.0, 1.0, // 5
    2.0, 1.0, // 6
    3.0, 1.0, // 7
];
const PTS_F64: [f64; 16] = [
    0.0, 0.0, // 0
    1.0, 0.0, // 1
    2.0, 0.0, // 2
    3.0, 0.0, // 3
    0.0, 1.0, // 4
    1.0, 1.0, // 5
    2.0, 1.0, // 6
    3.0, 1.0, // 7
];
const QUERY_F32: [f32; 2] = [0.2, 0.3];
const QUERY_F64: [f64; 2] = [0.2, 0.3];

const LEAF_MAX_SIZE: usize = 10;
const MAX_POINT_COUNT: usize = 1000;

// ---- Test 1: empty forest (current_n=0, no addPoints call at all) --------
//
// C++'s ctor auto-adds existing points ONLY if
// dataset_.kdtree_get_point_count() > 0 at construction time; our
// RowMajorDynAdaptor starts current_n at 0 so the ctor's auto-add path never
// fires (see cpp/wrapper.cpp's nfrd_build_impl). findNeighbors on the outer
// KDTreeSingleIndexDynamicAdaptor just loops treeCount_ empty sub-trees, each
// of which returns false immediately (size()==0) without touching
// root_node_ -- no special emptiness check anywhere, and no crash.

#[test]
fn empty_forest_knn_returns_zero_found_f32() {
    let idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.0);
    assert_eq!(indices.len(), 0, "empty forest must return 0 found");
    assert_eq!(dists.len(), 0);
}

#[test]
fn empty_forest_radius_returns_zero_found_f32() {
    let idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    // Document the OBSERVED C++ behavior on an empty forest: every sub-tree
    // is empty, radiusSearchCustomCallback -> findNeighbors loops
    // zero-or-empty sub-trees (each bails out on size()==0), so the result
    // count is 0 and result.full() (always true for RadiusResultSet) does
    // not crash despite nothing ever being added to it.
    let results = idx.radius(&QUERY_F32, 100.0, true, 0.0);
    assert_eq!(results.len(), 0, "empty forest radius search must return 0 found");
}

// ---- Test 2: set_current_n + addPoints(0, 3) (end-inclusive) -------------

#[test]
fn add_points_0_to_3_matches_hand_computed_knn_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3); // END-INCLUSIVE: adds 0, 1, 2, 3 (four points, not three)

    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.0);
    assert_eq!(indices, vec![0, 1], "expected nearest-2 indices [0, 1]");
    assert!((dists[0] - 0.13).abs() < 1e-6, "dist[0] = {} (want ~0.13)", dists[0]);
    assert!((dists[1] - 0.73).abs() < 1e-6, "dist[1] = {} (want ~0.73)", dists[1]);

    assert!(idx.tree_count() >= 1, "tree_count() must be >= 1");

    // Union of all slots' vAcc_ must be exactly {0,1,2,3} (Bentley-Saxe
    // forest: every physically-present point lives in exactly one sub-tree's
    // vAcc_ at any given time).
    let mut union: Vec<u32> = Vec::new();
    for slot in 0..idx.tree_count() {
        union.extend(idx.slot_vacc(slot));
    }
    union.sort_unstable();
    assert_eq!(union, vec![0, 1, 2, 3], "slot vAcc union must be exactly {{0,1,2,3}}");

    let ti = idx.tree_index();
    for (i, &v) in ti.iter().enumerate().take(4) {
        assert_ne!(v, -1, "tree_index()[{i}] must not be -1 (point {i} is active)");
    }
}

// ---- Test 3: removePoint (lazy deletion) ----------------------------------

#[test]
fn remove_point_1_excludes_it_from_knn_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3);

    idx.remove_point(1);

    // knn over the whole live set (k=4, more than the 3 remaining live
    // points) must never return 1.
    let (indices, _dists) = idx.knn(&QUERY_F32, 4, 0.0);
    assert!(!indices.contains(&1), "removed point 1 must never appear in knn results, got {:?}", indices);
    assert_eq!(indices, vec![0, 2, 3], "expected ascending [0, 2, 3] over the live set");

    let ti = idx.tree_index();
    assert_eq!(ti[1], -1, "tree_index()[1] must be -1 after removal");
    assert_eq!(idx.removed_count(), 1, "removed_count() must be 1");
}

// ---- Test 4: re-add (reactivation, not duplicate insertion) ---------------

#[test]
fn readd_point_1_reactivates_it_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3);
    idx.remove_point(1);

    idx.add_points(1, 1); // end-inclusive single-point re-add: reactivation, not a duplicate insert

    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.0);
    assert_eq!(indices, vec![0, 1], "point 1 must be queryable again after reactivation");
    assert!((dists[0] - 0.13).abs() < 1e-6);
    assert!((dists[1] - 0.73).abs() < 1e-6);
    assert_eq!(idx.removed_count(), 0, "removed_count() must be 0 after reactivation");

    let ti = idx.tree_index();
    assert_ne!(ti[1], -1, "tree_index()[1] must not be -1 after reactivation");

    // Reactivation must not duplicate the point: the union of live vAcc_
    // entries (points whose tree_index() != -1) must still be exactly 4.
    let mut union: Vec<u32> = Vec::new();
    for slot in 0..idx.tree_count() {
        union.extend(idx.slot_vacc(slot));
    }
    union.sort_unstable();
    assert_eq!(union.len(), 4, "vAcc_ must hold exactly 4 physical entries, no duplicate");
    assert_eq!(union, vec![0, 1, 2, 3]);
}

// ---- Test 5: grow to 8 points (Bentley-Saxe merge across slots) ----------

#[test]
fn grow_to_8_points_all_queryable_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3);
    idx.remove_point(1);
    idx.add_points(1, 1); // net effect after test 2-4's sequence: 0..3 all live, none removed

    idx.set_current_n(8);
    idx.add_points(4, 7); // END-INCLUSIVE: adds 4, 5, 6, 7

    // All 8 points must be queryable: full ascending order is
    // 0, 4, 1, 5, 2, 6, 3, 7 (see module doc comment).
    let (indices, dists) = idx.knn(&QUERY_F32, 8, 0.0);
    assert_eq!(indices, vec![0, 4, 1, 5, 2, 6, 3, 7], "expected the full hand-computed ascending order");
    let expected_dists = [0.13f32, 0.53, 0.73, 1.13, 3.33, 3.73, 7.93, 8.33];
    for (got, want) in dists.iter().zip(expected_dists.iter()) {
        assert!((got - want).abs() < 1e-5, "got={got} want={want}");
    }

    // Bentley-Saxe merge sanity: with no points currently removed, the sum
    // of every slot's vAcc_ length must equal the live point count (8) --
    // consistent with First0Bit-driven slot occupancy after 8 sequential
    // inserts (binary counter pattern: slot 3 alone holds all 8, or some
    // other consistent partition depending on the exact merge history).
    let total_vacc: usize = (0..idx.tree_count()).map(|slot| idx.slot_vacc(slot).len()).sum();
    assert_eq!(total_vacc, 8, "sum of slot vAcc_ lengths must equal the live point count");
}

// ---- Test 6: radius two-call round-trip on the live (8-point) set --------

#[test]
fn radius_round_trip_sorted_ascending_on_live_set_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(8);
    idx.add_points(0, 7);

    // radius_sq = 5.0: points with squared distance < 5.0 are
    // 0(0.13),4(0.53),1(0.73),5(1.13),2(3.33),6(3.73) -- 3(7.93) and 7(8.33)
    // are excluded. Ascending order: 0,4,1,5,2,6.
    let results = idx.radius(&QUERY_F32, 5.0, true, 0.0);
    let indices: Vec<u32> = results.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, vec![0, 4, 1, 5, 2, 6]);
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1, "results must be ascending by distance");
    }
}

// ---- Test 7: f64 twin of test 2 -------------------------------------------

#[test]
fn add_points_0_to_3_matches_hand_computed_knn_f64() {
    let mut idx = RefDynIndexF64::build(&PTS_F64, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3);

    let (indices, dists) = idx.knn(&QUERY_F64, 2, 0.0);
    assert_eq!(indices, vec![0, 1]);
    assert!((dists[0] - 0.13).abs() < 1e-9, "dist[0] = {}", dists[0]);
    assert!((dists[1] - 0.73).abs() < 1e-9, "dist[1] = {}", dists[1]);
}

// ---- Test 8: build/free loop with adds/removes (leak/UB smoke) -----------

#[test]
fn build_free_loop_100_iterations_with_adds_removes_no_crash() {
    for iter in 0..100 {
        let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
        idx.set_current_n(8);
        idx.add_points(0, 7);
        // Deterministic add/remove churn, varied per iteration.
        let victim = (iter % 8) as usize;
        idx.remove_point(victim);
        idx.add_points(victim as u32, victim as u32); // re-add (reactivation)
        idx.remove_point((iter % 8) as usize);

        let (indices, _dists) = idx.knn(&QUERY_F32, 3, 0.0);
        assert!(indices.len() <= 3);
        // idx dropped here each iteration.
    }
}

// ---- extra: knn_into / radius_into zero-alloc variants match allocating --

#[test]
fn knn_into_matches_allocating_knn_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(4);
    idx.add_points(0, 3);

    let (want_idx, want_dist) = idx.knn(&QUERY_F32, 2, 0.0);
    let mut got_idx = vec![0u32; 2];
    let mut got_dist = vec![0.0f32; 2];
    let found = idx.knn_into(&QUERY_F32, 2, 0.0, &mut got_idx, &mut got_dist);
    assert_eq!(found, want_idx.len());
    assert_eq!(&got_idx[..found], want_idx.as_slice());
    assert_eq!(&got_dist[..found], want_dist.as_slice());
}

#[test]
fn radius_into_matches_allocating_radius_f32() {
    let mut idx = RefDynIndexF32::build(&PTS_F32, DIM, LEAF_MAX_SIZE, MAX_POINT_COUNT);
    idx.set_current_n(8);
    idx.add_points(0, 7);

    let want = idx.radius(&QUERY_F32, 5.0, true, 0.0);
    let mut got_idx: Vec<u32> = Vec::new();
    let mut got_dist: Vec<f32> = Vec::new();
    let count = idx.radius_into(&QUERY_F32, 5.0, true, 0.0, &mut got_idx, &mut got_dist);
    assert_eq!(count, want.len());
    for i in 0..want.len() {
        assert_eq!(got_idx[i], want[i].0);
        assert_eq!(got_dist[i], want[i].1);
    }
}
