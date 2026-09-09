//! TDD tests for the `nanoflann-ref` C++ oracle, written against the safe
//! Rust API described in the Task 7 brief BEFORE that API exists (RED),
//! then made to pass once `build.rs` / `cpp/wrapper.cpp` / `src/lib.rs` are
//! implemented (GREEN).
//!
//! Fixed dim-3 f32 dataset used throughout (row-major, n=5, dim=3):
//!   0: [0,0,0]
//!   1: [1,0,0]
//!   2: [0,2,0]
//!   3: [3,3,3]
//!   4: [1,1,1]
//! query: [0.9, 0.1, 0.1]
//!
//! Hand-computed SQUARED L2 distances from the query:
//!   to 0: 0.9^2 + 0.1^2 + 0.1^2 = 0.83
//!   to 1: 0.1^2 + 0.1^2 + 0.1^2 = 0.03
//!   to 2: 0.9^2 + 1.9^2 + 0.1^2 = 4.43
//!   to 3: 2.1^2 + 2.9^2 + 2.9^2 = 21.23
//!   to 4: 0.1^2 + 0.9^2 + 0.9^2 = 1.63
//! So ascending order by distance is: 1 (0.03), 0 (0.83), 4 (1.63), 2 (4.43), 3 (21.23).

use nanoflann_ref::{Metric, RefIndex3F32, RefIndex3F64, RefIndexF32, RefIndexF64};

const DIM: usize = 3;
const PTS_F32: [f32; 15] = [
    0.0, 0.0, 0.0, // 0
    1.0, 0.0, 0.0, // 1
    0.0, 2.0, 0.0, // 2
    3.0, 3.0, 3.0, // 3
    1.0, 1.0, 1.0, // 4
];
const PTS_F64: [f64; 15] = [
    0.0, 0.0, 0.0, // 0
    1.0, 0.0, 0.0, // 1
    0.0, 2.0, 0.0, // 2
    3.0, 3.0, 3.0, // 3
    1.0, 1.0, 1.0, // 4
];
const QUERY_F32: [f32; 3] = [0.9, 0.1, 0.1];
const QUERY_F64: [f64; 3] = [0.9, 0.1, 0.1];

// ---- Test 1: knn, squared L2 distances, indices [1, 0] ----

#[test]
fn knn_k2_returns_expected_indices_and_squared_distances_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.0);
    assert_eq!(indices, vec![1, 0], "expected nearest-2 indices [1, 0]");
    assert!(
        (dists[0] - 0.03).abs() < 1e-6,
        "dist[0] = {} (want ~0.03)",
        dists[0]
    );
    assert!(
        (dists[1] - 0.83).abs() < 1e-6,
        "dist[1] = {} (want ~0.83)",
        dists[1]
    );
}

// ---- Test 1b: knn with nonzero eps (f32 FFI plumbing) ----
//
// leaf_max_size=10 >= n=5, so the whole dataset lives in a single leaf: no
// internal node ever gets a prune decision to make, so the eps-approximate
// search visits exactly the same points as an exact (eps=0) search would --
// the result is well-defined regardless of eps's value. This isolates the
// `eps: f32` FFI parameter itself (does the value even reach the C++ side
// and get accepted by `SearchParameters`?) from any question of whether
// approximate pruning is correct.

#[test]
fn knn_k2_with_nonzero_eps_returns_same_result_as_exact_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.5);
    assert_eq!(indices, vec![1, 0], "expected nearest-2 indices [1, 0]");
    assert!((dists[0] - 0.03).abs() < 1e-6, "dist[0] = {}", dists[0]);
    assert!((dists[1] - 0.83).abs() < 1e-6, "dist[1] = {}", dists[1]);
}

#[test]
fn knn_k2_with_nonzero_eps_returns_same_result_as_exact_f64() {
    let idx = RefIndexF64::build(&PTS_F64, DIM, Metric::L2, 10, 1);
    let (indices, dists) = idx.knn(&QUERY_F64, 2, 0.5);
    assert_eq!(indices, vec![1, 0], "expected nearest-2 indices [1, 0]");
    assert!((dists[0] - 0.03).abs() < 1e-9, "dist[0] = {}", dists[0]);
    assert!((dists[1] - 0.83).abs() < 1e-9, "dist[1] = {}", dists[1]);
}

// ---- Test 2: radius search, ascending order + strict-< exclusion ----

#[test]
fn radius_search_returns_ascending_points_within_radius_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let results = idx.radius(&QUERY_F32, 1.0, true, 0.0);
    let indices: Vec<u32> = results.iter().map(|(i, _)| *i).collect();
    assert_eq!(
        indices,
        vec![1, 0],
        "expected ascending [1, 0] within radius 1.0"
    );
    assert!(
        results[0].1 <= results[1].1,
        "results must be ascending by distance"
    );
}

#[test]
fn radius_search_excludes_point_exactly_at_radius_strict_less_than_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    // Compute the actual f32 distance to point 1 via knn (avoids relying on
    // an exact decimal literal matching the f32 arithmetic bit-for-bit).
    let (indices, dists) = idx.knn(&QUERY_F32, 1, 0.0);
    assert_eq!(indices, vec![1]);
    let exact_dist = dists[0];

    // radius == exact_dist must EXCLUDE point 1 (strict < contract).
    let results = idx.radius(&QUERY_F32, exact_dist, true, 0.0);
    let indices: Vec<u32> = results.iter().map(|(i, _)| *i).collect();
    assert!(
        !indices.contains(&1),
        "point 1 at exactly the radius must be excluded (strict <), got {:?}",
        indices
    );
}

// ---- Test 3: rknn, k with a radius cap ----

#[test]
fn rknn_returns_only_points_within_radius_cap_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    // radius=0.5: only point 1 (dist^2 = 0.03) qualifies; 0.83, 1.63, 4.43,
    // 21.23 are all >= 0.5.
    let (indices, dists) = idx.rknn(&QUERY_F32, 3, 0.5, 0.0);
    assert_eq!(indices, vec![1], "expected only point 1 within radius 0.5");
    assert_eq!(dists.len(), 1);
    assert!((dists[0] - 0.03).abs() < 1e-6);
}

// ---- Test 4: find_within_box, inclusive faces ----

#[test]
fn find_within_box_is_inclusive_on_faces_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let lo = [0.0f32, 0.0, 0.0];
    let hi = [1.0f32, 1.0, 1.0];
    let mut got = idx.find_within_box(&lo, &hi);
    got.sort_unstable();
    // point 0 = [0,0,0] (inside), point 1 = [1,0,0] (on the x=1 face),
    // point 4 = [1,1,1] (on the corner) -- all inclusive.
    // point 2 = [0,2,0] (y=2 > 1, excluded), point 3 = [3,3,3] (excluded).
    assert_eq!(got, vec![0, 1, 4]);
}

// ---- Test 5: vind is a permutation of 0..n ----

#[test]
fn vind_is_a_permutation_of_point_indices_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let mut v = idx.vind();
    v.sort_unstable();
    assert_eq!(v, vec![0, 1, 2, 3, 4]);
}

// ---- Test 6: f64 twin of test 1 ----

#[test]
fn knn_k2_returns_expected_indices_and_squared_distances_f64() {
    let idx = RefIndexF64::build(&PTS_F64, DIM, Metric::L2, 10, 1);
    let (indices, dists) = idx.knn(&QUERY_F64, 2, 0.0);
    assert_eq!(indices, vec![1, 0]);
    assert!((dists[0] - 0.03).abs() < 1e-9, "dist[0] = {}", dists[0]);
    assert!((dists[1] - 0.83).abs() < 1e-9, "dist[1] = {}", dists[1]);
}

// ---- Test 7: build/free loop smoke test (leak/UB) ----

#[test]
fn build_free_loop_1000_points_200_iterations_no_crash() {
    // Deterministic LCG fill so the dataset is fixed across runs.
    let n = 1000usize;
    let dim = 3usize;
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut pts = vec![0.0f32; n * dim];
    for v in pts.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 40) as u32; // 24 usable bits
        *v = (bits as f32) / (1u32 << 24) as f32 * 10.0;
    }
    let query = [pts[0], pts[1], pts[2]];

    for _ in 0..200 {
        let idx = RefIndexF32::build(&pts, dim, Metric::L2, 10, 1);
        let (indices, _dists) = idx.knn(&query, 5, 0.0);
        assert!(!indices.is_empty());
        // idx dropped here each iteration.
    }
}

// ---- Test 8: empty dataset ----

#[test]
fn empty_dataset_knn_returns_zero_results_f32() {
    let empty: [f32; 0] = [];
    let idx = RefIndexF32::build(&empty, 3, Metric::L2, 10, 1);
    assert_eq!(idx.size(), 0);
    let (indices, dists) = idx.knn(&QUERY_F32, 2, 0.0);
    assert_eq!(indices.len(), 0);
    assert_eq!(dists.len(), 0);
}

// ---- Test 9: dim-3 fast path matches runtime-dim L2 ----

#[test]
fn dim3_fast_path_matches_runtime_dim_l2_f32() {
    let runtime_idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let (runtime_indices, runtime_dists) = runtime_idx.knn(&QUERY_F32, 2, 0.0);

    let fast_idx = RefIndex3F32::build(&PTS_F32, 10, 1);
    let (fast_indices, fast_dists) = fast_idx.knn(&QUERY_F32, 2);

    assert_eq!(runtime_indices, fast_indices);
    for (a, b) in runtime_dists.iter().zip(fast_dists.iter()) {
        assert!((a - b).abs() < 1e-6, "runtime={a} fast={b}");
    }
}

#[test]
fn dim3_fast_path_matches_runtime_dim_l2_f64() {
    let runtime_idx = RefIndexF64::build(&PTS_F64, DIM, Metric::L2, 10, 1);
    let (runtime_indices, runtime_dists) = runtime_idx.knn(&QUERY_F64, 2, 0.0);

    let fast_idx = RefIndex3F64::build(&PTS_F64, 10, 1);
    let (fast_indices, fast_dists) = fast_idx.knn(&QUERY_F64, 2);

    assert_eq!(runtime_indices, fast_indices);
    for (a, b) in runtime_dists.iter().zip(fast_dists.iter()) {
        assert!((a - b).abs() < 1e-9, "runtime={a} fast={b}");
    }
}

// ---- extra: used_memory smoke (non-zero for a non-empty tree) ----

#[test]
fn used_memory_is_nonzero_for_nonempty_tree_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    assert!(idx.used_memory() > 0);
}

// ---- extra: every metric's switch-dispatch branch actually runs (not just
// compiles). The 9 brief tests above only exercise Metric::L2; these smoke
// tests make sure the other 4 branches of the metric `switch` in
// cpp/wrapper.cpp's `dispatch()` are runtime-correct too, not just that they
// compile (all 5 are instantiated into the object file regardless).

#[test]
fn every_metric_builds_and_answers_knn_without_crashing_f32() {
    for metric in [
        Metric::L1,
        Metric::L2,
        Metric::L2Simple,
        Metric::SO2,
        Metric::SO3,
    ] {
        let idx = RefIndexF32::build(&PTS_F32, DIM, metric, 10, 1);
        assert_eq!(idx.size(), 5);
        let (indices, dists) = idx.knn(&QUERY_F32, 3, 0.0);
        assert_eq!(indices.len(), 3, "metric {:?}", metric);
        assert_eq!(dists.len(), 3, "metric {:?}", metric);
        // Every returned index must be a valid point index.
        for &i in &indices {
            assert!(
                i < 5,
                "metric {:?} returned out-of-range index {}",
                metric,
                i
            );
        }
    }
}

#[test]
fn so2_last_dim_only_matches_hand_computed_wrap_f32() {
    // SO2 uses only the LAST coordinate, treated as an angle in [-pi, pi];
    // dim=2 per the brief's note ("only meaningful for the last dimension").
    let pts: [f32; 6] = [
        0.0, 3.0, // 0
        0.0, -3.0, // 1
        0.0, 0.0, // 2
    ];
    let idx = RefIndexF32::build(&pts, 2, Metric::SO2, 10, 1);
    let query = [0.0f32, 3.0];
    let (indices, dists) = idx.knn(&query, 1, 0.0);
    // Nearest by wrapped angular distance on the last dim only: point 0
    // (angle 3.0) is an exact match (dist 0), regardless of the first
    // (ignored) coordinate.
    assert_eq!(indices, vec![0]);
    assert!((dists[0] - 0.0).abs() < 1e-6, "dist={}", dists[0]);
}

// ---- extra: zero-allocation `_into` variants match their allocating
// counterparts bit-for-bit (Task 13 fix-round-2: allocation-symmetric C++
// query paths for benchmarks). ----

#[test]
fn knn_into_matches_allocating_knn_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let (want_idx, want_dist) = idx.knn(&QUERY_F32, 3, 0.0);

    let mut got_idx = vec![0u32; 3];
    let mut got_dist = vec![0.0f32; 3];
    let found = idx.knn_into(&QUERY_F32, 3, 0.0, &mut got_idx, &mut got_dist);

    assert_eq!(found, want_idx.len());
    assert_eq!(&got_idx[..found], want_idx.as_slice());
    assert_eq!(
        &got_dist[..found],
        want_dist.as_slice(),
        "distances must be bit-identical"
    );
}

#[test]
fn knn_into_panics_on_wrong_buffer_length() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let mut short_idx = vec![0u32; 2];
    let mut dist = vec![0.0f32; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        idx.knn_into(&QUERY_F32, 3, 0.0, &mut short_idx, &mut dist)
    }));
    assert!(
        result.is_err(),
        "knn_into must panic when out_idx.len() != k"
    );
}

#[test]
fn knn_into_matches_allocating_knn_f64() {
    let idx = RefIndexF64::build(&PTS_F64, DIM, Metric::L2, 10, 1);
    let (want_idx, want_dist) = idx.knn(&QUERY_F64, 4, 0.0);

    let mut got_idx = vec![0u32; 4];
    let mut got_dist = vec![0.0f64; 4];
    let found = idx.knn_into(&QUERY_F64, 4, 0.0, &mut got_idx, &mut got_dist);

    assert_eq!(found, want_idx.len());
    assert_eq!(&got_idx[..found], want_idx.as_slice());
    assert_eq!(
        &got_dist[..found],
        want_dist.as_slice(),
        "distances must be bit-identical"
    );
}

#[test]
fn radius_into_matches_allocating_radius_f32() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let want = idx.radius(&QUERY_F32, 5.0, true, 0.0);

    let mut got_idx: Vec<u32> = Vec::new();
    let mut got_dist: Vec<f32> = Vec::new();
    let count = idx.radius_into(&QUERY_F32, 5.0, true, 0.0, &mut got_idx, &mut got_dist);

    assert_eq!(count, want.len());
    assert_eq!(got_idx.len(), want.len());
    assert_eq!(got_dist.len(), want.len());
    for i in 0..want.len() {
        assert_eq!(got_idx[i], want[i].0);
        assert_eq!(
            got_dist[i], want[i].1,
            "distances must be bit-identical at rank {i}"
        );
    }
}

#[test]
fn radius_into_reuses_and_resizes_buffers_across_calls_with_different_result_sizes() {
    let idx = RefIndexF32::build(&PTS_F32, DIM, Metric::L2, 10, 1);
    let mut out_idx: Vec<u32> = Vec::new();
    let mut out_dist: Vec<f32> = Vec::new();

    // Small radius (fewer results), then a large radius (more results),
    // then back to small -- the SAME buffers must end up correctly sized
    // (not stale) after each call, growing and shrinking as needed.
    let c1 = idx.radius_into(&QUERY_F32, 0.5, true, 0.0, &mut out_idx, &mut out_dist);
    assert_eq!(c1, 1);
    assert_eq!(out_idx.len(), 1);

    let c2 = idx.radius_into(&QUERY_F32, 100.0, true, 0.0, &mut out_idx, &mut out_dist);
    assert_eq!(c2, 5, "radius 100.0 should capture all 5 points");
    assert_eq!(out_idx.len(), 5);

    let c3 = idx.radius_into(&QUERY_F32, 0.5, true, 0.0, &mut out_idx, &mut out_dist);
    assert_eq!(c3, 1);
    assert_eq!(
        out_idx.len(),
        1,
        "buffer must shrink back down, not leave stale entries"
    );
}

#[test]
fn fixed3_knn_into_matches_allocating_knn_f32() {
    let idx = RefIndex3F32::build(&PTS_F32, 10, 1);
    let (want_idx, want_dist) = idx.knn(&QUERY_F32, 2);

    let mut got_idx = vec![0u32; 2];
    let mut got_dist = vec![0.0f32; 2];
    let found = idx.knn_into(&QUERY_F32, 2, &mut got_idx, &mut got_dist);

    assert_eq!(found, want_idx.len());
    assert_eq!(&got_idx[..found], want_idx.as_slice());
    assert_eq!(
        &got_dist[..found],
        want_dist.as_slice(),
        "distances must be bit-identical"
    );
}

#[test]
fn fixed3_knn_into_matches_allocating_knn_f64() {
    let idx = RefIndex3F64::build(&PTS_F64, 10, 1);
    let (want_idx, want_dist) = idx.knn(&QUERY_F64, 3);

    let mut got_idx = vec![0u32; 3];
    let mut got_dist = vec![0.0f64; 3];
    let found = idx.knn_into(&QUERY_F64, 3, &mut got_idx, &mut got_dist);

    assert_eq!(found, want_idx.len());
    assert_eq!(&got_idx[..found], want_idx.as_slice());
    assert_eq!(
        &got_dist[..found],
        want_dist.as_slice(),
        "distances must be bit-identical"
    );
}
