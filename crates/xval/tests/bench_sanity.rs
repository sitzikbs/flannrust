//! Sanity test (NOT a criterion bench): exercises every bench data path
//! (`benches/bench_build.rs`, `benches/bench_knn.rs`, `benches/bench_radius.rs`,
//! `benches/bench_dynamic.rs`) at a tiny n = 2000 (dynamic path: tiny
//! capacity/batch sizes -- see `bench_data_paths_do_not_panic_dynamic`),
//! asserting only "does not panic" -- so `cargo test` (which never runs
//! criterion) still catches bench-code rot (a signature change, a removed
//! method, a broken assumption about slice lengths) the moment it happens,
//! instead of only at the next manual `cargo bench` run.

use nanoflann_ref::{Metric, RefDynIndexF32, RefIndex3F32, RefIndex3F64, RefIndexF32, RefIndexF64};
use flannrust::{ConstDim, DynDim, DynamicKdTreeBuilder, KdTreeBuilder, L2};
use xval::{
    build_rust_f32, build_rust_f64, cfg_seed, queries, sample_distinct_indices, to_array3, to_f32, uniform,
    BuildThreads, GrowableFlat, RoundRobin, XMetric,
};

const N: usize = 2000;
const DIM: usize = 3;
const DIM8: usize = 8;
const LEAF: usize = 10;
const K: usize = 10;
const N_QUERIES_POOL: usize = 50;

#[test]
fn bench_data_paths_do_not_panic() {
    // ---- build matrix path (bench_build.rs: rust/seq, rust/par, cpp/seq, cpp/par) ----
    let data64 = uniform(cfg_seed("bench_sanity", &[N, DIM]), N, DIM);
    let data32 = to_f32(&data64);

    let rust_seq = build_rust_f32(&data32, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
    let rust_par = build_rust_f32(&data32, DIM, XMetric::L2, LEAF, BuildThreads::Auto);
    let cpp_seq = RefIndexF32::build(&data32, DIM, Metric::L2, LEAF, 1);
    let cpp_par = RefIndexF32::build(&data32, DIM, Metric::L2, LEAF, 0);
    assert_eq!(rust_seq.size(), N);
    assert_eq!(rust_par.size(), N);
    assert_eq!(cpp_seq.size(), N);
    assert_eq!(cpp_par.size(), N);

    // ---- build_fixed3 path (ConstDim<3> rust vs nfr3_build_f cpp) ----
    let arr3 = to_array3(&data32);
    let rust3 =
        KdTreeBuilder::new(ConstDim::<3>, arr3.as_slice()).with_metric(L2).leaf_max_size(LEAF).build_sequential();
    let cpp3 = RefIndex3F32::build(&data32, LEAF, 1);
    assert_eq!(rust3.size(), N);

    // ---- runtime-dim knn path (bench_knn.rs `knn` group), f32 and f64, dim 3 and dim 8 ----
    let q64 = queries(cfg_seed("bench_sanity_q", &[N, DIM]), &data64, DIM, N_QUERIES_POOL);
    let q32 = to_f32(&q64);
    let mut rr32 = RoundRobin::new(&q32, DIM);
    let probe32 = rr32.next();
    let (r_idx, r_dist) = rust_seq.knn(probe32, K, 0.0);
    let (c_idx, c_dist) = cpp_seq.knn(probe32, K, 0.0);
    assert!(!r_idx.is_empty() && !c_idx.is_empty());
    assert_eq!(r_dist.len(), r_idx.len());
    assert_eq!(c_dist.len(), c_idx.len());

    let data64_dim8 = uniform(cfg_seed("bench_sanity_dim8", &[N, DIM8]), N, DIM8);
    let rust64_dim8 = build_rust_f64(&data64_dim8, DIM8, XMetric::L2, LEAF, BuildThreads::Sequential);
    let cpp64_dim8 = RefIndexF64::build(&data64_dim8, DIM8, Metric::L2, LEAF, 1);
    let q64_dim8 = queries(cfg_seed("bench_sanity_q_dim8", &[N, DIM8]), &data64_dim8, DIM8, N_QUERIES_POOL);
    let probe64 = &q64_dim8[0..DIM8];
    let (r_idx8, r_dist8) = rust64_dim8.knn(probe64, K, 0.0);
    let (c_idx8, c_dist8) = cpp64_dim8.knn(probe64, K, 0.0);
    assert!(!r_idx8.is_empty() && !c_idx8.is_empty());
    assert_eq!(r_dist8.len(), r_idx8.len());
    assert_eq!(c_dist8.len(), c_idx8.len());

    // ---- knn_fixed3 path (ConstDim<3> rust knn_search vs RefIndex3F32 knn) ----
    let mut out_idx = vec![0u32; K];
    let mut out_dist = vec![0.0f32; K];
    let found = rust3.knn_search(probe32, &mut out_idx, &mut out_dist);
    assert!(found > 0);
    let (c3_idx, c3_dist) = cpp3.knn(probe32, K);
    assert!(!c3_idx.is_empty());
    assert_eq!(c3_dist.len(), c3_idx.len());

    let arr3_64 = to_array3(&data64);
    let rust3_64 =
        KdTreeBuilder::new(ConstDim::<3>, arr3_64.as_slice()).with_metric(L2).leaf_max_size(LEAF).build_sequential();
    let cpp3_64 = RefIndex3F64::build(&data64, LEAF, 1);
    let probe64_dim3 = &q64[0..DIM];
    let mut out_idx64 = vec![0u32; K];
    let mut out_dist64 = vec![0.0f64; K];
    let found64 = rust3_64.knn_search(probe64_dim3, &mut out_idx64, &mut out_dist64);
    assert!(found64 > 0);
    let (c3_idx64, c3_dist64) = cpp3_64.knn(probe64_dim3, K);
    assert!(!c3_idx64.is_empty());
    assert_eq!(c3_dist64.len(), c3_idx64.len());

    // ---- radius calibration + radius_search path (bench_radius.rs) ----
    let (_calib_idx, calib_dist) = cpp_seq.knn(probe32, K, 0.0);
    let radius = *calib_dist.last().expect("calibration knn must return at least one neighbor");
    let rust_radius_out = rust_seq.radius(probe32, radius, true, 0.0);
    let cpp_radius_out = cpp_seq.radius(probe32, radius, true, 0.0);
    assert!(!rust_radius_out.is_empty());
    assert!(!cpp_radius_out.is_empty());

    // ---- leaf_sweep data path (small leaf, build + knn) ----
    for &leaf in &[1usize, 128] {
        let rust_leaf = build_rust_f32(&data32, DIM, XMetric::L2, leaf, BuildThreads::Sequential);
        let cpp_leaf = RefIndexF32::build(&data32, DIM, Metric::L2, leaf, 1);
        assert_eq!(rust_leaf.size(), N);
        let (rl_idx, _) = rust_leaf.knn(probe32, K, 0.0);
        let (cl_idx, _) = cpp_leaf.knn(probe32, K, 0.0);
        assert!(!rl_idx.is_empty() && !cl_idx.is_empty());
    }
}

// ============================================================================
// M2 Task 5: tiny dynamic path (benches/bench_dynamic.rs's three groups --
// dyn_add, dyn_churn, dyn_knn_after_churn -- at a tiny capacity/batch size
// so `cargo test` catches dynamic-bench-code rot too).
// ============================================================================

#[test]
fn bench_data_paths_do_not_panic_dynamic() {
    const CAP: usize = 200;
    const DIM: usize = 3;
    const LEAF: usize = 10;
    const BATCH: usize = 20;
    const BATCHES: usize = 10; // 10 * 20 = 200 = CAP
    const CHURN: usize = 20;
    const K: usize = 5;
    const N_QUERIES_POOL: usize = 20;

    let data = to_f32(&uniform(cfg_seed("bench_sanity_dyn_add_data", &[CAP, DIM]), CAP, DIM));

    // ---- dyn_add path (rust) ----
    let growable = GrowableFlat::new(&data, DIM);
    let mut rust_tree =
        DynamicKdTreeBuilder::new(DynDim(DIM), &growable).leaf_max_size(LEAF).maximum_point_count(CAP).build();
    let mut start = 0usize;
    for _ in 0..BATCHES {
        let end = start + BATCH;
        growable.set_current_n(end);
        rust_tree.add_points(start, end - 1);
        start = end;
    }
    assert_eq!(rust_tree.active_count(), CAP);

    // ---- dyn_add path (cpp) ----
    let mut cpp_tree = RefDynIndexF32::build(&data, DIM, LEAF, CAP);
    let mut start = 0u32;
    for _ in 0..BATCHES {
        let end = start + BATCH as u32;
        cpp_tree.set_current_n(end as usize);
        cpp_tree.add_points(start, end - 1);
        start = end;
    }
    assert!(cpp_tree.tree_count() > 0);

    // ---- dyn_churn path (both sides already fully built above) ----
    let churn_idx = sample_distinct_indices(cfg_seed("bench_sanity_dyn_churn_idx", &[CAP, CHURN]), CAP, CHURN);
    for &idx in &churn_idx {
        rust_tree.remove_point(idx);
    }
    assert_eq!(rust_tree.removed_len(), CHURN);
    for &idx in &churn_idx {
        rust_tree.add_points(idx, idx);
    }
    assert_eq!(rust_tree.removed_len(), 0);

    for &idx in &churn_idx {
        cpp_tree.remove_point(idx);
    }
    assert_eq!(cpp_tree.removed_count(), CHURN);
    for &idx in &churn_idx {
        cpp_tree.add_points(idx as u32, idx as u32);
    }
    assert_eq!(cpp_tree.removed_count(), 0);

    // ---- dyn_knn_after_churn path: one more churn pass (left applied this
    // time, mirroring the bench's "separate post-churn forest"), then
    // zero-alloc knn on both sides ----
    for &idx in &churn_idx {
        rust_tree.remove_point(idx);
    }
    for &idx in &churn_idx {
        rust_tree.add_points(idx, idx);
    }
    for &idx in &churn_idx {
        cpp_tree.remove_point(idx);
    }
    for &idx in &churn_idx {
        cpp_tree.add_points(idx as u32, idx as u32);
    }

    let q64 = queries(cfg_seed("bench_sanity_dyn_knn_q", &[CAP, DIM]), &to_f32_as_f64(&data), DIM, N_QUERIES_POOL);
    let q = to_f32(&q64);
    let mut rr = RoundRobin::new(&q, DIM);
    let mut out_idx = vec![0u32; K];
    let mut out_dist = vec![0.0f32; K];
    let probe = rr.next();
    let found = rust_tree.knn_search(probe, &mut out_idx, &mut out_dist);
    assert!(found > 0, "rust dynamic knn must find at least one point");

    let mut out_idx2 = vec![0u32; K];
    let mut out_dist2 = vec![0.0f32; K];
    let found2 = cpp_tree.knn_into(probe, K, 0.0, &mut out_idx2, &mut out_dist2);
    assert!(found2 > 0, "cpp dynamic knn must find at least one point");
}

/// Tiny local helper: `queries()` wants an `f64` dataset (it generates
/// exact-dataset-point query copies from it); this test's dynamic data path
/// only has the `f32`-cast buffer on hand, so cast back up rather than
/// generating a second, independently-seeded `f64` buffer that wouldn't
/// share the same points.
fn to_f32_as_f64(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&x| x as f64).collect()
}
