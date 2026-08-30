//! Performance gate: automated pass/fail perf assertions vs the C++ oracle.
//! User-mandated milestone success criterion. All tests `#[ignore]`d --
//! meant to be run explicitly, in release, on an otherwise-quiet machine.
//!
//! Run: `PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval
//! --release -- --ignored perf_gate --test-threads=1 --nocapture`
//!
//! M2.6 Task 1: each gate's workload is timed via `xval::measure_pair`,
//! which interleaves rust/cpp PER REPETITION (2 warmup reps, discarded,
//! then an adaptive `n = clamp(10, 100, floor(budget_s*1000 / t_est_ms))`
//! timed reps of BOTH sides -- see `xval::measure_pair`'s doc comment) in
//! place of the old fixed median-of-7. `BUDGET_S` (30s, per side) is the
//! same value `examples/report_data.rs` uses for its speed rows -- both
//! consumers share the M2.6 controller's repetition-budget policy. The gate
//! decision is UNCHANGED: `rust_median <= cpp_median * 1.25` (the 1.25
//! margin absorbs machine noise), now computed over >=10 (usually 100)
//! reps instead of 7. Every gate prints `PERF_GATE {name}: rust={..}ms
//! cpp={..}ms ratio={..}` (medians, same prefix shape prior runs' doc
//! pastes grep for) UNCONDITIONALLY (pass or fail), with the full
//! mean/std/n stats for both sides appended after a `|` -- see `report`
//! below for the exact appended shape. A gate FAILURE is a milestone
//! blocker (route to the perf task), NOT a flaky test to loosen -- the 1.25
//! margin may only be changed by the controller.
//!
//! Guard: besides `#[ignore]`, every gate test FIRST checks whether
//! `PERF_GATE` is set and returns early (printing `PERF_GATE skipped: set
//! PERF_GATE=1`) if not -- so `cargo test --workspace --release --
//! --ignored` (which also runs the heavy 1M-scale build/xval tests, in
//! parallel) can never produce a false gate failure just by touching these
//! tests. The real invocation above sets `PERF_GATE=1` explicitly and forces
//! `--test-threads=1` (gate timings must not share the machine with a
//! sibling test's CPU load).

use flannrust::{ConstDim, DynDim, DynamicKdTreeBuilder, KdTreeBuilder, ResultItem, L2};
use nanoflann_ref::{Metric, RefDynIndexF32, RefIndex3F32, RefIndexF32, RefIndexF64};
use xval::{
    build_rust_f32, build_rust_f64, cfg_seed, measure_pair, queries, sample_distinct_indices, to_array3, to_f32,
    uniform, BuildThreads, GrowableFlat, RoundRobin, TimingStats, XMetric,
};

/// Per-side budget (seconds) for `measure_pair`'s adaptive `n` -- see
/// `xval::rep_count`'s policy. 30s per side, per the M2.6 controller
/// directive (shared with `examples/report_data.rs`'s `BUDGET_S`).
const BUDGET_S: f64 = 30.0;
const MARGIN: f64 = 1.25;

/// Pure decision function: given the *result* of reading the `PERF_GATE`
/// env var, should this gate test skip? Factored out from the actual
/// `std::env::var("PERF_GATE")` call so the guard logic is unit-testable
/// without mutating real process environment state (which would race under
/// `cargo test`'s default parallel test execution across other test files
/// in this crate).
fn should_skip_perf_gate(env_var: Result<String, std::env::VarError>) -> bool {
    env_var.is_err()
}

/// Prints the self-documenting `PERF_GATE {name}: ...` line and returns the
/// ratio of MEDIANS (the gate's decision quantity), for both the
/// unconditional print and the assert message below. Keeps the historical
/// `PERF_GATE {name}: rust={..}ms cpp={..}ms ratio={..}` prefix byte-for-
/// byte (older doc pastes grep for exactly that shape; `rust`/`cpp` are now
/// the MEDIANS instead of a bare median-of-7, same field names/positions),
/// then appends the full stats after a `|`: `rust mean=.. std=.. n=..`,
/// `cpp mean=.. std=.. n=..`, and an explicit `ratio_medians=..` (same
/// value as the leading `ratio=`, spelled out for the JSON/report
/// consumer's benefit -- see `examples/report_data.rs`'s emitted
/// `ratio_medians` field).
fn report(name: &str, rust: &TimingStats, cpp: &TimingStats) -> f64 {
    let ratio_medians = rust.median_ms / cpp.median_ms;
    println!(
        "PERF_GATE {name}: rust={:.3}ms cpp={:.3}ms ratio={:.3} | rust mean={:.3} std={:.3} n={} | cpp mean={:.3} std={:.3} n={} | ratio_medians={:.3}",
        rust.median_ms,
        cpp.median_ms,
        ratio_medians,
        rust.mean_ms,
        rust.std_ms,
        rust.n,
        cpp.mean_ms,
        cpp.std_ms,
        cpp.n,
        ratio_medians
    );
    ratio_medians
}

macro_rules! assert_gate {
    ($name:expr, $rust:expr, $cpp:expr) => {{
        let ratio = report($name, &$rust, &$cpp);
        assert!(
            ratio <= MARGIN,
            "PERF_GATE {} FAILED: rust_median={:.3}ms cpp_median={:.3}ms ratio={:.3} (must be <= {MARGIN})",
            $name,
            $rust.median_ms,
            $cpp.median_ms,
            ratio
        );
    }};
}

// `median_of`/`timed_median_ms` (the OLD fixed-median-of-7 helpers) and
// their unit tests live in `crates/xval/src/lib.rs` still (kept for
// `examples/m25_diag.rs`'s unrelated diagnostics, out of this task's
// scope) -- `measure_pair`/`TimingStats`/`rep_count` (this task's adaptive
// replacement) are unit-tested there too. Only the guard logic below is
// specific to this file.
#[cfg(test)]
mod guard_tests {
    use super::*;

    #[test]
    fn skips_when_var_unset() {
        assert!(should_skip_perf_gate(Err(std::env::VarError::NotPresent)));
    }

    #[test]
    fn runs_when_var_set_to_1() {
        assert!(!should_skip_perf_gate(Ok("1".to_string())));
    }

    #[test]
    fn runs_when_var_set_to_any_value() {
        // `.is_ok()` per the brief -- ANY Ok value counts, not just "1".
        assert!(!should_skip_perf_gate(Ok(String::new())));
        assert!(!should_skip_perf_gate(Ok("0".to_string())));
    }
}

// ============================================================================
// Gated workloads
// ============================================================================

#[test]
#[ignore]
fn perf_gate_build_100k_dim3_f32_seq() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    const N: usize = 100_000;
    const DIM: usize = 3;
    const LEAF: usize = 10;

    let data = to_f32(&uniform(cfg_seed("perf_gate_build", &[N, DIM]), N, DIM));

    let (rust, cpp) = measure_pair(
        || {
            let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
            std::hint::black_box(idx.size());
        },
        || {
            let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);
            std::hint::black_box(idx.size());
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_build_100k_dim3_f32_seq", rust, cpp);
}

#[test]
#[ignore]
fn perf_gate_knn_dim3_f32_k10() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    const N: usize = 100_000;
    const DIM: usize = 3;
    const LEAF: usize = 10;
    const K: usize = 10;
    const N_QUERIES: usize = 10_000;
    const POOL: usize = 1000;

    let data64 = uniform(cfg_seed("perf_gate_knn_fixed3", &[N]), N, DIM);
    let data32 = to_f32(&data64);
    let arr3 = to_array3(&data32);
    let q64 = queries(cfg_seed("perf_gate_knn_fixed3_q", &[N]), &data64, DIM, POOL);
    let q32 = to_f32(&q64);

    // Build ONCE, outside all timing.
    let rust_tree =
        KdTreeBuilder::new(ConstDim::<3>, arr3.as_slice()).with_metric(L2).leaf_max_size(LEAF).build_sequential();
    let cpp_tree = RefIndex3F32::build(&data32, LEAF, 1);

    let (rust, cpp) = measure_pair(
        || {
            let mut rr = RoundRobin::new(&q32, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f32; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.knn_search(query, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        || {
            let mut rr = RoundRobin::new(&q32, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f32; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.knn_into(query, K, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_knn_dim3_f32_k10", rust, cpp);
}

#[test]
#[ignore]
fn perf_gate_knn_dyn_dim8_f64_k10() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    const N: usize = 100_000;
    const DIM: usize = 8;
    const LEAF: usize = 10;
    const K: usize = 10;
    const N_QUERIES: usize = 10_000;
    const POOL: usize = 1000;

    let data = uniform(cfg_seed("perf_gate_knn_dyn8", &[N]), N, DIM);
    let q = queries(cfg_seed("perf_gate_knn_dyn8_q", &[N]), &data, DIM, POOL);

    // Build ONCE, outside all timing.
    let rust_tree = build_rust_f64(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
    let cpp_tree = RefIndexF64::build(&data, DIM, Metric::L2, LEAF, 1);

    let (rust, cpp) = measure_pair(
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f64; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f64; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_knn_dyn_dim8_f64_k10", rust, cpp);
}

#[test]
#[ignore]
fn perf_gate_radius_dim3_f32() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    const N: usize = 100_000;
    const DIM: usize = 3;
    const LEAF: usize = 10;
    const N_QUERIES: usize = 2_000;
    const POOL: usize = 1000;
    const SELECTIVITY_K: usize = 100; // "~100-point selectivity" per the brief

    let data64 = uniform(cfg_seed("perf_gate_radius", &[N]), N, DIM);
    let data = to_f32(&data64);
    let q64 = queries(cfg_seed("perf_gate_radius_q", &[N]), &data64, DIM, POOL);
    let q = to_f32(&q64);

    // Build ONCE, outside all timing.
    let rust_tree = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
    let cpp_tree = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);

    // Calibrate the radius ONCE, before timing, from the C++ index's own
    // k-nn (last returned distance == squared distance to the SELECTIVITY_K-th
    // nearest neighbor of the first pooled query).
    let probe = &q[0..DIM];
    let (_idx, dist) = cpp_tree.knn(probe, SELECTIVITY_K, 0.0);
    let radius = *dist.last().expect("calibration knn must return at least one neighbor");

    let (rust, cpp) = measure_pair(
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out: Vec<ResultItem<u32, f32>> = Vec::new();
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.radius_into(query, radius, true, 0.0, &mut out);
                std::hint::black_box(found);
            }
        },
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx: Vec<u32> = Vec::new();
            let mut out_dist: Vec<f32> = Vec::new();
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.radius_into(query, radius, true, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_radius_dim3_f32", rust, cpp);
}

// ============================================================================
// M2 Task 5 -- dynamic (add/remove) gates. Same guard/methodology as the
// static gates above; workloads mirror `benches/bench_dynamic.rs`'s `dyn_add`
// and `dyn_knn_after_churn` groups exactly, at gate-friendly sizes (20k
// instead of 100k for the add gate, so a `PERF_GATE=1` run stays fast) --
// see `bench_dynamic.rs`'s module doc for why nanoflann's dynamic
// `addPoints` is inherently O(heavy) per point and why that makes this a
// same-algorithm, not same-design, comparison.
// ============================================================================

#[test]
#[ignore]
fn perf_gate_dyn_add_20k_dim3_f32() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    const CAP: usize = 20_000;
    const DIM: usize = 3;
    const LEAF: usize = 10;
    const BATCH: usize = 1000;
    const BATCHES: usize = 20; // 20 * 1000 = 20k total

    let data = to_f32(&uniform(cfg_seed("perf_gate_dyn_add", &[CAP, DIM]), CAP, DIM));

    let (rust, cpp) = measure_pair(
        || {
            let growable = GrowableFlat::new(&data, DIM);
            let mut tree = DynamicKdTreeBuilder::new(DynDim(DIM), &growable)
                .leaf_max_size(LEAF)
                .maximum_point_count(CAP)
                .build();
            let mut start = 0usize;
            for _ in 0..BATCHES {
                let end = start + BATCH;
                growable.set_current_n(end);
                tree.add_points(start, end - 1);
                start = end;
            }
            std::hint::black_box(tree.tree_count());
        },
        || {
            let mut oracle = RefDynIndexF32::build(&data, DIM, LEAF, CAP);
            let mut start = 0u32;
            for _ in 0..BATCHES {
                let end = start + BATCH as u32;
                oracle.set_current_n(end as usize);
                oracle.add_points(start, end - 1);
                start = end;
            }
            std::hint::black_box(oracle.tree_count());
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_dyn_add_20k_dim3_f32", rust, cpp);
}

#[test]
#[ignore]
fn perf_gate_dyn_knn_after_churn_dim3_f32() {
    if should_skip_perf_gate(std::env::var("PERF_GATE")) {
        println!("PERF_GATE skipped: set PERF_GATE=1");
        return;
    }
    // Workload: build BASE points, remove REMOVE of them, re-add only HALF
    // of those (READD) -- leaving READD-worth of LIVE TOMBSTONES in place
    // -- then add a fresh CONTIGUOUS growth batch of GROWTH points starting
    // exactly at BASE (point_count at that moment). A window of GROWTH=5000
    // consecutive `add_points` increments is larger than 4096 = 2^12, so it
    // is GUARANTEED (pigeonhole: any 4096 consecutive integers contain
    // exactly one value congruent to 4095 mod 4096) to pass through a
    // point_count whose binary representation ends in >= 12 one-bits --
    // i.e. `first0bit` returns >= 12 for at least one point in the batch,
    // forcing a REAL cascading merge across slots 0..=11+ (see
    // `flannrust::dynamic::DynamicKdTree::add_points`'s doc comment for
    // the merge/rebuild schedule), which also MIGRATES the still-removed
    // tombstones' recorded slot. An earlier version of this workload
    // removed 5k and reactivated all 5k, leaving ZERO tombstones and ZERO
    // merges at query time -- a provable no-op churn that never exercised
    // tombstone-filtered search at all. This corrected workload provably
    // does not have that problem (see the `removed_len() > 0` assert
    // below, and `docs/EXPERIMENTS.md`'s reproduction section for a RED
    // capture proving the assert actually fires when the workload
    // regresses to the old no-op shape).
    const N: usize = 100_000; // total points after the growth batch
    const DIM: usize = 3;
    const LEAF: usize = 10;
    const BASE: usize = 95_000; // initial contiguous build size
    const REMOVE: usize = 5_000; // removed from the BASE range
    const READD: usize = 2_500; // reactivated -- half of REMOVE
    const GROWTH: usize = N - BASE; // fresh contiguous growth batch (5_000)
    const K: usize = 10;
    const N_QUERIES: usize = 10_000;
    const POOL: usize = 1000;

    let data64 = uniform(cfg_seed("perf_gate_dyn_churn", &[N]), N, DIM);
    let data = to_f32(&data64);
    let q64 = queries(cfg_seed("perf_gate_dyn_churn_q", &[N]), &data64, DIM, POOL);
    let q = to_f32(&q64);
    let remove_idx = sample_distinct_indices(cfg_seed("perf_gate_dyn_churn_idx", &[BASE, REMOVE]), BASE, REMOVE);
    let readd_idx = &remove_idx[..READD];

    // Build ONCE, outside timing; churn (REMOVE removes + READD re-adds,
    // leaving REMOVE-READD live tombstones) then a fresh GROWTH-sized
    // contiguous growth batch (triggering real merges + tombstone
    // migrations) ONCE, outside timing -- only the knn query loop below is
    // timed.
    let growable = GrowableFlat::new(&data, DIM);
    let mut rust_tree =
        DynamicKdTreeBuilder::new(DynDim(DIM), &growable).leaf_max_size(LEAF).maximum_point_count(N).build();
    growable.set_current_n(BASE);
    rust_tree.add_points(0, BASE - 1);
    for &idx in &remove_idx {
        rust_tree.remove_point(idx);
    }
    for &idx in readd_idx {
        rust_tree.add_points(idx, idx);
    }
    growable.set_current_n(N);
    rust_tree.add_points(BASE, BASE + GROWTH - 1);
    assert!(
        rust_tree.removed_len() > 0,
        "perf_gate_dyn_knn_after_churn_dim3_f32 setup: removed_len() must be > 0 at query \
         time (workload must leave live tombstones) -- got 0, which means this workload \
         regressed back to the provable-no-op churn shape this gate was fixed to avoid"
    );

    let mut cpp_tree = RefDynIndexF32::build(&data, DIM, LEAF, N);
    cpp_tree.set_current_n(BASE);
    cpp_tree.add_points(0, (BASE - 1) as u32);
    for &idx in &remove_idx {
        cpp_tree.remove_point(idx);
    }
    for &idx in readd_idx {
        cpp_tree.add_points(idx as u32, idx as u32);
    }
    cpp_tree.set_current_n(N);
    cpp_tree.add_points(BASE as u32, (N - 1) as u32);

    let (rust, cpp) = measure_pair(
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f32; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.knn_search(query, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f32; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        },
        BUDGET_S,
    );

    assert_gate!("perf_gate_dyn_knn_after_churn_dim3_f32", rust, cpp);
}
