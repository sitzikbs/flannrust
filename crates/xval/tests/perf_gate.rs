//! Performance gate: automated pass/fail perf assertions vs the C++ oracle.
//! User-mandated milestone success criterion. All tests `#[ignore]`d --
//! meant to be run explicitly, in release, on an otherwise-quiet machine.
//!
//! Run: `PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval
//! --release -- --ignored perf_gate --test-threads=1 --nocapture`
//!
//! Each gate: fixed-seed workload, BOTH sides warmed up (one untimed pass),
//! then timed with `std::time::Instant` -- MEDIAN of 7 timed runs per side.
//! Asserts `rust_median <= cpp_median * 1.25` (the 1.25 margin absorbs
//! machine noise). Every gate prints `PERF_GATE {name}: rust={..}ms
//! cpp={..}ms ratio={..}` unconditionally (pass or fail) so runs are
//! self-documenting; a failing assert repeats the same numbers in its panic
//! message. A gate FAILURE is a milestone blocker (route to the perf task),
//! NOT a flaky test to loosen -- the 1.25 margin may only be changed by the
//! controller.
//!
//! Guard: besides `#[ignore]`, every gate test FIRST checks whether
//! `PERF_GATE` is set and returns early (printing `PERF_GATE skipped: set
//! PERF_GATE=1`) if not -- so `cargo test --workspace --release --
//! --ignored` (which also runs the heavy 1M-scale build/xval tests, in
//! parallel) can never produce a false gate failure just by touching these
//! tests. The real invocation above sets `PERF_GATE=1` explicitly and forces
//! `--test-threads=1` (gate timings must not share the machine with a
//! sibling test's CPU load).

use nanoflann_ref::{Metric, RefIndex3F32, RefIndexF32, RefIndexF64};
use nanoflann_rs::{ConstDim, KdTreeBuilder, L2};
use xval::{build_rust_f32, build_rust_f64, cfg_seed, queries, timed_median_ms, to_array3, to_f32, uniform, BuildThreads, RoundRobin, XMetric};

const RUNS: usize = 7;
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
/// ratio, for both the unconditional print and the assert message below.
fn report(name: &str, rust_ms: f64, cpp_ms: f64) -> f64 {
    let ratio = rust_ms / cpp_ms;
    println!("PERF_GATE {name}: rust={rust_ms:.3}ms cpp={cpp_ms:.3}ms ratio={ratio:.3}");
    ratio
}

macro_rules! assert_gate {
    ($name:expr, $rust_ms:expr, $cpp_ms:expr) => {{
        let ratio = report($name, $rust_ms, $cpp_ms);
        assert!(
            ratio <= MARGIN,
            "PERF_GATE {} FAILED: rust={:.3}ms cpp={:.3}ms ratio={:.3} (must be <= {MARGIN})",
            $name,
            $rust_ms,
            $cpp_ms,
            ratio
        );
    }};
}

// `median_of`/`timed_median_ms` now live in `xval` itself (shared with
// `examples/report_data.rs`); their unit tests moved to
// `crates/xval/src/lib.rs`'s test mod. Only the guard logic below is
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

    let rust_ms = timed_median_ms(RUNS, || {
        let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
        std::hint::black_box(idx.size());
    });
    let cpp_ms = timed_median_ms(RUNS, || {
        let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);
        std::hint::black_box(idx.size());
    });

    assert_gate!("perf_gate_build_100k_dim3_f32_seq", rust_ms, cpp_ms);
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

    let rust_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q32, DIM);
        let mut out_idx = vec![0u32; K];
        let mut out_dist = vec![0.0f32; K];
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let found = rust_tree.knn_search(query, &mut out_idx, &mut out_dist);
            std::hint::black_box(found);
        }
    });
    let cpp_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q32, DIM);
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let (idx, dist) = cpp_tree.knn(query, K);
            std::hint::black_box((idx, dist));
        }
    });

    assert_gate!("perf_gate_knn_dim3_f32_k10", rust_ms, cpp_ms);
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

    let rust_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q, DIM);
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let (idx, dist) = rust_tree.knn(query, K, 0.0);
            std::hint::black_box((idx, dist));
        }
    });
    let cpp_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q, DIM);
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let (idx, dist) = cpp_tree.knn(query, K, 0.0);
            std::hint::black_box((idx, dist));
        }
    });

    assert_gate!("perf_gate_knn_dyn_dim8_f64_k10", rust_ms, cpp_ms);
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

    let data = to_f32(&uniform(cfg_seed("perf_gate_radius", &[N]), N, DIM));
    let q64 = queries(cfg_seed("perf_gate_radius_q", &[N]), &uniform(cfg_seed("perf_gate_radius", &[N]), N, DIM), DIM, POOL);
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

    let rust_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q, DIM);
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let out = rust_tree.radius(query, radius, true, 0.0);
            std::hint::black_box(out);
        }
    });
    let cpp_ms = timed_median_ms(RUNS, || {
        let mut rr = RoundRobin::new(&q, DIM);
        for _ in 0..N_QUERIES {
            let query = std::hint::black_box(rr.next());
            let out = cpp_tree.radius(query, radius, true, 0.0);
            std::hint::black_box(out);
        }
    });

    assert_gate!("perf_gate_radius_dim3_f32", rust_ms, cpp_ms);
}
