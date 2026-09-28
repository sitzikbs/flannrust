//! Report-data collector: emits ONE JSON document on stdout with everything
//! the project's Rust-vs-C++ comparison report needs (speed + accuracy vs a
//! brute-force ground truth). No new dependencies -- the JSON is
//! hand-written via `println!`/`format!` (every field here is a plain
//! number, string, or bool; nothing needs escaping).
//!
//! Run: `cargo run -p xval --release --example report_data > report_data.json`
//! (add `RUSTFLAGS="-C target-cpu=native"` for the fair-fight speed numbers,
//! same as the benches -- see `xval::lib` docs).
//!
//! Progress goes to STDERR; stdout is JSON only.
//!
//! - Speed: the perf-gate's four workloads (identical adaptive-repetition
//!   methodology, via `xval::measure_pair` -- see `tests/perf_gate.rs` and
//!   this task's `BUDGET_S`) plus `build_1M_dim3_f32` seq/par. Every query
//!   loop (knn/radius) is ALLOCATION-SYMMETRIC: both sides use reused,
//!   caller-owned out-buffers (`knn_into`/`radius_into`, not the
//!   allocating `knn()`/`radius()` convenience wrappers) -- see this file's
//!   emitted `meta.speed_methodology`. Each speed row emits full
//!   `xval::TimingStats` (mean/std/median/min/max/n) for BOTH sides plus
//!   `ratio_means`/`ratio_means_std`/`ratio_medians` -- see `speed_json`'s
//!   emission in `main` for the exact field names. The pre-M2.6 keys
//!   (`rust_ms`/`cpp_ms`/`ratio`) are KEPT, populated from the MEDIANS, so
//!   any not-yet-updated downstream consumer keeps working unchanged.
//! - Accuracy: three seeded datasets (uniform n=50k dim 3, uniform n=50k
//!   dim 8, with_duplicates n=20k dim 3), 2000 queries each, k=10. Ground
//!   truth is brute-force LINEAR SCAN (`xval::brute_force_knn_l2_f32`,
//!   selection independent of any tree), but its DISTANCE ARITHMETIC is the
//!   library's own `L2` metric kernel -- see `xval::lib`'s "Brute-force
//!   ground truth" module doc and this file's emitted `meta.gt_methodology`
//!   -- computed ONCE per dataset (independent of eps) and reused across
//!   eps in {0.0, 0.1, 1.0}. `exact_tie_aware_vs_bruteforce` (per impl) uses
//!   `xval::score_exact_tie_aware_f32` -- see that function's doc comment,
//!   and the `"scoring"` line in this file's emitted `meta` object, for the
//!   exact definition (the plain order-independent index-SET match is too
//!   strict on duplicate-heavy datasets, where multiple points can be
//!   genuinely tied at the k-th distance boundary).
//!   `mean_dist_rel_error_*`/`max_dist_rel_error_*` are unaffected -- still
//!   from `xval::score_query_f32`'s `rel_dist_errors`.

use flannrust::{
    ConstDim, DynDim, DynamicKdTreeBuilder, KdTreeBuilder, ResultItem, SearchParams, L2,
};
use nanoflann_ref::{Metric, RefDynIndexF32, RefIndex3F32, RefIndexF32};
use std::io::Write;
use xval::{
    apply_dyn_op_f32, brute_force_knn_l2_f32, brute_force_knn_l2_live_f32, build_rust_f32,
    cfg_seed, cpu_model, cxx_compiler_version, dyn_ops, dyn_ops_stats, git_sha, is_wsl,
    kernel_version, measure_pair, queries, sample_distinct_indices, score_exact_tie_aware_f32,
    score_exact_tie_aware_live_f32, score_query_f32, to_array3, to_f32, uniform, with_duplicates,
    BuildThreads, GrowableFlat, RoundRobin, TimingStats, XMetric,
};

/// One-line description of the accuracy scoring methodology, embedded
/// verbatim in the emitted JSON's `meta.scoring` field.
const SCORING_NOTE: &str = "exact_tie_aware_vs_bruteforce: true iff the k returned distances bit-match the ground truth's k smallest distances positionally (both sorted ascending) AND every returned index's recomputed true distance equals its reported distance -- accepts any valid k-th-boundary tie resolution while still catching a wrong point, wrong distance, or missed closer neighbor.";

/// One-line description of how ground-truth distances are computed,
/// embedded verbatim in the emitted JSON's `meta.gt_methodology` field
/// (an independently-written summation is not reliably bit-exact against
/// either tree's internal arithmetic, so GT must route through the same
/// kernel).
const GT_METHODOLOGY_NOTE: &str = "ground-truth distances computed via the library's own L2 metric kernel (flannrust::L2::eval, the same code path KdTree::knn_search uses internally for leaf points), NOT an independently-written summation; point SELECTION is a dumb linear scan over every index, independent of any tree's traversal order.";

/// One-line description of the speed-timing allocation methodology,
/// embedded verbatim in the emitted JSON's `meta.speed_methodology` field
/// (stated as the METHODOLOGY, not a caveat -- both sides are
/// allocation-symmetric, not just Rust).
const SPEED_METHODOLOGY_NOTE: &str = "both sides zero-allocation per query: out-buffers (Vec<u32>/Vec<T>) are allocated once per timed workload and reused across every query, via knn_search/radius_search (rust) and knn_into/radius_into (nanoflann_ref's caller-buffer methods, backed directly by the raw FFI's out-pointer writes) -- not the allocating knn()/radius() convenience wrappers on either side. Timing (M2.6): xval::measure_pair -- 2 discarded warmup reps, then an adaptive n = clamp(10, 100, floor(budget_s*1000/t_est_ms)) reps of BOTH sides, interleaved per repetition; budget_s=30 per side. mean/std(sample, n-1)/median/min/max/n published for both sides; gate PASS/FAIL and ratio_medians use the ratio of MEDIANS.";

/// Per-side budget (seconds) for `measure_pair`'s adaptive `n` -- shared
/// with `tests/perf_gate.rs`'s `BUDGET_S` (same M2.6 controller policy).
const BUDGET_S: f64 = 30.0;

fn progress(msg: &str) {
    eprintln!("[report_data] {msg}");
}

// ============================================================================
// meta
// ============================================================================

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn today_utc() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn cpu_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Best-effort signal for whether THIS run was invoked with
/// `RUSTFLAGS="-C target-cpu=native"`: `RUSTFLAGS`, when set on the
/// `cargo run`/`cargo bench` command line, is inherited by the running
/// process's own environment too (not just consumed by cargo/rustc at
/// compile time), so checking it here at runtime is a reasonable proxy --
/// not a guarantee (a build cached from an earlier native-flagged compile
/// could be reused under a differently-flagged rerun), but the simplest
/// honest signal available without a new dependency.
fn target_cpu_native() -> bool {
    std::env::var("RUSTFLAGS")
        .map(|v| v.contains("target-cpu=native"))
        .unwrap_or(false)
}

// ============================================================================
// Speed: reuse the perf-gate's four workloads + build_1M_dim3 seq/par
// ============================================================================

struct SpeedRow {
    workload: &'static str,
    rust: TimingStats,
    cpp: TimingStats,
}

fn speed_rows() -> Vec<SpeedRow> {
    let mut rows = Vec::new();

    // ---- perf_gate_build_100k_dim3_f32_seq ----
    progress("speed: build_100k_dim3_f32_seq");
    {
        const N: usize = 100_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        let data = to_f32(&uniform(cfg_seed("report_build_100k", &[N, DIM]), N, DIM));
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
        rows.push(SpeedRow {
            workload: "build_100k_dim3_f32_seq",
            rust,
            cpp,
        });
    }

    // ---- perf_gate_knn_dim3_f32_k10 (ConstDim<3> vs cpp fixed-DIM-3 fast path) ----
    progress("speed: knn_dim3_f32_k10");
    {
        const N: usize = 100_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        const K: usize = 10;
        const N_QUERIES: usize = 10_000;
        const POOL: usize = 1000;

        let data64 = uniform(cfg_seed("report_knn_fixed3", &[N]), N, DIM);
        let data32 = to_f32(&data64);
        let arr3 = to_array3(&data32);
        let q64 = queries(cfg_seed("report_knn_fixed3_q", &[N]), &data64, DIM, POOL);
        let q32 = to_f32(&q64);

        let rust_tree = KdTreeBuilder::new(ConstDim::<3>, arr3.as_slice())
            .with_metric(L2)
            .leaf_max_size(LEAF)
            .build_sequential();
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
        rows.push(SpeedRow {
            workload: "knn_dim3_f32_k10",
            rust,
            cpp,
        });
    }

    // ---- knn_dim3_f32_k50 (higher k -- more distance comparisons per query,
    // probes whether SIMD advantage grows with k) ----
    progress("speed: knn_dim3_f32_k50");
    {
        const N: usize = 100_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        const K: usize = 50;
        const N_QUERIES: usize = 10_000;
        const POOL: usize = 1000;

        let data64 = uniform(cfg_seed("report_knn_fixed3_k50", &[N]), N, DIM);
        let data32 = to_f32(&data64);
        let arr3 = to_array3(&data32);
        let q64 = queries(cfg_seed("report_knn_fixed3_k50_q", &[N]), &data64, DIM, POOL);
        let q32 = to_f32(&q64);

        let rust_tree = KdTreeBuilder::new(ConstDim::<3>, arr3.as_slice())
            .with_metric(L2)
            .leaf_max_size(LEAF)
            .build_sequential();
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
        rows.push(SpeedRow {
            workload: "knn_dim3_f32_k50",
            rust,
            cpp,
        });
    }

    // ---- perf_gate_knn_dyn_dim8_f64_k10 ----
    progress("speed: knn_dyn_dim8_f64_k10");
    {
        use nanoflann_ref::RefIndexF64;
        use xval::build_rust_f64;

        const N: usize = 100_000;
        const DIM: usize = 8;
        const LEAF: usize = 10;
        const K: usize = 10;
        const N_QUERIES: usize = 10_000;
        const POOL: usize = 1000;

        let data = uniform(cfg_seed("report_knn_dyn8", &[N]), N, DIM);
        let q = queries(cfg_seed("report_knn_dyn8_q", &[N]), &data, DIM, POOL);

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
        rows.push(SpeedRow {
            workload: "knn_dyn_dim8_f64_k10",
            rust,
            cpp,
        });
    }

    // ---- perf_gate_radius_dim3_f32 ----
    progress("speed: radius_dim3_f32");
    {
        const N: usize = 100_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        const N_QUERIES: usize = 2_000;
        const POOL: usize = 1000;
        const SELECTIVITY_K: usize = 100;

        let data64 = uniform(cfg_seed("report_radius", &[N]), N, DIM);
        let data = to_f32(&data64);
        let q64 = queries(cfg_seed("report_radius_q", &[N]), &data64, DIM, POOL);
        let q = to_f32(&q64);

        let rust_tree = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
        let cpp_tree = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);

        let probe = &q[0..DIM];
        let (_idx, dist) = cpp_tree.knn(probe, SELECTIVITY_K, 0.0);
        let radius = *dist
            .last()
            .expect("calibration knn must return at least one neighbor");

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
                    let found =
                        cpp_tree.radius_into(query, radius, true, 0.0, &mut out_idx, &mut out_dist);
                    std::hint::black_box(found);
                }
            },
            BUDGET_S,
        );
        rows.push(SpeedRow {
            workload: "radius_dim3_f32",
            rust,
            cpp,
        });
    }

    // ---- build_1M_dim3_f32 seq/par ----
    progress("speed: build_1M_dim3_f32 seq/par");
    {
        const N: usize = 1_000_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        let data = to_f32(&uniform(cfg_seed("report_build_1m", &[N, DIM]), N, DIM));

        let (rust_seq, cpp_seq) = measure_pair(
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
        rows.push(SpeedRow {
            workload: "build_1M_dim3_f32_seq",
            rust: rust_seq,
            cpp: cpp_seq,
        });

        let (rust_par, cpp_par) = measure_pair(
            || {
                let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Auto);
                std::hint::black_box(idx.size());
            },
            || {
                let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 0);
                std::hint::black_box(idx.size());
            },
            BUDGET_S,
        );
        rows.push(SpeedRow {
            workload: "build_1M_dim3_f32_par",
            rust: rust_par,
            cpp: cpp_par,
        });
    }

    // ---- M2 Task 5: perf_gate_dyn_add_20k_dim3_f32 (same workload as the
    // gate -- see tests/perf_gate.rs) ----
    progress("speed: dyn_add_20k_dim3_f32");
    {
        const CAP: usize = 20_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        const BATCH: usize = 1000;
        const BATCHES: usize = 20; // 20 * 1000 = 20k total

        let data = to_f32(&uniform(cfg_seed("report_dyn_add", &[CAP, DIM]), CAP, DIM));

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
        rows.push(SpeedRow {
            workload: "dyn_add_20k_dim3_f32",
            rust,
            cpp,
        });
    }

    // ---- M2 Task 5: perf_gate_dyn_knn_after_churn_dim3_f32 (same workload
    // as the gate -- see tests/perf_gate.rs) ----
    progress("speed: dyn_knn_after_churn_dim3_f32");
    {
        const N: usize = 100_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        const CHURN: usize = 5_000;
        const K: usize = 10;
        const N_QUERIES: usize = 10_000;
        const POOL: usize = 1000;

        let data64 = uniform(cfg_seed("report_dyn_churn", &[N]), N, DIM);
        let data = to_f32(&data64);
        let q64 = queries(cfg_seed("report_dyn_churn_q", &[N]), &data64, DIM, POOL);
        let q = to_f32(&q64);
        let churn_idx =
            sample_distinct_indices(cfg_seed("report_dyn_churn_idx", &[N, CHURN]), N, CHURN);

        // Build ONCE, outside timing; churn (5k removes + 5k re-adds) ONCE,
        // outside timing -- only the knn query loop below is timed.
        let growable = GrowableFlat::new(&data, DIM);
        let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(DIM), &growable)
            .leaf_max_size(LEAF)
            .maximum_point_count(N)
            .build();
        growable.set_current_n(N);
        rust_tree.add_points(0, N - 1);
        for &idx in &churn_idx {
            rust_tree.remove_point(idx);
        }
        for &idx in &churn_idx {
            rust_tree.add_points(idx, idx);
        }

        let mut cpp_tree = RefDynIndexF32::build(&data, DIM, LEAF, N);
        cpp_tree.set_current_n(N);
        cpp_tree.add_points(0, (N - 1) as u32);
        for &idx in &churn_idx {
            cpp_tree.remove_point(idx);
        }
        for &idx in &churn_idx {
            cpp_tree.add_points(idx as u32, idx as u32);
        }

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
        rows.push(SpeedRow {
            workload: "dyn_knn_after_churn_dim3_f32",
            rust,
            cpp,
        });
    }

    rows
}

// ============================================================================
// Accuracy: uniform dim3/dim8 (n=50k), with_duplicates dim3 (n=20k) -- 2000
// queries each, k=10, eps in {0.0, 0.1, 1.0}, scored against brute-force GT.
// ============================================================================

struct AccuracyRow {
    workload: String,
    n_queries: usize,
    rust_exact_tie_aware_vs_bruteforce: f64,
    cpp_exact_tie_aware_vs_bruteforce: f64,
    rust_eq_cpp_bitexact: bool,
    mean_dist_rel_error_rust: f64,
    max_dist_rel_error_rust: f64,
    mean_dist_rel_error_cpp: f64,
    max_dist_rel_error_cpp: f64,
    /// Task 5b: direct churn evidence for the dynamic accuracy rows only
    /// (`None` for the three static-dataset rows) -- see
    /// `dynamic_accuracy_rows`'s module doc.
    dyn_evidence: Option<DynEvidence>,
}

/// Task 5b: direct evidence, emitted alongside `dyn_churn_*` accuracy
/// rows, that the forest being scored is genuinely churned -- not just
/// freshly built. `live_count`/`removed_count` come straight from the
/// Rust side's own post-churn `tree_index()` bookkeeping (already
/// cross-validated bit-exact against the oracle by M2 Task 4); the four
/// op-kind totals come from `xval::dyn_ops_stats` over the exact op
/// sequence that was replayed.
#[derive(Clone, Copy)]
struct DynEvidence {
    live_count: usize,
    removed_count: usize,
    grow_and_add_count: usize,
    remove_count: usize,
    readd_count: usize,
    tombstone_migrations: usize,
}

const K: usize = 10;
const N_ACC_QUERIES: usize = 2000;
const EPS_VALUES: [(&str, f32); 3] = [("eps0", 0.0), ("eps0.1", 0.1), ("eps1", 1.0)];

fn accuracy_rows_for_dataset(name: &str, dim: usize, data: &[f32]) -> Vec<AccuracyRow> {
    progress(&format!(
        "accuracy: {name} (n={}, dim={dim})",
        data.len() / dim
    ));

    let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let q64 = queries(
        cfg_seed("report_accuracy_q", &[name.len(), dim]),
        &data64,
        dim,
        N_ACC_QUERIES,
    );
    let q = to_f32(&q64);

    let rust_tree = build_rust_f32(data, dim, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_tree = RefIndexF32::build(data, dim, Metric::L2, 10, 1);

    // Ground truth: computed ONCE per dataset (independent of eps), reused
    // across every eps arm below.
    progress(&format!(
        "accuracy: {name} brute-force GT for {N_ACC_QUERIES} queries (O(n_queries*n), slow part)"
    ));
    let mut gts: Vec<Vec<(u32, f32)>> = Vec::with_capacity(N_ACC_QUERIES);
    for i in 0..N_ACC_QUERIES {
        let query = &q[i * dim..(i + 1) * dim];
        gts.push(brute_force_knn_l2_f32(data, dim, query, K));
    }

    let mut out = Vec::with_capacity(EPS_VALUES.len());
    for (eps_label, eps) in EPS_VALUES {
        let mut rust_exact_count = 0usize;
        let mut cpp_exact_count = 0usize;
        let mut rust_errs: Vec<f64> = Vec::new();
        let mut cpp_errs: Vec<f64> = Vec::new();
        let mut bitexact = true;

        for i in 0..N_ACC_QUERIES {
            let query = &q[i * dim..(i + 1) * dim];
            let gt = &gts[i];

            let (r_idx, r_dist) = rust_tree.knn(query, K, eps);
            let (c_idx, c_dist) = cpp_tree.knn(query, K, eps);

            let r_pairs: Vec<(u32, f32)> =
                r_idx.iter().copied().zip(r_dist.iter().copied()).collect();
            let c_pairs: Vec<(u32, f32)> =
                c_idx.iter().copied().zip(c_dist.iter().copied()).collect();

            // `rel_dist_errors` (mean/max error stats) still comes from the
            // plain per-rank scorer -- unaffected by the tie-aware fix.
            let r_score = score_query_f32(gt, &r_pairs);
            let c_score = score_query_f32(gt, &c_pairs);
            rust_errs.extend(r_score.rel_dist_errors);
            cpp_errs.extend(c_score.rel_dist_errors);

            // Exactness (the JSON's `exact_tie_aware_vs_bruteforce` field)
            // uses the tie-aware scorer -- see the module doc / controller
            // ruling referenced there.
            if score_exact_tie_aware_f32(data, dim, query, gt, &r_pairs) {
                rust_exact_count += 1;
            }
            if score_exact_tie_aware_f32(data, dim, query, gt, &c_pairs) {
                cpp_exact_count += 1;
            }

            if bitexact {
                if r_idx.len() != c_idx.len() {
                    bitexact = false;
                } else {
                    for k in 0..r_idx.len() {
                        if r_idx[k] != c_idx[k] || r_dist[k].to_bits() != c_dist[k].to_bits() {
                            bitexact = false;
                            break;
                        }
                    }
                }
            }
        }

        let mean = |v: &[f64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let max = |v: &[f64]| v.iter().cloned().fold(0.0f64, f64::max);

        out.push(AccuracyRow {
            workload: format!("{name}_dim{dim}_f32_k{K}_{eps_label}"),
            n_queries: N_ACC_QUERIES,
            rust_exact_tie_aware_vs_bruteforce: rust_exact_count as f64 / N_ACC_QUERIES as f64,
            cpp_exact_tie_aware_vs_bruteforce: cpp_exact_count as f64 / N_ACC_QUERIES as f64,
            rust_eq_cpp_bitexact: bitexact,
            mean_dist_rel_error_rust: mean(&rust_errs),
            max_dist_rel_error_rust: max(&rust_errs),
            mean_dist_rel_error_cpp: mean(&cpp_errs),
            max_dist_rel_error_cpp: max(&cpp_errs),
            dyn_evidence: None,
        });
    }
    out
}

fn accuracy_rows() -> Vec<AccuracyRow> {
    let mut rows = Vec::new();

    let uniform_dim3 = to_f32(&uniform(
        cfg_seed("report_acc_uniform_dim3", &[]),
        50_000,
        3,
    ));
    rows.extend(accuracy_rows_for_dataset("uniform", 3, &uniform_dim3));

    let uniform_dim8 = to_f32(&uniform(
        cfg_seed("report_acc_uniform_dim8", &[]),
        50_000,
        8,
    ));
    rows.extend(accuracy_rows_for_dataset("uniform", 8, &uniform_dim8));

    let dup_dim3 = to_f32(&with_duplicates(
        cfg_seed("report_acc_dup_dim3", &[]),
        20_000,
        3,
        0.3,
    ));
    rows.extend(accuracy_rows_for_dataset("with_duplicates", 3, &dup_dim3));

    rows.extend(dynamic_accuracy_rows());

    rows
}

// ============================================================================
// M2 Task 5 -- dynamic accuracy: after a seeded 120-op churn (add/remove/
// re-add) sequence over a capacity-20k, dim-3, f32 buffer (reusing M2 Task
// 4's cross-validation plumbing directly -- `xval::dyn_ops`/
// `xval::apply_dyn_op_f32`/`xval::GrowableFlat`, NOT reimplemented here),
// score BOTH forests' knn against brute-force ground truth restricted to the
// LIVE set (`xval::brute_force_knn_l2_live_f32` / `xval::
// score_exact_tie_aware_live_f32` -- see those functions' doc comments: a
// removed point's coordinates stay physically present in the backing buffer
// under lazy deletion, so an unfiltered ground truth or scorer would treat a
// tombstoned index as a valid neighbor). eps in {0.0, 0.1} only (no eps=1.0
// arm here, unlike the static accuracy datasets -- two eps values is enough
// to exercise the same eps-pruning codepath the static rows already cover
// in depth; this section's job is the churn/live-set methodology, not a
// second full eps sweep).
// ============================================================================

const DYN_ACC_CAPACITY: usize = 20_000;
const DYN_ACC_DIM: usize = 3;
const DYN_ACC_N_OPS: usize = 120;
const DYN_ACC_N_QUERIES: usize = 2000;
const DYN_ACC_EPS_VALUES: [(&str, f32); 2] = [("eps0", 0.0), ("eps0.1", 0.1)];

fn dynamic_accuracy_rows() -> Vec<AccuracyRow> {
    progress("accuracy: dyn_churn_dim3_f32 (120-op churn sequence, capacity 20k)");

    let data64 = uniform(
        cfg_seed("report_dyn_acc_data", &[]),
        DYN_ACC_CAPACITY,
        DYN_ACC_DIM,
    );
    let data = to_f32(&data64);

    let growable = GrowableFlat::new(&data, DYN_ACC_DIM);
    let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(DYN_ACC_DIM), &growable)
        .maximum_point_count(DYN_ACC_CAPACITY)
        .build();
    let mut oracle = RefDynIndexF32::build(&data, DYN_ACC_DIM, 10, DYN_ACC_CAPACITY);

    let ops = dyn_ops(
        cfg_seed("report_dyn_acc_ops", &[]),
        DYN_ACC_CAPACITY,
        DYN_ACC_N_OPS,
    );
    for op in &ops {
        apply_dyn_op_f32(op, &growable, &mut rust_tree, &mut oracle);
    }

    // Live set, read straight from the rust side's own post-churn
    // bookkeeping (`tree_index[i] != -1` iff `i` is live) -- already
    // cross-validated bit-exact against the oracle's own `tree_index()` by
    // M2 Task 4 (`tests/xval_dynamic.rs`), so there is no need to
    // independently re-derive it from the op sequence here.
    let tree_index = rust_tree.tree_index();
    let live: Vec<bool> = (0..DYN_ACC_CAPACITY)
        .map(|i| i < tree_index.len() && tree_index[i] != -1)
        .collect();
    let live_count = live.iter().filter(|&&l| l).count();
    progress(&format!(
        "accuracy: dyn_churn_dim3_f32 churn done -- {live_count}/{DYN_ACC_CAPACITY} live after {DYN_ACC_N_OPS} ops"
    ));

    // Task 5b: direct evidence that the scored forest is genuinely
    // churned, not freshly built -- `removed_count` from the same
    // `tree_index()` bookkeeping as `live_count` above (a point is
    // "removed" iff it was ever added but is no longer live), plus the
    // op-kind totals (including tombstone migrations) from
    // `dyn_ops_stats` over the exact `ops` sequence just replayed.
    let point_count = tree_index.len();
    let removed_count = point_count - live_count;
    let op_stats = dyn_ops_stats(&ops);
    let evidence = DynEvidence {
        live_count,
        removed_count,
        grow_and_add_count: op_stats.grow_and_add_count,
        remove_count: op_stats.remove_count,
        readd_count: op_stats.readd_count,
        tombstone_migrations: op_stats.tombstone_migrations,
    };

    let q64 = queries(
        cfg_seed("report_dyn_acc_q", &[]),
        &data64,
        DYN_ACC_DIM,
        DYN_ACC_N_QUERIES,
    );
    let q = to_f32(&q64);

    // Ground truth: computed ONCE (independent of eps), over the LIVE set
    // only, reused across both eps arms below.
    progress(&format!(
        "accuracy: dyn_churn_dim3_f32 brute-force live-set GT for {DYN_ACC_N_QUERIES} queries"
    ));
    let mut gts: Vec<Vec<(u32, f32)>> = Vec::with_capacity(DYN_ACC_N_QUERIES);
    for i in 0..DYN_ACC_N_QUERIES {
        let query = &q[i * DYN_ACC_DIM..(i + 1) * DYN_ACC_DIM];
        gts.push(brute_force_knn_l2_live_f32(
            &data,
            DYN_ACC_DIM,
            query,
            K,
            &live,
        ));
    }

    let mut out = Vec::with_capacity(DYN_ACC_EPS_VALUES.len());
    for (eps_label, eps) in DYN_ACC_EPS_VALUES {
        let mut rust_exact_count = 0usize;
        let mut cpp_exact_count = 0usize;
        let mut rust_errs: Vec<f64> = Vec::new();
        let mut cpp_errs: Vec<f64> = Vec::new();
        let mut bitexact = true;

        let params = SearchParams { eps, sorted: true };
        for i in 0..DYN_ACC_N_QUERIES {
            let query = &q[i * DYN_ACC_DIM..(i + 1) * DYN_ACC_DIM];
            let gt = &gts[i];

            let mut r_out_idx = vec![0u32; K];
            let mut r_out_dist = vec![0.0f32; K];
            let r_found =
                rust_tree.knn_search_with(query, &mut r_out_idx, &mut r_out_dist, &params);
            let r_pairs: Vec<(u32, f32)> = r_out_idx[..r_found]
                .iter()
                .copied()
                .zip(r_out_dist[..r_found].iter().copied())
                .collect();

            let (c_idx, c_dist) = oracle.knn(query, K, eps);
            let c_pairs: Vec<(u32, f32)> =
                c_idx.iter().copied().zip(c_dist.iter().copied()).collect();

            // `rel_dist_errors` (mean/max error stats) -- plain per-rank
            // scorer, unaffected by the live-set fix (same as the static
            // accuracy rows above).
            let r_score = score_query_f32(gt, &r_pairs);
            let c_score = score_query_f32(gt, &c_pairs);
            rust_errs.extend(r_score.rel_dist_errors);
            cpp_errs.extend(c_score.rel_dist_errors);

            // Exactness -- the LIVE-SET-AWARE tie-aware scorer (see this
            // section's module doc for why plain `score_exact_tie_aware_f32`
            // would be wrong here).
            if score_exact_tie_aware_live_f32(&data, DYN_ACC_DIM, query, gt, &r_pairs, &live) {
                rust_exact_count += 1;
            }
            if score_exact_tie_aware_live_f32(&data, DYN_ACC_DIM, query, gt, &c_pairs, &live) {
                cpp_exact_count += 1;
            }

            if bitexact {
                if r_pairs.len() != c_pairs.len() {
                    bitexact = false;
                } else {
                    for k in 0..r_pairs.len() {
                        if r_pairs[k].0 != c_pairs[k].0
                            || r_pairs[k].1.to_bits() != c_pairs[k].1.to_bits()
                        {
                            bitexact = false;
                            break;
                        }
                    }
                }
            }
        }

        let mean = |v: &[f64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let max = |v: &[f64]| v.iter().cloned().fold(0.0f64, f64::max);

        out.push(AccuracyRow {
            workload: format!("dyn_churn_dim{DYN_ACC_DIM}_f32_k{K}_{eps_label}"),
            n_queries: DYN_ACC_N_QUERIES,
            rust_exact_tie_aware_vs_bruteforce: rust_exact_count as f64 / DYN_ACC_N_QUERIES as f64,
            cpp_exact_tie_aware_vs_bruteforce: cpp_exact_count as f64 / DYN_ACC_N_QUERIES as f64,
            rust_eq_cpp_bitexact: bitexact,
            mean_dist_rel_error_rust: mean(&rust_errs),
            max_dist_rel_error_rust: max(&rust_errs),
            mean_dist_rel_error_cpp: mean(&cpp_errs),
            max_dist_rel_error_cpp: max(&cpp_errs),
            dyn_evidence: Some(evidence),
        });
    }
    out
}

// ============================================================================
// JSON emission -- hand-written, no serde (the RENDERING side,
// `xval::report::render`, parses this JSON via serde_json -- see that
// module's doc for why parsing gets the real dependency and emission
// doesn't). Every value emitted directly by this
// crate's own data (workload names, etc.) is a plain identifier; nothing
// here ever puts a `"`/`\`/control character into one, so those still
// need no escaping. Task 5b's new meta fields (`cpu_model`/`kernel`/
// `cxx_compiler`/`git_sha`) are captured from EXTERNAL commands/files
// (`/proc/cpuinfo`, `uname`, `$CXX --version`, `git`), so -- unlike
// everything else in this file -- they get run through `json_escape`
// defensively before being embedded.
// ============================================================================

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        // JSON has no NaN/Infinity literal; should not occur on real data,
        // but fall back to `null` rather than emit invalid JSON.
        "null".to_string()
    }
}

/// Escapes `"`, `\`, and control characters for a JSON string body. Only
/// used on the handful of Task 5b meta fields captured from external
/// commands/files (see this section's module doc) -- everything else
/// emitted by this file is a plain internally-generated identifier that
/// structurally cannot contain these characters.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out
}

/// One side's `xval::TimingStats` as a nested JSON object -- field names
/// mirror the struct 1:1 (`mean_ms`/`std_ms`/`median_ms`/`min_ms`/`max_ms`/
/// `n`).
fn timing_stats_json(s: &TimingStats) -> String {
    format!(
        "{{ \"mean_ms\": {}, \"std_ms\": {}, \"median_ms\": {}, \"min_ms\": {}, \"max_ms\": {}, \"n\": {} }}",
        json_num(s.mean_ms),
        json_num(s.std_ms),
        json_num(s.median_ms),
        json_num(s.min_ms),
        json_num(s.max_ms),
        s.n
    )
}

/// Standard error of `ratio_means` (`rust.mean_ms / cpp.mean_ms`) via
/// first-order (delta-method) error propagation over each side's OWN
/// standard error OF THE MEAN (`std_ms / sqrt(n)` -- NOT the raw per-rep
/// `std_ms`, which describes single-rep-to-rep variability, not the
/// precision of the mean ESTIMATE itself). Assumes rust/cpp are
/// independent (true here: separate `measure_pair` timed loops, no shared
/// random draw between the two sides).
fn ratio_means_std(rust: &TimingStats, cpp: &TimingStats) -> f64 {
    let ratio = rust.mean_ms / cpp.mean_ms;
    let sem_rust = rust.std_ms / (rust.n as f64).sqrt();
    let sem_cpp = cpp.std_ms / (cpp.n as f64).sqrt();
    let rel_var = (sem_rust / rust.mean_ms).powi(2) + (sem_cpp / cpp.mean_ms).powi(2);
    ratio * rel_var.sqrt()
}

/// One accuracy row's JSON object, conditionally including the dynamic-row
/// evidence fields (`live_count`/`removed_count`/the four `dyn_ops_stats`
/// op-kind totals) only when `dyn_evidence` is `Some` -- i.e. only for the
/// `dyn_churn_*` rows (see `dynamic_accuracy_rows`).
fn accuracy_row_json(r: &AccuracyRow) -> String {
    let base = format!(
        "\"workload\": \"{}\", \"n_queries\": {}, \"rust_exact_tie_aware_vs_bruteforce\": {}, \"cpp_exact_tie_aware_vs_bruteforce\": {}, \"rust_eq_cpp_bitexact\": {}, \"mean_dist_rel_error_rust\": {}, \"max_dist_rel_error_rust\": {}, \"mean_dist_rel_error_cpp\": {}, \"max_dist_rel_error_cpp\": {}",
        r.workload,
        r.n_queries,
        json_num(r.rust_exact_tie_aware_vs_bruteforce),
        json_num(r.cpp_exact_tie_aware_vs_bruteforce),
        r.rust_eq_cpp_bitexact,
        json_num(r.mean_dist_rel_error_rust),
        json_num(r.max_dist_rel_error_rust),
        json_num(r.mean_dist_rel_error_cpp),
        json_num(r.max_dist_rel_error_cpp),
    );
    match r.dyn_evidence {
        Some(e) => format!(
            "    {{ {base}, \"live_count\": {}, \"removed_count\": {}, \"grow_and_add_count\": {}, \"remove_count\": {}, \"readd_count\": {}, \"tombstone_migrations\": {} }}",
            e.live_count, e.removed_count, e.grow_and_add_count, e.remove_count, e.readd_count, e.tombstone_migrations
        ),
        None => format!("    {{ {base} }}"),
    }
}

fn main() {
    progress("starting");

    progress("meta: capturing machine/toolchain fingerprint");
    let meta = format!(
        "  \"meta\": {{\n    \"date\": \"{}\",\n    \"nanoflann_version\": \"1.12.1\",\n    \"rustc\": \"{}\",\n    \"cpu_threads\": {},\n    \"target_cpu_native\": {},\n    \"m2_dynamic\": true,\n    \"cpu_model\": \"{}\",\n    \"kernel\": \"{}\",\n    \"cxx_compiler\": \"{}\",\n    \"git_sha\": \"{}\",\n    \"wsl\": {},\n    \"scoring\": \"{}\",\n    \"gt_methodology\": \"{}\",\n    \"speed_methodology\": \"{}\"\n  }}",
        today_utc(),
        rustc_version(),
        cpu_threads(),
        target_cpu_native(),
        json_escape(&cpu_model()),
        json_escape(&kernel_version()),
        json_escape(&cxx_compiler_version()),
        json_escape(&git_sha()),
        is_wsl(),
        SCORING_NOTE,
        GT_METHODOLOGY_NOTE,
        SPEED_METHODOLOGY_NOTE
    );

    let speed = speed_rows();
    let speed_json = speed
        .iter()
        .map(|r| {
            let ratio_medians = r.rust.median_ms / r.cpp.median_ms;
            let ratio_means = r.rust.mean_ms / r.cpp.mean_ms;
            format!(
                "    {{ \"workload\": \"{}\", \"rust\": {}, \"cpp\": {}, \"rust_ms\": {}, \"cpp_ms\": {}, \"ratio\": {}, \"ratio_means\": {}, \"ratio_means_std\": {}, \"ratio_medians\": {} }}",
                r.workload,
                timing_stats_json(&r.rust),
                timing_stats_json(&r.cpp),
                json_num(r.rust.median_ms),
                json_num(r.cpp.median_ms),
                json_num(ratio_medians),
                json_num(ratio_means),
                json_num(ratio_means_std(&r.rust, &r.cpp)),
                json_num(ratio_medians),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let accuracy = accuracy_rows();
    let accuracy_json = accuracy
        .iter()
        .map(accuracy_row_json)
        .collect::<Vec<_>>()
        .join(",\n");

    let doc = format!(
        "{{\n{meta},\n  \"speed\": [\n{speed_json}\n  ],\n  \"accuracy\": [\n{accuracy_json}\n  ]\n}}"
    );

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    writeln!(lock, "{doc}").expect("failed to write JSON to stdout");

    progress("done");
}
