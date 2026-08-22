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
//! - Speed: the perf-gate's four workloads (identical median-of-7
//!   methodology, via `xval::timed_median_ms`/`xval::median_of` -- see
//!   `tests/perf_gate.rs`) plus `build_1M_dim3_f32` seq/par. Every query
//!   loop (knn/radius) is ALLOCATION-SYMMETRIC: both sides use reused,
//!   caller-owned out-buffers (`knn_into`/`radius_into`, not the
//!   allocating `knn()`/`radius()` convenience wrappers) -- see this file's
//!   emitted `meta.speed_methodology`.
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
//!   exact definition (controller ruling: the plain order-independent
//!   index-SET match is too strict on duplicate-heavy datasets, where
//!   multiple points can be genuinely tied at the k-th distance boundary).
//!   `mean_dist_rel_error_*`/`max_dist_rel_error_*` are unaffected -- still
//!   from `xval::score_query_f32`'s `rel_dist_errors`.

use nanoflann_ref::{Metric, RefIndex3F32, RefIndexF32};
use nanoflann_rs::{ConstDim, KdTreeBuilder, ResultItem, L2};
use std::io::Write;
use xval::{
    brute_force_knn_l2_f32, build_rust_f32, cfg_seed, queries, score_exact_tie_aware_f32, score_query_f32,
    timed_median_ms, to_array3, to_f32, uniform, with_duplicates, BuildThreads, RoundRobin, XMetric,
};

/// One-line description of the accuracy scoring methodology, embedded
/// verbatim in the emitted JSON's `meta.scoring` field.
const SCORING_NOTE: &str = "exact_tie_aware_vs_bruteforce: true iff the k returned distances bit-match the ground truth's k smallest distances positionally (both sorted ascending) AND every returned index's recomputed true distance equals its reported distance -- accepts any valid k-th-boundary tie resolution while still catching a wrong point, wrong distance, or missed closer neighbor.";

/// One-line description of how ground-truth distances are computed,
/// embedded verbatim in the emitted JSON's `meta.gt_methodology` field
/// (controller ruling round 2 -- see task-13-report.md's "Investigation"
/// section: an independently-written summation is not reliably bit-exact
/// against either tree's internal arithmetic, so GT must route through the
/// same kernel).
const GT_METHODOLOGY_NOTE: &str = "ground-truth distances computed via the library's own L2 metric kernel (nanoflann_rs::L2::eval, the same code path KdTree::knn_search uses internally for leaf points), NOT an independently-written summation; point SELECTION is a dumb linear scan over every index, independent of any tree's traversal order.";

/// One-line description of the speed-timing allocation methodology,
/// embedded verbatim in the emitted JSON's `meta.speed_methodology` field
/// (controller ruling: state this as the METHODOLOGY, not a caveat -- both
/// sides are allocation-symmetric, not just Rust).
const SPEED_METHODOLOGY_NOTE: &str = "both sides zero-allocation per query: out-buffers (Vec<u32>/Vec<T>) are allocated once per timed workload and reused across every query, via knn_search/radius_search (rust) and knn_into/radius_into (nanoflann_ref's caller-buffer methods, backed directly by the raw FFI's out-pointer writes) -- not the allocating knn()/radius() convenience wrappers on either side.";

const RUNS: usize = 7;

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
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
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
    std::env::var("RUSTFLAGS").map(|v| v.contains("target-cpu=native")).unwrap_or(false)
}

// ============================================================================
// Speed: reuse the perf-gate's four workloads + build_1M_dim3 seq/par
// ============================================================================

struct SpeedRow {
    workload: &'static str,
    rust_ms: f64,
    cpp_ms: f64,
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
        let rust_ms = timed_median_ms(RUNS, || {
            let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
            std::hint::black_box(idx.size());
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);
            std::hint::black_box(idx.size());
        });
        rows.push(SpeedRow { workload: "build_100k_dim3_f32_seq", rust_ms, cpp_ms });
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
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f32; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.knn_into(query, K, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        });
        rows.push(SpeedRow { workload: "knn_dim3_f32_k10", rust_ms, cpp_ms });
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

        let rust_ms = timed_median_ms(RUNS, || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f64; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx = vec![0u32; K];
            let mut out_dist = vec![0.0f64; K];
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        });
        rows.push(SpeedRow { workload: "knn_dyn_dim8_f64_k10", rust_ms, cpp_ms });
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
        let radius = *dist.last().expect("calibration knn must return at least one neighbor");

        let rust_ms = timed_median_ms(RUNS, || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out: Vec<ResultItem<u32, f32>> = Vec::new();
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = rust_tree.radius_into(query, radius, true, 0.0, &mut out);
                std::hint::black_box(found);
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            let mut rr = RoundRobin::new(&q, DIM);
            let mut out_idx: Vec<u32> = Vec::new();
            let mut out_dist: Vec<f32> = Vec::new();
            for _ in 0..N_QUERIES {
                let query = std::hint::black_box(rr.next());
                let found = cpp_tree.radius_into(query, radius, true, 0.0, &mut out_idx, &mut out_dist);
                std::hint::black_box(found);
            }
        });
        rows.push(SpeedRow { workload: "radius_dim3_f32", rust_ms, cpp_ms });
    }

    // ---- build_1M_dim3_f32 seq/par ----
    progress("speed: build_1M_dim3_f32 seq/par");
    {
        const N: usize = 1_000_000;
        const DIM: usize = 3;
        const LEAF: usize = 10;
        let data = to_f32(&uniform(cfg_seed("report_build_1m", &[N, DIM]), N, DIM));

        let rust_seq_ms = timed_median_ms(RUNS, || {
            let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
            std::hint::black_box(idx.size());
        });
        let cpp_seq_ms = timed_median_ms(RUNS, || {
            let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 1);
            std::hint::black_box(idx.size());
        });
        rows.push(SpeedRow { workload: "build_1M_dim3_f32_seq", rust_ms: rust_seq_ms, cpp_ms: cpp_seq_ms });

        let rust_par_ms = timed_median_ms(RUNS, || {
            let idx = build_rust_f32(&data, DIM, XMetric::L2, LEAF, BuildThreads::Auto);
            std::hint::black_box(idx.size());
        });
        let cpp_par_ms = timed_median_ms(RUNS, || {
            let idx = RefIndexF32::build(&data, DIM, Metric::L2, LEAF, 0);
            std::hint::black_box(idx.size());
        });
        rows.push(SpeedRow { workload: "build_1M_dim3_f32_par", rust_ms: rust_par_ms, cpp_ms: cpp_par_ms });
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
}

const K: usize = 10;
const N_ACC_QUERIES: usize = 2000;
const EPS_VALUES: [(&str, f32); 3] = [("eps0", 0.0), ("eps0.1", 0.1), ("eps1", 1.0)];

fn accuracy_rows_for_dataset(name: &str, dim: usize, data: &[f32]) -> Vec<AccuracyRow> {
    progress(&format!("accuracy: {name} (n={}, dim={dim})", data.len() / dim));

    let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let q64 = queries(cfg_seed("report_accuracy_q", &[name.len(), dim]), &data64, dim, N_ACC_QUERIES);
    let q = to_f32(&q64);

    let rust_tree = build_rust_f32(data, dim, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_tree = RefIndexF32::build(data, dim, Metric::L2, 10, 1);

    // Ground truth: computed ONCE per dataset (independent of eps), reused
    // across every eps arm below.
    progress(&format!("accuracy: {name} brute-force GT for {N_ACC_QUERIES} queries (O(n_queries*n), slow part)"));
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

            let r_pairs: Vec<(u32, f32)> = r_idx.iter().copied().zip(r_dist.iter().copied()).collect();
            let c_pairs: Vec<(u32, f32)> = c_idx.iter().copied().zip(c_dist.iter().copied()).collect();

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

        let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
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
        });
    }
    out
}

fn accuracy_rows() -> Vec<AccuracyRow> {
    let mut rows = Vec::new();

    let uniform_dim3 = to_f32(&uniform(cfg_seed("report_acc_uniform_dim3", &[]), 50_000, 3));
    rows.extend(accuracy_rows_for_dataset("uniform", 3, &uniform_dim3));

    let uniform_dim8 = to_f32(&uniform(cfg_seed("report_acc_uniform_dim8", &[]), 50_000, 8));
    rows.extend(accuracy_rows_for_dataset("uniform", 8, &uniform_dim8));

    let dup_dim3 = to_f32(&with_duplicates(cfg_seed("report_acc_dup_dim3", &[]), 20_000, 3, 0.3));
    rows.extend(accuracy_rows_for_dataset("with_duplicates", 3, &dup_dim3));

    rows
}

// ============================================================================
// JSON emission -- hand-written, no serde. Every value here is a plain
// number/string/bool; nothing in this crate ever puts a `"` or control
// character into a workload name, so no escaping logic is needed.
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

fn main() {
    progress("starting");

    let meta = format!(
        "  \"meta\": {{\n    \"date\": \"{}\",\n    \"nanoflann_version\": \"1.12.1\",\n    \"rustc\": \"{}\",\n    \"cpu_threads\": {},\n    \"target_cpu_native\": {},\n    \"scoring\": \"{}\",\n    \"gt_methodology\": \"{}\",\n    \"speed_methodology\": \"{}\"\n  }}",
        today_utc(),
        rustc_version(),
        cpu_threads(),
        target_cpu_native(),
        SCORING_NOTE,
        GT_METHODOLOGY_NOTE,
        SPEED_METHODOLOGY_NOTE
    );

    let speed = speed_rows();
    let speed_json = speed
        .iter()
        .map(|r| {
            format!(
                "    {{ \"workload\": \"{}\", \"rust_ms\": {}, \"cpp_ms\": {}, \"ratio\": {} }}",
                r.workload,
                json_num(r.rust_ms),
                json_num(r.cpp_ms),
                json_num(r.rust_ms / r.cpp_ms)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let accuracy = accuracy_rows();
    let accuracy_json = accuracy
        .iter()
        .map(|r| {
            format!(
                "    {{ \"workload\": \"{}\", \"n_queries\": {}, \"rust_exact_tie_aware_vs_bruteforce\": {}, \"cpp_exact_tie_aware_vs_bruteforce\": {}, \"rust_eq_cpp_bitexact\": {}, \"mean_dist_rel_error_rust\": {}, \"max_dist_rel_error_rust\": {}, \"mean_dist_rel_error_cpp\": {}, \"max_dist_rel_error_cpp\": {} }}",
                r.workload,
                r.n_queries,
                json_num(r.rust_exact_tie_aware_vs_bruteforce),
                json_num(r.cpp_exact_tie_aware_vs_bruteforce),
                r.rust_eq_cpp_bitexact,
                json_num(r.mean_dist_rel_error_rust),
                json_num(r.max_dist_rel_error_rust),
                json_num(r.mean_dist_rel_error_cpp),
                json_num(r.max_dist_rel_error_cpp)
            )
        })
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
