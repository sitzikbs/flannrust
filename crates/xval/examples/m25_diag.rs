//! M2.5 Task 1 DIAGNOSTIC PROBES (permanent provenance tooling for the
//! published dim-32/64 numbers — not part of the library or its test
//! surface, but retained, not temporary: `docs/EXPERIMENTS.md` names this
//! file's `knn` subcommand as the regenerating command for the README's/
//! `docs/benchmarks.md`'s dim-32/64 tables).
//!
//! Answers the M2.5 T1 brief's questions with measurements. Every subcommand
//! uses the perf-gate methodology (`xval::timed_median_ms`: one untimed
//! warmup + median of 7 timed runs) and expects
//! `RUSTFLAGS="-C target-cpu=native"` + `--release`.
//!
//! Subcommands:
//!   kernel   -- L2 kernel in isolation (library eval vs hand-written access-path variants)
//!   leafn    -- leaf_max_size = n knn (pure kernel sweep) rust vs cpp, several dims
//!   count    -- eval / accum_dist call counts per query via a counting metric wrapper
//!   sweep    -- leaf_max_size sweep with rust-vs-cpp timing (call-overhead regression)
//!   knn      -- plain gate-style knn timing at a given dim (rust vs cpp)

// The `chunks_exact(4)` variants below are DELIBERATE: they are one of the
// access-path forms under measurement, benchmarked head-to-head against the
// `as_chunks::<4>()` form clippy prefers (both measured, both bit-identical).
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::sync::atomic::{AtomicU64, Ordering};

use nanoflann_ref::{Metric, RefIndex3F32, RefIndexF32, RefIndexF64};
use nanoflann_rs::{
    ConstDim, DataSource, Dim, Distance, DynDim, FlatSlice, KdTreeBuilder, L2,
};
use xval::{
    build_rust_f32, build_rust_f64, cfg_seed, queries, timed_median_ms, to_array3, to_f32, uniform,
    BuildThreads, XMetric,
};

const RUNS: usize = 7;

// ===========================================================================
// Hand-written kernels: SAME summation order as `metric.rs`'s `impl_l2!`
// (4-wide unroll with `(d0^2+d1^2)+(d2^2+d3^2)`, descending remainder), only
// the ACCESS PATH differs.
// ===========================================================================

/// (a) per-component access, exactly mirroring `FlatSlice::point_component`
/// (`data[idx * stride + d]`, bounds-checked).
#[inline(always)]
fn l2_component_f32(q: &[f32], data: &[f32], stride: usize, idx: usize, dim: usize) -> f32 {
    let mut result = 0.0f32;
    let multof4 = (dim >> 2) << 2;
    let mut d = 0usize;
    while d < multof4 {
        let diff0 = q[d] - data[idx * stride + d];
        let diff1 = q[d + 1] - data[idx * stride + d + 1];
        let diff2 = q[d + 2] - data[idx * stride + d + 2];
        let diff3 = q[d + 3] - data[idx * stride + d + 3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
        d += 4;
    }
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = q[d + 2] - data[idx * stride + d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - data[idx * stride + d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - data[idx * stride + d];
        result += diff * diff;
    }
    result
}

/// (b) row slice obtained ONCE, then walked. Identical arithmetic order.
#[inline(always)]
fn l2_row_f32(q: &[f32], row: &[f32], dim: usize) -> f32 {
    let mut result = 0.0f32;
    let multof4 = (dim >> 2) << 2;
    let mut d = 0usize;
    while d < multof4 {
        let diff0 = q[d] - row[d];
        let diff1 = q[d + 1] - row[d + 1];
        let diff2 = q[d + 2] - row[d + 2];
        let diff3 = q[d + 3] - row[d + 3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
        d += 4;
    }
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = q[d + 2] - row[d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - row[d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - row[d];
        result += diff * diff;
    }
    result
}

/// (c) row slice, chunked iteration (no explicit indexing at all) --
/// identical arithmetic order, zero bounds checks by construction.
#[inline(always)]
fn l2_row_chunks_f32(q: &[f32], row: &[f32], dim: usize) -> f32 {
    let mut result = 0.0f32;
    let multof4 = (dim >> 2) << 2;
    let qc = q[..multof4].chunks_exact(4);
    let rc = row[..multof4].chunks_exact(4);
    for (a, b) in qc.zip(rc) {
        let diff0 = a[0] - b[0];
        let diff1 = a[1] - b[1];
        let diff2 = a[2] - b[2];
        let diff3 = a[3] - b[3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
    }
    let rem = dim - multof4;
    let d = multof4;
    if rem >= 3 {
        let diff = q[d + 2] - row[d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - row[d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - row[d];
        result += diff * diff;
    }
    result
}

// f64 twins of (a) and (b), for the scalar-width comparison.
#[inline(always)]
fn l2_component_f64(q: &[f64], data: &[f64], stride: usize, idx: usize, dim: usize) -> f64 {
    let mut result = 0.0f64;
    let multof4 = (dim >> 2) << 2;
    let mut d = 0usize;
    while d < multof4 {
        let diff0 = q[d] - data[idx * stride + d];
        let diff1 = q[d + 1] - data[idx * stride + d + 1];
        let diff2 = q[d + 2] - data[idx * stride + d + 2];
        let diff3 = q[d + 3] - data[idx * stride + d + 3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
        d += 4;
    }
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = q[d + 2] - data[idx * stride + d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - data[idx * stride + d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - data[idx * stride + d];
        result += diff * diff;
    }
    result
}

#[inline(always)]
fn l2_row_f64(q: &[f64], row: &[f64], dim: usize) -> f64 {
    let mut result = 0.0f64;
    let multof4 = (dim >> 2) << 2;
    let mut d = 0usize;
    while d < multof4 {
        let diff0 = q[d] - row[d];
        let diff1 = q[d + 1] - row[d + 1];
        let diff2 = q[d + 2] - row[d + 2];
        let diff3 = q[d + 3] - row[d + 3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
        d += 4;
    }
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = q[d + 2] - row[d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - row[d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - row[d];
        result += diff * diff;
    }
    result
}

#[inline(always)]
fn l2_row_chunks_f64(q: &[f64], row: &[f64], dim: usize) -> f64 {
    let mut result = 0.0f64;
    let multof4 = (dim >> 2) << 2;
    let qc = q[..multof4].chunks_exact(4);
    let rc = row[..multof4].chunks_exact(4);
    for (a, b) in qc.zip(rc) {
        let diff0 = a[0] - b[0];
        let diff1 = a[1] - b[1];
        let diff2 = a[2] - b[2];
        let diff3 = a[3] - b[3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
    }
    let rem = dim - multof4;
    let d = multof4;
    if rem >= 3 {
        let diff = q[d + 2] - row[d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - row[d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - row[d];
        result += diff * diff;
    }
    result
}

/// (d) per-component access with bounds checks REMOVED (`get_unchecked`) but
/// otherwise identical to (a): separates "bounds-check cost" from
/// "straight-line-block / vectorization" effects.
#[inline(always)]
fn l2_component_unchecked_f32(q: &[f32], data: &[f32], stride: usize, idx: usize, dim: usize) -> f32 {
    unsafe {
        let mut result = 0.0f32;
        let multof4 = (dim >> 2) << 2;
        let mut d = 0usize;
        while d < multof4 {
            let diff0 = *q.get_unchecked(d) - *data.get_unchecked(idx * stride + d);
            let diff1 = *q.get_unchecked(d + 1) - *data.get_unchecked(idx * stride + d + 1);
            let diff2 = *q.get_unchecked(d + 2) - *data.get_unchecked(idx * stride + d + 2);
            let diff3 = *q.get_unchecked(d + 3) - *data.get_unchecked(idx * stride + d + 3);
            result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
            d += 4;
        }
        let rem = dim - multof4;
        if rem >= 3 {
            let diff = *q.get_unchecked(d + 2) - *data.get_unchecked(idx * stride + d + 2);
            result += diff * diff;
        }
        if rem >= 2 {
            let diff = *q.get_unchecked(d + 1) - *data.get_unchecked(idx * stride + d + 1);
            result += diff * diff;
        }
        if rem >= 1 {
            let diff = *q.get_unchecked(d) - *data.get_unchecked(idx * stride + d);
            result += diff * diff;
        }
        result
    }
}

/// (e) row slice walked with `as_chunks::<4>()` (clippy's preferred form of
/// (c)) — identical arithmetic order, fixed-size array chunks.
#[inline(always)]
fn l2_row_arraychunks_f32(q: &[f32], row: &[f32], dim: usize) -> f32 {
    let mut result = 0.0f32;
    let multof4 = (dim >> 2) << 2;
    let (qc, _) = q[..multof4].as_chunks::<4>();
    let (rc, _) = row[..multof4].as_chunks::<4>();
    for (a, b) in qc.iter().zip(rc.iter()) {
        let diff0 = a[0] - b[0];
        let diff1 = a[1] - b[1];
        let diff2 = a[2] - b[2];
        let diff3 = a[3] - b[3];
        result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
    }
    let rem = dim - multof4;
    let d = multof4;
    if rem >= 3 {
        let diff = q[d + 2] - row[d + 2];
        result += diff * diff;
    }
    if rem >= 2 {
        let diff = q[d + 1] - row[d + 1];
        result += diff * diff;
    }
    if rem >= 1 {
        let diff = q[d] - row[d];
        result += diff * diff;
    }
    result
}

// ===========================================================================
// Counting metric: delegates to L2 verbatim (bit-identical results), counts
// `eval` (= point visits / leaf-scan distance evaluations) and `accum_dist`
// (= interior-node visits + one per dim in compute_initial_distances).
// ===========================================================================

static EVAL_COUNT: AtomicU64 = AtomicU64::new(0);
static ACCUM_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Default, Clone, Copy)]
struct CountingL2;

macro_rules! impl_counting {
    ($t:ty) => {
        impl Distance<$t> for CountingL2 {
            type DistanceType = $t;
            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                d: D,
            ) -> $t {
                EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
                L2.eval(query, ds, idx, d)
            }
            fn accum_dist(&self, a: $t, b: $t, axis: usize) -> $t {
                ACCUM_COUNT.fetch_add(1, Ordering::Relaxed);
                L2.accum_dist(a, b, axis)
            }
        }
    };
}
impl_counting!(f32);
impl_counting!(f64);

// ===========================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    match cmd {
        "kernel" => kernel_probe(),
        "leafn" => leafn_probe(),
        "count" => count_probe(),
        "sweep" => sweep_probe(),
        "knn" => knn_probe(),
        "dump" => dump_probe(&args[2]),
        other => {
            eprintln!("unknown subcommand {other}");
            std::process::exit(2);
        }
    }
}

// --- Q1/Q3: kernel isolation + access path -------------------------------

fn kernel_probe() {
    const N: usize = 100_000;
    println!("# kernel isolation, n={N}, one query swept over ALL points, ns/point");
    println!("# order: sequential (idx 0..n) and permuted (leaf-scan-like scatter)");
    println!("dim,scalar,order,variant,ns_per_point,checksum");

    for &dim in &[8usize, 16, 32, 64] {
        let data64 = uniform(cfg_seed("m25_kernel", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_kernel_q", &[dim]), &data64, dim, 1);
        let q32 = to_f32(&q64);
        // Permutation mimicking a leaf scan's scattered `vind` accessors.
        let perm: Vec<usize> = {
            let mut v: Vec<usize> = (0..N).collect();
            // deterministic LCG shuffle
            let mut s: u64 = 0x9E3779B97F4A7C15;
            for i in (1..N).rev() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (s >> 33) as usize % (i + 1);
                v.swap(i, j);
            }
            v
        };
        let flat32 = FlatSlice::new(&data32, dim);
        let flat64 = FlatSlice::new(&data64, dim);

        // BIT-PARITY CHECK: every hand-written variant must reproduce the
        // library kernel's result bit-for-bit on every point (same summation
        // order, different access path only).
        for i in 0..N {
            let base = L2.eval(&q32, &flat32, i, DynDim(dim)).to_bits();
            let row = &data32[i * dim..i * dim + dim];
            assert_eq!(base, l2_component_f32(&q32, &data32, dim, i, dim).to_bits());
            assert_eq!(base, l2_row_f32(&q32, row, dim).to_bits());
            assert_eq!(base, l2_row_chunks_f32(&q32, row, dim).to_bits());
            assert_eq!(base, l2_row_arraychunks_f32(&q32, row, dim).to_bits());
            assert_eq!(base, l2_component_unchecked_f32(&q32, &data32, dim, i, dim).to_bits());
            let base64 = L2.eval(&q64, &flat64, i, DynDim(dim)).to_bits();
            let row64 = &data64[i * dim..i * dim + dim];
            assert_eq!(base64, l2_component_f64(&q64, &data64, dim, i, dim).to_bits());
            assert_eq!(base64, l2_row_f64(&q64, row64, dim).to_bits());
            assert_eq!(base64, l2_row_chunks_f64(&q64, row64, dim).to_bits());
        }
        eprintln!("# dim {dim}: all hand-written variants bit-identical to L2::eval on all {N} points");

        for order in ["seq", "perm"] {
            let idxs: Vec<usize> = if order == "seq" { (0..N).collect() } else { perm.clone() };

            // f32 library eval, DynDim
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += L2.eval(&q32, &flat32, i, DynDim(dim));
                }
                acc = s;
            });
            report(dim, "f32", order, "lib_dyndim", ms, N, acc as f64);

            // f32 library eval, ConstDim (dim 32 arm only where it matters, but do all)
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += match dim {
                        8 => L2.eval(&q32, &flat32, i, ConstDim::<8>),
                        16 => L2.eval(&q32, &flat32, i, ConstDim::<16>),
                        32 => L2.eval(&q32, &flat32, i, ConstDim::<32>),
                        _ => L2.eval(&q32, &flat32, i, ConstDim::<64>),
                    };
                }
                acc = s;
            });
            report(dim, "f32", order, "lib_constdim", ms, N, acc as f64);

            // (a) per-component
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += l2_component_f32(&q32, &data32, dim, i, dim);
                }
                acc = s;
            });
            report(dim, "f32", order, "hand_component", ms, N, acc as f64);

            // (b) row slice
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += l2_row_f32(&q32, &data32[i * dim..i * dim + dim], dim);
                }
                acc = s;
            });
            report(dim, "f32", order, "hand_row", ms, N, acc as f64);

            // (c) row slice, chunks_exact
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += l2_row_chunks_f32(&q32, &data32[i * dim..i * dim + dim], dim);
                }
                acc = s;
            });
            report(dim, "f32", order, "hand_row_chunks", ms, N, acc as f64);

            // (e) row slice, as_chunks::<4>
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += l2_row_arraychunks_f32(&q32, &data32[i * dim..i * dim + dim], dim);
                }
                acc = s;
            });
            report(dim, "f32", order, "hand_row_arraychunks", ms, N, acc as f64);

            // (d) per-component, unchecked
            let mut acc = 0.0f32;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f32;
                for &i in &idxs {
                    s += l2_component_unchecked_f32(&q32, &data32, dim, i, dim);
                }
                acc = s;
            });
            report(dim, "f32", order, "hand_component_unchecked", ms, N, acc as f64);

            // f64 library eval, DynDim
            let mut acc = 0.0f64;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f64;
                for &i in &idxs {
                    s += L2.eval(&q64, &flat64, i, DynDim(dim));
                }
                acc = s;
            });
            report(dim, "f64", order, "lib_dyndim", ms, N, acc);

            // f64 (a)
            let mut acc = 0.0f64;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f64;
                for &i in &idxs {
                    s += l2_component_f64(&q64, &data64, dim, i, dim);
                }
                acc = s;
            });
            report(dim, "f64", order, "hand_component", ms, N, acc);

            // f64 (b)
            let mut acc = 0.0f64;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f64;
                for &i in &idxs {
                    s += l2_row_f64(&q64, &data64[i * dim..i * dim + dim], dim);
                }
                acc = s;
            });
            report(dim, "f64", order, "hand_row", ms, N, acc);

            // f64 (c)
            let mut acc = 0.0f64;
            let ms = timed_median_ms(RUNS, || {
                let mut s = 0.0f64;
                for &i in &idxs {
                    s += l2_row_chunks_f64(&q64, &data64[i * dim..i * dim + dim], dim);
                }
                acc = s;
            });
            report(dim, "f64", order, "hand_row_chunks", ms, N, acc);
        }
    }
}

fn report(dim: usize, scalar: &str, order: &str, variant: &str, ms: f64, n: usize, chk: f64) {
    let ns_per = ms * 1e6 / n as f64;
    println!("{dim},{scalar},{order},{variant},{ns_per:.3},{chk:e}");
}

// --- Q1: leaf_max_size = n makes knn a pure kernel sweep ------------------

fn leafn_probe() {
    const N: usize = 100_000;
    const NQ: usize = 200;
    const K: usize = 10;
    println!("# leaf_max_size = n (single-leaf tree) -> knn is a pure linear kernel sweep");
    println!("# n={N}, {NQ} queries, k={K}, median of {RUNS}");
    println!("dim,scalar,rust_ms,cpp_ms,ratio");
    for &dim in &[8usize, 16, 32, 64] {
        let data64 = uniform(cfg_seed("m25_leafn", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_leafn_q", &[dim]), &data64, dim, NQ);
        let q32 = to_f32(&q64);

        // f32
        let rust = build_rust_f32(&data32, dim, XMetric::L2, N, BuildThreads::Sequential);
        let cpp = RefIndexF32::build(&data32, dim, Metric::L2, N, 1);
        let mut oi = vec![0u32; K];
        let mut od = vec![0.0f32; K];
        let rust_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * dim..(c + 1) * dim];
                std::hint::black_box(rust.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * dim..(c + 1) * dim];
                std::hint::black_box(cpp.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        println!("{dim},f32,{rust_ms:.3},{cpp_ms:.3},{:.3}", rust_ms / cpp_ms);

        // f64
        let rust = build_rust_f64(&data64, dim, XMetric::L2, N, BuildThreads::Sequential);
        let cpp = RefIndexF64::build(&data64, dim, Metric::L2, N, 1);
        let mut od = vec![0.0f64; K];
        let rust_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q64[c * dim..(c + 1) * dim];
                std::hint::black_box(rust.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q64[c * dim..(c + 1) * dim];
                std::hint::black_box(cpp.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        println!("{dim},f64,{rust_ms:.3},{cpp_ms:.3},{:.3}", rust_ms / cpp_ms);
    }
}

// --- Q4/Q5: visit counts --------------------------------------------------

fn count_probe() {
    const N: usize = 100_000;
    const NQ: usize = 1000;
    const K: usize = 10;
    println!("# visit counts per query via a delegating CountingL2 metric");
    println!("# eval_calls = points scanned in leaves; accum_calls = interior-node visits + dim (initial dists)");
    println!("dim,leaf,scalar,queries,eval_per_query,interior_per_query,frac_points_scanned");
    for (dim, leaf) in [(3usize, 10usize), (8, 10), (16, 10), (32, 10), (3, 1), (3, 4), (3, 32), (3, 128), (3, 1024)] {
        let data64 = uniform(cfg_seed("m25_count", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_count_q", &[dim]), &data64, dim, NQ);
        let q32 = to_f32(&q64);
        let ds = FlatSlice::new(&data32, dim);
        let tree = KdTreeBuilder::new(DynDim(dim), ds)
            .with_metric(CountingL2)
            .leaf_max_size(leaf)
            .threads(nanoflann_rs::BuildThreads::Sequential)
            .build();
        EVAL_COUNT.store(0, Ordering::Relaxed);
        ACCUM_COUNT.store(0, Ordering::Relaxed);
        let mut oi = vec![0u32; K];
        let mut od = vec![0.0f32; K];
        for c in 0..NQ {
            let q = &q32[c * dim..(c + 1) * dim];
            tree.knn_search(q, &mut oi, &mut od);
        }
        let ev = EVAL_COUNT.load(Ordering::Relaxed) as f64 / NQ as f64;
        let ac = ACCUM_COUNT.load(Ordering::Relaxed) as f64 / NQ as f64 - dim as f64;
        println!(
            "{dim},{leaf},f32,{NQ},{ev:.1},{ac:.1},{:.4}",
            ev / N as f64
        );
    }
}

// --- Q4: dump the exact data/query set so a standalone instrumented C++
// build can count ITS node/point visits on identical input ----------------

fn dump_probe(dir: &str) {
    const N: usize = 100_000;
    const NQ: usize = 1000;
    const K: usize = 10;
    // The exact dataset/query set `sweep_probe` times, so the instrumented
    // C++ probe can count node visits for the SAME workload the sweep timed.
    {
        let data64 = uniform(cfg_seed("m25_sweep", &[3]), N, 3);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_sweep_q", &[3]), &data64, 3, 10_000);
        let q32 = to_f32(&q64);
        let write = |name: &str, v: &[f32]| {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(format!("{dir}/{name}"), bytes).unwrap();
        };
        write("data_3s.bin", &data32);
        write("query_3s.bin", &q32);
    }
    for (dim, leaf) in [(3usize, 10usize), (32, 10)] {
        let data64 = uniform(cfg_seed("m25_count", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_count_q", &[dim]), &data64, dim, NQ);
        let q32 = to_f32(&q64);
        let write = |name: &str, v: &[f32]| {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(format!("{dir}/{name}"), bytes).unwrap();
        };
        write(&format!("data_{dim}.bin"), &data32);
        write(&format!("query_{dim}.bin"), &q32);
        // Re-emit the Rust-side counts for exactly this config, plus the
        // bit-exact knn output digest, so the C++ probe can be compared to it.
        let ds = FlatSlice::new(&data32, dim);
        let tree = KdTreeBuilder::new(DynDim(dim), ds)
            .with_metric(CountingL2)
            .leaf_max_size(leaf)
            .threads(nanoflann_rs::BuildThreads::Sequential)
            .build();
        EVAL_COUNT.store(0, Ordering::Relaxed);
        ACCUM_COUNT.store(0, Ordering::Relaxed);
        let mut oi = vec![0u32; K];
        let mut od = vec![0.0f32; K];
        let mut digest: u64 = 0xcbf29ce484222325;
        for c in 0..NQ {
            let q = &q32[c * dim..(c + 1) * dim];
            tree.knn_search(q, &mut oi, &mut od);
            for j in 0..K {
                digest ^= oi[j] as u64;
                digest = digest.wrapping_mul(0x100000001b3);
                digest ^= od[j].to_bits() as u64;
                digest = digest.wrapping_mul(0x100000001b3);
            }
        }
        // `compute_initial_distances` calls `accum_dist` only for axes where
        // the query lies OUTSIDE the root bbox — count those independently so
        // `accum_calls - initial` is exactly the interior-node visit count.
        let bbox = tree.root_bbox();
        let mut initial = 0u64;
        for c in 0..NQ {
            let q = &q32[c * dim..(c + 1) * dim];
            for i in 0..dim {
                if q[i] < bbox[i].low || q[i] > bbox[i].high {
                    initial += 1;
                }
            }
        }
        let accum = ACCUM_COUNT.load(Ordering::Relaxed);
        println!(
            "rust dim={dim} leaf={leaf} n={N} nq={NQ} k={K} eval_calls={} accum_calls={accum} initial_accum={initial} interior={} digest={digest:#x}",
            EVAL_COUNT.load(Ordering::Relaxed),
            accum - initial,
        );
    }
}

// --- Q5/Q6: leaf sweep at dim 3, rust vs cpp ------------------------------

fn sweep_probe() {
    const N: usize = 100_000;
    const NQ: usize = 10_000;
    const K: usize = 10;
    println!("# dim-3 f32 ConstDim<3>/&[[f32;3]] vs cpp fixed-DIM=3, leaf_max_size sweep");
    println!("# n={N}, {NQ} queries per timed run, k={K}, median of {RUNS}");
    println!("leaf,rust_ms,cpp_ms,ratio,delta_ns_per_query");
    let data64 = uniform(cfg_seed("m25_sweep", &[3]), N, 3);
    let data32 = to_f32(&data64);
    let arr3 = to_array3(&data32);
    let q64 = queries(cfg_seed("m25_sweep_q", &[3]), &data64, 3, NQ);
    let q32 = to_f32(&q64);
    for &leaf in &[1usize, 2, 4, 10, 16, 32, 64, 128, 512, 1024] {
        let slice: &[[f32; 3]] = &arr3;
        let rust = KdTreeBuilder::new(ConstDim::<3>, slice)
            .with_metric(L2)
            .leaf_max_size(leaf)
            .threads(nanoflann_rs::BuildThreads::Sequential)
            .build();
        let cpp = RefIndex3F32::build(&data32, leaf, 1);
        let mut oi = vec![0u32; K];
        let mut od = vec![0.0f32; K];
        let rust_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * 3..(c + 1) * 3];
                std::hint::black_box(rust.knn_search(q, &mut oi, &mut od));
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * 3..(c + 1) * 3];
                std::hint::black_box(cpp.knn_into(q, K, &mut oi, &mut od));
            }
        });
        let delta_ns = (rust_ms - cpp_ms) * 1e6 / NQ as f64;
        println!("{leaf},{rust_ms:.3},{cpp_ms:.3},{:.4},{delta_ns:.2}", rust_ms / cpp_ms);
    }
}

// --- plain knn timing at a dim (gate-style) -------------------------------

fn knn_probe() {
    const N: usize = 100_000;
    const NQ: usize = 200;
    const K: usize = 10;
    const LEAF: usize = 10;
    println!("# gate-style knn (DynDim/FlatSlice vs DIM=-1), n={N}, {NQ} queries, k={K}, leaf={LEAF}");
    println!("dim,scalar,rust_ms,cpp_ms,ratio");
    for &dim in &[8usize, 16, 32, 64] {
        let data64 = uniform(cfg_seed("m25_knn", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("m25_knn_q", &[dim]), &data64, dim, NQ);
        let q32 = to_f32(&q64);

        let rust = build_rust_f32(&data32, dim, XMetric::L2, LEAF, BuildThreads::Sequential);
        let cpp = RefIndexF32::build(&data32, dim, Metric::L2, LEAF, 1);
        let mut oi = vec![0u32; K];
        let mut od = vec![0.0f32; K];
        let rust_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * dim..(c + 1) * dim];
                std::hint::black_box(rust.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q32[c * dim..(c + 1) * dim];
                std::hint::black_box(cpp.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        println!("{dim},f32,{rust_ms:.3},{cpp_ms:.3},{:.3}", rust_ms / cpp_ms);

        let rust = build_rust_f64(&data64, dim, XMetric::L2, LEAF, BuildThreads::Sequential);
        let cpp = RefIndexF64::build(&data64, dim, Metric::L2, LEAF, 1);
        let mut od = vec![0.0f64; K];
        let rust_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q64[c * dim..(c + 1) * dim];
                std::hint::black_box(rust.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        let cpp_ms = timed_median_ms(RUNS, || {
            for c in 0..NQ {
                let q = &q64[c * dim..(c + 1) * dim];
                std::hint::black_box(cpp.knn_into(q, K, 0.0, &mut oi, &mut od));
            }
        });
        println!("{dim},f64,{rust_ms:.3},{cpp_ms:.3},{:.3}", rust_ms / cpp_ms);
    }
}
