//! Criterion BUILD benchmarks: Rust `KdTree` vs C++ nanoflann, same process,
//! same timer, same data, `f32`. Three groups:
//!
//! - `build`: matrix n in {10_000, 100_000, 1_000_000} x dim in {3, 8} x
//!   {rust/seq, rust/par, cpp/seq, cpp/par}. BUDGET NOTE: n = 1_000_000 SKIPS
//!   dim = 8 (only dim = 3 runs at 1M) to hold the whole-suite runtime target
//!   of ~15 minutes -- 1M x dim8 x 4 arms would roughly double the heaviest
//!   part of the matrix for one more data point.
//! - `build_fixed3`: `ConstDim::<3>` + `&[[f32; 3]]` (rust) vs `nfr3_build_f`
//!   (cpp) at n in {100_000, 1_000_000}, sequential only -- isolates
//!   `ConstDim`'s per-node bbox allocation shape from the DynDim matrix
//!   above (the audit finding this bench exists to cover).
//! - `leaf_sweep`: n = 100_000, dim = 3, f32, leaf_max_size in {1, 4, 10, 16,
//!   32, 50, 128, 1024} -- BOTH build time and k=10 knn time, for both libs.
//!   Reproduces nanoflann's README Sec 2.1 methodology and validates
//!   leaf = 10 for our node layout.
//!
//! ID scheme: `build/{lib}/{n}/{dim}/{threads}`, `build_fixed3/{lib}/{n}`,
//! `leaf_sweep/{lib}/{leaf}/{phase}` -- `lib` in {rust, cpp}.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval` for a fair
//! fight -- the C++ oracle is always built `-O3 -march=native
//! -ffp-contract=off` (crates/nanoflann-ref/build.rs), so without
//! `target-cpu=native` the Rust side is handicapped to a generic baseline
//! ISA. See also `xval::lib` module docs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nanoflann_ref::{Metric, RefIndex3F32, RefIndexF32};
use nanoflann_rs::{ConstDim, KdTreeBuilder, L2};
use std::time::Duration;
use xval::{build_rust_f32, cfg_seed, queries, to_array3, to_f32, uniform, BuildThreads, RoundRobin, XMetric};

const LEAF: usize = 10;

fn gen_f32(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    to_f32(&uniform(seed, n, dim))
}

// ============================================================================
// build: n x dim x {rust/seq, rust/par, cpp/seq, cpp/par}
// ============================================================================

fn bench_build_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &n in &[10_000usize, 100_000usize, 1_000_000usize] {
        if n == 1_000_000 {
            // Heavy case: fewer samples, explicit measurement window (see
            // module doc's budget note).
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(5));
        } else {
            group.sample_size(100);
            group.measurement_time(Duration::from_secs(5));
        }
        for &dim in &[3usize, 8usize] {
            if n == 1_000_000 && dim == 8 {
                continue; // budget skip, see module doc
            }
            let data = gen_f32(cfg_seed("bench_build_matrix", &[n, dim]), n, dim);

            group.bench_function(format!("rust/{n}/{dim}/seq"), |b| {
                b.iter(|| {
                    let idx = build_rust_f32(black_box(&data), dim, XMetric::L2, LEAF, BuildThreads::Sequential);
                    black_box(idx.size());
                })
            });
            group.bench_function(format!("rust/{n}/{dim}/par"), |b| {
                b.iter(|| {
                    let idx = build_rust_f32(black_box(&data), dim, XMetric::L2, LEAF, BuildThreads::Auto);
                    black_box(idx.size());
                })
            });
            group.bench_function(format!("cpp/{n}/{dim}/seq"), |b| {
                b.iter(|| {
                    let idx = RefIndexF32::build(black_box(&data), dim, Metric::L2, LEAF, 1);
                    black_box(idx.size());
                })
            });
            group.bench_function(format!("cpp/{n}/{dim}/par"), |b| {
                b.iter(|| {
                    let idx = RefIndexF32::build(black_box(&data), dim, Metric::L2, LEAF, 0);
                    black_box(idx.size());
                })
            });
        }
    }
    group.finish();
}

// ============================================================================
// build_fixed3: ConstDim<3> + &[[f32;3]] (rust) vs nfr3_build_f (cpp)
// ============================================================================

fn bench_build_fixed3(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_fixed3");
    for &n in &[100_000usize, 1_000_000usize] {
        if n == 1_000_000 {
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(5));
        } else {
            group.sample_size(100);
            group.measurement_time(Duration::from_secs(5));
        }
        let flat = gen_f32(cfg_seed("bench_build_fixed3", &[n]), n, 3);
        let arr3 = to_array3(&flat);

        group.bench_function(format!("rust/{n}"), |b| {
            b.iter(|| {
                let ds: &[[f32; 3]] = black_box(arr3.as_slice());
                let tree = KdTreeBuilder::new(ConstDim::<3>, ds).with_metric(L2).leaf_max_size(LEAF).build_sequential();
                black_box(tree.size());
            })
        });
        group.bench_function(format!("cpp/{n}"), |b| {
            b.iter(|| {
                let idx = RefIndex3F32::build(black_box(&flat), LEAF, 1);
                black_box(&idx);
            })
        });
    }
    group.finish();
}

// ============================================================================
// leaf_sweep: n = 100_000, dim = 3, f32 -- build AND k=10 knn time, both libs
// ============================================================================

fn bench_leaf_sweep(c: &mut Criterion) {
    const N: usize = 100_000;
    const DIM: usize = 3;
    const K: usize = 10;
    const N_QUERIES_POOL: usize = 1000;

    let data_f64 = uniform(cfg_seed("bench_leaf_sweep_data", &[N, DIM]), N, DIM);
    let data = to_f32(&data_f64);
    let q_f64 = queries(cfg_seed("bench_leaf_sweep_q", &[N, DIM]), &data_f64, DIM, N_QUERIES_POOL);
    let q = to_f32(&q_f64);

    let mut group = c.benchmark_group("leaf_sweep");
    group.sample_size(10); // builds are the expensive phase; applies to both phases in this group

    for &leaf in &[1usize, 4, 10, 16, 32, 50, 128, 1024] {
        group.bench_function(format!("rust/{leaf}/build"), |b| {
            b.iter(|| {
                let idx = build_rust_f32(black_box(&data), DIM, XMetric::L2, leaf, BuildThreads::Sequential);
                black_box(idx.size());
            })
        });
        group.bench_function(format!("cpp/{leaf}/build"), |b| {
            b.iter(|| {
                let idx = RefIndexF32::build(black_box(&data), DIM, Metric::L2, leaf, 1);
                black_box(idx.size());
            })
        });

        // knn phase: build ONCE outside the timing loop, then time k=10
        // queries round-robin over the pre-generated pool.
        let rust_idx = build_rust_f32(&data, DIM, XMetric::L2, leaf, BuildThreads::Sequential);
        let cpp_idx = RefIndexF32::build(&data, DIM, Metric::L2, leaf, 1);

        group.bench_function(format!("rust/{leaf}/knn"), |b| {
            let mut rr = RoundRobin::new(&q, DIM);
            b.iter(|| {
                let query = black_box(rr.next());
                let (idx, dist) = rust_idx.knn(query, K, 0.0);
                black_box((idx, dist));
            })
        });
        group.bench_function(format!("cpp/{leaf}/knn"), |b| {
            let mut rr = RoundRobin::new(&q, DIM);
            b.iter(|| {
                let query = black_box(rr.next());
                let (idx, dist) = cpp_idx.knn(query, K, 0.0);
                black_box((idx, dist));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build_matrix, bench_build_fixed3, bench_leaf_sweep);
criterion_main!(benches);
