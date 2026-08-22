//! Criterion KNN benchmarks: Rust `KdTree` vs C++ nanoflann, same process,
//! same timer, same data, n = 100_000 uniform, leaf 10, L2. Two groups:
//!
//! - `knn`: runtime-dim (`DynDim` rust vs `DIM=-1` cpp). dim in {2, 3, 8,
//!   16, 32} x scalar in {f32, f64} x k in {1, 10, 100}, eps 0.
//! - `knn_fixed3`: the HEADLINE number -- `ConstDim::<3>` + `&[[T; 3]]`
//!   (rust) vs the fixed-DIM=3 fast path `RefIndex3F32`/`RefIndex3F64`
//!   (cpp), k in {1, 10, 100}, both scalars.
//!
//! Both groups build the index ONCE outside the timing loop, then iterate a
//! pre-generated 1000-query set round-robin (index counter mod len) inside
//! `b.iter`; `black_box` on inputs and outputs.
//!
//! ID scheme: `knn/{lib}/{dim}/{scalar}/k{k}`, `knn_fixed3/{lib}/{scalar}/k{k}`
//! -- `lib` in {rust, cpp}.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval` for a fair
//! fight -- see `bench_build.rs`'s module doc / `xval::lib` docs for why.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nanoflann_ref::{Metric, RefIndex3F32, RefIndex3F64, RefIndexF32, RefIndexF64};
use nanoflann_rs::{ConstDim, KdTreeBuilder, L2};
use xval::{build_rust_f32, build_rust_f64, cfg_seed, queries, to_array3, to_f32, uniform, BuildThreads, RoundRobin, XMetric};

const N: usize = 100_000;
const LEAF: usize = 10;
const N_QUERIES_POOL: usize = 1000;
const DIMS: [usize; 5] = [2, 3, 8, 16, 32];
const KS: [usize; 3] = [1, 10, 100];

// ============================================================================
// knn: runtime-dim, f32 and f64
// ============================================================================

fn bench_knn_runtime_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn");

    for &dim in &DIMS {
        // f32 arm
        let data64 = uniform(cfg_seed("bench_knn_data_f32", &[dim]), N, dim);
        let data32 = to_f32(&data64);
        let q64 = queries(cfg_seed("bench_knn_q_f32", &[dim]), &data64, dim, N_QUERIES_POOL);
        let q32 = to_f32(&q64);
        let rust32 = build_rust_f32(&data32, dim, XMetric::L2, LEAF, BuildThreads::Sequential);
        let cpp32 = RefIndexF32::build(&data32, dim, Metric::L2, LEAF, 1);

        for &k in &KS {
            group.bench_function(format!("rust/{dim}/f32/k{k}"), |b| {
                let mut rr = RoundRobin::new(&q32, dim);
                b.iter(|| {
                    let query = black_box(rr.next());
                    let (idx, dist) = rust32.knn(query, k, 0.0);
                    black_box((idx, dist));
                })
            });
            group.bench_function(format!("cpp/{dim}/f32/k{k}"), |b| {
                let mut rr = RoundRobin::new(&q32, dim);
                b.iter(|| {
                    let query = black_box(rr.next());
                    let (idx, dist) = cpp32.knn(query, k, 0.0);
                    black_box((idx, dist));
                })
            });
        }

        // f64 arm
        let q64_dataset = &data64;
        let rust64 = build_rust_f64(&data64, dim, XMetric::L2, LEAF, BuildThreads::Sequential);
        let cpp64 = RefIndexF64::build(&data64, dim, Metric::L2, LEAF, 1);
        let q64_pool = queries(cfg_seed("bench_knn_q_f64", &[dim]), q64_dataset, dim, N_QUERIES_POOL);

        for &k in &KS {
            group.bench_function(format!("rust/{dim}/f64/k{k}"), |b| {
                let mut rr = RoundRobin::new(&q64_pool, dim);
                b.iter(|| {
                    let query = black_box(rr.next());
                    let (idx, dist) = rust64.knn(query, k, 0.0);
                    black_box((idx, dist));
                })
            });
            group.bench_function(format!("cpp/{dim}/f64/k{k}"), |b| {
                let mut rr = RoundRobin::new(&q64_pool, dim);
                b.iter(|| {
                    let query = black_box(rr.next());
                    let (idx, dist) = cpp64.knn(query, k, 0.0);
                    black_box((idx, dist));
                })
            });
        }
    }
    group.finish();
}

// ============================================================================
// knn_fixed3: ConstDim<3> + &[[T;3]] (rust) vs RefIndex3F32/F64 (cpp) --
// THE HEADLINE NUMBER.
// ============================================================================

fn bench_knn_fixed3(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_fixed3");
    const DIM: usize = 3;

    // f32
    let data64 = uniform(cfg_seed("bench_knn_fixed3_data_f32", &[]), N, DIM);
    let data32 = to_f32(&data64);
    let arr3_32 = to_array3(&data32);
    let q64 = queries(cfg_seed("bench_knn_fixed3_q_f32", &[]), &data64, DIM, N_QUERIES_POOL);
    let q32 = to_f32(&q64);

    let rust32 = KdTreeBuilder::new(ConstDim::<3>, arr3_32.as_slice()).with_metric(L2).leaf_max_size(LEAF).build_sequential();
    let cpp32 = RefIndex3F32::build(&data32, LEAF, 1);

    for &k in &KS {
        group.bench_function(format!("rust/f32/k{k}"), |b| {
            let mut rr = RoundRobin::new(&q32, DIM);
            let mut out_idx = vec![0u32; k];
            let mut out_dist = vec![0.0f32; k];
            b.iter(|| {
                let query = black_box(rr.next());
                let found = rust32.knn_search(query, &mut out_idx, &mut out_dist);
                black_box(found);
                black_box(&out_idx);
                black_box(&out_dist);
            })
        });
        group.bench_function(format!("cpp/f32/k{k}"), |b| {
            let mut rr = RoundRobin::new(&q32, DIM);
            b.iter(|| {
                let query = black_box(rr.next());
                let (idx, dist) = cpp32.knn(query, k);
                black_box((idx, dist));
            })
        });
    }

    // f64
    let data64b = uniform(cfg_seed("bench_knn_fixed3_data_f64", &[]), N, DIM);
    let arr3_64 = to_array3(&data64b);
    let q64b = queries(cfg_seed("bench_knn_fixed3_q_f64", &[]), &data64b, DIM, N_QUERIES_POOL);

    let rust64 = KdTreeBuilder::new(ConstDim::<3>, arr3_64.as_slice()).with_metric(L2).leaf_max_size(LEAF).build_sequential();
    let cpp64 = RefIndex3F64::build(&data64b, LEAF, 1);

    for &k in &KS {
        group.bench_function(format!("rust/f64/k{k}"), |b| {
            let mut rr = RoundRobin::new(&q64b, DIM);
            let mut out_idx = vec![0u32; k];
            let mut out_dist = vec![0.0f64; k];
            b.iter(|| {
                let query = black_box(rr.next());
                let found = rust64.knn_search(query, &mut out_idx, &mut out_dist);
                black_box(found);
                black_box(&out_idx);
                black_box(&out_dist);
            })
        });
        group.bench_function(format!("cpp/f64/k{k}"), |b| {
            let mut rr = RoundRobin::new(&q64b, DIM);
            b.iter(|| {
                let query = black_box(rr.next());
                let (idx, dist) = cpp64.knn(query, k);
                black_box((idx, dist));
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_knn_runtime_dim, bench_knn_fixed3);
criterion_main!(benches);
