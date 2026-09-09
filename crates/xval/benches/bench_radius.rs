//! Criterion RADIUS benchmark: Rust `KdTree` vs C++ nanoflann, same process,
//! same timer, n = 100_000, dim 3, f32, leaf 10, sorted = true.
//!
//! Two selectivities, calibrated ONCE before timing (not re-derived per
//! iteration): `sel10`/`sel1000` are the squared L2 distance to the 10th and
//! 1000th nearest neighbor of one fixed probe query, asked of the C++
//! index's own `knn` (taking the last returned distance) -- so the radius
//! passed to `radius_search` on BOTH sides always selects (approximately)
//! that many points, letting the comparison isolate radius-search cost from
//! an arbitrarily-chosen, possibly-degenerate radius.
//!
//! ID scheme: `radius/{lib}/{sel}` -- `lib` in {rust, cpp}, `sel` in
//! {sel10, sel1000}.
//!
//! Both sides are ZERO-(RE)ALLOCATION per query: `radius_into` writes into
//! caller-owned buffers (allocated once per `bench_function`, outside
//! `b.iter`, and reused across every iteration) instead of the allocating
//! `radius()` convenience wrappers.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval` for a fair
//! fight -- see `bench_build.rs`'s module doc / `xval::lib` docs for why.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flannrust::ResultItem;
use nanoflann_ref::{Metric, RefIndexF32};
use xval::{build_rust_f32, cfg_seed, queries, to_f32, uniform, BuildThreads, RoundRobin, XMetric};

const N: usize = 100_000;
const DIM: usize = 3;
const LEAF: usize = 10;
const N_QUERIES_POOL: usize = 1000;

fn bench_radius(c: &mut Criterion) {
    let data64 = uniform(cfg_seed("bench_radius_data", &[]), N, DIM);
    let data32 = to_f32(&data64);
    let q64 = queries(
        cfg_seed("bench_radius_q", &[]),
        &data64,
        DIM,
        N_QUERIES_POOL,
    );
    let q32 = to_f32(&q64);

    let rust_idx = build_rust_f32(&data32, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
    let cpp_idx = RefIndexF32::build(&data32, DIM, Metric::L2, LEAF, 1);

    // Calibrate ONCE, before any timing: probe query is the first entry of
    // the pre-generated query pool (deterministic via the fixed seed above).
    let probe = &q32[0..DIM];
    let (_idx10, dist10) = cpp_idx.knn(probe, 10, 0.0);
    let (_idx1000, dist1000) = cpp_idx.knn(probe, 1000, 0.0);
    let sel10 = *dist10
        .last()
        .expect("cpp knn(k=10) must return at least one neighbor");
    let sel1000 = *dist1000
        .last()
        .expect("cpp knn(k=1000) must return at least one neighbor");

    let mut group = c.benchmark_group("radius");
    for (sel_name, radius) in [("sel10", sel10), ("sel1000", sel1000)] {
        group.bench_function(format!("rust/{sel_name}"), |b| {
            let mut rr = RoundRobin::new(&q32, DIM);
            let mut out: Vec<ResultItem<u32, f32>> = Vec::new();
            b.iter(|| {
                let query = black_box(rr.next());
                let found = rust_idx.radius_into(query, radius, true, 0.0, &mut out);
                black_box(found);
            })
        });
        group.bench_function(format!("cpp/{sel_name}"), |b| {
            let mut rr = RoundRobin::new(&q32, DIM);
            let mut out_idx: Vec<u32> = Vec::new();
            let mut out_dist: Vec<f32> = Vec::new();
            b.iter(|| {
                let query = black_box(rr.next());
                let found =
                    cpp_idx.radius_into(query, radius, true, 0.0, &mut out_idx, &mut out_dist);
                black_box(found);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_radius);
criterion_main!(benches);
