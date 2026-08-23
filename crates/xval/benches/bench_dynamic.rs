//! Criterion DYNAMIC (add/remove/re-add) benchmarks: Rust `DynamicKdTree`
//! (`nanoflann_rs::dynamic`) vs the C++ oracle's
//! `KDTreeSingleIndexDynamicAdaptor` (`nanoflann_ref::RefDynIndexF32`), same
//! process, same timer, same data -- n = 100_000 capacity, dim 3, f32, leaf
//! 10. Reuses M2 Task 4's cross-validation plumbing directly (`xval::
//! GrowableFlat`, the same op-application shapes `xval::apply_dyn_op_f32`
//! encodes) -- NOT reimplemented here.
//!
//! METHODOLOGY NOTE: nanoflann's dynamic `addPoints` is inherently
//! O(heavy) per point -- every call re-walks the merge-and-rebuild loop
//! (nanoflann.hpp:2647-2675; see `crates/nanoflann-rs/src/dynamic.rs`'s
//! `add_points` doc comment for the exact schedule), NOT an amortized-O(1)
//! insert: a genuinely-new point can trigger a cascading rebuild of every
//! slot up to the highest slot a binary-counter-style merge touches. Our
//! Rust port matches those semantics EXACTLY (same `first0bit` slot
//! selection, same "rebuild every slot up to the highest touched, even a
//! no-op call's slot 0" quirk) -- so these benchmarks compare the SAME
//! algorithm implemented twice, not two different designs. A speed
//! difference here reflects implementation quality (allocation patterns,
//! codegen, etc.), not an algorithmic one.
//!
//! Three groups:
//!
//! - `dyn_add`: from an EMPTY forest, add in 100 contiguity-legal batches of
//!   1000 (100k total), timed as ONE build-up per iteration -- a fresh
//!   forest is constructed inside `b.iter` every time (there is no
//!   "rewind" for an add-only build-up). `sample_size(10)`.
//! - `dyn_churn`: a forest pre-built with all 100k points (ONCE, outside all
//!   timing), then EACH timed iteration removes the same seeded 1000
//!   indices and re-adds them (reactivation). This is SELF-RESTORING: after
//!   a remove-then-re-add pass, both the tombstone bookkeeping and slot
//!   residency are back to their exact pre-iteration state (re-adding a
//!   removed index reactivates it in place, `dynamic.rs`'s module doc), so
//!   the SAME forest object is safely reused, net-unmutated, across every
//!   sample/iteration without an explicit rebuild-per-iteration cost.
//!   `sample_size(10)`.
//! - `dyn_knn_after_churn`: a SEPARATE forest, also pre-built with 100k
//!   points then churned ONCE (1000 removes + 1000 re-adds, outside all
//!   timing) to reach a realistic post-churn structure (tombstone
//!   migrations, multi-slot layout), then a 1000-query round-robin k=10 knn
//!   loop, zero-allocation both sides (`knn_search`/`knn_into` writing into
//!   reused out-buffers).
//!
//! ID scheme: `dyn_add/{lib}`, `dyn_churn/{lib}`, `dyn_knn_after_churn/{lib}`
//! -- `lib` in {rust, cpp}.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval` for a
//! fair fight -- see `bench_build.rs`'s module doc / `xval::lib` docs for
//! why.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nanoflann_ref::RefDynIndexF32;
use nanoflann_rs::{DynDim, DynamicKdTreeBuilder};
use xval::{cfg_seed, queries, sample_distinct_indices, to_f32, uniform, GrowableFlat, RoundRobin};

const CAP: usize = 100_000;
const DIM: usize = 3;
const LEAF: usize = 10;
const BATCH: usize = 1000;
const BATCHES: usize = 100;
const CHURN: usize = 1000;
const K: usize = 10;
const N_QUERIES_POOL: usize = 1000;

// ============================================================================
// dyn_add: from empty, 100 contiguous batches of 1000 (100k total)
// ============================================================================

fn bench_dyn_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("dyn_add");
    group.sample_size(10);

    let data = to_f32(&uniform(cfg_seed("bench_dyn_add_data", &[CAP, DIM]), CAP, DIM));

    group.bench_function("rust", |b| {
        b.iter(|| {
            let growable = GrowableFlat::new(black_box(&data), DIM);
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
            black_box(tree.tree_count());
        })
    });

    group.bench_function("cpp", |b| {
        b.iter(|| {
            let mut oracle = RefDynIndexF32::build(black_box(&data), DIM, LEAF, CAP);
            let mut start = 0u32;
            for _ in 0..BATCHES {
                let end = start + BATCH as u32;
                oracle.set_current_n(end as usize);
                oracle.add_points(start, end - 1);
                start = end;
            }
            black_box(oracle.tree_count());
        })
    });

    group.finish();
}

// ============================================================================
// dyn_churn: pre-built 100k (outside timing), each iteration = 1000 removes
// + 1000 re-adds (self-restoring -- see module doc)
// ============================================================================

fn bench_dyn_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("dyn_churn");
    group.sample_size(10);

    let data = to_f32(&uniform(cfg_seed("bench_dyn_churn_data", &[CAP, DIM]), CAP, DIM));
    let churn_idx = sample_distinct_indices(cfg_seed("bench_dyn_churn_idx", &[CAP, CHURN]), CAP, CHURN);

    // Rust: build once, add all CAP points -- outside timing.
    let growable = GrowableFlat::new(&data, DIM);
    let mut rust_tree =
        DynamicKdTreeBuilder::new(DynDim(DIM), &growable).leaf_max_size(LEAF).maximum_point_count(CAP).build();
    growable.set_current_n(CAP);
    rust_tree.add_points(0, CAP - 1);

    group.bench_function("rust", |b| {
        b.iter(|| {
            for &idx in &churn_idx {
                rust_tree.remove_point(black_box(idx));
            }
            for &idx in &churn_idx {
                rust_tree.add_points(idx, idx);
            }
            black_box(rust_tree.removed_len());
        })
    });

    // C++: build once, add all CAP points -- outside timing.
    let mut cpp_tree = RefDynIndexF32::build(&data, DIM, LEAF, CAP);
    cpp_tree.set_current_n(CAP);
    cpp_tree.add_points(0, (CAP - 1) as u32);

    group.bench_function("cpp", |b| {
        b.iter(|| {
            for &idx in &churn_idx {
                cpp_tree.remove_point(black_box(idx));
            }
            for &idx in &churn_idx {
                cpp_tree.add_points(idx as u32, idx as u32);
            }
            black_box(cpp_tree.removed_count());
        })
    });

    group.finish();
}

// ============================================================================
// dyn_knn_after_churn: separate 100k forest, churned ONCE (1000 removes +
// 1000 re-adds, outside timing), then 1000-query round-robin k=10 knn,
// zero-allocation both sides.
// ============================================================================

fn bench_dyn_knn_after_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("dyn_knn_after_churn");

    let data64 = uniform(cfg_seed("bench_dyn_knn_after_churn_data", &[CAP, DIM]), CAP, DIM);
    let data = to_f32(&data64);
    let q64 = queries(cfg_seed("bench_dyn_knn_after_churn_q", &[CAP, DIM]), &data64, DIM, N_QUERIES_POOL);
    let q = to_f32(&q64);
    let churn_idx = sample_distinct_indices(cfg_seed("bench_dyn_knn_after_churn_idx", &[CAP, CHURN]), CAP, CHURN);

    // Rust: build, add all, churn once -- all outside timing.
    let growable = GrowableFlat::new(&data, DIM);
    let mut rust_tree =
        DynamicKdTreeBuilder::new(DynDim(DIM), &growable).leaf_max_size(LEAF).maximum_point_count(CAP).build();
    growable.set_current_n(CAP);
    rust_tree.add_points(0, CAP - 1);
    for &idx in &churn_idx {
        rust_tree.remove_point(idx);
    }
    for &idx in &churn_idx {
        rust_tree.add_points(idx, idx);
    }

    // C++: same, outside timing.
    let mut cpp_tree = RefDynIndexF32::build(&data, DIM, LEAF, CAP);
    cpp_tree.set_current_n(CAP);
    cpp_tree.add_points(0, (CAP - 1) as u32);
    for &idx in &churn_idx {
        cpp_tree.remove_point(idx);
    }
    for &idx in &churn_idx {
        cpp_tree.add_points(idx as u32, idx as u32);
    }

    group.bench_function("rust", |b| {
        let mut rr = RoundRobin::new(&q, DIM);
        let mut out_idx = vec![0u32; K];
        let mut out_dist = vec![0.0f32; K];
        b.iter(|| {
            let query = black_box(rr.next());
            let found = rust_tree.knn_search(query, &mut out_idx, &mut out_dist);
            black_box(found);
        })
    });
    group.bench_function("cpp", |b| {
        let mut rr = RoundRobin::new(&q, DIM);
        let mut out_idx = vec![0u32; K];
        let mut out_dist = vec![0.0f32; K];
        b.iter(|| {
            let query = black_box(rr.next());
            let found = cpp_tree.knn_into(query, K, 0.0, &mut out_idx, &mut out_dist);
            black_box(found);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_dyn_add, bench_dyn_churn, bench_dyn_knn_after_churn);
criterion_main!(benches);
