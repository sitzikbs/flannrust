//! M2.5 Task 1 DIAGNOSTIC: asm-extraction probe (temporary measurement
//! tooling, not part of the library or its test surface).
//!
//! Forces monomorphization of the library's `L2::eval` at the shapes the
//! benchmarks use, plus the hand-written access-path variants, each behind an
//! `#[inline(never)] #[no_mangle]` wrapper so the emitted asm has a stable,
//! greppable label.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo rustc -p xval --release \
//!     --example m25_asm -- --emit asm -C llvm-args=-x86-asm-syntax=att
//! ```

// The `chunks_exact(4)` variants below are DELIBERATE: they are one of the
// access-path forms under measurement, benchmarked head-to-head against the
// `as_chunks::<4>()` form clippy prefers (both measured, both bit-identical).
#![allow(clippy::chunks_exact_to_as_chunks)]

use nanoflann_rs::{ConstDim, DynDim, FlatSlice, L2, Distance};

#[inline(never)]
#[no_mangle]
pub fn probe_lib_dyndim32_f32(q: &[f32], ds: &FlatSlice<'_, f32>, idx: usize, dim: usize) -> f32 {
    L2.eval(q, ds, idx, DynDim(dim))
}

#[inline(never)]
#[no_mangle]
pub fn probe_lib_constdim32_f32(q: &[f32], ds: &FlatSlice<'_, f32>, idx: usize) -> f32 {
    L2.eval(q, ds, idx, ConstDim::<32>)
}

#[inline(never)]
#[no_mangle]
pub fn probe_lib_dyndim32_f64(q: &[f64], ds: &FlatSlice<'_, f64>, idx: usize, dim: usize) -> f64 {
    L2.eval(q, ds, idx, DynDim(dim))
}

#[inline(never)]
#[no_mangle]
pub fn probe_lib_constdim3_f32(q: &[f32], ds: &&[[f32; 3]], idx: usize) -> f32 {
    L2.eval(q, ds, idx, ConstDim::<3>)
}

/// Row slice walked with `chunks_exact(4)` — identical summation order,
/// zero per-component bounds checks.
#[inline(never)]
#[no_mangle]
pub fn probe_row_chunks_f32(q: &[f32], row: &[f32], dim: usize) -> f32 {
    let mut result = 0.0f32;
    let multof4 = (dim >> 2) << 2;
    for (a, b) in q[..multof4].chunks_exact(4).zip(row[..multof4].chunks_exact(4)) {
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

type Tree3<'a> = nanoflann_rs::KdTree<f32, ConstDim<3>, &'a [[f32; 3]], L2, u32>;
type TreeDyn<'a> = nanoflann_rs::KdTree<f32, DynDim, FlatSlice<'a, f32>, L2, u32>;

/// Forces the `ConstDim<3>` / `&[[f32;3]]` monomorphization of the whole
/// query path, so `search_level::<f32, ConstDim<3>, ...>` shows up as its own
/// symbol in the emitted asm.
#[inline(never)]
#[no_mangle]
pub fn probe_knn3(t: &Tree3<'_>, q: &[f32], oi: &mut [u32], od: &mut [f32]) -> usize {
    t.knn_search(q, oi, od)
}

/// Same for the runtime-dim `DynDim`/`FlatSlice` path (the dim-32 workload).
#[inline(never)]
#[no_mangle]
pub fn probe_knn_dyn(t: &TreeDyn<'_>, q: &[f32], oi: &mut [u32], od: &mut [f32]) -> usize {
    t.knn_search(q, oi, od)
}

fn main() {
    let data = vec![1.0f32; 32 * 4];
    let ds = FlatSlice::new(&data, 32);
    let q = vec![0.5f32; 32];
    println!("{}", probe_lib_dyndim32_f32(&q, &ds, 1, std::hint::black_box(32)));
    println!("{}", probe_lib_constdim32_f32(&q, &ds, 1));
    println!("{}", probe_row_chunks_f32(&q, &data[32..64], std::hint::black_box(32)));
    let d64 = vec![1.0f64; 32 * 4];
    let ds64 = FlatSlice::new(&d64, 32);
    let q64 = vec![0.5f64; 32];
    println!("{}", probe_lib_dyndim32_f64(&q64, &ds64, 1, std::hint::black_box(32)));
    let a3: &[[f32; 3]] = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    println!("{}", probe_lib_constdim3_f32(&q, &a3, 1));

    let pts3: Vec<[f32; 3]> = (0..1000).map(|i| [i as f32, (i * 7 % 13) as f32, (i * 3 % 5) as f32]).collect();
    let sl: &[[f32; 3]] = &pts3;
    let t3 = nanoflann_rs::KdTreeBuilder::new(ConstDim::<3>, sl).with_metric(L2).leaf_max_size(10).build();
    let mut oi = vec![0u32; 10];
    let mut od = vec![0.0f32; 10];
    println!("{}", probe_knn3(&t3, &[1.0, 2.0, 3.0], &mut oi, &mut od));

    let flat: Vec<f32> = (0..1000 * 32).map(|i| (i % 97) as f32).collect();
    let tdyn = nanoflann_rs::KdTreeBuilder::new(DynDim(32), FlatSlice::new(&flat, 32))
        .with_metric(L2)
        .leaf_max_size(10)
        .build();
    println!("{}", probe_knn_dyn(&tdyn, &q, &mut oi, &mut od));
}
