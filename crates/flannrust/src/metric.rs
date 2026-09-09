//! Distance metric adaptors, ported verbatim from nanoflann 1.12.1's
//! `L1_Adaptor` / `L2_Adaptor` / `L2_Simple_Adaptor` / `SO2_Adaptor` /
//! `SO3_Adaptor` (nanoflann.hpp lines 552-783). The 4-way unroll and
//! descending-remainder summation order is preserved exactly so our
//! distances match the C++ bit-for-bit on well-behaved (e.g. integer-valued)
//! data — later milestones cross-validate against the reference
//! implementation.

use crate::data_source::DataSource;
use crate::dim::Dim;
use crate::scalar::{DistanceValue, Scalar};

/// A nanoflann-style metric. Implementations are plain values, so stateful
/// metrics (weights, scales) work: construct one and hand it to the builder.
///
/// Contract (same as C++): `eval` is the full point-to-point distance
/// (= `evalMetric`); `accum_dist` is the per-axis component distance
/// (= `accum_dist`), which the tree uses for split-plane bounds — every
/// metric MUST be per-axis decomposable.
///
/// L1/L2/L2Simple return SQUARED (L2) / summed-absolute (L1) distances;
/// SO2 returns an UNsquared wrapped angle of only the last dimension.
pub trait Distance<T: Scalar>: Send + Sync {
    /// The metric's accumulated distance/radius type (may differ from `T`).
    type DistanceType: DistanceValue;

    /// `d` carries the dimensionality — pass a [`crate::ConstDim`] where the
    /// caller's dimension is a compile-time constant (e.g. the hot search
    /// path over a `ConstDim<N>`-built tree) so the monomorphized body sees
    /// `d.dim()` constant-fold to `N`, letting the optimizer fully unroll
    /// fixed-width kernels (like `L2`'s 4-wide unroll) and elide bounds
    /// checks; pass [`crate::DynDim`] for a runtime dimension. A custom
    /// `Distance` implementation's `eval` should be generic over `D: Dim`
    /// and call `d.dim()` wherever a plain dimension count is needed.
    fn eval<DS: DataSource<T> + ?Sized, D: Dim>(
        &self,
        query: &[T],
        ds: &DS,
        idx: usize,
        d: D,
    ) -> Self::DistanceType;

    /// Per-axis component distance between two coordinate values on `axis`
    /// — the tree uses this for split-plane bound checks, so every metric
    /// MUST be per-axis decomposable.
    fn accum_dist(&self, a: T, b: T, axis: usize) -> Self::DistanceType;
}

/// Sum-of-absolute-differences ("Manhattan") metric (nanoflann's `L1_Adaptor`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L1;

/// **Squared** Euclidean metric, unrolled 4-wide for high-dimensional data
/// (nanoflann's `L2_Adaptor`). Returns squared distance, matching the C++.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2;

/// **Squared** Euclidean metric with a plain (non-unrolled) summation loop,
/// suited to low-dimensional data (nanoflann's `L2_Simple_Adaptor`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2Simple;

/// **Squared** Euclidean metric using hardware FMA (`mul_add`). Faster than
/// [`L2`] on FMA-capable hardware (all x86-64-v3+ and aarch64) but **NOT
/// bit-identical** to `L2` or the C++ oracle — results may differ at the
/// last bit due to fused multiply-add rounding. Use when throughput matters
/// more than exact reproducibility.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2Fma;

/// Shortest angular distance on the last dimension only, assuming inputs are
/// already wrapped into `[-pi, pi]` (nanoflann's `SO2_Adaptor`). NOT squared;
/// the result lies in `[0, pi]`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SO2;

/// Delegates to `L2Simple` verbatim (nanoflann's `SO3_Adaptor`, intended for
/// quaternion-like coordinates where plain squared Euclidean distance over
/// all components is the right metric).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SO3;

/// `if x < 0.0 { -x } else { x }` — NOT `.abs()`. This mirrors the C++'s
/// `diff < DistanceType(0) ? -diff : diff` idiom exactly (used verbatim by
/// nanoflann's SO2 adaptor, and functionally equivalent to `std::abs` used
/// by L1 for all finite inputs). Note this differs from `f32::abs`/`f64::abs`
/// on NaN: the ternary leaves NaN's sign bit untouched (`NaN < 0.0` is always
/// `false`, so it returns `x` unchanged), whereas `.abs()` always clears the
/// sign bit.
macro_rules! abs_ternary {
    ($e:expr) => {{
        let v = $e;
        if v < 0.0 {
            -v
        } else {
            v
        }
    }};
}

/// Debug-only trust-boundary check on a third-party `DataSource::point_row`
/// implementation: an O(1) spot check (first and last of the `dim`
/// components the row path is about to consume) against `point_component`,
/// so a contract-violating `point_row` (see its doc) diverges LOUDLY in
/// tests/debug builds instead of silently in the row-path fast arithmetic.
/// A full O(dim) check would defeat the point of the fast path even in
/// debug builds at high dim, so this is deliberately a spot check, not
/// exhaustive verification -- `DataSource::point_row`'s doc still calls out
/// that a violating implementation is undefined-result (not unsafe), and
/// this is a best-effort net, not a soundness guarantee. Compiles to
/// nothing in release (`debug_assert!`).
#[inline]
fn debug_check_point_row_contract<T: Scalar, DS: DataSource<T> + ?Sized>(
    ds: &DS,
    idx: usize,
    row: &[T],
    dim: usize,
) {
    if dim == 0 {
        return;
    }
    // NaN-tolerant equality: a conforming `point_row` over NaN-poisoned data
    // must not trip this check (`NaN == NaN` is false). `Scalar` has no
    // `is_nan`; `x.partial_cmp(&x).is_none()` is the `PartialOrd`-only test.
    #[inline]
    fn same<T: Scalar>(a: T, b: T) -> bool {
        a == b || (a.partial_cmp(&a).is_none() && b.partial_cmp(&b).is_none())
    }
    debug_assert!(
        same(row[0], ds.point_component(idx, 0)),
        "DataSource::point_row contract violated at idx={idx}: row[0] != point_component(idx, 0)"
    );
    debug_assert!(
        same(row[dim - 1], ds.point_component(idx, dim - 1)),
        "DataSource::point_row contract violated at idx={idx}: row[dim-1] != point_component(idx, dim-1)"
    );
}

/// Bounds-check-free chunked row walk for `L1::eval` -- same lever as
/// `l2_eval_row` below, mirrored for sum-of-absolute-differences. The
/// ternary matches `abs_ternary!` exactly (`if v < 0 { -v } else { v }`);
/// `zero - v` stands in for unary negation (`Scalar` has no `Neg` bound) and
/// is bit-identical to `-v` here because this arm is only reached when `v`
/// is strictly negative (never `-0.0`/`+0.0`), so there is no zero-sign
/// ambiguity between the two spellings.
#[inline]
fn l1_eval_row<T: Scalar>(query: &[T], row: &[T], dim: usize) -> T {
    let zero = T::default();
    #[inline]
    fn abs_t<T: Scalar>(v: T, zero: T) -> T {
        if v < zero {
            zero - v
        } else {
            v
        }
    }
    let mut result: T = zero;
    let multof4 = (dim >> 2) << 2;
    let (qc, _) = query[..multof4].as_chunks::<4>();
    let (rc, _) = row[..multof4].as_chunks::<4>();
    for (a, b) in qc.iter().zip(rc.iter()) {
        let diff0 = abs_t(a[0] - b[0], zero);
        let diff1 = abs_t(a[1] - b[1], zero);
        let diff2 = abs_t(a[2] - b[2], zero);
        let diff3 = abs_t(a[3] - b[3], zero);
        result = result + ((diff0 + diff1) + (diff2 + diff3));
    }
    let d = multof4;
    let rem = dim - multof4;
    if rem >= 3 {
        result = result + abs_t(query[d + 2] - row[d + 2], zero);
    }
    if rem >= 2 {
        result = result + abs_t(query[d + 1] - row[d + 1], zero);
    }
    if rem >= 1 {
        result = result + abs_t(query[d] - row[d], zero);
    }
    result
}

macro_rules! impl_l1 {
    ($t:ty) => {
        impl Distance<$t> for L1 {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                let dim = dim.dim();
                if let Some(row) = ds.point_row(idx) {
                    if row.len() >= dim && query.len() >= dim {
                        debug_check_point_row_contract(ds, idx, row, dim);
                        return l1_eval_row(query, row, dim);
                    }
                }
                let mut result: $t = 0.0;
                let multof4 = (dim >> 2) << 2; // largest multiple of 4
                let mut d = 0usize;
                while d < multof4 {
                    let diff0 = abs_ternary!(query[d] - ds.point_component(idx, d));
                    let diff1 = abs_ternary!(query[d + 1] - ds.point_component(idx, d + 1));
                    let diff2 = abs_ternary!(query[d + 2] - ds.point_component(idx, d + 2));
                    let diff3 = abs_ternary!(query[d + 3] - ds.point_component(idx, d + 3));
                    // Parentheses break the dependency chain — same order as C++.
                    result += (diff0 + diff1) + (diff2 + diff3);
                    d += 4;
                }
                // Process last 0-3 components. Replicates the C++'s
                // fall-through switch: terms added in descending order
                // d+2, d+1, d+0.
                let rem = dim - multof4;
                if rem >= 3 {
                    result += abs_ternary!(query[d + 2] - ds.point_component(idx, d + 2));
                }
                if rem >= 2 {
                    result += abs_ternary!(query[d + 1] - ds.point_component(idx, d + 1));
                }
                if rem >= 1 {
                    result += abs_ternary!(query[d] - ds.point_component(idx, d));
                }
                result
            }

            fn accum_dist(&self, a: $t, b: $t, _axis: usize) -> $t {
                abs_ternary!(a - b)
            }
        }
    };
}

/// Bounds-check-free chunked row walk for `L2::eval`, used whenever
/// `DataSource::point_row` hands back a contiguous slice. Computes the
/// IDENTICAL summation as the per-component fallback loop below --
/// `(d0*d0 + d1*d1) + (d2*d2 + d3*d3)` per 4-wide chunk (via `as_chunks`, one
/// slice-length check instead of 8 per-element bounds checks), then the same
/// descending-remainder tail (d+2, d+1, d+0) -- so results are bit-identical;
/// only the *loads* move (row-indexed rather than `point_component`-indexed).
/// Row-vs-fallback bit-equality is asserted by
/// `metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_{f32,f64}`.
#[inline]
fn l2_eval_row<T: Scalar>(query: &[T], row: &[T], dim: usize) -> T {
    let mut result: T = T::default(); // zero, per Scalar's contract
    let multof4 = (dim >> 2) << 2; // largest multiple of 4
    let (qc, _) = query[..multof4].as_chunks::<4>();
    let (rc, _) = row[..multof4].as_chunks::<4>();
    for (a, b) in qc.iter().zip(rc.iter()) {
        let diff0 = a[0] - b[0];
        let diff1 = a[1] - b[1];
        let diff2 = a[2] - b[2];
        let diff3 = a[3] - b[3];
        // Parentheses break the dependency chain -- same order as the
        // fallback loop / C++.
        result = result + ((diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3));
    }
    // Process last 0-3 components. Replicates the fallback loop's /
    // C++'s fall-through switch: terms added in descending order d+2, d+1,
    // d+0.
    let d = multof4;
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = query[d + 2] - row[d + 2];
        result = result + diff * diff;
    }
    if rem >= 2 {
        let diff = query[d + 1] - row[d + 1];
        result = result + diff * diff;
    }
    if rem >= 1 {
        let diff = query[d] - row[d];
        result = result + diff * diff;
    }
    result
}

macro_rules! impl_l2 {
    ($t:ty) => {
        impl Distance<$t> for L2 {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                let dim = dim.dim();
                if let Some(row) = ds.point_row(idx) {
                    if row.len() >= dim && query.len() >= dim {
                        debug_check_point_row_contract(ds, idx, row, dim);
                        return l2_eval_row(query, row, dim);
                    }
                }
                let mut result: $t = 0.0;
                let multof4 = (dim >> 2) << 2; // largest multiple of 4
                let mut d = 0usize;
                while d < multof4 {
                    let diff0 = query[d] - ds.point_component(idx, d);
                    let diff1 = query[d + 1] - ds.point_component(idx, d + 1);
                    let diff2 = query[d + 2] - ds.point_component(idx, d + 2);
                    let diff3 = query[d + 3] - ds.point_component(idx, d + 3);
                    // Parentheses break the dependency chain — same order as C++.
                    result += (diff0 * diff0 + diff1 * diff1) + (diff2 * diff2 + diff3 * diff3);
                    d += 4;
                }
                // Process last 0-3 components. Replicates the C++'s
                // fall-through switch: terms added in descending order
                // d+2, d+1, d+0.
                let rem = dim - multof4;
                if rem >= 3 {
                    let diff = query[d + 2] - ds.point_component(idx, d + 2);
                    result += diff * diff;
                }
                if rem >= 2 {
                    let diff = query[d + 1] - ds.point_component(idx, d + 1);
                    result += diff * diff;
                }
                if rem >= 1 {
                    let diff = query[d] - ds.point_component(idx, d);
                    result += diff * diff;
                }
                result
            }

            fn accum_dist(&self, a: $t, b: $t, _axis: usize) -> $t {
                let diff = a - b;
                diff * diff
            }
        }
    };
}

macro_rules! impl_l2_simple {
    ($t:ty) => {
        impl Distance<$t> for L2Simple {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                let dim = dim.dim();
                let mut result: $t = 0.0;
                for d in 0..dim {
                    let diff = query[d] - ds.point_component(idx, d);
                    result += diff * diff;
                }
                result
            }

            fn accum_dist(&self, a: $t, b: $t, _axis: usize) -> $t {
                let diff = a - b;
                diff * diff
            }
        }
    };
}

macro_rules! impl_l2_fma {
    ($t:ty) => {
        impl Distance<$t> for L2Fma {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                let dim = dim.dim();
                let mut result: $t = 0.0;
                if let Some(row) = ds.point_row(idx) {
                    if row.len() >= dim && query.len() >= dim {
                        for (q, p) in query[..dim].iter().zip(&row[..dim]) {
                            let diff = *q - *p;
                            result = diff.mul_add(diff, result);
                        }
                        return result;
                    }
                }
                for d in 0..dim {
                    let diff = query[d] - ds.point_component(idx, d);
                    result = diff.mul_add(diff, result);
                }
                result
            }

            fn accum_dist(&self, a: $t, b: $t, _axis: usize) -> $t {
                let diff = a - b;
                diff * diff
            }
        }
    };
}

macro_rules! impl_so2 {
    ($t:ty, $pi:expr) => {
        impl Distance<$t> for SO2 {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                let dim = dim.dim();
                self.accum_dist(query[dim - 1], ds.point_component(idx, dim - 1), dim - 1)
            }

            /// Returns the shortest angular distance between `a` and `b`,
            /// assuming both are already wrapped into `[-pi, pi]` (single-shot
            /// wrap: does NOT correctly handle inputs further outside that
            /// range). Orientation matters: `diff = b - a`, matching the C++
            /// exactly. Result lies in `[0, pi]` and is UNsquared (unlike
            /// L1/L2/L2Simple).
            fn accum_dist(&self, a: $t, b: $t, _axis: usize) -> $t {
                let mut diff = b - a;
                let pi: $t = $pi;
                if diff > pi {
                    diff -= 2.0 * pi;
                } else if diff < -pi {
                    diff += 2.0 * pi;
                }
                abs_ternary!(diff)
            }
        }
    };
}

macro_rules! impl_so3 {
    ($t:ty) => {
        impl Distance<$t> for SO3 {
            type DistanceType = $t;

            fn eval<DS: DataSource<$t> + ?Sized, D: Dim>(
                &self,
                query: &[$t],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> $t {
                L2Simple.eval(query, ds, idx, dim)
            }

            fn accum_dist(&self, a: $t, b: $t, axis: usize) -> $t {
                L2Simple.accum_dist(a, b, axis)
            }
        }
    };
}

impl_l1!(f32);
impl_l1!(f64);
impl_l2!(f32);
impl_l2!(f64);
impl_l2_simple!(f32);
impl_l2_simple!(f64);

impl_l2_fma!(f32);
impl_l2_fma!(f64);
impl_so2!(f32, core::f32::consts::PI);
impl_so2!(f64, core::f64::consts::PI);
impl_so3!(f32);
impl_so3!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_source::{FlatSlice, OwnedRows};
    use crate::dim::{ConstDim, DynDim};

    // ---- Test 1: L2 dim-2 exact ----

    #[test]
    fn l2_dim2_exact_f64() {
        let q = [1.0f64, 2.0];
        let p: &[[f64; 2]] = &[[4.0, 6.0]];
        let got = L2.eval(&q, &p, 0, ConstDim::<2>);
        assert_eq!(got, 25.0);
    }

    #[test]
    fn l2_dim2_exact_f32() {
        let q = [1.0f32, 2.0];
        let p: &[[f32; 2]] = &[[4.0, 6.0]];
        let got = L2.eval(&q, &p, 0, ConstDim::<2>);
        assert_eq!(got, 25.0);
    }

    // ---- Test 2/3: L2 / L1 vs reference over several dims (0/1/2/3 remainders) ----

    fn gen_int_points_f64(dim: usize) -> (Vec<f64>, Vec<f64>) {
        let q: Vec<f64> = (0..dim).map(|i| (i as f64) * 7.0 + 3.0).collect();
        let p: Vec<f64> = (0..dim).map(|i| (i as f64) * 3.0 + 11.0).collect();
        (q, p)
    }

    fn gen_int_points_f32(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..dim).map(|i| (i as f32) * 7.0 + 3.0).collect();
        let p: Vec<f32> = (0..dim).map(|i| (i as f32) * 3.0 + 11.0).collect();
        (q, p)
    }

    const TEST_DIMS: &[usize] = &[3, 5, 6, 7, 8, 16, 32];

    #[test]
    fn l2_matches_reference_various_dims_f64() {
        for &dim in TEST_DIMS {
            let (q, p) = gen_int_points_f64(dim);
            let ds = FlatSlice::new(&p, dim);
            let got = L2.eval(&q, &ds, 0, DynDim(dim));
            let want: f64 = q.iter().zip(p.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            assert_eq!(got, want, "dim={dim}");
        }
    }

    #[test]
    fn l2_matches_reference_various_dims_f32() {
        for &dim in TEST_DIMS {
            let (q, p) = gen_int_points_f32(dim);
            let ds = FlatSlice::new(&p, dim);
            let got = L2.eval(&q, &ds, 0, DynDim(dim));
            let want: f32 = q.iter().zip(p.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            assert_eq!(got, want, "dim={dim}");
        }
    }

    #[test]
    fn l1_matches_reference_various_dims_f64() {
        for &dim in TEST_DIMS {
            let (q, p) = gen_int_points_f64(dim);
            let ds = FlatSlice::new(&p, dim);
            let got = L1.eval(&q, &ds, 0, DynDim(dim));
            let want: f64 = q.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert_eq!(got, want, "dim={dim}");
        }
    }

    #[test]
    fn l1_matches_reference_various_dims_f32() {
        for &dim in TEST_DIMS {
            let (q, p) = gen_int_points_f32(dim);
            let ds = FlatSlice::new(&p, dim);
            let got = L1.eval(&q, &ds, 0, DynDim(dim));
            let want: f32 = q.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert_eq!(got, want, "dim={dim}");
        }
    }

    // ---- Test 4: L2Simple == L2 ----

    #[test]
    fn l2_simple_matches_l2_exact_integer_dim8() {
        let (q, p) = gen_int_points_f64(8);
        let ds = FlatSlice::new(&p, 8);
        let l2 = L2.eval(&q, &ds, 0, ConstDim::<8>);
        let l2s = L2Simple.eval(&q, &ds, 0, ConstDim::<8>);
        assert_eq!(l2, l2s);
    }

    #[test]
    fn l2_simple_matches_l2_within_4ulp_irrational_dim8() {
        let dim = 8;
        let q: Vec<f64> = (0..dim).map(|i| 0.1 * (i as f64)).collect();
        let p: Vec<f64> = (0..dim).map(|i| 0.1 * ((i as f64) + 1.0)).collect();
        let ds = FlatSlice::new(&p, dim);
        let l2 = L2.eval(&q, &ds, 0, DynDim(dim));
        let l2s = L2Simple.eval(&q, &ds, 0, DynDim(dim));
        // Different summation order (4-way unroll vs plain sequential loop)
        // can produce a tiny floating-point discrepancy on non-exact inputs.
        assert!((l2 - l2s).abs() < 4.0 * f64::EPSILON, "l2={l2} l2s={l2s}");
    }

    // ---- Test 4b: L2Fma close to L2, uses mul_add ----

    #[test]
    fn l2_fma_matches_l2_exact_integer_dim8() {
        let (q, p) = gen_int_points_f64(8);
        let ds = FlatSlice::new(&p, 8);
        let l2 = L2.eval(&q, &ds, 0, ConstDim::<8>);
        let fma = L2Fma.eval(&q, &ds, 0, ConstDim::<8>);
        assert_eq!(l2, fma);
    }

    #[test]
    fn l2_fma_close_to_l2_irrational_dim8() {
        let dim = 8;
        let q: Vec<f64> = (0..dim).map(|i| 0.1 * (i as f64)).collect();
        let p: Vec<f64> = (0..dim).map(|i| 0.1 * ((i as f64) + 1.0)).collect();
        let ds = FlatSlice::new(&p, dim);
        let l2 = L2.eval(&q, &ds, 0, DynDim(dim));
        let fma = L2Fma.eval(&q, &ds, 0, DynDim(dim));
        assert!((l2 - fma).abs() < 4.0 * f64::EPSILON, "l2={l2} fma={fma}");
    }

    #[test]
    fn l2_fma_various_dims() {
        for &dim in TEST_DIMS {
            let (q, p) = gen_int_points_f64(dim);
            let ds = FlatSlice::new(&p, dim);
            let l2 = L2.eval(&q, &ds, 0, DynDim(dim));
            let fma = L2Fma.eval(&q, &ds, 0, DynDim(dim));
            assert!(
                (l2 - fma).abs() < 4.0 * f64::EPSILON * l2.max(1.0),
                "dim={dim} l2={l2} fma={fma}"
            );
        }
    }

    // ---- Test 5: SO2 last-dim-only, UNsquared ----

    #[test]
    fn so2_ignores_all_but_last_dim_and_is_unsquared() {
        let q = [10.0f64, 0.1];
        let p: &[[f64; 2]] = &[[-10.0, -0.1]];
        let got = SO2.eval(&q, &p, 0, ConstDim::<2>);
        // First component (10.0 vs -10.0) is IGNORED entirely.
        // Result is UNsquared: 0.2, not 0.04.
        assert!((got - 0.2).abs() < 1e-12, "got={got}");
    }

    // ---- Test 6: SO2 wrap-around ----

    #[test]
    fn so2_wraps_from_positive_diff() {
        // a=3.0, b=-3.0 -> raw diff = b - a = -6.0, wraps to -6.0 + 2*pi.
        let expected = (2.0 * core::f64::consts::PI - 6.0).abs();
        let got = SO2.accum_dist(3.0f64, -3.0, 1);
        assert!(
            (got - expected).abs() < 1e-12,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn so2_wraps_from_negative_diff() {
        // a=-3.0, b=3.0 -> raw diff = b - a = 6.0, wraps to 6.0 - 2*pi.
        let expected = (6.0 - 2.0 * core::f64::consts::PI).abs();
        let got = SO2.accum_dist(-3.0f64, 3.0, 1);
        assert!(
            (got - expected).abs() < 1e-12,
            "got={got} expected={expected}"
        );
    }

    #[test]
    fn so2_no_wrap_when_within_pi() {
        let got = SO2.accum_dist(0.0f64, 3.0, 1);
        assert_eq!(got, 3.0);
    }

    #[test]
    fn so2_result_always_in_0_pi_range() {
        let pi = core::f64::consts::PI;
        let n = 25;
        for i in 0..n {
            for j in 0..n {
                let a = -pi + 2.0 * pi * (i as f64) / ((n - 1) as f64);
                let b = -pi + 2.0 * pi * (j as f64) / ((n - 1) as f64);
                let got = SO2.accum_dist(a, b, 1);
                assert!(
                    got >= 0.0 && got <= pi + 1e-12,
                    "a={a} b={b} got={got} out of [0, pi]"
                );
            }
        }
    }

    // ---- Test 8: SO3 == L2Simple bit-for-bit ----

    #[test]
    fn so3_matches_l2_simple_bit_for_bit_dim4() {
        let q = [0.3f64, -1.7, 2.25, 0.001];
        let p: &[[f64; 4]] = &[[1.1, 0.05, -0.4, 3.333]];
        let so3 = SO3.eval(&q, &p, 0, ConstDim::<4>);
        let l2s = L2Simple.eval(&q, &p, 0, ConstDim::<4>);
        assert_eq!(so3.to_bits(), l2s.to_bits());

        let so3_acc = SO3.accum_dist(0.3f64, 1.1, 0);
        let l2s_acc = L2Simple.accum_dist(0.3f64, 1.1, 0);
        assert_eq!(so3_acc.to_bits(), l2s_acc.to_bits());
    }

    // ---- Test 9: accum_dist unit checks ----

    #[test]
    fn accum_dist_l1_is_abs_diff() {
        assert_eq!(L1.accum_dist(5.0f64, 2.0, 0), 3.0);
        assert_eq!(L1.accum_dist(2.0f64, 5.0, 0), 3.0);
    }

    #[test]
    fn accum_dist_l2_is_squared_diff() {
        assert_eq!(L2.accum_dist(5.0f64, 2.0, 0), 9.0);
    }

    #[test]
    fn accum_dist_l2_simple_is_squared_diff() {
        assert_eq!(L2Simple.accum_dist(5.0f64, 2.0, 0), 9.0);
    }

    #[test]
    fn accum_dist_so3_is_squared_diff() {
        assert_eq!(SO3.accum_dist(5.0f64, 2.0, 0), 9.0);
    }

    #[test]
    fn accum_dist_so2_wrapped_matches_wrap_test_values() {
        let expected = (2.0 * core::f64::consts::PI - 6.0).abs();
        assert!((SO2.accum_dist(3.0f64, -3.0, 1) - expected).abs() < 1e-12);
        assert!((SO2.accum_dist(-3.0f64, 3.0, 1) - expected).abs() < 1e-12);
    }

    // ---- Test 11: L2/L1::eval row path bit-equals the point_component
    // fallback path. This is the load-bearing invariant for the chunked
    // bounds-check-free row walk: the row-path source (e.g. `FlatSlice`)
    // takes the row path (it overrides `point_row`), while `FallbackOnly`
    // wraps the SAME source but forwards only `point_component`, never
    // overriding `point_row` (stays `None`), so it is forced through the
    // untouched per-component loop. Both must produce bit-identical results.
    //
    // Fix-round-1 note: a single fixed salt per test is NOT enough --
    // review proved the original single-salt f64 tests were blind to BOTH
    // sabotage mutations (ascending-remainder order, left-associative
    // chunk grouping) at every swept dim, because that one salt happened to
    // land on values where the reassociation didn't change the rounded bit
    // pattern. Every test below now sweeps EVERY dim x EVERY salt in
    // `SALTS_F32`/`SALTS_F64`, and both sabotages (reproduced in this
    // file's history / this round's RED capture, see the task-3 report
    // addendum) were independently confirmed to make EACH of the four
    // tests fail on its own against this exact multi-salt sweep. ----

    /// Wraps any `DataSource` but does NOT override `point_row` -- forces
    /// `L2::eval`/`L1::eval` through the per-component fallback path even
    /// when the wrapped source's own storage is contiguous.
    struct FallbackOnly<DS>(DS);

    impl<T: Scalar, DS: DataSource<T>> DataSource<T> for FallbackOnly<DS> {
        fn point_count(&self) -> usize {
            self.0.point_count()
        }
        fn point_component(&self, idx: usize, dim: usize) -> T {
            self.0.point_component(idx, dim)
        }
        // `point_row` intentionally NOT overridden -- default `None`.
    }

    const BIT_EQ_DIMS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 64];

    /// Six salts, deliberately varied in sign and magnitude (and clear of
    /// clippy's `approx_constant` list) so that across the full dim x salt
    /// sweep, every 4-wide remainder class (via the various dims) AND the
    /// main chunked-loop grouping (via dims >= 4) both get exercised with
    /// enough distinct rounding behavior to be genuinely discriminating.
    const SALTS_F64: &[f64] = &[0.4173, -1.9021, 3.5588, -4.2214, 5.7731, -6.6102];
    const SALTS_F32: &[f32] = &[0.4173, -1.9021, 3.5588, -4.2214, 5.7731, -6.6102];

    /// Deterministic, non-integer "random-ish" data -- irrational-ish
    /// multipliers so summation-order differences would actually show up as
    /// rounding differences (unlike small exact integers).
    fn gen_random_ish_f64(dim: usize, salt: f64) -> (Vec<f64>, Vec<f64>) {
        let q: Vec<f64> = (0..dim)
            .map(|i| ((i as f64) * 0.837421 + salt).sin() * 137.035999)
            .collect();
        let p: Vec<f64> = (0..dim)
            .map(|i| ((i as f64) * 1.928374 + salt * 1.5).cos() * 271.8281828 + 0.5)
            .collect();
        (q, p)
    }

    fn gen_random_ish_f32(dim: usize, salt: f32) -> (Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.837421 + salt).sin() * 137.036)
            .collect();
        let p: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 1.928374 + salt * 1.5).cos() * 271.828_2 + 0.5)
            .collect();
        (q, p)
    }

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_all_dims_f32() {
        for &dim in BIT_EQ_DIMS {
            for &salt in SALTS_F32 {
                let (q, p) = gen_random_ish_f32(dim, salt);
                let flat = FlatSlice::new(&p, dim);
                let fallback = FallbackOnly(flat);
                let row_result = L2.eval(&q, &flat, 0, DynDim(dim));
                let fallback_result = L2.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_all_dims_f64() {
        for &dim in BIT_EQ_DIMS {
            for &salt in SALTS_F64 {
                let (q, p) = gen_random_ish_f64(dim, salt);
                let flat = FlatSlice::new(&p, dim);
                let fallback = FallbackOnly(flat);
                let row_result = L2.eval(&q, &flat, 0, DynDim(dim));
                let fallback_result = L2.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l1_eval_row_path_bit_equals_fallback_path_all_dims_f32() {
        for &dim in BIT_EQ_DIMS {
            for &salt in SALTS_F32 {
                let (q, p) = gen_random_ish_f32(dim, salt);
                let flat = FlatSlice::new(&p, dim);
                let fallback = FallbackOnly(flat);
                let row_result = L1.eval(&q, &flat, 0, DynDim(dim));
                let fallback_result = L1.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l1_eval_row_path_bit_equals_fallback_path_all_dims_f64() {
        for &dim in BIT_EQ_DIMS {
            for &salt in SALTS_F64 {
                let (q, p) = gen_random_ish_f64(dim, salt);
                let flat = FlatSlice::new(&p, dim);
                let fallback = FallbackOnly(flat);
                let row_result = L1.eval(&q, &flat, 0, DynDim(dim));
                let fallback_result = L1.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    // ---- Test 11b: the ConstDim<3> + `&[[T;3]]` combination the fixed-3
    // perf gate actually runs -- separate from the DynDim/FlatSlice sweep
    // above, since `&[[T;N]]`'s `point_row` override is a distinct code
    // path from `FlatSlice`'s. ----

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f32() {
        for &salt in SALTS_F32 {
            let (q, p3) = gen_random_ish_f32(3, salt);
            let points: [[f32; 3]; 1] = [[p3[0], p3[1], p3[2]]];
            let arr: &[[f32; 3]] = &points;
            let fallback = FallbackOnly(arr);
            let row_result = L2.eval(&q, &arr, 0, ConstDim::<3>);
            let fallback_result = L2.eval(&q, &fallback, 0, ConstDim::<3>);
            assert_eq!(
                row_result.to_bits(),
                fallback_result.to_bits(),
                "salt={salt} row={row_result} fallback={fallback_result}"
            );
        }
    }

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f64() {
        for &salt in SALTS_F64 {
            let (q, p3) = gen_random_ish_f64(3, salt);
            let points: [[f64; 3]; 1] = [[p3[0], p3[1], p3[2]]];
            let arr: &[[f64; 3]] = &points;
            let fallback = FallbackOnly(arr);
            let row_result = L2.eval(&q, &arr, 0, ConstDim::<3>);
            let fallback_result = L2.eval(&q, &fallback, 0, ConstDim::<3>);
            assert_eq!(
                row_result.to_bits(),
                fallback_result.to_bits(),
                "salt={salt} row={row_result} fallback={fallback_result}"
            );
        }
    }

    // ---- Test 11d: same row-vs-fallback bit-equality invariant, but over
    // `OwnedRows` (M-py) instead of `FlatSlice` -- confirms the M2.5 row
    // kernel engages identically regardless of which row-major `DataSource`
    // backs it. Narrower dim sweep {3, 8, 32} than the full `BIT_EQ_DIMS`
    // list above (that list's job is already done by the `FlatSlice`
    // tests); this just needs to show `OwnedRows` itself takes the row
    // path bit-for-bit. ----

    const OWNED_ROWS_BIT_EQ_DIMS: &[usize] = &[3, 8, 32];

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_owned_rows_f32() {
        for &dim in OWNED_ROWS_BIT_EQ_DIMS {
            for &salt in SALTS_F32 {
                let (q, p) = gen_random_ish_f32(dim, salt);
                let owned = OwnedRows::new(p, dim);
                let fallback = FallbackOnly(&owned);
                let row_result = L2.eval(&q, &owned, 0, DynDim(dim));
                let fallback_result = L2.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l2_eval_row_path_bit_equals_fallback_path_owned_rows_f64() {
        for &dim in OWNED_ROWS_BIT_EQ_DIMS {
            for &salt in SALTS_F64 {
                let (q, p) = gen_random_ish_f64(dim, salt);
                let owned = OwnedRows::new(p, dim);
                let fallback = FallbackOnly(&owned);
                let row_result = L2.eval(&q, &owned, 0, DynDim(dim));
                let fallback_result = L2.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l1_eval_row_path_bit_equals_fallback_path_owned_rows_f32() {
        for &dim in OWNED_ROWS_BIT_EQ_DIMS {
            for &salt in SALTS_F32 {
                let (q, p) = gen_random_ish_f32(dim, salt);
                let owned = OwnedRows::new(p, dim);
                let fallback = FallbackOnly(&owned);
                let row_result = L1.eval(&q, &owned, 0, DynDim(dim));
                let fallback_result = L1.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    #[test]
    fn l1_eval_row_path_bit_equals_fallback_path_owned_rows_f64() {
        for &dim in OWNED_ROWS_BIT_EQ_DIMS {
            for &salt in SALTS_F64 {
                let (q, p) = gen_random_ish_f64(dim, salt);
                let owned = OwnedRows::new(p, dim);
                let fallback = FallbackOnly(&owned);
                let row_result = L1.eval(&q, &owned, 0, DynDim(dim));
                let fallback_result = L1.eval(&q, &fallback, 0, DynDim(dim));
                assert_eq!(
                    row_result.to_bits(),
                    fallback_result.to_bits(),
                    "dim={dim} salt={salt} row={row_result} fallback={fallback_result}"
                );
            }
        }
    }

    // ---- Test 11c: `-0.0` locks the ternary-vs-`.abs()` distinction
    // documented on `abs_ternary!` -- `-x` where `x == -0.0` must return
    // `-0.0` unchanged (the ternary's `else` branch, since `-0.0 < 0.0` is
    // `false`), NOT `+0.0` (which `.abs()` would give). ----

    #[test]
    fn l1_accum_dist_negative_zero_diff_stays_negative_zero() {
        // a - b = -0.0 exactly when a == b == 0.0 (both signs of zero
        // compare equal, and 0.0 - 0.0 rounds to +0.0 in IEEE754 -- so to
        // land on a diff of EXACTLY -0.0 we supply it as the difference
        // via `a = -0.0, b = 0.0`: -0.0 - 0.0 = -0.0).
        let got = L1.accum_dist(-0.0f64, 0.0, 0);
        assert!(
            got.is_sign_negative(),
            "abs_ternary! must leave -0.0 unchanged, got {got} (bits {:#x})",
            got.to_bits()
        );
        assert_eq!(got.to_bits(), (-0.0f64).to_bits());
    }

    // ---- Test 10: stateful custom metric compiles against the trait ----

    #[test]
    fn custom_stateful_metric_compiles_and_computes() {
        struct WeightedL2 {
            weights: Vec<f64>,
        }

        impl Distance<f64> for WeightedL2 {
            type DistanceType = f64;

            // `d` indexes three independent sources at once (`query`, the
            // `DataSource` trait method, and `self.weights`) — no single
            // iterator adaptor covers all three cleanly.
            #[allow(clippy::needless_range_loop)]
            fn eval<DS: DataSource<f64> + ?Sized, D: Dim>(
                &self,
                query: &[f64],
                ds: &DS,
                idx: usize,
                dim: D,
            ) -> f64 {
                let dim = dim.dim();
                let mut result = 0.0;
                for d in 0..dim {
                    let diff = query[d] - ds.point_component(idx, d);
                    result += self.weights[d] * diff * diff;
                }
                result
            }

            fn accum_dist(&self, a: f64, b: f64, axis: usize) -> f64 {
                let diff = a - b;
                self.weights[axis] * diff * diff
            }
        }

        let metric = WeightedL2 {
            weights: vec![1.0, 4.0, 0.5],
        };
        let q = [0.0f64, 0.0, 0.0];
        let p: &[[f64; 3]] = &[[1.0, 1.0, 1.0]];
        let got = metric.eval(&q, &p, 0, ConstDim::<3>);
        // 1*1^2 + 4*1^2 + 0.5*1^2 = 1 + 4 + 0.5 = 5.5
        assert_eq!(got, 5.5);

        let acc = metric.accum_dist(0.0, 2.0, 1);
        // weight[1] * (0-2)^2 = 4 * 4 = 16
        assert_eq!(acc, 16.0);
    }
}
