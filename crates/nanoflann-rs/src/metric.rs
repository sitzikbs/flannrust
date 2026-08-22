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
    type DistanceType: DistanceValue;

    /// `d` carries the dimensionality — pass a [`crate::ConstDim`] where the
    /// caller's dimension is a compile-time constant (e.g. the hot search
    /// path over a `ConstDim<N>`-built tree) so the monomorphized body sees
    /// `d.dim()` constant-fold to `N`, letting the optimizer fully unroll
    /// fixed-width kernels (like `L2`'s 4-wide unroll) and elide bounds
    /// checks; pass [`crate::DynDim`] for a runtime dimension. **API
    /// CHANGE**: this parameter used to be a plain `dim: usize` — existing
    /// external `Distance` implementations must update their `eval` to take
    /// `d: D` (`D: Dim`) and call `d.dim()` wherever the old `dim` was used.
    fn eval<DS: DataSource<T> + ?Sized, D: Dim>(
        &self,
        query: &[T],
        ds: &DS,
        idx: usize,
        d: D,
    ) -> Self::DistanceType;

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
impl_so2!(f32, core::f32::consts::PI);
impl_so2!(f64, core::f64::consts::PI);
impl_so3!(f32);
impl_so3!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_source::FlatSlice;
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
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
    }

    #[test]
    fn so2_wraps_from_negative_diff() {
        // a=-3.0, b=3.0 -> raw diff = b - a = 6.0, wraps to 6.0 - 2*pi.
        let expected = (6.0 - 2.0 * core::f64::consts::PI).abs();
        let got = SO2.accum_dist(-3.0f64, 3.0, 1);
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
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

    // ---- Test 10: stateful custom metric compiles against the trait ----

    #[test]
    fn custom_stateful_metric_compiles_and_computes() {
        struct WeightedL2 {
            weights: Vec<f64>,
        }

        impl Distance<f64> for WeightedL2 {
            type DistanceType = f64;

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
