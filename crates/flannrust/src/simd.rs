//! SIMD-accelerated L2 distance kernels for x86_64.
//!
//! Two families:
//! - **L2 (bit-exact):** AVX2 kernels using `mul + add` with per-chunk
//!   horizontal reduction matching the scalar `(d0²+d1²)+(d2²+d3²)` order.
//! - **L2Fma:** AVX2+FMA and AVX-512F kernels using fused multiply-add.
//!   Results are within FMA tolerance of the scalar `mul_add` loop.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Bit-exact L2 SIMD dispatch. Per-chunk horizontal reduction preserves
/// the scalar `l2_eval_row` accumulation order.
pub(crate) trait L2Simd: Sized {
    fn dispatch(q: &[Self], r: &[Self], dim: usize) -> Option<Self>;
}

#[cfg(target_arch = "x86_64")]
impl L2Simd for f64 {
    #[inline]
    fn dispatch(q: &[f64], r: &[f64], dim: usize) -> Option<f64> {
        if dim < 8 {
            return None;
        }
        if cfg!(target_feature = "avx2") {
            return Some(unsafe { l2_f64_avx2(q, r, dim) });
        }
        if is_x86_feature_detected!("avx2") {
            return Some(unsafe { l2_f64_avx2(q, r, dim) });
        }
        None
    }
}

#[cfg(target_arch = "x86_64")]
impl L2Simd for f32 {
    #[inline]
    fn dispatch(q: &[f32], r: &[f32], dim: usize) -> Option<f32> {
        if dim < 8 {
            return None;
        }
        if cfg!(target_feature = "avx2") {
            return Some(unsafe { l2_f32_avx2(q, r, dim) });
        }
        if is_x86_feature_detected!("avx2") {
            return Some(unsafe { l2_f32_avx2(q, r, dim) });
        }
        None
    }
}

/// Trait for type-generic SIMD dispatch, so the `impl_l2_fma!` macro can
/// call the right kernel for both f32 and f64 without `paste!`.
pub(crate) trait L2FmaSimd: Sized {
    fn dispatch(q: &[Self], r: &[Self], dim: usize) -> Option<Self>;
}

#[cfg(target_arch = "x86_64")]
impl L2FmaSimd for f64 {
    #[inline]
    fn dispatch(q: &[f64], r: &[f64], dim: usize) -> Option<f64> {
        if dim < 8 {
            return None;
        }
        // Compile-time path (native builds -- branch folds away).
        if cfg!(target_feature = "avx512f") && dim >= 32 {
            return Some(unsafe { l2fma_f64_avx512(q, r, dim) });
        }
        if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
            return Some(unsafe { l2fma_f64_avx2(q, r, dim) });
        }
        // Runtime path (generic builds -- std caches cpuid).
        if is_x86_feature_detected!("avx512f") && dim >= 32 {
            return Some(unsafe { l2fma_f64_avx512(q, r, dim) });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Some(unsafe { l2fma_f64_avx2(q, r, dim) });
        }
        None
    }
}

#[cfg(target_arch = "x86_64")]
impl L2FmaSimd for f32 {
    #[inline]
    fn dispatch(q: &[f32], r: &[f32], dim: usize) -> Option<f32> {
        if dim < 8 {
            return None;
        }
        if cfg!(target_feature = "avx512f") && dim >= 32 {
            return Some(unsafe { l2fma_f32_avx512(q, r, dim) });
        }
        if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
            return Some(unsafe { l2fma_f32_avx2(q, r, dim) });
        }
        if is_x86_feature_detected!("avx512f") && dim >= 32 {
            return Some(unsafe { l2fma_f32_avx512(q, r, dim) });
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Some(unsafe { l2fma_f32_avx2(q, r, dim) });
        }
        None
    }
}

// ---- AVX2+FMA f64 kernel ----

/// Squared L2 via FMA over 4-wide f64 lanes, 2 accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2fma_f64_avx2(q: &[f64], r: &[f64], dim: usize) -> f64 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut i = 0usize;
    // Main loop: 8 f64s per iteration (2x4-wide)
    while i + 8 <= dim {
        let d0 = _mm256_sub_pd(_mm256_loadu_pd(qp.add(i)), _mm256_loadu_pd(rp.add(i)));
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
        let d1 = _mm256_sub_pd(
            _mm256_loadu_pd(qp.add(i + 4)),
            _mm256_loadu_pd(rp.add(i + 4)),
        );
        acc1 = _mm256_fmadd_pd(d1, d1, acc1);
        i += 8;
    }
    // 4-wide remainder
    if i + 4 <= dim {
        let d0 = _mm256_sub_pd(_mm256_loadu_pd(qp.add(i)), _mm256_loadu_pd(rp.add(i)));
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
        i += 4;
    }
    // Combine accumulators and horizontal sum
    acc0 = _mm256_add_pd(acc0, acc1);
    let hi128 = _mm256_extractf128_pd(acc0, 1);
    let lo128 = _mm256_castpd256_pd128(acc0);
    let sum128 = _mm_add_pd(lo128, hi128);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    let mut result = _mm_cvtsd_f64(_mm_add_sd(sum128, hi64));
    // Scalar tail (0-3 elements)
    while i < dim {
        let d = *q.get_unchecked(i) - *r.get_unchecked(i);
        result = d.mul_add(d, result);
        i += 1;
    }
    result
}

// ---- AVX2+FMA f32 kernel ----

/// Squared L2 via FMA over 8-wide f32 lanes, 2 accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2fma_f32_avx2(q: &[f32], r: &[f32], dim: usize) -> f32 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    // Main loop: 16 f32s per iteration (2x8-wide)
    while i + 16 <= dim {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(rp.add(i)));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(qp.add(i + 8)),
            _mm256_loadu_ps(rp.add(i + 8)),
        );
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        i += 16;
    }
    // 8-wide remainder
    if i + 8 <= dim {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(rp.add(i)));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        i += 8;
    }
    // Combine accumulators and horizontal sum
    acc0 = _mm256_add_ps(acc0, acc1);
    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf1 = _mm_movehdup_ps(sum128);
    let sum1 = _mm_add_ps(sum128, shuf1);
    let shuf2 = _mm_movehl_ps(sum1, sum1);
    let mut result = _mm_cvtss_f32(_mm_add_ss(sum1, shuf2));
    // Scalar tail (0-7 elements)
    while i < dim {
        let d = *q.get_unchecked(i) - *r.get_unchecked(i);
        result = d.mul_add(d, result);
        i += 1;
    }
    result
}

// ---- AVX-512F f64 kernel ----

/// Squared L2 via FMA over 8-wide f64 lanes, 2 accumulators, masked tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2fma_f64_avx512(q: &[f64], r: &[f64], dim: usize) -> f64 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();
    let mut i = 0usize;
    // Main loop: 16 f64s per iteration (2x8-wide)
    while i + 16 <= dim {
        let d0 = _mm512_sub_pd(_mm512_loadu_pd(qp.add(i)), _mm512_loadu_pd(rp.add(i)));
        acc0 = _mm512_fmadd_pd(d0, d0, acc0);
        let d1 = _mm512_sub_pd(
            _mm512_loadu_pd(qp.add(i + 8)),
            _mm512_loadu_pd(rp.add(i + 8)),
        );
        acc1 = _mm512_fmadd_pd(d1, d1, acc1);
        i += 16;
    }
    // 8-wide remainder
    if i + 8 <= dim {
        let d0 = _mm512_sub_pd(_mm512_loadu_pd(qp.add(i)), _mm512_loadu_pd(rp.add(i)));
        acc0 = _mm512_fmadd_pd(d0, d0, acc0);
        i += 8;
    }
    // Masked tail (0-7 remaining f64s)
    let tail = dim - i;
    if tail > 0 {
        let mask: __mmask8 = (1u8 << tail) - 1;
        let qv = _mm512_maskz_loadu_pd(mask, qp.add(i));
        let rv = _mm512_maskz_loadu_pd(mask, rp.add(i));
        let d = _mm512_sub_pd(qv, rv);
        acc1 = _mm512_fmadd_pd(d, d, acc1);
    }
    // Combine and horizontal sum: 512 -> 256 -> 128 -> scalar
    acc0 = _mm512_add_pd(acc0, acc1);
    let lo256 = _mm512_castpd512_pd256(acc0);
    let hi256 = _mm512_extractf64x4_pd(acc0, 1);
    let sum256 = _mm256_add_pd(lo256, hi256);
    let hi128 = _mm256_extractf128_pd(sum256, 1);
    let lo128 = _mm256_castpd256_pd128(sum256);
    let sum128 = _mm_add_pd(lo128, hi128);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
}

// ---- AVX-512F f32 kernel ----

/// Squared L2 via FMA over 16-wide f32 lanes, 2 accumulators, masked tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2fma_f32_avx512(q: &[f32], r: &[f32], dim: usize) -> f32 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0usize;
    // Main loop: 32 f32s per iteration (2x16-wide)
    while i + 32 <= dim {
        let d0 = _mm512_sub_ps(_mm512_loadu_ps(qp.add(i)), _mm512_loadu_ps(rp.add(i)));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        let d1 = _mm512_sub_ps(
            _mm512_loadu_ps(qp.add(i + 16)),
            _mm512_loadu_ps(rp.add(i + 16)),
        );
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        i += 32;
    }
    // 16-wide remainder
    if i + 16 <= dim {
        let d0 = _mm512_sub_ps(_mm512_loadu_ps(qp.add(i)), _mm512_loadu_ps(rp.add(i)));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        i += 16;
    }
    // Masked tail (0-15 remaining f32s)
    let tail = dim - i;
    if tail > 0 {
        let mask: __mmask16 = (1u16 << tail) - 1;
        let qv = _mm512_maskz_loadu_ps(mask, qp.add(i));
        let rv = _mm512_maskz_loadu_ps(mask, rp.add(i));
        let d = _mm512_sub_ps(qv, rv);
        acc1 = _mm512_fmadd_ps(d, d, acc1);
    }
    // Combine and horizontal sum: 512 -> 256 -> 128 -> scalar
    // (avx512f has no _mm512_extractf32x8_ps -- that's avx512dq --
    //  so bitcast through f64 to use extractf64x4.)
    acc0 = _mm512_add_ps(acc0, acc1);
    let pd = _mm512_castps_pd(acc0);
    let hi256 = _mm256_castpd_ps(_mm512_extractf64x4_pd(pd, 1));
    let lo256 = _mm512_castps512_ps256(acc0);
    let sum256 = _mm256_add_ps(lo256, hi256);
    let hi128 = _mm256_extractf128_ps(sum256, 1);
    let lo128 = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf1 = _mm_movehdup_ps(sum128);
    let sum1 = _mm_add_ps(sum128, shuf1);
    let shuf2 = _mm_movehl_ps(sum1, sum1);
    _mm_cvtss_f32(_mm_add_ss(sum1, shuf2))
}

// ---- Bit-exact L2 AVX2 kernels ----
//
// Per-chunk horizontal reduction: each 4-wide vector is reduced to a scalar
// matching `(d0²+d1²)+(d2²+d3²)` before accumulating into `result`. This
// preserves the exact FP order of `l2_eval_row`'s scalar path.

/// Bit-exact squared L2 via AVX2 over 4-wide f64. Per-chunk horizontal sum
/// matches the scalar `(d0²+d1²)+(d2²+d3²)` parenthesization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_f64_avx2(q: &[f64], r: &[f64], dim: usize) -> f64 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut result = 0.0f64;
    let mut i = 0usize;
    while i + 4 <= dim {
        let d = _mm256_sub_pd(_mm256_loadu_pd(qp.add(i)), _mm256_loadu_pd(rp.add(i)));
        let d2 = _mm256_mul_pd(d, d);
        // (d0²+d1²)+(d2²+d3²): swap adjacent pairs, add, then cross-lane add
        let shuf = _mm256_permute_pd(d2, 0b0101);
        let pair_sums = _mm256_add_pd(d2, shuf);
        let hi128 = _mm256_extractf128_pd(pair_sums, 1);
        let lo128 = _mm256_castpd256_pd128(pair_sums);
        result = result + _mm_cvtsd_f64(_mm_add_sd(lo128, hi128));
        i += 4;
    }
    // Scalar tail: 0-3 elements in descending order (matches l2_eval_row)
    let rem = dim - i;
    if rem >= 3 {
        let d = *q.get_unchecked(i + 2) - *r.get_unchecked(i + 2);
        result = result + d * d;
    }
    if rem >= 2 {
        let d = *q.get_unchecked(i + 1) - *r.get_unchecked(i + 1);
        result = result + d * d;
    }
    if rem >= 1 {
        let d = *q.get_unchecked(i) - *r.get_unchecked(i);
        result = result + d * d;
    }
    result
}

/// Bit-exact squared L2 via AVX2 over 8-wide f32. Each vector covers two
/// scalar chunks; each half is reduced independently to preserve order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_f32_avx2(q: &[f32], r: &[f32], dim: usize) -> f32 {
    let qp = q.as_ptr();
    let rp = r.as_ptr();
    let mut result = 0.0f32;
    let mut i = 0usize;
    // Main loop: 8 f32s = 2 scalar chunks per iteration
    while i + 8 <= dim {
        let d = _mm256_sub_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(rp.add(i)));
        let d2 = _mm256_mul_ps(d, d);
        let lo = _mm256_castps256_ps128(d2);
        let hi = _mm256_extractf128_ps(d2, 1);
        // Reduce lo 4 lanes: (d0²+d1²)+(d2²+d3²)
        let lo_shuf = _mm_movehdup_ps(lo);
        let lo_pairs = _mm_add_ps(lo, lo_shuf);
        let lo_high = _mm_movehl_ps(lo_pairs, lo_pairs);
        result = result + _mm_cvtss_f32(_mm_add_ss(lo_pairs, lo_high));
        // Reduce hi 4 lanes: (d4²+d5²)+(d6²+d7²)
        let hi_shuf = _mm_movehdup_ps(hi);
        let hi_pairs = _mm_add_ps(hi, hi_shuf);
        let hi_high = _mm_movehl_ps(hi_pairs, hi_pairs);
        result = result + _mm_cvtss_f32(_mm_add_ss(hi_pairs, hi_high));
        i += 8;
    }
    // 4-wide remainder (one scalar chunk)
    if i + 4 <= dim {
        let d = _mm_sub_ps(_mm_loadu_ps(qp.add(i)), _mm_loadu_ps(rp.add(i)));
        let d2 = _mm_mul_ps(d, d);
        let shuf = _mm_movehdup_ps(d2);
        let pairs = _mm_add_ps(d2, shuf);
        let high = _mm_movehl_ps(pairs, pairs);
        result = result + _mm_cvtss_f32(_mm_add_ss(pairs, high));
        i += 4;
    }
    // Scalar tail: 0-3 elements in descending order
    let rem = dim - i;
    if rem >= 3 {
        let d = *q.get_unchecked(i + 2) - *r.get_unchecked(i + 2);
        result = result + d * d;
    }
    if rem >= 2 {
        let d = *q.get_unchecked(i + 1) - *r.get_unchecked(i + 1);
        result = result + d * d;
    }
    if rem >= 1 {
        let d = *q.get_unchecked(i) - *r.get_unchecked(i);
        result = result + d * d;
    }
    result
}
