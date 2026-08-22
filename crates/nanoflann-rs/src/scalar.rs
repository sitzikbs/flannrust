use core::ops::{Add, Div, Mul, Sub};

/// Element type of dataset coordinates. Implemented for `f32` and `f64` in M1.
/// `Default` doubles as the zero value (mirrors nanoflann's `ElementType()` zero-init).
///
/// `Mul`/`Div` and `from_f64` are needed starting with the tree builder
/// (`build.rs`, task 6): nanoflann's `middleSplit_` computes `(1 - EPS) *
/// max_span` and `(low + high) / 2` directly in `ElementType`, and its
/// constants (`EPS = 0.00001`, the halving divisor, the `-1` spread seed)
/// need a generic way to materialize as `T`. `from_f64` mirrors
/// `DistanceValue::from_f32` (widen-by-cast) but narrows for `f32`, exactly
/// like C++'s `static_cast<ElementType>(0.00001)` from a `double` literal.
pub trait Scalar:
    Copy
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Cast a `f64` literal to `Self` (`as f32` / identity for `f64`).
    fn from_f64(v: f64) -> Self;
}

impl Scalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl Scalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Accumulated distance type (nanoflann's `DistanceType`).
pub trait DistanceValue:
    Copy + PartialOrd + Default + Add<Output = Self> + Sub<Output = Self> + core::ops::Mul<Output = Self> + Send + Sync + 'static
{
    /// Sentinel "worst possible" distance (= `std::numeric_limits<T>::max()`).
    const MAX: Self;
    const ZERO: Self;
    /// Widen a user-supplied `eps` (nanoflann's `SearchParameters::eps` is `float`).
    fn from_f32(v: f32) -> Self;
}

impl DistanceValue for f32 {
    const MAX: Self = f32::MAX;
    const ZERO: Self = 0.0;
    fn from_f32(v: f32) -> Self { v }
}

impl DistanceValue for f64 {
    const MAX: Self = f64::MAX;
    const ZERO: Self = 0.0;
    fn from_f32(v: f32) -> Self { v as f64 }
}

/// Public point-index type (promoted from `build.rs`'s crate-private
/// `IndexAccess`, task 6 — same contract, same method names, now part of the
/// public API so callers can pick `u32`/`u64`/`usize` via
/// `KdTreeBuilder::index_type`).
pub trait IndexType: Copy + PartialOrd + Send + Sync + 'static {
    fn to_usize(self) -> usize;
    /// Panics (via `try_into().unwrap()`) if `v` doesn't fit in `Self`.
    fn from_usize(v: usize) -> Self;
}

impl IndexType for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_usize(v: usize) -> Self {
        v.try_into().unwrap()
    }
}

impl IndexType for u64 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_usize(v: usize) -> Self {
        v.try_into().unwrap()
    }
}

impl IndexType for usize {
    #[inline]
    fn to_usize(self) -> usize {
        self
    }
    #[inline]
    fn from_usize(v: usize) -> Self {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_distance_value_constants() {
        assert_eq!(<f32 as DistanceValue>::MAX, f32::MAX);
        assert_eq!(<f32 as DistanceValue>::ZERO, 0.0f32);
    }

    #[test]
    fn test_f64_distance_value_constants() {
        assert_eq!(<f64 as DistanceValue>::MAX, f64::MAX);
        assert_eq!(<f64 as DistanceValue>::ZERO, 0.0f64);
    }

    #[test]
    fn test_from_f32_f32() {
        let result = <f32 as DistanceValue>::from_f32(1.5f32);
        assert_eq!(result, 1.5f32);
    }

    #[test]
    fn test_from_f32_f64() {
        let result = <f64 as DistanceValue>::from_f32(1.5f32);
        assert_eq!(result, 1.5f64);
    }

    #[test]
    fn test_scalar_default_f32_is_zero() {
        let zero: f32 = f32::default();
        assert_eq!(zero, 0.0f32);
    }

    #[test]
    fn test_scalar_default_f64_is_zero() {
        let zero: f64 = f64::default();
        assert_eq!(zero, 0.0f64);
    }

    #[test]
    fn test_index_type_round_trip_u32_u64_usize() {
        assert_eq!(<u32 as IndexType>::from_usize(7).to_usize(), 7);
        assert_eq!(<u64 as IndexType>::from_usize(7).to_usize(), 7);
        assert_eq!(<usize as IndexType>::from_usize(7).to_usize(), 7);
    }

    #[test]
    #[should_panic]
    fn test_index_type_u32_from_usize_panics_on_overflow() {
        let _ = <u32 as IndexType>::from_usize(u32::MAX as usize + 1);
    }
}
