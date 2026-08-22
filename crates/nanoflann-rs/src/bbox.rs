use crate::scalar::Scalar;

/// One per-dimension `[low, high]` interval (nanoflann's `KDTreeBaseClass::Interval`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Interval<T> {
    pub low: T,
    pub high: T,
}

impl<T: Scalar> Interval<T> {
    /// Inclusive containment on BOTH faces — nanoflann's box-search convention
    /// (`point < low || point > high` → reject; nanoflann.hpp:2182-2192).
    #[inline]
    pub fn contains(&self, v: T) -> bool {
        !(v < self.low || v > self.high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_contains_below_low() {
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(!interval.contains(0.9));
    }

    #[test]
    fn test_interval_contains_at_low_boundary() {
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(interval.contains(1.0));
    }

    #[test]
    fn test_interval_contains_inside() {
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(interval.contains(1.5));
    }

    #[test]
    fn test_interval_contains_at_high_boundary() {
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(interval.contains(2.0));
    }

    #[test]
    fn test_interval_contains_above_high() {
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(!interval.contains(2.1));
    }

    #[test]
    fn test_interval_contains_nan() {
        // NaN behavior: in C++, both `(point < low)` and `(point > high)` are false
        // when point is NaN (IEEE 754 semantics), so the double negation
        // `!(nan < low || nan > high)` evaluates to `!(false || false)` = true.
        // This documents parity with nanoflann's NaN handling.
        let interval = Interval { low: 1.0, high: 2.0 };
        assert!(interval.contains(f64::NAN));
    }
}
