use crate::scalar::Scalar;
use crate::data_source::DataSource;

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

/// Fill `bbox` (length = dim) with the dataset's bounds: the dataset's own
/// `fill_bbox` if it provides one, else a full scan. Same comparison order
/// as C++ (`< low` then `> high`, two independent ifs).
pub(crate) fn compute_bounding_box<T: Scalar, DS: DataSource<T> + ?Sized>(
    ds: &DS,
    dim: usize,
    bbox: &mut [Interval<T>],
) {
    debug_assert_eq!(bbox.len(), dim);
    if ds.fill_bbox(bbox) {
        return;
    }
    let n = ds.point_count();
    debug_assert!(n > 0, "compute_bounding_box on empty dataset");
    for i in 0..dim {
        let v = ds.point_component(0, i);
        bbox[i] = Interval { low: v, high: v };
    }
    for k in 1..n {
        for i in 0..dim {
            let v = ds.point_component(k, i);
            if v < bbox[i].low {
                bbox[i].low = v;
            }
            if v > bbox[i].high {
                bbox[i].high = v;
            }
        }
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

    #[test]
    fn test_compute_bounding_box_scan() {
        let points: &[[f32; 2]] = &[[1.0, 5.0], [-2.0, 7.0], [0.5, 6.0]];
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 2];
        compute_bounding_box(&points, 2, &mut bbox);
        assert_eq!(bbox[0].low, -2.0);
        assert_eq!(bbox[0].high, 1.0);
        assert_eq!(bbox[1].low, 5.0);
        assert_eq!(bbox[1].high, 7.0);
    }

    #[test]
    fn test_compute_bounding_box_single_point() {
        let points: &[[f64; 3]] = &[[2.5, 3.5, 4.5]];
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 3];
        compute_bounding_box(&points, 3, &mut bbox);
        assert_eq!(bbox[0].low, 2.5);
        assert_eq!(bbox[0].high, 2.5);
        assert_eq!(bbox[1].low, 3.5);
        assert_eq!(bbox[1].high, 3.5);
        assert_eq!(bbox[2].low, 4.5);
        assert_eq!(bbox[2].high, 4.5);
    }

    #[test]
    fn test_compute_bounding_box_fill_bbox_short_circuit() {
        struct CustomDataSource;

        impl DataSource<f32> for CustomDataSource {
            fn point_count(&self) -> usize {
                2
            }

            fn point_component(&self, _idx: usize, _dim: usize) -> f32 {
                15.0 // This is between 10 and 20
            }

            fn fill_bbox(&self, bbox: &mut [Interval<f32>]) -> bool {
                bbox[0] = Interval { low: 0.0, high: 100.0 };
                true
            }
        }

        let ds = CustomDataSource;
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 1];
        compute_bounding_box(&ds, 1, &mut bbox);
        // Should use fill_bbox result, not scan
        assert_eq!(bbox[0].low, 0.0);
        assert_eq!(bbox[0].high, 100.0);
    }
}
