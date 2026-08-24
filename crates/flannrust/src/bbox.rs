//! Per-dimension `[low, high]` bounding intervals (`Interval`) and the
//! dataset-bounding-box scan (`compute_bounding_box`) used to seed the
//! tree builder's root box.

use crate::data_source::DataSource;
use crate::scalar::{IndexType, Scalar};

/// One per-dimension `[low, high]` interval (nanoflann's `KDTreeBaseClass::Interval`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Interval<T> {
    /// Lower bound (inclusive).
    pub low: T,
    /// Upper bound (inclusive).
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
// `i` indexes BOTH `bbox` and `ds.point_component(_, i)` (a trait method,
// not a slice) in lockstep with the C++ source's per-axis loop order --
// kept as an explicit range loop (not an iterator adaptor over `bbox`
// alone) for that C++-order fidelity.
#[allow(clippy::needless_range_loop)]
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

/// Same scan as [`compute_bounding_box`], but restricted to an explicit
/// index subset `ind` (dataset indices, not `0..n`) instead of the whole
/// dataset. Used by the dynamic forest's (`dynamic.rs`) per-slot rebuild,
/// which mirrors the C++ `KDTreeBaseClass::computeBoundingBox`
/// (nanoflann.hpp:1215-1233) — the SAME function the static build uses,
/// but there it scans `vAcc_[0..size_]`, not `0..point_count()` directly;
/// the two coincide for the static tree only because its `vAcc_` is always
/// the identity permutation at build time. A dynamic forest slot's `vind`
/// (= that slot's `vAcc_`) is an arbitrary permuted SUBSET of dataset
/// indices after a merge, so this restricted scan is required for a
/// bit-exact port: `ds.fill_bbox` is still tried first (same "quirk" as the
/// static path — if a `DataSource` overrides it, the DATASET-WIDE bbox it
/// returns is used verbatim for the slot too, exactly like the C++, even
/// though it may be far looser than the slot's own points; this only
/// affects candidate-dimension selection in `middle_split`, never
/// correctness, since leaf bboxes are always tightened from real
/// coordinates).
#[allow(clippy::needless_range_loop)]
pub(crate) fn compute_bounding_box_over_indices<T, DS, Idx>(
    ds: &DS,
    dim: usize,
    ind: &[Idx],
    bbox: &mut [Interval<T>],
) where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    debug_assert_eq!(bbox.len(), dim);
    if ds.fill_bbox(bbox) {
        return;
    }
    debug_assert!(!ind.is_empty(), "compute_bounding_box_over_indices on empty ind");
    for i in 0..dim {
        let v = ds.point_component(ind[0].to_usize(), i);
        bbox[i] = Interval { low: v, high: v };
    }
    for k in 1..ind.len() {
        for i in 0..dim {
            let v = ds.point_component(ind[k].to_usize(), i);
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

    #[test]
    fn test_compute_bounding_box_over_indices_restricted_subset() {
        // Full dataset spans much wider than the {1, 3} subset used below —
        // proves the scan is restricted to `ind`, not the whole dataset
        // (unlike `compute_bounding_box`, which always scans everything).
        let points: &[[f64; 2]] = &[[100.0, 100.0], [1.0, 5.0], [-100.0, -100.0], [3.0, 1.0]];
        let ind: Vec<u32> = vec![1, 3];
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 2];
        compute_bounding_box_over_indices(&points, 2, &ind, &mut bbox);
        assert_eq!(bbox[0], Interval { low: 1.0, high: 3.0 });
        assert_eq!(bbox[1], Interval { low: 1.0, high: 5.0 });
    }
}
