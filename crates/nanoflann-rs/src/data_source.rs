//! Zero-copy dataset access: the `DataSource` trait callers implement to
//! hand their point cloud to a `KdTreeBuilder`, plus two built-in
//! implementations (`&[[T; N]]` and the row-major `FlatSlice`).

use crate::bbox::Interval;
use crate::scalar::Scalar;

/// Zero-copy dataset access (nanoflann's DatasetAdaptor duck-type contract:
/// `kdtree_get_point_count` / `kdtree_get_pt` / `kdtree_get_bbox`).
pub trait DataSource<T: Scalar> {
    /// Number of points in the dataset.
    fn point_count(&self) -> usize;

    /// Component `dim` of point `idx`. `idx < point_count()`, `dim < dimensionality`.
    fn point_component(&self, idx: usize, dim: usize) -> T;

    /// Optionally fill a precomputed bounding box (= `kdtree_get_bbox`).
    /// Return `false` (the default) to have the tree compute it by scanning.
    fn fill_bbox(&self, _bbox: &mut [Interval<T>]) -> bool {
        false
    }

    /// Contiguous row access fast path. Implementations whose points are
    /// stored contiguously SHOULD override this and return the full row;
    /// the default `None` keeps every existing implementation
    /// source-compatible and routes metrics through [`point_component`]
    /// unchanged. Contract: when `Some(row)` is returned, `row.len() >=
    /// dim` and `row[d] == self.point_component(idx, d)` for all `d <
    /// dim`, where `dim` is whatever dimensionality the caller is
    /// currently evaluating with (the row is not required to be exactly
    /// `dim` long -- callers that need fewer than the full row's
    /// components just read a prefix).
    ///
    /// [`point_component`]: DataSource::point_component
    #[inline]
    fn point_row(&self, _idx: usize) -> Option<&[T]> {
        None
    }
}

impl<T: Scalar, const N: usize> DataSource<T> for &[[T; N]] {
    #[inline]
    fn point_count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn point_component(&self, idx: usize, dim: usize) -> T {
        self[idx][dim]
    }

    #[inline]
    fn point_row(&self, idx: usize) -> Option<&[T]> {
        Some(&self[idx][..])
    }
}

/// Row-major flat buffer: point `i` occupies `data[i*dim .. (i+1)*dim]`.
#[derive(Debug, Clone, Copy)]
pub struct FlatSlice<'a, T> {
    data: &'a [T],
    dim: usize,
}

impl<'a, T: Scalar> FlatSlice<'a, T> {
    /// Panics if `dim == 0` or `data.len()` is not a multiple of `dim`.
    pub fn new(data: &'a [T], dim: usize) -> Self {
        assert!(dim > 0, "FlatSlice dimension must be > 0");
        assert!(
            data.len().is_multiple_of(dim),
            "FlatSlice data length {} is not a multiple of dim {}",
            data.len(),
            dim
        );
        Self { data, dim }
    }

    /// The dimensionality this `FlatSlice` was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<'a, T: Scalar> DataSource<T> for FlatSlice<'a, T> {
    #[inline]
    fn point_count(&self) -> usize {
        self.data.len() / self.dim
    }

    #[inline]
    fn point_component(&self, idx: usize, dim: usize) -> T {
        self.data[idx * self.dim + dim]
    }

    #[inline]
    fn point_row(&self, idx: usize) -> Option<&[T]> {
        self.data.get(idx * self.dim..idx * self.dim + self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_slice_adaptor_point_count() {
        let points: &[[f32; 3]] = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(points.point_count(), 2);
    }

    #[test]
    fn test_array_slice_adaptor_point_component() {
        let points: &[[f32; 3]] = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(points.point_component(0, 0), 1.0);
        assert_eq!(points.point_component(0, 2), 3.0);
        assert_eq!(points.point_component(1, 1), 5.0);
        assert_eq!(points.point_component(1, 2), 6.0);
    }

    #[test]
    fn test_flat_slice_point_count() {
        let data = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flat = FlatSlice::new(data, 3);
        assert_eq!(flat.point_count(), 2);
    }

    #[test]
    fn test_flat_slice_point_component() {
        let data = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flat = FlatSlice::new(data, 3);
        assert_eq!(flat.point_component(1, 2), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_flat_slice_dim_zero_panics() {
        let data = &[1.0f32, 2.0, 3.0];
        let _ = FlatSlice::new(data, 0);
    }

    #[test]
    #[should_panic]
    fn test_flat_slice_non_multiple_length_panics() {
        let data = &[1.0f32, 2.0, 3.0, 4.0, 5.0];
        let _ = FlatSlice::new(data, 3);
    }

    // ---- `point_row` contract tests ----

    #[test]
    fn test_array_slice_point_row_length_and_values() {
        let points: &[[f32; 3]] = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let row0 = points.point_row(0).expect("array-slice point_row must be Some");
        assert_eq!(row0.len(), 3);
        for (d, &v) in row0.iter().enumerate() {
            assert_eq!(v, points.point_component(0, d), "d={d}");
        }
    }

    #[test]
    fn test_array_slice_point_row_boundary_idx() {
        let points: &[[f64; 4]] = &[[0.0; 4], [1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]];
        let last = points.len() - 1;
        let row = points.point_row(last).expect("boundary idx must return Some");
        assert_eq!(row, &[9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_flat_slice_point_row_length_and_values() {
        let data = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let flat = FlatSlice::new(data, 3);
        let row1 = flat.point_row(1).expect("flat-slice point_row must be Some");
        assert_eq!(row1.len(), 3);
        for (d, &v) in row1.iter().enumerate() {
            assert_eq!(v, flat.point_component(1, d), "d={d}");
        }
    }

    #[test]
    fn test_flat_slice_point_row_boundary_idx() {
        let data = &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flat = FlatSlice::new(data, 3);
        let last = flat.point_count() - 1;
        let row = flat.point_row(last).expect("boundary idx must return Some");
        assert_eq!(row, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flat_slice_point_row_out_of_range_returns_none() {
        let data = &[1.0f32, 2.0, 3.0, 4.0];
        let flat = FlatSlice::new(data, 2);
        // point_count() == 2, so idx 2 is one past the last valid point.
        assert!(flat.point_row(2).is_none());
    }

    #[test]
    fn test_default_fill_bbox_returns_false() {
        let points: &[[f32; 2]] = &[[1.0, 2.0], [3.0, 4.0]];
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 2];
        let result = points.fill_bbox(&mut bbox);
        assert!(!result);
    }

    #[test]
    fn test_custom_fill_bbox_impl() {
        struct CustomDataSource;

        impl DataSource<f32> for CustomDataSource {
            fn point_count(&self) -> usize {
                2
            }

            fn point_component(&self, _idx: usize, _dim: usize) -> f32 {
                0.0
            }

            fn fill_bbox(&self, bbox: &mut [Interval<f32>]) -> bool {
                bbox[0] = Interval { low: 1.0, high: 2.0 };
                bbox[1] = Interval { low: 3.0, high: 4.0 };
                true
            }
        }

        let ds = CustomDataSource;
        let mut bbox = [Interval { low: 0.0, high: 0.0 }; 2];
        let result = ds.fill_bbox(&mut bbox);
        assert!(result);
        assert_eq!(bbox[0].low, 1.0);
        assert_eq!(bbox[0].high, 2.0);
        assert_eq!(bbox[1].low, 3.0);
        assert_eq!(bbox[1].high, 4.0);
    }

    #[test]
    fn test_default_point_row_returns_none_for_custom_data_source() {
        // A hand-rolled DataSource that does not override `point_row` must
        // keep routing every metric through `point_component` -- the whole
        // point of the default being `None`.
        struct CustomDataSource;

        impl DataSource<f32> for CustomDataSource {
            fn point_count(&self) -> usize {
                2
            }

            fn point_component(&self, _idx: usize, _dim: usize) -> f32 {
                0.0
            }
        }

        let ds = CustomDataSource;
        assert!(ds.point_row(0).is_none());
        assert!(ds.point_row(1).is_none());
    }
}
