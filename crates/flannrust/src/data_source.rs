//! Zero-copy dataset access: the `DataSource` trait callers implement to
//! hand their point cloud to a `KdTreeBuilder`, plus three built-in
//! implementations (`&[[T; N]]`, the row-major `FlatSlice`, and the owned
//! row-major `OwnedRows`).

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
    /// Behavior for `idx >= point_count()` (out-of-range) is deliberately
    /// UNSPECIFIED by this contract -- an implementation may return `None`
    /// (e.g. by bounds-checking against its backing buffer, as
    /// [`FlatSlice`] does via `slice::get`) or it may return `Some` of
    /// whatever the backing storage happens to hold at that offset (as
    /// `&GrowableFlat` in the `xval` crate does, deliberately matching its
    /// own [`point_component`]'s out-of-range behavior, which also doesn't
    /// consult `point_count()`). Either is a conforming implementation:
    /// callers of `point_row` are never expected to pass an out-of-range
    /// `idx` in the first place (same as `point_component`'s existing
    /// `idx < point_count()` precondition), so this is about implementors
    /// having latitude, not about callers relying on either behavior.
    ///
    /// A `debug_assert`-only spot check against `point_component` guards
    /// the row path's in-tree callers (see `metric::debug_check_point_row_
    /// contract`) against a `point_row` implementation that silently
    /// violates the *in-range* contract above; it does not run in release
    /// builds and is not a soundness mechanism.
    ///
    /// **Performance note (M2.5)**: overriding this is also the precondition
    /// for the dim-32/64 `L2`/`L1` kernel speedup (a bounds-check-free
    /// chunked row walk over the slice this method returns, see the
    /// README's "dim-32/64 knn" section and `docs/benchmarks.md`'s "M2.5 —
    /// performance deep-dive"). A `DataSource` that only implements
    /// [`point_component`] falls back to the per-component loop at every
    /// dimensionality and gets none of that fix — the built-in
    /// [`FlatSlice`] and `&[[T; N]]` impls both override this method for
    /// exactly that reason.
    ///
    /// [`point_component`]: DataSource::point_component
    /// [`FlatSlice`]: crate::data_source::FlatSlice
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

/// Owned row-major buffer: point `i` occupies `data[i*dim .. (i+1)*dim]`
/// (the owning counterpart to [`FlatSlice`], for callers -- e.g. the
/// upcoming Python bindings -- that need to build up a dataset
/// incrementally rather than handing over a borrowed slice up front).
#[derive(Debug, Clone)]
pub struct OwnedRows<T> {
    data: Vec<T>,
    dim: usize,
}

impl<T: Scalar> OwnedRows<T> {
    /// Panics if `dim == 0` or `data.len()` is not a multiple of `dim`.
    pub fn new(data: Vec<T>, dim: usize) -> Self {
        assert!(dim > 0, "OwnedRows dimension must be > 0");
        assert!(
            data.len().is_multiple_of(dim),
            "OwnedRows data length {} is not a multiple of dim {}",
            data.len(),
            dim
        );
        Self { data, dim }
    }

    /// An empty buffer with `dim` fixed and room pre-reserved for
    /// `n_points` rows (`Vec::with_capacity(dim * n_points)`). Panics if
    /// `dim == 0`.
    pub fn with_capacity(dim: usize, n_points: usize) -> Self {
        assert!(dim > 0, "OwnedRows dimension must be > 0");
        Self {
            data: Vec::with_capacity(dim * n_points),
            dim,
        }
    }

    /// Appends whole rows. Panics if `rows.len()` is not a multiple of
    /// `dim()`.
    pub fn push_rows(&mut self, rows: &[T]) {
        assert!(
            rows.len().is_multiple_of(self.dim),
            "OwnedRows::push_rows length {} is not a multiple of dim {}",
            rows.len(),
            self.dim
        );
        self.data.extend_from_slice(rows);
    }

    /// The dimensionality this `OwnedRows` was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of points currently stored.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// `true` iff no points have been stored yet.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The full backing buffer, row-major (`data[i*dim + d]` = point `i`'s
    /// component `d`).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: Scalar> DataSource<T> for OwnedRows<T> {
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

impl<T: Scalar> DataSource<T> for &OwnedRows<T> {
    #[inline]
    fn point_count(&self) -> usize {
        (**self).point_count()
    }

    #[inline]
    fn point_component(&self, idx: usize, dim: usize) -> T {
        (**self).point_component(idx, dim)
    }

    #[inline]
    fn point_row(&self, idx: usize) -> Option<&[T]> {
        (**self).point_row(idx)
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
        let row0 = points
            .point_row(0)
            .expect("array-slice point_row must be Some");
        assert_eq!(row0.len(), 3);
        for (d, &v) in row0.iter().enumerate() {
            assert_eq!(v, points.point_component(0, d), "d={d}");
        }
    }

    #[test]
    fn test_array_slice_point_row_boundary_idx() {
        let points: &[[f64; 4]] = &[[0.0; 4], [1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]];
        let last = points.len() - 1;
        let row = points
            .point_row(last)
            .expect("boundary idx must return Some");
        assert_eq!(row, &[9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_flat_slice_point_row_length_and_values() {
        let data = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let flat = FlatSlice::new(data, 3);
        let row1 = flat
            .point_row(1)
            .expect("flat-slice point_row must be Some");
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
        let mut bbox = [Interval {
            low: 0.0,
            high: 0.0,
        }; 2];
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
                bbox[0] = Interval {
                    low: 1.0,
                    high: 2.0,
                };
                bbox[1] = Interval {
                    low: 3.0,
                    high: 4.0,
                };
                true
            }
        }

        let ds = CustomDataSource;
        let mut bbox = [Interval {
            low: 0.0,
            high: 0.0,
        }; 2];
        let result = ds.fill_bbox(&mut bbox);
        assert!(result);
        assert_eq!(bbox[0].low, 1.0);
        assert_eq!(bbox[0].high, 2.0);
        assert_eq!(bbox[1].low, 3.0);
        assert_eq!(bbox[1].high, 4.0);
    }

    // ---- `OwnedRows` ----

    #[test]
    fn test_owned_rows_new_point_count_and_component() {
        let rows = OwnedRows::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!(rows.point_count(), 2);
        assert_eq!(rows.point_component(0, 0), 1.0);
        assert_eq!(rows.point_component(0, 2), 3.0);
        assert_eq!(rows.point_component(1, 2), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_owned_rows_new_dim_zero_panics() {
        let _ = OwnedRows::new(vec![1.0f32, 2.0, 3.0], 0);
    }

    #[test]
    #[should_panic]
    fn test_owned_rows_new_non_multiple_length_panics() {
        let _ = OwnedRows::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], 3);
    }

    #[test]
    fn test_owned_rows_with_capacity_starts_empty() {
        let rows: OwnedRows<f64> = OwnedRows::with_capacity(4, 10);
        assert_eq!(rows.dim(), 4);
        assert_eq!(rows.len(), 0);
        assert!(rows.is_empty());
        assert!(rows.as_slice().is_empty());
    }

    #[test]
    fn test_owned_rows_push_rows_grows_len_and_values() {
        let mut rows: OwnedRows<f64> = OwnedRows::with_capacity(3, 0);
        rows.push_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(rows.len(), 2);
        assert!(!rows.is_empty());
        assert_eq!(rows.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(rows.point_component(1, 0), 4.0);

        rows.push_rows(&[7.0, 8.0, 9.0]);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows.point_component(2, 2), 9.0);
    }

    #[test]
    #[should_panic]
    fn test_owned_rows_push_rows_ragged_input_panics() {
        let mut rows: OwnedRows<f32> = OwnedRows::with_capacity(3, 0);
        rows.push_rows(&[1.0, 2.0]); // not a multiple of dim 3
    }

    #[test]
    fn test_owned_rows_point_row_length_and_values() {
        let rows = OwnedRows::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3);
        // "`None` never returned for valid idx": exercise every valid idx.
        for idx in 0..rows.len() {
            let row = rows
                .point_row(idx)
                .expect("OwnedRows point_row must be Some for valid idx");
            assert!(row.len() >= 3);
            for (d, &v) in row.iter().enumerate().take(3) {
                assert_eq!(v, rows.point_component(idx, d), "idx={idx} d={d}");
            }
        }
    }

    #[test]
    fn test_owned_rows_point_row_boundary_idx() {
        let rows = OwnedRows::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        let last = rows.len() - 1;
        let row = rows.point_row(last).expect("boundary idx must return Some");
        assert_eq!(row, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_owned_rows_data_source_impl_for_reference() {
        // `DataSource` must be implemented for `&OwnedRows<T>` too (needed
        // so a caller can borrow instead of consuming the owned buffer).
        let rows = OwnedRows::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        let by_ref: &OwnedRows<f32> = &rows;
        assert_eq!(by_ref.point_count(), 2);
        assert_eq!(by_ref.point_component(1, 1), 5.0);
        assert_eq!(by_ref.point_row(0), Some(&[1.0f32, 2.0, 3.0][..]));
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
