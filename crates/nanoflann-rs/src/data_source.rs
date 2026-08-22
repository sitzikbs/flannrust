use crate::bbox::Interval;
use crate::scalar::Scalar;

/// Zero-copy dataset access (nanoflann's DatasetAdaptor duck-type contract:
/// `kdtree_get_point_count` / `kdtree_get_pt` / `kdtree_get_bbox`).
pub trait DataSource<T: Scalar> {
    fn point_count(&self) -> usize;

    /// Component `dim` of point `idx`. `idx < point_count()`, `dim < dimensionality`.
    fn point_component(&self, idx: usize, dim: usize) -> T;

    /// Optionally fill a precomputed bounding box (= `kdtree_get_bbox`).
    /// Return `false` (the default) to have the tree compute it by scanning.
    fn fill_bbox(&self, _bbox: &mut [Interval<T>]) -> bool {
        false
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
            data.len() % dim == 0,
            "FlatSlice data length {} is not a multiple of dim {}",
            data.len(),
            dim
        );
        Self { data, dim }
    }

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
}
