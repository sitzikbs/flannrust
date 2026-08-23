//! Dimensionality strategy: compile-time (`ConstDim<N>`) or runtime
//! (`DynDim`), both implementing the `Dim` trait so the rest of the crate is
//! generic over which one a tree was built with.

/// Dimension strategy: compile-time (`ConstDim<N>`) or runtime (`DynDim`).
pub trait Dim: Copy + Send + Sync + 'static {
    /// `[T; N]` for `ConstDim<N>`, `Vec<T>` for `DynDim` — per-dimension scratch/bbox storage.
    type Array<T: Copy + Default + Send + Sync + 'static>: AsRef<[T]>
        + AsMut<[T]>
        + Clone
        + Send
        + Sync;

    /// The dimensionality. `#[inline]` so it constant-folds for `ConstDim`.
    fn dim(self) -> usize;

    /// An `Array` with every slot set to `v`.
    fn filled<T: Copy + Default + Send + Sync + 'static>(self, v: T) -> Self::Array<T>;
}

/// Compile-time dimensionality (C++ `DIM > 0`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstDim<const N: usize>;

impl<const N: usize> Dim for ConstDim<N> {
    type Array<T: Copy + Default + Send + Sync + 'static> = [T; N];

    #[inline(always)]
    fn dim(self) -> usize { N }

    #[inline]
    fn filled<T: Copy + Default + Send + Sync + 'static>(self, v: T) -> [T; N] { [v; N] }
}

/// Runtime dimensionality (C++ `DIM = -1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynDim(pub usize);

impl Dim for DynDim {
    type Array<T: Copy + Default + Send + Sync + 'static> = Vec<T>;

    #[inline(always)]
    fn dim(self) -> usize { self.0 }

    #[inline]
    fn filled<T: Copy + Default + Send + Sync + 'static>(self, v: T) -> Vec<T> { vec![v; self.0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_dim_dim() {
        assert_eq!(ConstDim::<3>.dim(), 3);
    }

    #[test]
    fn test_dyn_dim_dim() {
        assert_eq!(DynDim(5).dim(), 5);
    }

    #[test]
    fn test_const_dim_filled_f32() {
        let a: [f32; 3] = ConstDim::<3>.filled(0.25f32);
        assert_eq!(a, [0.25f32, 0.25f32, 0.25f32]);
    }

    #[test]
    fn test_dyn_dim_filled_f64() {
        let v = DynDim(5).filled(1.0f64);
        assert_eq!(v.len(), 5);
        assert!(v.iter().all(|&x| x == 1.0f64));
    }

    #[test]
    fn test_const_dim_zero_sized() {
        let dim = ConstDim::<0>;
        assert_eq!(dim.dim(), 0);
        let a: [f32; 0] = dim.filled(0.0f32);
        assert_eq!(a.len(), 0);
    }

    fn sum_dims<D: Dim>(d: D) -> usize {
        d.dim()
    }

    #[test]
    fn test_generic_helper_const_dim() {
        let result = sum_dims(ConstDim::<3>);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_generic_helper_dyn_dim() {
        let result = sum_dims(DynDim(7));
        assert_eq!(result, 7);
    }
}
