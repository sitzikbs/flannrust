//! Rust port of nanoflann's static kd-tree; behavioral parity with nanoflann 1.12.1

pub mod scalar;
pub mod dim;

pub use scalar::{Scalar, DistanceValue};
pub use dim::{Dim, ConstDim, DynDim};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {}
}
