//! Rust port of nanoflann's static kd-tree; behavioral parity with nanoflann 1.12.1

pub mod scalar;
pub mod dim;
pub mod bbox;
pub mod data_source;

pub use scalar::{Scalar, DistanceValue};
pub use dim::{Dim, ConstDim, DynDim};
pub use bbox::Interval;
pub use data_source::{DataSource, FlatSlice};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {}
}
