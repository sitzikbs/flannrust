//! Rust port of nanoflann's static kd-tree; behavioral parity with nanoflann 1.12.1

pub mod scalar;
pub mod dim;
pub mod bbox;
pub mod data_source;
pub mod metric;
pub mod result_set;
mod node;

pub use scalar::{Scalar, DistanceValue};
pub use dim::{Dim, ConstDim, DynDim};
pub use bbox::Interval;
pub use data_source::{DataSource, FlatSlice};
pub use metric::{Distance, L1, L2, L2Simple, SO2, SO3};
pub use result_set::{ResultItem, ResultSet, TieBreak, KeepInsertionOrder, SmallestIndexWins, KnnResultSet, RknnResultSet, RadiusResultSet};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {}
}
