//! Rust port of nanoflann's static kd-tree; behavioral parity with nanoflann 1.12.1
//!
//! # Example
//!
//! ```
//! use nanoflann_rs::{ConstDim, KdTreeBuilder};
//!
//! let pts: &[[f64; 3]] = &[
//!     [0.0, 0.0, 0.0],
//!     [10.0, 10.0, 10.0],
//!     [1.0, 1.0, 1.0],
//! ];
//! let tree = KdTreeBuilder::new(ConstDim::<3>, pts).build();
//!
//! let mut indices = [0u32; 2];
//! let mut dists = [0.0f64; 2];
//! let found = tree.knn_search(&[0.1, 0.1, 0.1], &mut indices, &mut dists);
//!
//! assert_eq!(found, 2);
//! assert_eq!(indices[0], 0); // nearest point is [0.0, 0.0, 0.0]
//! ```

pub mod scalar;
pub mod dim;
pub mod bbox;
pub mod data_source;
pub mod metric;
pub mod result_set;
pub mod filter;
pub mod params;
pub mod tree;
mod node;
mod build;
mod search;

pub use scalar::{Scalar, DistanceValue, IndexType};
pub use dim::{Dim, ConstDim, DynDim};
pub use bbox::Interval;
pub use data_source::{DataSource, FlatSlice};
pub use metric::{Distance, L1, L2, L2Simple, SO2, SO3};
pub use result_set::{ResultItem, ResultSet, TieBreak, KeepInsertionOrder, SmallestIndexWins, KnnResultSet, RknnResultSet, RadiusResultSet};
pub use filter::{PointFilter, AcceptAll};
pub use params::SearchParams;
pub use tree::{KdTree, KdTreeBuilder};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {}
}
