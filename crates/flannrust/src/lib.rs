#![warn(missing_docs)]
// The internal `(vind, root_bbox, nodes, n)`-shaped build-result tuples
// (`build_sequential_core` and its parallel-build/test-helper analogs)
// are crate-private plumbing shared by exactly a handful of call sites;
// factoring them into a named struct wouldn't make any of those call
// sites clearer, just add a type to look up. Allowed crate-wide since the
// pattern recurs in a few private helpers, not just one.
#![allow(clippy::type_complexity)]
//! Rust port of nanoflann: a STATIC kd-tree ([`tree::KdTree`],
//! `KDTreeSingleIndexAdaptor`) and a DYNAMIC Bentley-Saxe forest
//! ([`dynamic::DynamicKdTree`], `KDTreeSingleIndexDynamicAdaptor`)
//! supporting point add/remove after construction; behavioral parity with
//! nanoflann 1.12.1.
//!
//! # Contracts (see the README for the full list, deviations, and input domain)
//!
//! - `L2`/`L2Simple` distances and radii are SQUARED (`L1` is summed absolute
//!   value); `SO2` is an UNsquared wrapped angle of only the LAST dimension
//!   (single-shot wrap, inputs assumed already in `[-pi, pi]`).
//! - Radius search is strictly `dist < radius` (boundary excluded); box
//!   search is inclusive on all faces, unsorted, traversal order, no params.
//! - kNN ties keep traversal order by default ([`result_set::KeepInsertionOrder`]);
//!   opt into [`result_set::SmallestIndexWins`] for `NANOFLANN_FIRST_MATCH`.
//! - `eps`: a node is visited iff `mindist * (1 + eps) <= worst_dist`, `eps`
//!   widened to the distance type BEFORE the multiply-add.
//! - Queries snapshot the dataset at `build()`; growth afterward is invisible
//!   until a rebuild.
//! - [`ResultItem`] is `#[repr(C)] { index, distance }`.
//! - Coordinates must be finite: NaN/±inf inputs are outside this crate's
//!   domain (see the README's "Input domain" section).
//! - Under the default `parallel` feature, `KdTreeBuilder::build()` requires
//!   `DataSource: Sync`; non-`Sync` data sources use
//!   [`tree::KdTreeBuilder::build_sequential`] instead. [`dynamic::DynamicKdTreeBuilder::build`]
//!   never needs `Sync` at all (the dynamic forest has no parallel build path).
//! - [`dynamic::DynamicKdTree::add_points`]'s contiguous-append contract
//!   (DEVIATION from C++): a genuinely-new (non-reactivation) point index
//!   must equal the forest's running `point_count` at the moment it is
//!   processed — nanoflann silently corrupts its bookkeeping on a
//!   misaligned call, this port panics instead. Reactivating a
//!   previously-removed index is exempt (legally any `start`/`end`).
//! - [`dynamic::DynamicKdTree::find_neighbors`]'s empty-forest quirk
//!   (inherited from C++): with zero occupied slots, `result.full()`
//!   reflects an untouched result set — `false` for knn/rknn, but
//!   hardwired `true` for [`result_set::RadiusResultSet`] regardless of
//!   whether anything was ever added.
//! - The dim-32/64 `L2`/`L1` kernel speedup (M2.5, see the README's
//!   "dim-32/64 knn" section) requires the `DataSource` impl to override
//!   [`data_source::DataSource::point_row`]; a `DataSource` that only
//!   implements `point_component` gets none of it, at any dimensionality.
//!
//! # Example
//!
//! ```
//! use flannrust::{ConstDim, KdTreeBuilder};
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

pub mod bbox;
mod build;
#[cfg(feature = "parallel")]
mod build_parallel;
pub mod data_source;
pub mod dim;
pub mod dynamic;
pub mod filter;
pub mod metric;
mod node;
pub mod params;
pub mod result_set;
pub mod scalar;
mod search;
#[cfg(target_arch = "x86_64")]
mod simd;
pub mod tree;

pub use bbox::Interval;
pub use data_source::{DataSource, FlatSlice, OwnedRows};
pub use dim::{ConstDim, Dim, DynDim};
pub use dynamic::{DynamicKdTree, DynamicKdTreeBuilder};
pub use filter::{AcceptAll, PointFilter};
pub use metric::{Distance, L2Fma, L2Simple, L1, L2, SO2, SO3};
pub use params::{BuildThreads, SearchParams};
pub use result_set::{
    KeepInsertionOrder, KnnResultSet, RadiusResultSet, ResultItem, ResultSet, RknnResultSet,
    SmallestIndexWins, TieBreak,
};
pub use scalar::{DistanceValue, IndexType, Scalar};
pub use tree::{KdTree, KdTreeBuilder};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {}
}
