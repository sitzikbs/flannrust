//! Public façade: `KdTreeBuilder` (configuration) and `KdTree` (the built,
//! queryable index). Wires together `dim.rs`/`data_source.rs`/`metric.rs`
//! (configuration), `build.rs` (construction), and `search.rs` (queries).
//!
//! An unbuilt index is UNREPRESENTABLE in this API: `KdTreeBuilder::build`
//! consumes the builder and returns a ready `KdTree` — there is no separate
//! "call build() before querying" step to forget, and therefore no runtime
//! state to check at query time. This deliberately replaces nanoflann C++'s
//! runtime throw (`std::runtime_error` from `findNeighbors` etc. when the
//! index was never built): the invalid state simply cannot be constructed in
//! Rust's type system, so there is nothing to test with a `compile_fail`
//! doctest either — there is no unbuilt-index misuse to demonstrate.
//!
//! A8 (nanoflann's dead `size_at_index_build_` field is deliberately NOT
//! ported — see `docs/nanoflann-notes.md`): `KdTree` snapshots `vind`/`size`
//! at `build()` time and never re-consults `dataset.point_count()` at query
//! time, so growing the dataset after `build()` cannot change query results
//! until the index is rebuilt. See `stale_growth_after_build_is_ignored`
//! below.

use core::marker::PhantomData;
use core::mem;

use crate::bbox::{compute_bounding_box, Interval};
use crate::build::{init_vind, SubtreeBuilder};
use crate::data_source::DataSource;
use crate::dim::Dim;
use crate::filter::AcceptAll;
use crate::metric::{Distance, L2};
use crate::node::Node;
use crate::params::{BuildThreads, SearchParams};
use crate::result_set::{
    KeepInsertionOrder, KnnResultSet, RadiusResultSet, ResultItem, ResultSet, RknnResultSet,
    TieBreak,
};
use crate::scalar::{DistanceValue, IndexType, Scalar};
use crate::search::{
    find_neighbors as search_find_neighbors, find_within_box as search_find_within_box, SearchCtx,
};

/// Configures and builds a [`KdTree`]. Defaults: `L2` metric, `u32` point
/// indices, insertion-order tie-breaking, `leaf_max_size` 10.
pub struct KdTreeBuilder<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    dim: D,
    dataset: DS,
    metric: M,
    leaf_max_size: usize,
    threads: BuildThreads,
    // T doesn't otherwise appear in a field (only in where-bounds on D/DS/M),
    // so it needs a phantom slot too, alongside Idx/TB.
    _marker: PhantomData<(T, Idx, TB)>,
}

impl<T: Scalar, D: Dim, DS: DataSource<T>> KdTreeBuilder<T, D, DS>
where
    L2: Distance<T>,
{
    /// Defaults: `L2` metric, `u32` indices, insertion-order ties,
    /// `leaf_max_size` 10 (nanoflann's default), `threads`
    /// `BuildThreads::Sequential` (nanoflann's default `n_thread_build` 1).
    ///
    /// `dim` and `dataset` are independent, un-checked-against-each-other
    /// parameters — nothing here verifies that `dim`'s dimensionality
    /// actually matches how many coordinates `dataset` holds per point.
    /// Getting them out of sync is a footgun with two different failure
    /// modes depending on which side is "too big":
    /// - `DynDim(3)` over a `DataSource` whose points actually have 5
    ///   coordinates (e.g. a dim-5 `FlatSlice`) silently uses only the first
    ///   3 of each point — no panic, no error, just a tree built over the
    ///   wrong subset of each point's coordinates.
    /// - `ConstDim::<5>` over `&[[T; 3]]` (a `DataSource` impl whose own
    ///   `point_component` only has 3 valid `dim` values per point) panics
    ///   at build/query time on an out-of-bounds array index.
    ///
    /// Always pass a `dim` that matches `dataset`'s actual per-point
    /// coordinate count.
    pub fn new(dim: D, dataset: DS) -> Self {
        Self {
            dim,
            dataset,
            metric: L2,
            leaf_max_size: 10,
            threads: BuildThreads::default(),
            _marker: PhantomData,
        }
    }
}

impl<T, D, DS, M, Idx, TB> KdTreeBuilder<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Takes a metric INSTANCE — stateful metrics (weights etc.) are
    /// supported, mirroring nanoflann's variadic metric-constructor
    /// forwarding.
    pub fn with_metric<M2: Distance<T>>(self, metric: M2) -> KdTreeBuilder<T, D, DS, M2, Idx, TB> {
        KdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric,
            leaf_max_size: self.leaf_max_size,
            threads: self.threads,
            _marker: PhantomData,
        }
    }

    /// Panics if `n == 0`. Default 10 (nanoflann's default).
    pub fn leaf_max_size(mut self, n: usize) -> Self {
        assert!(n > 0, "leaf_max_size: n must be > 0");
        self.leaf_max_size = n;
        self
    }

    /// Build-thread policy — see [`BuildThreads`]. Default `Sequential`.
    /// `Auto`/`Threads(_)` require the "parallel" cargo feature (default
    /// on); without it, [`Self::build`] panics — see `BuildThreads`'s doc
    /// comment for the exact C++ mapping.
    pub fn threads(mut self, t: BuildThreads) -> Self {
        self.threads = t;
        self
    }

    /// Selects the point-index type stored in query results (`u32`, `u64`,
    /// or `usize`). Default `u32`. Leaf offsets inside the tree's internal
    /// node arena are always `u32` regardless of this choice — see the
    /// "Index types" section of the crate docs.
    pub fn index_type<Idx2: IndexType>(self) -> KdTreeBuilder<T, D, DS, M, Idx2, TB> {
        KdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric: self.metric,
            leaf_max_size: self.leaf_max_size,
            threads: self.threads,
            _marker: PhantomData,
        }
    }

    /// Selects the kNN/RKNN equal-distance tie policy — see
    /// [`crate::result_set::KeepInsertionOrder`] (default) and
    /// [`crate::result_set::SmallestIndexWins`].
    pub fn tie_break<TB2: TieBreak>(self) -> KdTreeBuilder<T, D, DS, M, Idx, TB2> {
        KdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric: self.metric,
            leaf_max_size: self.leaf_max_size,
            threads: self.threads,
            _marker: PhantomData,
        }
    }

    /// Sequential build, ALWAYS available regardless of the "parallel"
    /// feature and regardless of whether `DS: Sync` — unlike [`Self::build`]
    /// (see that method's doc comment), this impl block carries only the
    /// plain `DS: DataSource<T>` bound, so a non-`Sync` `DataSource` (e.g.
    /// `Rc`-backed interior mutability) can still build an index, as long as
    /// `threads` is left at the default `BuildThreads::Sequential`.
    ///
    /// Panics if `threads` was set to `Auto`/`Threads(_)` — those variants
    /// inherently require `DS: Sync` (the parallel build shares `&DS` across
    /// real worker threads), which this method's bound does not guarantee;
    /// use [`Self::build`] instead for those, which has the `DS: Sync` bound
    /// needed to actually service them. Also panics if `point_count() >
    /// u32::MAX as usize` (leaf offsets are `u32`), same as `build()`.
    pub fn build_sequential(self) -> KdTree<T, D, DS, M, Idx, TB> {
        let KdTreeBuilder {
            dim,
            dataset,
            metric,
            leaf_max_size,
            threads,
            ..
        } = self;

        assert!(
            matches!(threads, BuildThreads::Sequential),
            "build_sequential() requires BuildThreads::Sequential; non-Sync datasets cannot build in parallel"
        );

        let (vind, root_bbox, nodes, n) = build_sequential_core(dim, &dataset, leaf_max_size);

        KdTree {
            dataset,
            metric,
            dim,
            vind,
            nodes,
            root_bbox,
            leaf_max_size,
            size: n,
            _marker: PhantomData,
        }
    }
}

/// Shared sequential-build core used by [`KdTreeBuilder::build_sequential`]
/// and both feature-gated [`KdTreeBuilder::build`] impls' `Sequential` path
/// — kept as a free function (rather than duplicated inline three times) so
/// the one true sequential-build sequence has one implementation. Bounded
/// only by plain `DataSource<T>` (no `Sync`), matching `build_sequential`'s
/// contract. Returns `(vind, root_bbox, nodes, n)`; `n == 0` short-circuits
/// to empty everything without touching `SubtreeBuilder` at all (mirrors the
/// original inline `if n == 0` early-return in every one of the three call
/// sites this replaces).
fn build_sequential_core<T, D, DS, Idx>(
    dim: D,
    dataset: &DS,
    leaf_max_size: usize,
) -> (Vec<Idx>, D::Array<Interval<T>>, Vec<Node<T>>, usize)
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    let n = dataset.point_count();

    if n == 0 {
        return (Vec::new(), dim.filled(Interval::default()), Vec::new(), 0);
    }

    let dim_n = dim.dim();
    let mut vind: Vec<Idx> = init_vind(n);
    let mut root_bbox = dim.filled(Interval::default());
    compute_bounding_box(dataset, dim_n, root_bbox.as_mut());

    let mut nodes: Vec<Node<T>> = Vec::new();
    let root = {
        let mut builder = SubtreeBuilder {
            ds: dataset,
            dim: dim_n,
            leaf_max_size,
            base: 0,
            vind: &mut vind,
            arena: &mut nodes,
        };
        builder.build(root_bbox.as_mut())
    };
    debug_assert_eq!(root, 0, "root is expected to always be arena index 0");

    (vind, root_bbox, nodes, n)
}

/// `build()`, feature `"parallel"` ON: `DS: Sync` is required here (and only
/// here — every other builder method stays available for non-`Sync` data
/// sources) because the parallel path shares `&DS` across `rayon::join`'s
/// two closures, which run on separate worker threads; a shared reference
/// crossing threads is `Send` only when the referent is `Sync`. `Sequential`
/// never needs this — see the `#[cfg(not(feature = "parallel"))]` impl below
/// for the non-`Sync`-friendly build available without the feature.
#[cfg(feature = "parallel")]
impl<T, D, DS, M, Idx, TB> KdTreeBuilder<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + Sync,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Builds per `threads` (default `Sequential`). Infallible: an empty
    /// dataset yields an empty index whose searches return 0/false. Panics
    /// if `point_count() > u32::MAX as usize` (leaf offsets are `u32`).
    ///
    /// `Sequential` routes through the same `build_sequential_core` helper
    /// [`KdTreeBuilder::build_sequential`] uses — one true sequential-build
    /// sequence, not a second inline copy. `Auto` builds on rayon's
    /// ambient/global thread pool. `Threads(n)` builds inside a scoped
    /// `n`-thread rayon pool — DEVIATION (documented on [`BuildThreads`]):
    /// this guarantees "at most `n` rayon workers", not C++'s exact
    /// async-gating behavior. Both non-sequential paths produce a tree
    /// BIT-IDENTICAL to `Sequential` — see `build_parallel.rs`'s module doc
    /// for the determinism argument.
    pub fn build(self) -> KdTree<T, D, DS, M, Idx, TB> {
        let KdTreeBuilder {
            dim,
            dataset,
            metric,
            leaf_max_size,
            threads,
            ..
        } = self;

        // `Sequential` is handled entirely by `build_sequential_core` (same
        // helper `build_sequential()` calls) rather than duplicating its
        // n==0/vind/bbox/SubtreeBuilder sequence inline here.
        if let BuildThreads::Sequential = threads {
            let (vind, root_bbox, nodes, n) = build_sequential_core(dim, &dataset, leaf_max_size);
            return KdTree {
                dataset,
                metric,
                dim,
                vind,
                nodes,
                root_bbox,
                leaf_max_size,
                size: n,
                _marker: PhantomData,
            };
        }

        let n = dataset.point_count();

        if n == 0 {
            return KdTree {
                dataset,
                metric,
                dim,
                vind: Vec::new(),
                nodes: Vec::new(),
                root_bbox: dim.filled(Interval::default()),
                leaf_max_size,
                size: 0,
                _marker: PhantomData,
            };
        }

        let dim_n = dim.dim();
        let mut vind: Vec<Idx> = init_vind(n);
        let mut root_bbox = dim.filled(Interval::default());
        compute_bounding_box(&dataset, dim_n, root_bbox.as_mut());

        let nodes: Vec<Node<T>> = match threads {
            BuildThreads::Sequential => unreachable!("Sequential is handled above, before this point"),
            BuildThreads::Auto => crate::build_parallel::build_tree_parallel(
                &dataset,
                dim_n,
                leaf_max_size,
                &mut vind,
                root_bbox.as_mut(),
            ),
            BuildThreads::Threads(n_threads) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads.get() as usize)
                    .build()
                    .expect("rayon pool");
                pool.install(|| {
                    crate::build_parallel::build_tree_parallel(
                        &dataset,
                        dim_n,
                        leaf_max_size,
                        &mut vind,
                        root_bbox.as_mut(),
                    )
                })
            }
        };

        KdTree {
            dataset,
            metric,
            dim,
            vind,
            nodes,
            root_bbox,
            leaf_max_size,
            size: n,
            _marker: PhantomData,
        }
    }
}

/// `build()`, feature `"parallel"` OFF: only `Sequential` is possible (there
/// is no rayon dependency to build with), so this impl does not need `DS:
/// Sync` at all — a non-`Sync` `DataSource` (e.g. `Rc`-backed) works fine
/// here. `Auto`/`Threads(_)` panic — the moral equivalent of C++
/// `NANOFLANN_NO_THREADS`'s throw ("Multithreading is disabled"), per
/// [`BuildThreads`]'s doc comment.
#[cfg(not(feature = "parallel"))]
impl<T, D, DS, M, Idx, TB> KdTreeBuilder<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Sequential build. Infallible: an empty dataset yields an empty index
    /// whose searches return 0/false. Panics if `point_count() >
    /// u32::MAX as usize` (leaf offsets are `u32`), or if `threads` is
    /// anything other than `BuildThreads::Sequential` (see this impl's doc
    /// comment).
    pub fn build(self) -> KdTree<T, D, DS, M, Idx, TB> {
        let KdTreeBuilder {
            dim,
            dataset,
            metric,
            leaf_max_size,
            threads,
            ..
        } = self;

        if !matches!(threads, BuildThreads::Sequential) {
            panic!(
                "flannrust was compiled without the 'parallel' feature; BuildThreads::Auto/Threads need it"
            );
        }

        let (vind, root_bbox, nodes, n) = build_sequential_core(dim, &dataset, leaf_max_size);

        KdTree {
            dataset,
            metric,
            dim,
            vind,
            nodes,
            root_bbox,
            leaf_max_size,
            size: n,
            _marker: PhantomData,
        }
    }
}

/// A built, immutable kd-tree index. See the module docs for why "unbuilt"
/// is not a representable state, and for the A8 stale-growth contract.
pub struct KdTree<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    dataset: DS,
    metric: M,
    dim: D,
    vind: Vec<Idx>,
    nodes: Vec<Node<T>>,
    root_bbox: D::Array<Interval<T>>,
    leaf_max_size: usize,
    /// Dataset size AT BUILD TIME (= `vind.len()`), snapshotted once and
    /// never re-derived from `dataset.point_count()` — see the A8 module doc.
    size: usize,
    _marker: PhantomData<TB>,
}

impl<T, D, DS, M, Idx, TB> KdTree<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Dataset size at build time (= `vind.len()`). See the A8 module doc:
    /// this NEVER changes even if the underlying `DataSource` grows.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The tree's dimensionality.
    pub fn dim(&self) -> usize {
        self.dim.dim()
    }

    /// The `leaf_max_size` this tree was built with.
    pub fn leaf_max_size(&self) -> usize {
        self.leaf_max_size
    }

    /// Heap bytes held by the index structures (nodes + vind) — the moral
    /// equivalent of nanoflann's `usedMemory()`: `len`-based (not actual
    /// allocator-reported capacity), documented as such.
    pub fn used_memory_bytes(&self) -> usize {
        self.nodes.len() * mem::size_of::<Node<T>>() + self.vind.len() * mem::size_of::<Idx>()
    }

    /// The tree's tight root bounding box. Empty slice content for an empty
    /// tree is fine (still `dim()` entries, each `Interval::default()`).
    pub fn root_bbox(&self) -> &[Interval<T>] {
        self.root_bbox.as_ref()
    }

    /// The build-time permutation of point indices, = nanoflann's `vAcc_`;
    /// exposed for introspection and cross-validation.
    pub fn point_indices(&self) -> &[Idx] {
        &self.vind
    }

    fn ctx(&self) -> SearchCtx<'_, T, D, DS, M, Idx> {
        SearchCtx {
            ds: &self.dataset,
            metric: &self.metric,
            dim: self.dim,
            nodes: &self.nodes,
            vind: &self.vind,
            root_bbox: self.root_bbox.as_ref(),
        }
    }

    /// `k` = out slices' length (must be equal; panics otherwise). Returns
    /// the found count (`< k` iff `size() < k`). `eps = 0`, sorted — like
    /// C++'s static `knnSearch`.
    pub fn knn_search(&self, query: &[T], out_indices: &mut [Idx], out_dists: &mut [M::DistanceType]) -> usize {
        self.knn_search_with(query, out_indices, out_dists, &SearchParams::default())
    }

    /// Same as [`Self::knn_search`] with explicit params (ADDITIVE over C++,
    /// which has no params on static `knnSearch`).
    pub fn knn_search_with(
        &self,
        query: &[T],
        out_indices: &mut [Idx],
        out_dists: &mut [M::DistanceType],
        params: &SearchParams,
    ) -> usize {
        assert_eq!(
            out_indices.len(),
            out_dists.len(),
            "knn_search: out_indices/out_dists length mismatch"
        );
        let ctx = self.ctx();
        let mut rs = KnnResultSet::<M::DistanceType, Idx, TB>::new(out_indices, out_dists);
        let mut scratch = self.dim.filled(M::DistanceType::ZERO);
        search_find_neighbors(&ctx, &mut rs, query, params, &AcceptAll, scratch.as_mut());
        rs.size()
    }

    /// `radius` is in the metric's scale: SQUARED for L2-family. Strictness
    /// matters at two different boundaries: while fewer than `k` points have
    /// been found, the cutoff IS `radius` (strict `<`: a point at exactly
    /// `radius` is excluded); once `k` points have been found, the effective
    /// cutoff tightens to the k-th smallest distance found so far (same
    /// strict `<`), exactly like [`Self::knn_search`] from that point on.
    pub fn rknn_search(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out_indices: &mut [Idx],
        out_dists: &mut [M::DistanceType],
    ) -> usize {
        self.rknn_search_with(query, radius, out_indices, out_dists, &SearchParams::default())
    }

    /// Same as [`Self::rknn_search`] with explicit params.
    pub fn rknn_search_with(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out_indices: &mut [Idx],
        out_dists: &mut [M::DistanceType],
        params: &SearchParams,
    ) -> usize {
        assert_eq!(
            out_indices.len(),
            out_dists.len(),
            "rknn_search: out_indices/out_dists length mismatch"
        );
        let ctx = self.ctx();
        let mut rs = RknnResultSet::<M::DistanceType, Idx, TB>::new(out_indices, out_dists, radius);
        let mut scratch = self.dim.filled(M::DistanceType::ZERO);
        search_find_neighbors(&ctx, &mut rs, query, params, &AcceptAll, scratch.as_mut());
        rs.size()
    }

    /// Clears `out`. STRICTLY `dist < radius`. `eps = 0`, sorted — like
    /// [`Self::knn_search`]'s defaults.
    pub fn radius_search(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out: &mut Vec<ResultItem<Idx, M::DistanceType>>,
    ) -> usize {
        self.radius_search_with(query, radius, out, &SearchParams::default())
    }

    /// Same as [`Self::radius_search`] with explicit params.
    pub fn radius_search_with(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out: &mut Vec<ResultItem<Idx, M::DistanceType>>,
        params: &SearchParams,
    ) -> usize {
        let ctx = self.ctx();
        let mut rs = RadiusResultSet::new(radius, out);
        let mut scratch = self.dim.filled(M::DistanceType::ZERO);
        search_find_neighbors(&ctx, &mut rs, query, params, &AcceptAll, scratch.as_mut());
        rs.size()
    }

    /// Inclusive faces, traversal order, never sorts, no params. Clears
    /// `out`. Returns the found count. Panics if `bounds.len() != dim()`.
    pub fn find_within_box(&self, bounds: &[Interval<T>], out: &mut Vec<Idx>) -> usize {
        assert_eq!(
            bounds.len(),
            self.dim.dim(),
            "find_within_box: bounds.len() != dim"
        );
        let ctx = self.ctx();
        search_find_within_box(&ctx, bounds, out)
    }

    /// Generic escape hatch (= C++ `findNeighbors`). Returns
    /// `result.full()`. **Empty-tree quirk** (mirrors the dynamic forest's,
    /// see `dynamic`'s module doc): with zero points, `result.full()`
    /// reflects an untouched result set — `false` for `KnnResultSet`/
    /// `RknnResultSet`, but **`true`** for `RadiusResultSet` (hardwired,
    /// nanoflann.hpp:433) regardless of whether anything was ever added.
    pub fn find_neighbors<R: ResultSet<M::DistanceType, Idx>>(
        &self,
        result: &mut R,
        query: &[T],
        params: &SearchParams,
    ) -> bool {
        let ctx = self.ctx();
        let mut scratch = self.dim.filled(M::DistanceType::ZERO);
        search_find_neighbors(&ctx, result, query, params, &AcceptAll, scratch.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{ConstDim, DynDim};
    use crate::data_source::FlatSlice;
    use crate::metric::L1;
    use crate::result_set::SmallestIndexWins;

    // ---------------------------------------------------------------
    // Shared helpers
    // ---------------------------------------------------------------

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn seeded_points<const N: usize>(seed: u64, n: usize, scale: f64) -> Vec<[f64; N]> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let mut p = [0.0f64; N];
                for v in p.iter_mut() {
                    *v = rng.next_f64() * scale;
                }
                p
            })
            .collect()
    }

    /// Brute-force KNN reference computed via the SAME metric `eval` used by
    /// the tree (not a hand-rolled sum), so summation-order ULP differences
    /// can never cause a spurious mismatch — bit-exact comparison is valid.
    fn brute_force_knn<const N: usize, Mtr: Distance<f64>>(
        metric: &Mtr,
        pts: &[[f64; N]],
        query: &[f64; N],
        k: usize,
    ) -> (Vec<u32>, Vec<Mtr::DistanceType>) {
        let mut indices = vec![0u32; k];
        let mut dists = vec![Mtr::DistanceType::ZERO; k];
        let count;
        {
            let mut rs = KnnResultSet::<Mtr::DistanceType, u32>::new(&mut indices, &mut dists);
            for i in 0..pts.len() {
                let d = metric.eval(query.as_slice(), &pts, i, ConstDim::<N>);
                rs.add_point(d, i as u32);
            }
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists)
    }

    // ---------------------------------------------------------------
    // Test 1: defaults — leaf_max_size 10, knn vs brute force (exact)
    // ---------------------------------------------------------------

    #[test]
    fn defaults_leaf_max_size_10_and_knn_matches_brute_force() {
        let pts = seeded_points::<3>(0xC0FFEE, 100, 100.0);
        let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();

        assert_eq!(tree.leaf_max_size(), 10);
        assert_eq!(tree.size(), 100);
        assert_eq!(tree.dim(), 3);

        let query = [12.3, 45.6, 78.9];
        let k = 5;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let found = tree.knn_search(&query, &mut indices, &mut dists);

        let (want_idx, want_dist) = brute_force_knn(&L2, &pts, &query, k);
        assert_eq!(found, k);
        assert_eq!(indices, want_idx);
        assert_eq!(dists, want_dist);
    }

    // ---------------------------------------------------------------
    // Test 2: DynDim == ConstDim (bit-equal)
    // ---------------------------------------------------------------

    #[test]
    fn dyn_dim_matches_const_dim_bit_equal() {
        let pts = seeded_points::<3>(0xDEADBEEF, 100, 50.0);
        let flat: Vec<f64> = pts.iter().flat_map(|p| p.iter().copied()).collect();

        let const_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();
        let dyn_tree = KdTreeBuilder::new(DynDim(3), FlatSlice::new(&flat, 3)).build();

        let query = [10.0, 20.0, 30.0];
        let k = 7;
        let mut c_idx = vec![0u32; k];
        let mut c_dist = vec![0.0f64; k];
        let c_found = const_tree.knn_search(&query, &mut c_idx, &mut c_dist);

        let mut d_idx = vec![0u32; k];
        let mut d_dist = vec![0.0f64; k];
        let d_found = dyn_tree.knn_search(&query, &mut d_idx, &mut d_dist);

        assert_eq!(c_found, d_found);
        assert_eq!(c_idx, d_idx);
        assert_eq!(c_dist, d_dist, "DynDim/ConstDim distances must be bit-equal");
    }

    // ---------------------------------------------------------------
    // Test 3: radius_search — clears out, sorted vs traversal order,
    // strict `<` boundary.
    // ---------------------------------------------------------------

    #[test]
    fn radius_search_clears_pre_populated_out() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0]];
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();

        let mut out = vec![ResultItem { index: 999u32, distance: 12345.0f64 }];
        let params = SearchParams::default();
        let found = tree.radius_search_with(&[0.0, 0.0], 100.0, &mut out, &params);

        assert_eq!(found, 2);
        assert!(!out.iter().any(|r| r.index == 999), "pre-populated junk must be cleared");
    }

    #[test]
    fn radius_search_sorted_true_gives_ascending_sorted_false_gives_traversal_order() {
        // Same construction as search.rs's hand-derived traversal-order
        // test, but built end-to-end through the public builder (not a
        // hand-spliced arena): points 0.0,1.0,10.0,11.0 (dim-1),
        // leaf_max_size=2, query=0.4. The builder produces the identical
        // node layout search.rs derived by hand (same middle_split/
        // plane_split algorithm), so traversal order is [0, 1, 3, 2].
        let pts: Vec<[f64; 1]> = vec![[0.0], [1.0], [10.0], [11.0]];
        let tree = KdTreeBuilder::new(ConstDim::<1>, pts.as_slice())
            .leaf_max_size(2)
            .build();

        let mut unsorted = Vec::new();
        let unsorted_params = SearchParams { eps: 0.0, sorted: false };
        tree.radius_search_with(&[0.4], 200.0, &mut unsorted, &unsorted_params);
        let order: Vec<u32> = unsorted.iter().map(|r| r.index).collect();
        assert_eq!(order, vec![0, 1, 3, 2], "sorted=false must yield raw traversal order");

        let mut sorted = Vec::new();
        let sorted_params = SearchParams { eps: 0.0, sorted: true };
        tree.radius_search_with(&[0.4], 200.0, &mut sorted, &sorted_params);
        let dists: Vec<f64> = sorted.iter().map(|r| r.distance).collect();
        let mut ascending = dists.clone();
        ascending.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, ascending, "sorted=true must yield ascending distances");
    }

    #[test]
    fn radius_search_strict_less_than_at_exact_boundary() {
        // Point (2,0) is at EXACTLY squared distance 4.0 from the origin.
        let pts: Vec<[f64; 2]> = vec![[2.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();
        let params = SearchParams::default();

        let mut out = Vec::new();
        tree.radius_search_with(&[0.0, 0.0], 4.0, &mut out, &params);
        assert!(!out.iter().any(|r| r.index == 0), "exact-boundary point must be excluded");

        let radius_next_up = f64::from_bits(4.0f64.to_bits() + 1);
        let mut out2 = Vec::new();
        tree.radius_search_with(&[0.0, 0.0], radius_next_up, &mut out2, &params);
        assert!(out2.iter().any(|r| r.index == 0), "next_up(4.0) must include the boundary point");
    }

    // ---------------------------------------------------------------
    // Test 4: rknn_search — partial coverage vs full coverage/closest-k
    // ---------------------------------------------------------------

    #[test]
    fn rknn_search_partial_and_full_coverage() {
        // 10 points at squared distances 1,4,...,100 from the origin along
        // one axis.
        let pts: Vec<[f64; 1]> = (1..=10).map(|i| [i as f64]).collect();
        let tree = KdTreeBuilder::new(ConstDim::<1>, pts.as_slice()).build();

        // radius 10.0 (squared) covers only points at dist^2 in {1,4,9} -> 3 points.
        let mut idx = [0u32; 5];
        let mut dist = [0.0f64; 5];
        let found = tree.rknn_search(&[0.0], 10.0, &mut idx, &mut dist);
        assert_eq!(found, 3, "expected exactly 3 points within radius 10.0 (squared)");
        assert_eq!(&idx[..3], &[0, 1, 2]);

        // radius 1000.0 covers all 10 -> closest 5 returned.
        let mut idx2 = [0u32; 5];
        let mut dist2 = [0.0f64; 5];
        let found2 = tree.rknn_search(&[0.0], 1000.0, &mut idx2, &mut dist2);
        assert_eq!(found2, 5);
        assert_eq!(idx2, [0, 1, 2, 3, 4]);
    }

    // ---------------------------------------------------------------
    // Test 5: find_within_box — panics on wrong bounds len, brute-force
    // inclusive filter on a grid.
    // ---------------------------------------------------------------

    #[test]
    #[should_panic]
    fn find_within_box_panics_on_wrong_bounds_len() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0]];
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();
        let mut out = Vec::new();
        let bad_bounds = [Interval { low: 0.0, high: 1.0 }]; // len 1, dim is 2
        tree.find_within_box(&bad_bounds, &mut out);
    }

    #[test]
    fn find_within_box_matches_brute_force_inclusive_filter_on_grid() {
        let pts: Vec<[f64; 2]> = (0..5)
            .flat_map(|x| (0..5).map(move |y| [x as f64, y as f64]))
            .collect();
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();

        let bounds = [
            Interval { low: 1.0, high: 3.0 },
            Interval { low: 1.0, high: 3.0 },
        ];
        let mut out = Vec::new();
        let found = tree.find_within_box(&bounds, &mut out);

        let mut want: Vec<u32> = pts
            .iter()
            .enumerate()
            .filter(|(_, p)| bounds[0].contains(p[0]) && bounds[1].contains(p[1]))
            .map(|(i, _)| i as u32)
            .collect();
        let mut got = out.clone();
        want.sort_unstable();
        got.sort_unstable();

        assert_eq!(found, want.len());
        assert_eq!(got, want);
    }

    // ---------------------------------------------------------------
    // Test 6: k > n, and empty dataset — no panics, everything returns 0/false
    // ---------------------------------------------------------------

    #[test]
    fn k_greater_than_n_returns_partial_result_not_full() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();

        let mut idx = vec![0u32; 10];
        let mut dist = vec![0.0f64; 10];
        let found = tree.knn_search(&[0.5, 0.5], &mut idx, &mut dist);
        assert_eq!(found, 3);

        let (want_idx, want_dist) = brute_force_knn(&L2, &pts, &[0.5, 0.5], 3);
        assert_eq!(&idx[..3], want_idx.as_slice());
        assert_eq!(&dist[..3], want_dist.as_slice());
    }

    #[test]
    fn empty_dataset_every_search_returns_zero_no_panics() {
        let pts: Vec<[f64; 3]> = Vec::new();
        let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();

        assert_eq!(tree.size(), 0);
        assert_eq!(tree.dim(), 3);

        let mut idx = [0u32; 3];
        let mut dist = [0.0f64; 3];
        assert_eq!(tree.knn_search(&[0.0, 0.0, 0.0], &mut idx, &mut dist), 0);
        assert_eq!(tree.rknn_search(&[0.0, 0.0, 0.0], 100.0, &mut idx, &mut dist), 0);

        let mut radius_out = Vec::new();
        assert_eq!(
            tree.radius_search_with(&[0.0, 0.0, 0.0], 100.0, &mut radius_out, &SearchParams::default()),
            0
        );
        assert!(radius_out.is_empty());

        let mut box_out = Vec::new();
        let bounds = [
            Interval { low: -1.0, high: 1.0 },
            Interval { low: -1.0, high: 1.0 },
            Interval { low: -1.0, high: 1.0 },
        ];
        assert_eq!(tree.find_within_box(&bounds, &mut box_out), 0);
        assert!(box_out.is_empty());

        let mut rs = KnnResultSet::<f64, u32>::new(&mut idx, &mut dist);
        let full = tree.find_neighbors(&mut rs, &[0.0, 0.0, 0.0], &SearchParams::default());
        assert!(!full);
    }

    // ---------------------------------------------------------------
    // Test 7: with_metric(L1) end-to-end vs brute-force L1
    // ---------------------------------------------------------------

    #[test]
    fn with_metric_l1_end_to_end_matches_brute_force_l1() {
        let pts = seeded_points::<3>(0xAB1234, 80, 40.0);
        let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice())
            .with_metric(L1)
            .build();

        let query = [11.0, 22.0, 33.0];
        let k = 6;
        let mut idx = vec![0u32; k];
        let mut dist = vec![0.0f64; k];
        let found = tree.knn_search(&query, &mut idx, &mut dist);

        let (want_idx, want_dist) = brute_force_knn(&L1, &pts, &query, k);
        assert_eq!(found, k);
        assert_eq!(idx, want_idx);
        assert_eq!(dist, want_dist);
    }

    // ---------------------------------------------------------------
    // Test 8: tie_break — SmallestIndexWins vs KeepInsertionOrder
    // ---------------------------------------------------------------

    /// BRIEF NOTE: the brief suggests "4 duplicate points, k=2" as the
    /// scenario. That specific shape does NOT discriminate between the two
    /// tie-break policies: `search_level`'s prune gate
    /// (`dist < result.worst_dist()`, checked BEFORE `add_point` is ever
    /// called) blocks every candidate whose distance ties the current
    /// worst once the result set is full — so once 2 duplicates fill a
    /// `k=2` buffer, the remaining 2 duplicates never reach `add_point` at
    /// all, regardless of `TieBreak`. Both policies land on the same
    /// "first two visited" answer. This mirrors task-8's proven
    /// discrepancy for the eps test (see `search.rs`).
    ///
    /// Instead: 4 identical points, `leaf_max_size = 2` (forces one split;
    /// `plane_split` never swaps when every value equals `cutval`, so
    /// `vind` for the all-duplicate case stays split into two CONTIGUOUS,
    /// order-preserving halves `[0,1]`/`[2,3]`), `k = 4` (== point count,
    /// so the result set never fills before every point is offered — no
    /// gate-blocking at all). With `divlow == divhigh == cutval == query`,
    /// the search heuristic's `(diff1+diff2) < 0` is false (sum is exactly
    /// 0), so it visits the RIGHT subtree first: traversal (= insertion)
    /// order is `[2, 3, 0, 1]`. `KeepInsertionOrder` (`prev_d > d`, ties
    /// never shift) preserves that raw order exactly. `SmallestIndexWins`
    /// (`prev_d > d || (d == prev_d && prev_i > i)`) shifts every tie by
    /// index, fully re-sorting to `[0, 1, 2, 3]` — genuinely discriminating
    /// and deterministic. Hand-traced in the task-10 report.
    #[test]
    fn tie_break_smallest_index_wins_vs_keep_insertion_order() {
        let pts: Vec<[f64; 1]> = vec![[5.0], [5.0], [5.0], [5.0]];

        let keep_tree = KdTreeBuilder::new(ConstDim::<1>, pts.as_slice())
            .leaf_max_size(2)
            .build();
        let mut keep_idx = [0u32; 4];
        let mut keep_dist = [0.0f64; 4];
        let keep_found = keep_tree.knn_search(&[5.0], &mut keep_idx, &mut keep_dist);
        assert_eq!(keep_found, 4);
        assert_eq!(keep_idx, [2, 3, 0, 1], "default KeepInsertionOrder must be raw traversal order");

        let smallest_tree = KdTreeBuilder::new(ConstDim::<1>, pts.as_slice())
            .leaf_max_size(2)
            .tie_break::<SmallestIndexWins>()
            .build();
        let mut smallest_idx = [0u32; 4];
        let mut smallest_dist = [0.0f64; 4];
        let smallest_found = smallest_tree.knn_search(&[5.0], &mut smallest_idx, &mut smallest_dist);
        assert_eq!(smallest_found, 4);
        assert_eq!(smallest_idx, [0, 1, 2, 3], "SmallestIndexWins must fully re-sort by index on ties");
    }

    // ---------------------------------------------------------------
    // Test 9: stateful custom metric end-to-end (Task-3's WeightedL2)
    // ---------------------------------------------------------------

    struct WeightedL2 {
        weights: Vec<f64>,
    }

    impl Distance<f64> for WeightedL2 {
        type DistanceType = f64;

        // `d` indexes three independent sources at once (`query`, the
        // `DataSource` trait method, and `self.weights`) — no single
        // iterator adaptor covers all three cleanly.
        #[allow(clippy::needless_range_loop)]
        fn eval<Ds: DataSource<f64> + ?Sized, D: Dim>(&self, query: &[f64], ds: &Ds, idx: usize, dim: D) -> f64 {
            let dim = dim.dim();
            let mut result = 0.0;
            for d in 0..dim {
                let diff = query[d] - ds.point_component(idx, d);
                result += self.weights[d] * diff * diff;
            }
            result
        }

        fn accum_dist(&self, a: f64, b: f64, axis: usize) -> f64 {
            let diff = a - b;
            self.weights[axis] * diff * diff
        }
    }

    #[test]
    fn stateful_weighted_metric_end_to_end_changes_the_nearest_neighbor() {
        // Weighted heavily on dim 1: PA=(1,0) is raw-farther but
        // weight-nearer than PB=(0,0.5) which is raw-nearer but
        // weight-farther.
        let pts: Vec<[f64; 2]> = vec![[1.0, 0.0], [0.0, 0.5]];
        let query = [0.0, 0.0];

        // Sanity: unweighted L2 would pick PB (index 1).
        let (unweighted_idx, _) = brute_force_knn(&L2, &pts, &query, 1);
        assert_eq!(unweighted_idx, vec![1], "test setup: unweighted nearest should be PB");

        let metric = WeightedL2 { weights: vec![1.0, 100.0] };
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice())
            .with_metric(metric)
            .build();

        let mut idx = [0u32; 1];
        let mut dist = [0.0f64; 1];
        let found = tree.knn_search(&query, &mut idx, &mut dist);
        assert_eq!(found, 1);
        assert_eq!(idx, [0], "weighted nearest must be PA (index 0), not the unweighted winner");
    }

    // ---------------------------------------------------------------
    // Test 10: used_memory_bytes, size(), dim()
    // ---------------------------------------------------------------

    #[test]
    fn used_memory_bytes_positive_and_grows_with_n_size_and_dim_correct() {
        let small_pts = seeded_points::<2>(0x5, 5, 10.0);
        let small_tree = KdTreeBuilder::new(ConstDim::<2>, small_pts.as_slice()).build();
        assert_eq!(small_tree.size(), 5);
        assert_eq!(small_tree.dim(), 2);
        assert!(small_tree.used_memory_bytes() > 0);

        let big_pts = seeded_points::<2>(0x5, 500, 10.0);
        let big_tree = KdTreeBuilder::new(ConstDim::<2>, big_pts.as_slice()).build();
        assert_eq!(big_tree.size(), 500);
        assert!(
            big_tree.used_memory_bytes() > small_tree.used_memory_bytes(),
            "used_memory_bytes should grow with n"
        );
    }

    // ---------------------------------------------------------------
    // Test 11: point_indices() is a permutation of 0..n
    // ---------------------------------------------------------------

    #[test]
    fn point_indices_is_a_permutation_of_0_to_n() {
        let pts = seeded_points::<3>(0x1234, 37, 25.0);
        let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();
        let mut got: Vec<u32> = tree.point_indices().to_vec();
        assert_eq!(got.len(), 37);
        got.sort_unstable();
        assert_eq!(got, (0u32..37).collect::<Vec<u32>>());
    }

    // ---------------------------------------------------------------
    // Test 12 (A8): stale-growth snapshot
    // ---------------------------------------------------------------

    use std::cell::Cell;
    use std::rc::Rc;

    /// A `DataSource` with interior mutability: `point_count()` reports a
    /// `Cell<usize>` that can be grown AFTER the tree is built, over a fixed
    /// backing array. `Rc<Cell<usize>>` is deliberately NOT `Sync` — this is
    /// the regression-proof (task 12 fix round 1) that non-`Sync`
    /// `DataSource`s can still build an index via
    /// `KdTreeBuilder::build_sequential()`, which carries no `Sync` bound
    /// (unlike `build()`, which requires `DS: Sync` whenever the "parallel"
    /// feature is on — see `params.rs`'s `BuildThreads` doc comment). The
    /// FIRST `count` slots (visible at build time) are deliberately placed
    /// FAR from the test query; slots that only become visible after growth
    /// are placed NEAR the query — so if the implementation ever
    /// (incorrectly) re-derived tree state from `dataset.point_count()` at
    /// query time, this test would catch it by observing a changed nearest
    /// neighbor after growth.
    struct GrowableDataSource {
        data: [[f64; 2]; 8],
        count: Rc<Cell<usize>>,
    }

    impl DataSource<f64> for GrowableDataSource {
        fn point_count(&self) -> usize {
            self.count.get()
        }
        fn point_component(&self, idx: usize, dim: usize) -> f64 {
            self.data[idx][dim]
        }
    }

    #[test]
    fn stale_growth_after_build_is_ignored() {
        let count = Rc::new(Cell::new(3));
        let ds = GrowableDataSource {
            data: [
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                // Only visible after growth; all much nearer the query
                // (0,0) than anything above.
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
            ],
            count: count.clone(),
        };

        // build_sequential(), NOT build(): `Rc<Cell<usize>>` is not `Sync`,
        // so `build()` would fail to compile under the "parallel" feature —
        // this is exactly the case `build_sequential()` exists for.
        let tree = KdTreeBuilder::new(ConstDim::<2>, ds).build_sequential();
        assert_eq!(tree.size(), 3);

        let query = [0.0, 0.0];
        let k = 2;
        let mut idx_before = [0u32; 2];
        let mut dist_before = [0.0f64; 2];
        let found_before = tree.knn_search(&query, &mut idx_before, &mut dist_before);

        // Grow the dataset AFTER build.
        count.set(8);

        assert_eq!(tree.size(), 3, "size() must stay snapshotted at build time");

        let mut idx_after = [0u32; 2];
        let mut dist_after = [0.0f64; 2];
        let found_after = tree.knn_search(&query, &mut idx_after, &mut dist_after);

        assert_eq!(found_before, k);
        assert_eq!(found_before, found_after);
        assert_eq!(idx_before, idx_after, "growth must not change knn results");
        assert_eq!(dist_before, dist_after);
        // Sanity: the query results are the ORIGINAL far points, not the
        // newly-visible near ones — proves the tree really ignored growth
        // rather than coincidentally matching.
        assert_eq!(idx_before, [0, 1]);
    }

    // ---------------------------------------------------------------
    // Test 13 (task 12): threads() end-to-end via the public builder API —
    // belt-and-braces alongside build_parallel.rs's lower-level tests 1-4,
    // exercised here through the SAME surface real callers use
    // (`KdTreeBuilder::threads(..).build()` + `knn_search`).
    // ---------------------------------------------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn threads_auto_and_threads2_end_to_end_knn_matches_sequential() {
        use crate::params::BuildThreads;

        let pts = seeded_points::<3>(0x7EED, 5000, 200.0);

        let seq_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice())
            .leaf_max_size(10)
            .threads(BuildThreads::Sequential)
            .build();
        let auto_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice())
            .leaf_max_size(10)
            .threads(BuildThreads::Auto)
            .build();
        let threads2_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice())
            .leaf_max_size(10)
            .threads(BuildThreads::Threads(core::num::NonZeroU32::new(2).unwrap()))
            .build();

        assert_eq!(seq_tree.point_indices(), auto_tree.point_indices());
        assert_eq!(seq_tree.point_indices(), threads2_tree.point_indices());

        let mut rng = Lcg(0x51DE);
        for _ in 0..50 {
            let q = [rng.next_f64() * 200.0, rng.next_f64() * 200.0, rng.next_f64() * 200.0];
            let k = 8;

            let mut si = vec![0u32; k];
            let mut sd = vec![0.0f64; k];
            let sf = seq_tree.knn_search(&q, &mut si, &mut sd);

            let mut ai = vec![0u32; k];
            let mut ad = vec![0.0f64; k];
            let af = auto_tree.knn_search(&q, &mut ai, &mut ad);

            let mut ti = vec![0u32; k];
            let mut td = vec![0.0f64; k];
            let tf = threads2_tree.knn_search(&q, &mut ti, &mut td);

            assert_eq!(sf, af);
            assert_eq!(si, ai);
            assert_eq!(sd, ad);
            assert_eq!(sf, tf);
            assert_eq!(si, ti);
            assert_eq!(sd, td);
        }
    }

    // ---------------------------------------------------------------
    // Test 14 (task 12 fix round 1): build_sequential() — always available
    // regardless of "parallel"/`DS: Sync`. The positive non-Sync case is
    // `stale_growth_after_build_is_ignored` above (uses
    // `Rc<Cell<usize>>::build_sequential()` directly — that test IS the
    // regression proof); the positive Sync-dataset-via-plain-`build()`-plus-
    // -Sequential case is already covered by
    // `defaults_leaf_max_size_10_and_knn_matches_brute_force` (default
    // `threads` is `Sequential`, dataset is `&[[f64; N]]`, which is `Sync`)
    // and, under the "parallel" feature, by
    // `threads_auto_and_threads2_end_to_end_knn_matches_sequential` above
    // (`.threads(BuildThreads::Sequential).build()`). This test covers the
    // one behavior not exercised elsewhere: `build_sequential()` panics on
    // non-`Sequential` `threads`.
    // ---------------------------------------------------------------

    #[test]
    #[should_panic(expected = "build_sequential() requires BuildThreads::Sequential")]
    fn build_sequential_panics_on_non_sequential_threads() {
        use crate::params::BuildThreads;

        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let _ = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice())
            .threads(BuildThreads::Auto)
            .build_sequential();
    }

    #[test]
    fn build_sequential_matches_plain_build_on_a_sync_dataset() {
        let pts = seeded_points::<3>(0x5EA1, 300, 40.0);

        let via_build = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();
        let via_build_sequential = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build_sequential();

        assert_eq!(via_build.point_indices(), via_build_sequential.point_indices());
        assert_eq!(via_build.size(), via_build_sequential.size());
    }

    // ---------------------------------------------------------------
    // Test 15 (I9): `index_type` actually instantiated for u64/usize —
    // knn results must match the default u32 tree exactly (indices equal as
    // usize, distances bit-equal).
    // ---------------------------------------------------------------

    #[test]
    fn index_type_u64_and_usize_match_default_u32_tree() {
        let pts = seeded_points::<3>(0x1D57E, 250, 60.0);

        let u32_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();
        let u64_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).index_type::<u64>().build();
        let usize_tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).index_type::<usize>().build();

        let mut rng = Lcg(0xC0DE_1D5E);
        for _ in 0..30 {
            let query = [rng.next_f64() * 60.0, rng.next_f64() * 60.0, rng.next_f64() * 60.0];
            let k = 6;

            let mut u32_idx = vec![0u32; k];
            let mut u32_dist = vec![0.0f64; k];
            let u32_found = u32_tree.knn_search(&query, &mut u32_idx, &mut u32_dist);

            let mut u64_idx = vec![0u64; k];
            let mut u64_dist = vec![0.0f64; k];
            let u64_found = u64_tree.knn_search(&query, &mut u64_idx, &mut u64_dist);

            let mut usize_idx = vec![0usize; k];
            let mut usize_dist = vec![0.0f64; k];
            let usize_found = usize_tree.knn_search(&query, &mut usize_idx, &mut usize_dist);

            assert_eq!(u32_found, u64_found);
            assert_eq!(u32_found, usize_found);

            let u32_idx_as_usize: Vec<usize> = u32_idx.iter().map(|&i| i as usize).collect();
            let u64_idx_as_usize: Vec<usize> = u64_idx.iter().map(|&i| i as usize).collect();
            assert_eq!(u32_idx_as_usize, u64_idx_as_usize, "u64 indices must match u32 tree's (as usize)");
            assert_eq!(u32_idx_as_usize, usize_idx, "usize indices must match u32 tree's (as usize)");

            assert_eq!(u32_dist, u64_dist, "u64 tree distances must be bit-equal to u32 tree's");
            assert_eq!(u32_dist, usize_dist, "usize tree distances must be bit-equal to u32 tree's");
        }
    }
}

/// Feature-gating (brief test 6): without the "parallel" feature,
/// `BuildThreads::Auto`/`Threads(_)` must panic with the documented message
/// instead of silently building sequentially or failing to compile. A
/// separate module (rather than a `#[cfg]`'d test inside `mod tests` above)
/// because it needs to exist ONLY when "parallel" is off — `mod tests` above
/// is unconditional.
#[cfg(all(test, not(feature = "parallel")))]
mod no_parallel_tests {
    use super::*;
    use crate::dim::ConstDim;
    use crate::params::BuildThreads;

    #[test]
    #[should_panic(expected = "flannrust was compiled without the 'parallel' feature")]
    fn build_with_auto_threads_panics_without_parallel_feature() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let _ = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice())
            .threads(BuildThreads::Auto)
            .build();
    }

    #[test]
    fn build_with_sequential_threads_still_works_without_parallel_feature() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let tree = KdTreeBuilder::new(ConstDim::<2>, pts.as_slice())
            .threads(BuildThreads::Sequential)
            .build();
        assert_eq!(tree.size(), 3);
    }
}
