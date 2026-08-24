//! Bentley-Saxe dynamic forest: a faithful port of nanoflann's
//! `KDTreeSingleIndexDynamicAdaptor` (nanoflann.hpp:2521-2718) plus the
//! sub-tree class it wraps, `KDTreeSingleIndexDynamicAdaptor_`
//! (nanoflann.hpp:2248-2504, in particular its `buildIndex()` at
//! nanoflann.hpp:2345-2367). This module provides the FOREST BOOKKEEPING —
//! `add_points`/`remove_point`/the merge-and-rebuild schedule — and SEARCH:
//! [`DynamicKdTree::find_neighbors`] (= C++'s forest `findNeighbors`,
//! nanoflann.hpp:2704-2713) plus the additive `knn_search`/`rknn_search`/
//! `radius_search` wrappers, all tombstone-filtered via `TombstoneFilter`.
//!
//! # The forest idea
//!
//! `tree_count` independent static kd-trees ("slots"), indexed 0..tree_count.
//! Each newly-added point walks a binary-counter pattern
//! (`first0bit`) to decide which slot absorbs it: slot `pos` absorbs
//! every LOWER slot's entire current point list plus the new point itself,
//! and every lower slot becomes empty. This is exactly incrementing a
//! binary counter by one (`pos` = position of the counter's lowest unset
//! bit) — after `n` sequential adds with no removals, the set of non-empty
//! slots is exactly the set bits of `n`'s binary representation, each
//! holding `2^slot` points. A slot is REBUILT FROM SCRATCH (the static
//! builder's `SubtreeBuilder`, `base = 0`) every time its point list changes, and
//! — critically — every time ANY slot up to the highest-touched slot
//! changes, per nanoflann.hpp:2670-2675's `for (int i = 0; i <= maxIndex;
//! ++i)` rebuild loop (see [`DynamicKdTree::add_points`]'s doc comment).
//!
//! # Removal is lazy
//!
//! `remove_point` never touches a slot's point list (`vind`, = C++'s
//! `vAcc_`) — it only flips `tree_index[idx]` to `-1` and remembers which
//! slot still physically holds the point in `removed`, so a later
//! `add_points` call covering that same index can restore it in place
//! (`tree_index[idx] = removed[idx]`) instead of inserting a duplicate.
//! Because slot membership doesn't shrink on removal, a later MERGE can
//! move a tombstoned point's physical storage from one slot to another —
//! `add_points`'s merge loop keeps `removed`'s recorded slot current for
//! exactly this reason (nanoflann.hpp:2657-2663's `else removedPoints_[e] =
//! pos;` branch; see [`DynamicKdTree::add_points`]'s doc comment).

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::bbox::{compute_bounding_box_over_indices, Interval};
use crate::data_source::DataSource;
use crate::dim::Dim;
use crate::filter::PointFilter;
use crate::metric::{Distance, L2};
use crate::node::Node;
use crate::params::SearchParams;
use crate::result_set::{
    KeepInsertionOrder, KnnResultSet, RadiusResultSet, ResultItem, ResultSet, RknnResultSet, TieBreak,
};
use crate::scalar::{DistanceValue, IndexType, Scalar};
use crate::search::{find_neighbors as search_find_neighbors, SearchCtx};

/// Position of the least-significant UNSET (zero) bit of `n` — nanoflann's
/// `First0Bit` (nanoflann.hpp:2565-2574, `private` member of
/// `KDTreeSingleIndexDynamicAdaptor`, ported verbatim as a free function
/// since it has no other state dependency):
/// ```cpp
/// int First0Bit(Size num) {
///     int pos = 0;
///     while (num & 1) { num = num >> 1; pos++; }
///     return pos;
/// }
/// ```
/// This is the standard "binary counter increment" slot-selection rule: the
/// bit position that would flip 0->1 if `n` were incremented by one.
pub(crate) fn first0bit(n: usize) -> usize {
    let mut num = n;
    let mut pos = 0usize;
    while num & 1 == 1 {
        num >>= 1;
        pos += 1;
    }
    pos
}

/// One forest slot: an independent static kd-tree over a subset of the
/// dataset's point indices. `vind` (= C++ `vAcc_`) is the slot's OWN
/// permuted point-index list; `nodes` (= C++'s node pool `pool_`, exposed
/// here as the same flat arena M1's static tree uses) and `root_bbox` are
/// rebuilt from scratch by [`SubtreeBuilder`](crate::build::SubtreeBuilder)
/// every time `vind` changes (or, per the C++ rebuild-loop quirk, every
/// time any slot up to the highest slot touched this call changes — see
/// [`DynamicKdTree::add_points`]). An empty slot (`vind.is_empty()`) has an
/// empty `nodes` arena and a stale/never-written `root_bbox` (never read:
/// search always checks `vind.is_empty()` first, mirroring nanoflann's
/// `size(*this) == 0` early-return, nanoflann.hpp:2380).
struct Slot<T: Scalar, D: Dim, Idx: IndexType> {
    vind: Vec<Idx>,
    // NOTE: `nodes`/`root_bbox` don't need `#[allow(dead_code)]` even though
    // no query method reads them until a search actually runs --
    // `add_points`'s rebuild pass already writes AND clears them (`.clear()`,
    // struct-literal init, `slot.root_bbox = bbox`), which is enough for
    // rustc's dead_code analysis to consider them used.
    nodes: Vec<Node<T>>,
    root_bbox: D::Array<Interval<T>>,
}

impl<T: Scalar, D: Dim, Idx: IndexType> Slot<T, D, Idx> {
    fn empty(dim: D) -> Self {
        Slot { vind: Vec::new(), nodes: Vec::new(), root_bbox: dim.filled(Interval::default()) }
    }
}

/// The forest's `isActive` (nanoflann.hpp:2289: `return treeIndex_[idx] !=
/// -1;`), wired into M1's [`PointFilter`] seam so a slot's search skips
/// tombstoned (lazily removed) points, exactly like the C++ dynamic
/// sub-tree adaptor.
///
/// # Safety note (parity deviation)
/// `is_active` indexes `tree_index[idx.to_usize()]` directly, with no
/// bounds check beyond what Rust's slice indexing does automatically. A
/// LEGAL forest never yields an out-of-range `vind` entry here: every
/// accessor a slot's leaf loop offers this filter originates from that
/// slot's own `vind`, itself only ever populated by `add_points` with
/// dataset indices `< tree_index.len()` at the moment they were pushed
/// (`tree_index` is resized to cover every `idx` before it's ever written),
/// and `tree_index` only ever grows, never shrinks or gets reordered. So a
/// panic here can only happen if that forest-wide invariant is somehow
/// violated by a bug elsewhere — unlike C++, where the equivalent
/// out-of-range `treeIndex_[idx]` access would be silent undefined
/// behavior, safe Rust panics loudly instead.
struct TombstoneFilter<'a> {
    tree_index: &'a [i32],
}

impl<'a, Idx: IndexType> PointFilter<Idx> for TombstoneFilter<'a> {
    #[inline]
    fn is_active(&self, idx: Idx) -> bool {
        self.tree_index[idx.to_usize()] != -1
    }
}

/// Configures and builds a [`DynamicKdTree`]. Defaults mirror
/// [`crate::tree::KdTreeBuilder`]: `L2` metric, `u32` indices,
/// insertion-order ties, `leaf_max_size` 10. `maximum_point_count` defaults
/// to `1_000_000_000` (nanoflann's `KDTreeSingleIndexDynamicAdaptor`
/// constructor default, nanoflann.hpp:2608).
///
/// Sequential-only build (parity: the C++ forest ctor has no parallel
/// slot-rebuild path — `n_thread_build` only ever reaches the PER-SLOT
/// `buildIndex()`, and this crate's forest always rebuilds slots one at a
/// time on the calling thread, matching every observed C++ call site). One
/// consequence: unlike [`crate::tree::KdTreeBuilder::build`] (which, under
/// the default `parallel` feature, requires `DataSource: Sync` because it
/// genuinely shares `&DS` across rayon worker threads), [`Self::build`]
/// never needs a `Sync` bound at all — there is no parallel path here to
/// need one, so a non-`Sync` dataset (e.g. `Rc<Cell<_>>`-backed interior
/// mutability) works out of the box, with no `build_sequential`-style
/// escape hatch required.
pub struct DynamicKdTreeBuilder<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder>
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
    maximum_point_count: usize,
    _marker: PhantomData<(T, Idx, TB)>,
}

impl<T: Scalar, D: Dim, DS: DataSource<T>> DynamicKdTreeBuilder<T, D, DS>
where
    L2: Distance<T>,
{
    /// Defaults: `L2` metric, `u32` indices, insertion-order ties,
    /// `leaf_max_size` 10, `maximum_point_count` 1_000_000_000.
    pub fn new(dim: D, dataset: DS) -> Self {
        Self {
            dim,
            dataset,
            metric: L2,
            leaf_max_size: 10,
            maximum_point_count: 1_000_000_000,
            _marker: PhantomData,
        }
    }
}

impl<T, D, DS, M, Idx, TB> DynamicKdTreeBuilder<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Takes a metric INSTANCE, same as [`crate::tree::KdTreeBuilder::with_metric`].
    pub fn with_metric<M2: Distance<T>>(self, metric: M2) -> DynamicKdTreeBuilder<T, D, DS, M2, Idx, TB> {
        DynamicKdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric,
            leaf_max_size: self.leaf_max_size,
            maximum_point_count: self.maximum_point_count,
            _marker: PhantomData,
        }
    }

    /// Panics if `n == 0`. Default 10.
    pub fn leaf_max_size(mut self, n: usize) -> Self {
        assert!(n > 0, "leaf_max_size: n must be > 0");
        self.leaf_max_size = n;
        self
    }

    /// Bounds `tree_count` via `tree_count = floor(log2(maximum_point_count)) + 1`
    /// (nanoflann.hpp:2610, see [`DynamicKdTree::tree_count`]'s doc comment
    /// for the exact cast). Default 1_000_000_000 (30 slots). This is a real
    /// capacity, not a hint: `add_points` panics (naming
    /// `maximum_point_count` in its message) if the number of points ever
    /// added would require a slot beyond `tree_count`, so pass a value big
    /// enough for the largest total point count the forest will ever hold.
    /// Panics if `n == 0` (a zero-capacity forest with no slots at all can
    /// never legally add a point).
    pub fn maximum_point_count(mut self, n: usize) -> Self {
        assert!(n >= 1, "maximum_point_count: n must be >= 1");
        self.maximum_point_count = n;
        self
    }

    /// Selects the point-index type. Default `u32`.
    pub fn index_type<Idx2: IndexType>(self) -> DynamicKdTreeBuilder<T, D, DS, M, Idx2, TB> {
        DynamicKdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric: self.metric,
            leaf_max_size: self.leaf_max_size,
            maximum_point_count: self.maximum_point_count,
            _marker: PhantomData,
        }
    }

    /// Selects the kNN/RKNN equal-distance tie policy (consumed by
    /// [`DynamicKdTree`]'s search methods; stored here so the type parameter
    /// is fixed at build time).
    pub fn tie_break<TB2: TieBreak>(self) -> DynamicKdTreeBuilder<T, D, DS, M, Idx, TB2> {
        DynamicKdTreeBuilder {
            dim: self.dim,
            dataset: self.dataset,
            metric: self.metric,
            leaf_max_size: self.leaf_max_size,
            maximum_point_count: self.maximum_point_count,
            _marker: PhantomData,
        }
    }

    /// Builds the (initially possibly-empty) forest. Mirrors
    /// `KDTreeSingleIndexDynamicAdaptor`'s constructor (nanoflann.hpp:2604-2623):
    /// allocates `tree_count` empty slots, then — ONLY if the dataset already
    /// has points at build time (`dataset.point_count() > 0`) — immediately
    /// calls `add_points(0, point_count() - 1)`, exactly like the C++'s
    /// `if (num_initial_points > 0) addPoints(0, num_initial_points - 1);`
    /// (nanoflann.hpp:2620-2622). An empty dataset yields an empty forest
    /// with zero occupied slots; `add_points` can be called later once the
    /// dataset (if it uses interior mutability) actually grows.
    pub fn build(self) -> DynamicKdTree<T, D, DS, M, Idx, TB> {
        let DynamicKdTreeBuilder { dim, dataset, metric, leaf_max_size, maximum_point_count, .. } = self;

        // nanoflann.hpp:2610: `static_cast<size_t>(std::log2(maximumPointCount)) + 1`.
        // Rust's `as usize` on f64 saturates (0 for negative/NaN/-inf) rather
        // than C++'s UB-on-negative-cast, but is bit-identical for every
        // `maximum_point_count >= 1` value this crate's tests or any sane
        // caller would pass.
        let tree_count = ((maximum_point_count as f64).log2() as usize) + 1;

        let slots: Vec<Slot<T, D, Idx>> = (0..tree_count).map(|_| Slot::empty(dim)).collect();

        let mut tree = DynamicKdTree {
            dataset,
            metric,
            dim,
            leaf_max_size,
            slots,
            tree_index: Vec::new(),
            removed: HashMap::new(),
            point_count: 0,
            _marker: PhantomData,
        };

        let n = tree.dataset.point_count();
        if n > 0 {
            tree.add_points(0, n - 1);
        }

        tree
    }
}

/// The dynamic (Bentley-Saxe) forest: `tree_count` independent static
/// kd-tree slots plus the bookkeeping (`tree_index`/`removed`/
/// `point_count`) that decides which slot each dataset point currently
/// lives in. See the module doc for the overall scheme.
///
/// # API symmetry with the static [`crate::tree::KdTree`]
///
/// This type deliberately does NOT offer a `size()`/`used_memory_bytes()`-
/// style accessor, even though [`crate::tree::KdTree`] has both: for a
/// forest, "size" is ambiguous between "points ever added" (`point_count`,
/// the C++ `pointCount_` counter, which removal never decrements) and
/// "currently live points" ([`Self::active_count`], additive over C++). The
/// C++ forest itself picks neither consistently across its own surface, so
/// rather than port that ambiguity, this crate exposes only the
/// unambiguous [`Self::active_count`]/[`Self::tree_index`]/
/// [`Self::removed_len`] accessors and leaves a size-parity decision (which
/// semantics, if any, `size()` should report) to a future roadmap item
/// rather than guessing now.
///
/// # Example
///
/// ```
/// use flannrust::{ConstDim, DynamicKdTreeBuilder};
///
/// let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]];
/// let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).build();
///
/// let mut idx = [0u32; 1];
/// let mut dist = [0.0f64; 1];
/// let found = tree.knn_search(&[0.1, 0.1], &mut idx, &mut dist);
/// assert_eq!(found, 1);
/// assert_eq!(idx, [0]); // nearest is [0.0, 0.0]
///
/// assert!(tree.remove_point(0));
/// let found_after = tree.knn_search(&[0.1, 0.1], &mut idx, &mut dist);
/// assert_eq!(found_after, 1);
/// assert_eq!(idx, [1]); // [0.0, 0.0] is gone; nearest is now [5.0, 5.0]
/// ```
pub struct DynamicKdTree<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder>
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
    leaf_max_size: usize,
    slots: Vec<Slot<T, D, Idx>>,
    /// `tree_index[i]` = the slot dataset-index `i` currently lives in, or
    /// `-1` if `i` is currently removed. Grows monotonically (never
    /// shrinks) as `add_points` processes genuinely new indices — matches
    /// C++'s `treeIndex_.resize(pointCount_ + 1)` resize-on-demand
    /// (nanoflann.hpp:2644-2645): its length always equals `point_count`
    /// after any `add_points` call, i.e. it tracks "points EVER added",
    /// not `dataset.point_count()`.
    tree_index: Vec<i32>,
    /// dataset-index -> the slot that STILL PHYSICALLY HOLDS a currently-removed
    /// point (lazy deletion never touches a slot's `vind`). Reactivating that
    /// index later restores `tree_index[idx]` from here instead of inserting
    /// a duplicate. A merge can migrate an entry's value (see `add_points`'s
    /// doc comment) without ever removing/re-inserting the key.
    removed: HashMap<usize, i32>,
    /// C++'s `pointCount_`: a monotonically-increasing counter of how many
    /// DISTINCT dataset indices have ever been passed to `add_points` as a
    /// genuinely-new (non-reactivation) point. Drives `first0bit` slot
    /// selection. Confirmed by reading `removePoint` (nanoflann.hpp:2678-2685):
    /// it touches ONLY `removedPoints_`/`treeIndex_`, never `pointCount_` —
    /// so removal does NOT decrement this counter, ever. A point removed and
    /// never re-added still occupies its `first0bit` "slot budget" forever.
    point_count: usize,
    _marker: PhantomData<TB>,
}

impl<T, D, DS, M, Idx, TB> DynamicKdTree<T, D, DS, M, Idx, TB>
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T>,
    M: Distance<T>,
    Idx: IndexType,
    TB: TieBreak,
{
    /// Number of forest slots (= C++ `treeCount_`, fixed for the forest's
    /// lifetime at `floor(log2(maximum_point_count)) + 1`,
    /// nanoflann.hpp:2610). Default `maximum_point_count` 1_000_000_000
    /// gives 30.
    pub fn tree_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of currently-live points: dataset indices ever added minus
    /// those currently removed (`tree_index[i] != -1` among the
    /// `point_count` indices ever processed). Not a C++ method (nanoflann's
    /// forest has no such accessor) — additive, for tests/introspection.
    pub fn active_count(&self) -> usize {
        self.tree_index.iter().filter(|&&v| v != -1).count()
    }

    /// A slot's own point-index list (= that sub-tree's `vAcc_`) in its
    /// CURRENT build-time permuted order — for cross-validation against the
    /// C++ oracle's `nfrd_slot_vacc_*`. Panics if `slot >= tree_count()`.
    pub fn point_indices_of_slot(&self, slot: usize) -> &[Idx] {
        &self.slots[slot].vind
    }

    /// `tree_index[i]` = the slot dataset-index `i` lives in, `-1` if
    /// removed. Length = `point_count` (points ever added), NOT
    /// `dataset.point_count()` — see the field's doc comment. For
    /// cross-validation against `nfrd_tree_index_*`.
    pub fn tree_index(&self) -> &[i32] {
        &self.tree_index
    }

    /// Number of currently-removed (tombstoned) points.
    pub fn removed_len(&self) -> usize {
        self.removed.len()
    }

    /// Add every dataset index in `[start, end_inclusive]` (END-INCLUSIVE,
    /// matching C++'s `addPoints(IndexType start, IndexType end)` with its
    /// `idx <= end` loop condition, nanoflann.hpp:2629-2676). A `start >
    /// end_inclusive` range is a legal no-op over the per-index loop (same
    /// as C++, where the `for` loop simply never executes) — but see below,
    /// the trailing rebuild pass still always touches slot 0.
    ///
    /// # Contiguous-append contract (DEVIATION from C++)
    ///
    /// nanoflann's `addPoints` writes `treeIndex_[pointCount_] = pos` —
    /// indexed by the running `pointCount_` COUNTER, not by the real point
    /// index `idx` being processed (nanoflann.hpp:2644-2648). For this to
    /// correctly record `idx`'s slot, `idx` must equal `pointCount_` at the
    /// moment it is processed, for every GENUINELY NEW index (one not found
    /// in `removed`/`removedPoints_`) in the call. In the common case (a
    /// pure-growth call adding brand-new points), this reduces to: `start`
    /// must equal `point_count` at call time.
    ///
    /// Reactivating a previously-removed index is EXEMPT — nanoflann's
    /// reactivation branch (`continue`s immediately, nanoflann.hpp:2639-2643)
    /// never touches `pointCount_` at all, so a reactivation-only call can
    /// legally use ANY `start`/`end_inclusive`, regardless of the current
    /// `point_count`. This is real, oracle-verified C++ behavior — see
    /// `nanoflann-ref`'s `readd_point_1_reactivates_it_f32`, which calls
    /// `add_points(1, 1)` to reactivate index 1 while `pointCount_` is
    /// already 4. A stricter BLANKET `assert_eq!(start, self.point_count)`
    /// at the top of this method would PANIC on exactly that legal,
    /// oracle-verified sequence — reactivation-only calls routinely start
    /// well below the current `point_count`, and that is not a bug. The
    /// per-index invariant actually implemented below asserts only against
    /// GENUINELY-NEW indices, which is precisely the case nanoflann's own
    /// bookkeeping silently corrupts on misalignment — narrower than a
    /// blanket check, but correct for every legal call shape.
    ///
    /// So: where C++ silently corrupts `treeIndex_` when a genuinely-new
    /// index doesn't line up with `pointCount_` (see `nanoflann-ref`'s
    /// `add_points_misaligned_start_documents_silent_corruption_f32`
    /// regression test for the observed symptom), this method instead
    /// PANICS the instant that misalignment would occur — asserted
    /// PER-INDEX, only against genuinely-new (non-reactivation) indices.
    /// This is a documented, strictly-safer deviation: behavior is
    /// IDENTICAL to C++ for every legal call sequence (including mixed
    /// reactivation-then-growth calls, as long as the growth portion is
    /// itself contiguous from `point_count`), and panics instead of
    /// silently corrupting bookkeeping for every illegal one.
    ///
    /// # Per-point loop (nanoflann.hpp:2629-2676)
    ///
    /// For each `idx` in `start..=end_inclusive`, in order:
    /// 1. **Reactivation short-circuit**: if `idx` is a key in `removed`,
    ///    restore `tree_index[idx] = removed[idx]`, remove the key, and move
    ///    to the next `idx` — the point's old slot still physically holds it,
    ///    so slots are untouched.
    /// 2. Otherwise (genuinely new — see the contiguity contract above):
    ///    `pos = first0bit(point_count)`; every LOWER slot `0..pos`'s ENTIRE
    ///    point list is merged into slot `pos` (append order: slots
    ///    ascending, within a slot in its current `vind` order) and cleared;
    ///    during the merge, each moved dataset index `e` has
    ///    `tree_index[e] = pos` if it's currently live, or — critically —
    ///    `removed[e] = pos` if it's currently a TOMBSTONE (the tombstone's
    ///    recorded slot MIGRATES to track where its point physically ended
    ///    up, nanoflann.hpp:2657-2663). `idx` itself is then pushed onto
    ///    slot `pos`'s list and `point_count` increments.
    ///
    /// # Rebuild pass (nanoflann.hpp:2670-2675)
    ///
    /// After the per-point loop, EVERY slot `0..=max_index` (where
    /// `max_index` is the highest `pos` reached by any genuinely-new index
    /// this call, defaulting to 0 even if the call did nothing at all — see
    /// below) is rebuilt: its arena is freed, and if its `vind` is
    /// non-empty, `crate::build::SubtreeBuilder` rebuilds it FROM THE
    /// SLOT'S OWN CURRENT `vind` ORDER (not reset to identity — mirrors
    /// `KDTreeSingleIndexDynamicAdaptor_::buildIndex()`,
    /// nanoflann.hpp:2345-2367, which calls `computeBoundingBox`/
    /// `divideTree` directly over whatever `vAcc_` currently holds). This
    /// INCLUDES slots whose contents didn't change this call — because
    /// `max_index` starts at 0 regardless of what happened in the per-point
    /// loop, slot 0 is unconditionally freed-and-maybe-rebuilt on EVERY
    /// `add_points` call, even a no-op one (empty range, or a call that
    /// only reactivated points). Slot-level `vind` parity with the C++
    /// oracle therefore depends on replicating this exact rebuild
    /// SCHEDULE, not just tracking final slot membership.
    pub fn add_points(&mut self, start: usize, end_inclusive: usize) {
        let dim_n = self.dim.dim();
        let leaf_max_size = self.leaf_max_size;
        let mut max_index: usize = 0;

        for idx in start..=end_inclusive {
            if let Some(&slot) = self.removed.get(&idx) {
                self.tree_index[idx] = slot;
                self.removed.remove(&idx);
                continue;
            }

            assert_eq!(
                idx,
                self.point_count,
                "add_points: index {idx} is a genuinely-new point but does not equal \
                 point_count ({}) -- nanoflann's addPoints contract requires brand-new \
                 indices to be a contiguous append starting exactly at point_count \
                 (treeIndex_ is indexed by the running pointCount_ counter, not by idx); \
                 see DynamicKdTree::add_points's doc comment",
                self.point_count
            );

            let pos = first0bit(self.point_count);
            assert!(
                pos < self.slots.len(),
                "add_points: point_count ({}) has outgrown this forest's capacity ({} slots) \
                 -- construct the forest with a larger DynamicKdTreeBuilder::maximum_point_count",
                self.point_count,
                self.slots.len()
            );
            if pos > max_index {
                max_index = pos;
            }

            if self.tree_index.len() <= self.point_count {
                self.tree_index.resize(self.point_count + 1, -1);
            }
            self.tree_index[self.point_count] = pos as i32;

            for i in 0..pos {
                let entries = std::mem::take(&mut self.slots[i].vind);
                for e in entries {
                    self.slots[pos].vind.push(e);
                    let e_usize = e.to_usize();
                    if self.tree_index[e_usize] != -1 {
                        self.tree_index[e_usize] = pos as i32;
                    } else {
                        self.removed.insert(e_usize, pos as i32);
                    }
                }
            }

            self.slots[pos].vind.push(Idx::from_usize(idx));
            self.point_count += 1;
        }

        for i in 0..=max_index {
            self.slots[i].nodes.clear();
            if self.slots[i].vind.is_empty() {
                continue;
            }

            let mut bbox = self.dim.filled(Interval::default());
            compute_bounding_box_over_indices(&self.dataset, dim_n, &self.slots[i].vind, bbox.as_mut());

            let slot = &mut self.slots[i];
            {
                let mut builder = crate::build::SubtreeBuilder {
                    ds: &self.dataset,
                    dim: dim_n,
                    leaf_max_size,
                    base: 0,
                    vind: &mut slot.vind,
                    arena: &mut slot.nodes,
                };
                builder.build(bbox.as_mut());
            }
            slot.root_bbox = bbox;
        }
    }

    /// Lazily remove `idx` (= C++ `removePoint`, nanoflann.hpp:2678-2685):
    /// ```cpp
    /// void removePoint(size_t idx) {
    ///     if (idx >= pointCount_) return;
    ///     if (treeIndex_[idx] == -1) return;  // already removed
    ///     removedPoints_[idx] = treeIndex_[idx];
    ///     treeIndex_[idx] = -1;
    /// }
    /// ```
    /// Returns `false` (no-op) if `idx >= point_count` (never added) or
    /// `idx` is already removed; `true` otherwise. Never touches a slot's
    /// `vind` — the point stays physically present until a rebuild happens
    /// to touch that slot for an unrelated reason. `point_count` is NOT
    /// decremented (see this struct's `point_count` field doc comment) —
    /// ported exactly, matching the C++ source, which never references
    /// `pointCount_` anywhere in `removePoint`.
    pub fn remove_point(&mut self, idx: usize) -> bool {
        if idx >= self.point_count {
            return false;
        }
        if self.tree_index[idx] == -1 {
            return false;
        }
        self.removed.insert(idx, self.tree_index[idx]);
        self.tree_index[idx] = -1;
        true
    }

    /// = C++ forest `findNeighbors` (nanoflann.hpp:2704-2713): calls every
    /// slot's FULL sub-tree `findNeighbors` (nanoflann.hpp:2400-2417) in
    /// turn, slots in ASCENDING order, against the SAME shared `result`,
    /// tombstone-filtered via `TombstoneFilter`. Reuses the static builder's
    /// `search::find_neighbors` verbatim per slot — that function already
    /// re-zeroes its `dists_scratch` argument at the top of every call
    /// (checked against both the vendored C++, whose sub-tree
    /// `findNeighbors` constructs a fresh zeroed `distance_vector_t` on
    /// EVERY call, nanoflann.hpp:2409-2411, and M1's actual
    /// `search::find_neighbors` body, `for d in dists_scratch.iter_mut() {
    /// *d = ZERO; }` at its top) — so the single `scratch` buffer below is
    /// safely reused across every slot pass without any extra re-zeroing
    /// here; each pass still independently computes its own initial
    /// distances from ITS OWN slot's root bbox, honors `params.eps`, and
    /// (a real C++ quirk, mirrored exactly, not an optimization
    /// opportunity to skip) calls `result.sort()` again if `params.sorted`
    /// — so with `sorted: true` and more than one non-empty slot, the
    /// result is re-sorted after EVERY slot pass, which is harmless
    /// (idempotent on an already-sorted set) but is the literal C++
    /// behavior. Each slot's own `bool` return is discarded, exactly like
    /// the C++ forest loop discards `index_[i].findNeighbors(...)`'s
    /// return value; only the FINAL `result.full()` is returned.
    ///
    /// **Empty-forest quirk (inherited from nanoflann; M1 Corrections #4):**
    /// with zero occupied slots, every slot pass hits the size-0 early
    /// return inside `search::find_neighbors` (`ctx.nodes.is_empty()`)
    /// without ever touching `result` — so the final `result.full()`
    /// reflects whatever an UNTOUCHED result set naturally reports:
    /// `false` for [`KnnResultSet`]/[`RknnResultSet`] (count `0 <`
    /// capacity), but **`true`** for [`RadiusResultSet`] (its `full()` is
    /// hardwired `true`, nanoflann.hpp:433, regardless of whether anything
    /// was ever added). So calling `find_neighbors` directly with an empty
    /// `RadiusResultSet` on an empty (or all-slots-empty) forest returns
    /// `true` with ZERO items — this is the literal C++ contract, not a
    /// bug. The quirk is only observable through this generic escape
    /// hatch: the additive [`Self::radius_search`]/[`Self::radius_search_with`]
    /// wrappers below return the found COUNT (`0` in that case), where the
    /// quirk is invisible to callers who only use the wrapper API.
    ///
    /// No forest-level box search exists in the vendored C++ (only the
    /// per-slot static class has `findWithinBox`, and it is never exposed
    /// through the dynamic forest adaptor) — this port does not add one
    /// either, matching upstream's surface exactly.
    pub fn find_neighbors<R: ResultSet<M::DistanceType, Idx>>(
        &self,
        result: &mut R,
        query: &[T],
        params: &SearchParams,
    ) -> bool {
        let filter = TombstoneFilter { tree_index: &self.tree_index };
        let mut scratch = self.dim.filled(M::DistanceType::ZERO);

        for slot in &self.slots {
            let ctx = SearchCtx {
                ds: &self.dataset,
                metric: &self.metric,
                dim: self.dim,
                nodes: &slot.nodes,
                vind: &slot.vind,
                root_bbox: slot.root_bbox.as_ref(),
            };
            let _ = search_find_neighbors(&ctx, result, query, params, &filter, scratch.as_mut());
        }

        result.full()
    }

    /// `k` = out slices' length (must be equal; panics otherwise). Returns
    /// the found count. `eps = 0`, sorted — like M1's `KdTree::knn_search`.
    /// ADDITIVE over C++ (the dynamic forest adaptor exposes only
    /// `findNeighbors`; this wraps it with M1's naming/ergonomics, tombstone
    /// filtering included automatically via [`Self::find_neighbors`]).
    pub fn knn_search(&self, query: &[T], out_indices: &mut [Idx], out_dists: &mut [M::DistanceType]) -> usize {
        self.knn_search_with(query, out_indices, out_dists, &SearchParams::default())
    }

    /// Same as [`Self::knn_search`] with explicit params.
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
        let mut rs = KnnResultSet::<M::DistanceType, Idx, TB>::new(out_indices, out_dists);
        self.find_neighbors(&mut rs, query, params);
        rs.size()
    }

    /// ADDITIVE (see [`Self::knn_search`]). `radius` is in the metric's
    /// native scale (SQUARED for L2-family).
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
        let mut rs = RknnResultSet::<M::DistanceType, Idx, TB>::new(out_indices, out_dists, radius);
        self.find_neighbors(&mut rs, query, params);
        rs.size()
    }

    /// ADDITIVE (see [`Self::knn_search`]). Clears `out`. STRICTLY `dist <
    /// radius`. `eps = 0`, sorted.
    pub fn radius_search(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out: &mut Vec<ResultItem<Idx, M::DistanceType>>,
    ) -> usize {
        self.radius_search_with(query, radius, out, &SearchParams::default())
    }

    /// Same as [`Self::radius_search`] with explicit params. Returns the
    /// found COUNT — the empty-forest quirk documented on
    /// [`Self::find_neighbors`] is invisible here (`rs.size()` is `0` either
    /// way for an untouched result set).
    pub fn radius_search_with(
        &self,
        query: &[T],
        radius: M::DistanceType,
        out: &mut Vec<ResultItem<Idx, M::DistanceType>>,
        params: &SearchParams,
    ) -> usize {
        let mut rs = RadiusResultSet::new(radius, out);
        self.find_neighbors(&mut rs, query, params);
        rs.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::ConstDim;

    /// A dataset with real backing coordinates but a permanently-zero
    /// reported `point_count()` -- lets a test drive every `add_points`
    /// call itself, explicitly, exactly as a real caller of the dynamic
    /// forest is expected to (see `add_points`'s contiguous-append
    /// contract doc comment). Using a plain `&[[T; N]]` slice instead
    /// would report its true (non-zero) length immediately, so
    /// `DynamicKdTreeBuilder::build`'s ctor auto-add
    /// (`if n > 0 { add_points(0, n - 1) }`) would eagerly add every point
    /// up front -- correct C++-mirroring behavior in its own right (see
    /// test 8 below, which exercises exactly that), but it would make any
    /// FOLLOW-UP `add_points` call in a hand-traced test immediately
    /// misaligned against `point_count`, which is not what these
    /// finer-grained bookkeeping tests are exercising.
    struct Ungated<const N: usize>(Vec<[f64; N]>);

    impl<const N: usize> DataSource<f64> for Ungated<N> {
        fn point_count(&self) -> usize {
            0
        }
        fn point_component(&self, idx: usize, dim: usize) -> f64 {
            self.0[idx][dim]
        }
    }

    // ---------------------------------------------------------------
    // Test 1: first0bit vector
    // ---------------------------------------------------------------

    /// Hand-derived from the C++ algorithm (`while (num & 1) { num >>= 1;
    /// pos++; }`), i.e. "position of the least-significant unset bit":
    /// - 0  (0b0000)    -> bit0 is 0            -> 0
    /// - 1  (0b0001)    -> bit0=1,bit1=0         -> 1
    /// - 2  (0b0010)    -> bit0=0                -> 0
    /// - 3  (0b0011)    -> bit0=1,bit1=1,bit2=0  -> 2
    /// - 4  (0b0100)    -> bit0=0                -> 0
    /// - 5  (0b0101)    -> bit0=1,bit1=0          -> 1
    /// - 6  (0b0110)    -> bit0=0                -> 0
    /// - 7  (0b0111)    -> bit0=1,bit1=1,bit2=1,bit3=0 -> 3
    /// - 8  (0b1000)    -> bit0=0                -> 0
    /// - 15 (0b1111)    -> four set bits, bit4=0 -> 4
    #[test]
    fn first0bit_matches_hand_derived_vector() {
        assert_eq!(first0bit(0), 0);
        assert_eq!(first0bit(1), 1);
        assert_eq!(first0bit(2), 0);
        assert_eq!(first0bit(3), 2);
        assert_eq!(first0bit(4), 0);
        assert_eq!(first0bit(5), 1);
        assert_eq!(first0bit(6), 0);
        assert_eq!(first0bit(7), 3);
        assert_eq!(first0bit(8), 0);
        assert_eq!(first0bit(15), 4);
    }

    // ---------------------------------------------------------------
    // Test 2: Add 4 points one batch -> First0Bit-derived slot occupancy
    // ---------------------------------------------------------------

    /// Hand-derived First0Bit sequence for point_count 0,1,2,3 (before each
    /// point is placed): first0bit(0)=0, first0bit(1)=1, first0bit(2)=0,
    /// first0bit(3)=2. Tracing the merge rule (see `add_points`'s doc
    /// comment), including the exact APPEND ORDER (lower slots ascending,
    /// within a slot in its current `vind` order, then the new idx last):
    /// - idx=0: pos=0 -> slot0.vind = [0].
    /// - idx=1: pos=1 -> merge slot0 (`[0]`) into slot1, then push 1 ->
    ///   slot1.vind = [0, 1]; slot0 cleared.
    /// - idx=2: pos=0 -> slot0.vind = [2].
    /// - idx=3: pos=2 -> merge slot0 (`[2]`) THEN slot1 (`[0, 1]`) into
    ///   slot2 (ascending slot order: slot0 before slot1), then push 3 ->
    ///   slot2.vind = [2, 0, 1, 3]; slot0/slot1 cleared.
    ///
    /// With `leaf_max_size` 10 (the default) and only 4 points, the
    /// rebuild pass's `SubtreeBuilder::build` never calls `middle_split`/
    /// `plane_split` at all (`count(4) <= leaf_max_size(10)` makes it an
    /// immediate single leaf, nanoflann.hpp:1335 / `build.rs`'s `if count
    /// <= self.leaf_max_size` branch) — the leaf's `vind` slice is used
    /// AS-IS, unpermuted. So the exact post-rebuild slot 2 order is the
    /// merge-append order itself: `[2, 0, 1, 3]`. (A rebuild that DOES
    /// permute is covered separately below by
    /// `permuting_rebuild_with_leaf_max_size_one_matches_hand_derived_sort`.)
    #[test]
    fn add_4_points_one_batch_matches_first0bit_slot_occupancy() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();

        tree.add_points(0, 3);

        assert!(tree.point_indices_of_slot(0).is_empty(), "slot 0 must be empty");
        assert!(tree.point_indices_of_slot(1).is_empty(), "slot 1 must be empty");

        // EXACT append order (not just membership) -- this is the signal
        // T4's per-slot cross-validation against the C++ oracle's vAcc_
        // depends on; a single-leaf rebuild (count <= leaf_max_size) never
        // permutes, so this must equal the raw merge-append order exactly.
        assert_eq!(
            tree.point_indices_of_slot(2),
            &[2u32, 0, 1, 3],
            "slot 2 must hold the EXACT merge-append order, not just the right membership"
        );

        assert_eq!(tree.tree_index(), &[2, 2, 2, 2]);

        // Separate membership-only check (order-independent), kept
        // alongside the exact-order assertion above rather than instead of
        // it.
        let mut union: Vec<u32> = (0..tree.tree_count())
            .flat_map(|s| tree.point_indices_of_slot(s).iter().copied())
            .collect();
        union.sort_unstable();
        assert_eq!(union, vec![0, 1, 2, 3]);
    }

    // ---------------------------------------------------------------
    // Test 3: Merge rule migrates a tombstone (the `else` branch)
    // ---------------------------------------------------------------

    /// Sequence (hand-traced against nanoflann.hpp:2629-2676):
    /// - `add_points(0, 1)`: point 0 -> slot 0 (pos=first0bit(0)=0); point 1
    ///   -> slot 1 (pos=first0bit(1)=1), absorbing slot 0's [0] ->
    ///   `tree_index[0] = 1`. `leaf_max_size` 10 (default) >= count(2), so
    ///   this call's rebuild is a single leaf: slot1.vind stays exactly
    ///   `[0, 1]` (append order, unpermuted -- same "count <=
    ///   leaf_max_size never permutes" reasoning as the previous test).
    /// - `remove_point(0)`: `removed = {0: 1}`, `tree_index[0] = -1`.
    /// - `add_points(2, 2)`: point 2 -> slot 0 (pos=first0bit(2)=0);
    ///   slot0.vind = `[2]`.
    /// - `add_points(3, 3)`: point 3 -> slot 2 (pos=first0bit(3)=2),
    ///   merging slot0 (`[2]`) THEN slot1 (`[0, 1]`, exactly the order the
    ///   PRIOR single-leaf rebuild left it in) into slot2, ascending slot
    ///   order, within-slot vind order preserved, then pushing 3 last:
    ///   slot2.vind = `[2, 0, 1, 3]`. During the slot1 merge, the entry
    ///   that is index 0 (`tree_index[0] == -1`, a tombstone) takes the
    ///   `else` branch -- `removed[0]` MIGRATES from `1` to `2`. The entry
    ///   that is index 1 (live) takes the `if` branch: `tree_index[1] =
    ///   2`. `count(4) <= leaf_max_size(10)` again -> single leaf, no
    ///   permutation -> slot2.vind stays exactly `[2, 0, 1, 3]`.
    #[test]
    fn merge_migrates_a_tombstones_recorded_slot() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();

        tree.add_points(0, 1);
        assert!(tree.remove_point(0));
        tree.add_points(2, 2);
        tree.add_points(3, 3);

        // tree_index[0] must STILL read -1 (removal semantics preserved
        // across the merge -- the point never became "live" again).
        assert_eq!(tree.tree_index()[0], -1, "removed point must stay -1 through a merge");
        assert_eq!(tree.removed_len(), 1);

        // Point 1 (live) must have migrated to slot 2 along with everyone else.
        assert_eq!(tree.tree_index()[1], 2);
        assert_eq!(tree.tree_index()[2], 2);
        assert_eq!(tree.tree_index()[3], 2);

        // EXACT append order (including the tombstoned 0 -- lazy deletion
        // never removes from `vind`): [2, 0, 1, 3], per the hand-derivation
        // above.
        assert_eq!(
            tree.point_indices_of_slot(2),
            &[2u32, 0, 1, 3],
            "slot 2 must hold the EXACT merge-append order"
        );

        // Separate membership-only check.
        let mut slot2_sorted: Vec<u32> = tree.point_indices_of_slot(2).to_vec();
        slot2_sorted.sort_unstable();
        assert_eq!(slot2_sorted, vec![0, 1, 2, 3]);
    }

    // ---------------------------------------------------------------
    // Test 3b: a rebuild that actually PERMUTES (parity item 5's whole
    // point) -- `leaf_max_size(1)` forces `SubtreeBuilder` to run real
    // `middle_split`/`plane_split` splits over a merged 8-point slot,
    // instead of the previous two tests' single-leaf (count <=
    // leaf_max_size) pass-through.
    // ---------------------------------------------------------------

    /// dim=2 dataset, dim1 held CONSTANT (100.0) for every point so it can
    /// never be a `middle_split` candidate axis (its bbox span is always
    /// exactly 0, unconditionally below `threshold = (1-EPS)*max_span` at
    /// every recursion level, since `max_span` is dim0's genuinely positive
    /// span) -- this pins the split axis to dim0 at EVERY level, without
    /// needing to hand-trace candidate-axis selection at each step, while
    /// still genuinely exercising `dim >= 2` (a real 2-D dataset, not a
    /// 1-D one wearing a `ConstDim::<2>` label).
    ///
    /// dim0 values (index -> value): 0->70, 1->10, 2->60, 3->20, 4->50,
    /// 5->30, 6->80, 7->40 -- all distinct, chosen so the FINAL fully-sorted
    /// order (see below) is neither ascending-by-index `[0..7]` nor the raw
    /// merge-append order `[6,4,5,2,0,1,3,7]` derived next.
    ///
    /// # Step 1: merge-append order (bookkeeping only, no splits yet)
    ///
    /// Single call `add_points(0, 7)` (first0bit sequence for point_count
    /// 0..7: 0,1,0,2,0,1,0,3 -- same binary-counter pattern as tests 2/3,
    /// extended two more "carries"):
    /// - idx0: pos=0 -> slot0=[0].
    /// - idx1: pos=1 -> merge slot0([0]) into slot1, push 1 -> slot1=[0,1]; slot0=[].
    /// - idx2: pos=0 -> slot0=[2].
    /// - idx3: pos=2 -> merge slot0([2]) then slot1([0,1]) into slot2, push 3
    ///   -> slot2=[2,0,1,3]; slot0=[],slot1=[].
    /// - idx4: pos=0 -> slot0=[4].
    /// - idx5: pos=1 -> merge slot0([4]) into slot1, push 5 -> slot1=[4,5]; slot0=[].
    /// - idx6: pos=0 -> slot0=[6].
    /// - idx7: pos=3 -> merge slot0([6]) then slot1([4,5]) then slot2([2,0,1,3])
    ///   into slot3 (ascending slot order 0,1,2), push 7 ->
    ///   slot3 = [6, 4, 5, 2, 0, 1, 3, 7]; slots 0,1,2 = [].
    ///
    /// Because this is ONE `add_points` call, the per-index bookkeeping
    /// loop runs to completion for all 8 indices BEFORE the rebuild pass
    /// runs even once (see `add_points`'s doc comment) -- so slot1/slot2's
    /// TRANSIENT intermediate contents above (e.g. slot2's `[2,0,1,3]`
    /// after idx3) are never independently rebuilt; only slot3's FINAL
    /// content is ever fed to `SubtreeBuilder`, with `max_index = 3`
    /// (slots 0..3 freed, only slot3 non-empty).
    ///
    /// # Step 2: `SubtreeBuilder::build` over slot3 = `[6,4,5,2,0,1,3,7]`,
    /// `leaf_max_size = 1` (dim0 values by position: 80,50,30,60,70,10,20,40)
    ///
    /// bbox: dim0=`[10,80]` (min idx1=10, max idx6=80), dim1=`[100,100]`.
    /// `max_span = 70`, `threshold ~= 69.9993`; dim1's span (0) is below
    /// threshold (never a candidate); dim0 is the ONLY candidate ->
    /// `cutfeat=0`. `cutval = clamp((10+80)/2=45, [10,80]) = 45`.
    ///
    /// `plane_split(cutfeat=0, cutval=45)` over `[6,4,5,2,0,1,3,7]`
    /// (values 80,50,30,60,70,10,20,40), traced swap-for-swap exactly like
    /// `build.rs`'s `plane_split_exact_permutation_hand_simulated`:
    /// ```text
    /// left=0 mid=0 right=7
    /// mid=0: v(6)=80 > 45      -> swap(0,7); right=6        [7,4,5,2,0,1,3,6]
    /// mid=0: v(7)=40 < 45      -> swap(0,0); left=1; mid=1  (no-op)
    /// mid=1: v(4)=50 > 45      -> swap(1,6); right=5        [7,3,5,2,0,1,4,6]
    /// mid=1: v(3)=20 < 45      -> swap(1,1); left=2; mid=2  (no-op)
    /// mid=2: v(5)=30 < 45      -> swap(2,2); left=3; mid=3  (no-op)
    /// mid=3: v(2)=60 > 45      -> swap(3,5); right=4        [7,3,5,1,0,2,4,6]
    /// mid=3: v(1)=10 < 45      -> swap(3,3); left=4; mid=4  (no-op)
    /// mid=4: v(0)=70 > 45      -> swap(4,4); right=3        (no-op)
    /// mid(4) <= right(3)? no -> loop ends
    /// ```
    /// Final array `[7,3,5,1,0,2,4,6]`; `lim1=left=4`, `lim2=mid=4`
    /// (no equal-to-cutval middle band). `count=8`, `half=4`: `lim1(4) >
    /// half(4)`? no. `lim2(4) < half(4)`? no. `index = half = 4`. LEFT =
    /// `[7,3,5,1]` (values 40,20,30,10, all < 45); RIGHT = `[0,2,4,6]`
    /// (values 70,60,50,80, all > 45).
    ///
    /// # Step 3: recurse (one more level shown concretely; the rest follow
    /// by the same argument)
    ///
    /// LEFT half `[7,3,5,1]` (values 40,20,30,10): bbox dim0=`[10,45]`
    /// (inherited loose upper bound from the parent clip), `cutval =
    /// clamp((10+45)/2=27.5, [10,40]) = 27.5`. `plane_split`:
    /// ```text
    /// left=0 mid=0 right=3
    /// mid=0: v(7)=40 > 27.5 -> swap(0,3); right=2   [1,3,5,7]
    /// mid=0: v(1)=10 < 27.5 -> swap(0,0); left=1; mid=1  (no-op)
    /// mid=1: v(3)=20 < 27.5 -> swap(1,1); left=2; mid=2  (no-op)
    /// mid=2: v(5)=30 > 27.5 -> swap(2,2); right=1   (no-op)
    /// mid(2) <= right(1)? no -> loop ends
    /// ```
    /// `[1,3,5,7]`, `lim1=lim2=2=half` -> LEFT-LEFT=`[1,3]` (10,20),
    /// LEFT-RIGHT=`[5,7]` (30,40). With `leaf_max_size=1`, each 2-element
    /// pair takes exactly one more identical split (single candidate axis,
    /// two distinct values, cutval strictly between them) into two
    /// singleton leaves in ascending order: `[1,3]` -> `[1],[3]`; `[5,7]`
    /// -> `[5],[7]`. LEFT subtree's final leaf order: `[1, 3, 5, 7]`
    /// (ascending by dim0 value: 10,20,30,40) -- matching LEFT half's
    /// values sorted ascending, exactly.
    ///
    /// By the identical argument (single always-candidate axis + all
    /// distinct values -> a spatial-median 3-way partition with a strict
    /// `<`/`>` boundary and no possible cross-contamination between sides,
    /// applied recursively down to singleton leaves, is exactly a
    /// comparison sort by that axis -- this is what steps 2-3 verify
    /// concretely for 8 and 4 elements), RIGHT half `[0,2,4,6]` (values
    /// 70,60,50,80) sorts ascending to `[4, 2, 0, 6]` (50,60,70,80).
    ///
    /// # Final derived vind
    ///
    /// `slot3.vind = [1, 3, 5, 7, 4, 2, 0, 6]` -- differs from BOTH the
    /// merge-append order `[6,4,5,2,0,1,3,7]` and ascending-index order
    /// `[0,1,2,3,4,5,6,7]`, as required.
    #[test]
    fn permuting_rebuild_with_leaf_max_size_one_matches_hand_derived_sort() {
        let ds = Ungated(vec![
            [70.0, 100.0], // 0
            [10.0, 100.0], // 1
            [60.0, 100.0], // 2
            [20.0, 100.0], // 3
            [50.0, 100.0], // 4
            [30.0, 100.0], // 5
            [80.0, 100.0], // 6
            [40.0, 100.0], // 7
        ]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, ds)
            .leaf_max_size(1)
            .maximum_point_count(1000)
            .build();

        tree.add_points(0, 7);

        assert!(tree.point_indices_of_slot(0).is_empty());
        assert!(tree.point_indices_of_slot(1).is_empty());
        assert!(tree.point_indices_of_slot(2).is_empty());
        assert_eq!(
            tree.point_indices_of_slot(3),
            &[1u32, 3, 5, 7, 4, 2, 0, 6],
            "leaf_max_size=1 must fully sort slot 3 by dim0 -- a genuinely PERMUTING rebuild, \
             not just a membership-preserving pass-through"
        );

        for &v in tree.tree_index() {
            assert_eq!(v, 3, "all 8 points must live in slot 3");
        }
        assert_eq!(tree.active_count(), 8);
    }

    // ---------------------------------------------------------------
    // Test 4: Reactivation short-circuit (no merge involved)
    // ---------------------------------------------------------------

    #[test]
    fn reactivation_restores_tree_index_from_removed_map_without_growing_slots() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();

        tree.add_points(0, 3); // all four end up in slot 2 (see test 2)
        let slot_before_removal = tree.tree_index()[1];
        assert_ne!(slot_before_removal, -1);

        assert!(tree.remove_point(1));
        assert_eq!(tree.tree_index()[1], -1);
        assert_eq!(tree.removed_len(), 1);

        let physical_count_before: usize =
            (0..tree.tree_count()).map(|s| tree.point_indices_of_slot(s).len()).sum();

        tree.add_points(1, 1); // reactivation, not a duplicate insert

        assert_eq!(
            tree.tree_index()[1],
            slot_before_removal,
            "reactivation must restore the ORIGINAL (removed-from) slot value"
        );
        assert_eq!(tree.removed_len(), 0, "removed map must be empty after reactivation");

        let physical_count_after: usize =
            (0..tree.tree_count()).map(|s| tree.point_indices_of_slot(s).len()).sum();
        assert_eq!(
            physical_count_before, physical_count_after,
            "reactivation must not grow any slot's physical point list (no duplicate)"
        );
    }

    // ---------------------------------------------------------------
    // Test 5: remove_point return value + active_count
    // ---------------------------------------------------------------

    #[test]
    fn remove_point_returns_true_once_then_false_and_active_count_drops() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();
        tree.add_points(0, 2);

        assert_eq!(tree.active_count(), 3);
        assert!(tree.remove_point(1), "first removal must return true");
        assert_eq!(tree.active_count(), 2);
        assert!(!tree.remove_point(1), "repeat removal must return false");
        assert_eq!(tree.active_count(), 2, "active_count must not drop again");

        assert!(!tree.remove_point(999), "out-of-range removal must return false");
        assert!(!tree.remove_point(2usize.wrapping_add(1_000_000)), "wildly out-of-range must return false too");
    }

    // ---------------------------------------------------------------
    // Test 6: Remove-then-merge-then-query-readiness -- reactivate INTO
    // the migrated slot (rules 3 + 4 together; the resurrect-in-wrong-tree
    // bug this design exists to prevent).
    // ---------------------------------------------------------------

    #[test]
    fn reactivation_after_a_migrating_merge_lands_in_the_migrated_slot() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();

        // Same sequence as test 3: removed[0] ends up migrated to slot 2.
        tree.add_points(0, 1);
        assert!(tree.remove_point(0));
        tree.add_points(2, 2);
        tree.add_points(3, 3);
        assert_eq!(tree.tree_index()[0], -1);

        let slot2_len_before = tree.point_indices_of_slot(2).len();

        tree.add_points(0, 0); // reactivate index 0

        // Must land in slot 2 (the MIGRATED slot), never slot 1 (its
        // original, now-stale slot) -- the bug this test guards against is
        // reactivating into the wrong (no-longer-holding-the-point) tree.
        assert_eq!(tree.tree_index()[0], 2, "must reactivate into the MIGRATED slot, not the original one");
        assert_eq!(tree.removed_len(), 0);

        // No duplicate insert: slot 2's physical list did not grow.
        assert_eq!(tree.point_indices_of_slot(2).len(), slot2_len_before);
        assert!(tree.point_indices_of_slot(1).is_empty(), "slot 1 (the stale original) must stay empty");
    }

    // ---------------------------------------------------------------
    // Test 7: Rebuild-schedule determinism
    // ---------------------------------------------------------------

    #[test]
    fn identical_op_sequences_produce_identical_bookkeeping() {
        let pts: Vec<[f64; 2]> = (0..20).map(|i| [i as f64 * 1.3, (i * i) as f64 * 0.7]).collect();

        let run = |pts: &[[f64; 2]]| -> (Vec<Vec<u32>>, Vec<i32>, usize) {
            let ds = Ungated(pts.to_vec());
            let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, ds).maximum_point_count(1000).build();
            tree.add_points(0, 7);
            assert!(tree.remove_point(2));
            assert!(tree.remove_point(5));
            tree.add_points(8, 13);
            tree.add_points(2, 2); // reactivate
            tree.add_points(14, 19);
            let slots: Vec<Vec<u32>> = (0..tree.tree_count()).map(|s| tree.point_indices_of_slot(s).to_vec()).collect();
            (slots, tree.tree_index().to_vec(), tree.removed_len())
        };

        let (slots_a, ti_a, removed_a) = run(&pts);
        let (slots_b, ti_b, removed_b) = run(&pts);

        assert_eq!(slots_a, slots_b, "slot vind must be element-wise identical across identical runs");
        assert_eq!(ti_a, ti_b);
        assert_eq!(removed_a, removed_b);
    }

    // ---------------------------------------------------------------
    // Test 8: Ctor auto-add + empty-dataset-then-grow
    // ---------------------------------------------------------------

    #[test]
    fn ctor_auto_adds_existing_points_immediately_queryable_state() {
        let pts: Vec<[f64; 2]> = (0..5).map(|i| [i as f64, i as f64 * 2.0]).collect();
        let tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts.as_slice()).maximum_point_count(1000).build();

        assert_eq!(tree.active_count(), 5, "ctor must auto-add every point already in the dataset");
        let mut union: Vec<u32> = (0..tree.tree_count()).flat_map(|s| tree.point_indices_of_slot(s).iter().copied()).collect();
        union.sort_unstable();
        assert_eq!(union, vec![0, 1, 2, 3, 4]);
        for &v in tree.tree_index() {
            assert_ne!(v, -1);
        }
    }

    use std::cell::Cell;
    use std::rc::Rc;

    /// Interior-mutability `DataSource` (M1 A8-test style): `point_count()`
    /// starts at 0 (so the ctor's auto-add never fires), then grows AFTER
    /// `build()` so a later manual `add_points` call has real data behind it.
    struct GrowableDataSource {
        data: [[f64; 2]; 4],
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
    fn empty_dataset_yields_empty_forest_then_add_points_works_after_growth() {
        let count = Rc::new(Cell::new(0));
        let ds = GrowableDataSource {
            data: [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            count: count.clone(),
        };

        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, ds).maximum_point_count(1000).build();

        assert_eq!(tree.active_count(), 0, "empty dataset at build time must yield an empty forest");
        for s in 0..tree.tree_count() {
            assert!(tree.point_indices_of_slot(s).is_empty());
        }

        count.set(4);
        tree.add_points(0, 3);

        assert_eq!(tree.active_count(), 4);
        let mut union: Vec<u32> = (0..tree.tree_count()).flat_map(|s| tree.point_indices_of_slot(s).iter().copied()).collect();
        union.sort_unstable();
        assert_eq!(union, vec![0, 1, 2, 3]);
    }

    // ---------------------------------------------------------------
    // Test 9: Drain-and-refill
    // ---------------------------------------------------------------

    #[test]
    fn drain_and_refill_all_live_and_bookkeeping_consistent() {
        let pts: Vec<[f64; 1]> = (0..8).map(|i| [i as f64 * 3.3]).collect();
        let ds = Ungated(pts);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds).maximum_point_count(1000).build();

        tree.add_points(0, 7);
        assert_eq!(tree.active_count(), 8);

        for i in 0..8 {
            assert!(tree.remove_point(i));
        }
        assert_eq!(tree.active_count(), 0);
        assert_eq!(tree.removed_len(), 8);

        tree.add_points(0, 7); // reactivate all 8 in one call

        assert_eq!(tree.active_count(), 8);
        assert_eq!(tree.removed_len(), 0);
        for &v in tree.tree_index() {
            assert_ne!(v, -1);
        }
        let mut union: Vec<u32> = (0..tree.tree_count()).flat_map(|s| tree.point_indices_of_slot(s).iter().copied()).collect();
        union.sort_unstable();
        assert_eq!(union, (0u32..8).collect::<Vec<u32>>());
    }

    // ---------------------------------------------------------------
    // Extra: contiguity-contract panic on a genuinely misaligned call
    // ---------------------------------------------------------------

    #[test]
    #[should_panic(expected = "does not equal point_count")]
    fn add_points_panics_on_misaligned_genuinely_new_index() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds).maximum_point_count(1000).build();
        // First-ever call, but starts at 1 instead of 0 -- point_count is 0.
        tree.add_points(1, 1);
    }

    #[test]
    #[should_panic(expected = "outgrown this forest's capacity")]
    fn add_points_panics_when_point_count_outgrows_maximum_point_count() {
        // maximum_point_count(1) -> tree_count() = floor(log2(1)) + 1 = 1
        // (a single slot 0). The first point (pos = first0bit(0) = 0) fits;
        // the second (pos = first0bit(1) = 1) needs slot 1, which doesn't
        // exist -- must panic naming the capacity, not silently
        // index-out-of-bounds panic on `self.slots[pos]`.
        let ds = Ungated(vec![[10.0], [20.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds).maximum_point_count(1).build();
        assert_eq!(tree.tree_count(), 1);
        tree.add_points(0, 1);
    }

    #[test]
    fn maximum_point_count_zero_panics() {
        let result = std::panic::catch_unwind(|| {
            DynamicKdTreeBuilder::new(ConstDim::<1>, Ungated::<1>(vec![])).maximum_point_count(0)
        });
        assert!(result.is_err(), "maximum_point_count(0) must panic, not silently accept an unusable capacity");
    }

    #[test]
    fn reactivation_only_call_is_exempt_from_the_contiguity_assert() {
        // Mirrors nanoflann-ref's readd_point_1_reactivates_it_f32: legal
        // even though `start` (1) != `point_count` (4) at call time, because
        // every index in the range is a reactivation.
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds).maximum_point_count(1000).build();
        tree.add_points(0, 3);
        assert!(tree.remove_point(1));
        tree.add_points(1, 1); // must NOT panic
        assert_eq!(tree.removed_len(), 0);
    }

    // =================================================================
    // M2 Task 3: forest search
    // =================================================================

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// Brute-force KNN restricted to `live[i]` points, using the SAME
    /// `KnnResultSet` push pattern M1's own tests use (search.rs /
    /// tree.rs's `brute_force_knn`), and `metric.eval` (not a hand-rolled
    /// sum) so summation order matches the tree search bit-for-bit.
    fn brute_force_knn_live<const N: usize>(
        pts: &[[f64; N]],
        live: &[bool],
        query: &[f64; N],
        k: usize,
    ) -> (Vec<u32>, Vec<f64>) {
        let metric = L2;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            for (i, &is_live) in live.iter().enumerate() {
                if !is_live {
                    continue;
                }
                let d = metric.eval(query.as_slice(), &pts, i, ConstDim::<N>);
                rs.add_point(d, i as u32);
            }
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists)
    }

    /// Brute-force RKNN restricted to `live[i]` points: all live points
    /// within `radius` (strict `<`), closest `k` first, tie-broken by index
    /// (irrelevant on this test's tie-free data, kept for determinism).
    fn brute_force_rknn_live<const N: usize>(
        pts: &[[f64; N]],
        live: &[bool],
        query: &[f64; N],
        radius: f64,
        k: usize,
    ) -> (Vec<u32>, Vec<f64>) {
        let metric = L2;
        let mut scored: Vec<(f64, u32)> = (0..pts.len())
            .filter(|&i| live[i])
            .map(|i| (metric.eval(query.as_slice(), &pts, i, ConstDim::<N>), i as u32))
            .filter(|&(d, _)| d < radius)
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        scored.truncate(k);
        let indices = scored.iter().map(|&(_, i)| i).collect();
        let dists = scored.iter().map(|&(d, _)| d).collect();
        (indices, dists)
    }

    /// Brute-force radius set restricted to `live[i]` points, sorted
    /// ascending by distance (then index) -- mirrors search.rs's
    /// `brute_force_radius`.
    fn brute_force_radius_live<const N: usize>(
        pts: &[[f64; N]],
        live: &[bool],
        query: &[f64; N],
        radius: f64,
    ) -> Vec<(u32, f64)> {
        let metric = L2;
        let mut out: Vec<(u32, f64)> = (0..pts.len())
            .filter(|&i| live[i])
            .filter_map(|i| {
                let d = metric.eval(query.as_slice(), &pts, i, ConstDim::<N>);
                if d < radius { Some((i as u32, d)) } else { None }
            })
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        out
    }

    // ---------------------------------------------------------------
    // T3 Test 1: removed points never returned
    // ---------------------------------------------------------------

    #[test]
    fn removed_points_never_returned_by_knn_or_radius() {
        let pts: Vec<[f64; 2]> = (0..20).map(|i| [i as f64, (i * 2) as f64]).collect();
        let pts_slice: &[[f64; 2]] = pts.as_slice();
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts_slice)
            .maximum_point_count(1000)
            .build();

        let removed_indices = [1usize, 3, 5, 7, 9, 11, 13];
        for &i in &removed_indices {
            assert!(tree.remove_point(i));
        }
        assert_eq!(tree.active_count(), 13);

        let mut idx = [0u32; 20];
        let mut dist = [0.0f64; 20];
        let found = tree.knn_search(&[0.0, 0.0], &mut idx, &mut dist);
        assert_eq!(found, 13, "knn(k=20) over 13 live points must return exactly 13");
        for &i in &idx[..found] {
            assert!(!removed_indices.contains(&(i as usize)), "removed point {i} leaked into knn results");
        }

        let mut radius_out = Vec::new();
        let radius_found = tree.radius_search(&[0.0, 0.0], 1_000_000.0, &mut radius_out);
        assert_eq!(radius_found, 13, "radius over everything must return exactly the 13 live points");
        for item in &radius_out {
            assert!(
                !removed_indices.contains(&(item.index as usize)),
                "removed point leaked into radius results"
            );
        }
    }

    // ---------------------------------------------------------------
    // T3 Test 2: filtered brute-force equality (tie-free data)
    // ---------------------------------------------------------------

    #[test]
    fn filtered_brute_force_equality_100_points_dim3_random_removals() {
        // Random continuous-valued coordinates are tie-free with
        // overwhelming probability. NOTE: multi-slot forest traversal order
        // can genuinely differ from a single static tree's order
        // specifically ON TIES (each slot pass restarts its own
        // mindist/eps bookkeeping from its own root bbox) -- this test's
        // data is effectively tie-free, so exact index + bit-equal-distance
        // equality against a brute force is a safe assertion here.
        // Cross-language (C++ oracle) tie parity is Task 4's job, not this
        // one.
        let mut rng = Lcg(0xF0A_15E7u64);
        let n = 100;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                [
                    rng.next_f64() * 200.0 - 100.0,
                    rng.next_f64() * 200.0 - 100.0,
                    rng.next_f64() * 200.0 - 100.0,
                ]
            })
            .collect();
        let pts_slice: &[[f64; 3]] = pts.as_slice();

        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<3>, pts_slice)
            .maximum_point_count(1000)
            .build();

        let mut live = vec![true; n];
        for (i, live_i) in live.iter_mut().enumerate() {
            if rng.next_f64() < 0.3 {
                assert!(tree.remove_point(i));
                *live_i = false;
            }
        }

        for _case in 0..15 {
            let query = [
                rng.next_f64() * 200.0 - 100.0,
                rng.next_f64() * 200.0 - 100.0,
                rng.next_f64() * 200.0 - 100.0,
            ];
            let k = 10;

            let mut idx = vec![0u32; k];
            let mut dist = vec![0.0f64; k];
            let found = tree.knn_search(&query, &mut idx, &mut dist);
            let (want_idx, want_dist) = brute_force_knn_live(pts_slice, &live, &query, k);
            assert_eq!(found, want_idx.len());
            assert_eq!(&idx[..found], want_idx.as_slice());
            assert_eq!(&dist[..found], want_dist.as_slice());

            let radius = 5000.0;
            let mut ridx = vec![0u32; k];
            let mut rdist = vec![0.0f64; k];
            let rfound = tree.rknn_search(&query, radius, &mut ridx, &mut rdist);
            let (want_ridx, want_rdist) = brute_force_rknn_live(pts_slice, &live, &query, radius, k);
            assert_eq!(rfound, want_ridx.len());
            assert_eq!(&ridx[..rfound], want_ridx.as_slice());
            assert_eq!(&rdist[..rfound], want_rdist.as_slice());

            let mut radius_out = Vec::new();
            tree.radius_search(&query, radius, &mut radius_out);
            let want_radius = brute_force_radius_live(pts_slice, &live, &query, radius);
            let got_radius: Vec<(u32, f64)> = radius_out.iter().map(|it| (it.index, it.distance)).collect();
            assert_eq!(got_radius, want_radius);
        }
    }

    // ---------------------------------------------------------------
    // T3 Test 3: reactivated point returned again, same distance bits
    // ---------------------------------------------------------------

    #[test]
    fn reactivated_point_is_returned_again_with_same_distance_bits() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [3.0, 4.0], [10.0, 10.0], [-5.0, -5.0]];
        let pts_slice: &[[f64; 2]] = pts.as_slice();
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts_slice)
            .maximum_point_count(1000)
            .build();

        let query = [0.0, 0.0];
        let expected_dist = L2.eval(query.as_slice(), &pts_slice, 1, ConstDim::<2>);

        assert!(tree.remove_point(1));

        let mut idx = [0u32; 4];
        let mut dist = [0.0f64; 4];
        let found = tree.knn_search(&query, &mut idx, &mut dist);
        assert!(!idx[..found].contains(&1), "removed point must not be found by query");

        // Reactivation: `start == 1` is not `point_count` (4) at call time,
        // but every index in `1..=1` is a reactivation, so this is
        // contiguity-legal (see `add_points`'s doc comment).
        tree.add_points(1, 1);

        let mut idx2 = [0u32; 4];
        let mut dist2 = [0.0f64; 4];
        let found2 = tree.knn_search(&query, &mut idx2, &mut dist2);
        let pos = idx2[..found2]
            .iter()
            .position(|&i| i == 1)
            .expect("reactivated point must be found again");
        assert_eq!(
            dist2[pos], expected_dist,
            "reactivated point's distance must be bit-identical to its pre-removal distance"
        );
    }

    // ---------------------------------------------------------------
    // T3 Test 4: empty-forest quirk
    // ---------------------------------------------------------------

    #[test]
    fn empty_forest_quirk_true_for_radius_false_for_knn_zero_for_radius_search_wrapper() {
        let pts: Vec<[f64; 2]> = Vec::new();
        let pts_slice: &[[f64; 2]] = pts.as_slice();
        let tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts_slice)
            .maximum_point_count(1000)
            .build();
        assert_eq!(tree.active_count(), 0);

        let mut idx = [0u32; 3];
        let mut dist = [0.0f64; 3];
        let mut knn_rs = KnnResultSet::<f64, u32>::new(&mut idx, &mut dist);
        let full = tree.find_neighbors(&mut knn_rs, &[0.0, 0.0], &SearchParams::default());
        assert!(!full, "KNN find_neighbors on an empty forest must return false");
        assert_eq!(knn_rs.size(), 0);

        // Inherited nanoflann quirk (M1 Corrections #4):
        // `RadiusResultSet::full()` is HARDWIRED true (nanoflann.hpp:433)
        // regardless of whether anything was ever added -- so an empty
        // forest (zero slot passes ever touch `result`) still reports
        // `find_neighbors(..) == true`, with ZERO items. This is the
        // literal C++ contract, not a bug; it's only observable through
        // this generic escape hatch (the `radius_search` wrapper below
        // returns the found COUNT and hides the quirk).
        let mut items = Vec::new();
        let mut radius_rs = RadiusResultSet::new(100.0, &mut items);
        let radius_full = tree.find_neighbors(&mut radius_rs, &[0.0, 0.0], &SearchParams::default());
        assert!(radius_full, "RadiusResultSet::full() is hardwired true -- the quirk");
        assert_eq!(items.len(), 0);

        let mut out = Vec::new();
        let count = tree.radius_search(&[0.0, 0.0], 100.0, &mut out);
        assert_eq!(count, 0, "the additive radius_search wrapper returns the COUNT, hiding the quirk");
    }

    // ---------------------------------------------------------------
    // T3 Test 5: multi-slot correctness, then forced merge
    // ---------------------------------------------------------------

    #[test]
    fn multi_slot_correctness_spanning_two_slots_then_merged() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0], [8.0, -1.0]];
        let pts_slice: &[[f64; 2]] = pts.as_slice();

        let ds = Ungated(pts.clone());
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, ds)
            .maximum_point_count(1000)
            .build();

        // 3 points -> first0bit(0)=0, first0bit(1)=1, first0bit(2)=0:
        // slot0=[2], slot1=[0,1] -- 2 slots simultaneously occupied (see
        // `add_4_points_one_batch_matches_first0bit_slot_occupancy` above
        // for the same derivation one step further).
        tree.add_points(0, 2);

        assert!(!tree.point_indices_of_slot(0).is_empty(), "slot 0 must be occupied");
        assert!(!tree.point_indices_of_slot(1).is_empty(), "slot 1 must be occupied");
        assert!(tree.point_indices_of_slot(2).is_empty());

        let query = [1.0, 1.0];
        let k = 3;
        let mut idx = vec![0u32; k];
        let mut dist = vec![0.0f64; k];
        let found = tree.knn_search(&query, &mut idx, &mut dist);
        let (want_idx, want_dist) = brute_force_knn_live(pts_slice, &[true, true, true, false], &query, k);
        assert_eq!(found, 3);
        assert_eq!(idx, want_idx.as_slice());
        assert_eq!(dist, want_dist.as_slice());

        // Force a merge: the 4th point's pos = first0bit(3) = 2, which
        // absorbs slot0 AND slot1's entire contents into slot2.
        tree.add_points(3, 3);

        assert!(tree.point_indices_of_slot(0).is_empty());
        assert!(tree.point_indices_of_slot(1).is_empty());
        assert!(!tree.point_indices_of_slot(2).is_empty(), "everything merged into slot 2");

        let k2 = 4;
        let mut idx2 = vec![0u32; k2];
        let mut dist2 = vec![0.0f64; k2];
        let found2 = tree.knn_search(&query, &mut idx2, &mut dist2);
        let (want_idx2, want_dist2) = brute_force_knn_live(pts_slice, &[true; 4], &query, k2);
        assert_eq!(found2, 4);
        assert_eq!(idx2, want_idx2.as_slice());
        assert_eq!(dist2, want_dist2.as_slice());
    }

    // ---------------------------------------------------------------
    // T3 Test 6: eps plumbing
    // ---------------------------------------------------------------

    #[test]
    fn eps_plumbing_zero_matches_brute_force_ten_returns_valid_member() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]];
        let pts_slice: &[[f64; 2]] = pts.as_slice();
        let tree = DynamicKdTreeBuilder::new(ConstDim::<2>, pts_slice)
            .maximum_point_count(1000)
            .build();

        // 3 points -> slot0=[2], slot1=[0,1]: 2 simultaneously-occupied slots.
        assert!(!tree.point_indices_of_slot(0).is_empty());
        assert!(!tree.point_indices_of_slot(1).is_empty());

        let query = [1.0, 1.0];
        let k = 1;

        let params_zero = SearchParams { eps: 0.0, sorted: true };
        let mut idx0 = [0u32; 1];
        let mut dist0 = [0.0f64; 1];
        tree.knn_search_with(&query, &mut idx0, &mut dist0, &params_zero);
        let (want_idx, want_dist) = brute_force_knn_live(pts_slice, &[true; 3], &query, k);
        assert_eq!(idx0.as_slice(), want_idx.as_slice(), "eps=0 must match brute force exactly");
        assert_eq!(dist0.as_slice(), want_dist.as_slice());

        let params_eps = SearchParams { eps: 10.0, sorted: true };
        let mut idx_eps = [0u32; 1];
        let mut dist_eps = [0.0f64; 1];
        let found = tree.knn_search_with(&query, &mut idx_eps, &mut dist_eps, &params_eps);
        assert_eq!(found, 1);
        assert!(
            (idx_eps[0] as usize) < pts.len(),
            "approximate result must still be a member of the live set"
        );
    }

    // ---------------------------------------------------------------
    // T3 Test 7: sorted radius across slots
    // ---------------------------------------------------------------

    #[test]
    fn sorted_radius_search_ascending_across_slots() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0], [8.0, -1.0]];
        let ds = Ungated(pts.clone());
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<2>, ds)
            .maximum_point_count(1000)
            .build();
        tree.add_points(0, 2); // slot0=[2], slot1=[0,1] -- 2 occupied slots
        assert!(!tree.point_indices_of_slot(0).is_empty());
        assert!(!tree.point_indices_of_slot(1).is_empty());

        let mut out = Vec::new();
        let params = SearchParams { eps: 0.0, sorted: true };
        tree.radius_search_with(&[1.0, 1.0], 1000.0, &mut out, &params);
        let dists: Vec<f64> = out.iter().map(|it| it.distance).collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, sorted, "sorted=true must yield ascending distances across multiple slot passes");
        assert!(out.len() >= 2, "test must actually exercise multiple slots' worth of results");
    }
}
