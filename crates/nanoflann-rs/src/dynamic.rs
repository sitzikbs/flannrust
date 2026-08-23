//! Bentley-Saxe dynamic forest: a faithful port of nanoflann's
//! `KDTreeSingleIndexDynamicAdaptor` (nanoflann.hpp:2521-2718) plus the
//! sub-tree class it wraps, `KDTreeSingleIndexDynamicAdaptor_`
//! (nanoflann.hpp:2248-2504, in particular its `buildIndex()` at
//! nanoflann.hpp:2345-2367). This task is FOREST BOOKKEEPING ONLY —
//! `add_points`/`remove_point`/the merge-and-rebuild schedule. Search lands
//! in M2 Task 3; fields/paths that only matter to a future query
//! implementation are marked `#[allow(dead_code)] // TODO(m2-task-3)`.
//!
//! # The forest idea
//!
//! `tree_count` independent static kd-trees ("slots"), indexed 0..tree_count.
//! Each newly-added point walks a binary-counter pattern
//! ([`first0bit`]) to decide which slot absorbs it: slot `pos` absorbs
//! every LOWER slot's entire current point list plus the new point itself,
//! and every lower slot becomes empty. This is exactly incrementing a
//! binary counter by one (`pos` = position of the counter's lowest unset
//! bit) — after `n` sequential adds with no removals, the set of non-empty
//! slots is exactly the set bits of `n`'s binary representation, each
//! holding `2^slot` points. A slot is REBUILT FROM SCRATCH (M1's
//! [`SubtreeBuilder`], `base = 0`) every time its point list changes, and
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
use crate::metric::{Distance, L2};
use crate::node::Node;
use crate::result_set::{KeepInsertionOrder, TieBreak};
use crate::scalar::{IndexType, Scalar};

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
/// search — Task 3 — always checks `vind.is_empty()` first, mirroring
/// nanoflann's `size(*this) == 0` early-return, nanoflann.hpp:2380).
struct Slot<T: Scalar, D: Dim, Idx: IndexType> {
    vind: Vec<Idx>,
    // NOTE: `nodes`/`root_bbox` don't need `#[allow(dead_code)]` even though
    // no query method reads them yet (Task 3) -- `add_points`'s rebuild pass
    // already writes AND clears them (`.clear()`, struct-literal init,
    // `slot.root_bbox = bbox`), which is enough for rustc's dead_code
    // analysis to consider them used. Confirmed by temporarily stripping
    // `#[allow(dead_code)]` here during self-review: no warning fired for
    // either field, only for `DynamicKdTree::metric` below (which nothing
    // in this task reads OR writes after construction).
    nodes: Vec<Node<T>>,
    root_bbox: D::Array<Interval<T>>,
}

impl<T: Scalar, D: Dim, Idx: IndexType> Slot<T, D, Idx> {
    fn empty(dim: D) -> Self {
        Slot { vind: Vec::new(), nodes: Vec::new(), root_bbox: dim.filled(Interval::default()) }
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
/// time on the calling thread, matching every observed C++ call site).
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
    /// for the exact cast). Default 1_000_000_000 (30 slots).
    pub fn maximum_point_count(mut self, n: usize) -> Self {
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

    /// Selects the kNN/RKNN equal-distance tie policy (consumed by Task 3's
    /// search methods; stored now so the type parameter is fixed at build time).
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
    #[allow(dead_code)] // TODO(m2-task-3): consumed once query methods land
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
    /// # Contiguous-append contract (controller ruling, DEVIATION from C++)
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
    /// already 4. (NOTE: the M2 Task 2 brief's literal text proposes a
    /// blanket `assert_eq!(start, self.point_count)` at the top of this
    /// method; that would PANIC on this exact legal, oracle-verified
    /// sequence, so it is not what's implemented — see the "Brief-vs-source
    /// discrepancies" section of this task's report.)
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
    /// non-empty, [`crate::build::SubtreeBuilder`] rebuilds it FROM THE
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
    /// comment): point 0 -> slot 0; point 1 -> slot 1 (absorbs slot 0's
    /// [0]); point 2 -> slot 0; point 3 -> slot 2 (absorbs slot 0's [2] AND
    /// slot 1's [0,1]). All four points therefore end up in slot 2, with
    /// slots 0 and 1 left empty.
    #[test]
    fn add_4_points_one_batch_matches_first0bit_slot_occupancy() {
        let ds = Ungated(vec![[10.0], [20.0], [30.0], [40.0]]);
        let mut tree = DynamicKdTreeBuilder::new(ConstDim::<1>, ds)
            .maximum_point_count(1000)
            .build();

        tree.add_points(0, 3);

        assert!(tree.point_indices_of_slot(0).is_empty(), "slot 0 must be empty");
        assert!(tree.point_indices_of_slot(1).is_empty(), "slot 1 must be empty");
        let mut slot2: Vec<u32> = tree.point_indices_of_slot(2).to_vec();
        slot2.sort_unstable();
        assert_eq!(slot2, vec![0, 1, 2, 3], "slot 2 must hold all 4 points");

        assert_eq!(tree.tree_index(), &[2, 2, 2, 2]);

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
    ///   `tree_index[0] = 1`.
    /// - `remove_point(0)`: `removed = {0: 1}`, `tree_index[0] = -1`.
    /// - `add_points(2, 2)`: point 2 -> slot 0 (pos=first0bit(2)=0).
    /// - `add_points(3, 3)`: point 3 -> slot 2 (pos=first0bit(3)=2),
    ///   absorbing slot 0's [2] (tree_index[2] live -> tree_index[2]=2) AND
    ///   slot 1's [0 or 1 in whatever order the earlier rebuild left them]:
    ///   for the entry that is index 0 (`tree_index[0] == -1`, a tombstone),
    ///   the `else` branch fires -- `removed[0]` MIGRATES from `1` to `2`.
    ///   For the entry that is index 1 (live), `tree_index[1] = 2`.
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

        // Slot 2 physically holds all 4 (including the tombstoned 0 --
        // lazy deletion never removes from `vind`).
        let mut slot2: Vec<u32> = tree.point_indices_of_slot(2).to_vec();
        slot2.sort_unstable();
        assert_eq!(slot2, vec![0, 1, 2, 3]);
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
}
