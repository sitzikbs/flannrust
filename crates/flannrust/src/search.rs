//! Query core — the exact port of nanoflann's `searchLevel` +
//! `computeInitialDistances` + the static `findNeighbors` driver
//! (nanoflann.hpp: `searchLevel` ~1228-1287, `computeInitialDistances`
//! ~1595-1605, `findNeighbors` ~1990-2011).

use core::mem::MaybeUninit;

use crate::bbox::Interval;
use crate::data_source::DataSource;
use crate::dim::Dim;
use crate::filter::PointFilter;
use crate::metric::Distance;
use crate::node::Node;
use crate::params::SearchParams;
use crate::result_set::{BoxResultSet, ResultSet};
use crate::scalar::{DistanceValue, IndexType, Scalar};

/// Everything a query needs, borrowed: dataset, metric, dimensionality, the
/// built arena + permuted index vector, and the tree's root bounding box.
/// The root node is always arena index 0 (task 6's `SubtreeBuilder::build`
/// always allocates the top-level node first).
///
/// `dim` is carried as a typed `D: Dim` (not a plain `usize`) so it reaches
/// `Distance::eval` in the leaf-scan hot path (`search_level`) as a
/// [`crate::ConstDim`] whenever the caller's tree was built over one —
/// letting the metric kernel's dimension loop constant-fold and fully
/// unroll under monomorphization instead of branching on a runtime value
/// (task 14's ConstDim-unroll fix; see `metric.rs`'s `Distance::eval` doc).
pub(crate) struct SearchCtx<'a, T: Scalar, D: Dim, DS: DataSource<T> + ?Sized, M: Distance<T>, Idx>
{
    pub ds: &'a DS,
    pub metric: &'a M,
    pub dim: D,
    pub nodes: &'a [Node<T>],
    pub vind: &'a [Idx],
    pub root_bbox: &'a [Interval<T>],
}

/// = static `findNeighbors` (nanoflann.hpp ~1990-2011) minus the unbuilt-tree
/// throw (unrepresentable here: `SearchCtx` can only be constructed from an
/// already-built tree in later tasks). Empty tree (`ctx.nodes` empty) →
/// `false`, matching C++'s `if (this->size(*this) == 0) return false;`.
///
/// `dists_scratch` is caller-provided, length `ctx.dim`, zeroed here (callers
/// with `ConstDim` pass a stack array — no per-query heap allocation).
pub(crate) fn find_neighbors<T, D, DS, M, Idx, R, F>(
    ctx: &SearchCtx<T, D, DS, M, Idx>,
    result: &mut R,
    query: &[T],
    params: &SearchParams,
    filter: &F,
    dists_scratch: &mut [M::DistanceType],
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
    R: ResultSet<M::DistanceType, Idx>,
    F: PointFilter<Idx>,
{
    if ctx.nodes.is_empty() {
        return false;
    }
    // Release-mode check (mirrors `find_within_box`'s bounds assert in
    // `tree.rs`): one check per query, off the leaf-scan hot loop.
    assert!(
        query.len() >= ctx.dim.dim(),
        "query vector length {} is shorter than dim {}",
        query.len(),
        ctx.dim.dim()
    );
    debug_assert_eq!(
        dists_scratch.len(),
        ctx.dim.dim(),
        "dists_scratch.len() != dim"
    );

    // C++ (nanoflann.hpp:1999): `DistanceType epsError = 1 + static_cast<DistanceType>(searchParams.eps);`
    // — widens `eps` (f32) to `DistanceType` FIRST, then adds `1` in
    // `DistanceType` arithmetic. Mirrored exactly: widen each operand via
    // `DistanceValue::from_f32` (an exact f32->f64 widening, no rounding),
    // then add in `DistanceType` space. (Fix round 1: an earlier version of
    // this computed `1.0f32 + params.eps` in `f32` FIRST and widened the
    // sum — that rounds to f32 precision before widening and is NOT
    // bit-identical to the C++; see `eps_error_widen_first_parity` below for
    // a discriminating test.)
    let eps_error = M::DistanceType::from_f32(1.0f32) + M::DistanceType::from_f32(params.eps);

    for d in dists_scratch.iter_mut() {
        *d = M::DistanceType::ZERO;
    }

    let mindist = compute_initial_distances(ctx, query, dists_scratch);

    // C++ discards `searchLevel`'s own return value here — it's only used to
    // propagate an abort UP THROUGH the recursion, not by the top-level
    // driver.
    let _ = search_level_hybrid(
        ctx,
        result,
        query,
        0,
        mindist,
        dists_scratch,
        eps_error,
        filter,
        0,
    );

    if params.sorted {
        result.sort();
    }

    result.full()
}

/// Port of `computeInitialDistances` (nanoflann.hpp ~1595-1605). For each
/// dim `i`: if `query[i] < root_bbox[i].low`, accumulate the per-axis
/// component distance to the low edge; ELSE IF `query[i] > root_bbox[i].high`
/// (an `else if`, NOT an independent second `if` — the brief's prose says
/// "two independent ifs", but the vendored C++ source at nanoflann.hpp:1608
/// uses `else if`; functionally equivalent for a well-formed bbox where
/// `low <= high`, but we mirror the source exactly per task instructions).
/// Returns the summed lower-bound distance.
fn compute_initial_distances<T, D, DS, M, Idx>(
    ctx: &SearchCtx<T, D, DS, M, Idx>,
    query: &[T],
    dists: &mut [M::DistanceType],
) -> M::DistanceType
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
{
    let dim = ctx.dim.dim();
    let query = &query[..dim];
    let dists = &mut dists[..dim];
    let mut dist = M::DistanceType::ZERO;
    for i in 0..dim {
        if query[i] < ctx.root_bbox[i].low {
            dists[i] = ctx.metric.accum_dist(query[i], ctx.root_bbox[i].low, i);
            dist = dist + dists[i];
        } else if query[i] > ctx.root_bbox[i].high {
            dists[i] = ctx.metric.accum_dist(query[i], ctx.root_bbox[i].high, i);
            dist = dist + dists[i];
        }
    }
    dist
}

/// Native-recursion depth limit for the hybrid search path. Balanced trees
/// (leaf_max_size 10, 100k points) reach depth ~13–17; 96 covers every
/// realistic workload with wide headroom while staying well within a
/// default 8 MB thread stack (~10 KB of frames). Degenerate trees that
/// exceed this depth fall back to the explicit `FrameStack` for the
/// remaining subtree — no stack overflow risk, same correctness.
const HYBRID_RECURSION_DEPTH_LIMIT: u32 = 96;

/// Inline capacity of `search_level`'s explicit frame stack (see `Frame`
/// below). Deep enough for any realistically balanced tree (leaf_max_size
/// 10 over 100k points needs depth ~13) with wide headroom; degenerate
/// trees deeper than this spill onto the heap (see `FrameStack`) — a query
/// never allocates on its own unless it actually needs a frame past this
/// depth.
///
/// **This value (128) is ARBITRARY-WITH-HEADROOM, not derived from a
/// principled depth-distribution analysis.** Carried over unchanged from
/// the M2.5-T1 prototype measurement, chosen only as "comfortably above
/// every depth this crate's own gate/xval workloads produce" (~13-17 for
/// the standard 100k-point/leaf-10 gate configuration; task 6's own
/// deliberately-adversarial degenerate-tree stress tests reach depth
/// ~2 115 and ~16 794 — see `heavy_query_degenerate_trees` below — which
/// are FAR beyond 128 regardless of what this constant is set to, so no
/// choice of a "reasonable" inline capacity avoids the spill path for
/// those trees; 128 was never meant to cover them, only the realistic
/// case). No sweep over alternative capacities against realistic
/// (non-adversarial) tree-depth distributions has been done — a future
/// task that wants to tune this number down (to shrink the fixed
/// `size_of::<FrameStack>()` stack reservation) or up (to push the spill
/// boundary further out) should measure first, not assume 128 is load-
/// bearing. (Reviewer fix-round item 5's one-off A/B did test whether
/// LOWERING this to 32 recovers the `radius`/`knn_dyn_dim8_f64` losses
/// documented in `task-2-report.md` §6 — it does NOT: both regressions are
/// intrinsic to the store/reload + code-size cost of even an inlined
/// explicit stack vs. register-carried native recursion, not a function of
/// this constant.)
///
/// **Spill heap-churn cost (documented next to the no-stack-overflow
/// upgrade it trades against):** a query whose tree is deeper than this
/// constant pays a REAL per-query heap allocation/reallocation cost once
/// `FrameStack::push` starts routing into `overflow: Vec` (`Vec`'s normal
/// doubling-growth reallocation, repeated as the query's stack depth grows
/// past 128, 256, 512, ... frames) — this is new cost that did NOT exist
/// under the previous native-recursion implementation (which paid zero
/// heap allocation at any depth, only native call-stack frames). The
/// trade this constant embodies: bounded, `MaybeUninit`-cheap common-case
/// behavior plus IMMUNITY to native-stack overflow at arbitrary depth (the
/// old recursive form could in principle overflow a small worker thread's
/// stack on a sufficiently degenerate tree — see `heavy_query_degenerate_trees`'s
/// updated doc comment), in exchange for real, measurable heap churn on
/// the rare deep-tree query. This is judged a good trade (the M1
/// degenerate-tree ceiling is real and adversarial-tree-shaped datasets
/// are exactly the case this crate's own test suite deliberately
/// stress-tests), but it is a genuine give-back, not a free upgrade, and a
/// workload that runs MANY queries against a tree deeper than 128 on every
/// query would pay this repeatedly (once per query, since `FrameStack` is
/// not reused across queries — see `search_level`'s doc for why).
const SEARCH_STACK_INLINE_CAPACITY: usize = 128;

/// One interior node's saved "resume" state on `search_level`'s explicit
/// stack (see that function's doc for the full mapping). A frame is pushed
/// the moment we decide to descend into an interior node's `best_child`,
/// in `Phase::PostBest` — "when this frame is resumed, run the logic that
/// follows the recursive function's FIRST (`best_child`) call". If that
/// logic decides to visit `other_child`, the SAME frame stays on the stack
/// with `phase` flipped to `Phase::PostOther` — "when resumed, run the
/// logic that follows the SECOND (`other_child`) call" (just the restore).
/// A frame that gets pruned (no `other_child` visit) is popped directly out
/// of `Phase::PostBest` and never sees `PostOther`.
#[derive(Clone, Copy)]
struct Frame<Dist: Copy> {
    /// `node.split_dim()` of the interior node this frame belongs to.
    idx: usize,
    /// `dists[idx]` as it was when this node was entered — the recursive
    /// function's `let dst = dists[idx];` local, captured up front (safe to
    /// capture before, rather than after, descending into `best_child`:
    /// nothing touches `dists[idx]` during that subtree's traversal except
    /// possibly a same-axis descendant, which always restores it to this
    /// same value before returning — see the module doc below). Read again
    /// in `Phase::PostOther` to restore.
    dst: Dist,
    /// `cut_dist` for entering `other_child` (recursive local of the same
    /// name). Only meaningful in `Phase::PostBest`.
    cut_dist: Dist,
    /// Arena index of the not-yet-visited sibling. Only meaningful in
    /// `Phase::PostBest`.
    other_child: u32,
    /// `mindist` as passed INTO this node (the recursive function's own
    /// `mindist` parameter, before its `mindist = mindist + cut_dist -
    /// dst` update). Only meaningful in `Phase::PostBest`.
    entry_mindist: Dist,
    phase: Phase,
}

#[derive(Clone, Copy)]
enum Phase {
    PostBest,
    PostOther,
}

/// `search_level`'s explicit node-visit stack: a fixed inline array
/// (`SEARCH_STACK_INLINE_CAPACITY` frames) with a heap-`Vec` spill for the
/// rare tree deeper than that. A single logical LIFO stack split across two
/// backing stores — `inline` always holds the BOTTOM
/// `SEARCH_STACK_INLINE_CAPACITY` frames (or fewer), `overflow` holds
/// everything past that — so `push`/`pop`/`top_mut` simply route to
/// whichever store currently holds the top. `Vec::new()` doesn't allocate
/// until its first `push`, so a query whose tree never exceeds the inline
/// capacity (every realistic balanced tree) performs no heap allocation at
/// all; only a query that actually walks past frame
/// `SEARCH_STACK_INLINE_CAPACITY` pays for (and reuses, across its own
/// remaining pushes) one `Vec` growth.
///
/// `inline` is `[MaybeUninit<Frame<Dist>>; N]`, NOT `[Frame<Dist>; N]` —
/// deliberately, after measuring the plain-array version as a genuine
/// regression: `FrameStack::new()` runs once per QUERY (see `search_level`'s
/// doc), so a typical query — whose tree is only ~15 levels deep — pays to
/// initialize (zero) all 128 slots for nothing. `[Frame::default(); 128]`
/// compiles to a ~4 KiB `memset` call plus an inlined stack-clash guard-page
/// probe on every query (measured: `crates/flannrust`'s dim-3 gate median
/// ratio moved 1.06 -> 1.14, a real regression — see task-2-report.md's
/// asm/measurement evidence). `[const { MaybeUninit::uninit() }; N]` — an
/// inline-`const` array-repeat expression, so it doesn't need `Dist: Copy`
/// the way a plain `[expr; N]` repeat would — is ordinary safe Rust and
/// costs nothing at runtime: no element is considered "initialized" until
/// `push` actually writes one.
///
/// INITIALIZATION invariant: `inline[i]` holds a live, `push`-written
/// `Frame` for every `i < inline_len`, and NEVER for `i >= inline_len` —
/// `inline_len` is the only place that boundary is adjusted, always in
/// lock-step with a `write` (`push`) or a consuming read (`pop`/`top_mut`,
/// both of which only ever touch index `inline_len - 1`, i.e. the slot the
/// most recent `push` wrote and no `pop` has consumed yet).
///
/// DESTRUCTION/PANIC note: `Dist: Copy` (bound on this struct and on
/// `Frame`) is a leak guard, not a soundness requirement. `MaybeUninit<T>`
/// never runs `T`'s destructor, and this struct has no manual `Drop`, so
/// a non-`Copy` `Frame` holding a `Drop` resource would be LEAKED (never
/// UB — `mem::forget` is safe) for every slot still initialized when the
/// stack is dropped: on the ordinary path, on an aborted search's early
/// return, or on an unwinding panic. A type cannot be both `Copy` and
/// `Drop`, so the bound forecloses that leak statically. `M::DistanceType`
/// is already `Copy` via `DistanceValue` (`scalar.rs`), so the bound costs
/// nothing at any current call site.
struct FrameStack<Dist: Copy> {
    inline: [MaybeUninit<Frame<Dist>>; SEARCH_STACK_INLINE_CAPACITY],
    inline_len: usize,
    overflow: Vec<Frame<Dist>>,
    /// Test-only high-water mark: the largest `inline_len + overflow.len()`
    /// ever observed by `push`, i.e. the deepest this stack ever actually
    /// got during ONE query. Exists purely so tests that mean to exercise
    /// the `overflow` spill path can ASSERT they actually did — reviewer
    /// fix-round finding: a test that only checks the final knn/radius
    /// result can pass even when the constructed tree's query path never
    /// pushes past `SEARCH_STACK_INLINE_CAPACITY` (the tree's overall
    /// `max_depth` and one SPECIFIC query's push depth are different
    /// numbers — see `spill_boundary_deep_tree_knn_and_radius_match_brute_force`'s
    /// updated doc comment for the concrete miss this caught: query index
    /// 60 only reached inline depth 69, never spilling, while the test's
    /// assertions still passed on correctness alone). Zero cost outside
    /// `cfg(test)` — the field doesn't exist in a normal build.
    #[cfg(test)]
    max_depth_seen: usize,
}

impl<Dist: Copy> FrameStack<Dist> {
    #[inline]
    fn new() -> Self {
        // `[const { MaybeUninit::uninit() }; N]` (an inline-`const` array
        // repeat expression, not a `Copy`-based one) sidesteps needing
        // `Frame<Dist>: Copy` for the array-repeat itself — each element is
        // independently const-evaluated as uninitialized, so this compiles
        // for ANY `Dist` and costs nothing at runtime.
        Self {
            inline: [const { MaybeUninit::uninit() }; SEARCH_STACK_INLINE_CAPACITY],
            inline_len: 0,
            overflow: Vec::new(),
            #[cfg(test)]
            max_depth_seen: 0,
        }
    }

    #[inline]
    fn push(&mut self, frame: Frame<Dist>) {
        if self.inline_len < SEARCH_STACK_INLINE_CAPACITY {
            self.inline[self.inline_len].write(frame);
            self.inline_len += 1;
        } else {
            self.overflow.push(frame);
        }
        #[cfg(test)]
        {
            let depth = self.inline_len + self.overflow.len();
            if depth > self.max_depth_seen {
                self.max_depth_seen = depth;
            }
        }
    }

    /// Pops and returns the top frame, or `None` if the stack is empty —
    /// `None` signals the whole search is finished (mirrors the recursive
    /// form's outermost call finally returning).
    #[inline]
    fn pop(&mut self) -> Option<Frame<Dist>> {
        if let Some(f) = self.overflow.pop() {
            return Some(f);
        }
        if self.inline_len == 0 {
            return None;
        }
        self.inline_len -= 1;
        // SAFETY: `inline[inline_len]` (post-decrement) is exactly the slot
        // the most recent `push` wrote — see the struct's invariant doc.
        Some(unsafe { self.inline[self.inline_len].assume_init_read() })
    }

    #[inline]
    fn top_mut(&mut self) -> Option<&mut Frame<Dist>> {
        if let Some(f) = self.overflow.last_mut() {
            return Some(f);
        }
        if self.inline_len == 0 {
            return None;
        }
        // SAFETY: same invariant as `pop` — `inline[inline_len - 1]` was
        // written by the most recent `push` and hasn't been popped since.
        Some(unsafe { self.inline[self.inline_len - 1].assume_init_mut() })
    }
}

// Test-only side channel for `FrameStack::max_depth_seen` (see that
// field's doc): `search_level`'s `FrameStack` is a purely local variable
// (deliberately — no per-query allocation on the common path, no public
// API change), so a test that wants to know how deep a specific query's
// stack actually got needs somewhere to read that number from after the
// call returns. Published right before every `search_level` return site
// via the `record_max_stack_depth!` macro. `thread_local!` (not a plain
// global `static`) so parallel test execution (`cargo test` runs test
// functions on multiple threads by default) can't cross-contaminate two
// tests' readings.
#[cfg(test)]
thread_local! {
    static LAST_SEARCH_STACK_MAX_DEPTH: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Hybrid search: native recursion for the first `HYBRID_RECURSION_DEPTH_LIMIT`
/// levels, falling back to the explicit-stack `search_level_explicit` for
/// degenerate trees that exceed that depth. Native recursion keeps `dst`,
/// `cut_dist`, `mindist`, and `other_child` in CPU registers instead of
/// heap-allocated `Frame` structs — measured ~7% faster on dim-8 f64 knn.
#[allow(clippy::too_many_arguments)]
fn search_level_hybrid<T, D, DS, M, Idx, R, F>(
    ctx: &SearchCtx<T, D, DS, M, Idx>,
    result: &mut R,
    query: &[T],
    node_idx: u32,
    mindist: M::DistanceType,
    dists: &mut [M::DistanceType],
    eps_error: M::DistanceType,
    filter: &F,
    depth: u32,
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
    R: ResultSet<M::DistanceType, Idx>,
    F: PointFilter<Idx>,
{
    let node = &ctx.nodes[node_idx as usize];

    if node.is_leaf() {
        let (left, right) = node.leaf_range();
        for &accessor in &ctx.vind[left..right] {
            if !filter.is_active(accessor) {
                continue;
            }
            let dist = ctx.metric.eval(query, ctx.ds, accessor.to_usize(), ctx.dim);
            if dist < result.worst_dist() && !result.add_point(dist, accessor) {
                return false;
            }
        }
        return true;
    }

    if depth >= HYBRID_RECURSION_DEPTH_LIMIT {
        return search_level_explicit(ctx, result, query, node_idx, mindist, dists, eps_error, filter);
    }

    let idx = node.split_dim();
    assert!(idx < dists.len());
    let val = query[idx];
    let diff1 = val - node.div_low();
    let diff2 = val - node.div_high();

    let (c1, c2) = node.children();
    let (best_child, other_child, cut_dist) = if (diff1 + diff2) < T::default() {
        (c1, c2, ctx.metric.accum_dist(val, node.div_high(), idx))
    } else {
        (c2, c1, ctx.metric.accum_dist(val, node.div_low(), idx))
    };

    if !search_level_hybrid(ctx, result, query, best_child, mindist, dists, eps_error, filter, depth + 1) {
        return false;
    }

    let dst = dists[idx];
    let new_mindist = mindist + cut_dist - dst;
    dists[idx] = cut_dist;

    if new_mindist * eps_error <= result.worst_dist()
        && !search_level_hybrid(ctx, result, query, other_child, new_mindist, dists, eps_error, filter, depth + 1)
    {
        return false;
    }

    dists[idx] = dst;
    true
}

/// Port of `searchLevel` (nanoflann.hpp ~1228-1287) as an EXPLICIT-STACK
/// iteration (M2.5 task 2) instead of native recursion. The recursive
/// shape was:
///
/// ```text
/// fn search_level(node, mindist):
///     if leaf: scan, return early(false) on abort, else return true
///     compute idx, best_child, other_child, cut_dist
///     if !search_level(best_child, mindist): return false      // (i)
///     dst = dists[idx]; mindist = mindist + cut_dist - dst; dists[idx] = cut_dist
///     if mindist*eps <= worst_dist && !search_level(other_child, mindist):
///         return false                                          // (ii), no restore
///     dists[idx] = dst                                          // (iii) restore: normal AND pruned
///     return true
/// ```
///
/// Converted 1:1 onto an explicit `FrameStack`: descending into
/// `best_child` == pushing a `Frame` (capturing `idx`/`dst`/`cut_dist`
/// /`other_child`/the entry `mindist`) and continuing the descent loop from
/// `best_child`; reaching a leaf breaks out of the descent loop into a
/// "resume" loop that pops/patches frames. The three exit paths above map
/// onto three (and only three) places `dists` is touched or a `false`
/// escapes:
///
/// | # | Recursive | Iterative |
/// |---|---|---|
/// | (i) | best_child call returns `false` -> `return false` immediately, `dists[idx]` never touched by this node | leaf-scan abort inside the descent loop -> `return false` immediately; every `Frame` currently on the stack (whether `PostBest`, meaning we're still "inside" its best-child subtree and never wrote `dists[idx]`, or `PostOther`, meaning it already wrote `dists[frame.idx] = frame.cut_dist` and hasn't restored yet) is simply abandoned, exactly matching the unwound-but-not-restored recursive stack frames |
/// | (ii) | other_child call returns `false` -> `return false`, `dists[idx]` LEFT as `cut_dist` (no restore) | same leaf-scan abort as (i), reached while resuming a `Phase::PostOther` frame; that frame (and any of its own ancestors) is likewise abandoned without its restore running |
/// | (iii) | falls through to `dists[idx] = dst` on EITHER a pruned eps test OR a normal (non-aborting) `other_child` return | `Phase::PostBest`, eps test fails -> restore + pop immediately (no `other_child` visit); `Phase::PostOther` reached normally (its `other_child` subtree's descent loop broke out via a normal leaf scan, not an abort) -> restore + pop |
///
/// Traversal order is unchanged: the descent loop always continues into
/// `best_child` first (exactly the first recursive call), and a
/// `Phase::PostBest` frame that is not pruned always initiates the
/// `other_child` descent next — for the EXACT SAME node, at the EXACT SAME
/// point (immediately after `best_child`'s subtree fully finishes) — before
/// any node further up the (conceptual) call stack is resumed, because
/// `FrameStack` is strictly LIFO: the most recently pushed frame is always
/// the first one `top_mut`/`pop` sees.
///
/// `dst` is captured at push time (right when `Frame` is built, BEFORE
/// descending into `best_child`) rather than after `best_child`'s subtree
/// returns. Provably equivalent to reading it after: nothing in
/// `best_child`'s subtree can leave `dists[idx]` changed by the time
/// control returns to this frame on a non-aborting path — a descendant
/// interior node that happens to share this same split axis writes
/// `dists[idx] = its_own_cut_dist` only transiently, inside its OWN
/// `Phase::PostOther` window, and unconditionally restores it back to
/// whatever value it read on entry (which is this same `dst`, by the same
/// inductive argument, all the way down) before its own frame is popped —
/// i.e. by the time THIS frame's `best_child` subtree is fully unwound,
/// `dists[idx]` is provably back to its pre-descent value. (On an aborting
/// path this frame's `dst` is never read at all — the function returns
/// `false` straight out of the leaf scan before `Phase::PostBest` is ever
/// resumed for this frame.)
///
/// Frame storage: a fixed inline array with a heap-spill `Vec` fallback
/// (see `FrameStack`) — depth is bounded by tree depth, and task 6's
/// builder can (deliberately, in its own adversarial stress tests) produce
/// degenerate trees far deeper than any balanced tree over a realistic
/// point count, so a plain fixed array without a spill would not be sound
/// in general (see `heavy_query_degenerate_trees` below, depth ~16 794, and
/// `spill_boundary_deep_tree_knn_and_radius_match_brute_force`, which
/// specifically exercises the spill transition at a much cheaper depth).
///
/// Returns `false` to abort (propagated from `ResultSet::add_point`
/// returning `false`, nanoflann.hpp ~1245-1250), `true` to continue.
#[allow(clippy::too_many_arguments)]
fn search_level_explicit<T, D, DS, M, Idx, R, F>(
    ctx: &SearchCtx<T, D, DS, M, Idx>,
    result: &mut R,
    query: &[T],
    mut node_idx: u32,
    mut mindist: M::DistanceType,
    dists: &mut [M::DistanceType],
    eps_error: M::DistanceType,
    filter: &F,
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
    R: ResultSet<M::DistanceType, Idx>,
    F: PointFilter<Idx>,
{
    let mut stack: FrameStack<M::DistanceType> = FrameStack::new();

    // Test-only instrumentation: `FrameStack` is a purely local variable
    // here (by design — see its doc), so a test that wants to assert a
    // query actually pushed past `SEARCH_STACK_INLINE_CAPACITY` (i.e. truly
    // exercised the `overflow` spill, not just "the tree is deep") needs a
    // side channel to read `stack.max_depth_seen` after the call returns.
    // `record_max_stack_depth!` publishes it into a thread-local right
    // before every return site (defined just below `search_level`).
    macro_rules! record_max_stack_depth {
        () => {
            #[cfg(test)]
            {
                LAST_SEARCH_STACK_MAX_DEPTH.with(|c| c.set(stack.max_depth_seen));
            }
        };
    }

    'walk: loop {
        // ---- Descend from `node_idx` to a leaf, pushing one `Frame` per
        // interior node passed through — exactly the recursive function's
        // "compute best/other child, recurse into best_child" prologue. ----
        loop {
            let node = &ctx.nodes[node_idx as usize];

            if node.is_leaf() {
                let (left, right) = node.leaf_range();
                // Same bounds-check-elision restructuring as before this
                // conversion: index the sub-slice once, then walk it by
                // pointer.
                for &accessor in &ctx.vind[left..right] {
                    if !filter.is_active(accessor) {
                        continue;
                    }
                    let dist = ctx.metric.eval(query, ctx.ds, accessor.to_usize(), ctx.dim);
                    // C++ (nanoflann.hpp ~1259): `result_set.worstDist()` is
                    // called LIVE, inline in the loop condition, on every
                    // iteration — NOT hoisted into a local cached before
                    // the loop. Mirrored exactly, same as before.
                    if dist < result.worst_dist() && !result.add_point(dist, accessor) {
                        // Abort: exit paths (i)/(ii) above — propagate
                        // immediately, no frame's restore runs.
                        record_max_stack_depth!();
                        return false;
                    }
                }
                break;
            }

            // Which child branch should be taken first? (unchanged)
            let idx = node.split_dim();
            assert!(idx < dists.len());
            let val = query[idx];
            let diff1 = val - node.div_low();
            let diff2 = val - node.div_high();

            let (c1, c2) = node.children();
            let (best_child, other_child, cut_dist) = if (diff1 + diff2) < T::default() {
                (c1, c2, ctx.metric.accum_dist(val, node.div_high(), idx))
            } else {
                (c2, c1, ctx.metric.accum_dist(val, node.div_low(), idx))
            };

            stack.push(Frame {
                idx,
                dst: dists[idx],
                cut_dist,
                other_child,
                entry_mindist: mindist,
                phase: Phase::PostBest,
            });
            node_idx = best_child;
        }

        // ---- Resume: pop/patch frames until one initiates an
        // `other_child` descent (continue the outer loop from there), or
        // the stack empties (the whole search is done). ----
        loop {
            let Some(frame) = stack.top_mut() else {
                record_max_stack_depth!();
                return true;
            };
            match frame.phase {
                Phase::PostBest => {
                    // = the recursive function's post-best-child block:
                    // `let dst = dists[idx]; mindist = mindist + cut_dist -
                    // dst; dists[idx] = cut_dist;` (unconditional), then the
                    // eps-pruned `if`.
                    let idx = frame.idx;
                    let new_mindist = frame.entry_mindist + frame.cut_dist - frame.dst;
                    dists[idx] = frame.cut_dist;
                    if new_mindist * eps_error <= result.worst_dist() {
                        // Not pruned: descend into `other_child`. Leave
                        // THIS frame in place (flipped to `PostOther`) so
                        // its restore — exit path (iii) — runs once that
                        // subtree finishes normally, or is silently
                        // abandoned on an abort — exit path (ii).
                        frame.phase = Phase::PostOther;
                        node_idx = frame.other_child;
                        mindist = new_mindist;
                        continue 'walk;
                    }
                    // Pruned: exit path (iii)'s other branch — the
                    // recursive `&&` short-circuits, falling through to
                    // the SAME unconditional `dists[idx] = dst` restore.
                    dists[idx] = frame.dst;
                    stack.pop();
                }
                Phase::PostOther => {
                    // `other_child`'s subtree returned normally (an abort
                    // takes the early `return false` above and never
                    // reaches here) — exit path (iii): restore, keep
                    // resuming the parent.
                    dists[frame.idx] = frame.dst;
                    stack.pop();
                }
            }
        }
    }
}

/// = static `findWithinBox` (nanoflann.hpp ~2030-2074): all points inside the
/// axis-aligned box, boundaries INCLUSIVE, output in traversal order, NEVER
/// sorted, no `SearchParameters`. `bounds.len() == ctx.dim`. Empty tree -> 0,
/// `out` cleared.
///
/// Explicit `Vec<u32>` node stack, seeded with the root (always arena index
/// 0, per `SearchCtx`'s doc comment). Push order per interior node mirrors
/// the source EXACTLY: `child1` pushed first (if `bounds[divfeat].low <=
/// node.div_low()`), then `child2` (if `bounds[divfeat].high >=
/// node.div_high()`) — both INCLUSIVE compares against the node's div
/// bounds (nanoflann.hpp:2068-2069). Because this is a LIFO stack, when both
/// pushes happen `child2` is popped and processed FIRST; this determines the
/// output order (locked by `find_within_box_exact_traversal_order` above).
///
/// No `PointFilter` parameter: C++'s static `findWithinBox` has no
/// `isActive`-style hook (unlike `findNeighbors`), so this port has none.
///
/// C++'s inner loop honors `result.addPoint`'s bool return as a mid-loop
/// early-out (nanoflann.hpp:2053-2058: `if (!result.addPoint(...)) return
/// result.size();`) — if the result set signals "stop", the search returns
/// immediately without visiting the rest of the stack. `BoxResultSet::add`
/// here returns `()` (see `result_set.rs`), not `bool`: the built-in
/// collector never wants to stop early (it always accepts every point), so
/// that early-out is unobservable for this collector and is deliberately
/// not ported.
pub(crate) fn find_within_box<T, D, DS, M, Idx>(
    ctx: &SearchCtx<'_, T, D, DS, M, Idx>,
    bounds: &[Interval<T>],
    out: &mut Vec<Idx>,
) -> usize
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
{
    let mut result = BoxResultSet::new(out);

    // C++ (nanoflann.hpp:2032): `if (this->size(*this) == 0) return 0;` —
    // an empty dataset has no built tree (no root node) to walk.
    if ctx.nodes.is_empty() {
        return 0;
    }

    debug_assert_eq!(bounds.len(), ctx.dim.dim(), "bounds.len() != dim");

    let mut stack: Vec<u32> = vec![0];

    while let Some(node_idx) = stack.pop() {
        let node = &ctx.nodes[node_idx as usize];

        if node.is_leaf() {
            let (left, right) = node.leaf_range();
            // Same bounds-check-elision restructuring as search_level's
            // leaf loop above.
            for &accessor in &ctx.vind[left..right] {
                if contains_point(ctx, bounds, accessor) {
                    result.add(accessor);
                }
            }
        } else {
            let idx = node.split_dim();
            let low_bound = node.div_low();
            let high_bound = node.div_high();
            let (child1, child2) = node.children();

            if bounds[idx].low <= low_bound {
                stack.push(child1);
            }
            if bounds[idx].high >= high_bound {
                stack.push(child2);
            }
        }
    }

    result.size()
}

/// Port of `contains` (nanoflann.hpp:2182-2192): a point is inside the box
/// iff EVERY dimension's coordinate satisfies `Interval::contains` (the
/// C++-parity inclusive `point < low || point > high` -> reject predicate,
/// see `bbox.rs`).
// `i` indexes BOTH `bounds` and `ctx.ds.point_component(_, i)` (a trait
// method, not a slice) in the same per-axis order as nanoflann.hpp:2182-2192.
#[allow(clippy::needless_range_loop)]
fn contains_point<T, D, DS, M, Idx>(
    ctx: &SearchCtx<'_, T, D, DS, M, Idx>,
    bounds: &[Interval<T>],
    idx: Idx,
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
{
    let dim = ctx.dim.dim();
    if let Some(row) = ctx.ds.point_row(idx.to_usize()) {
        if row.len() >= dim {
            for i in 0..dim {
                if !bounds[i].contains(row[i]) {
                    return false;
                }
            }
            return true;
        }
    }
    for i in 0..dim {
        let point = ctx.ds.point_component(idx.to_usize(), i);
        if !bounds[i].contains(point) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::compute_bounding_box;
    use crate::build::{init_vind, SubtreeBuilder};
    use crate::dim::DynDim;
    use crate::filter::AcceptAll;
    use crate::metric::L2;
    use crate::result_set::KnnResultSet;

    /// Build a full tree over `pts` (T = f64) via `SubtreeBuilder`, base 0,
    /// identity `vind`, tight root bbox from `compute_bounding_box`. The
    /// root is always arena index 0 (see `SearchCtx` doc comment).
    fn build_tree<const N: usize>(
        pts: &[[f64; N]],
        leaf_max_size: usize,
    ) -> (Vec<u32>, Vec<Node<f64>>, Vec<Interval<f64>>) {
        let dim = N;
        let n = pts.len();
        let mut vind: Vec<u32> = init_vind(n);
        let mut bbox = vec![
            Interval {
                low: 0.0,
                high: 0.0
            };
            dim
        ];
        compute_bounding_box(&pts, dim, &mut bbox);
        let mut arena = Vec::new();
        {
            let mut builder = SubtreeBuilder {
                ds: &pts,
                dim,
                leaf_max_size,
                base: 0,
                vind: &mut vind,
                arena: &mut arena,
            };
            let root = builder.build(&mut bbox);
            assert_eq!(root, 0, "root is expected to always be arena index 0");
        }
        (vind, arena, bbox)
    }

    // ---------------------------------------------------------------
    // Test 1: brute-force property (RED evidence for now — stub always
    // returns `false` / touches nothing, so this fails against ANY non-empty
    // case).
    // ---------------------------------------------------------------

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn brute_force_knn<const N: usize>(
        pts: &[[f64; N]],
        query: &[f64; N],
        k: usize,
    ) -> (Vec<u32>, Vec<f64>) {
        let dim = N;
        let metric = L2;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            for i in 0..pts.len() {
                let d = metric.eval(query.as_slice(), &pts, i, DynDim(dim));
                rs.add_point(d, i as u32);
            }
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists)
    }

    fn tree_knn<const N: usize>(
        vind: &[u32],
        arena: &[Node<f64>],
        bbox: &[Interval<f64>],
        pts: &[[f64; N]],
        query: &[f64; N],
        k: usize,
        eps: f32,
    ) -> (Vec<u32>, Vec<f64>, bool) {
        let dim = N;
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(dim),
            nodes: arena,
            vind,
            root_bbox: bbox,
        };
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        let full;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = vec![0.0f64; dim];
            let params = SearchParams { eps, sorted: true };
            full = find_neighbors(
                &ctx,
                &mut rs,
                query.as_slice(),
                &params,
                &AcceptAll,
                &mut scratch,
            );
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists, full)
    }

    fn brute_force_radius<const N: usize>(
        pts: &[[f64; N]],
        query: &[f64; N],
        radius: f64,
    ) -> Vec<(u32, f64)> {
        let dim = N;
        let metric = L2;
        let mut out: Vec<(u32, f64)> = (0..pts.len())
            .filter_map(|i| {
                let d = metric.eval(query.as_slice(), &pts, i, DynDim(dim));
                if d < radius {
                    Some((i as u32, d))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        out
    }

    fn tree_radius<const N: usize>(
        vind: &[u32],
        arena: &[Node<f64>],
        bbox: &[Interval<f64>],
        pts: &[[f64; N]],
        query: &[f64; N],
        radius: f64,
    ) -> Vec<(u32, f64)> {
        let dim = N;
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(dim),
            nodes: arena,
            vind,
            root_bbox: bbox,
        };
        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(radius, &mut items);
            let mut scratch = vec![0.0f64; dim];
            let params = SearchParams {
                eps: 0.0,
                sorted: true,
            };
            find_neighbors(
                &ctx,
                &mut rs,
                query.as_slice(),
                &params,
                &AcceptAll,
                &mut scratch,
            );
        }
        items
            .into_iter()
            .map(|it| (it.index, it.distance))
            .collect()
    }

    fn run_brute_force_case<const N: usize>(seed: u64, n: usize, k: usize, leaf_max_size: usize) {
        let mut rng = Lcg(seed);
        let pts: Vec<[f64; N]> = (0..n)
            .map(|_| {
                let mut p = [0.0f64; N];
                for v in p.iter_mut() {
                    *v = rng.next_f64() * 200.0 - 100.0;
                }
                p
            })
            .collect();
        let mut query = [0.0f64; N];
        for v in query.iter_mut() {
            *v = rng.next_f64() * 200.0 - 100.0;
        }

        let (vind, arena, bbox) = build_tree(&pts, leaf_max_size);
        let (tree_idx, tree_dists, tree_full) =
            tree_knn(&vind, &arena, &bbox, &pts, &query, k, 0.0);
        let (bf_idx, bf_dists) = brute_force_knn(&pts, &query, k);

        assert_eq!(
            tree_idx, bf_idx,
            "seed={seed} n={n} dim={N} k={k}: index mismatch"
        );
        assert_eq!(
            tree_dists, bf_dists,
            "seed={seed} n={n} dim={N} k={k}: distance mismatch"
        );
        assert_eq!(
            tree_full,
            bf_idx.len() == k,
            "seed={seed} n={n} dim={N} k={k}: full() mismatch"
        );
    }

    #[test]
    fn brute_force_property_200_seeded_cases() {
        let mut seed = 0xF00Du64;
        for case in 0..200u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let dim_choice = case % 3;
            let mut rng = Lcg(seed ^ 0xABCD);
            let n = 1 + (rng.next_f64() * 200.0) as usize;
            let n = n.clamp(1, 200);
            let k_max = n.min(20);
            let k = 1 + (rng.next_f64() * (k_max as f64)) as usize;
            let k = k.min(k_max).max(1);

            match dim_choice {
                0 => run_brute_force_case::<2>(seed, n, k, 10),
                1 => run_brute_force_case::<3>(seed, n, k, 10),
                _ => run_brute_force_case::<8>(seed, n, k, 10),
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 2: k > n
    // ---------------------------------------------------------------

    #[test]
    fn k_greater_than_n_returns_exactly_n_results_and_not_full() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let (idx, _dists, full) = tree_knn(&vind, &arena, &bbox, &pts, &[0.5, 0.5], 10, 0.0);
        assert_eq!(idx.len(), 3);
        assert!(!full);
    }

    // ---------------------------------------------------------------
    // Test 3: query far outside root bbox
    // ---------------------------------------------------------------

    #[test]
    fn query_far_outside_bbox_50_seeded_cases() {
        let mut seed = 0xBEEFu64;
        for case in 0..50u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut rng = Lcg(seed);
            let n = 1 + (rng.next_f64() * 50.0) as usize;
            let pts: Vec<[f64; 3]> = (0..n)
                .map(|_| {
                    [
                        rng.next_f64() * 20.0,
                        rng.next_f64() * 20.0,
                        rng.next_f64() * 20.0,
                    ]
                })
                .collect();
            let sign = if case % 2 == 0 { 1.0 } else { -1.0 };
            let query = [
                sign * 1000.0 + rng.next_f64(),
                sign * 1000.0 + rng.next_f64(),
                sign * 1000.0 + rng.next_f64(),
            ];
            let k = 1 + (rng.next_f64() * (n.min(5) as f64)) as usize;
            let k = k.min(n).max(1);

            let (vind, arena, bbox) = build_tree(&pts, 10);
            let (tree_idx, tree_dists, _) = tree_knn(&vind, &arena, &bbox, &pts, &query, k, 0.0);
            let (bf_idx, bf_dists) = brute_force_knn(&pts, &query, k);
            assert_eq!(tree_idx, bf_idx, "case={case}");
            assert_eq!(tree_dists, bf_dists, "case={case}");
        }
    }

    // ---------------------------------------------------------------
    // Test 9: empty tree
    // ---------------------------------------------------------------

    #[test]
    fn empty_tree_returns_false_and_leaves_result_untouched() {
        let pts: &[[f64; 2]] = &[];
        let arena: Vec<Node<f64>> = Vec::new();
        let vind: Vec<u32> = Vec::new();
        let bbox: Vec<Interval<f64>> = vec![
            Interval {
                low: 0.0,
                high: 0.0
            };
            2
        ];
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut indices = [99u32; 3];
        let mut dists = [77.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
        let mut scratch = [0.0f64; 2];
        let params = SearchParams::default();

        let full = find_neighbors(
            &ctx,
            &mut rs,
            &[0.0, 0.0],
            &params,
            &AcceptAll,
            &mut scratch,
        );
        assert!(!full);
        assert_eq!(indices, [99, 99, 99]);
        assert_eq!(dists, [77.0, 77.0, 77.0]);
    }

    // ---------------------------------------------------------------
    // Test 10: leaf_max_size 1 and n
    // ---------------------------------------------------------------

    #[test]
    fn leaf_max_size_one_brute_force_equality_20_seeded_cases() {
        for case in 0..20u64 {
            let seed = 0x1EAF_0001u64.wrapping_add(case.wrapping_mul(747796405));
            run_brute_force_case::<3>(seed, 1 + (case as usize) * 3, 1 + (case as usize % 5), 1);
        }
    }

    #[test]
    fn leaf_max_size_n_brute_force_equality_20_seeded_cases() {
        for case in 0..20u64 {
            let seed = 0x1EAF_000Fu64.wrapping_add(case.wrapping_mul(747796405));
            let n = 1 + (case as usize) * 3;
            run_brute_force_case::<3>(seed, n, 1 + (case as usize % n.max(1)), n.max(1));
        }
    }

    // ---------------------------------------------------------------
    // Test 6: RKNN
    // ---------------------------------------------------------------

    #[test]
    fn rknn_partial_coverage_returns_false() {
        // 5 points at squared distances 1,4,9,16,25 from the query along one
        // axis; radius 10.0 covers only the first 3 (dist < 10.0: 1,4,9).
        let pts: Vec<[f64; 1]> = vec![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut indices = [0u32; 5];
        let mut dists = [0.0f64; 5];
        let full;
        {
            let mut rs =
                crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 10.0);
            let mut scratch = [0.0f64; 1];
            full = find_neighbors(
                &ctx,
                &mut rs,
                &[0.0],
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
            assert_eq!(rs.size(), 3);
        }
        assert!(!full);
        assert_eq!(&indices[..3], &[0, 1, 2]);
    }

    #[test]
    fn rknn_full_coverage_returns_closest_k_and_true() {
        let pts: Vec<[f64; 1]> = vec![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0],
        ];
        let (vind, arena, bbox) = build_tree(&pts, 3);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut indices = [0u32; 5];
        let mut dists = [0.0f64; 5];
        let full;
        {
            // radius 100.0 (squared) covers all 10 points.
            let mut rs =
                crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 100.0);
            let mut scratch = [0.0f64; 1];
            full = find_neighbors(
                &ctx,
                &mut rs,
                &[0.0],
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
            assert_eq!(rs.size(), 5);
        }
        assert!(full);
        // Closest 5 to 0.0 are points at 1,2,3,4,5 -> indices 0..5.
        assert_eq!(indices, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn rknn_exact_radius_boundary_excludes_point() {
        // Single point at exact distance^2 == radius from the query.
        let pts: Vec<[f64; 1]> = vec![[2.0]]; // dist^2 from 0.0 = 4.0
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut indices = [0u32; 1];
        let mut dists = [0.0f64; 1];
        {
            let mut rs =
                crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 4.0);
            let mut scratch = [0.0f64; 1];
            find_neighbors(
                &ctx,
                &mut rs,
                &[0.0],
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
            assert_eq!(
                rs.size(),
                0,
                "point at exact squared radius must be excluded (dist < worst_dist gate)"
            );
        }
    }

    // ---------------------------------------------------------------
    // Test 5: radius strict boundary + sort order
    // ---------------------------------------------------------------

    #[test]
    fn radius_strict_boundary_excludes_exact_and_includes_next_up() {
        // Grid points; one point at exact squared distance 4.0 from query (0,0): (2,0).
        let pts: Vec<[f64; 2]> = vec![[2.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(4.0f64, &mut items);
            let mut scratch = [0.0f64; 2];
            find_neighbors(
                &ctx,
                &mut rs,
                &[0.0, 0.0],
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
        }
        assert!(
            !items.iter().any(|it| it.index == 0),
            "point at exact squared radius 4.0 must be absent"
        );

        let radius_next_up = f64::from_bits(4.0f64.to_bits() + 1);
        let mut items2 = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(radius_next_up, &mut items2);
            let mut scratch = [0.0f64; 2];
            find_neighbors(
                &ctx,
                &mut rs,
                &[0.0, 0.0],
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
        }
        assert!(
            items2.iter().any(|it| it.index == 0),
            "point at radius next_up(4.0) must be present"
        );
    }

    #[test]
    fn radius_sorted_true_gives_ascending_distances() {
        let pts: Vec<[f64; 2]> = vec![[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(100.0f64, &mut items);
            let mut scratch = [0.0f64; 2];
            let params = SearchParams {
                eps: 0.0,
                sorted: true,
            };
            find_neighbors(
                &ctx,
                &mut rs,
                &[0.0, 0.0],
                &params,
                &AcceptAll,
                &mut scratch,
            );
        }
        let ds: Vec<f64> = items.iter().map(|it| it.distance).collect();
        let mut sorted = ds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ds, sorted, "sorted=true must yield ascending distances");
    }

    #[test]
    fn radius_sorted_false_gives_tree_traversal_order() {
        // Small fixed 2-leaf tree, hand-derived traversal order.
        //
        // Points (dim-1, original index -> value): 0->0.0, 1->1.0, 2->10.0,
        // 3->11.0. Initial vind=[0,1,2,3], bbox=[0.0,11.0].
        //
        // middle_split: only candidate dim is 0 (trivially, dim=1); cutval =
        // clamp((0+11)/2, min=0, max=11) = 5.5. plane_split(ind=[0,1,2,3],
        // cutfeat=0, cutval=5.5), tracing nanoflann's 3-cursor partition
        // (build.rs::plane_split) exactly:
        //   left=0 mid=0 right=3
        //   mid=0: v[ind[0]]=v[0]=0.0  < 5.5 -> swap(0,0); left=1; mid=1   ind unchanged
        //   mid=1: v[ind[1]]=v[1]=1.0  < 5.5 -> swap(1,1); left=2; mid=2   ind unchanged
        //   mid=2: v[ind[2]]=v[2]=10.0 > 5.5 -> swap(2,3); right=2        ind=[0,1,3,2]
        //   mid=2: v[ind[2]]=v[3]=11.0 > 5.5 -> swap(2,2); right=1        ind unchanged
        //   mid=2 > right=1 -> loop ends. lim1=left=2, lim2=mid=2.
        // half = 4/2 = 2. lim1(2)>half(2)? no. lim2(2)<half(2)? no. -> index=half=2.
        //
        // So the split places vind[0..2]=[0,1] in the LEFT leaf (values
        // 0.0,1.0, in that order) and vind[2..4]=[3,2] in the RIGHT leaf —
        // note the swap left the right partition as [3,2], NOT input order
        // [2,3] (values 11.0,10.0 in that order): `plane_split` is not a
        // stable partition, per build.rs's own documented swap semantics.
        //
        // finalizeSplitNode: divlow = max(left subtree, dim0) = max(0,1) =
        // 1.0; divhigh = min(right subtree, dim0) = min(11,10) = 10.0
        // (order-independent, unaffected by the swap above).
        //
        // Query = 0.4 (near leaf A). Root: diff1=val-divlow=0.4-1.0=-0.6;
        // diff2=val-divhigh=0.4-10.0=-9.6; sum=-10.2 < 0 -> best_child =
        // child1 = left (A), visited FIRST, leaf loop scans its range in
        // vind order [0,1] (unsorted by distance). On the `<0` branch
        // cut_dist uses div_HIGH (it's the entry cost for the OTHER child,
        // which is child2/right/B here): cut_dist=accum_dist(0.4,10.0)=
        // (0.4-10.0)^2=92.16 — NOT div_low (that's the other branch's
        // formula, for when child2/right is picked as best instead). Both
        // A's points are well within radius 200 -> added in order 0, then 1.
        // mindist = 0 + 92.16 - 0 = 92.16; eps=0 -> 92.16*1 <= 200 -> visit
        // other (B), leaf loop scans ITS vind range [3,2] (per the swap
        // above) -> added 3, then 2.
        //
        // Expected unsorted (traversal) order: indices [0, 1, 3, 2].
        let pts: Vec<[f64; 1]> = vec![[0.0], [1.0], [10.0], [11.0]];
        let (vind, arena, bbox) = build_tree(&pts, 2);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(200.0f64, &mut items);
            let mut scratch = [0.0f64; 1];
            let params = SearchParams {
                eps: 0.0,
                sorted: false,
            };
            find_neighbors(&ctx, &mut rs, &[0.4], &params, &AcceptAll, &mut scratch);
        }
        let order: Vec<u32> = items.iter().map(|it| it.index).collect();
        assert_eq!(order, vec![0, 1, 3, 2]);
    }

    // ---------------------------------------------------------------
    // Test 7: abort plumbing
    // ---------------------------------------------------------------

    struct AbortAfterN {
        limit: usize,
        adds: usize,
    }

    impl ResultSet<f64, u32> for AbortAfterN {
        fn worst_dist(&self) -> f64 {
            f64::MAX
        }
        fn add_point(&mut self, _dist: f64, _index: u32) -> bool {
            self.adds += 1;
            self.adds < self.limit
        }
        fn full(&self) -> bool {
            self.adds >= self.limit
        }
        fn size(&self) -> usize {
            self.adds
        }
    }

    #[test]
    fn abort_plumbing_stops_search_and_propagates_full() {
        // leaf_max_size=1 so every point lives in its own leaf; with a
        // ResultSet that always accepts (worst_dist == MAX) but returns
        // `false` from add_point after the 3rd add, the search must stop
        // dead after exactly 3 adds, regardless of how many points remain.
        let pts: Vec<[f64; 1]> = (0..20).map(|i| [i as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 1);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut rs = AbortAfterN { limit: 3, adds: 0 };
        let mut scratch = [0.0f64; 1];
        let full = find_neighbors(
            &ctx,
            &mut rs,
            &[0.0],
            &SearchParams::default(),
            &AcceptAll,
            &mut scratch,
        );

        assert_eq!(
            rs.adds, 3,
            "exactly 3 adds should have happened before the 4th add_point aborts"
        );
        assert_eq!(full, rs.full());
    }

    // ---------------------------------------------------------------
    // Test 8: filter plumbing
    // ---------------------------------------------------------------

    struct RejectEven;
    impl PointFilter<u32> for RejectEven {
        fn is_active(&self, idx: u32) -> bool {
            idx % 2 == 1
        }
    }

    #[test]
    fn filter_plumbing_rejects_even_indices() {
        let mut rng = Lcg(0x0DD_F17E7u64);
        let n = 37;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                [
                    rng.next_f64() * 50.0,
                    rng.next_f64() * 50.0,
                    rng.next_f64() * 50.0,
                ]
            })
            .collect();
        let query = [
            rng.next_f64() * 50.0,
            rng.next_f64() * 50.0,
            rng.next_f64() * 50.0,
        ];

        let (vind, arena, bbox) = build_tree(&pts, 4);
        let metric = L2;
        let pts: &[[f64; 3]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(3),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let k = 10;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 3];
            find_neighbors(
                &ctx,
                &mut rs,
                &query,
                &SearchParams::default(),
                &RejectEven,
                &mut scratch,
            );
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);

        assert!(
            indices.iter().all(|&i| i % 2 == 1),
            "even indices leaked through the filter"
        );

        // Brute force restricted to odd indices only. Uses `metric.eval`
        // (not a hand-rolled sum) so the summation order matches the tree
        // search bit-for-bit — L2's 4-wide-unrolled remainder loop does NOT
        // sum components in index order for dim=3 (see metric.rs), so a
        // naive `(0..3).sum()` can differ by an ULP.
        let mut scored: Vec<(f64, u32)> = (0..pts.len())
            .filter(|&i| i % 2 == 1)
            .map(|i| (metric.eval(&query, &pts, i, DynDim(3)), i as u32))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        scored.truncate(k);
        let want_indices: Vec<u32> = scored.iter().map(|(_, i)| *i).collect();
        let want_dists: Vec<f64> = scored.iter().map(|(d, _)| *d).collect();

        assert_eq!(indices, want_indices);
        assert_eq!(dists, want_dists);
    }

    // ---------------------------------------------------------------
    // Test 11: SO2 end-to-end
    // ---------------------------------------------------------------

    #[test]
    fn so2_end_to_end_wrapped_angle_wins_over_raw_difference() {
        use crate::metric::SO2;
        let pi = core::f64::consts::PI;
        // dim-2: first component varies wildly (irrelevant to SO2, which
        // only looks at the LAST dim), second component is the angle.
        // Point A: angle = -pi + 0.05 (just past -pi, wraps close to +pi).
        // Point B: angle = pi - 0.5 (raw-closer to query's pi - 0.1 by plain
        // subtraction, but farther by wrapped angular distance).
        let query = [999.0, pi - 0.1];
        let pts: Vec<[f64; 2]> = vec![
            [12345.0, -pi + 0.05], // wrapped dist to query: pi-0.1 -> -pi+0.05 wraps to ~0.15
            [-6789.0, pi - 0.5],   // raw dist: 0.4 (no wrap needed)
        ];

        // Sanity: point 0's wrapped distance (~0.15) < point 1's distance (0.4).
        let metric = SO2;
        let d0 = metric.accum_dist(query[1], pts[0][1], 1);
        let d1 = metric.accum_dist(query[1], pts[1][1], 1);
        assert!(
            d0 < d1,
            "test setup: expected point 0 to be the wrapped-nearer angle, d0={d0} d1={d1}"
        );

        let (vind, arena, bbox) = build_tree(&pts, 10);
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let mut indices = [0u32; 1];
        let mut dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 2];
            find_neighbors(
                &ctx,
                &mut rs,
                &query,
                &SearchParams::default(),
                &AcceptAll,
                &mut scratch,
            );
        }
        assert_eq!(
            indices,
            [0],
            "wrapped-nearer angle should win, first component ignored"
        );
    }

    // ---------------------------------------------------------------
    // Test 4: eps
    // ---------------------------------------------------------------

    struct CountingFilter(core::cell::Cell<usize>);
    impl PointFilter<u32> for CountingFilter {
        fn is_active(&self, _idx: u32) -> bool {
            self.0.set(self.0.get() + 1);
            true
        }
    }

    #[test]
    fn eps_zero_visits_both_leaves_eps_large_visits_one() {
        // Fix round 1, finding 3: the original version of this test used 8
        // widely-separated points, which made it PASS identically at eps=0
        // too (every prune check failed regardless of eps_error, since
        // mindist >> worst_dist by a huge margin at every node) — i.e. it
        // never actually exercised eps's effect, only that AcceptAll-style
        // full traversal reaches one leaf. Replaced with a 2-point,
        // leaf_max_size=1 tree hand-picked so the SINGLE prune check that
        // exists (at the root) sits exactly ON the eps=0 boundary, making
        // the eps=0 vs eps=10 outcomes genuinely different.
        //
        // Points -0.1 (index 0) and 0.1 (index 1), query 0.0. middle_split
        // (bbox=[-0.1,0.1], cutval=0.0) puts index0 in the LEFT leaf, index1
        // in the RIGHT leaf; divlow=-0.1 (left's own point), divhigh=0.1
        // (right's own point). Heuristic: diff1=0-(-0.1)=0.1, diff2=0-0.1=
        // -0.1, sum=0.0, NOT < 0 -> best=RIGHT(index1), cut_dist (for
        // OTHER=LEFT)=accum_dist(0,-0.1)=(0.1)^2=0.01=mindist.
        //
        // worst_dist after visiting RIGHT (k=1): dist to 0.1 = (0-0.1)^2 =
        // 0.01 (exactly equal to mindist — a genuine tie, not a margin).
        //
        // eps=0 (eps_error=1.0): 0.01*1.0 <= 0.01 -> TRUE (inclusive `<=`)
        // -> visits LEFT too -> both points seen -> count == 2.
        // eps=10 (eps_error=11.0): 0.01*11.0=0.11 <= 0.01 -> FALSE -> LEFT
        // pruned -> only RIGHT seen -> count == 1.
        let pts: Vec<[f64; 1]> = vec![[-0.1], [0.1]];
        let (vind, arena, bbox) = build_tree(&pts, 1);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let run = |eps: f32| -> (usize, u32) {
            let filter = CountingFilter(core::cell::Cell::new(0));
            let mut indices = [0u32; 1];
            let mut dists = [0.0f64; 1];
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 1];
            let params = SearchParams { eps, sorted: true };
            find_neighbors(&ctx, &mut rs, &[0.0], &params, &filter, &mut scratch);
            (filter.0.get(), indices[0])
        };

        let (count_exact, nearest_exact) = run(0.0);
        assert_eq!(
            count_exact, 2,
            "eps=0: the root's prune check is an exact tie (<=), both leaves should be visited"
        );
        assert_eq!(nearest_exact, 1, "nearest to 0.0 is index 1 (0.1)");

        let (count_large, nearest_large) = run(10.0);
        assert_eq!(
            count_large, 1,
            "eps=10: the tie should be broken by the widened eps_error, pruning the other leaf"
        );
        assert_eq!(
            nearest_large, 1,
            "eps=10 should still find the correct nearest point in the one visited leaf"
        );
    }

    // A dim-1 dataset where the pruned branch holds the true nearest (to
    // exercise the eps-approximate bound for k=1) is UNCONSTRUCTIBLE in
    // dim-1 — proof below.
    //
    // Proof sketch: at any interior node, `div_low` is the ACTUAL max
    // coordinate of a real point in the left subtree (build.rs's
    // `finalizeSplitNode` sets it from the left subtree's own tight bbox,
    // not merely a bound), and `div_high` is likewise the real min of the
    // right subtree. The heuristic condition `(diff1+diff2) < 0` is
    // algebraically identical to `|val - div_low| < |val - div_high|`
    // (expand both cases: `val` inside the [div_low,div_high] gap, or
    // outside it on either side — the two forms agree in every case). Since
    // `div_low`/`div_high` are themselves points reachable inside their
    // respective subtrees, `TRUE_MIN(subtree) <= dist(val, its own
    // boundary)` always (the boundary point is a candidate, so it can only
    // be beaten or tied, never exceeded, by the subtree's true minimum).
    // Chaining: `TRUE_MIN(chosen_best) <= |val-div_low or div_high| <=
    // |val - other's boundary| = TRUE_MIN(other)`. So whichever child the
    // heuristic labels "best" PROVABLY has a true minimum <= the other
    // child's true minimum, at EVERY node, all the way to a leaf — meaning
    // pure greedy best-child descent (no "other" branch ever needed) already
    // finds the exact global 1-NN in 1D, for ANY eps (the "other" branch
    // visit is exact-search-only insurance for ties, never a source of
    // improvement). Confirmed empirically: 20,000 random dim-1 trials
    // (leaf_max_size in {1,2}, n up to 48, uniform query range) never
    // produced a single case where eps=0.5's k=1 answer differed from
    // eps=0's.
    //
    // The mechanism DOES exist in dim>=2, because `mindist` there is a
    // SINGLE-AXIS bound (the split axis's `accum_dist` only) while
    // `worst_dist` is the FULL multi-axis distance — so `mindist` is a
    // valid but genuinely non-tight lower bound, and a point in the "other"
    // branch can have a small full-distance despite the pruned axis alone
    // suggesting otherwise. This test hand-derives such a dim-2 case (bypasses
    // the builder for exact control over `div_low`/`div_high`, same
    // technique as `radius_sorted_false_gives_tree_traversal_order`).
    //
    // Construction (all values exact in f64, verified by hand below and by
    // running the resulting assertions): points P0=(0.35, 0.3) ["found"/best
    // child, visited unconditionally first], P1=(-0.3, 0.05) ["other"/pruned
    // child — the TRUE nearest]. Split axis 0, div_low=-0.3 (=P1's own x,
    // tight), div_high=0.35 (=P0's own x, tight). Query=(0.06, 0.0).
    //
    // Heuristic: diff1=val-div_low=0.06-(-0.3)=0.36; diff2=val-div_high=
    // 0.06-0.35=-0.29; sum=0.07 >= 0 -> best=P0 (child2), matching the
    // labeling above.
    //
    // Exact (eps=0): visit P0 first, worst_dist=(0.06-0.35)^2+(0-0.3)^2=
    // 0.0841+0.09=0.1741. cut_dist=accum_dist(0.06,-0.3,axis0)=(0.36)^2=
    // 0.1296=mindist. eps=0 -> 0.1296*1<=0.1741 -> visit P1: true dist=
    // (0.06+0.3)^2+(0-0.05)^2=0.1296+0.0025=0.1321 < 0.1741 -> UPDATES,
    // exact result = P1, dist 0.1321 (the true global nearest).
    //
    // eps=0.5 (eps_error=1.5): after P0, worst_dist=0.1741, mindist=0.1296.
    // 0.1296*1.5=0.1944 > 0.1741 -> PRUNED, P1 never visited. Approx result
    // = P0, dist 0.1741.
    //
    // Bound check: 0.1741 <= 1.5 * 0.1321 = 0.19815 -- holds, with margin.
    #[test]
    fn eps_moderate_bounds_approximate_error_dim2_hand_derived() {
        let pts: Vec<[f64; 2]> = vec![[0.35, 0.3], [-0.3, 0.05]]; // index0=P0, index1=P1
        let pts: &[[f64; 2]] = &pts;

        // vind = [1, 0]: leaf-left (arena idx 1) = vind[0..1] = [1] (P1);
        // leaf-right (arena idx 2) = vind[1..2] = [0] (P0).
        let vind: Vec<u32> = vec![1, 0];
        // Pushed one at a time (not a `vec![]` literal) so each node's
        // arena-index role can carry its own inline comment.
        #[allow(clippy::vec_init_then_push)]
        let mut arena: Vec<Node<f64>> = {
            let mut arena = Vec::new();
            arena.push(Node::split(0, -0.3, 0.35)); // root, will patch children below
            arena.push(Node::leaf(0, 1)); // left leaf: P1
            arena.push(Node::leaf(1, 2)); // right leaf: P0
            arena
        };
        {
            // Node has no public setter accessible from outside build.rs's
            // module boundary other than what's already pub(crate); reuse
            // the same crate-private API the builder itself uses.
            let root = &mut arena[0];
            root.set_children(1, 2);
        }

        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &[],
        };
        let query = [0.06f64, 0.0];

        // Exact (eps_error = 1.0): bypass find_neighbors/compute_initial_distances
        // entirely (its bbox-based initial mindist isn't part of this
        // hand-derivation, which assumes mindist starts at 0 exactly as if
        // query were inside the root bbox on every axis) — call
        // `search_level` directly with mindist=0, dists=[0,0].
        let mut exact_indices = [0u32; 1];
        let mut exact_dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut exact_indices, &mut exact_dists);
            let mut dists = [0.0f64, 0.0];
            search_level_explicit(
                &ctx, &mut rs, &query, 0, 0.0f64, &mut dists, 1.0f64, &AcceptAll,
            );
        }
        assert_eq!(
            exact_indices,
            [1],
            "exact search should find P1 (the true nearest)"
        );
        assert!(
            (exact_dists[0] - 0.1321).abs() < 1e-9,
            "exact_dists[0]={}",
            exact_dists[0]
        );

        // Approximate (eps=0.5 -> eps_error = 1.5).
        let mut approx_indices = [0u32; 1];
        let mut approx_dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut approx_indices, &mut approx_dists);
            let mut dists = [0.0f64, 0.0];
            search_level_explicit(
                &ctx, &mut rs, &query, 0, 0.0f64, &mut dists, 1.5f64, &AcceptAll,
            );
        }
        assert_eq!(
            approx_indices,
            [0],
            "eps=0.5 should have pruned P1 and kept the (suboptimal) P0"
        );
        assert!(
            (approx_dists[0] - 0.1741).abs() < 1e-9,
            "approx_dists[0]={}",
            approx_dists[0]
        );

        // The eps-error contract: returned distance <= (1+eps) * true nearest.
        let true_best = exact_dists[0];
        let got = approx_dists[0];
        assert!(
            got > true_best,
            "eps=0.5 result should be strictly worse than exact here (that's the point)"
        );
        assert!(
            got <= 1.5 * true_best + 1e-9,
            "approx dist {got} exceeds (1+eps)*true_best {}",
            1.5 * true_best
        );
    }

    // ---------------------------------------------------------------
    // Fix round 1, finding 1: eps_error must widen-first
    // ---------------------------------------------------------------

    /// A `ResultSet` whose `worst_dist()` is a fixed, caller-chosen constant
    /// — completely decoupled from any point actually found. Used to pin
    /// down `search_level`'s prune-gate arithmetic (`mindist * eps_error <=
    /// worst_dist`) to an exact target ratio without fighting floating-point
    /// rounding in a geometric construction.
    struct FixedWorstDist {
        worst: f64,
    }
    impl ResultSet<f64, u32> for FixedWorstDist {
        fn worst_dist(&self) -> f64 {
            self.worst
        }
        fn add_point(&mut self, _dist: f64, _index: u32) -> bool {
            true
        }
        fn full(&self) -> bool {
            false
        }
        fn size(&self) -> usize {
            0
        }
    }

    /// Counts `is_active` calls for one specific index only (the "other"
    /// branch's sole point), so visiting-or-not is directly observable.
    struct TargetOnlyCounter {
        target: u32,
        count: core::cell::Cell<usize>,
    }
    impl PointFilter<u32> for TargetOnlyCounter {
        fn is_active(&self, idx: u32) -> bool {
            if idx == self.target {
                self.count.set(self.count.get() + 1);
            }
            true
        }
    }

    /// `find_neighbors` computes `eps_error` as `DistanceValue::from_f32(1.0)
    /// + DistanceValue::from_f32(params.eps)` — widening `1.0f32` and
    /// `eps: f32` to `f64` FIRST, then adding in `f64` — mirroring
    /// nanoflann.hpp:1999's `1 + static_cast<DistanceType>(searchParams.eps)`
    /// exactly. An earlier version of this code instead computed
    /// `1.0f32 + params.eps` in `f32` (rounding to f32 precision) and widened
    /// the SUM, which is NOT bit-identical. At `eps = 0.1f32` the two forms
    /// diverge in the low bits:
    ///   widen-first (correct):  from_f32(1.0) + from_f32(0.1) = 1.1000000014901161
    ///   add-then-widen (old bug): from_f32(1.0f32 + 0.1f32)   = 1.100000023841858
    /// This test builds a hand-crafted node (bypassing the builder, same
    /// technique as `eps_moderate_bounds_approximate_error_dim2_hand_derived`)
    /// where the prune gate `mindist * eps_error <= worst_dist` sits exactly
    /// between the two constants (mindist=1.0 exactly, worst_dist mocked to
    /// the exact midpoint of the two eps_error values via `FixedWorstDist`),
    ///   so the CORRECT constant visits the other branch and the OLD (buggy)
    ///   constant prunes it — a genuinely discriminating test.
    #[test]
    fn eps_error_widen_first_parity() {
        let eps = 0.1f32;
        let eps_correct = f64::from(1.0f32) + f64::from(eps); // widen-first: matches nanoflann.hpp:1999
        let eps_old_buggy = f64::from(1.0f32 + eps); // add-in-f32-then-widen: the bug this fixes
        assert!(
            eps_correct < eps_old_buggy,
            "sanity: the two constants must actually differ"
        );
        assert!(
            (eps_correct - 1.1000000014901161).abs() < 1e-16,
            "eps_correct={eps_correct}"
        );
        assert!(
            (eps_old_buggy - 1.100000023841858).abs() < 1e-16,
            "eps_old_buggy={eps_old_buggy}"
        );

        // mindist = 1.0 exactly. worst_dist is mocked to the exact midpoint
        // of the two constants, so the gate outcome depends ENTIRELY on
        // which eps_error value is used.
        let worst_dist_target = (eps_correct + eps_old_buggy) / 2.0;

        // index0 = dummy point at 0.5 (RIGHT leaf, "best" — always visited
        // unconditionally; its actual distance is irrelevant since
        // worst_dist is mocked). index1 = target point at -1.0 (LEFT leaf,
        // "other" — the one whose visitation this test observes).
        let pts: Vec<[f64; 1]> = vec![[0.5], [-1.0]];
        let pts: &[[f64; 1]] = &pts;
        let vind: Vec<u32> = vec![1, 0]; // left leaf = vind[0..1] = [1]; right leaf = vind[1..2] = [0]
        let mut arena: Vec<Node<f64>> = vec![
            Node::split(0, -1.0, 0.5), // root: div_low = index1's own x, div_high = index0's own x
            Node::leaf(0, 1),          // left leaf: index1 (-1.0)
            Node::leaf(1, 2),          // right leaf: index0 (0.5)
        ];
        arena[0].set_children(1, 2);

        // Heuristic check: diff1=val-div_low=0-(-1.0)=1.0; diff2=val-div_high
        // =0-0.5=-0.5; sum=0.5 >= 0 -> best=child2=RIGHT(index0),
        // other=child1=LEFT(index1). cut_dist (for LEFT)=accum_dist(0,-1.0)=
        // (0-(-1.0))^2=1.0=mindist, exactly as designed.
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &[],
        };
        let query = [0.0f64];

        let run_with_eps_error = |eps_error: f64| -> usize {
            let mut rs = FixedWorstDist {
                worst: worst_dist_target,
            };
            let filter = TargetOnlyCounter {
                target: 1,
                count: core::cell::Cell::new(0),
            };
            let mut dists = [0.0f64];
            search_level_explicit(
                &ctx, &mut rs, &query, 0, 0.0f64, &mut dists, eps_error, &filter,
            );
            filter.count.get()
        };

        assert_eq!(
            run_with_eps_error(eps_correct),
            1,
            "the CORRECT (widen-first) eps_error should visit the other leaf"
        );
        assert_eq!(
            run_with_eps_error(eps_old_buggy),
            0,
            "the OLD (add-then-widen, buggy) eps_error would have pruned it"
        );

        // Confirm the ACTUAL production code path (`find_neighbors`, which
        // computes `eps_error` internally from `params.eps`) matches the
        // corrected widen-first behavior — same mocked ResultSet, plumbed
        // through the real driver this time (needs a real root_bbox so
        // `compute_initial_distances` doesn't contribute; query=0.0 is
        // inside [-1.0, 0.5] on every axis, so it contributes exactly 0).
        let ctx_with_bbox = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(1),
            nodes: &arena,
            vind: &vind,
            root_bbox: &[Interval {
                low: -1.0,
                high: 0.5,
            }],
        };
        let mut rs = FixedWorstDist {
            worst: worst_dist_target,
        };
        let filter = TargetOnlyCounter {
            target: 1,
            count: core::cell::Cell::new(0),
        };
        let mut scratch = [0.0f64];
        let params = SearchParams { eps, sorted: true };
        find_neighbors(
            &ctx_with_bbox,
            &mut rs,
            &query,
            &params,
            &filter,
            &mut scratch,
        );
        assert_eq!(
            filter.count.get(),
            1,
            "find_neighbors's internally-computed eps_error must match the corrected (widen-first) constant"
        );
    }

    // ---------------------------------------------------------------
    // Fix round 1, finding 4: deep-tree QUERY coverage (release-only, #[ignore])
    // ---------------------------------------------------------------

    /// Iterative max-depth (root = depth 1) — local copy of build.rs's test
    /// helper of the same name (that one is private to build.rs's own test
    /// module, so it isn't reachable from here).
    fn max_depth(arena: &[Node<f64>], root: u32) -> usize {
        let mut stack = vec![(root, 1usize)];
        let mut max_d = 0usize;
        while let Some((idx, d)) = stack.pop() {
            if d > max_d {
                max_d = d;
            }
            let node = &arena[idx as usize];
            if !node.is_leaf() {
                let (c1, c2) = node.children();
                stack.push((c1, d + 1));
                stack.push((c2, d + 1));
            }
        }
        max_d
    }

    /// Same exponential-spine construction as build.rs's
    /// `heavy_exponential_build_1m` test (duplicated here — that generator
    /// is private to build.rs's own test module). Measured depth there:
    /// ~2115.
    fn heavy_dim1_points(n: usize) -> Vec<[f64; 1]> {
        let mut spine: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(1023);
        while v > 0.0 && spine.len() < n - 1 {
            spine.push(v);
            v /= 2.0;
        }
        let mut values = spine;
        values.resize(n, 0.0);
        values.into_iter().map(|v| [v]).collect()
    }

    /// Same round-robin construction as build.rs's
    /// `heavy_exponential_build_1m_dim8` test. Measured depth there: ~16794.
    fn heavy_dim8_points(n: usize) -> Vec<[f64; 8]> {
        const DIM: usize = 8;
        let mut ladder: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(1023);
        while v > 0.0 {
            ladder.push(v);
            v /= 2.0;
        }
        let spine_len = (ladder.len() * DIM).min(n - 1);
        let mut points: Vec<[f64; DIM]> = Vec::with_capacity(n);
        for k in 0..spine_len {
            let d = k % DIM;
            let m = k / DIM;
            let mut coords = [0.0f64; DIM];
            coords[d] = ladder[m];
            points.push(coords);
        }
        points.resize(n, [0.0f64; DIM]);
        points
    }

    /// UPDATED by M2.5 task 2: `search_level` used to be native recursion,
    /// so a sufficiently deep tree could in principle overflow a worker
    /// thread's stack during a QUERY (not just during the build task 6
    /// already stress-tested) — hence this test originally ran on a
    /// deliberately small, rayon-worker-sized 2 MiB thread stack, to prove
    /// the deepest degenerate trees task 6's builder can produce stayed
    /// within that budget. Task 2 converted `search_level` to the explicit
    /// `FrameStack` iteration (see its doc comment) — depth is now bounded
    /// by `FrameStack`'s heap spill, not by ANY native call stack, so a
    /// native worker-thread stack overflow during a query is no longer
    /// possible in Rust regardless of tree depth (C++'s `searchLevel` is
    /// still native recursion and keeps the old exposure, unchanged by this
    /// port). The 2 MiB worker thread is kept anyway — it is still a
    /// faithful "no regression vs. a real rayon worker" harness, and
    /// running here specifically exercises `FrameStack`'s SPILL path at
    /// real scale (frame 129 onward, all the way to depth ~16 794, vs.
    /// `spill_boundary_deep_tree_knn_and_radius_match_brute_force` above,
    /// which checks the spill transition itself far more cheaply). This
    /// reuses task 6's two deepest known constructions
    /// (`build.rs::heavy_exponential_build_1m`, depth ~2115, and
    /// `..._dim8`, depth ~16794) and runs real knn + radius searches
    /// against them.
    #[test]
    #[ignore]
    fn heavy_query_degenerate_trees() {
        const STACK_SIZE: usize = 2 * 1024 * 1024;
        eprintln!(
            "heavy_query_degenerate_trees: using {STACK_SIZE} byte ({} MiB) worker stack",
            STACK_SIZE / (1024 * 1024)
        );

        let n = 1_000_000usize;

        let pts1 = heavy_dim1_points(n);
        let (vind1, arena1, bbox1) = build_tree(&pts1, 10);
        let depth1 = max_depth(&arena1, 0);
        eprintln!("dim-1 heavy tree depth = {depth1}");
        assert!(
            depth1 > 2_000,
            "expected the dim-1 heavy tree to stay deep, got {depth1}"
        );

        let pts8 = heavy_dim8_points(n);
        let (vind8, arena8, bbox8) = build_tree(&pts8, 10);
        let depth8 = max_depth(&arena8, 0);
        eprintln!("dim-8 heavy tree depth = {depth8}");
        assert!(
            depth8 > 10_000,
            "expected the dim-8 heavy tree to stay deep, got {depth8}"
        );

        let handle = std::thread::Builder::new()
            .name("heavy-query-worker".into())
            .stack_size(STACK_SIZE)
            .spawn(move || {
                // Query selection note: an EARLIER version of this test used
                // query=[0.0] (and dim-8's all-zero analog) and hit a real
                // methodology bug, not a search bug — this heavy dataset has
                // ~1M-2115 (dim-1) / ~1M-16794 (dim-8) points at EXACTLY
                // 0.0 (both the explicit padding AND the deep spine tail,
                // which underflows to exactly 0.0 once squared, since IEEE
                // f64 has a finite denormal floor). A query at/near the
                // origin faces a many-thousand-way EXACT tie for "nearest",
                // and `brute_force_knn`'s linear index-order scan resolves
                // ties by SMALLEST INDEX while the tree's resolution follows
                // TRAVERSAL order (leaf-visit order, not index order) — both
                // are valid readings of `KeepInsertionOrder`'s contract (see
                // result_set.rs), but they don't have to agree with each
                // other on WHICH tied points win when there are thousands of
                // them, so comparing exact index lists against such a query
                // is not a meaningful test. Instead: queries here are exact
                // copies of specific, well-separated (non-tail, non-padding)
                // data points, so their nearest neighbors are unambiguous
                // (adjacent geometric-ladder rungs, strictly ordered, no
                // ties) — this still exercises the SAME deep recursive
                // traversal paths (the tree still has to walk all the way
                // down to a leaf and prune back up through ~2000-17000
                // levels to answer these), which is what this test is
                // actually checking (stack safety + correctness under deep
                // recursion), without the tie-methodology trap.
                // Indices chosen comfortably mid-spine (magnitude ~2^423,
                // ~2^323, ~2^23): NOT from the top of the spine (index 0's
                // magnitude ~2^1023 — squaring a difference between two
                // such huge values overflows f64's ~2^1024 range to
                // `+inf`, which then fails EVERY `dist < worst_dist` gate
                // and silently drops genuinely-nearer points; caught this
                // empirically — an earlier version of this test used
                // `pts1[0]`/`pts1[500]` and got a real but spurious
                // `tree_idx.len()==1` "mismatch" purely from squared-L2
                // overflow, not a search bug).
                let queries1: Vec<[f64; 1]> = vec![pts1[600], pts1[700], pts1[1000]];
                for q in &queries1 {
                    let (tree_idx, tree_dists, _) =
                        tree_knn(&vind1, &arena1, &bbox1, &pts1, q, 10, 0.0);
                    let (bf_idx, bf_dists) = brute_force_knn(&pts1, q, 10);
                    assert_eq!(tree_idx, bf_idx, "dim-1 knn index mismatch for query {q:?}");
                    assert_eq!(
                        tree_dists, bf_dists,
                        "dim-1 knn dist mismatch for query {q:?}"
                    );

                    // Radius strictly between the 1st and 2nd nearest
                    // distances -> exactly one point within radius (the
                    // query's own exact match, distance 0), no tie/boundary
                    // ambiguity possible.
                    let radius = (bf_dists[0] + bf_dists[1]) / 2.0;
                    let mut tree_r = tree_radius(&vind1, &arena1, &bbox1, &pts1, q, radius);
                    tree_r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
                    let bf_r = brute_force_radius(&pts1, q, radius);
                    assert_eq!(tree_r, bf_r, "dim-1 radius mismatch for query {q:?}");
                    assert_eq!(
                        tree_r.len(),
                        1,
                        "radius was chosen to admit exactly the query's own match"
                    );
                }

                // Same overflow-avoidance as dim-1: indices k=8*m+axis with
                // m in {600, 700, 5000/8=625} — all comfortably past the
                // point where squared differences between adjacent ladder
                // rungs would overflow f64.
                let queries8: Vec<[f64; 8]> =
                    vec![pts8[8 * 600 + 2], pts8[8 * 700 + 5], pts8[5000]];
                for q in &queries8 {
                    let (tree_idx, tree_dists, _) =
                        tree_knn(&vind8, &arena8, &bbox8, &pts8, q, 10, 0.0);
                    let (bf_idx, bf_dists) = brute_force_knn(&pts8, q, 10);
                    assert_eq!(tree_idx, bf_idx, "dim-8 knn index mismatch for query {q:?}");
                    assert_eq!(
                        tree_dists, bf_dists,
                        "dim-8 knn dist mismatch for query {q:?}"
                    );

                    let radius = (bf_dists[0] + bf_dists[1]) / 2.0;
                    let mut tree_r = tree_radius(&vind8, &arena8, &bbox8, &pts8, q, radius);
                    tree_r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
                    let bf_r = brute_force_radius(&pts8, q, radius);
                    assert_eq!(tree_r, bf_r, "dim-8 radius mismatch for query {q:?}");
                    assert_eq!(
                        tree_r.len(),
                        1,
                        "radius was chosen to admit exactly the query's own match"
                    );
                }
            })
            .expect("failed to spawn heavy-query-worker thread");

        handle.join().expect("heavy-query-worker thread panicked (a stack overflow instead aborts the whole process, with no Result to observe)");
        eprintln!("heavy_query_degenerate_trees: all knn/radius queries matched brute force on both heavy trees");
    }

    // ---------------------------------------------------------------
    // Task 9: find_within_box
    // ---------------------------------------------------------------

    fn brute_force_box<const N: usize>(pts: &[[f64; N]], bounds: &[Interval<f64>]) -> Vec<u32> {
        (0..pts.len())
            .filter(|&i| (0..N).all(|d| bounds[d].contains(pts[i][d])))
            .map(|i| i as u32)
            .collect()
    }

    fn tree_box<const N: usize>(
        vind: &[u32],
        arena: &[Node<f64>],
        bbox: &[Interval<f64>],
        pts: &[[f64; N]],
        bounds: &[Interval<f64>],
    ) -> Vec<u32> {
        let dim = N;
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(dim),
            nodes: arena,
            vind,
            root_bbox: bbox,
        };
        let mut out = Vec::new();
        let count = find_within_box(&ctx, bounds, &mut out);
        assert_eq!(
            count,
            out.len(),
            "find_within_box return value must equal out.len()"
        );
        out
    }

    // Test 1: face inclusion — dim-3 integer grid 0..=3^3 (64 points), box
    // [1,2]x[0,3]x[2,2]. Result must equal the brute-force inclusive filter,
    // and points exactly on every face (low AND high, each axis) must be
    // included.
    #[test]
    fn find_within_box_face_inclusion() {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for x in 0..=3 {
            for y in 0..=3 {
                for z in 0..=3 {
                    pts.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let (vind, arena, bbox) = build_tree(&pts, 4);

        let bounds = vec![
            Interval {
                low: 1.0,
                high: 2.0,
            },
            Interval {
                low: 0.0,
                high: 3.0,
            },
            Interval {
                low: 2.0,
                high: 2.0,
            },
        ];

        let mut got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
        let mut expected = brute_force_box(&pts, &bounds);
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            got, expected,
            "find_within_box must equal the brute-force inclusive filter (as a set)"
        );

        // Explicit face checks: both faces of dim0 (x=1 low, x=2 high) and
        // both faces of dim1 (y=0 low, y=3 high), all at the degenerate
        // dim2 value z=2 (low == high on that axis).
        for corner in [
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [1.0, 3.0, 2.0],
            [2.0, 3.0, 2.0],
        ] {
            let idx = pts
                .iter()
                .position(|p| *p == corner)
                .expect("corner point must exist in the grid") as u32;
            assert!(
                got.contains(&idx),
                "face point {corner:?} (index {idx}) must be included (inclusive boundaries)"
            );
        }
    }

    // Test 2: exact traversal order lock.
    //
    // Dataset: pts = [[1.0], [2.0], ..., [10.0]] (dim 1, 10 points),
    // leaf_max_size = 2. Hand-derived tree (by simulating `middle_split` /
    // `plane_split` from build.rs exactly, since arena index assignment is
    // preorder DFS, left child fully built before right per the Finalize
    // push-order comment in `SubtreeBuilder::build`):
    //
    //   Node0 (root):   interior, cutfeat=0, divlow=5,  divhigh=6,  children=(1,6)
    //   Node1:          interior, cutfeat=0, divlow=3,  divhigh=4,  children=(2,5)
    //   Node2:          interior, cutfeat=0, divlow=2,  divhigh=3,  children=(3,4)
    //   Node3:          leaf, vind[0..2)  = point-indices [0,1]   (values 1,2)
    //   Node4:          leaf, vind[2..3)  = point-indices [2]     (value 3)
    //   Node5:          leaf, vind[3..5)  = point-indices [4,3]   (values 5,4)
    //   Node6:          interior, cutfeat=0, divlow=7,  divhigh=8,  children=(7,8)
    //   Node7:          leaf, vind[5..7)  = point-indices [6,5]   (values 7,6)
    //   Node8:          interior, cutfeat=0, divlow=8,  divhigh=9,  children=(9,10)
    //   Node9:          leaf, vind[7..8)  = point-indices [7]     (value 8)
    //   Node10:         leaf, vind[8..10) = point-indices [8,9]   (values 9,10)
    //
    // (Node4/Node5's split at Node1 happens to leave the right leaf's vind
    // order as [4,3] rather than [3,4] because `plane_split`'s in-place
    // Dutch-flag partition swaps elements > cutval down from the right end;
    // similarly Node7's leaf ends up [6,5].)
    //
    // Query box = [2.0, 9.0] (dim 1). Hand-traced stack walk (push child1
    // then child2 -> child2 pops FIRST):
    //   pop 0(root): low2<=5 push 1; high9>=6 push 6.        stack=[1,6]
    //   pop 6:       low2<=7 push 7; high9>=8 push 8.        stack=[1,7,8]
    //   pop 8:       low2<=8 push 9; high9>=9 push 10.       stack=[1,7,9,10]
    //   pop 10: leaf {8,9} vals{9,10} -> 9 in [2,9]? yes(idx8); 10? no.  out=[8]
    //   pop 9:  leaf {7} val{8} -> in [2,9]? yes.                        out=[8,7]
    //   pop 7:  leaf {6,5} vals{7,6} (iterated in that vind order)
    //           -> both in [2,9]: idx6 then idx5.                       out=[8,7,6,5]
    //   pop 1:       low2<=3 push 2; high9>=4 push 5.        stack=[2,5]
    //   pop 5:  leaf {4,3} vals{5,4} -> both in range: idx4 then idx3.   out=[8,7,6,5,4,3]
    //   pop 2:       low2<=2 push 3; high9>=3 push 4.        stack=[3,4]
    //   pop 4:  leaf {2} val{3} -> in range: idx2.                      out=[8,7,6,5,4,3,2]
    //   pop 3:  leaf {0,1} vals{1,2} -> val1 NOT in [2,9] (excluded);
    //           val2 in range: idx1.                                    out=[8,7,6,5,4,3,2,1]
    //
    // Final expected sequence: [8,7,6,5,4,3,2,1] (locks child2-pops-first).
    #[test]
    fn find_within_box_exact_traversal_order() {
        let pts: Vec<[f64; 1]> = (1..=10).map(|v| [v as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 2);

        let bounds = vec![Interval {
            low: 2.0,
            high: 9.0,
        }];
        let got = tree_box(&vind, &arena, &bbox, &pts, &bounds);

        assert_eq!(
            got,
            vec![8u32, 7, 6, 5, 4, 3, 2, 1],
            "traversal order must match the hand-derived stack walk"
        );
    }

    // Test 3: prune boundaries — inclusive descent (`<=`/`>=`), not the
    // exclusive `<`/`>` a port bug would use. Reuses the tree from Test 2.
    #[test]
    fn find_within_box_prune_boundary_low_equals_div_low() {
        let pts: Vec<[f64; 1]> = (1..=10).map(|v| [v as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 2);

        // box.low == Node2's divlow (2.0) exactly. The only matching point
        // (value 2, index 1) lives in Node2's child1 (Node3); a `<` port
        // bug would fail `2.0 < 2.0` and prune Node3, losing it.
        let bounds = vec![Interval {
            low: 2.0,
            high: 2.0,
        }];
        let got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
        assert_eq!(got, vec![1u32]);
    }

    #[test]
    fn find_within_box_prune_boundary_high_equals_div_high() {
        let pts: Vec<[f64; 1]> = (1..=10).map(|v| [v as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 2);

        // box.high == the root's divhigh (6.0) exactly. The only matching
        // point (value 6, index 5) lives under the root's child2 (Node6); a
        // `>` port bug would fail `6.0 > 6.0` and prune Node6, losing it.
        let bounds = vec![Interval {
            low: 6.0,
            high: 6.0,
        }];
        let got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
        assert_eq!(got, vec![5u32]);
    }

    // Test 4: brute-force property — 100 seeded LCG cases, n <= 300, dim 2/3,
    // random boxes (including degenerate low==high boxes per axis).
    fn run_box_case<const N: usize>(seed: u64, n: usize, leaf_max_size: usize) {
        let mut rng = Lcg(seed);
        let pts: Vec<[f64; N]> = (0..n)
            .map(|_| {
                let mut p = [0.0f64; N];
                for v in p.iter_mut() {
                    *v = rng.next_f64() * 200.0 - 100.0;
                }
                p
            })
            .collect();

        let mut bounds = vec![
            Interval {
                low: 0.0,
                high: 0.0
            };
            N
        ];
        for b in bounds.iter_mut() {
            let a = rng.next_f64() * 200.0 - 100.0;
            let b2 = rng.next_f64() * 200.0 - 100.0;
            let (lo, hi) = if a <= b2 { (a, b2) } else { (b2, a) };
            // ~1-in-5 chance of collapsing this axis to a degenerate point.
            *b = if rng.next_f64() < 0.2 {
                Interval { low: lo, high: lo }
            } else {
                Interval { low: lo, high: hi }
            };
        }

        let (vind, arena, bbox) = build_tree(&pts, leaf_max_size);
        let mut got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
        let mut expected = brute_force_box(&pts, &bounds);
        assert_eq!(
            got.len(),
            expected.len(),
            "seed={seed} n={n} dim={N} bounds={bounds:?}: count mismatch"
        );
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            got, expected,
            "seed={seed} n={n} dim={N} bounds={bounds:?}: set mismatch"
        );
    }

    #[test]
    fn find_within_box_brute_force_property() {
        for seed in 0..100u64 {
            let n = 1 + (seed as usize * 37) % 300;
            let leaf_max = 1 + (seed as usize % 8);
            if seed % 2 == 0 {
                run_box_case::<2>(seed.wrapping_mul(7919).wrapping_add(1), n, leaf_max);
            } else {
                run_box_case::<3>(seed.wrapping_mul(7919).wrapping_add(1), n, leaf_max);
            }
        }
    }

    // Test 5: never sorted — search seeded random cases (wide box, so the
    // FULL point set is returned in natural traversal order) for one whose
    // output is not ascending-sorted, and lock that. Guards against an
    // accidental "helpful" sort creeping into `find_within_box` (unlike
    // knn/radius search, box search takes no `SearchParams` and never sorts).
    #[test]
    fn find_within_box_never_sorted() {
        let mut found = false;
        for seed in 0..200u64 {
            let n = 1 + (seed as usize * 53) % 300;
            let leaf_max = 1 + (seed as usize % 6);
            let mut rng = Lcg(seed.wrapping_mul(104729).wrapping_add(3));
            let pts: Vec<[f64; 3]> = (0..n)
                .map(|_| {
                    let mut p = [0.0f64; 3];
                    for v in &mut p {
                        *v = rng.next_f64() * 200.0 - 100.0;
                    }
                    p
                })
                .collect();
            let bounds = vec![
                Interval {
                    low: -100.0,
                    high: 100.0
                };
                3
            ];
            let (vind, arena, bbox) = build_tree(&pts, leaf_max);
            let got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
            if got.len() >= 3 && got.windows(2).any(|w| w[0] > w[1]) {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "expected at least one seeded case with non-ascending find_within_box output"
        );
    }

    // Test 6: empty tree -> 0, `out` cleared (pre-populated to prove it).
    #[test]
    fn find_within_box_empty_tree() {
        let arena: Vec<Node<f64>> = Vec::new();
        let vind: Vec<u32> = Vec::new();
        let bbox: Vec<Interval<f64>> = Vec::new();
        let pts: Vec<[f64; 2]> = Vec::new();
        let pts: &[[f64; 2]] = &pts;
        let metric = L2;
        let ctx = SearchCtx {
            ds: &pts,
            metric: &metric,
            dim: DynDim(2),
            nodes: &arena,
            vind: &vind,
            root_bbox: &bbox,
        };

        let bounds = vec![
            Interval {
                low: -1.0,
                high: 1.0
            };
            2
        ];
        let mut out = vec![42u32, 43, 44];
        let count = find_within_box(&ctx, &bounds, &mut out);

        assert_eq!(count, 0);
        assert!(out.is_empty());
    }

    // Test 7: no-match box -> 0.
    #[test]
    fn find_within_box_no_match() {
        let pts: Vec<[f64; 1]> = (1..=10).map(|v| [v as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 2);

        let bounds = vec![Interval {
            low: 1000.0,
            high: 2000.0,
        }];
        let got = tree_box(&vind, &arena, &bbox, &pts, &bounds);
        assert!(got.is_empty());
    }

    // ---------------------------------------------------------------
    // Spill-boundary test: explicit-stack `search_level` frame storage is a
    // fixed inline array (capacity `SEARCH_STACK_INLINE_CAPACITY`, see the
    // top of this file) with a heap-spill `Vec` fallback for deeper trees.
    // This builds a tree whose depth crosses that inline capacity and
    // checks knn + radius against brute force — i.e. it exercises the
    // spill path itself, not just "does it not panic".
    //
    // Reviewer fix-round finding (do NOT regress this): checking the
    // TREE's overall `max_depth` (via the same helper
    // `heavy_query_degenerate_trees` uses) is necessary but NOT sufficient
    // — a specific query's push depth through that tree can be far
    // shallower than the tree's deepest leaf. The first version of this
    // test asserted `max_depth > SEARCH_STACK_INLINE_CAPACITY` and queried
    // `pts[60]`; the tree's `max_depth` was correctly 140, but that
    // specific query's `FrameStack` only ever reached depth 69 — the spill
    // path (`overflow: Vec`, the `pop`ping/reading of it) was NEVER
    // EXECUTED by this test, despite every assertion passing. Caught only
    // because `FrameStack::max_depth_seen` (test-only instrumentation, see
    // that field's doc) makes the actual push depth directly assertable
    // instead of inferred from the tree's shape. This test now asserts
    // `max_depth_seen` directly, for BOTH the knn and the radius call
    // (`pts[135]` measured: knn depth 139, radius depth 135 — both cross
    // 128 with real margin, not by a hair).
    //
    // `n=150` of the same exponential-spine generator as
    // `heavy_query_degenerate_trees` (just a far smaller `n`) empirically
    // gives tree depth 140 (see the depth/n relationship documented on
    // `heavy_dim1_points`) while staying cheap enough to run in every
    // `cargo test` invocation, unlike the `--ignored` depth ~16 794 heavy
    // test below.
    // ---------------------------------------------------------------

    /// Same exponential-spine shape as `heavy_dim1_points` (each value half
    /// the previous, forcing `middle_split` to peel ~1 point per level —
    /// see that function's doc and `build.rs::heavy_exponential_build_1m`'s
    /// comment for why this specific pattern degenerates), but anchored at
    /// `2^300` instead of `2^1023`. At `2^1023` every point in a short
    /// (n~150) spine is still astronomically large (down to only ~`2^874`),
    /// so squaring any two spine values' difference overflows `f64` to
    /// `+inf` — which breaks `dist < worst_dist()`'s strict inequality (an
    /// `+inf` challenger never beats an `+inf`-initialized "worst", so nothing
    /// past the first exact match gets added) and makes brute-force/tree
    /// comparison meaningless, exactly the trap
    /// `heavy_query_degenerate_trees` documents and avoids by querying deep
    /// into a MUCH longer (n=1e6) spine instead. Anchoring at `2^300`
    /// instead keeps every value in a short spine's squared difference
    /// (worst case ~`2^600`) comfortably under `f64::MAX` (~`2^1024`) while
    /// preserving the identical relative (ratio-based) construction that
    /// makes `middle_split` degenerate — i.e. depth still scales as `n-10`,
    /// empirically confirmed at this anchor.
    fn spill_boundary_points(n: usize) -> Vec<[f64; 1]> {
        let mut spine: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(300);
        while v > 0.0 && spine.len() < n - 1 {
            spine.push(v);
            v /= 2.0;
        }
        let mut values = spine;
        values.resize(n, 0.0);
        values.into_iter().map(|v| [v]).collect()
    }

    #[test]
    fn spill_boundary_deep_tree_knn_and_radius_match_brute_force() {
        // The hybrid search handles the first HYBRID_RECURSION_DEPTH_LIMIT
        // levels via native recursion; only depth beyond that reaches the
        // explicit FrameStack. For the spill path to fire, the FrameStack
        // portion alone must exceed SEARCH_STACK_INLINE_CAPACITY, i.e. tree
        // depth must exceed both thresholds combined.
        const MIN_TREE_DEPTH: usize =
            HYBRID_RECURSION_DEPTH_LIMIT as usize + SEARCH_STACK_INLINE_CAPACITY + 1;
        // n=250 gives tree depth ~240 (spine peels ~1 point per level with
        // leaf_max_size=10), explicit-stack portion ~144, crossing 128.
        const _: () = assert!(240 > MIN_TREE_DEPTH as usize);

        let n = 250usize;
        let pts = spill_boundary_points(n);
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let depth = max_depth(&arena, 0);
        assert!(
            depth > MIN_TREE_DEPTH,
            "expected tree depth ({depth}) to exceed hybrid cutoff + inline capacity \
             ({MIN_TREE_DEPTH}) so this test actually exercises the spill path"
        );

        // Query deep into the spine so the explicit-stack portion crosses
        // the inline capacity. pts[235] reaches depth ~235, of which ~139
        // land on the explicit stack (235 - 96), crossing the 128 boundary.
        let query = pts[235];

        LAST_SEARCH_STACK_MAX_DEPTH.with(|c| c.set(0));
        let (tree_idx, tree_dists, full) = tree_knn(&vind, &arena, &bbox, &pts, &query, 10, 0.0);
        let knn_max_depth = LAST_SEARCH_STACK_MAX_DEPTH.with(|c| c.get());
        assert!(
            knn_max_depth > SEARCH_STACK_INLINE_CAPACITY,
            "knn query never actually spilled: FrameStack max depth reached was {knn_max_depth}, \
             capacity is {SEARCH_STACK_INLINE_CAPACITY} — this test would silently stop \
             discriminating the spill path if the query index above stopped reaching this deep"
        );
        let (bf_idx, bf_dists) = brute_force_knn(&pts, &query, 10);
        assert_eq!(
            tree_idx, bf_idx,
            "knn index mismatch across the spill boundary"
        );
        assert_eq!(
            tree_dists, bf_dists,
            "knn dist mismatch across the spill boundary"
        );
        assert!(full);

        // Radius strictly between the 1st and 2nd nearest distances -> the
        // query's own exact match only, no tie ambiguity.
        let radius = (bf_dists[0] + bf_dists[1]) / 2.0;
        LAST_SEARCH_STACK_MAX_DEPTH.with(|c| c.set(0));
        let mut tree_r = tree_radius(&vind, &arena, &bbox, &pts, &query, radius);
        let radius_max_depth = LAST_SEARCH_STACK_MAX_DEPTH.with(|c| c.get());
        assert!(
            radius_max_depth > SEARCH_STACK_INLINE_CAPACITY,
            "radius query never actually spilled: FrameStack max depth reached was {radius_max_depth}, \
             capacity is {SEARCH_STACK_INLINE_CAPACITY}"
        );
        tree_r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        let bf_r = brute_force_radius(&pts, &query, radius);
        assert_eq!(tree_r, bf_r, "radius mismatch across the spill boundary");
        assert_eq!(
            tree_r.len(),
            1,
            "radius was chosen to admit exactly the query's own match"
        );
    }
}
