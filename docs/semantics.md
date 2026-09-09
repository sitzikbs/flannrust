# flannrust semantics

Exact behavioral contracts of `flannrust`, and every place it deliberately
differs from nanoflann 1.12.1's C++. Extracted from the project README;
how these contracts are *verified* is documented in [testing.md](testing.md).

## Behavioral contracts

These are the exact semantics this crate commits to. Every one mirrors
nanoflann 1.12.1's C++ behavior unless the "Deliberate deviations" table
below says otherwise.

- **Distances and radii are SQUARED** for `L2`/`L2_Simple` (and summed
  absolute value, not squared, for `L1`); `SO2` returns an **unsquared**
  wrapped angle of **only the last dimension** — a single-shot wrap assuming
  inputs are already in `[-π, π]` (it does not correctly re-wrap inputs
  further outside that range).
- **Radius search is strictly `dist < radius`** — a point exactly on the
  boundary is excluded. **Box search is inclusive on all faces** and returns
  results in raw traversal order — it is never sorted and takes no search
  parameters.
- **kNN ties keep traversal order by default** (`KeepInsertionOrder`, C++'s
  default behavior); opt into `SmallestIndexWins` for `NANOFLANN_FIRST_MATCH`
  (fully re-sort ties by ascending index). `SmallestIndexWins`'s rule mirrors
  `NANOFLANN_FIRST_MATCH`'s documented behavior, but it is **not**
  oracle-verified by the cross-validation suite: the vendored C++ oracle
  (`crates/nanoflann-ref`) is compiled with `NANOFLANN_FIRST_MATCH`
  undefined, so there is no C++ build to cross-validate that tie rule
  against.
- **eps**: a node is visited iff `mindist * (1 + eps) <= worst_dist`, where
  `eps` is widened to the distance type **before** the multiply-add (matches
  nanoflann.hpp:1999's `epsError = 1 + static_cast<DistanceType>(eps)` order
  — widen first, not "compute in f32 then widen").
- **Queries snapshot the dataset at `build()` time**: growing the underlying
  `DataSource` after `build()` is invisible to every subsequent query until
  the index is rebuilt (nanoflann's `size_at_index_build_` field is dead code
  upstream — see [`nanoflann-notes.md`](nanoflann-notes.md) — the real snapshot behavior comes
  from `vAcc_`/`size_`, which is what we actually port).
- **`ResultItem` is `#[repr(C)] { index, distance }`**, the layout analog of
  nanoflann's `first`/`second` standard-layout `ResultItem`.

## Deliberate deviations from C++

| Deviation | Rationale |
|---|---|
| An unbuilt index is unrepresentable: `KdTreeBuilder::build()` consumes the builder and returns a ready `KdTree` | Replaces C++'s runtime `std::runtime_error` throw from `findNeighbors` etc. on an unbuilt index with a compile-time impossibility — there is no "forgot to call build()" bug class in this API. |
| Radius results' `sort()` uses Rust's **stable** sort | C++'s `IndexDist_Sorter` uses `std::sort`, which is **unstable** — this is the one place where output order may legally differ between the two implementations, and only among exactly-tied distances. |
| `BuildThreads::Threads(n)` means "at most `n` rayon workers" | Not C++'s exact async per-node thread-gating (`++thread_count < n_thread_build_`); both produce a tree bit-identical to a sequential build (partition precedes spawning on both sides), so this only affects *how* the work is scheduled, never the result. |
| No `NANOFLANN_NODE_ALIGNMENT` (16-byte node alignment) | Evaluated as a bench-gated perf candidate, not applied unconditionally as a default — see [`benchmarks.md`](benchmarks.md). |
| `KdTree::knn_search_with` accepts a `SearchParams` | C++'s static `knnSearch` takes none — this is additive, not a narrowing. |
| `BoxResultSet::sort()` not ported | Dead code upstream: nanoflann's own box-search path never calls it (callers own the output `Vec` and can sort it themselves if desired). |
| `search_level` is an explicit-stack iteration (`FrameStack`: 128 inline frames, heap-`Vec` spill beyond), not C++'s native `searchLevel` recursion (M2.5) | Same traversal order and results, bit-exact. Benefit: the query path cannot overflow the native stack on a degenerate tree (C++'s recursion can). Costs: a query deeper than 128 levels pays per-query heap allocation the recursive form never did, and the conversion gave back margin on `knn_dyn_dim8_f64_k10` and `radius_dim3_f32` (both still < 1.0 vs C++) — see [testing.md](testing.md) and [`benchmarks.md`](benchmarks.md)'s "M2.5 — performance deep-dive". |

## Input domain

Floating-point coordinates must be **finite**. Two edge cases are
out-of-domain on *both* implementations, not just this port:

- **NaN coordinates** give C++-parity-*undefined* containment: the box
  predicate `!(point < low || point > high)` is `true` for NaN (neither
  comparison is ever true under IEEE 754), so NaN is "contained" by every
  box — this matches nanoflann's C++ behavior exactly, but is not a
  meaningful geometric answer.
- **±infinity coordinates make BUILD non-terminating/degenerate on BOTH
  implementations.** An all-`+inf` bounding box has `span = inf - inf = NaN`,
  and NaN fails every `<` comparison used in split-axis/cutval selection, so
  partitioning can never shrink the candidate set. Confirmed during
  cross-validation testing: the C++ oracle **segfaults** (unbounded
  native recursion overflows the stack) and the Rust builder **exhausts
  memory** (observed growing past 18 GB RSS before being killed) on
  identical `+inf`-heavy input. This is unsupported — validate coordinates
  upstream of this crate.

## Feature matrix

- **`parallel`** (enabled by default): pulls in `rayon` and enables
  `BuildThreads::Auto`/`Threads(n)`. With the feature off, those variants
  panic with a message mirroring C++'s `NANOFLANN_NO_THREADS` throw
  ("Multithreading is disabled"); `KdTreeBuilder::build_sequential()` always
  works regardless of this feature and regardless of whether the
  `DataSource` is `Sync` — see below.
- **Index types**: `u32` (default), `u64`, and `usize` are all accepted via
  `KdTreeBuilder::index_type::<Idx>()`. Regardless of which you pick, leaf
  offsets inside the tree's internal node arena are stored as `u32`, so the
  practical dataset-size ceiling is `n <= u32::MAX` no matter the chosen
  index type — `build()`/`build_sequential()` panic if `point_count() >
  u32::MAX as usize`.

### The `Sync` escape hatch: `build_sequential()`

Under the default (`parallel`-on) feature set, `KdTreeBuilder::build()`
requires `DataSource: Sync` — the parallel build path shares `&DS` across
real rayon worker threads, and a shared reference can only cross threads
when the referent is `Sync`. This means a non-`Sync` `DataSource` (for
example, one backed by `Rc<Cell<_>>` interior mutability) cannot call
`build()` at all under default features, **even to request a purely
sequential build**. `KdTreeBuilder::build_sequential()` exists for exactly
this case: it carries no `Sync` bound, always compiles, and always builds —
but only honors `BuildThreads::Sequential` (the default); it panics if
`threads` was set to `Auto`/`Threads(_)`, since those genuinely require
`Sync` to run at all.

## Dynamic adaptor (`DynamicKdTree`)

`DynamicKdTreeBuilder`/`DynamicKdTree` port nanoflann's
`KDTreeSingleIndexDynamicAdaptor` (nanoflann.hpp:2521-2718) plus the
internal sub-tree class it wraps (`KDTreeSingleIndexDynamicAdaptor_`,
nanoflann.hpp:2248-2519): a Bentley–Saxe forest of `tree_count` independent
static kd-tree "slots" (`floor(log2(maximum_point_count)) + 1`, default 30)
supporting point add/remove after construction, without a full rebuild on
every mutation. Every newly-added point walks a binary-counter pattern
(`first0bit`) to pick which slot absorbs it, merging and rebuilding every
lower slot into it; removal is lazy (a tombstone flag, `vind` untouched
until an unrelated merge happens to touch that slot). Full mechanism,
verified against the C++ source line-for-line: `crates/flannrust/src/dynamic.rs`'s
module and `add_points` doc comments; source facts:
[`nanoflann-notes.md`](nanoflann-notes.md)'s "M2 outcome" section.

### API overview

- **`DynamicKdTreeBuilder::new(dim, dataset).build()`** — same defaults as
  the static `KdTreeBuilder` (`L2`, `u32` indices, insertion-order ties,
  `leaf_max_size` 10), plus `maximum_point_count` (default
  1_000_000_000, i.e. 30 slots). If the dataset already reports points at
  build time, they're added immediately (`add_points(0, n-1)`), mirroring
  C++'s constructor. Sequential-only: the forest ctor has no parallel
  slot-rebuild path on either side.
- **`add_points(start, end_inclusive)`** (END-INCLUSIVE, matching C++'s
  `addPoints(start, end)`): adds every dataset index in the range.
  **Contiguity contract (deviation from C++)**: nanoflann indexes its
  internal bookkeeping array by the running point-count COUNTER, not by
  the point index being processed, so every genuinely-new (non-reactivation)
  index must equal `point_count` at the moment it's processed — violating
  this **silently corrupts** the C++ tree's bookkeeping (confirmed by a
  dedicated regression test, `nanoflann-ref`'s
  `add_points_misaligned_start_documents_silent_corruption_f32`). This
  port instead **panics** the instant that misalignment would occur,
  asserted per-index, only against genuinely-new indices — a documented,
  strictly-safer deviation: identical behavior to C++ on every legal call
  sequence, a loud panic instead of silent corruption on an illegal one.
  Reactivating a previously-removed index is *exempt* from the contiguity
  requirement (C++'s reactivation branch never touches the point-count
  counter) — `add_points(idx, idx)` legally reactivates index `idx`
  regardless of the current `point_count`.
- **`remove_point(idx) -> bool`** — lazy removal (C++'s `removePoint`):
  flips a tombstone flag, never touches a slot's point list. Returns
  `false` (no-op) if `idx` was never added or is already removed. The
  point-count counter is never decremented, even by removal — a removed
  index still occupies its original slot budget forever (ported exactly;
  C++'s `removePoint` never references the counter either).
- **Reactivation**: re-adding a removed index via `add_points` restores it
  in place (no duplicate insertion) — the point's slot never actually
  forgot it; only the tombstone flag flips back.
- **Queries — `find_neighbors` (parity) + additive wrappers**: the vendored
  C++ forest exposes only `findNeighbors` (plus `addPoints`/`removePoint`/
  `getAllIndices`) — **no `knnSearch`/`radiusSearch` methods exist on the
  dynamic forest at all** in upstream nanoflann (only the per-slot internal
  sub-tree class has richer methods, never exposed through the forest).
  This port's `find_neighbors` is the direct parity surface (every slot's
  full sub-tree search, tombstone-filtered, against one shared result set).
  `knn_search`/`rknn_search`/`radius_search` (and their `_with` variants
  taking explicit `SearchParams`) are **additive, not upstream parity** —
  convenience wrappers built the same way M1's `KdTree` wraps its own
  `find_neighbors`, so the dynamic forest's ergonomics match the static
  tree's. **No `box_search`**: the C++ forest has no box-search surface to
  wrap (only the static per-slot class does, and it's never exposed through
  the dynamic adaptor) — this port matches that gap exactly rather than
  inventing one.
- **`tree_count()`**, **`active_count()`** (live points = added minus
  currently-removed; additive, no C++ equivalent), **`point_indices_of_slot(slot)`**,
  **`tree_index()`**, **`removed_len()`** — introspection/cross-validation
  accessors, all additive.

### Empty-forest `full()` quirk (inherited from C++, unchanged)

Calling `find_neighbors` directly (the generic parity surface, not the
wrappers) against a forest with zero occupied slots returns whatever an
untouched result set naturally reports on `.full()`: `false` for
knn/rknn, but **`true`** for a fresh `RadiusResultSet` (its `full()` is
hardwired `true` regardless of whether anything was ever added — same
quirk M1 already documented for the static tree, nanoflann.hpp:433). The
additive wrappers return the found COUNT instead, so this is invisible
through the normal API — it only surfaces if you reach for
`find_neighbors` directly on an empty (or all-tombstoned) forest.

## Python API

**Distances (and every radius argument — `query(..., r=...)`,
`query_radius(..., r=...)`) are SQUARED for `l2`/`l2_simple`, exactly like
the Rust core above — NOT euclidean like `scipy.spatial.cKDTree`.** `l1` is
already unsquared (summed absolute value). This is the single most common
mistake porting code from `cKDTree`: square your radius before calling, and
expect squared values back.

| | `KDTree` (static) | `DynamicKDTree` |
|---|---|---|
| construct | `KDTree(points, leaf_size=10, metric="l2", threads=None)` | `DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)` |
| `.query(x, k=1, r=None, eps=0.0, workers=1)` | `-> (dists, idxs)` | same |
| `.query_radius(x, r, sorted=True, eps=0.0, workers=1)` | `-> (idxs, dists)` — two lists of ragged 1-D arrays (note: opposite order from `.query`) | same |
| `.query_box(lo, hi)` | `-> uint32` array, inclusive `[lo, hi]`, traversal order | not exposed — nanoflann's C++ dynamic forest has no box-search surface either, see the "Dynamic adaptor" section above |
| `.add_points(points)` / `.remove_point(idx)` | n/a (static) | append (returns the new half-open index range) / lazily tombstone |
| `.data` | read-only NumPy **copy** of the dataset (not a view — mutating it does not affect the tree) | n/a |
| getters | `.n` `.dim` `.dtype` `.leaf_size` `.metric` | `.n_active` `.n_total` `.removed_len` `.dim` `.dtype` `.leaf_size` `.metric` `.capacity` |

`workers` follows `scipy`'s convention: `1` = sequential, `-1` = all cores,
`n > 1` = capped rayon pool; `threads`/`workers` release the GIL during the
build/query. `metric` is one of `"l2"`, `"l1"`, `"l2_simple"`. Full
docstrings live on each pyclass/pymethod
(`crates/flannrust-py/src/static_tree.rs`, `dynamic_tree.rs`).
