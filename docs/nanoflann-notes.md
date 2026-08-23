# nanoflann 1.12.1 source notes (verified 2026-08-22)

Reference: tag `1.12.1`, commit `7812aa0`, single header `include/nanoflann.hpp` (4469
lines), vendored at `crates/nanoflann-ref/cpp/nanoflann.hpp`. Line numbers below index
that file. These notes exist so Milestones 2+ don't have to re-derive them.

## Class inventory

Adaptors: `KDTreeBaseClass` (1051, CRTP), `KDTreeSingleIndexAdaptor` (1834, static — the
M1 target), `KDTreeSingleIndexDynamicAdaptor_` (2248, internal sub-tree),
`KDTreeSingleIndexDynamicAdaptor` (2521, Bentley–Saxe forest),
`KDTreeSingleIndexIncrementalAdaptor` (2774, scapegoat-style single tree),
`KDTreeSingleIndexIncrementalAdaptorMT` (3916, background-rebuild wrapper, inside
`#ifndef NANOFLANN_NO_THREADS`), `KDTreeEigenMatrixAdaptor` (4354).
Result sets: `KNNResultSet` (291), `RKNNResultSet` (347), `RadiusResultSet` (410),
`BoxResultSet` (472), shared insert `detail::addPointToSortedResultSet` (252),
`ResultItem` (235, `first`=index/`second`=distance, standard-layout, issue #166).
Metrics: L1 (552), L2 (614, 4-way unroll), L2_Simple (681), SO2 (720, last dim only,
unsquared, ±π single wrap), SO3 (763, delegates to L2_Simple). All must provide
`evalMetric` AND per-axis `accum_dist` (used for split-plane bounds).

## Stale / questionable spots — do NOT port verbatim

- `size_at_index_build_` (1127): written in 6 places, read nowhere, no accessor. Dead.
  Stale-query behavior actually comes from `vAcc_`/`size_` fixed at build.
- `RKNNResultSet::init` seeds `dists[capacity-1] = maximumSearchDistanceSquared` —
  vestigial: `worstDist()` (395) returns the member when not full.
- `KDTreeSingleIndexDynamicAdaptor::findNeighbors` (2705) has NO emptiness checks →
  returns `true` on an empty index for radius/box result sets. The forest also has no
  `knnSearch`/`radiusSearch` at all, only `findNeighbors`/`addPoints`/`removePoint`/
  `getAllIndices`.
- README §2.2 documents a `checks` parameter that no longer exists.
- `KDTreeSingleIndexDynamicAdaptor_::operator=` (2331) cannot rebind its `treeIndex_`
  reference member; duplicated `public:` label at 2489.

## Dynamic adaptor (M2) essentials

- Ctor: `maximumPointCount = 1000000000U` default; `treeCount_ = log2(max)+1`; auto
  `addPoints(0, n-1)` for pre-existing points.
- Merge into tree `pos` (2654): for each moved point `e`:
  `if (treeIndex_[e] != -1) treeIndex_[e] = pos; else removedPoints_[e] = pos;` —
  the tombstone update is the `else` branch. Re-adds reactivate tombstones in place
  (2640). Deleted-point skip is CRTP `isActive()` (2288: `treeIndex_[idx] != -1`)
  consulted in `searchLevel` (1240).

## Incremental adaptor (M3) essentials

One-point-per-node scapegoat tree (`INode` 2801: ptIdx, divfeat, deleted, treeDeleted,
child1/child2/parent, subtree_size, invalid_count, box, pcoord; child1 doubles as
free-list link). Constants: `kMinBalanceRebuild = 4`, `kBulkInsertFraction = 0.5`,
alphas 0.75/0.5 (`KDTreeIncrementalIndexParams`, 2728). `pendingRebuild_`: only the
HIGHEST unbalanced node found in one insertion is rebuilt (set 3487, consumed 3467).
`removePoint` (2942) walks ancestors for `treeDeleted` and no-ops if any is set (no
double decrement). Bulk switch (2916): batch >= 0.5·liveCount → flatten + rebuild.
API: setInlineRebuild, snapshotLiveIndices, collectPhysicalIndices, referencesIndex,
buildFromIndices (validates before mutating), setCollectRemovedPoints /
acquireRemovedPoints, physicalSize, boundingBox (O(1)), reserve, empty.
Serialization: magic `NFLI` (0x4E464C49, 3145), field-by-field topology + flags;
bbox/subtree_size/parent/pcoord recomputed on load. Coord cache gated by
`NANOFLANN_INCREMENTAL_NO_COORD_CACHE`; `kCacheCoords = (DIM > 0)` (2825).

## MT adaptor (M4) essentials

`KDTreeSingleIndexIncrementalAdaptorMT` (3916): ONE persistent worker thread (4308),
`workerCvJob_`/`workerCvDone_` condition variables, `std::exception_ptr buildError_`
(4315) marshals background-build exceptions to the foreground. Model: op-log + snapshot
build + atomic swap in `integrateIfReady()` (4220). Ctor defaults `rebuild_growth=1.3`,
`min_rebuild_size=10000`. Requires dataset storage stable during rebuild (3908). API:
size, empty, physicalSize, isRebuilding, boundingBox, snapshotLiveIndices, reserve,
sync, activeIndex, setRebuildCallback, setCollectRemovedPoints, acquireRemovedPoints.
No usedMemory.

## Static serialization (M-later)

Magic `NFLN` (0x4E464C4E, 1642); header: magic(u32)|version(u32)|sizeof(size_t)|
sizeof(IndexType)|sizeof(ElementType)|sizeof(DistanceType) (u8 each). `save_tree` dumps
raw `Node` structs; child-pointer NON-NULLNESS encodes topology, values are garbage and
overwritten on load (1616-1638). Our index-based nodes cannot be byte-compatible.

## Misc facts

- Macros: `NANOFLANN_FIRST_MATCH` (tie → smaller index, in shared insert 257),
  `NANOFLANN_NODE_ALIGNMENT` (default 16, applied to static `Node` only, not INode),
  `NANOFLANN_NO_THREADS` (throws "Multithreading is disabled" whenever normalized
  n_thread_build != 1; 0 on single-core normalizes to 1 → no throw).
- PooledAllocator: WORDSIZE=16, BLOCKSIZE=8192, first word of block = prev-block ptr.
- `n_thread_build`: 0 → `max(hardware_concurrency(),1)`, 1 → sequential, n → cap via
  `++thread_count < n_thread_build_` (1439); only the RIGHT subtree is offloaded;
  partition precedes spawning → parallel tree is IDENTICAL to sequential.
- `usedMemory` = pool used + wasted + vAcc bytes (1173).
- Static ctor forwards variadic args to the metric (1892) — stateful metrics; only the
  static adaptor does this.
- leaf_max_size default 10; README recommends 10–50.

## Implementation-fidelity notes (gap-audit N-3, N-9)

Two subtle line-for-line parity details worth knowing before touching
`search.rs` or `bbox.rs` again:

- **`worst_dist()` is called live, not hoisted** (`search.rs`'s leaf loop,
  mirrors nanoflann.hpp ~1259): the C++ source re-evaluates
  `result_set.worstDist()` inline in the loop condition on every leaf-point
  iteration, rather than caching it in a local before the loop starts. This
  port mirrors that exact call site rather than hoisting it, even though the
  two are functionally equivalent (`worst_dist` only changes via
  `add_point`, so hoisting would be a legal optimization) — kept as
  written-to-match rather than "improved", so future line-by-line parity
  audits against the C++ source don't have to reconcile a structural
  difference that isn't actually a behavior difference.
- **`compute_bounding_box`'s scan uses two independent one-sided
  comparisons per axis**, not a combined min/max: `if v < bbox[i].low {
  bbox[i].low = v; } if v > bbox[i].high { bbox[i].high = v; }` (`bbox.rs`),
  matching nanoflann's own bbox-scan comparison order exactly. This matters
  for bit-parity because `min`/`max`-style implementations can differ from
  this two-independent-`if` form on NaN inputs (which are out of this
  crate's domain anyway — see the README's "Input domain" section) or under
  aggressive fast-math reassociation (not used on either side here, but the
  explicit form leaves nothing for a future optimization pass to
  accidentally reassociate into a divergent comparison order).

## M1 outcome (2026-08-22)

M1 (the static kd-tree, `KDTreeSingleIndexAdaptor` only) is complete. Summary
for M2+ planning; the repo root `README.md` has the full user-facing story.

- **Parity:** full cross-validation suite (`crates/xval`) passes bit-exact
  (positional, `max_ulps = 0`) across the matrix — {f32,f64} × dims
  {2,3,8,16,32} runtime + `ConstDim<3>` × metrics {L1,L2,L2Simple,SO3}
  (+SO2 dim-2) × datasets {uniform, clustered, 30% duplicates, all-identical,
  exponential-spacing} × leaf {1,10,64}, including `vind` (tree-permutation)
  equality, not just query-result equality. Tie latitude was relaxed only
  within exactly-tied (bit-equal) distance groups, never across a real
  divergence — the one legitimate source of tie-order ambiguity is C++'s
  `std::sort` being unstable (this port's radius-result sort is stable
  instead, a documented deviation).
- **Vestigial C++ spots confirmed in practice** (matching this file's
  earlier "Stale / questionable spots" section, now empirically verified
  rather than just read off the source): `size_at_index_build_` really is
  dead and the query-time snapshot behavior really does come from
  `vAcc_`/`size_` alone (the A8 stale-growth test in `tree.rs` exercises
  this); `BoxResultSet::sort()` really is unreachable from the box-search
  path in both implementations (neither side calls it), so this port
  omitted it entirely rather than porting dead code.
- **The inf-input finding** (originally surfaced while building xval's
  exponential-spacing f32 test data, Task 11): feeding near-all-`+inf` point
  coordinates to either implementation's builder is out of the algorithm's
  domain, not a bug in either. An all-`+inf` bounding box has
  `span = inf - inf = NaN`, and NaN fails every `<` comparison used in
  split-axis/cutval selection, so partitioning can never shrink the
  candidate set. Verified independently on both sides on identical data: the
  C++ oracle **segfaults** (unbounded native recursion overflows the stack);
  the Rust builder **exhausts memory** (observed past 18 GB RSS before being
  killed, since its iterative builder has no stack-depth limit to hit
  first). Neither is a regression to fix in M1 — coordinates are assumed
  finite — but M2's dynamic adaptor (or a shared validation layer) is a
  reasonable place to add an explicit finite-coordinate check if silent
  non-termination on bad input ever becomes a real concern.
- **Speed:** all four perf gates pass with wide margin; build reached
  parity (ratio 1.018) and `knn_fixed3` closed to a ~4% asm-analyzed residual
  gap (recursive `search_level`/`searchLevel` call overhead, present on both
  sides — see `docs/benchmarks.md`). A pre-existing dim-32 knn gap
  (~1.3-1.45x) was found and is out of M1's scope (dim-3/dim-8 only);
  flagged for M2.

## M2 outcome (2026-08-23)

M2 (the dynamic Bentley–Saxe forest, `KDTreeSingleIndexDynamicAdaptor` +
its internal sub-tree class `KDTreeSingleIndexDynamicAdaptor_`) is
complete. Summary for M3+ planning; the repo root `README.md` has the full
user-facing story (see its "Dynamic adaptor (M2)" section), and
`docs/benchmarks.md`'s "M2 — dynamic forest" section has the numbers.

- **What was ported**: the forest bookkeeping (`add_points`/`remove_point`,
  the `first0bit` binary-counter slot-selection rule, the merge-and-rebuild
  schedule that rebuilds every slot up to the highest one touched — even a
  no-op call's slot 0 — every single call) plus the forest's own
  `findNeighbors` (every slot's full sub-tree search against one shared
  result set, tombstone-filtered). Each slot reuses M1's `SubtreeBuilder`
  (`base = 0`) and M1's `search::find_neighbors` verbatim — no
  parallel-search or parallel-build path for the forest exists on either
  side, so this crate's forest never spawns rayon workers regardless of
  the `parallel` feature.
- **The contiguity-contract discovery (a real C++ UB finding, not just a
  style note)**: nanoflann's `addPoints` indexes its `treeIndex_` array by
  the running `pointCount_` counter, not by the point index actually being
  processed — so a genuinely-new point's index MUST equal `pointCount_` at
  the moment it's added, or `treeIndex_` gets silently corrupted (a wrong
  slot recorded for a real index, with no error, no crash, just wrong
  bookkeeping that then propagates into every future merge). Confirmed
  empirically, not just read off the source: `nanoflann-ref`'s
  `add_points_misaligned_start_documents_silent_corruption_f32` regression
  test feeds the C++ oracle a genuinely-new index that doesn't line up with
  `pointCount_` and observes the corrupted bookkeeping directly. This
  port's `add_points` instead asserts the alignment per-index (only against
  genuinely-new indices — reactivations are exempt, since C++'s
  reactivation branch never touches `pointCount_` either) and panics loudly
  the instant it would occur — a documented, strictly-safer deviation:
  identical behavior to C++ on every legal call sequence, a loud panic
  instead of silent corruption on an illegal one. Full contract, including
  the exact reactivation exemption and its own oracle-verified regression
  test (`readd_point_1_reactivates_it_f32`), is in
  `crates/nanoflann-rs/src/dynamic.rs`'s `add_points` doc comment.
- **The per-call maxIndex/slot-0-retouch quirk**: nanoflann's rebuild pass
  after the per-point loop always runs `for (int i = 0; i <= maxIndex; ++i)`
  where `maxIndex` starts at `0` regardless of what happened in the loop —
  so slot 0 is unconditionally freed-and-maybe-rebuilt on EVERY
  `add_points` call, even a fully no-op one (an empty range, or a call that
  only reactivated already-tombstoned points and touched no slot at all).
  This port mirrors that exact rebuild schedule (not just final slot
  membership) — verified structurally correct because the M2 cross-
  validation suite checks per-slot `vind` order after every op, which would
  catch a schedule divergence even when final membership happened to agree.
- **Protected-member access**: like M1's static adaptor, the C++ sub-tree
  class's fields the port needs (`vAcc_`, node pool, bounding box) are
  `protected` in the vendored header, reached via the same pure
  using-subclass pattern the T1 oracle wrapper already used for the static
  adaptor in M1 — no new access-pattern discovery needed here, M1's
  approach generalized cleanly.
- **No `radiusSearch`/`knnSearch` on the forest**: confirmed empirically
  (not just by reading the header) that the vendored C++ forest class
  exposes only `findNeighbors`/`addPoints`/`removePoint`/`getAllIndices` —
  no richer search methods, and no box search at all (only the internal
  per-slot sub-tree class has those, and they're never called through the
  forest in any observed C++ call site or example). This port's additive
  `knn_search`/`rknn_search`/`radius_search` wrappers are composed the same
  way M1's `KdTree` composes its own convenience wrappers over
  `find_neighbors` — genuinely additive ergonomics, not upstream parity —
  and no `box_search` was added, matching the upstream gap exactly rather
  than inventing API surface nanoflann itself never shipped.
- **Empty-forest quirk, re-confirmed for the forest** (M1 Corrections #4's
  same finding, now re-verified at the forest level): a forest with zero
  occupied slots hits every slot's size-0 early return without ever
  touching the result set, so `find_neighbors`'s final `.full()` reflects
  an untouched result set — `false` for knn/rknn, but hardwired `true` for
  a fresh `RadiusResultSet` regardless of whether anything was ever added.
  Only observable through the generic `find_neighbors` escape hatch; the
  additive wrappers return the found count (`0`), where the quirk is
  invisible.
- **Parity:** the dynamic op-sequence cross-validation suite
  (`crates/xval/tests/xval_dynamic.rs`) passes bit-exact across its matrix
  (dims {2,3,8} × datasets {uniform, 30% duplicates} × leaf {1,10} × 3
  seeds, 120 generated ops/sequence) — per-op structure equality (every
  slot's point list, `tree_index`, `removed_len`) plus periodic knn/radius
  query parity, four scripted scenarios, and mutation canaries that pin the
  specific field/slot a deliberately-introduced divergence surfaces in. No
  genuine Rust-vs-C++ divergence was ever found.
- **Speed:** both new dynamic perf gates pass, though the margin is not
  wide — `dyn_add_20k_dim3_f32` has ranged **1.106–1.237** across repeated
  re-runs (within 1.1% of the 1.25 threshold at its high end) and
  `dyn_knn_after_churn_dim3_f32` (a workload with LIVE tombstones and real
  cross-slot merges, not a self-cancelling churn) has ranged **0.971–0.976**
  — see `docs/benchmarks.md`'s "M2 — dynamic forest" section for the single
  source of these numbers. The churned-forest accuracy row is bit-exact
  (`1.0`/`1.0` at `eps=0`). The
  pre-existing dim-3 knn residual and dim-32 knn gap flagged at the end of
  M1 remain open — M2 was scoped to the dynamic adaptor itself, not to
  closing those; both, plus parallel slot rebuilds for the dynamic
  adaptor, move to the M2.5 backlog (see `docs/ROADMAP.md`).

## M2.5 outcome (2026-08-23)

M2.5 (performance deep-dive) is complete. Two changes landed, both bit-exact
parity-preserving (full xval suite green throughout): **T3** root-caused and
closed the dim-32 knn gap (per-component bounds checks in the `L2`/`L1`
kernel were blocking LLVM's SLP vectorizer at runtime-known dim — not a
SIMD/batching problem as M1 had hypothesized; fixed with a bounds-check-free
chunked row walk, `DataSource::point_row` + `as_chunks::<4>()`, bit-exact
with the fallback: dim-32 f32 1.423x → 0.966–1.008x). **T2** converted
`search_level` from native self-recursion to an explicit-stack iteration
(`Frame`/`Phase`/`FrameStack`, `MaybeUninit`-backed inline array + heap-`Vec`
spill past depth 128), closing most of the fixed-dim-3 residual
(1.042–1.058x → 1.011–1.040x) and adding query-path stack-overflow immunity
on degenerate trees (C++'s native `searchLevel` recursion remains exposed)
— **at a reviewed, documented cost**: it gave back part of
`knn_dyn_dim8_f64_k10`'s and `radius_dim3_f32`'s prior improvement (both
remain Rust wins vs C++ throughout, ratio < 1.0) in exchange for the
fixed-dim-3 win plus a substantial `dyn_knn_after_churn` win
(0.971–0.976x → 0.873–0.916x in T4's fresh sweep). This is documented as a
trade-off, not a "no regression" outcome — full accounting in
`docs/benchmarks.md`'s "M2.5 — performance deep-dive" section. Open
residuals explicitly not chased this milestone (dim-64 f32, dim-32 f64, the
fixed-dim-3 last ~1-4%, a fast-math feature flag, parallel slot rebuilds):
`docs/ROADMAP.md`'s M2.5 entry.
