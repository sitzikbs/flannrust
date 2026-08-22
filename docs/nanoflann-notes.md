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
