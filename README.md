# nanoflann-rs

A Rust port of [nanoflann](https://github.com/jlblancoc/nanoflann) 1.12.1,
targeting bit-exact result parity with the C++ reference implementation and
equal-or-better speed. Two indexes are implemented so far: the **static**
kd-tree (`KDTreeSingleIndexAdaptor` -> `KdTree`), where the point set is
fixed at build time, and the **dynamic** Bentley–Saxe forest
(`KDTreeSingleIndexDynamicAdaptor` -> `DynamicKdTree`, see "Dynamic
adaptor" below) supporting point add/remove after construction. Incremental
(single self-balancing tree) and multithreaded-background-rebuild indexes
remain future milestones (see "Roadmap" below).

## Attribution & license

`nanoflann-rs` is a derivative port of [nanoflann](https://github.com/jlblancoc/nanoflann),
credited to Jose Luis Blanco-Claraco et al., which itself builds on FLANN by
Marius Muja and David G. Lowe. This crate is licensed under BSD-2-Clause (see
[`LICENSE`](LICENSE)); the vendored, unmodified C++ header
(`crates/nanoflann-ref/cpp/nanoflann.hpp`, used only as a cross-validation and
benchmark oracle, not part of the Rust library) retains its own original
copyright notice, reproduced verbatim in `LICENSE`.

Milestones 1 (M1 — static kd-tree) and 2 (M2 — dynamic adaptor) are
complete. The full design records live in
[`docs/superpowers/plans/2026-08-22-nanoflann-rs-m1.md`](docs/superpowers/plans/2026-08-22-nanoflann-rs-m1.md)
and
[`docs/superpowers/plans/2026-08-22-nanoflann-rs-m2-dynamic.md`](docs/superpowers/plans/2026-08-22-nanoflann-rs-m2-dynamic.md);
verified nanoflann 1.12.1 source notes (class inventory, stale spots, and
everything M3+ needs to know about the incremental/MT adaptors) live in
[`docs/nanoflann-notes.md`](docs/nanoflann-notes.md). Reproducible
benchmarks and experimental setup: [`docs/benchmarks.md`](docs/benchmarks.md)
and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Quickstart

```rust
use nanoflann_rs::{ConstDim, KdTreeBuilder};

let pts: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],
    [10.0, 10.0, 10.0],
    [1.0, 1.0, 1.0],
];
let tree = KdTreeBuilder::new(ConstDim::<3>, pts).build();

let mut indices = [0u32; 2];
let mut dists = [0.0f64; 2];
let found = tree.knn_search(&[0.1, 0.1, 0.1], &mut indices, &mut dists);

assert_eq!(found, 2);
assert_eq!(indices[0], 0); // nearest point is [0.0, 0.0, 0.0]
```

(This is `crates/nanoflann-rs/src/lib.rs`'s crate-doc doctest, run under
`cargo test --workspace` on every commit.)

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
  upstream — see `docs/nanoflann-notes.md` — the real snapshot behavior comes
  from `vAcc_`/`size_`, which is what we actually port).
- **`ResultItem` is `#[repr(C)] { index, distance }`**, the layout analog of
  nanoflann's `first`/`second` standard-layout `ResultItem`.

## Deliberate deviations from C++

| Deviation | Rationale |
|---|---|
| An unbuilt index is unrepresentable: `KdTreeBuilder::build()` consumes the builder and returns a ready `KdTree` | Replaces C++'s runtime `std::runtime_error` throw from `findNeighbors` etc. on an unbuilt index with a compile-time impossibility — there is no "forgot to call build()" bug class in this API. |
| Radius results' `sort()` uses Rust's **stable** sort | C++'s `IndexDist_Sorter` uses `std::sort`, which is **unstable** — this is the one place where output order may legally differ between the two implementations, and only among exactly-tied distances. |
| `BuildThreads::Threads(n)` means "at most `n` rayon workers" | Not C++'s exact async per-node thread-gating (`++thread_count < n_thread_build_`); both produce a tree bit-identical to a sequential build (partition precedes spawning on both sides), so this only affects *how* the work is scheduled, never the result. |
| No `NANOFLANN_NODE_ALIGNMENT` (16-byte node alignment) | Evaluated as a bench-gated perf candidate, not applied unconditionally as a default — see `docs/benchmarks.md`. |
| `KdTree::knn_search_with` accepts a `SearchParams` | C++'s static `knnSearch` takes none — this is additive, not a narrowing. |
| `BoxResultSet::sort()` not ported | Dead code upstream: nanoflann's own box-search path never calls it (callers own the output `Vec` and can sort it themselves if desired). |

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

## Parity & testing

The `xval` crate cross-validates every query kind against the vendored C++
nanoflann 1.12.1 source, compiled to a native library and called in-process
through an `extern "C"` FFI wrapper (`crates/nanoflann-ref`) — not a
subprocess, not a serialized comparison against pre-recorded output.

- **Bit-exact positional comparison** by default: indices and distances must
  match exactly, in the same order, between Rust and C++.
- **Tie latitude only within exact ties**: `ties = true` comparators relax
  index *order* only among groups of results that are bit-equal in distance
  — never across a genuine distance difference. This exists because C++'s
  `std::sort` (used by radius search's optional sort) is unstable, so
  equal-distance order is the one place outputs may legally differ (see
  "Deliberate deviations" above).
- **Tree-permutation equality**: `vind` (the build-time point-index
  permutation, `point_indices()` in this API) is checked directly, not just
  query results, so structural divergences can't hide behind coincidentally
  matching queries.
- **Mutation canaries**: tests that deliberately perturb a comparator (tie
  rule, distance-bit-equality check) to confirm it actually *can* fail —
  guards against tautological assertions.
- **Matrix coverage** (`{f32, f64}` scalars throughout; queries are a mix of
  uniform-random, exact copies of dataset points, and far-outside-bbox
  points — see `xval::queries`):
  - **knn/rknn suite**: dims `{2, 3, 8, 16, 32}` (runtime `DynDim`) × metrics
    `{L1, L2, L2Simple, SO3}` × datasets `{uniform, clustered, 30%
    duplicates, all-identical}` × `leaf_max_size ∈ {1, 10, 64}` × `k ∈ {1,
    10}`, 60 seeded queries per case (no exponential-spacing dataset here);
    `eps ∈ {0, 0.1, 1.0}` is a separate, narrower pass (dims `{3, 16}`, 40
    queries); rknn is its own pass (dim 3, 30 queries); `k = 101 > n = 50` is
    a dedicated k>n test, not part of the main matrix; `SO2` gets its own
    dim-2 pass (leaf `{1, 10}`, k `{1, 10}`, 60 queries); plus a `ConstDim<3>`
    spot check against the C++ runtime-dim index.
  - **radius/box suite**: dims `{2, 3, 8}` × datasets `{uniform, 30%
    duplicates}` × `leaf_max_size ∈ {1, 10}`, `L2` only, 60 seeded queries at
    two radii (selective and broad) per case; `L1`, `SO2` (dim 3), and
    `all-identical` are separate, narrower breadth tests, not folded into the
    main dim×dataset×leaf loop.
  - **build-parity (`vind`) suite**: the one place exponential-spacing data
    is exercised, cross-validating the build-time point permutation
    (`vAcc_`/`point_indices()`) directly against the C++ oracle — build
    only; degenerate-tree **query** parity against C++ is not covered by any
    suite.
  - Empty-tree edge cases are covered for every search kind.

### Dynamic (M2) cross-validation

`crates/xval/tests/xval_dynamic.rs` extends the same bit-exact,
in-process, against-the-vendored-oracle methodology above to mutation
*sequences* on `DynamicKdTree`, not just single builds/queries:

- **Matrix**: dims `{2, 3, 8}` × datasets `{uniform, 30% duplicates}` ×
  `leaf_max_size ∈ {1, 10}` × 3 seeds, 120 generated ops per sequence.
- **Op-sequence generator's legality model**: `xval::dyn_ops` produces
  `GrowAndAdd{count}`/`Remove{live_idx}`/`ReAdd{removed_idx}` (weighted
  50/30/20, each step restricted to whichever kinds are currently legal
  given the sequence-so-far, and renormalized), so it never emits an op
  that would violate `add_points`'s contiguity contract or remove/re-add
  a point that isn't in the right state. This legality model is not just
  trusted: `validate_dyn_ops_legal` is an *independent*, from-scratch
  reimplementation of the same legality rules (not calling back into
  `dyn_ops`'s internal state), run over many seeds as its own property
  test — a bug shared between generation and validation couldn't hide.
- **Per-op structure equality**: after every op in every sequence,
  element-for-element (not membership-only) comparison of `tree_count`,
  **every slot's own point list** (`vind`, exact order — merge-append
  order and permuting-rebuild order are both pinned, not just final
  membership), **`tree_index()`** (every dataset index's current slot
  *or* tombstone value, not just the occupied/live ones), and
  `removed_len()`. Every 10th op additionally checks knn + radius query
  parity over 30 fixed queries.
- **Scripted scenarios**: tombstone migration across a merge (a removed
  point's recorded slot correctly follows its physical storage when an
  unrelated merge moves it), drain-and-refill (remove everything, re-add
  everything, full parity restored), empty-forest parity, and a
  non-vacuous eps-pruning case (60 live points, `leaf_max_size(1)`, `k=2`
  — deep enough that `eps` genuinely changes which nodes get pruned; an
  earlier version of this scenario was provably vacuous at `k` too close
  to the live count and was replaced, not just patched, once that was
  caught).
- **Mutation canaries**: a `remove_point` skipped only on the Rust side
  (oracle still applies it) and a doctored per-slot point-list copy, each
  confirmed via `catch_unwind` to actually panic, with the panic message
  asserted to name the *specific* field/slot that diverged (`tree_index`
  or `slot N vind`) — not just that some assertion fired, which could
  hide the wrong comparator catching an unrelated bug.

No genuine Rust-vs-C++ divergence has ever been found by this suite —
every structure comparison, all scripted scenarios, and the full matrix
pass bit-exact. The canaries and coverage-band assertions (which pin the
generator's observed op-kind/tombstone-migration counts to a tolerance
band, so the matrix can't silently degrade into a suite that never
exercises a merge) exist to keep that finding trustworthy, not to paper
over a real one.

### Success criteria (binding, from the plan's §5b)

M1 was defined as done only when all four hold — current status:

1. **Parity** — entire xval suite passes with `max_ulps = 0` defaults: vind
   (tree permutation) equality, knn/rknn/radius/box result equality across
   the full config matrix, boundary conventions locked. **Met**: full parity
   green (bit-exact knn/rknn/radius/box/vind across the matrix).
2. **Speed** — (a) automated perf gate (`xval/tests/perf_gate.rs`, release,
   `#[ignore]`-gated): median Rust time ≤ 1.25× median C++ time on fixed
   workloads (a failure here blocks completion); (b) headline criterion
   numbers (dim-3 f32/f64 knn `k ∈ {1, 10, 100}`; build 100k/1M
   sequential+parallel) show Rust ≥ C++ (ratio ≤ 1.0) or a documented gap
   analysis. **Met**: all four perf gates pass with wide margin (see
   "Benchmarks" below); build is at parity (ratio 1.018); knn_fixed3's
   residual ~4% gap is analyzed, not hand-waved (asm-evidenced, see
   `docs/benchmarks.md`).
3. **Robustness** — heavy degenerate builds (dim-1 depth ~2115, dim-8 depth
   ~16794) pass in release; empty/`k > n`/duplicate edge cases covered.
   **Met**.
4. **Hygiene** — `cargo test --workspace` green with zero warnings (forced
   rebuild), `--no-default-features` builds, doctests pass. **Met**.

M2 (dynamic adaptor) was held to the same bar — bit-exact structure and
query parity across its own op-sequence matrix, both dynamic perf gates
passing, `cargo test --workspace` still green with zero warnings — also
**Met**; see this file's "Dynamic adaptor (M2)" section above for the
numbers and `.superpowers/sdd/2026-08-22-nanoflann-rs-m2-dynamic/progress.md`
for the full per-task review trail.

## Benchmarks (M1 — static kd-tree)

Machine: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), AMD Ryzen 7 9800X3D,
8 threads visible, `rustc 1.98.0`. Methodology: `RUSTFLAGS="-C
target-cpu=native"`, release profile (`codegen-units = 1`, `lto = "thin"` —
matched against the C++ oracle's single-translation-unit `-O3` build so
neither side gets an unforced codegen advantage); C++ built `-O3
-march=native -ffp-contract=off` (`-ffp-contract=off` specifically so
clang's default FMA contraction doesn't produce a bit-parity mismatch
against rustc, which never contracts). Ratio = `rust_time / cpp_time`; lower
is better for Rust. Full detail, including an asm-level analysis of the
residual `knn_fixed3` gap, is in
[`docs/benchmarks.md`](docs/benchmarks.md) (this crate's single, unified
benchmarks doc — M1's static-tree numbers and M2's dynamic-forest numbers
both live there now, in separate sections). **Reproducing our numbers**:
every command behind every figure on this page and in `docs/benchmarks.md`
is listed, exact and copy-pasteable, in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), including the WSL2
measurement-noise caveat and the bare-metal re-run this crate still owes
before any public announcement.

### Perf gate (four gated workloads; pass threshold is ratio ≤ 1.25)

| Workload | Final ratio |
|---|---|
| `build_100k_dim3_f32_seq` | 1.018 |
| `knn_fixed3_dim3_f32_k10` | 1.042 |
| `knn_dyn_dim8_f64_k10` | 0.973 |
| `radius_dim3_f32` | 0.767 |

All four pass with comfortable margin. Build is effectively at parity;
`knn_fixed3`'s ~4% residual gap is architectural (recursive `search_level`
call overhead, present on both sides — confirmed via asm inspection that
LLVM does not inline the self-recursive tree walk) rather than a missed
optimization in the hot kernel, and is documented at length in
`docs/benchmarks.md` rather than silently accepted.

### Known gap: dim-32 knn

Outside the perf gate's covered workloads (dim-3, dim-8) and outside M1's
success-criterion scope (dim-3 f32/f64 knn and build), dim-32 knn shows a
real, reproducible **~1.3-1.45x** ratio, while dim-8 sits close to parity
(0.97-1.00) — confirmed pre-dating the M1 performance pass by checking out an
earlier commit and re-running the identical benchmark, so it isn't something
that pass introduced or regressed. Root cause not yet isolated; the working
hypothesis is that closing it needs SIMD/batching to amortize the per-axis
L2 kernel over a wider dimension, which is out of scope for M1. Still open
after M2 (M2 was scoped to the dynamic adaptor, not this gap) — tracked in
the M2.5 backlog (see "Roadmap" below) for investigation.

### `leaf_max_size` sweep

Swept `{1, 4, 10, 16, 32, 50, 128, 1024}` for both libraries (criterion,
`xval/benches/bench_build.rs`) — the default of 10 (nanoflann's own default,
also this crate's default) remains a good choice for this node layout; see
`docs/benchmarks.md` for the full table.

## Dynamic adaptor (M2)

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
verified against the C++ source line-for-line: `crates/nanoflann-rs/src/dynamic.rs`'s
module and `add_points` doc comments; source facts:
[`docs/nanoflann-notes.md`](docs/nanoflann-notes.md)'s "M2 outcome" section.

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

### Performance

Both new dynamic perf gates (same 1.25-ratio pass bar as M1's four static
gates) pass with comfortable margin: `dyn_add_20k_dim3_f32` (contiguous
adds from empty) at **1.113**, `dyn_knn_after_churn_dim3_f32` (knn after a
realistic remove/re-add churn) at **0.964**. The churned-forest accuracy
row is bit-exact against the C++ oracle (`1.0`/`1.0`/bit-exact at `eps=0`
on a live-set-restricted brute-force ground truth). Full numbers, the
`dyn_add`/`dyn_churn`/`dyn_knn_after_churn` criterion benchmarks, and why
nanoflann's dynamic `addPoints` is inherently O(heavy)-per-point rather
than amortized O(1) (on both sides, by design — not a Rust regression):
[`docs/benchmarks.md`](docs/benchmarks.md)'s "M2 — dynamic forest" section.

Cross-validated the same way M1's static tree is (bit-exact, in-process,
against the vendored C++ oracle) — the dynamic-specific op-sequence suite
is documented as part of "Parity & testing" below, not repeated here.

## Roadmap

M1 (static kd-tree) and M2 (dynamic Bentley–Saxe forest, this document's
"Dynamic adaptor (M2)" section above) are both **complete**. Remaining and
future milestones (design notes already captured in
`docs/nanoflann-notes.md` so they don't need re-deriving from the C++
source; full tracker: [`docs/ROADMAP.md`](docs/ROADMAP.md)):

- **M2.5 — performance deep-dive** (up next): the fixed-dim-3 knn residual
  (~4-6%, asm-attributed to recursive call overhead) and the dim-32 knn gap
  (~1.3-1.45×, root cause not yet isolated — see "Known gap: dim-32 knn"
  above) both remain open past M2, plus parallel slot rebuilds for the
  dynamic adaptor (C++ rebuilds sequentially; a deviation here would need
  documenting like `BuildThreads` already is).
- **M3 — incremental adaptor** (`KDTreeSingleIndexIncrementalAdaptor`, a
  single scapegoat-style self-balancing tree).
- **M4 — multithreaded wrapper** (`KDTreeSingleIndexIncrementalAdaptorMT`,
  background rebuild via a persistent worker thread, op-log + snapshot +
  atomic swap).
- **Serialization** (static tree save/load; nanoflann's `NFLN`/`NFLI` binary
  formats don't map byte-for-byte onto this crate's index-based node arena,
  so this needs its own format).
- **M-py — Python bindings** and **M-pub — announcement readiness**
  (bare-metal re-run, claims audit, CI, crates.io dry run) — unchanged from
  `docs/ROADMAP.md`, not started.

See `docs/nanoflann-notes.md` for the verified C++ source facts (class
inventory, constants, API surfaces, stale/dead code to avoid porting
verbatim) backing each of these.
