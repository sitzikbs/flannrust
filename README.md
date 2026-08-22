# nanoflann-rs

A Rust port of [nanoflann](https://github.com/jlblancoc/nanoflann) 1.12.1's
**static** kd-tree (`KDTreeSingleIndexAdaptor`), targeting bit-exact result
parity with the C++ reference implementation and equal-or-better speed.
"Static" means the point set is fixed at build time — dynamic insertion,
incremental rebuilding, and the multithreaded background-rebuild wrapper are
future milestones (see "Roadmap" below).

This is Milestone 1 (M1). The full design record lives in
[`docs/superpowers/plans/2026-08-22-nanoflann-rs-m1.md`](docs/superpowers/plans/2026-08-22-nanoflann-rs-m1.md);
verified nanoflann 1.12.1 source notes (class inventory, stale spots, and
everything M2+ needs to know about the dynamic/incremental/MT adaptors) live
in [`docs/nanoflann-notes.md`](docs/nanoflann-notes.md).

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
  (fully re-sort ties by ascending index).
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
| No `NANOFLANN_NODE_ALIGNMENT` (16-byte node alignment) | Evaluated as a Task 14 perf candidate and bench-gated, not applied unconditionally as a default — see `docs/benchmarks-m1.md`. |
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
  cross-validation testing (Task 11): the C++ oracle **segfaults** (unbounded
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
- **Matrix coverage**: `{f32, f64}` scalars, dims `{2, 3, 8, 16, 32}`
  (runtime `DynDim`) plus `ConstDim<3>`, metrics `{L1, L2, L2Simple, SO3}`
  (+ `SO2` at dim 2), datasets `{uniform, clustered, 30% duplicates,
  all-identical, exponential-spacing}`, `leaf_max_size ∈ {1, 10, 64}`, knn
  `k ∈ {1, 10, 101}` (including `k > n`), 100 seeded queries per case
  including on-dataset and far-outside points, `eps ∈ {0, 0.1, 1.0}` on both
  sides, and empty-tree edge cases for every search kind.

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
   `docs/benchmarks-m1.md`).
3. **Robustness** — heavy degenerate builds (dim-1 depth ~2115, dim-8 depth
   ~16794) pass in release; empty/`k > n`/duplicate edge cases covered.
   **Met**.
4. **Hygiene** — `cargo test --workspace` green with zero warnings (forced
   rebuild), `--no-default-features` builds, doctests pass. **Met**.

## Benchmarks

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
[`docs/benchmarks-m1.md`](docs/benchmarks-m1.md).

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
`docs/benchmarks-m1.md` rather than silently accepted.

### Known gap: dim-32 knn

Outside the perf gate's covered workloads (dim-3, dim-8) and outside M1's
success-criterion scope (dim-3 f32/f64 knn and build), dim-32 knn shows a
real, reproducible **~1.3-1.45x** ratio, while dim-8 sits close to parity
(0.97-1.00) — confirmed pre-dating Task 14's perf pass by checking out an
earlier commit and re-running the identical benchmark, so it isn't something
Task 14 introduced or regressed. Root cause not yet isolated; the working
hypothesis is that closing it needs SIMD/batching to amortize the per-axis
L2 kernel over a wider dimension, which is out of scope for M1. Tracked here
and in the M2 backlog for investigation.

### `leaf_max_size` sweep

Swept `{1, 4, 10, 16, 32, 50, 128, 1024}` for both libraries (criterion,
`xval/benches/bench_build.rs`) — the default of 10 (nanoflann's own default,
also this crate's default) remains a good choice for this node layout; see
`docs/benchmarks-m1.md` for the full table.

## Roadmap

M1 covers only the static kd-tree. Future milestones (design notes already
captured in `docs/nanoflann-notes.md` so they don't need re-deriving from
the C++ source):

- **M2 — dynamic adaptor** (`KDTreeSingleIndexDynamicAdaptor`, a
  Bentley–Saxe forest of static trees supporting point add/remove). Also
  where dim-32 knn's gap and general input-validation (NaN/inf rejection)
  are natural candidates to revisit.
- **M3 — incremental adaptor** (`KDTreeSingleIndexIncrementalAdaptor`, a
  single scapegoat-style self-balancing tree).
- **M4 — multithreaded wrapper** (`KDTreeSingleIndexIncrementalAdaptorMT`,
  background rebuild via a persistent worker thread, op-log + snapshot +
  atomic swap).
- **Serialization** (static tree save/load; nanoflann's `NFLN`/`NFLI` binary
  formats don't map byte-for-byte onto this crate's index-based node arena,
  so this needs its own format).

See `docs/nanoflann-notes.md` for the verified C++ source facts (class
inventory, constants, API surfaces, stale/dead code to avoid porting
verbatim) backing each of these.
