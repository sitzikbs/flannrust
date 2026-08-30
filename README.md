# flannrust

A Rust port of [nanoflann](https://github.com/jlblancoc/nanoflann) 1.12.1,
targeting bit-exact result parity with the C++ reference implementation and
equal-or-better speed. Two indexes are implemented so far: the **static**
kd-tree (`KDTreeSingleIndexAdaptor` -> `KdTree`), where the point set is
fixed at build time, and the **dynamic** Bentley–Saxe forest
(`KDTreeSingleIndexDynamicAdaptor` -> `DynamicKdTree`, see "Dynamic
adaptor" below) supporting point add/remove after construction. Incremental
(single self-balancing tree) and multithreaded-background-rebuild indexes
remain future milestones (see "Roadmap" below). Both indexes are also
available from Python (`KDTree`/`DynamicKDTree`, PyO3 bindings — see
"Python bindings (M-py)" below).

## Attribution & license

`flannrust` is a derivative port of [nanoflann](https://github.com/jlblancoc/nanoflann),
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

## How this was built

This codebase was implemented by an AI agent (Claude Code), directed and
reviewed by Itzik Ben-Shabat, who does not write Rust. Every line of Rust,
C++ FFI, and Python-binding code here is agent-written; the author's role
was specifying requirements, reviewing the generated code and each task's
report, and directing dedicated rigor/fidelity-audit passes (see
`docs/superpowers/plans/` and `docs/reports/`). Because the author cannot
independently vet Rust idioms or catch subtle logic errors by reading the
code, correctness here does not rest on the author's Rust expertise — it
rests on the bit-exact cross-validation suite run against the vendored
nanoflann 1.12.1 C++ reference implementation (see "Parity & testing"
below), which checks every result index, distance, and internal tree
permutation against the real C++ library, in-process, on every change.
Every performance figure quoted in this README and `docs/benchmarks.md`
traces to a reproducible, pasted command and run recorded in
`docs/EXPERIMENTS.md`; see `docs/reports/m-pub/claims-audit.md` for the
audit that verified that claim against both files' current text.

## Quickstart

```rust
use flannrust::{ConstDim, KdTreeBuilder};

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

(This is `crates/flannrust/src/lib.rs`'s crate-doc doctest, run under
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
| `search_level` is an explicit-stack iteration (`FrameStack`: 128 inline frames, heap-`Vec` spill beyond), not C++'s native `searchLevel` recursion (M2.5) | Same traversal order and results, bit-exact. Benefit: the query path cannot overflow the native stack on a degenerate tree (C++'s recursion can). Costs: a query deeper than 128 levels pays per-query heap allocation the recursive form never did, and the conversion gave back margin on `knn_dyn_dim8_f64_k10` and `radius_dim3_f32` (both still < 1.0 vs C++) — see "Safety" below and `docs/benchmarks.md`'s "M2.5 — performance deep-dive". |

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

## Safety (`unsafe` in this crate)

This crate contains exactly two `unsafe` blocks, both in
`crates/flannrust/src/search.rs`'s `FrameStack` (M2.5): a
`MaybeUninit::assume_init_read` in `pop` and an `assume_init_mut` in
`top_mut`, reading back frames of the explicit-stack query walk from a
fixed 128-slot inline array that is deliberately left uninitialized
(zero-filling it on every query was measured as a real regression). The
invariant is one line — slot `i` is initialized iff `i < inline_len` —
stated on the struct and referenced at each use site; it is the same
technique `arrayvec`/`smallvec` use. Why it is kept rather than replaced
by a safe `Vec`: a zero-`unsafe` `Vec::with_capacity(128)` frame stack was
A/B-tested against it and was slower in 8/8 interleaved repeats, median
**+4.7%** on the fixed-dim-3 knn gate (pasted in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)'s "M2.5 task 2" subsection).
Both blocks are miri-clean under both aliasing models (Stacked Borrows and
Tree Borrows, `-Zmiri-tree-borrows`), including a test that really
exercises the heap-spill path past 128 frames — commands and the pasted
runs are in `docs/EXPERIMENTS.md`'s "Miri" subsection; `docs/ROADMAP.md`
makes that miri job a required CI check for any change to `search.rs`.

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
   residual ~4% gap is analyzed, not hand-waved (M1-era; that recursion
   analysis is superseded by M2.5 — see "Benchmarks" below).
3. **Robustness** — heavy degenerate builds (dim-1 depth ~2115, dim-8 depth
   ~16794) pass in release; empty/`k > n`/duplicate edge cases covered.
   **Met**.
4. **Hygiene** — `cargo test --workspace` green with zero warnings (forced
   rebuild, including `cargo doc -p flannrust --no-deps`),
   `--no-default-features` builds, doctests pass. **Met**.

M2 (dynamic adaptor) was held to the same bar — bit-exact structure and
query parity across its own op-sequence matrix, both dynamic perf gates
passing, `cargo test --workspace` still green with zero warnings — also
**Met**; see this file's "Dynamic adaptor (M2)" section above for the
numbers and [`docs/nanoflann-notes.md`](docs/nanoflann-notes.md)'s "M2
outcome" section for the full source-fact record.

## Benchmarks

Machine: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), AMD Ryzen 7 9800X3D,
8 threads visible, `rustc 1.98.0`. Methodology: `RUSTFLAGS="-C
target-cpu=native"`, release profile (`codegen-units = 1`, `lto = "thin"` —
matched against the C++ oracle's single-translation-unit `-O3` build so
neither side gets an unforced codegen advantage); C++ built `-O3
-march=native -ffp-contract=off` (`-ffp-contract=off` specifically so
clang's default FMA contraction doesn't produce a bit-parity mismatch
against rustc, which never contracts). Ratio = `rust_time / cpp_time`; lower
is better for Rust. Full detail — including the M1-era asm analysis of
the then-recursive `knn_fixed3` walk (kept as history, superseded by the
M2.5 section that records the iterative walk that replaced it) — is in
[`docs/benchmarks.md`](docs/benchmarks.md) (this crate's single, unified
benchmarks doc — M1's static-tree, M2's dynamic-forest, and M2.5's
performance-deep-dive numbers all live there, in separate sections).
**Reproducing our numbers**:
every command behind every figure on this page and in `docs/benchmarks.md`
is listed, exact and copy-pasteable, in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), including the WSL2
measurement-noise caveat and the bare-metal re-run this crate still owes
before any public announcement.

### Perf gate (six gated workloads; pass threshold is ratio ≤ 1.25)

**Post-M2.5** (T4's fresh two-run sweep, commit `2d23db4`, median-of-7 —
full pasted runs in `docs/EXPERIMENTS.md`, milestone-by-milestone comparison
in `docs/benchmarks.md`'s "M2.5 — performance deep-dive" section):

| Workload | Ratio range (2 fresh runs) |
|---|---|
| `build_100k_dim3_f32_seq` | 0.974–1.079 |
| `knn_fixed3_dim3_f32_k10`¹ | 1.011–1.040 |
| `knn_dyn_dim8_f64_k10` | 0.950–0.966 |
| `radius_dim3_f32` | 0.827–0.828 |
| `dyn_add_20k_dim3_f32` | 1.121–1.170 |
| `dyn_knn_after_churn_dim3_f32` | 0.873–0.916 |

¹ within this host's documented noise floor — see the Update below.

**Update (M2.6, commit `a30a819`, 2026-08-25):** the table above is T4's
2-run, median-of-7 sweep, kept as-is (pasted evidence, never edited in
place). A statistically stronger re-verification — `xval::measure_pair`'s
adaptive `n=clamp(10,100,...)` harness, n=100 reps/side, mean/std/median
published, 4 independent sessions (2 gate + 2 report-chain) — widens or
corrects several of these ranges (full account: `docs/EXPERIMENTS.md`
"M2.6 task 2", `docs/benchmarks.md` "M2.6 — statistical re-verification"):

| Workload | Ratio range (4 fresh n=100 sessions) |
|---|---|
| `build_100k_dim3_f32_seq` | 0.967–1.039 (reproducibly split by configuration, not noise — see below; **Update (M2.6 task 6): re-hedged, did not reproduce a third time — see below**) |
| `knn_fixed3_dim3_f32_k10`¹ | 1.030–1.067 |
| `knn_dyn_dim8_f64_k10` | 0.929–0.937 (better than the entire pre-M2.5 range) |
| `radius_dim3_f32` | 0.807–0.867 |
| `dyn_add_20k_dim3_f32` | 1.112–1.149 (narrower than every prior range) |
| `dyn_knn_after_churn_dim3_f32` | 0.940–0.947 |

¹ within this host's documented noise floor, **now the n=100-per-side,
4-session characterization** (session medians 1.030–1.067, approx.
conservative envelope of 0.94–1.16 (extreme session medians ± 2×max
per-session σ) — `docs/EXPERIMENTS.md` "M2.6 task 2"), superseding
the old 8-run 0.956–1.192 figure (`docs/EXPERIMENTS.md` "M2.5 task 1",
kept as historical). Individual `knn_fixed3` old noise-sweep runs of
1.104/1.192 remain historical context, not the current floor.
**Update (M2.6 task 7): re-confirmed on the idle host** — 4 fresh
`knn_fixed3` gate sessions (loadavg-bracketed, host confirmed idle
throughout) compute their own conservative envelope at 0.884–1.176,
slightly *wider* than 0.94–1.16, not narrower, so there is no evidence
this figure was inflated by the game-load incident; 0.94–1.16 is kept as
the canonical floor unchanged (`docs/EXPERIMENTS.md` "M2.6 task 7").

All six pass the ≤1.25 gate in every session, both sweeps. `dyn_add_20k_dim3_f32`
looked like the tightest-margin gate under the old n=7 methodology (up to
1.237, ~1% below threshold); the new n=100 data settles at a narrower
1.112–1.149, suggesting the old high-water mark was itself small-sample
noise, not the true steady-state ratio — see the "Dynamic adaptor (M2)" →
"Performance" section below. Build's `build_100k_dim3_f32_seq` spread
(previously read as "suspiciously wide," 0.974–1.139) turns out to be
reproducibly split by configuration (**Update (M2.6 task 6): re-hedged,
did not reproduce a third time — see below**) — not primarily noise or drift, though
dataset seed and process context were never crossed to isolate which one
is the cause (`docs/EXPERIMENTS.md` conclusion (g)): the perf-gate binary's own dataset builds
consistently ~0.967 across sessions, while `report_data`'s independently
seeded dataset of the same shape builds consistently ~1.037 — each stable
to <1% on its own, ~7 points apart from the other (`docs/EXPERIMENTS.md`
"M2.6 task 2" conclusion (g)). `knn_fixed3`'s residual is no longer
recursive-call overhead: M2.5's T2 task
converted the query walk to a fully-inlined explicit-stack iteration — 0
`search_level` call targets remain in the compiled asm (reproduction
command and grep output in `docs/EXPERIMENTS.md`'s "M2.5 asm re-capture"
subsection). What the residual now IS has not been separately measured:
T2's reviewer attributes it to frame store/reload cost intrinsic to the
explicit-stack form, a diagnosis consistent with the A/B data but not
asm-quantified (see `docs/benchmarks.md`'s "Honest residuals" section); the
n=100 data honestly widens the residual itself to 1.030–1.067 (3.0–6.7%),
still comfortably inside the (also newly-characterized) noise floor.
**That conversion is a trade-off, not a free win**: it gave back all of
`dim8`'s M2.5 kernel-fix win (0.867 → 0.929–0.937 in the n=100 re-run —
now BETTER than dim8's entire pre-M2.5 range, not merely back inside it)
and left `radius` measurably worse than its pre-M2.5 range (0.767–0.792 →
0.807–0.867 in the n=100 re-run, wider than the single "~0.83"/0.827–0.828
point estimate previously published); both still beat C++ (ratio < 1.0).
In exchange: the `knn_fixed3` win, a substantial (if slightly narrower in
this fresh sweep, 0.940–0.947 vs. the earlier 0.873–0.916) `dyn_knn_after_churn`
win, and query-path stack-overflow immunity on degenerate trees that the
C++ oracle's native recursion still lacks — at the cost of per-query heap
allocation for queries deeper than 128 levels (see "Deliberate deviations"
above). Full honest accounting, both sweeps: `docs/benchmarks.md`'s "M2.5
— performance deep-dive" and "M2.6 — statistical re-verification" sections
(this is explicitly not a "no regression" story).

**Update (M2.6 task 5, commits `3d64c8b`/`a68df86`):** a line-by-line
fidelity audit against vendored nanoflann 1.12.1 found and fixed two real
divergences in `dyn_add`'s hot path (dynamic-forest merge-loop `Vec`
capacity preservation, and a 4-wide unrolled min/max scan matching
`middleSplit_`'s actual shape) — `dyn_add_20k_dim3_f32` **1.112–1.149 →
1.034–1.038**, closing roughly 90% of the only real (non-noise) gap this
milestone's fidelity audit found, with bit-exact parity preserved
throughout. `build_100k`/`knn_fixed3`/`radius`/`dim8` were audited too and
found to have no unported C++ behavior — their residuals are session
noise or the recorded, accepted iterative-search-conversion trade-off
above, not fixed. **Update (M2.6 task 6):** a fresh 3-session idle-host
sweep (2 gate + 1 report-chain) re-confirms every range above and the
`dyn_add` fix's durability:

| Workload | M2.6 task 2 (4 sessions) | M2.6 task 6 (3 fresh sessions) |
|---|---|---|
| `build_100k_dim3_f32_seq` | 0.967–1.039 | **0.993–1.009** |
| `knn_fixed3_dim3_f32_k10` | 1.030–1.067 | **1.011–1.040** |
| `knn_dyn_dim8_f64_k10` | 0.929–0.937 | **0.927–0.941** |
| `radius_dim3_f32` | 0.807–0.867 | **0.831–0.868** |
| `dyn_add_20k_dim3_f32` | 1.112–1.149 (pre-fix) | **1.0315–1.039** (post-fix) |
| `dyn_knn_after_churn_dim3_f32` | 0.940–0.947 | **0.931–0.958** |

**`build_100k`'s "reproducibly split by configuration" claim is corrected
here.** T4's fidelity audit re-ran the identical unpatched code/seeds and
could not reproduce the split (0.9825/0.984, no trace of the old
0.967-vs-1.037/1.039 clustering); this task's own fresh sweep corroborates
that a second, independent way — both gate sessions measured 1.009/1.009
and the report-chain session measured 0.9926, the **opposite** ordering
from the original split, all three clustered tightly at 0.993–1.009.
**Current verdict: `build_100k_dim3_f32_seq` shows no reproducible
configuration-dependent gap.** The original split most likely reflected
session/host-load state — consistent with (though not independently
proven to be caused by) a background game process the user identified as
consuming host compute during some earlier M2.6 sessions (see
`docs/EXPERIMENTS.md` §1's "Measurement-conditions protocol", added this
task). Full evidence: `docs/EXPERIMENTS.md` "M2.6 task 6" subsection;
`docs/benchmarks.md`'s "M2.6 task 6" section.

### dim-32/64 knn: the M1-era gap is closed at f32 (M2.5)

M1 flagged a real, reproducible dim-32 gap (**~1.3-1.45x**, outside M1's
success-criterion scope of dim-3/dim-8) that stayed open through M2. M2.5's
T3 task root-caused it (per-component bounds checks in the `L2`/`L1` kernel
blocking LLVM's SLP vectorizer at runtime-known dim — gcc compiled the
*identical* summation order to AVX2/AVX-512, proving the order was always
vectorizable) and fixed it with a bounds-check-free chunked row walk
(`DataSource::point_row` + `as_chunks::<4>()`, bit-exact with the fallback,
zero SIMD intrinsics, zero arithmetic reordering) — **not** the
SIMD/batching approach originally hypothesized as necessary. **This win has
a precondition**: it only applies when the `DataSource` impl overrides
`point_row` (the built-in `&[[T; N]]`, `FlatSlice`, and the dynamic
adaptor's `GrowableFlat` all do); a custom `DataSource` that only
implements `point_component` falls back to the per-component loop and gets
none of this fix — see `DataSource::point_row`'s doc comment
(`crates/flannrust/src/data_source.rs`) for what to implement to opt in.
Result, this task's (T4's) fresh two-run sweep:

| dim | scalar | before M2.5 | after M2.5 |
|---|---|---|---|
| 32 | f32 | 1.423 | **0.966–1.008** |
| 32 | f64 | 1.315 | 1.103–1.111 |
| 64 | f32 | 1.918 | 1.162–1.171 |
| 64 | f64 | 1.213 | 0.879–0.887 |

dim-32 f32 flips from a real C++ win to near-parity in T4's 2-run sweep;
dim-32 f64 and dim-64 f32 both improve substantially but remain open
residuals (not this task's target); dim-64 f64 lands under parity. Full
mechanism, A/B methodology, and residual analysis: `docs/benchmarks.md`'s
"M2.5 — performance deep-dive" section.

**Update (M2.6):** a fresh median-of-15, 2-session re-run
(`docs/EXPERIMENTS.md` "M2.6 task 2" conclusion (a)) widens dim-32 f32 to
**0.986–1.062** — straddling parity, including one session where C++ is
measurably faster (+6.2%). The headline gap closure (1.423x → ~1.0x) is
unaffected and re-confirmed; only the "parity-or-better" framing is
corrected — read this row as "near parity, occasionally a few percent
either way," not a guaranteed Rust win at every measurement.

**Update (M2.6 task 7, user-directed idle-host re-measurement):**
dim-32 f32 above is CONFIRMED (a fresh 2-session idle-host re-run nests
inside 0.986–1.062, `docs/EXPERIMENTS.md` "M2.6 task 7"). The other three
rows in the table above are CORRECTED — a fresh idle-host re-run,
combined with the M2.6-task-2 raw sweep data those rows never previously
incorporated, widens all three: dim-32 f64 **1.080–1.104** (was
1.103–1.111), dim-64 f32 **1.082–1.235** (was 1.162–1.171), dim-64 f64
**0.862–0.956** (was 0.879–0.887). None of this changes the qualitative
read (all three remain "improved substantially, not closed to parity" or
"under parity") — only the numeric bounds widen, and dim-64 in particular
turns out to be an inherently higher-variance workload on this host
(session-to-session spread persists even under confirmed-idle
conditions) rather than a contamination artifact. Full evidence and
per-row reasoning: `docs/EXPERIMENTS.md` "M2.6 task 7".

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
verified against the C++ source line-for-line: `crates/flannrust/src/dynamic.rs`'s
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

Both new dynamic perf gates pass (same 1.25-ratio pass bar as M1's four
static gates). `dyn_add_20k_dim3_f32` (contiguous adds from empty) has
ranged **1.106–1.237** across repeated re-runs on this WSL2 host under the
old median-of-7 methodology — the margin looked real, not "comfortable"
(1.237 is only 1.1% below the 1.25 threshold) — see
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for every pasted run.
**Update (M2.6):** a fresh n=100-per-side, 4-session re-verification
settles at a narrower **1.112–1.149** (`docs/EXPERIMENTS.md` "M2.6 task 2"
conclusion (f)) — comfortably inside the old range, with the old 1.237
high-water mark now reading as small-sample noise rather than the true
steady-state ratio; every session still comfortably clears the 1.25 gate.
**Update (M2.6 task 5, commits `3d64c8b`/`a68df86`):** a C++-fidelity audit
found and fixed two real divergences in the merge/rebuild hot path
(dynamic-forest merge-loop `Vec` capacity preservation + a 4-wide unrolled
min/max scan matching `middleSplit_`'s actual C++ shape) — `dyn_add`
**1.112–1.149 → 1.034–1.038**, bit-exact parity preserved throughout.
**Update (M2.6 task 6):** a fresh 3-session idle-host re-sweep confirms
the fix holds: **1.0315–1.039**.
`dyn_knn_after_churn_dim3_f32` (knn against a forest with LIVE
tombstones and real cross-slot merges, not a self-cancelling churn — see
`docs/benchmarks.md` for why the workload was corrected) originally ranged
**0.971–0.976**; **M2.5's T2 task** (explicit-stack iterative
`search_level`, see "Benchmarks" above) improved it substantially further,
to **0.873–0.916** in T4's fresh two-run sweep — one of M2.5's two clear
wins, not a give-back workload. **Update (M2.6):** a fresh n=100-per-side,
4-session re-verification narrows this to **0.940–0.947** — still a clear,
comfortable win over the pre-M2.5 0.956–0.976 band, though not as large a
margin as T4's own 2-run sweep suggested. **Update (M2.6 task 6):** a
fresh 3-session idle-host re-sweep widens this slightly to **0.931–0.958**
— ordinary session movement, still a clear win. The churned-forest accuracy row is
bit-exact against the C++ oracle (`1.0`/`1.0`/bit-exact at `eps=0` on a
live-set-restricted brute-force ground truth). Full numbers, the
`dyn_add`/`dyn_churn`/`dyn_knn_after_churn` criterion benchmarks, and why
nanoflann's dynamic `addPoints` is inherently O(heavy)-per-point rather
than amortized O(1) (on both sides, by design — not a Rust regression):
[`docs/benchmarks.md`](docs/benchmarks.md)'s "M2 — dynamic forest" and
"M2.5 — performance deep-dive" sections, the single source for these
numbers (not repeated in more detail here).

Cross-validated the same way M1's static tree is (bit-exact, in-process,
against the vendored C++ oracle) — the dynamic-specific op-sequence suite
is documented as part of "Parity & testing" below, not repeated here.

## Python bindings (M-py)

**Distances (and every radius argument — `query(..., r=...)`,
`query_radius(..., r=...)`) are SQUARED for `l2`/`l2_simple`, exactly like
the Rust core above — NOT euclidean like `scipy.spatial.cKDTree`.** `l1` is
already unsquared (summed absolute value). This is the single most common
mistake porting code from `cKDTree`: square your radius before calling, and
expect squared values back.

Install (once published to PyPI as **`flannrust`** — see "Roadmap" below for
the M-pub packaging plan) or build from source today:

```bash
cd crates/flannrust-py
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

```python
import numpy as np
import flannrust

pts = np.random.default_rng(0).uniform(-10, 10, size=(100_000, 3)).astype(np.float32)
tree = flannrust.KDTree(pts, leaf_size=10, metric="l2", threads=None)

q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
dists, idxs = tree.query(q, k=5)                        # dists are SQUARED l2
r_idxs, r_dists = tree.query_radius(q, r=4.0)            # r is SQUARED too, strict `<`
box = tree.query_box(np.full(3, -1.0, dtype=np.float32),
                      np.full(3, 1.0, dtype=np.float32))  # inclusive [lo, hi]

dyn = flannrust.DynamicKDTree(dim=3, dtype="float32")
start, end = dyn.add_points(pts)                         # half-open [start, end)
dyn.remove_point(0)                                       # lazy tombstone
dists, idxs = dyn.query(q, k=5)
```

`crates/flannrust-py/python/tests/` (pytest, 348 passed at this milestone's
close) is the full behavior suite; `crates/flannrust-py/python/bench/bench_py.py`
is the benchmark script behind the numbers below.

### API

| | `KDTree` (static) | `DynamicKDTree` |
|---|---|---|
| construct | `KDTree(points, leaf_size=10, metric="l2", threads=None)` | `DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)` |
| `.query(x, k=1, r=None, eps=0.0, workers=1)` | `-> (dists, idxs)` | same |
| `.query_radius(x, r, sorted=True, eps=0.0, workers=1)` | `-> (idxs, dists)` — two lists of ragged 1-D arrays (note: opposite order from `.query`) | same |
| `.query_box(lo, hi)` | `-> uint32` array, inclusive `[lo, hi]`, traversal order | not exposed — nanoflann's C++ dynamic forest has no box-search surface either, see "Dynamic adaptor (M2)" above |
| `.add_points(points)` / `.remove_point(idx)` | n/a (static) | append (returns the new half-open index range) / lazily tombstone |
| `.data` | read-only NumPy **copy** of the dataset (not a view — mutating it does not affect the tree) | n/a |
| getters | `.n` `.dim` `.dtype` `.leaf_size` `.metric` | `.n_active` `.n_total` `.removed_len` `.dim` `.dtype` `.leaf_size` `.metric` `.capacity` |

`workers` follows `scipy`'s convention: `1` = sequential, `-1` = all cores,
`n > 1` = capped rayon pool; `threads`/`workers` release the GIL during the
build/query. `metric` is one of `"l2"`, `"l1"`, `"l2_simple"`. Full
docstrings live on each pyclass/pymethod
(`crates/flannrust-py/src/static_tree.rs`, `dynamic_tree.rs`).

### Parity — scoped, not blanket (binding controller ruling)

Cross-validated against `pynanoflann` (a pybind11 wrapper around C++
nanoflann) and `scipy.spatial.cKDTree`. `pynanoflann` 0.10.0 (the only PyPI
release) vendors nanoflann **1.5.5**, while flannrust's own C++ xval oracle
— the version the Rust kernel is deliberately bit-matched against — is
**1.12.1**; between those two versions nanoflann's distance-summation order
changed, producing exactly-1-ULP squared-distance differences for `dim >=
3` (a version-gap environment artifact against this one `pynanoflann`
build, not a flannrust defect — full root-cause writeup:
`docs/EXPERIMENTS.md`'s "M-py" subsection). The parity claim is therefore
**tie-free KNN index-sequence parity across the full spec'd matrix (96/96
nodes, bit-exact) plus `dim=2` KNN distance-value bit-exactness (4/4,
provably immune to the version gap)** — not a blanket "bit-exact
everywhere" claim; the 16 nodes attributable to the documented 1-ULP
boundary-flip mechanism (radius search at dim ∈ {8,32}/float32, and
near-tie/duplicate data at dim ∈ {3,8,32}/float32) are pinned as strict
`xfail`, not silently dropped. No tolerance was ever loosened to force a
green result.

### Benchmarks

Two full bench runs (range honesty, not averaged); full per-run tables and
every pasted command: `docs/benchmarks.md`'s "M-py" section and
`docs/EXPERIMENTS.md`'s "M-py" subsection.

| Criterion | Gate | Range (both runs) | Verdict |
|---|---|---|---|
| Batched knn dim3 f32 vs cKDTree (workers 1 and −1) | ≤ 1.00 | 0.712–0.830 | **MET** |
| Build vs pynanoflann (100k / 1M) | ≤ 1.00 | 0.543–0.561 | **MET** |
| dim-32 knn vs pynanoflann | ≤ 1.10 | 0.878–0.899 | **MET** |
| Per-call overhead | measured | flannrust ≈2.0–2.3µs, cKDTree ≈7.2–7.5µs, pynanoflann ≈2.3µs | **MET** |

All gated criteria pass, with margin, in both runs. **Honest misses**
(published alongside, not gated): `knn_batched_dim3_f32_..._workers1` vs
pynanoflann ranges 1.058–1.216 (flannrust 6–22% slower single-threaded;
near parity at `workers=-1`, 0.896–1.000); `knn_dim8_float64_..._workers1`
vs pynanoflann ranges 1.233–1.249 (flannrust 23–25% slower, the most
consistent miss in the matrix). flannrust beats cKDTree on every single
row, both runs, without exception. See `docs/ROADMAP.md`'s M-pub section
for the open investigation the dim8 f64 gap motivates.

**Update (M2.6 task 3, 2026-08-25):** `bench_py.py` gained the same
adaptive-`n` (`clamp(10,100,...)`), mean/std/median/min/max/n statistical
harness M2.6 task 1 gave the Rust-vs-C++ perf gate/report chain,
generalized to the 3 interleaved engines this bench compares — one fresh
full run (full pasted table, per-cell `n`, per-conclusion detail:
`docs/EXPERIMENTS.md` "M2.6 task 3", `docs/benchmarks.md` M-py section):

| Criterion | Gate | New range/value (1 run, adaptive n) | Verdict |
|---|---|---|---|
| Batched knn dim3 f32 vs cKDTree, workers=1 | ≤ 1.00 | **0.493–0.830** (this run's 0.493 is a flagged single-session outlier — see below) | **MET** |
| Batched knn dim3 f32 vs cKDTree, workers=−1 | ≤ 1.00 | **0.703–0.827** | **MET** |
| Build vs pynanoflann (100k / 1M) | ≤ 1.00 | 0.543–0.561 (100k **0.553** confirmed; 1M **0.553**, 0.002 above the old 0.551 high end) | **MET** |
| dim-32 knn vs pynanoflann | ≤ 1.10 | **0.878–0.922** | **MET** |
| Per-call overhead | measured | flannrust ≈2.0–2.4µs, cKDTree ≈7.2–8.5µs, pynanoflann ≈2.3–2.8µs | **MET** |

All gated criteria still pass, most with wide margin. The `workers=1`
**0.493** figure is a genuine single-session outlier, not a flannrust
speedup: this run's cKDTree median for that cell (698.9ms) is ~63% slower
than every prior recorded run (426–430ms) while flannrust's own median
(344.3ms) sits right where prior runs put it — a scheduler/thermal stall
landing on cKDTree's share of that cell (this task touched only the bench
harness and renderer, not `crates/flannrust` or `flannrust-py`'s query
code) — flagged for confirmation, not folded silently into "the new
range." **Honest misses, re-measured**: `knn_batched_dim3_f32_..._workers1`
vs pynanoflann is now **1.009** (near-parity — same single-session caveat
as above, NOT declared resolved; new range 1.009–1.216);
`knn_dim8_float64_..._workers1` vs pynanoflann is now **1.271**, an
ordinary widening of the miss (new range 1.233–1.271, still the most
consistent miss in the matrix, still an open M-pub investigation item).

**Update (M2.6 task 6, full fresh run, idle host, 2026-08-25):** the
flagged `workers=1` cell was re-measured on a confirmed-idle host with
loadavg captured before/after. `ratio_ckdtree` now measures **0.688**
(n=69) — cKDTree's own absolute median (423.19ms) is back in its
historical 426–430ms band, decisively away from the flagged session's
698.9ms — and `ratio_pynanoflann` now measures **1.055** (n=69), back
inside the old 1.058–1.216 range. **Verdict: CONFIRMED — the flagged
0.493/1.009 pair was host-load contamination (consistent with a
background game process the user identified as running during some
earlier M2.6 sessions), not a new flannrust-vs-cKDTree/pynanoflann steady
state.** Honest combined ranges going forward: `ratio_ckdtree`
**0.493–0.830**, `ratio_pynanoflann` **1.009–1.216** (both flagged outlier
points kept in the range, not deleted, per this repo's range-honesty
convention). Full detail: `docs/EXPERIMENTS.md` "M2.6 task 6" subsection.

## Roadmap

M1 (static kd-tree), M2 (dynamic Bentley–Saxe forest, this document's
"Dynamic adaptor (M2)" section above), **M2.5 (performance deep-dive)**, and
**M-py (Python bindings, this document's "Python bindings (M-py)" section
above)** are all **complete**. Remaining and future milestones (design notes
already captured in `docs/nanoflann-notes.md` so they don't need
re-deriving from the C++ source; full tracker:
[`docs/ROADMAP.md`](docs/ROADMAP.md)):

- **M2.5 — performance deep-dive (complete)**: two headline outcomes — the
  dim-32 knn gap (~1.3-1.45x) is closed at f32 to near-parity via a
  bounds-check-free chunked kernel row walk, and the fixed-dim-3 knn
  residual is substantially closed via an explicit-stack iterative
  `search_level` (a documented trade-off: it gave back all of `dim8`'s
  kernel-fix win and left `radius` measurably worse than its pre-M2.5 range,
  both still Rust wins vs C++, in exchange for that win, the churn win, and
  query-path stack-overflow immunity on degenerate trees C++ still lacks).
  See "Benchmarks" and "Dynamic adaptor (M2)" → "Performance" above, and
  `docs/benchmarks.md`'s "M2.5 — performance deep-dive" and "M2.6 —
  statistical re-verification" sections for the full honest accounting
  (M2.6 re-measured every one of these ranges at n=100/side across 4
  sessions — several widened or corrected, see the sections linked above).
  Future perf leads explicitly **not** taken this
  milestone: dim-64 f32 (1.16–1.23x across T3+T4 runs, improved not
  closed), dim-32 f64 (~1.10-1.11x, improved not closed), the fixed-dim-3
  residual's last ~3-7% (attributed by T2's reviewer to frame store/reload
  cost — not asm-quantified — and within this host's newly-characterized
  n=100 noise floor), a
  fast-math/reordered-arithmetic kernel feature flag, and parallel slot
  rebuilds for the dynamic adaptor. **Update (M2.6 task 4, fidelity audit):**
  the dim-32/64 f64 residual was asm-verified to be compiler codegen, not a
  Rust fidelity defect — the Rust kernel already mirrors nanoflann 1.12.1's
  arithmetic exactly (bit-parity holds), but LLVM emits conservative
  256-bit AVX2 for the kernel body while g++ emits 512-bit AVX-512 for the
  identical vectorizable summation on this AVX-512-capable host —
  **REJECTED as a further port-fidelity change**; a `RUSTFLAGS`-level
  512-bit-vector experiment remains a possible toolchain-level lever, still
  unexplored (`docs/reports/m2.6/task-4-report.md` §6). The fast-math flag and parallel slot
  rebuilds remain untaken ideas, unchanged. **Update (M2.6 task 7,
  user-directed idle-host re-measurement):** dim-64 f32 and dim-32 f64
  above are CORRECTED wider — dim-64 f32 **1.082–1.235** (was
  1.16–1.23x), dim-32 f64 **1.080–1.104** (was ~1.10-1.11x); dim-64 f64
  (not previously called out in this bullet) is also CORRECTED wider,
  **0.862–0.956** (was 0.879–0.887). "Improved substantially, not closed"
  still holds for all three; only the bounds move (`docs/EXPERIMENTS.md`
  "M2.6 task 7").
- **M3 — incremental adaptor** (`KDTreeSingleIndexIncrementalAdaptor`, a
  single scapegoat-style self-balancing tree).
- **M4 — multithreaded wrapper** (`KDTreeSingleIndexIncrementalAdaptorMT`,
  background rebuild via a persistent worker thread, op-log + snapshot +
  atomic swap).
- **Serialization** (static tree save/load; nanoflann's `NFLN`/`NFLI` binary
  formats don't map byte-for-byte onto this crate's index-based node arena,
  so this needs its own format).
- **M-py — Python bindings (complete)**: `KDTree`/`DynamicKDTree` PyO3
  bindings, copy-in NumPy build input (one copy at construction; zero-copy
  input deferred), GIL released during build/query,
  benchmarked against `scipy.spatial.cKDTree` and `pynanoflann` — see
  "Python bindings (M-py)" above for the API, the scoped parity claim, and
  the honest bench accounting (all three spec'd speed gates met; two
  non-gating misses vs pynanoflann published alongside; M2.6 task 3
  re-measured every one of these ranges with an adaptive-`n` statistical
  harness — one flagged single-session outlier, resolved by M2.6 task 6's
  fresh idle-host re-run as host-load contamination, not a new steady
  state — still all MET, see the "Benchmarks" → "Update (M2.6 task 3)"/
  "Update (M2.6 task 6)" entries above).
- **M-pub — announcement readiness** (bare-metal re-run under the M2.6
  task 6 measurement-conditions protocol, claims audit, CI matrix +
  wheel/PyPI publish, crates.io dry run) — not started; see
  `docs/ROADMAP.md`.

See `docs/nanoflann-notes.md` for the verified C++ source facts (class
inventory, constants, API surfaces, stale/dead code to avoid porting
verbatim) backing each of these.
