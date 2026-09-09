# flannrust testing & parity methodology

How flannrust's bit-exact-parity claim is verified. The contracts being
verified are documented in [semantics.md](semantics.md). Extracted from the
project README.

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
[`EXPERIMENTS.md`](EXPERIMENTS.md)'s "M2.5 task 2" subsection).
Both blocks are miri-clean under both aliasing models (Stacked Borrows and
Tree Borrows, `-Zmiri-tree-borrows`), including a test that really
exercises the heap-spill path past 128 frames — commands and the pasted
runs are in `EXPERIMENTS.md`'s "Miri" subsection; [`ROADMAP.md`](ROADMAP.md)
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
  [semantics.md](semantics.md)'s "Deliberate deviations").
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

### Dynamic cross-validation

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

## Python parity — scoped, not blanket (binding controller ruling)

Cross-validated against `pynanoflann` (a pybind11 wrapper around C++
nanoflann) and `scipy.spatial.cKDTree`. `pynanoflann` 0.10.0 (the only PyPI
release) vendors nanoflann **1.5.5**, while flannrust's own C++ xval oracle
— the version the Rust kernel is deliberately bit-matched against — is
**1.12.1**; between those two versions nanoflann's distance-summation order
changed, producing exactly-1-ULP squared-distance differences for `dim >=
3` (a version-gap environment artifact against this one `pynanoflann`
build, not a flannrust defect — full root-cause writeup:
`EXPERIMENTS.md`'s "M-py" subsection). The parity claim is therefore
**tie-free KNN index-sequence parity across the full spec'd matrix (96/96
nodes, bit-exact) plus `dim=2` KNN distance-value bit-exactness (4/4,
provably immune to the version gap)** — not a blanket "bit-exact
everywhere" claim; the 16 nodes attributable to the documented 1-ULP
boundary-flip mechanism (radius search at dim ∈ {8,32}/float32, and
near-tie/duplicate data at dim ∈ {3,8,32}/float32) are pinned as strict
`xfail`, not silently dropped. No tolerance was ever loosened to force a
green result.
