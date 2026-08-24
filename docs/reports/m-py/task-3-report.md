# Task 3 report: `query_radius` + `query_box` + pynanoflann parity suite

## Status: DONE_WITH_CONCERNS

`query_radius`/`query_box` are implemented, correct, and fully verified by a
deterministic, Rust-unit-test-mirroring behavior suite. The pynanoflann
parity suite is implemented per spec across the full required matrix and is
**intentionally left with real, root-caused, documented failures** — per
the brief's binding escalation rule, no tolerance was loosened to force
green. See "Escalation" below for the full evidence trail.

## Files changed

- `crates/flannrust-py/src/static_tree.rs` — added `KDTree.query_radius`
  and `KDTree.query_box` pymethods, plus their generic `do_query_radius`/
  `do_query_box` helpers (same `for_each_variant!`/`py.detach` pattern as
  the existing `do_query`).
- `crates/flannrust-py/python/tests/test_query_radius_box.py` (new) — TDD
  behavior tests: strict `<` at an exact radius boundary, `sorted=False`
  traversal order (exact index-sequence lock, mirrors
  `crates/flannrust/src/tree.rs`'s
  `radius_search_sorted_true_gives_ascending_sorted_false_gives_traversal_order`
  unit test), box face-inclusion, box exact-traversal-order lock (mirrors
  `crates/flannrust/src/search.rs`'s `find_within_box_exact_traversal_order`),
  box on an empty tree.
- `crates/flannrust-py/python/tests/test_parity_pynanoflann.py` (new) —
  pynanoflann parity suite (see below).

No other files modified. `crates/flannrust`, `crates/nanoflann-ref`,
`crates/xval`, and Task 2's `test_behavior.py`/`test_bruteforce.py`/
`conftest.py` are untouched.

## API

```
.query_radius(x, r, sorted=True, eps=0.0, workers=1)
    -> (list[ndarray idxs (uint32)], list[ndarray dists])
```
`r` is the SQUARED radius for l2/l2_simple (unsquared for l1), strict `<`.
Returns two Python lists of length `m` (or 1 for a `(d,)` query) of ragged
1-D numpy arrays — order is `(idxs, dists)` per the brief's verbatim
signature (note: the opposite order from `query`'s `(dists, idxs)` — kept
exactly as specified).

```
.query_box(lo, hi) -> ndarray[uint32]
```
Inclusive `[lo, hi]` on both faces, traversal order, single box (no
batching, no params — nanoflann's box search has none).

No `unsafe` added (crate-wide invariant preserved).

## TDD: RED evidence

Before rebuilding, the new `test_query_radius_box.py` file failed with
`AttributeError` (methods didn't exist yet):

```
$ .venv/bin/pytest python/tests/test_query_radius_box.py -q
...
E       AttributeError: 'flannrust.KDTree' object has no attribute 'query_box'
...
FAILED python/tests/test_query_radius_box.py::test_query_radius_strict_less_than_at_exact_boundary[float32]
FAILED python/tests/test_query_radius_box.py::test_query_radius_strict_less_than_at_exact_boundary[float64]
FAILED python/tests/test_query_radius_box.py::test_query_radius_sorted_false_gives_traversal_order[float32]
FAILED python/tests/test_query_radius_box.py::test_query_radius_sorted_false_gives_traversal_order[float64]
FAILED python/tests/test_query_radius_box.py::test_query_radius_1d_query_returns_length_1_lists[float32]
FAILED python/tests/test_query_radius_box.py::test_query_radius_1d_query_returns_length_1_lists[float64]
FAILED python/tests/test_query_radius_box.py::test_query_box_face_inclusion[float32]
FAILED python/tests/test_query_radius_box.py::test_query_box_face_inclusion[float64]
FAILED python/tests/test_query_radius_box.py::test_query_box_exact_traversal_order[float32]
FAILED python/tests/test_query_radius_box.py::test_query_box_exact_traversal_order[float64]
FAILED python/tests/test_query_radius_box.py::test_query_box_empty_tree[float32]
FAILED python/tests/test_query_radius_box.py::test_query_box_empty_tree[float64]
12 failed in 0.05s
```

After implementing and `maturin develop --release`: all 12 GREEN (one fix
mid-flight — see below).

## Escalation: pynanoflann's vendored nanoflann version differs from flannrust's reference

**Verified finding.** pynanoflann 0.10.0 (the only release on PyPI — not
fixable by upgrading) vendors its own copy of `nanoflann.hpp`
(`nanoflann/include/nanoflann.hpp` in its sdist), with
`#define NANOFLANN_VERSION 0x155` → **nanoflann 1.5.5**. flannrust's own
xval reference (`crates/nanoflann-ref/cpp/nanoflann.hpp`) is
`NANOFLANN_VERSION 0x010C01` → **nanoflann 1.12.1**, and the Rust core is
*deliberately* bit-matched against that exact 1.12.1 kernel: see
`crates/flannrust/src/metric.rs`'s `l2_eval_row`, whose comment
`// Parentheses break the dependency chain -- same order as C++` is a
verbatim port of 1.12.1's `L2_Adaptor::evalMetric`.

**Root cause (code-diffed).** Between 1.5.5 and 1.12.1, nanoflann's
`L2_Adaptor`/`L1_Adaptor::evalMetric` changed how the 4-wide unrolled
loop's partial sums combine:

```cpp
// 1.5.5 (pynanoflann's vendored copy) -- sequential chain:
result += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;

// 1.12.1 (flannrust's reference, and flannrust's own Rust kernel) -- pairwise:
result += (diff0*diff0 + diff1*diff1) + (diff2*diff2 + diff3*diff3);
```

and the `dim < 4` remainder loop's iteration order also flipped (1.5.5:
ascending, `while`; 1.12.1: descending, fall-through `switch`). Floating
point addition is commutative but **not associative**, so these two valid
summation orders can produce squared-distance results differing by exactly
1 ULP for any `dim >= 3` (a 3+-term, order-sensitive sum).

**Empirical confirmation (positive control).** `dim == 2` is a pure
2-term, commutative-only sum — provably immune. Measured on this host:

| dim | distance mismatches (of 3000 probed entries) |
|---|---|
| 2  | 0 |
| 3  | 339 |
| 8  | 626 |
| 32 | 1195 |

Every mismatch found was verified (via bit-pattern inspection) to be
**exactly 1 ULP**, and in every probed case the correct nearest point was
still identified (the wobble did not flip *which* point was found nearest
in tie-free data at ordinary scale).

**Three observable consequences, same root cause:**
1. `query`'s distance value (after the l2 sqrt mapping) can be 1 ULP off
   pynanoflann's.
2. `query_radius`'s strict `<` threshold occasionally disagrees at the
   boundary — a candidate 1 ULP inside/outside the radius on one engine
   can land on the other side of the threshold on the other engine.
   Confirmed even for tie-free ("uniform") data at dim ∈ {8, 32}, n=20000.
3. Near-tied ("clustered") or exactly-duplicated ("duplicates") data can
   have its exact-tie-group *boundaries* shift between engines (two points
   that tie exactly under one summation order may differ by 1 ULP under
   the other, or vice versa) — on top of, and distinct from, the
   already-expected tie-break-*order* ambiguity between two independently
   implemented `KnnResultSet`s.

**Consistency with existing precedent.** `crates/xval/src/lib.rs`'s own
comparator (`impl_knn_comparator!`) requires bit-exact (`ulp_diff == 0`)
distances at *every* rank unconditionally, even in its "ties" mode — only
index *order* within an exact-tie run is relaxed there. xval's
zero-tolerance design works specifically because the Rust core is
kernel-matched to its C++ reference (1.12.1); it does not, and structurally
cannot, paper over a genuine different-summation-order artifact against a
*different* nanoflann version. No newer pynanoflann release exists to pin
to instead (`pip`/PyPI JSON API confirms 0.10.0 is latest; only releases
ever published are 0.0.9 and 0.10.0).

## Parity suite: what's actually verified, and what's honestly red

Per the escalation rule, tolerances were **not** loosened anywhere. One
legitimate test-design correction was made (documented inline, not a
tolerance change): a k-truncated trailing KNN tie-group's index *identity*
is skipped when the true tie-run may extend past `k` — this is an
inherently unprovable property (which specific exact-duplicate landed in
the last visible slot is implementation-defined once truncated), not a
loosened check.

`crates/flannrust-py/python/tests/test_parity_pynanoflann.py`, four test
functions:

1. **`test_knn_index_parity_uniform_full_matrix`** — full spec'd matrix
   (n ∈ {1000, 20000}, dims {2,3,8,32}, f32/f64, leaf {1,10,64}, metric
   {l2,l1}), k ∈ {1,10}, 200 seeded queries (20 for n=20000, incl.
   on-dataset points). Strict index-sequence equality. **96/96 nodes
   PASS** — this is the clean, unambiguous, fully-verified correctness
   result (tie-free data → no tie-break-order or ULP-boundary ambiguity
   possible; matches the finding above that the 1-ULP wobble never flips
   an argmin/ranking decision at this scale).
2. **`test_radius_index_parity_uniform_full_matrix`** — same matrix,
   `query_radius` at two selectivities (derived from our own
   already-verified knn, avoiding a redundant brute-force oracle), sorted
   sequence equality (float32-snapped radius so both sides use the
   bit-identical threshold — see module docstring's "Radius precision
   quirk": pynanoflann's C++ binding declares `radius_neighbors(...,
   float radius)`, narrowing even the float64-tree path to float32).
   **86/96 pass, 10 fail** — all 10 failures are dim ∈ {8,32}, dtype
   float32, exactly the documented boundary-flip mechanism.
3. **`test_knn_and_radius_tie_multiset_clustered_duplicates`** — kind ∈
   {clustered, duplicates}, dims {2,3,8,32}, both dtypes, leaf {1,64},
   both metrics (n=1000, reduced leaf axis to keep runtime down — the
   artifact is leaf-independent). Tie-group multiset comparison ported
   from xval's `impl_knn_comparator!` semantics. **58/64 pass, 6 fail** —
   all 6 are dtype float32, dim ∈ {3,8,32}, matching the documented
   near-tie-boundary-shift mechanism.
4. **`test_knn_distance_bitexact_dim2_positive_control`** /
   **`test_knn_distance_bitexact_dim_ge_3_documented_xfail`** — direct
   distance-value bit-exactness demonstration. dim=2: **4/4 pass**
   (hard, unconditional). dim ∈ {3,8,32}: `xfail(strict=False)` — **11
   xfailed, 1 xpassed** (float64/l1/dim=3 did not trigger a mismatch in
   30000 sampled comparisons in a follow-up probe; `dim==3` is a weaker,
   data-dependent manifestation than dim ∈ {8,32}'s much higher
   mismatch rate, so `strict=False` — an occasional xpass there is
   expected and is not evidence the artifact stopped existing).

**Combined run** (`pytest python/tests/test_parity_pynanoflann.py -q`):
`16 failed, 244 passed, 11 xfailed, 1 xpassed in 3.85s` — comfortably
under the ~60s budget. Every one of the 16 real failures is individually
attributable, by dim/dtype pattern, to the documented root cause; none are
unexplained.

eps: pynanoflann's native `kneighbors`/`radius_neighbors` bindings expose
no `eps` kwarg (`help(nanoflann_ext.KDTree64.kneighbors)` shows
`(self, X, n_neighbors=None) -> ...`; `src/pynanoflann.cpp` always
constructs the eps-less `nanoflann::SearchParameters()`). Documented in the
test file header per the brief; eps=0.1 plumbing (no oracle available) is
already covered by `test_behavior.py::test_eps_relaxes_search_bound`.

## Verification commands + decisive output

**RED (before rebuild)** — see above, 12/12 `AttributeError` failures.

**`cargo check -p flannrust-py`**
```
    Checking flannrust-py v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.14s
```

**`cargo clippy -p flannrust-py --all-targets -- -D warnings`**
```
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.03s
```
Clean, no warnings.

**`.venv/bin/maturin develop --release && .venv/bin/pytest python/tests -q`**
```
16 failed, 331 passed, 11 xfailed, 1 xpassed in 3.85s
```
331 passed = all of Task 2's 75 + the 12 new behavior tests + 244 parity
passes. The 16 failures are exactly the documented, root-caused parity
mismatches above (none in `test_behavior.py`, `test_bruteforce.py`, or
`test_query_radius_box.py` — those are all green).

**`unset PYO3_PYTHON && cargo test`** (workspace default members, no
Python env) — green, `flannrust-py` not compiled:
```
$ cargo test 2>&1 | grep -iE "flannrust-py|FAILED|error\["
(no output)
```
All `flannrust`/`nanoflann-ref`/`xval` unit/doc/xval tests pass (197 + 23
+ 12 + 80 + 2 + 13 + 12 across the various binaries, 0 failed).

**`grep -rn unsafe crates/flannrust-py/src/`** — only the module-doc
comment in `convert.rs` stating "No `unsafe`"; zero actual `unsafe` usage.

**`cargo doc -p flannrust-py --no-deps`** — clean, no rustdoc warnings.

## Self-review

- API signature matches the brief verbatim, including `query_radius`'s
  `(idxs, dists)` order (opposite of `query`'s `(dists, idxs)` — checked
  against the brief text twice since it's an easy place to transpose).
- `query_box`'s dim/dtype error messages and 1-D-only enforcement mirror
  `query`'s existing validation style (`ValueError`/`TypeError` split).
- `do_query_radius`/`do_query_box` reuse the exact `for_each_variant!` +
  `py.detach` + `workers` dispatch pattern from `do_query` — no new
  concurrency primitive introduced.
- The one behavior-test bug I hit and fixed myself: the strict-boundary
  test's `nextafter(4.0, inf)` must be computed in the **tree's own
  dtype**, not always float64 — a float64-domain nextafter is
  indistinguishable from the original value once narrowed to float32
  (`T::from_f64`), which silently made the float32 case of that test
  vacuous until fixed.
- I did not invent a distance tolerance anywhere; the one relaxation in
  the parity suite (skipping a possibly-k-truncated trailing tie-group's
  index-identity check) is a test-design correction of an unprovable
  property, argued explicitly in code comments, not a numeric loosening.

## Concerns for the controller

1. **Radius-search and tied-data KNN parity against pynanoflann cannot be
   made bit-exact** at this vendored-nanoflann-version gap (1.5.5 vs
   flannrust's 1.12.1 reference), for the reasons and evidence above. This
   affects the M-py spec's success criterion #1 ("Parity: all pytest
   parity tests pass... on the full matrix above") — it does **not** pass
   in full; 16 of 272 parametrized nodes fail, all attributable to this one
   documented, code-diffed root cause, none to a flannrust defect.
2. Recommend the controller decide how `benchmarks.md`/`EXPERIMENTS.md`
   and the M-py success-criteria language should characterize this (e.g.,
   explicitly scope "bit-exact parity" to tie-free KNN index-sequence
   comparisons, which hold unconditionally, and note radius/tie parity as
   "index-set correct at ordinary scale; occasional 1-ULP boundary
   disagreement vs this specific pynanoflann build, by design/environment,
   not a flannrust defect").
3. Left `test_knn_distance_bitexact_dim_ge_3_documented_xfail` as
   `xfail(strict=False)` rather than `strict=True` because the dim=3
   sub-case is data-dependent enough that it can occasionally not trigger
   in a given sample (verified: 0/30000 for one float64/l1 configuration)
   — flagged in case the controller wants a stronger (or different)
   guarantee there.

---

## Fix round 1/5: pin the 16 documented nodes as strict xfail

**Controller ruling (accepted):** the nanoflann 1.5.5-vs-1.12.1 version gap
is an environment artifact, not a flannrust defect. Parity is re-scoped to
what holds unconditionally (tie-free knn index sequences; dim==2 distance
bit-exactness — both already hard/strict/green and untouched). The 16
failing nodes must not stay red: convert exactly the attributable
parametrized nodes to `xfail(strict=True)`.

### What changed

`crates/flannrust-py/python/tests/test_parity_pynanoflann.py` only (no
Rust changes this round):

1. **Determinism check first.** Ran the un-pinned suite twice
   (`pytest python/tests/test_parity_pynanoflann.py -q`, before any edit)
   and diffed the failing-node-ID list — identical both times (only wall
   time differed). All 16 nodes fail deterministically (fully seeded, no
   randomness), so all 16 qualify for `strict=True`; none needed the
   `strict=False` escape hatch.
2. Added `_xfail_if_known(request, known_nodes, key, reason, strict=True)`
   — a **targeted condition helper** (per the ruling: "Do NOT blanket-mark
   whole test functions"). It calls `request.node.add_marker(pytest.mark
   .xfail(...))` on the *current* parametrized node only when that node's
   `(n, dim, dtype, leaf, metric)` (or `(kind, dim, dtype, leaf, metric)`)
   tuple is in a fixed set — registered dynamically, before the test body
   runs, so a node that unexpectedly starts passing is recorded as XPASS
   (loud), not silently absorbed.
3. Two fixed sets, exactly the 16 previously-failing node IDs, no more, no
   less: `_RADIUS_XFAIL_NODES` (10 tuples, for
   `test_radius_index_parity_uniform_full_matrix`) and
   `_TIE_MULTISET_XFAIL_NODES` (6 tuples, for
   `test_knn_and_radius_tie_multiset_clustered_duplicates`). Reason string
   (both sets): `"nanoflann 1.5.5 (pynanoflann) vs 1.12.1 summation-order
   gap: 1-ULP boundary flip"`.
4. Added `request` as the first parameter to both test functions and one
   `_xfail_if_known(...)` call at the top of each, before any tree
   build/assertion.
5. `test_knn_index_parity_uniform_full_matrix` (96/96 green) and
   `test_knn_distance_bitexact_dim2_positive_control` (4/4 green) —
   **untouched**, exactly as strict as before (no `request` param, no
   marker).
6. `test_knn_distance_bitexact_dim_ge_3_documented_xfail` (the pre-existing
   `xfail(strict=False)` from the prior round, kept for the data-dependent
   dim=3/float64/l1 flakiness documented in the original report) — left
   as-is; the ruling's fix list only named the 16 newly-red nodes.
7. Module docstring: added a "Controller ruling (fix round 1/5, accepted)"
   paragraph naming the re-scoped parity claim and pointing at
   `_xfail_if_known`, plus updated the two section-header comments (§2, §3)
   to reference the new pinning instead of "left as-is"/"left unmodified".
   The pre-existing "nanoflann version gap" explanation (1.5.5-vs-1.12.1
   `evalMetric` diff, one paragraph per consequence) was already present
   from the original implementation and needed no rewrite — re-checked it
   reads standalone (a reader hits the ruling paragraph, then the full
   root-cause paragraph immediately above it, without needing the SDD
   report).

### Verification

**Run 1** (`'.venv/bin/pytest python/tests -q'`):
```
331 passed, 27 xfailed, 1 xpassed in 3.86s
```

**Run 2** (immediately after, no changes in between):
```
331 passed, 27 xfailed, 1 xpassed in 3.87s
```

Identical outcome both runs — **0 failed**, confirmed stable. `27 xfailed`
= 16 newly pinned + 11 from the prior round's
`test_knn_distance_bitexact_dim_ge_3_documented_xfail`. `1 xpassed` is the
pre-existing, already-explained `strict=False` dim=3/float64/l1 flake (not
touched this round, not a new issue).

`ast.parse` syntax check passed before running; no Rust files changed, so
`cargo check`/`clippy`/`cargo test` are unaffected from the prior round's
results (still clean/green — see the original report above).

### Self-review

- Every one of the 16 pinned tuples is copy-pasted from the prior
  `FAILED ... [param-id]` lines (cross-checked against both determinism
  runs), not retyped by hand from memory — reduces the chance of a typo
  silently leaving a real failure unpinned (which would still show up as
  `X failed` here, so it's self-checking) or over-pinning a passing node
  (which would show up as a `strict=True` XPASS failure — also
  self-checking, and didn't happen).
- No assertion logic changed; `_xfail_if_known` only registers a marker,
  it never touches the comparison code path.
- Confirmed the two currently-green tests (`test_knn_index_parity_uniform_
  full_matrix`, `test_knn_distance_bitexact_dim2_positive_control`) have no
  `request` parameter and no xfail wiring — exactly as strict as before.

**Status:** DONE (ruling addressed in full)
