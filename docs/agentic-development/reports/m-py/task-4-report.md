# Task 4 report: `DynamicKDTree`

## Status: DONE

## Files changed

- `crates/flannrust-py/src/dynamic_tree.rs` (new) — the `DynamicKDTree`
  pyclass: a `DynTree` enum (same 10-way dtype x dim x metric
  monomorphization matrix as `static_tree::StaticTree`) plus a
  `for_each_dyn_variant!` dispatch macro, mirroring Task 2/3's pattern
  exactly. Constructor builds an initially-empty `DynamicKdTree` over an
  `OwnedRows::with_capacity(dim, capacity.unwrap_or(0))` dataset.
  `add_points`/`remove_point`/`query`/`query_radius` plus
  `dim`/`dtype`/`leaf_size`/`metric`/`capacity`/`n_active`/`n_total`/
  `removed_len` getters.
- `crates/flannrust-py/python/tests/test_dynamic.py` (new) — 15 TDD tests
  covering every checklist item in the brief.
- `crates/flannrust-py/src/lib.rs` — registered `mod dynamic_tree;` and
  `m.add_class::<DynamicKDTree>()`.

No other files modified. `crates/flannrust`, `crates/nanoflann-ref`,
`crates/xval`, and every existing test file are untouched.

## API

```
flannrust.DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)
    .add_points(points) -> (start, end)        # HALF-OPEN
    .remove_point(idx) -> bool
    .query(x, k=1, r=None, eps=0.0, workers=1)          -> (dists, idxs)   # same as static
    .query_radius(x, r, sorted=True, eps=0.0, workers=1) -> (idxs, dists)  # same as static
    .dim / .dtype / .leaf_size / .metric / .capacity
    .n_active / .n_total / .removed_len
```
Distances (and `r=`) are SQUARED for l2/l2_simple, unsquared for l1 — same
convention as the static `KDTree`. No `unsafe` added.

## Design notes

- **`add_points` half-open <-> core end-inclusive conversion.** Python
  contract: `start = tree.dataset().len()` (current total ever added,
  which this wrapper keeps in lock-step with the forest's own
  `point_count` since it is the *only* caller of `dataset_mut()`/
  `add_points` — never exposed to Python directly). For `m` new rows:
  `push_rows(&data)` then core `add_points(start, start + m - 1)`
  (end-inclusive), returning the Python-facing `(start, start + m)`
  (half-open). Verified at batch size 1 (`add_points(10, 10)` for a
  single `(d,)` point landing at index 10) via
  `test_add_in_three_batches_matches_static`'s middle batch.
- **Empty-batch short-circuit.** `m == 0` returns `(start, start)`
  immediately, before `push_rows`/`add_points` are ever called — avoids
  ever passing an inverted range (`start > end_inclusive`) to the core.
  Covered by `test_add_empty_batch_is_noop_and_returns_start_start`.
- **`capacity` is enforced in Python, not via the core's
  `maximum_point_count`.** Read `crates/flannrust/src/dynamic.rs`'s
  `maximum_point_count` doc/impl closely: it only bounds
  `tree_count = floor(log2(n)) + 1` slots, and `add_points` only panics
  when `first0bit(point_count) >= tree_count` — i.e. the *actual*
  addable capacity is `2^tree_count - 1`, which is generally **larger**
  than the user's requested `capacity` (e.g. `capacity=8` yields
  `tree_count=4`, actual core ceiling 15). So the wrapper tracks
  `start = dataset().len()` itself and raises `ValueError` whenever
  `start + m > capacity` **before** calling `push_rows`/`add_points` at
  all — the core is never touched on a rejected batch. `capacity` is
  still passed through to `.maximum_point_count(capacity)` at
  construction (so the forest itself is never under-provisioned), but
  the user-facing limit is the Python-side check. Verified by
  `test_capacity_honored_overflow_raises_before_touching_core`, which
  also asserts `n_total`/`n_active` are unchanged after the rejected
  call.
- **`n_active`/`n_total`/`removed_len`** map directly to
  `active_count()`/`dataset().len()`/`removed_len()` — `dataset().len()`
  (not a `size()`/`point_count()` the core deliberately doesn't expose,
  per that struct's own doc comment on the ambiguity) is the right
  "points ever added" number here because this wrapper's `add_points`
  always pushes exactly what it then hands to the core, keeping the two
  in sync by construction.
- **`query`/`query_radius`** are byte-for-byte the same `py.detach` +
  rayon-workers + padding (`pad_idx = n_total`, `pad_dist = inf`)
  pattern as `static_tree.rs`'s `do_query`/`do_query_radius`, just
  reading `tree.dataset().dim()`/`tree.dataset().len()` instead of the
  static tree's own `tree.dim()`/`tree.size()` (the dynamic forest has
  no such direct accessors — confirmed by grep, only `dataset()`/
  `dataset_mut()`). `knn_search_with`/`rknn_search_with`/
  `radius_search_with` have identical signatures to the static tree's,
  so the body is a straight copy.
- **`dtype` string at construction** is a separate validation path from
  the static tree's array-dtype extraction: `parse_dtype` rejects an
  unrecognized string with `ValueError` (consistent with `parse_metric`'s
  existing convention for a bad string parameter), while a genuine
  array/tree dtype *mismatch* at `add_points` time still comes from
  `as_rows_2d::<T>`'s existing `TypeError` path — no new error-mapping
  logic needed there.
- No shared trait/abstraction introduced between `static_tree.rs` and
  `dynamic_tree.rs`'s duplicated dispatch macros/query helpers — the
  brief explicitly asks for "own enum `DynTree`, same 10 variants",
  matching the established (already-duplicated) Task 2/3 pattern rather
  than inventing a premature generic-over-tree-kind abstraction for two
  call sites.

## TDD: RED evidence

Before implementing (module didn't have the class yet):

```
$ .venv/bin/pytest python/tests/test_dynamic.py -q
...
E       AttributeError: module 'flannrust' has no attribute 'DynamicKDTree'
...
FAILED python/tests/test_dynamic.py::test_construct_invalid_params_raise
FAILED python/tests/test_dynamic.py::test_basic_attributes
FAILED python/tests/test_dynamic.py::test_add_in_three_batches_matches_static[float32]
FAILED python/tests/test_dynamic.py::test_add_in_three_batches_matches_static[float64]
FAILED python/tests/test_dynamic.py::test_add_empty_batch_is_noop_and_returns_start_start[float32]
FAILED python/tests/test_dynamic.py::test_add_empty_batch_is_noop_and_returns_start_start[float64]
FAILED python/tests/test_dynamic.py::test_remove_point_excludes_and_second_remove_returns_false[float32]
FAILED python/tests/test_dynamic.py::test_remove_point_excludes_and_second_remove_returns_false[float64]
FAILED python/tests/test_dynamic.py::test_interleaved_churn_matches_static_rebuild_every_100_ops[float32]
FAILED python/tests/test_dynamic.py::test_interleaved_churn_matches_static_rebuild_every_100_ops[float64]
FAILED python/tests/test_dynamic.py::test_capacity_honored_overflow_raises_before_touching_core[float32]
FAILED python/tests/test_dynamic.py::test_capacity_honored_overflow_raises_before_touching_core[float64]
FAILED python/tests/test_dynamic.py::test_add_points_dtype_mismatch_raises_typeerror
FAILED python/tests/test_dynamic.py::test_query_on_empty_dynamic_tree_returns_padding[float32]
FAILED python/tests/test_dynamic.py::test_query_on_empty_dynamic_tree_returns_padding[float64]
15 failed in 0.06s
```

After implementing + `maturin develop --release`: all 15 GREEN (two
mid-flight compile fixes, not test-logic fixes — see below).

## Test coverage (maps to the brief's checklist)

1. `test_add_in_three_batches_matches_static` — 3 batches (10, then a
   single `(d,)` point at the off-by-one boundary, then 19 more) → knn
   (`k=5`, 5 queries) matches a fresh `flannrust.KDTree` built over the
   same (all still-live) 30 rows, index-for-index and distance-for-distance
   (`rtol=atol=1e-4`).
2. `test_remove_point_excludes_and_second_remove_returns_false` — removes
   index 0, confirms it's excluded from a `k=1` query at its own
   coordinates, confirms `remove_point(0)` a second time returns `False`,
   and separately confirms 19-live-point knn still matches a static
   rebuild over the surviving rows (index-remapped via a `live_idx` map,
   since dynamic-tree indices are original append positions while a
   compacted static-tree rebuild renumbers).
3. `test_interleaved_churn_matches_static_rebuild_every_100_ops` — 500
   seeded ops (60% add of a 1-3 row batch drawn from a 700-row pool, 40%
   remove of a random still-live index), checked against a static rebuild
   (over the live-index-mapped surviving rows) every 100 ops and once
   more at the end.
4. `test_capacity_honored_overflow_raises_before_touching_core` —
   `capacity=5`, fills exactly to 5, then a 1-point overflow add raises
   `ValueError` and leaves `n_total`/`n_active` at 5 (core untouched).
5. `test_add_points_dtype_mismatch_raises_typeerror` — `dtype="float32"`
   tree, `add_points` with a float64 array raises `TypeError`.
6. `test_query_on_empty_dynamic_tree_returns_padding` — `query`/
   `query_radius` on a zero-point tree: pad index equals `n_total` (0),
   pad distance is `inf`, `query_radius` returns an empty ragged array.

Plus two small supporting tests: `test_construct_invalid_params_raise`
(dim/leaf_size/metric/dtype/capacity=0 all `ValueError`) and
`test_basic_attributes` (getters read back what was passed in).

## Verification commands + decisive output

**RED** — see above, 15/15 `AttributeError` failures.

**`.venv/bin/maturin develop --release`** (two mid-flight compile fixes:
missing `L2: Distance<T>` bound on `build_dyn`'s `where` clause — same
bound the static tree's `build_tree` needs — and a missing
`numpy::PyArrayMethods` import for `.reshape(..)`; both are the exact same
requirements Task 2/3's `static_tree.rs` already has):
```
    Finished `release` profile [optimized] target(s) in 8.42s
🛠 Installed flannrust-0.1.0
```

**`.venv/bin/pytest python/tests/test_dynamic.py -q`**
```
15 passed
```

**`.venv/bin/pytest python/tests -q`** (full suite, must stay at the
baseline 331/27/1/0 plus the 15 new tests)
```
346 passed, 27 xfailed, 1 xpassed in 3.85s
```
346 = 331 (baseline) + 15 (new). 27 xfailed / 1 xpassed unchanged from
baseline. 0 failed.

**`cargo clippy -p flannrust-py --all-targets -- -D warnings`** (re-run
after `touch`ing `dynamic_tree.rs` to force a fresh check):
```
    Checking flannrust-py v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.09s
```
Clean, no warnings.

**`cargo test`** (workspace default members — `flannrust`,
`nanoflann-ref`, `xval`; `flannrust-py` is not a default member, confirmed
via `cargo metadata --no-deps` `workspace_default_members`, so this needs
no Python env):
```
$ cargo test 2>&1 | grep -E "FAILED|test result:"
test result: ok. 197 passed; 0 failed; 4 ignored; ...
test result: ok. 0 passed; 0 failed; 0 ignored; ...
test result: ok. 23 passed; 0 failed; 0 ignored; ...
test result: ok. 12 passed; 0 failed; 0 ignored; ...
test result: ok. 80 passed; 0 failed; 0 ignored; ...
test result: ok. 2 passed; 0 failed; 0 ignored; ...
test result: ok. 0 passed; 0 failed; 1 ignored; ...
test result: ok. 3 passed; 0 failed; 6 ignored; ...
test result: ok. 13 passed; 0 failed; 0 ignored; ...
test result: ok. 5 passed; 0 failed; 0 ignored; ...
test result: ok. 11 passed; 0 failed; 2 ignored; ...
test result: ok. 13 passed; 0 failed; 1 ignored; ...
test result: ok. 12 passed; 0 failed; 1 ignored; ...
test result: ok. 2 passed; 0 failed; 0 ignored; ...
test result: ok. 0 passed; 0 failed; 0 ignored; ...
test result: ok. 0 passed; 0 failed; 0 ignored; ...
```
Every suite `0 failed`.

**`grep -rn unsafe crates/flannrust-py/src/dynamic_tree.rs`** — no output
(zero `unsafe` usage).

**`git status --short`** — confirms only the two new files
(`dynamic_tree.rs`, `test_dynamic.py`) plus the 3-line `lib.rs`
registration diff:
```
 M crates/flannrust-py/src/lib.rs
?? crates/flannrust-py/python/tests/test_dynamic.py
?? crates/flannrust-py/src/dynamic_tree.rs
```

## Self-review

- Re-read `DynamicKdTree::add_points`'s full doc comment (contiguous-
  append contract, per-index panic on misalignment) before writing
  `do_add_points` — confirmed the wrapper's invariant (it is the *only*
  caller of `dataset_mut()`, always immediately followed by `add_points`
  covering exactly what was just pushed) can never trigger that panic.
- Re-read `maximum_point_count`'s doc comment closely rather than
  assuming it was an exact cap — the `tree_count = floor(log2(n))+1`
  slot-count relationship means the core's actual ceiling is looser than
  the user's requested number, which is exactly why the brief calls for
  a Python-side pre-check rather than relying on the core to enforce it.
- Checked (via `grep -n "fn dim\|fn size"`) that `DynamicKdTree` has
  neither — used `dataset().dim()`/`dataset().len()` instead of guessing
  an accessor existed.
- `query`/`query_radius`/padding/workers logic is a direct copy of
  `static_tree.rs`'s already-reviewed (Task 2/3) implementation with only
  the `tree.dim()/.size()` → `tree.dataset().dim()/.len()` substitution —
  minimized the surface for a new bug versus writing it from scratch.
- Confirmed capacity=0 raises before even calling `parse_metric`/
  `parse_dtype` order doesn't matter functionally (both produce
  `ValueError`), but checked construction order doesn't skip the dim/
  leaf_size checks either (all four validated before any tree is built).
- No abstraction shared between `static_tree.rs` and `dynamic_tree.rs`
  beyond the existing `convert.rs` helpers — matches the brief's "own
  enum `DynTree`" instruction and the established duplicated-per-file
  dispatch-macro convention; did not introduce a generic trait to
  de-duplicate the two `do_query` bodies (two call sites, not a pattern
  worth abstracting yet).

## Concerns

None. All brief checklist items implemented and verified; full pytest
suite and default-member `cargo test` both green; clippy clean; no
`unsafe`; only the 3 required/permitted files touched.
