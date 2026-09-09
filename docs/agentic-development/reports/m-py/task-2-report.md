# Task 2 report: `flannrust-py` scaffold + static `KDTree` with `query`

Branch: `m-py`. Commit: see bottom.

## Summary

Implemented the `flannrust-py` PyO3 0.29.2 / numpy 0.29.0 cdylib crate: a
static `KDTree` Python class wrapping a 10-way `{f32,f64} x {ConstDim<2>,
ConstDim<3>,DynDim} x {L2,L1,L2Simple}` monomorphization matrix (dispatched
via a `StaticTree` enum + `for_each_variant!` macro), with `.query(x, k,
r, eps, workers) -> (dists, idxs)`, attributes `n/dim/dtype/leaf_size/
metric/data`, GIL released during search via `Python::detach`, and rayon
`par_chunks_mut` parallelism for `workers != 1`. No `unsafe` anywhere in
the crate. All 75 pytest tests pass; plain `cargo test` (default members)
stays green without a Python environment.

## Files

- Created: `crates/flannrust-py/Cargo.toml`, `crates/flannrust-py/pyproject.toml`,
  `crates/flannrust-py/src/lib.rs`, `crates/flannrust-py/src/convert.rs`,
  `crates/flannrust-py/src/static_tree.rs`,
  `crates/flannrust-py/python/tests/{conftest.py,test_behavior.py,test_bruteforce.py}`
- Modified: workspace `Cargo.toml` (added `crates/flannrust-py` to `members`,
  NOT `default-members`), `.gitignore` (`.venv/`, `*.so`, `__pycache__/`, `*.pyc`
  — `/target` was already covered)

## RED evidence

Before implementing `KDTree`'s `#[new]`/methods (stub `#[pyclass] struct
KDTree;` with no constructor), the full pytest suite (already written)
against the built stub extension:

```
$ .venv/bin/pytest python/tests -x -q
.F
=================================== FAILURES ===================================
__________________ test_construct_3d_points_raises_valueerror __________________

    def test_construct_3d_points_raises_valueerror():
        pts = np.zeros((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
>           flannrust.KDTree(pts)
E           TypeError: cannot create 'builtins.KDTree' instances

python/tests/test_behavior.py:27: TypeError
=========================== short test summary info ============================
FAILED python/tests/test_behavior.py::test_construct_3d_points_raises_valueerror
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed, 1 passed in 0.02s
```

(1 passed was `test_construct_int_dtype_raises_typeerror` — coincidentally
also a `TypeError`, so it happened to pass against the stub; every other
test that reaches `flannrust.KDTree(...)` or `.query(...)` would fail the
same way. This is standard "no constructor yet" RED, not a designed
per-test RED matrix.)

## Design notes

### Toolchain confirmation (before writing tests)
- `cargo check -p flannrust-py` resolved `pyo3 0.29.2` / `numpy 0.29.0`
  cleanly against the stub.
- `uv venv .venv && uv pip install --python .venv/bin/python maturin numpy
  scipy pytest pynanoflann` → `maturin==1.14.1`, `numpy==2.5.2`,
  `pynanoflann==0.10.0`, `pytest==9.1.1`, `scipy==1.18.1`.
- `maturin develop --release` on the stub built and installed `flannrust`
  in ~4s; `python -c "import flannrust"`-equivalent confirmed via pytest
  collection succeeding (no import error).
- Checked the vendored `pyo3-0.29.2`/`numpy-0.29.0` sources directly
  (`~/.cargo/registry/src/...`) rather than guessing API names:
  - `Python::detach(self, f: F) -> T where F: Ungil + FnOnce() -> T` exists;
    `allow_threads` does **not** exist in this version (zero grep hits) —
    confirms the brief's naming note.
  - `Bound<'py, PyAny>::downcast::<T>()` does not exist in this pyo3
    version; the method is `.cast::<T>()`. Discovered via a real compile
    error, not guessed.
  - `numpy`'s `PyReadwriteArray::make_nonwriteable(self) -> PyReadonlyArray`
    (in `borrow/mod.rs`) is a **safe** API (its one `unsafe` block lives
    inside the `numpy` crate, not mine) that flips `NPY_ARRAY_WRITEABLE`
    off — used for the read-only `data` attribute instead of the design
    spec's zero-copy `PyArray::borrow_from_array` idea, which would need
    `unsafe` to tie a raw pointer's lifetime to the tree object. See
    "Deviations" below.

### `convert.rs`
- `to_ndarray(py, x)` — thin wrapper around Python `numpy.asarray(x)`.
  Normalizes ndarrays (no-op, no copy) *and* nested lists/tuples in one
  line, matching the design spec's "any array-like ... `np.ascontiguousarray`
  semantics" without hand-rolling list-to-array conversion.
- `as_rows_2d<'py, T: Element + Copy>(x) -> PyResult<(Vec<T>, usize, usize)>`
  — exact signature from the brief. Accepts already-numpy `x`; 1-D `(d,)`
  → `(data, 1, d)`, 2-D `(m,d)` → `(data, m, d)`; any other rank →
  `ValueError`. Dtype mismatch (checked via `PyReadonlyArray<T,_>`
  extraction failing) → `TypeError`. Row-major copy is done by iterating
  `ArrayView::outer_iter()` / `.iter()` rather than trusting the input's
  memory layout, so non-C-contiguous input (Fortran order, strided views)
  is still copied correctly with zero `unsafe`.
- `ndim(x) -> PyResult<usize>` — small additional helper (not in the
  brief's explicit interface list, but needed by every caller that must
  decide `(k,)`-vs-`(m,k)` output shaping *before* extracting typed data,
  and reusable as-is by Tasks 3/4's `query_radius`/`add_points`). Cheap
  (`PyUntypedArray::ndim()`, no dtype extraction).

### `static_tree.rs` — `StaticTree` enum / `for_each_variant!` macro
Exactly the 10-variant enum from the brief:
```rust
enum StaticTree {
    F32D2(KdTree<f32, ConstDim<2>, OwnedRows<f32>, L2>), F32D3(...), F64D2(...), F64D3(...),
    F32Dyn(KdTree<f32, DynDim, OwnedRows<f32>, L2>), F64Dyn(...),
    F32DynL1(...), F64DynL1(...), F32DynL2s(...), F64DynL2s(...),
}
```
`for_each_variant!($self, $t => $body)` expands to a 10-arm `match` binding
`$t: &KdTree<T,D,OwnedRows<T>,M>` (concrete `T`/`D`/`M` per arm) to `$body`.
Because a `match` requires one common result type, `$body` is always a call
into a `T`/`D`/`M`-generic free function (`data_array`, `do_query`) rather
than inline arm-specific code — that's what makes 10 arms type-check as one
expression. `KDTree::new` builds the right variant via two small
`build_f32`/`build_f64` matches on `(metric, dim)` calling a shared
`build_tree<T,D,M>(...)` helper (one true `KdTreeBuilder` call site, not
duplicated 10 times).

Dispatch rule (matches the design spec's table): `metric == "l2" && dim in
{2,3}` → `ConstDim` fast-path variant; `metric == "l2"` at any other dim,
or `metric` ∈ {`l1`,`l2_simple`} at any dim → `DynDim` variant. `L1`/
`L2Simple` are never paired with `ConstDim` (matches the enum's 4+6 split).

### `query`
- Validates `k >= 1` and `workers ∈ {-1} ∪ [1, ∞)` before touching the
  core (`ValueError` otherwise).
- `x` normalized via `to_ndarray`, `is_1d` captured via `ndim()`, then
  `as_rows_2d::<T>(&x)` extracts data (dtype mismatch → `TypeError`, dim
  mismatch vs `tree.dim()` → `ValueError`).
- Output buffers pre-filled with the padding sentinel (`idx = tree.size()`
  as `u32`, `dist = T::from_f64(f64::INFINITY)`) *before* calling into the
  core — since `knn_search_with`/`rknn_search_with` only write the found
  prefix and leave the tail untouched (per the brief), pre-filling makes
  padding "free" (no separate post-pass).
- `r = None` → `knn_search_with`; `r = Some(radius)` → `rknn_search_with`
  with `T::from_f64(radius)` — this is the "`r=` turns `query` into rknn"
  requirement from the brief, verified against the exact M1 test recipe
  (10 colinear points, squared distances 1..100, radius² 10.0 → 3 found,
  radius² 1000.0 → full coverage capped at `k`).
- Whole search loop (sequential or parallel) runs inside
  `py.detach(|| { ... })`. `workers == 1`: plain sequential `for` over
  `.chunks()/.chunks_mut()`. `workers > 1`: a scoped
  `rayon::ThreadPoolBuilder::new().num_threads(workers).build()...install(...)`
  capping the pool at exactly `workers` threads. `workers == -1`: the same
  `par_chunks_mut`/`par_chunks`/`zip`/`for_each` combinator called directly
  (uses rayon's ambient/global pool — all cores). All three paths share one
  `search_one` closure (`Fn`, capturing `&tree`/`r`/`&params`) so the
  knn-vs-rknn branch isn't duplicated three times.
- `KdTree: Send + Sync` (confirmed by grep of `crates/flannrust/src/scalar.rs`:
  `Scalar`/`DistanceValue` both require `Send + Sync`, and `Distance<T>:
  Send + Sync`) makes sharing `&tree` across rayon workers sound with zero
  `unsafe`.

### `data` attribute
Builds a fresh numpy `(n, dim)` array from `tree.dataset().as_slice()`
(`PyArray1::from_vec` → `.reshape((n,dim))`), then
`.into_readwrite().make_nonwriteable()` to flip the `WRITEABLE` numpy flag
off before returning it — genuinely read-only from Python's own
perspective (`arr[0,0] = 1.0` raises numpy's own `ValueError: assignment
destination is read-only`), all with zero `unsafe`.

## Deviation from the design spec (documented, not hidden)

The M-py design spec's `data` attribute description says "zero-copy OUT,
via `PyArray::borrow_from_array` equivalent ... tied to the tree object's
lifetime." I implemented it as a **fresh copy** per access instead (still
read-only, still correct, still O(n) — same order as construction's own
copy-in). Reason: making a numpy array's backing memory alias a Rust
struct's owned `Vec` for its full Python-visible lifetime is exactly the
aliasing/lifetime hazard `unsafe` exists to manage, and Task 2's global
constraint list is explicit: "NO `unsafe` anywhere in flannrust-py." I
took the constraint as binding over the spec's aspirational zero-copy
detail. Functionally indistinguishable to callers (round-trips, read-only,
correct values) — only the constant-factor cost of `.data` access changes
(one extra copy per access instead of zero), and `.data` is not on any hot
per-query path.

## Verification (commands + decisive output)

**1. `maturin develop --release && pytest -x -q` — green (post-implementation):**
```
$ .venv/bin/maturin develop --release
...
🛠 Installed flannrust-0.1.0
$ .venv/bin/pytest python/tests -x -q
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 0.11s
```
Also ran without `-x` (no early stop) to rule out masking: `75 passed in 0.09s`.

**2. `cargo clippy -p flannrust-py --all-targets -- -D warnings` — clean:**
```
    Checking flannrust v0.1.0 (...)
    Checking flannrust-py v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.21s
```
(zero warnings/errors)

**3. `cargo clippy --workspace --all-targets -- -D warnings` — clean** (confirms
`--workspace` explicitly reaches the non-default member too, with the rest
of the workspace unaffected):
```
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 10.80s
```

**4. `cargo doc -p flannrust-py --no-deps` — zero warnings:**
```
   Generated /home/sitzikbs/dev/flannrust/target/doc/flannrust/index.html
```

**5. Plain `cargo test` (default members) green WITHOUT the Python env**
(`PYO3_PYTHON` explicitly unset; `Compiling`/`Running` lines show only
`flannrust`, `nanoflann-ref`, `xval` — `flannrust-py` never touched):
```
$ unset PYO3_PYTHON
$ cargo test 2>&1 | grep -E "Compiling|Running|test result|error"
     Running unittests src/lib.rs (target/debug/deps/flannrust-...)
test result: ok. 197 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out
     Running unittests src/lib.rs (target/debug/deps/nanoflann_ref-...)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
     Running tests/oracle.rs ...
test result: ok. 23 passed; 0 failed; 0 ignored
     Running tests/oracle_dynamic.rs ...
test result: ok. 12 passed; 0 failed; 0 ignored
     Running unittests src/lib.rs (target/debug/deps/xval-...)
test result: ok. 80 passed; 0 failed; 0 ignored
     Running tests/bench_sanity.rs ...              ok. 2 passed
     Running tests/native_parity.rs ...              ok. 0 passed; 1 ignored
     Running tests/perf_gate.rs ...                  ok. 3 passed; 6 ignored
     Running tests/render_report_test.rs ...         ok. 13 passed
     Running tests/xval_build.rs ...                 ok. 5 passed
     Running tests/xval_dynamic.rs ...                ok. 11 passed; 2 ignored
     Running tests/xval_knn.rs ...                    ok. 13 passed; 1 ignored
     Running tests/xval_radius_box.rs ...             ok. 12 passed; 1 ignored
   Doc-tests flannrust: ok. 2 passed
```
No `flannrust-py` compile step appears anywhere — confirms `default-members`
exclusion works and the Python bindings crate imposes zero cost on the
plain Rust dev loop.

**6. Manual smoke test** (`import flannrust`, build, query, shapes/dtypes):
```
$ .venv/bin/python -c "..."
n,dim,dtype,leaf_size,metric = 1000 3 float32 10 l2
shapes (3, 4) (3, 4) float32 uint32
[0.        5.1636267 5.520803  5.868802 ] [  0 155 579 568]
1d shapes (4,) (4,)
import OK
```
`idxs` dtype is `uint32` (matches the core's default `u32` `IndexType`).

**7. `.gitignore` / staging hygiene** — confirmed via `git add -n
crates/flannrust-py/ .gitignore Cargo.toml Cargo.lock`: only source files
staged (`Cargo.toml`, `pyproject.toml`, `src/*.rs`, `python/tests/*.py`);
`.venv/`, `target/`, `__pycache__/` all excluded (added `.venv/`, `*.so`,
`__pycache__/`, `*.pyc` to `.gitignore`; `/target` was already present).

## Test coverage summary (75 tests)

- `test_behavior.py` (23 tests, some multiplied by the `dtype` fixture):
  construct dtype/shape errors (int dtype → `TypeError`, 3-D/1-D points →
  `ValueError`), basic attributes, metric validation (invalid → `ValueError`,
  case-insensitive accepted), `leaf_size=0` → `ValueError`, empty-points
  build, query `(d,)`-vs-`(m,d)` shapes, `k > n` padding (`idx==n`,
  `dist==inf`), empty-tree query, `k=0` → `ValueError`, query dtype/dim
  mismatch → `TypeError`/`ValueError`, batched query == per-row query,
  `threads ∈ {None,1,4}` identical results, `workers=-1`/`workers=3`
  bit-identical to `workers=1` on batches up to 1000 queries, the M1
  3-vs-full radius recipe via `r=`, `eps=0.5` slack bound
  (`exact ≤ approx ≤ (1+eps)*exact`), `data` round-trip + read-only.
- `test_bruteforce.py` (52 tests): `{2,3,8,32} dims × {f32,f64} × {l2,l1,
  l2_simple}` uniform (tie-free) index-sequence-exact-match vs
  `argsort`/4-ULP distance check, plus `{clustered,duplicates} ×
  {f32,f64} × {l2,l1,l2_simple}` distance-multiset + self-consistency
  check (ties in these kinds make a single "correct" index sequence
  non-unique, so index-order equality isn't asserted there — only
  correctness of the returned distances/indices as a set).

## Self-review

- Every core API used (`OwnedRows::new`, `KdTreeBuilder::{new,with_metric,
  leaf_max_size,threads,build}`, `knn_search_with`/`rknn_search_with`,
  `SearchParams{eps,sorted}`, `KdTree::{dataset,dim,size}`) was read
  directly from `crates/flannrust/src/{tree,params,data_source,scalar}.rs`
  before use — no guessing signatures.
- Confirmed the brief's PyO3-API naming note by grepping the actual
  vendored crate source rather than trusting memory; two mismatches found
  and fixed (`Python::detach` not `allow_threads` — brief predicted this
  correctly; `.cast()` not `.downcast()` — brief didn't call this one out,
  found via compiler error).
- No `unsafe` anywhere in the crate (`grep -rn unsafe crates/flannrust-py/src`
  → no hits, confirmed while writing this report).
- `cargo test` (default members) confirmed green with `PYO3_PYTHON`
  explicitly unset and grepped to prove `flannrust-py` was never compiled.
- Workspace member list changed on exactly the one line the brief allowed
  (`members`, not `default-members`); `crates/flannrust`, `crates/nanoflann-ref`,
  `crates/xval` untouched.

## Concerns

1. **`data` attribute is a copy, not the spec's zero-copy view** — see
   "Deviation" above. I judge the explicit no-`unsafe` constraint to
   dominate the spec's zero-copy aspiration for this task; flagging for
   controller sign-off since it's a real (if minor, and previously
   spec'd-as-optional-upgrade — "Zero-copy is a later upgrade only if the
   bench shows the copy matters") deviation from the written design.
2. **`ndim()` in `convert.rs` is an addition beyond the brief's explicit
   "Produces" list** (which names only `as_rows_2d` and the dispatch
   macro). It's small, has an obvious single purpose, and Tasks 3/4 will
   need identical `(d,)`-vs-`(m,d)` shape logic for `query_radius`/
   `add_points` — flagging so the controller can confirm it's an
   acceptable/expected extension of convert.rs's surface rather than scope
   creep.
3. Per-metric `for_each_variant!` dispatch means every `query`/`data` call
   monomorphizes through 10 arms; this is inherent to the design's
   compile-time dispatch matrix (not something I introduced) and was
   already accepted in the spec/plan.

## Commit

`feat: flannrust-py static KDTree with cKDTree-like query`

---

## Fix round 1: release GIL during `KDTree::new`'s build (Important finding)

**Finding (from review):** `KDTree::new` extracted the array data and then
called `build_f32`/`build_f64` → `KdTreeBuilder::build()` directly, without
`py.detach`. `do_query` released the GIL for the search path, but the
*build* path (which, with `threads=None` → `BuildThreads::Auto`, can be a
large parallel rayon build) held the GIL the whole time — blocking other
Python threads for the build's full duration. The brief requires GIL
release during both build and batched query.

**Fix** (`crates/flannrust-py/src/static_tree.rs`, `KDTree::new`): kept
`to_ndarray`/`ndim`/`as_rows_2d` (all Python-object-touching calls) exactly
where they were, entirely *before* the change — `as_rows_2d` already
copies the array's data out into a plain `Vec<f32>`/`Vec<f64>`, so by the
time `build_f32`/`build_f64` are called, nothing in scope touches Python
objects. Wrapped just the two build call sites in `py.detach`:

```rust
let tree = py.detach(|| build_f32(metric_norm, d, data, leaf_size, build_threads));
...
let tree = py.detach(|| build_f64(metric_norm, d, data, leaf_size, build_threads));
```

`data: Vec<T>` (`Send`), `metric_norm: &'static str` (`Copy`), `d`/
`leaf_size: usize` (`Copy`), `build_threads: BuildThreads` (`Copy`) — all
move into the closure cleanly; `build_f32`/`build_f64` return `StaticTree`,
whose only fields are `KdTree<T,D,OwnedRows<T>,M>` variants, all `Send`
(per `Scalar`/`Distance`'s `Send + Sync` bounds) — so `StaticTree: Send`
and the `py.detach`/`Ungil` bounds are satisfied with zero manual trait
work. Compiled clean on the first try (`cargo check -p flannrust-py`).

### Verification (post-fix)

**`cargo check -p flannrust-py`** — clean:
```
    Checking flannrust-py v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.10s
```

**`maturin develop --release && pytest python/tests -q`** — still green,
same 75/75:
```
$ .venv/bin/maturin develop --release
...
🛠 Installed flannrust-0.1.0
$ .venv/bin/pytest python/tests -q
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 0.09s
```

**`cargo clippy -p flannrust-py --all-targets -- -D warnings`** — clean:
```
    Checking flannrust-py v0.1.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.11s
```

**Plain `cargo test` (default members), `PYO3_PYTHON` unset** — still
green, `flannrust-py` still never compiled:
```
$ unset PYO3_PYTHON
$ cargo test 2>&1 | grep -E "Compiling flannrust-py|test result: FAILED|error\["
no flannrust-py compile, no failures/errors
```

No new pytest coverage was added for this fix specifically — GIL-release
correctness isn't observable from pure Python (no test can assert "the GIL
was released" without a second thread racing the build, which is out of
scope for this fix) — the existing `test_threads_variants_identical_results`
already exercises `threads ∈ {None,1,4}` (i.e. the `Auto`/`Sequential`/
`Threads(4)` build paths this fix wraps) for correctness; this fix changes
*when* the GIL is held, not what the build computes, so no new pytest is
needed to cover it (the fixed code path was already exercised by every
existing `KDTree(...)` construction in the whole suite).

**Commit:** `fix: release GIL during KDTree build` (hash below).
