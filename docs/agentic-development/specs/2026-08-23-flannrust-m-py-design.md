# flannrust — Milestone M-py: Python bindings (design)

Status: approved design (user, 2026-08-23). Implementation plan follows in
`docs/agentic-development/plans/`.

## Goal

Make the Rust kd-tree usable from Python under the name `flannrust`, with an
API scipy/pynanoflann users recognize, so it can replace `pynanoflann` and
compete with `scipy.spatial.cKDTree` — and measure that claim with the same
suite-generated, provenance-tracked rigor as M1/M2/M2.5.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| Published name | `flannrust` everywhere: Rust crate, PyPI package, `import flannrust`. The core crate `nanoflann-rs` is renamed to `flannrust` (T0). nanoflann attribution (BSD-2, Blanco-Claraco et al.) unchanged in README/LICENSE. |
| Python API shape | cKDTree-like: `KDTree(points, ...)`, `.query`, `.query_radius`, `.query_box`; batched `(m, d)` queries, GIL released, `workers` parallelism. |
| Dynamic tree | Included: `DynamicKDTree` with `add_points` / `remove_point` and the same query surface. |
| Input ownership | Copy into an owned row-major `Vec<T>` at construction (one memcpy, measured and documented; no `unsafe` in the bindings, no aliasing hazard). Zero-copy is a later upgrade only if the bench shows the copy matters. |
| Distances | Squared L2 (nanoflann semantics), radius arguments squared too. Documented loudly; differs from cKDTree (euclidean) and from pynanoflann's Python layer (which sqrt's). |
| Benchmark comparators | `scipy.spatial.cKDTree` and `pynanoflann`. |
| Parity oracle | `pynanoflann` (pybind11 over C++ nanoflann) for indices + `sqrt`-mapped distances, plus numpy brute force. The Rust core is already bit-exact vs nanoflann 1.12.1 (xval), so the bindings' risk is marshalling, not arithmetic. |

## Architecture

```
crates/flannrust/        # core (renamed from nanoflann-rs); gains OwnedRows + DynamicKdTree::dataset_mut
crates/flannrust-py/     # PyO3 0.29 + numpy 0.29 cdylib, maturin; native module `flannrust`, no pure-Python layer
  Cargo.toml, pyproject.toml, src/lib.rs, src/static_tree.rs, src/dynamic_tree.rs, src/convert.rs
  python/tests/          # pytest parity + behavior
  python/bench/bench_py.py   # emits JSON consumed by xval's render_report
crates/nanoflann-ref/    # unchanged
crates/xval/             # render_report grows `--python <json>`; GrowableFlat stays (test-only)
```

The bindings crate is a workspace member but excluded from `default-members`
(workspace `Cargo.toml`), so plain `cargo test`/`cargo build` never need a
Python dev environment; `cargo test --workspace` and `maturin` build it
explicitly (documented in EXPERIMENTS.md).

### Core additions (crates/flannrust)

1. `data_source::OwnedRows<T>` — `{ data: Vec<T>, dim: usize }`, row-major,
   `DataSource<T>` with `point_row` overridden (so the M2.5 chunked kernel
   is always hit), `point_count = data.len() / dim`. Constructors:
   `OwnedRows::new(data, dim)` (asserts `dim > 0`, `data.len() % dim == 0`),
   `OwnedRows::with_capacity(dim, n)`; `push_rows(&mut self, rows: &[T])`
   (asserts multiple of `dim`), `dim()`, `len()`, `as_slice()`.
   Implemented for `OwnedRows<T>` by value AND for `&OwnedRows<T>`.
2. `DynamicKdTree::dataset(&self) -> &DS` and `dataset_mut(&mut self) -> &mut DS`.
   `&mut self` guarantees no concurrent search; the caller must only APPEND
   (the existing contiguous-append contract of `add_points`) — documented on
   `dataset_mut`. `KdTree::dataset(&self) -> &DS` for symmetry.
3. Nothing else in the core changes. Parity suites are the judge of the rename.

### Bindings (crates/flannrust-py)

Monomorphization matrix, dispatched by an enum behind a macro:

| dtype | dim | metric |
|---|---|---|
| f32, f64 | `ConstDim<2>`, `ConstDim<3>` | L2 only |
| f32, f64 | `DynDim` | L2, L1, L2Simple |

= 4 fixed-dim + 6 dyn-dim = 10 static variants; same 10 for dynamic. SO2/SO3
not exposed in M-py.

`KDTree(points, leaf_size=10, metric="l2", threads=None)`
- `points`: any array-like coercible to 2-D `float32`/`float64`; converted
  with `np.ascontiguousarray` semantics on the Rust side (copy always
  happens — into `OwnedRows`). `dtype` is taken from the input; integer or
  other dtypes raise `TypeError`. `points.shape == (n, d)`, `d >= 1`; `n == 0`
  builds an empty tree (all queries return empty / padded).
- `leaf_size` ≥ 1; `metric` ∈ {"l2", "l1", "l2_simple"} (case-insensitive);
  `threads`: `None` → `BuildThreads::Auto` (rayon ambient pool), `1` →
  `Sequential`, `n > 1` → `Threads(n)`. Build runs with the GIL released.
- Attributes: `n`, `dim`, `dtype`, `leaf_size`, `metric`, `data` (a read-only
  NumPy view of the owned copy — zero-copy OUT, via `PyArray::borrow_from_array`
  equivalent in numpy 0.29, tied to the tree object's lifetime).

`query(x, k=1, r=None, eps=0.0, workers=1) -> (dists, idxs)`
- `x`: `(d,)` → outputs `(k,)`; `(m, d)` → outputs `(m, k)`. dtype must match
  the tree's (`TypeError` otherwise); `k >= 1`.
- Padding when fewer than `k` found (`n < k`): `idx = n`, `dist = inf`
  (cKDTree convention). `idxs` dtype `uint32` (the tree's `IndexType`);
  `dists` dtype = tree dtype (squared L2 / L1 sum).
- `workers`: `1` → sequential; `-1` → all cores; `n > 1` → capped at `n`.
  Implemented as rayon `par_chunks_mut` over output rows inside
  `py.allow_threads` (name per PyO3 0.29: `Python::detach`). `workers != 1`
  on `m == 1` is allowed and degenerates to sequential.
- Per-query scratch (`out_i`/`out_d` slices) is the output row itself — no
  per-query allocation.

`query_radius(x, r, sorted=True, eps=0.0, workers=1) -> (idxs, dists)`
- `r` is the SQUARED radius for L2; strict `<` as in nanoflann.
- Returns two lists (length `m`, or length 1 for a `(d,)` query) of 1-D
  arrays (ragged); `sorted=True` sorts by distance (nanoflann `sort`).
- `workers` semantics as `query`; each worker owns a `Vec<ResultItem>` and
  results are assembled into NumPy arrays under the GIL afterwards.

`rknn` is exposed as `query(x, k, r=None)`: when `r` is given, it is an
RKNN search (`k` nearest within squared radius `r`, padded like `query`).

`query_box(lo, hi) -> idxs` — inclusive box, single box only (no batching;
box search takes no params in nanoflann), traversal order, `uint32`.

`DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)`
- Empty at construction; `capacity` maps to `maximum_point_count` (None →
  builder default).
- `add_points(points)`: `(m, d)` (or `(d,)`) appended to the owned buffer,
  then `add_points(start, end_inclusive)` on the core. Contiguity is by
  construction. Returns the index range `(start, end)` (Python half-open).
- `remove_point(i) -> bool`; `n_active`, `n_total` (`removed_len` exposed too).
- Same `query` / `query_radius` surface (no `query_box` — nanoflann's dynamic
  forest has none; documented).
- Not thread-safe for mutation; `query` with `workers` is allowed (reads).

Errors: Python `ValueError`/`TypeError` for shape/dtype/argument problems;
Rust panics never cross the FFI (PyO3 converts panics to `PanicException`,
but every documented precondition is checked before calling into the core).

### Distances, ties, eps — parity contract

Identical to the Rust core (and therefore to nanoflann 1.12.1): squared L2,
strict radius `<`, inclusive box, insertion-order ties, `eps` multiplicative
on the node bound, k > n padded. Documented in the module docstring and in
README's Python section with the "squared, not euclidean" warning first.

## Testing

`crates/flannrust-py/python/tests/` (pytest, run via `uv run pytest` in the
crate's venv; `maturin develop --release` first):

1. **Parity vs pynanoflann** (`test_parity_pynanoflann.py`): seeded datasets
   (uniform / clustered / 30% duplicates; n ∈ {1k, 20k}; dims {2, 3, 8, 32};
   f32 + f64; leaf {1, 10, 64}), 200 queries incl. on-dataset points.
   knn (k ∈ {1, 10, 101 > n?}): index sequences equal; `np.sqrt(ours)`
   bit-equals pynanoflann's `kneighbors` distances for L2, raw equality for
   L1. Radius: `pynanoflann.radius_neighbors(X, radius=sqrt(r))` set
   equality + sorted sequence equality. Ties: compare tie groups as
   multisets (reuse xval's rule, reimplemented in numpy). eps ∈ {0, 0.1}
   both sides. If pynanoflann's vendored nanoflann differs from 1.12.1 in a
   way that shows up, the test FAILS and the implementer escalates — the
   report must say which nanoflann version pynanoflann embeds (`pip show`,
   source inspection) before any tolerance is loosened.
2. **Brute force** (`test_bruteforce.py`): numpy `argpartition` ground
   truth; index equality on tie-free data; distances within 4 ULP (numpy
   summation order differs).
3. **Behavior** (`test_behavior.py`): dtype/shape errors; `(d,)` vs `(m,d)`
   outputs; padding for `k > n`; empty tree; `workers=-1` bit-identical to
   `workers=1`; `data` attribute is a read-only view; `query_box` boundary
   inclusivity; strict radius on an exact-boundary point; `DynamicKDTree`
   add/remove sequence equals a fresh static `KDTree` over the live set
   (index sets), and `remove_point` twice returns `False`.
4. **Rust side**: unit tests for `OwnedRows` (contract of `point_row`,
   `push_rows` assertions) and `dataset_mut` (append-then-add_points round
   trip vs a static tree); the full xval suite unchanged after the rename.

## Benchmarks and report

`python/bench/bench_py.py` (seeded with the same generator parameters as
xval's `uniform`; datasets regenerated in numpy, seed documented):

| Workload | flannrust | cKDTree | pynanoflann |
|---|---|---|---|
| build 100k / 1M dim-3 f32, sequential and parallel | `KDTree(..., threads=1/None)` | `cKDTree(pts, leafsize=10, balanced_tree=False, compact_nodes=False)` (closest to a plain median kd-tree; documented) | `KDTree(leaf_size=10).fit` |
| knn k=10, 200k queries, dim-3 f32, workers 1 and -1 | `query(Q, 10, workers=w)` | `query(Q, 10, workers=w)` | `kneighbors(Q, 10, n_jobs=w)` |
| knn k=10 dim-8 f64, dim-32 f32 (100k) | same | same | same |
| radius (selectivity ~10 and ~1000), dim-3 f32 | `query_radius` | `query_ball_point` | `radius_neighbors` |
| single-query loop (1k Python calls, dim-3 f32, k=10) — the per-call overhead workload | `query(q, 10)` | same | same |

Method: median of 7 (perf-gate methodology), `RUSTFLAGS="-C target-cpu=native"`
for the flannrust build, `workers=-1` = `os.cpu_count()`, same process,
interleaved order, environment printed (numpy/scipy/pynanoflann versions,
Python, CPU). Output: `report_py.json`; `render_report --python report_py.json`
adds a "Python bindings" section (table + ratios flannrust/cKDTree and
flannrust/pynanoflann, lower is better). EXPERIMENTS.md gets the Python
environment, exact commands, and provenance rows; benchmarks.md gets an
M-py section with honest ranges; the scorecard artifact is regenerated.

The single-query workload is expected to be overhead-bound (PyO3 call +
array creation ≈ µs vs a ~100 ns search); it is measured to state that
honestly, not to win.

## Success criteria

1. Parity: all pytest parity tests pass (bit-exact index sequences; L2
   distances bit-exact after sqrt mapping) on the full matrix above.
2. Speed (ratio = flannrust time / competitor time, median×7, documented noise
   floor applies): batched knn dim-3 f32 ≤ 1.00 vs cKDTree at `workers=1`
   AND `workers=-1`; build ≤ 1.00 vs pynanoflann (sequential) and parallel
   build < 1.00 vs both. dim-32 f32 knn ≤ 1.10 vs pynanoflann. Any miss is
   reported with numbers, not hidden.
3. Per-call overhead of the single-query path measured and documented (µs
   per call, all three libraries).
4. `maturin build --release` produces an installable wheel; `uv run pytest`
   green from a clean venv; `cargo test --workspace` green with the Python dev env present
   (CI gating is M-pub).
5. Hygiene: clippy clean (`-D warnings`), zero rustdoc warnings, README
   Python section with install + 10-line example + the squared-distance
   warning, ROADMAP updated, EXPERIMENTS/benchmarks provenance complete.

## Out of scope (M-py)

Wheel CI matrix / PyPI publish (M-pub); SO2/SO3 and custom Python metrics;
pickling / index save-load; zero-copy input; `query_box` batching;
sklearn-style estimator interface.
