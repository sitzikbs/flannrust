# flannrust M-py — Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish-quality Python bindings (`import flannrust`) for the Rust kd-tree — static + dynamic, cKDTree-like API, bit-exact parity vs pynanoflann, benchmarked against scipy cKDTree and pynanoflann with suite-generated provenance.

**Architecture:** Rename the core crate to `flannrust`; add two small core pieces (`OwnedRows` DataSource, `DynamicKdTree::dataset_mut`); new PyO3 cdylib crate `flannrust-py` with an enum-dispatched monomorphization matrix; pytest parity/behavior suites; a Python bench script feeding the existing report renderer.

**Tech Stack:** Rust stable, PyO3 0.29.2 + numpy 0.29.0, maturin (via `uv`), pytest, numpy/scipy/pynanoflann (probe venv verified working on this host).

**Spec:** `docs/agentic-development/specs/2026-08-23-flannrust-m-py-design.md` — the plan argues from the spec; read it first. Roadmap context: `docs/ROADMAP.md` §M-py.

## Global Constraints

- Default-build results stay bit-exact vs the C++ oracle: full xval suite green after every task (the rename must be behaviorally invisible).
- No `unsafe` in `flannrust-py` (spec decision: copy-in ownership). Core `unsafe` count stays at the two existing `FrameStack` blocks.
- Squared L2 everywhere in the Python API; radius args squared; strict `<`; inclusive box; insertion-order ties; `k > n` padded `idx = n`, `dist = inf`.
- `cargo test`/`cargo build` (default members) must never require a Python environment; `flannrust-py` is excluded from `default-members`.
- Perf claims: median×7, interleaved, environment recorded; every number in docs traces to a pasted run (EXPERIMENTS.md provenance rules).
- clippy `-D warnings` clean, zero rustdoc warnings, `--no-default-features` builds, at every commit.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01T1ELzxC42AesQeKGCT7Kt1`
- Toolchain facts for this host: `cargo` needs `export PATH="$HOME/.cargo/bin:$PATH"`; `uv` is at `~/.local/bin/uv`; Python 3.12.3; probe venv showed `maturin 1.14.1`, `pynanoflann` installs and imports fine; WSL2 noise floor documented in `docs/EXPERIMENTS.md` §1.

---

### Task 0: Rename `nanoflann-rs` → `flannrust`

**Files:**
- Rename: `crates/nanoflann-rs/` → `crates/flannrust/` (`git mv`)
- Modify: `Cargo.toml` (workspace members + add `default-members`), `crates/flannrust/Cargo.toml` (`name = "flannrust"`), every `use nanoflann_rs::`/`nanoflann_rs::` in `crates/xval/**` and `crates/flannrust/**` (docs/examples included), `README.md` (title/usage), `docs/ROADMAP.md` (name decision recorded)
- Do NOT rewrite: verbatim pasted run outputs in `docs/EXPERIMENTS.md`/`docs/benchmarks.md`/`docs/agentic-development/reports/` that mention the old crate name — they are historical evidence. Add one note near the top of `docs/EXPERIMENTS.md` §1: "The library crate was renamed `nanoflann-rs` → `flannrust` on 2026-08-23 (M-py T0); pasted outputs earlier than that show the old name/paths."

**Interfaces:**
- Produces: crate `flannrust` importable as `use flannrust::{KdTreeBuilder, DynamicKdTreeBuilder, ...}`; workspace `default-members = ["crates/flannrust", "crates/nanoflann-ref", "crates/xval"]`.

- [ ] `git mv crates/nanoflann-rs crates/flannrust`; edit workspace `Cargo.toml` members + `default-members`; set package name `flannrust` in the crate's Cargo.toml; `xval`'s dependency entry becomes `flannrust = { path = "../flannrust", ... }`.
- [ ] Mechanical replace `nanoflann_rs` → `flannrust` in all `.rs` files; replace current-state (non-pasted-output) `nanoflann-rs`/`crates/nanoflann-rs` mentions in README.md and docs; add the EXPERIMENTS.md rename note. `grep -rn "nanoflann_rs" crates/` must be empty; `grep -rn "nanoflann-rs" README.md docs/*.md` hits only pasted-output blocks and the rename note.
- [ ] Verify: `cargo test --workspace` green (359 tests), `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build -p flannrust --no-default-features`, `cargo doc -p flannrust` zero warnings. Attribution intact: README + LICENSE still credit nanoflann/Blanco-Claraco (grep and cite).
- [ ] Commit: `refactor: rename crate nanoflann-rs -> flannrust (published name, user decision)`.

### Task 1: Core additions — `OwnedRows` + dataset accessors

**Files:**
- Modify: `crates/flannrust/src/data_source.rs` (add `OwnedRows`), `crates/flannrust/src/tree.rs` (`KdTree::dataset`), `crates/flannrust/src/dynamic.rs` (`DynamicKdTree::{dataset, dataset_mut}`), `crates/flannrust/src/lib.rs` (re-export `OwnedRows`)
- Test: inline `#[cfg(test)]` in `data_source.rs` and `dynamic.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct OwnedRows<T> { /* data: Vec<T>, dim: usize */ }
  impl<T: Scalar> OwnedRows<T> {
      pub fn new(data: Vec<T>, dim: usize) -> Self;        // asserts dim > 0, len % dim == 0
      pub fn with_capacity(dim: usize, n_points: usize) -> Self;
      pub fn push_rows(&mut self, rows: &[T]);             // asserts rows.len() % dim == 0
      pub fn dim(&self) -> usize;
      pub fn len(&self) -> usize;                          // point count
      pub fn is_empty(&self) -> bool;
      pub fn as_slice(&self) -> &[T];
  }
  // DataSource for OwnedRows<T> AND for &OwnedRows<T>; point_row overridden (Some(&data[idx*dim..idx*dim+dim]))
  impl KdTree { pub fn dataset(&self) -> &DS; }
  impl DynamicKdTree { pub fn dataset(&self) -> &DS; pub fn dataset_mut(&mut self) -> &mut DS; }
  ```
- `dataset_mut` doc: append-only between `add_points` calls (contiguous-append contract); never shrink below indices already added.

- [ ] TDD: failing unit tests first — `OwnedRows` count/component/`point_row` contract (length ≥ dim, values == `point_component`, boundary idx, `None` never returned for valid idx), `push_rows` assertion on ragged input, `with_capacity` starts empty; dynamic round-trip test: build `DynamicKdTree` over `OwnedRows::with_capacity`, `dataset_mut().push_rows(...)` then `add_points(start, end)` in 3 batches + one `remove_point`, knn results equal a fresh static `KdTree` over the same live rows (index sets + distances).
- [ ] Implement; run; also run the metric row-vs-fallback bit-equality tests with an `OwnedRows` source at dims {3, 8, 32} f32/f64 (extend the existing test's source list — the M2.5 kernel must engage).
- [ ] Verify hygiene trio (test/clippy/doc) + `--no-default-features`.
- [ ] Commit: `feat: OwnedRows row-major DataSource with point_row fast path; dataset accessors`.

### Task 2: `flannrust-py` scaffold + static `KDTree` with `query`

**Files:**
- Create: `crates/flannrust-py/Cargo.toml`, `crates/flannrust-py/pyproject.toml`, `crates/flannrust-py/src/lib.rs`, `crates/flannrust-py/src/convert.rs`, `crates/flannrust-py/src/static_tree.rs`, `crates/flannrust-py/python/tests/test_behavior.py`, `crates/flannrust-py/python/tests/test_bruteforce.py`, `crates/flannrust-py/python/tests/conftest.py`
- Modify: workspace `Cargo.toml` (member, NOT default-member)

**Interfaces:**
- Consumes: `flannrust::{KdTreeBuilder, OwnedRows, ConstDim, DynDim, L1, L2, L2Simple, SearchParams, BuildThreads}`.
- Produces (Python): `flannrust.KDTree(points, leaf_size=10, metric="l2", threads=None)`; `.query(x, k=1, r=None, eps=0.0, workers=1) -> (dists, idxs)`; attributes `n, dim, dtype, leaf_size, metric, data`.
- Produces (Rust, used by Tasks 3–4): `convert.rs` helpers `fn as_rows_2d<'py, T>(...) -> PyResult<(Vec<T>, usize, usize)>` (accepts `(d,)`/`(m,d)`, contiguity via `PyReadonlyArray` + copy) and the dispatch macro `for_each_variant!` over the enum:
  ```rust
  enum StaticTree {
      F32D2(KdTree<f32, ConstDim<2>, OwnedRows<f32>, L2>), F32D3(...), F64D2(...), F64D3(...),
      F32Dyn(KdTree<f32, DynDim, OwnedRows<f32>, L2>), F64Dyn(...),
      F32DynL1(...), F64DynL1(...), F32DynL2s(...), F64DynL2s(...),
  }
  ```

- [ ] Scaffold: Cargo.toml (`crate-type = ["cdylib"]`, `pyo3 = { version = "0.29.2", features = ["extension-module", "abi3-py39"] }`, `numpy = "0.29.0"`, `rayon`, `flannrust = { path = "../flannrust" }`); pyproject.toml (`[build-system] requires = ["maturin>=1.14"]`, module name `flannrust`); `uv venv crates/flannrust-py/.venv && uv pip install maturin numpy scipy pytest pynanoflann`; `maturin develop --release` from the crate dir; `python -c "import flannrust"` works with a stub `#[pymodule]`.
- [ ] TDD (pytest RED first, committed with the implementation as usual): behavior tests — dtype/shape errors (`TypeError` on int array, `ValueError` on 3-D), `(d,)` vs `(m,d)` output shapes, padding `k > n` (`idx == n`, `dist == inf`), empty tree, `metric` validation, `threads` values {None, 1, 4} give identical results, `workers=-1` bit-identical to `workers=1` on 1k queries, `r=` turns `query` into rknn (crafted 3-in-radius vs 10-in-radius cases from the M1 test recipe), `eps=0.5` distances ≤ (1+eps)·true on a crafted set, `data` attribute round-trips and is read-only.
- [ ] Brute-force tests: seeded numpy datasets (uniform, clustered, 30% duplicates; dims {2,3,8,32}; f32/f64), `query` indices equal numpy `argpartition` GT on tie-free data; distances within 4 ULP (`np.nextafter` bound); all metrics.
- [ ] Implement: `convert.rs` (copy into `OwnedRows`), enum + macro, build with GIL released (`Python::detach` — PyO3 0.29 name; verify against the pyo3 docs for this exact version and use what compiles: it is the successor of `allow_threads`), `query` writing straight into output `ndarray` rows, rayon `par_chunks_mut` when `workers != 1` (cap at `workers` via a scoped pool when `> 1`, ambient pool for `-1`).
- [ ] Verify: `maturin develop --release && pytest python/tests -x -q` green; `cargo clippy -p flannrust-py --all-targets -- -D warnings`; plain `cargo test` (default members) still green WITHOUT the Python env (prove by running it with `PYO3_PYTHON` unset).
- [ ] Commit: `feat: flannrust-py static KDTree with cKDTree-like query`.

### Task 3: `query_radius` + `query_box` + pynanoflann parity suite

**Files:**
- Modify: `crates/flannrust-py/src/static_tree.rs`
- Create: `crates/flannrust-py/python/tests/test_parity_pynanoflann.py`

**Interfaces:**
- Produces: `.query_radius(x, r, sorted=True, eps=0.0, workers=1) -> (list[ndarray idxs], list[ndarray dists])`; `.query_box(lo, hi) -> ndarray[uint32]` (traversal order, inclusive).

- [ ] TDD behavior additions: strict `<` at an exact boundary point (`dist² == r` excluded, `np.nextafter(r, inf)` includes), `sorted=False` returns traversal order, box face-point inclusion, box on empty tree.
- [ ] Parity suite per spec §Testing 1: matrix (uniform/clustered/duplicates × n {1k, 20k} × dims {2,3,8,32} × f32/f64 × leaf {1,10,64} × metric {l2, l1}), 200 seeded queries incl. on-dataset points; knn k ∈ {1, 10}: index sequences equal AND `np.sqrt(ours)` bit-equals pynanoflann for l2 (raw equality for l1), tie groups as multisets; radius: ours(`r`) vs pynanoflann `radius_neighbors(X, radius=np.sqrt(r))` — sorted sequences equal under the same sqrt mapping; eps ∈ {0, 0.1} both sides (pynanoflann has no eps kwarg — check its native signature; if eps is not exposed, run eps=0 only and DOCUMENT that in the test header). Keep runtime < ~60 s by trimming query counts on the 20k cases.
- [ ] Escalation rule (verbatim in the test file header): any parity mismatch is a STOP — identify pynanoflann's vendored nanoflann version (sdist inspection) before touching tolerances; report findings to the controller.
- [ ] Verify + commit: `feat: query_radius/query_box + bit-exact parity suite vs pynanoflann`.

### Task 4: `DynamicKDTree`

**Files:**
- Create: `crates/flannrust-py/src/dynamic_tree.rs`, `crates/flannrust-py/python/tests/test_dynamic.py`
- Modify: `crates/flannrust-py/src/lib.rs` (register class)

**Interfaces:**
- Consumes: `flannrust::{DynamicKdTreeBuilder, DynamicKdTree}`, Task 1's `dataset_mut`, Task 2's convert/dispatch pattern (own enum `DynTree`, same 10 variants).
- Produces (Python): `flannrust.DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)`; `.add_points(points) -> (start, end)` half-open; `.remove_point(i) -> bool`; `.query/.query_radius` as static; `n_active`, `n_total` attributes.

- [ ] TDD: add in 3 batches (incl. a `(d,)` single point) → knn equals fresh static tree over live rows; remove → excluded from results, second remove returns `False`; interleaved add/remove churn (seeded, 500 ops) vs static-rebuild reference after every 100 ops; capacity honored (`capacity=n` then overflow add raises `ValueError` before touching the core); dtype fixed at construction (`TypeError` on mismatched `add_points`); queries on empty dynamic tree.
- [ ] Implement (`dataset_mut().push_rows` then core `add_points(start, end_inclusive)`; note the core takes END-INCLUSIVE — convert carefully and test the off-by-one at batch size 1).
- [ ] Verify + commit: `feat: flannrust-py DynamicKDTree (add_points/remove_point)`.

### Task 5: Python benchmarks + report integration

**Files:**
- Create: `crates/flannrust-py/python/bench/bench_py.py`
- Modify: `crates/xval/examples/render_report.rs` (optional second arg: python JSON)

**Interfaces:**
- Produces: `bench_py.py` writes JSON `{meta: {python, numpy, scipy, pynanoflann, cpu, date, git_sha}, workloads: [{name, flannrust_ms, ckdtree_ms, pynanoflann_ms, ratio_ckdtree, ratio_pynanoflann}]}`; `render_report report.json report_py.json` appends a "Python bindings" section.

- [ ] Implement bench per spec table: build 100k/1M dim-3 f32 (threads 1 and None; cKDTree `leafsize=10, balanced_tree=False, compact_nodes=False` — documented as the closest-configuration choice; pynanoflann `.fit`); batched knn k=10, 200k queries dim-3 f32 at workers {1, -1} (cKDTree `workers`, pynanoflann `n_jobs`); knn dim-8 f64 + dim-32 f32 (100k pts, 20k queries, workers=1); radius selectivity ~10 and ~1000 dim-3 f32; single-query loop 1k calls k=10 (per-call µs, all three). Median×7 per cell, interleaved A/B/C within each repeat, seeds fixed, distances cross-checked between libraries on a 10-query sample before timing (sqrt mapping) so a benchmark never compares wrong answers.
- [ ] `render_report` python section: same visual language as existing sections; ratios < 1.0 = flannrust faster; single-query row labeled "overhead-bound (see EXPERIMENTS)".
- [ ] Run the full bench, save JSON + stdout; regenerate `report_data`/`render_report` chain end-to-end.
- [ ] Verify + commit: `feat: Python bench vs cKDTree/pynanoflann + report section`.

### Task 6: Docs + final sweep

**Files:**
- Modify: `README.md` (Python section: install via `pip install flannrust` (future) / `maturin develop` (now), 10-line example, SQUARED-distance warning first, API table, bench summary), `docs/EXPERIMENTS.md` (Python env, exact commands, provenance rows for every new number), `docs/benchmarks.md` (M-py section, honest ranges), `docs/ROADMAP.md` (M-py done; M-pub gains wheel-CI + PyPI-publish items)

- [ ] Re-run the Rust perf gates once (`PERF_GATE=1 ...`) to confirm the rename/additions didn't move anything; paste.
- [ ] Re-run `bench_py.py` a second time (range honesty); write docs from BOTH runs' pasted outputs; regenerate the scorecard artifact.
- [ ] Verify: workspace tests, clippy, rustdoc, `--no-default-features`, pytest — all green, all pasted.
- [ ] Commit: `docs: M-py results — Python bindings benchmarks and provenance`.

---

## Self-review notes (done at write time)

- Spec coverage: name/rename → T0; OwnedRows/dataset_mut → T1; KDTree/query/errors/workers/GIL → T2; radius/box/parity → T3; dynamic → T4; bench/report → T5; docs/success-criteria evidence → T6. Success criterion 2 (speed targets) is judged in T5/T6 with the miss-reporting rule from the spec.
- pynanoflann eps: its Python `kneighbors` exposes no eps — T3 carries the check-and-document instruction rather than assuming.
- Types: `StaticTree`/`DynTree` enums are task-local (T2/T4) but share `convert.rs`; `OwnedRows` signatures repeated where consumed.
