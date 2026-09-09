# Task 6 report: docs + final sweep (M-py close-out)

## Status: DONE

## Files changed

- `README.md` — new "Python bindings (M-py)" section (SQUARED-distance
  warning first, install/build instructions, 10-line example, API table,
  scoped parity sentence, bench verdict table + honest misses); "Roadmap"
  section marked M-py complete; top-of-file intro line mentions the Python
  bindings.
- `docs/benchmarks.md` — new "M-py — Python bindings" section (parity
  scope, success-criteria verdict table, honest misses, Rust perf-gate
  re-run, test-status addendum); fixed one stale `crates/nanoflann-rs`
  path pointer found while sweeping (line ~411, not in the brief's
  original list, same category as the named deferred minors).
- `docs/EXPERIMENTS.md` — new "M-py" subsection in §3 (env, all commands,
  parity root-cause + ruling, perf-gate re-run, both bench runs' full
  JSON-derived tables, honest ranges, success-criteria accounting), new
  §1 git-SHA row, new §5 provenance rows; fixed the two stale
  `crates/nanoflann-rs`/`nanoflann_rs::` current-state pointers at (former)
  lines 245/346.
- `docs/ROADMAP.md` — M-py marked complete with full summary; M-pub gained
  the wheel-CI-matrix + PyPI-publish (package name `flannrust`) item and
  the dim8-f64-vs-pynanoflann investigation lead.
- `docs/nanoflann-notes.md` — fixed stale `crates/nanoflann-rs` path
  (line 202).
- `LICENSE` — `nanoflann-rs` → `flannrust` self-references (lines 3, 35).
- `crates/flannrust/src/data_source.rs` — module doc "two built-in
  implementations" → "three" (names `OwnedRows`).
- `crates/flannrust/src/tree.rs` — one new unit test,
  `dataset_returns_the_built_over_data_source`.
- `crates/flannrust-py/python/tests/test_query_radius_box.py` — one new
  pytest test, `test_query_radius_workers_parallel_matches_sequential`
  (workers=1 vs workers=-1 determinism, analogous to `test_behavior.py`'s
  existing `query` pair).
- `crates/flannrust-py/python/bench/bench_py.py` — `_pnf_jobs(workers)`
  precomputed once per worker-count outside the timed closure in
  `batched_knn_dim3_workload` (was called inside the `pynanoflann` lambda
  on every timed repeat).

## What was verified

**Parity re-confirmation** (no changes to the parity suite itself, just
re-run as part of the full pytest sweep):
`test_parity_pynanoflann.py` still shows the ruled scope holds — 96/96
tie-free knn index-sequence nodes, 4/4 dim=2 distance bit-exactness, 16
nodes pinned `xfail(strict=True)` for the pynanoflann-1.5.5-vs-flannrust's-
1.12.1-oracle version gap. No tolerance touched.

**Rust perf gates, re-run once** (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native"
cargo test -p xval --release --test perf_gate -- --ignored perf_gate
--test-threads=1 --nocapture`):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=10.848ms cpp=9.520ms ratio=1.139
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.923ms cpp=3.530ms ratio=1.111
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.985ms cpp=33.286ms ratio=0.901
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.334ms cpp=7.171ms ratio=1.023
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=179.656ms cpp=187.852ms ratio=0.956
PERF_GATE perf_gate_radius_dim3_f32: rust=5.240ms cpp=6.204ms ratio=0.845
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 4.06s
```
All six pass the 1.25 margin with wide room. Confirms M-py's T0–T5
rename/additions never touched `crates/flannrust`'s query/build/kernel
code. `build_100k` (1.139) and `radius` (0.845) each widen their
previously-recorded M2.5 range by a small margin — reported honestly as
range-widening (single additional WSL2 data point), not a regression, per
this repo's established convention (`docs/EXPERIMENTS.md`'s "M-py"
subsection has the full comparison against the M2.5 ranges).

**Two `bench_py.py` runs** (range honesty, per the brief):
- Run 1: lifted verbatim from `task-5-report.md` (gitignored; git_sha
  `509cd36`, date `2026-08-24T08:19:25Z`), pasted in full into
  `docs/EXPERIMENTS.md`.
- Run 2: fresh, run for this task (rebuilt first with
  `RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release`;
  git_sha `0cea30c`, date `2026-08-24T08:49:29Z`). All cross-checks
  passed, including the radius-specific cross-check added in T5's fix
  round. Full result-row table and all cross-check output pasted into
  `docs/EXPERIMENTS.md`.

Combined honest ranges (both runs) back every success-criteria and
honest-miss claim published in README.md / `docs/benchmarks.md` /
`docs/EXPERIMENTS.md`:
- Batched knn dim3 f32 vs cKDTree (workers 1, −1): **0.712–0.830** — MET
  (gate ≤1.00).
- Build vs pynanoflann (100k / 1M): **0.543–0.561** — MET (gate ≤1.00).
- dim-32 knn vs pynanoflann: **0.878–0.899** — MET (gate ≤1.10).
- Per-call overhead: measured, ≈2.0–2.3µs (flannrust) vs ≈7.2–7.5µs
  (cKDTree) vs ≈2.3µs (pynanoflann).
- Honest misses (not gated): `knn_batched_..._workers1` vs pynanoflann
  **1.058–1.216**; `knn_dim8_float64_..._workers1` vs pynanoflann
  **1.233–1.249** (the most consistent miss — motivates the new M-pub
  investigation lead, since the same workload measures ~0.95x against
  flannrust's own 1.12.1 C++ oracle).

## Final verification sweep (all pasted, all green)

```
$ cargo test --workspace
374 passed, 0 failed, 15 ignored   (across all binaries incl. flannrust-py's
                                     0-Rust-unit-test lib — its correctness
                                     suite is entirely pytest)

$ cd crates/flannrust-py && .venv/bin/python -m pytest python/tests -q
348 passed, 27 xfailed, 1 xpassed in 3.73s

$ cargo clippy --workspace --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.05s   (clean)

$ RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust --no-deps
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.03s   (clean)

$ cargo build -p flannrust --no-default-features
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s   (clean)

$ cargo test -p flannrust --no-default-features
194 passed, 0 failed, 3 ignored + 2 doctests

$ ruff check crates/flannrust-py/python/bench/bench_py.py
All checks passed!
```

348 = 346 at T5's close-out + 2 new dtype-parametrized cases from the new
`query_radius` workers-determinism test. 374 = 359 (M2.5's Rust-only count)
+ M-py's own `OwnedRows`/`data_source.rs` unit tests (T1) + this task's 1
new `dataset()` test, workspace now including `flannrust-py`'s 0-test
binary.

Note on the unused `import pytest` in `test_query_radius_box.py`
(`ruff` flags it): pre-existing before this task (confirmed via
`git diff --stat` — only insertions in this file this task), not
introduced by my change, left as-is (out of this task's scope).

## Self-review

- Every number written into README/benchmarks.md/EXPERIMENTS.md traces to
  either a verbatim lift from `task-3-report.md`/`task-5-report.md` (both
  gitignored — decisive output blocks re-pasted into `EXPERIMENTS.md` §3
  per the brief) or a command I ran myself this session, dated
  2026-08-24, at `git rev-parse --short HEAD` (`0cea30c` at measurement
  time, T6's own doc/test/bench-line changes staged but uncommitted and
  documented as not touching any timed code path).
- The three binding facts (pynanoflann version-gap parity scope, `.data`
  is a read-only copy not a view, Python distances/radii are SQUARED) are
  stated in README.md exactly where the brief specified: the parity
  sentence carries the scope explicitly, the API table's `.data` row says
  "copy (not a view — mutating it does not affect the tree)", and the
  SQUARED-distance warning is the very first paragraph of the Python
  section, in bold.
- Did not regenerate the scorecard artifact (controller's job post-merge)
  and did not touch `.superpowers/`, per the brief's explicit exclusions.
- Deferred minors (a)–(g) from the context are all done: (a)/(b) stale
  path renames, (c) LICENSE self-references, (d) `data_source.rs` doc
  count, (e) `dataset()` unit test, (f) `query_radius` workers-determinism
  pytest test, (g) `bench_py.py`'s `n_jobs` precompute. Found and fixed
  one more stale `crates/nanoflann-rs` current-state pointer in
  `docs/benchmarks.md` (not in the brief's original list) while sweeping
  for the named ones — same category, same fix.

## Concerns

None blocking. Two honest, non-gating misses vs pynanoflann are published
(batched-knn-workers1 and dim8-f64), consistent with the brief's explicit
instruction to report them alongside the wins; the dim8 f64 gap is now a
tracked M-pub investigation item, not silently dropped. `docs/ROADMAP.md`'s
existing bare-metal-re-run mandate for M-pub now explicitly covers the
Python bench numbers too, not just the Rust-vs-C++ gates (both are still
WSL2-only today).
