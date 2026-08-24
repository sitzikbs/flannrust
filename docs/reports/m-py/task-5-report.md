# Task 5 report: Python benchmarks + report integration

## Summary

Added `crates/flannrust-py/python/bench/bench_py.py` (new) and extended
`crates/xval/examples/render_report.rs` with an optional second CLI
argument (Python bench JSON) that appends a "Python bindings" section to
the rendered scorecard. Full bench run completed end-to-end; all
cross-checks passed; all three spec'd success criteria met. Renderer
verified both ways (1-arg output byte-identical to pre-change; 2-arg
output renders the new section correctly).

## Files touched

- `crates/flannrust-py/python/bench/bench_py.py` (new, 371 lines)
- `crates/xval/examples/render_report.rs` (modified: optional argv[2], all
  new logic added, existing argv[1]-only path untouched)

No other files were touched (library code and docs were out of scope for
this task, per the brief).

## Rebuild confirmation (target-cpu=native)

Rebuilt the flannrust wheel before any timing, per the brief's requirement:

```
export PATH="$HOME/.cargo/bin:$PATH"
cd crates/flannrust-py
RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release
```
Output: `Finished \`release\` profile [optimized] target(s) in 13.14s` /
`Installed flannrust-0.1.0`. `meta.rustflags` in the emitted JSON records
`"-C target-cpu=native"` (captured from the actual bench-run process
environment, exported before invoking the script), and the rendered HTML's
machine line shows `&middot; target-cpu=native` for the Python section,
confirming the flag was live both at build time and (best-effort signal)
at bench-run time.

## Bench methodology, as run

- `time.perf_counter()`, median of 7 repeats per cell (`RUNS = 7`).
- Within each repeat, every engine in a given cell is timed once,
  interleaved in order (A, B, C, A, B, C, ...) rather than clustered --
  `timed_median_interleaved()`.
- `gc.disable()` for the entire timed block of each cell, restored after.
- All datasets seeded via `numpy.random.default_rng` with CRC32-derived
  integer seeds (`_seed()`, mirrors `test_parity_pynanoflann.py`'s own
  `_seed` helper) -- deterministic across runs/processes.
- `make_points` is imported directly from
  `crates/flannrust-py/python/tests/conftest.py` (via a `sys.path` insert)
  rather than reimplemented, reusing the same seeded-dataset generator the
  pytest parity suite already uses.
- Cross-check before timing: for every distinct (n, dim, dtype) dataset
  used (7 of them: build_100k, build_1M, knn_batched dim3, knn dim8 f64,
  knn dim32 f32, radius dim3, single_query_loop dim3), a 10-query tie-free
  ("uniform") sample is run through both `flannrust.KDTree.query` and
  `pynanoflann.KDTree.kneighbors` and asserted: (a) identical indices, (b)
  sqrt-mapped distances agree within 1e-6 relative tolerance. A failure
  aborts via `SystemExit` with a clear message -- never silently times a
  wrong-answer engine. **All 7 cross-checks passed** (see evidence below).
- Distance/radius unit mapping: cKDTree (`query`, `query_ball_point`) and
  pynanoflann (`kneighbors`, `radius_neighbors`) both take/return euclidean
  distances; flannrust's l2 metric is squared. Every radius argument fed to
  cKDTree/pynanoflann is `math.sqrt(r_sq)`; every cross-check distance
  comparison sqrt-maps flannrust's output before comparing.
- `n_jobs` correctness fix (found during smoke-testing, before the timed
  run): pynanoflann's `n_jobs` is a literal C++ thread count -- unlike
  scipy's `workers` and flannrust's `threads`/`workers`, it does **not**
  special-case `-1` as "all cores" (`kneighbors_multithreaded` raises a
  pybind11 `TypeError` on a negative arg, confirmed empirically). Added
  `_pnf_jobs(workers)` to map `-1` -> `os.cpu_count()` for pynanoflann calls
  only, so the `workers=-1` batched-knn cell is a genuine full-core
  comparison on all three engines, not a silent crash or a 1-thread
  pynanoflann run mislabeled as "-1".
- Build-thread caveat, labeled honestly: neither cKDTree nor pynanoflann
  offers a parallel-build option. The `*_threadsNone_parallel_build` rows
  compare flannrust `threads=None` against the *same* single-thread
  cKDTree/pynanoflann numbers already captured in the sibling `*_threads1`
  row, with an explicit `note` field on the JSON row (also surfaced as the
  rendered `<tr title="...">` tooltip) stating this is not a same-config
  parity claim.
- cKDTree build config: `leafsize=10, balanced_tree=False,
  compact_nodes=False` (closest-configuration choice per the brief).
- Workload sizing decisions (not fully pinned by the brief, documented
  here): batched knn uses n=1,000,000 points / 200,000 queries dim3 f32
  (large enough that `workers=-1` shows real thread-scaling, ~7x on this
  8-thread host); dim8/dim32 knn and radius/single-query use n=100,000
  points, matching `report_data.rs`'s own dim3/dim8 workload scale.

## Cross-check evidence (all 7 passed)

```
cross-check OK (build_100k_dim3_f32 dataset): 10/10 index match, max rel dist err 1.09e-07
cross-check OK (build_1M_dim3_f32 dataset): 10/10 index match, max rel dist err 1.18e-07
cross-check OK (knn_batched_dim3_f32 dataset): 10/10 index match, max rel dist err 1.07e-07
cross-check OK (knn_dim8_float64 dataset): 10/10 index match, max rel dist err 2.05e-16
cross-check OK (knn_dim32_float32 dataset): 10/10 index match, max rel dist err 1.40e-07
cross-check OK (radius_dim3_f32 dataset): 10/10 index match, max rel dist err 1.13e-07
cross-check OK (single_query_loop dataset): 10/10 index match, max rel dist err 1.18e-07
```
All well within the 1e-6 relative-tolerance bound (dim8 float64's 2e-16 is
essentially double-precision epsilon, as expected -- no summation-order
artifact in float64).

## Full bench run: command + meta

```
cd crates/flannrust-py
export RUSTFLAGS="-C target-cpu=native"
.venv/bin/python python/bench/bench_py.py > /tmp/report_py.json
```
Ran once end-to-end (progress on stderr, JSON on stdout), ~7 minutes wall
clock (dominated by the dim32 knn cell -- curse-of-dimensionality makes
kd-tree pruning far less effective at dim 32, so all three engines spend
tens of seconds there; see the row below).

```json
"meta": {
  "python": "3.12.11",
  "numpy": "2.5.2",
  "scipy": "1.18.1",
  "pynanoflann": "0.10.0",
  "cpu": "AMD Ryzen 7 9800X3D 8-Core Processor",
  "threads": 8,
  "date": "2026-08-24T08:19:25Z",
  "git_sha": "509cd36",
  "rustflags": "-C target-cpu=native",
  "wheel_profile": "release (.venv/bin/maturin develop --release); RUSTFLAGS=\"-C target-cpu=native\" REQUIRED at build time..."
}
```

## Full result rows (all 11 workloads, pasted verbatim from the run)

| workload | flannrust_ms | ckdtree_ms | pynanoflann_ms | ratio_ckdtree | ratio_pynanoflann |
|---|---|---|---|---|---|
| build_100k_dim3_f32_threads1 | 9.576 | 11.669 | 17.552 | **0.821** | **0.546** |
| build_100k_dim3_f32_threadsNone_parallel_build (labeled, apples-to-oranges) | 3.285 | 11.669 (reused) | 17.552 (reused) | 0.282 | 0.187 |
| build_1M_dim3_f32_threads1 | 130.388 | 132.147 | 239.969 | **0.987** | **0.543** |
| build_1M_dim3_f32_threadsNone_parallel_build (labeled, apples-to-oranges) | 32.092 | 132.147 (reused) | 239.969 (reused) | 0.243 | 0.134 |
| knn_batched_dim3_f32_k10_q200k_workers1 | 306.391 | 430.344 | 289.668 | **0.712** | 1.058 |
| knn_batched_dim3_f32_k10_q200k_workersNeg1 | 45.095 | 63.362 | 45.075 | **0.712** | 1.000 |
| knn_dim8_float64_k10_workers1 | 392.347 | 510.625 | 318.081 | 0.768 | 1.233 |
| knn_dim32_float32_k10_workers1 | 22023.360 | 43101.397 | 25082.533 | 0.511 | **0.878** |
| radius_dim3_f32_sel10 | 2.606 | 4.082 | 4.415 | 0.638 | 0.590 |
| radius_dim3_f32_sel1000 | 66.102 | 129.133 | 230.621 | 0.512 | 0.287 |
| single_query_loop_dim3_f32_k10_percall_ms (overhead-bound, per-call) | 0.001968 | 0.007245 | 0.002273 | 0.272 | 0.866 |

(Bold ratios are the three spec'd success-criteria cells.)

## Success criteria vs spec (honest accounting)

1. **Batched knn dim3 f32 <= 1.00 vs cKDTree at workers 1 and -1**: MET.
   workers=1: ratio_ckdtree = **0.712**. workers=-1: ratio_ckdtree =
   **0.712** (identical after both engines scale ~7x with 8 threads on
   this 8-core host -- 306ms/45ms flannrust ≈ 6.8x, 430ms/63ms cKDTree ≈
   6.8x, so the ratio is flat across worker counts here, not a coincidence
   of rounding).
2. **Build <= 1.00 vs pynanoflann**: MET. build_100k threads1:
   ratio_pynanoflann = **0.546**. build_1M threads1: ratio_pynanoflann =
   **0.543**. (The threadsNone/parallel-build rows are even lower, 0.187
   and 0.134, but those compare an apples-to-oranges config as documented
   above -- not claimed as the primary evidence for this criterion.)
3. **dim-32 <= 1.10 vs pynanoflann**: MET. knn_dim32_float32_k10_workers1:
   ratio_pynanoflann = **0.878**.

All three stated gates pass, with margin.

**Honest non-gating observations** (not spec'd success criteria, reported
for completeness): flannrust is *slower* than pynanoflann on two rows --
`knn_batched_dim3_f32_k10_q200k_workers1` (ratio_pynanoflann 1.058, ~6%
slower) and `knn_dim8_float64_k10_workers1` (ratio_pynanoflann 1.233, ~23%
slower); at workers=-1 the batched-dim3-vs-pynanoflann ratio is 1.000
(essential parity). flannrust is faster than cKDTree on every single row
without exception. These aren't failures against anything the brief gates
on, but are reported here rather than only citing the favorable rows.

## Renderer verification

**RED** (pre-change, current `main` binary): running the *unmodified*
`render_report` binary with a 2nd argument (a synthetic python JSON
fixture) produced **zero** occurrences of "Python bindings" in the output
-- the 2nd arg was silently ignored by the old `std::env::args().nth(1)`
call. Captured via `git stash` on `render_report.rs` alone, rebuild, run,
`git stash pop`, rebuild again.

**GREEN** (post-change):
- 1-arg invocation (`render_report report.json`) produces HTML **byte-identical**
  (`diff` clean) to the pre-change binary's 1-arg output -- confirms the
  existing behavior is untouched.
- 2-arg invocation with the real `report_py.json` from the full bench run:
  `cargo run -p xval --release --example render_report -- report.json report_py.json > report.html`
  exits 0; output contains exactly one `<section class="python-bindings">`,
  all 11 workload names, the `overhead-bound (see EXPERIMENTS)` badge
  exactly once (only on `single_query_loop_*`), correct `pill good`/`pill
  fail` classes matching each ratio's `< 1.0`/`>= 1.0` (spot-checked:
  `knn_batched...workers1` vs cKDTree gets `pill good` at 0.712x, vs
  pynanoflann gets `pill fail` at 1.058x), and zero leftover
  `{placeholder}` template residue.
- Full `report_data` -> `render_report` chain regenerated end-to-end using
  a recent `report.json` already present in this session's scratchpad
  (from an earlier run this session, git_sha `d64cfe0`, dated
  2026-08-23T23:24:18Z -- reused per the task instructions' "you may reuse
  an existing recent report.json" allowance) plus the fresh
  `report_py.json` from this task's own bench run. Rendered HTML: 556
  lines, both sections present, no template residue.

This is the "TDD adapted" evidence for the renderer change (no new Rust
test file was added -- out of the allowed-files list for this task, which
restricts changes to `crates/flannrust-py/python/bench/**` and
`crates/xval/examples/render_report.rs` only). The byte-identical-diff
check is a strong RED/GREEN pair: it proves the change is additive-only
for the existing single-arg path, not just "looks right."

## Verification commands + output (final gate run)

```
$ cargo clippy -p xval --all-targets -- -D warnings
    Checking xval v0.0.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.11s

$ cargo test          # default-members: flannrust, nanoflann-ref, xval
... (all crates) test result: ok. 0 failed ... across every suite,
including tests/render_report_test.rs: 13 passed; 0 failed.

$ cd crates/flannrust-py && .venv/bin/python -m pytest python/tests -q
346 passed, 27 xfailed, 1 xpassed in 3.72s   # matches stated 346-pass baseline

$ ruff check crates/flannrust-py/python/bench/bench_py.py
All checks passed!
```

## Self-review

- Reused `conftest.py`'s `make_points` rather than duplicating the seeded
  dataset generator (ponytail rung 2: already in the codebase).
- Reused `xval::report`'s CSS classes (`section`/`h2`/`legend`/`pill`/
  `table`/`scroll-x`/`num`) for the new HTML fragment -- no stylesheet
  changes, matches "same visual language as existing sections" verbatim.
- Found and fixed a real correctness bug during smoke-testing (pynanoflann
  `n_jobs=-1` crashing, not "all cores" like scipy) before it could corrupt
  a timed cell -- caught by testing the actual API surface at small scale
  first rather than trusting the brief's framing that `n_jobs` behaves
  like `workers`.
- Every honest-labeling requirement from the brief/context is implemented
  as data (a `note` field on the JSON row, surfaced as a `title` tooltip in
  the HTML), not just prose in this report: the build-threadsNone
  apples-to-oranges caveat, and the single-query overhead-bound caveat.
- Did not add a dependency, a new test framework, or an abstraction layer
  for a single call site (e.g., no generic "engine adapter" class -- three
  plain functions calling three plain APIs, per ponytail).

## Concerns

- None blocking. The two non-gating "flannrust slower than pynanoflann"
  rows (dim3 batched-workers1 at 1.058x, dim8 f64 at 1.233x) are reported
  above for honesty; they are not spec'd success criteria and are
  plausible given pynanoflann's simpler C++ binding path has less
  per-call Python/numpy marshalling overhead in some shapes, while
  flannrust's threaded rayon dispatch has fixed overhead that shows more
  at workers=1 on already-cheap dim3 queries. A future task could profile
  this if it becomes a stated gate.
- The full bench run takes several minutes (dominated by the dim32 cell,
  ~22-43s per single query call across 3 engines x 7 reps); T6 (docs) or
  any re-run should budget for that, as noted in the bench script's own
  docstring.
- `bench_py.py` intentionally does not commit its own JSON output (per
  instructions) -- all evidence above is pasted directly from the actual
  run rather than referencing an artifact file.

---

## Fix round 1/5 (code review response)

Two Important findings addressed.

### Finding 1: radius workload's pre-timing cross-check never exercised the radius codepath

`cross_check()` only compares `query`/`kneighbors` (knn). `query_radius` vs
`query_ball_point` vs `radius_neighbors` -- different unit mapping (squared
vs euclidean), different boundary-inclusion convention, different
selection-count logic -- were entirely unguarded before timing.

**Fix**: added `radius_cross_check(ftree, ctree, pnf, dim, dtype, r_sq,
seed, label)` in `bench_py.py` (right after `cross_check`). It builds a
fresh 10-query tie-free ("uniform") sample at an independent seed, runs all
three engines' radius call at the exact `r_sq`/`sqrt(r_sq)` a timed cell is
about to use, and compares **index sets** (not order -- both
`query_ball_point` and `radius_neighbors` return unsorted results) across
all three. Any row mismatch raises `SystemExit` with the three sets shown,
aborting the whole bench -- same pattern as the existing knn `cross_check`.
Wired into `radius_workload()`: called once per selectivity level
(`sel10`/`sel1000`), right after `r_sq` is calibrated and before that
cell's `timed_median_interleaved`.

**Why an independent seed, not the same 10-query sample `cross_check` uses,
and not reusing `q[0]`**: `radius_workload`'s `r_sq` is calibrated as
`q[0]`'s own k-th-nearest-neighbor distance, i.e. it lands EXACTLY on a
real point's distance from `q[0]`. At that exact boundary, flannrust's
strict `<` convention and cKDTree/pynanoflann's inclusive convention can
legitimately disagree about a single point -- that's a real cross-engine
convention difference, not a bug, and asserting equality there would be a
flaky false-positive test failure baked into the fix itself. A **fresh**
seeded sample (different seed, uniform/continuous data) has ~0 probability
of any query row's true neighbor distance exactly equaling that same
calibrated `r_sq` value, so the cross-check exercises the real
index-set-agreement property without inheriting that one specific
boundary-tie hazard.

**Proof it actually catches a real mismatch** (not just "runs without
crashing"): built real flannrust/cKDTree/pynanoflann trees on n=20,000
dim3 f32 points and ran `radius_cross_check` directly via a driver script
(not through the full bench):

```
[bench_py] radius cross-check OK (driver_smoke_sel10): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)
[bench_py] radius cross-check OK (driver_smoke_sel200): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)
[bench_py] radius cross-check OK (driver_smoke_zero_radius_ok): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)
```

Negative control (cKDTree's `query_ball_point` is a read-only Cython slot
and can't be monkeypatched directly, so a thin duck-typed proxy object was
used instead, wrapping the real cKDTree call and dropping one hit from
row 0's result before returning it):

```
negative control correctly aborted with SystemExit:
bench_py: radius cross-check FAILED (driver_smoke_negative_control, row 0): index sets differ -- flannrust=[32, 59, 71, ..., 19899] ckdtree=[32, 59, 71, ..., 19690] pynanoflann=[32, 59, 71, ..., 19899]
(flannrust/pynanoflann sets identical, 145 elements; ckdtree set missing the
last element, 144 elements -- exactly the induced 1-element drop)
```

This confirms the abort path fires on a genuine mismatch, not just that the
happy path is silent.

### Finding 2: fixed `.3` ms formatting made the single-query row illegible

`crates/xval/examples/render_report.rs`'s workload-row formatter used a
fixed `{:.3}` for every ms cell. The single-query-loop row's per-call
values (0.001968 ms, 0.007245 ms, 0.002273 ms) all round to "0.002"/
"0.007"/"0.002" -- the flannrust and pynanoflann columns become visually
identical even though the real values differ by ~15%.

**Fix chosen**: adaptive precision (the finding's second suggested option,
not the µs-unit-relabel option) -- added `fmt_ms(v: f64) -> String`: values
`< 1.0` render with 6 decimals, everything else keeps the original `.3`.
Chose this over switching to a µs unit because the column headers say "...
ms" generically across every row; relabeling only one row's numbers as µs
under an "ms" header would be a more confusing (and larger) change than
just widening precision where the magnitude needs it. Applied uniformly
per-cell (not per-row-by-name) -- simplest correct rule, and every other
row's three columns are already >> 1 ms so they render byte-identical to
before.

**Re-rendered from the already-saved `report_py.json`** (same file used in
the original Task 5 run, no re-bench needed):
```
cargo run -p xval --release --example render_report -- report.json report_py.json > report.html
```
Single-query row fragment, before vs after:
```
BEFORE:  0.002   0.007   0.002    <-- flannrust and pynanoflann indistinguishable
AFTER:   0.001968   0.007245   0.002273    <-- all three legible
```
Verified a normal row (`build_100k_dim3_f32_threads1`) still renders
`9.576 / 11.669 / 17.552` -- unchanged `.3`-decimal formatting, confirming
"keep other rows as-is".

### Verification (post-fix)

```
$ cargo test -p xval
... every test binary: test result: ok. 0 failed (render_report_test.rs: 13 passed; 0 failed)

$ cargo clippy -p xval --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s   # clean, no warnings

$ ruff check crates/flannrust-py/python/bench/bench_py.py
All checks passed!

$ cd crates/flannrust-py && .venv/bin/python -m pytest python/tests -q
346 passed, 27 xfailed, 1 xpassed in 3.72s   # baseline preserved, untouched by this fix
```

No new full-scale bench re-run was needed for this fix round: Finding 1 was
proved via a real (non-mocked) small-scale driver invocation against actual
built trees, and Finding 2 was proved by re-rendering the exact
`report_py.json` already captured from the original full run.
