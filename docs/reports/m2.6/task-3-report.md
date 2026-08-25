# Task 3 report — Python bench statistical harness (n>=10..100, mean±std) + M-py conclusion re-verification

Branch `m2p6-rigor`, starting commit `4b9fbc3`. This task ported T1's
`xval::measure_pair` adaptive-repetition timing policy to
`crates/flannrust-py/python/bench/bench_py.py` (generalized from 2 sides to
N interleaved engines: flannrust/cKDTree/pynanoflann), gave the emitted
JSON a `schema_version: 2` gate, moved the "Python bindings" HTML section
renderer from `examples/render_report.rs` into the `xval` library
(`crates/xval/src/report.rs`, as `pub fn render_python`) so it has unit
test coverage, ran the upgraded bench once in full, and re-verified every
M-py bench conclusion against the new mean/std/median/n data.

## Policy as implemented

`bench_py.py`'s new `timed_stats_interleaved(fns, budget_s=30.0, label=None)`
mirrors `xval::measure_pair` field-for-field, generalized from a fixed pair
to a `dict[name -> callable]` of any size:

- `WARMUP_REPS = 2` untimed-but-clocked warmup reps, every repeat running
  every engine once, interleaved in dict order (A, B, C, A, B, C, ... —
  same discipline for warmup and timed reps).
- `t_est_ms` = the SLOWEST engine's own mean of its 2 warmup reps (not the
  fastest, not an average across engines) — keeps every engine within
  `budget_s` at the shared `n`, identical reasoning to `measure_pair`'s
  "use the slower side" comment.
- `n = rep_count(budget_s, t_est_ms)` = `clamp(10, 100, floor(budget_s*1000
  / t_est_ms))`, `t_est_ms <= 0.0` maps to `100` — byte-for-byte the same
  formula as `xval::rep_count`, verified against `xval`'s own unit-test
  cases in a standalone smoke test before the full run (see "Verification"
  below).
- `n` timed reps, same interleave order, warmup excluded from the
  published sample.
- `gc.disable()` for the whole call (warmup + timed), same as the old
  `timed_median_interleaved` it replaces.
- `stats_from_samples` computes `{mean_ms, std_ms, median_ms, min_ms,
  max_ms, n}` — `std_ms` is the SAMPLE standard deviation (divisor `n-1`,
  `0.0` for a single-sample set), matching `TimingStats::from_samples`.
- Ratios (`ratio_ckdtree`/`ratio_pynanoflann`) are computed from
  `median_ms`, same decision statistic as every ratio claim elsewhere in
  this repo.
- `single_query_loop_workload`'s per-call figures now come from a new
  `scale_stats(stats, 1/n_calls)` helper — divides every field
  (mean/std/median/min/max, linear under a positive scalar) by `n_calls`,
  `n` (rep count) unchanged — rather than a bare scalar division.

`build_meta()` gained `budget_s`, `warmup_reps`, and a `rep_policy` string
recording the formula in the JSON itself (self-documenting provenance).
The top-level document gained `"schema_version": 2`.

`crates/xval/src/report.rs` gained `pub fn render_python(json: &str) ->
Result<String, RenderError>` (previously a private fn + structs living
only in `examples/render_report.rs`, untestable from an integration test
crate). Parses a tiny `PySchemaCheck { #[serde(default)] schema_version:
u64 }` probe FIRST; if `schema_version != 2`, returns
`RenderError::SchemaVersion { expected, found }` with a specific message
("re-run the updated bench_py.py to regenerate") BEFORE attempting the
full-document parse — so an old-format (pre-task-3, flat `flannrust_ms`
scalar fields) JSON is rejected with a message that names the actual
problem, not a generic serde "missing field flannrust_stats" error that
happens to fire for the same reason. `examples/render_report.rs` is now
just argument handling / stdin-vs-file reading / `inject_python_section`
splicing — all JSON-consuming logic moved to the library.

## Verification — RED before GREEN (renderer)

Ran the full `render_report_test.rs` suite with the `schema_version` check
temporarily deleted from `report.rs` (`cp` backup, python one-liner
removed the `if check.schema_version != EXPECTED...` block, restored
immediately after):

```
$ cargo test -p xval --test render_report_test render_python_rejects
running 2 tests
test render_python_rejects_old_format_json_without_schema_version ... FAILED
test render_python_rejects_wrong_schema_version_number ... FAILED

---- render_python_rejects_old_format_json_without_schema_version stdout ----
thread '...' panicked: error message must name the actual problem (schema_version mismatch), got:
failed to parse report JSON: missing field `flannrust_stats` at line 15 column 169
---- render_python_rejects_wrong_schema_version_number stdout ----
thread '...' panicked: schema_version: 1 must be rejected (expected 2): "<section class=\"python-bindings\">..." [rendered successfully instead of erroring]
```

Confirms the tests were genuinely exercising the new gate (not vacuously
passing), and that WITHOUT the explicit check, an old-format JSON would
either fail with a generic/unhelpful serde error (missing-field case) or —
worse — silently render (wrong-schema-number case, since the fixture there
still has all the new-format fields, just the wrong version tag). Restored
the check (`diff` confirmed byte-identical to before), then GREEN:

```
$ cargo test -p xval --test render_report_test
running 19 tests
... (all 19 ok, including the 3 new render_python_* tests)
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

End-to-end CLI smoke test (`render_report` example binary, not just the
library fn):

```
$ cargo run -p xval --release --example render_report -- report.json py_old.json   # schema_version absent
render_report: failed to render python bench section: python bench JSON schema_version mismatch: expected 2, found 0 -- re-run the updated bench_py.py to regenerate (...)
$ echo $?
1

$ cargo run -p xval --release --example render_report -- report.json py_new.json   # schema_version: 2
$ echo $?
0
```

Confirms exit code 1 + stderr message for the rejection case, exit 0 +
rendered HTML for the accepted case — matches the brief's "must fail with
a clear error (exit non-zero), not render silently wrong."

## Full bench run (single run, per brief — n>=10 IS the added statistical weight)

Rebuild + run:
```
export PATH="$HOME/.cargo/bin:$PATH"
cd crates/flannrust-py
RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release
RUSTFLAGS="-C target-cpu=native" .venv/bin/python python/bench/bench_py.py > /tmp/report_py_m26t3.json
```

`meta`: `python 3.12.11, numpy 2.5.2, scipy 1.18.1, pynanoflann 0.10.0`,
`cpu=AMD Ryzen 7 9800X3D`, `git_sha=0726e9f`, `date=2026-08-25T04:42:18Z`,
`rustflags=-C target-cpu=native`, `schema_version=2`. Exit 0. All 6 knn
cross-checks + 2 radius cross-checks passed (same seeded datasets as every
prior bench_py.py run — cross-check methodology itself untouched by this
task). Total wall time **1458.5s (~24.3 minutes)** — vs. the old fixed-n=7
run's ~7 minutes, and well inside the brief's ~75-minute ceiling (no
`BUDGET_S` reduction was needed anywhere).

### Per-cell `n` (adaptive; `rep_count(30.0, t_est_ms)`)

| Workload | n | t_est_ms | Note |
|---|---|---|---|
| `build_100k_dim3_f32_threads1` | 100 | 18.8 | upper clamp |
| `build_100k_dim3_f32_threadsNone_parallel_build` | 100 | 3.9 | upper clamp |
| `build_1M_dim3_f32_threads1` | 100 | 249.3 | upper clamp |
| `build_1M_dim3_f32_threadsNone_parallel_build` | 100 | 36.9 | upper clamp |
| `knn_batched_dim3_f32_..._workers1` | 35 | 835.2 | unclamped mid-range |
| `knn_batched_dim3_f32_..._workersNeg1` | 100 | 70.5 | upper clamp |
| `knn_dim8_float64_..._workers1` | 56 | 534.3 | unclamped mid-range |
| `knn_dim32_float32_..._workers1` | **10** | 49696.8 | **lower clamp (floor)** — dominates the run, ~82% of total wall time (~19.9 min) |
| `radius_dim3_f32_sel10` | 100 | 5.1 | upper clamp |
| `radius_dim3_f32_sel1000` | 100 | 275.4 | upper clamp |
| `single_query_loop_..._percall_ms` | 100 | 8.1 | upper clamp |

### Full result rows (median (mean±std, n) per engine; ratios are median-based)

| Workload | flannrust ms | cKDTree ms | pynanoflann ms | ratio_ckdtree | ratio_pynanoflann |
|---|---|---|---|---|---|
| `build_100k_dim3_f32_threads1` | 10.295 (10.338±0.251, n=100) | 12.154 (12.228±0.348, n=100) | 18.607 (18.685±0.405, n=100) | 0.847 | 0.553 |
| `build_100k_dim3_f32_threadsNone_parallel_build` | 3.337 (3.355±0.150, n=100) | (reused) | (reused) | 0.275 | 0.179 |
| `build_1M_dim3_f32_threads1` | 138.394 (143.181±30.913, n=100) | 143.085 (150.962±21.135, n=100) | 250.333 (259.096±44.384, n=100) | 0.967 | 0.553 |
| `build_1M_dim3_f32_threadsNone_parallel_build` | 35.553 (44.255±60.627, n=100) | (reused) | (reused) | 0.248 | 0.142 |
| `knn_batched_dim3_f32_k10_q200k_workers1` | 344.292 (344.881±13.160, n=35) | 698.875 (712.763±75.637, n=35) | 341.311 (346.571±16.859, n=35) | 0.493 | 1.009 |
| `knn_batched_dim3_f32_k10_q200k_workersNeg1` | 49.236 (50.313±3.631, n=100) | 70.028 (72.838±8.476, n=100) | 60.894 (58.719±7.836, n=100) | 0.703 | 0.809 |
| `knn_dim8_float64_k10_workers1` | 421.469 (432.263±35.072, n=56) | 533.628 (552.892±64.981, n=56) | 331.565 (342.506±38.732, n=56) | 0.790 | 1.271 |
| `knn_dim32_float32_k10_workers1` | 24018.707 (24010.968±144.628, n=10) | 49443.517 (49472.543±984.183, n=10) | 26055.463 (26045.315±160.961, n=10) | 0.486 | 0.922 |
| `radius_dim3_f32_sel10` | 2.838 (2.896±0.173, n=100) | 4.577 (4.617±0.229, n=100) | 4.809 (4.831±0.184, n=100) | 0.620 | 0.590 |
| `radius_dim3_f32_sel1000` | 70.806 (71.892±5.548, n=100) | 133.671 (137.770±20.464, n=100) | 246.557 (251.527±27.516, n=100) | 0.530 | 0.287 |
| `single_query_loop_dim3_f32_k10_percall_ms` | 0.002226 (0.002357±0.000342, n=100) | 0.008361 (0.008527±0.000809, n=100) | 0.002683 (0.002770±0.000392, n=100) | 0.266 | 0.830 |

Notable outliers surfaced by the new min/max fields (published for the
first time; the old median-of-7 methodology never captured this):
`build_1M_..._threadsNone` flannrust `max=635.907` vs `median=35.553` ms
(~18x, single severe stall); `build_1M_..._threads1` flannrust
`max=434.277` vs `median=138.394` (~3.1x); `knn_batched_..._workers1`
cKDTree `max=1004.586` vs `median=698.876` (~1.4x, but on the widest
relative spread of any `n>=35` cell — `std=75.637`). Same WSL2
scheduler/thermal-stall phenomenon M2.6 task 2 already documented for the
Rust-vs-C++ side, now visible on the Python side because `n=100`
publishes min/max.

## Per-conclusion verdicts vs. recorded M-py ranges

Since this task touched only the bench harness and renderer — never
`crates/flannrust`'s query/build/kernel code or `flannrust-py`'s Rust
binding code — every ratio shift below is measurement noise/environment
variance, not a code-driven change.

| # | Conclusion | Old range | New (n>=10) | Verdict |
|---|---|---|---|---|
| (a) | Batched knn workers=1 vs cKDTree | 0.712–0.830 | **0.493** (n=35) | **CORRECTED, flagged single-session outlier** — still MET (≤1.00). Root cause: this run's cKDTree median (698.9ms) is ~63% slower than any prior run (426–430ms) for this exact cell while flannrust (344.3ms) is unchanged; new range 0.493–0.830, NOT declared the new steady state pending a confirming re-run. |
| (b) | Batched knn workers=-1 vs cKDTree | 0.712–0.827 | **0.703** (n=100) | **CORRECTED, minor** — ordinary ~1-point widening. New range 0.703–0.827. |
| (c) | Build vs pynanoflann, 100k | 0.546–0.561 | **0.553** (n=100) | **CONFIRMED** — inside old range. |
| (d) | Build vs pynanoflann, 1M | 0.543–0.551 | **0.553** (n=100) | **CORRECTED, marginal** — 0.002 above old high end, far smaller than this workload's own session std (21–61ms on a ~140ms median). New range 0.543–0.553. |
| (e) | dim-32 knn vs pynanoflann | 0.878–0.899 | **0.922** (n=10, clamp floor) | **CORRECTED, widened, still MET** (≤1.10 gate, comfortable margin). New range 0.878–0.922. |
| (f) | Per-call overhead | flannrust ≈2.0–2.3µs, cKDTree ≈7.2–7.5µs, pynanoflann ≈2.3µs | flannrust 2.226µs (median)/2.357µs (mean); cKDTree 8.361/8.527µs; pynanoflann 2.683/2.770µs (all n=100) | **Essentially CONFIRMED for flannrust** (median in-range); **CORRECTED (widened) for cKDTree/pynanoflann**. Relative ordering (flannrust < pynanoflann < cKDTree) unchanged. |
| (g) | Honest miss: batched knn workers=1 vs pynanoflann | 1.058–1.216 | **1.009** (n=35) | **CORRECTED, near-parity — same volatility caveat as (a), NOT declared resolved.** Same cell as (a); pynanoflann's median this session (341.3ms) also somewhat above its historical range (289–291ms), though less dramatically than cKDTree's swing. New range 1.009–1.216, flagged for confirmation. |
| (h) | Honest miss: knn_dim8_float64 workers=1 vs pynanoflann | 1.233–1.249 | **1.271** (n=56) | **CORRECTED, widened, ordinary (non-outlier-shaped)** — this cell's distribution (`std=38.732` on `median=421.469`ms) is unremarkable, unlike (a)/(g). New range 1.233–1.271, still the most consistent miss in the matrix. |

**Summary**: every gated criterion still passes, most with wide margin.
Two ranges widen only marginally ((b), (d)). Two widen more substantially
but stay comfortably inside their gates, ordinary noise shape ((e), (h)).
Two show a large, specifically-flagged single-session swing traceable to
one interleaved cell's cKDTree/pynanoflann engines running unusually slow
((a), (g)) — published per this repo's range-honesty convention, explicitly
NOT folded silently into "the new steady state." This run stays **WSL2**;
bare-metal re-verification (including these Python numbers) remains an
open **M-pub** item, unchanged by this task.

## Docs updated (house style: pasted evidence never edited in place, only Update banners + new sections added)

- `docs/EXPERIMENTS.md`: new `### M2.6 task 3` subsection (full policy
  description, rebuild/run commands, cross-check output, per-cell `n`
  table, full result-row table, per-conclusion re-verification with the
  noise-vs-code-change reasoning spelled out) inserted after `### M2.6 task
  2` (before `## 5. Number provenance`); 2 new provenance-table rows.
- `docs/benchmarks.md`: `Update (M2.6 task 3, ...)` paragraph + new
  success-criteria comparison table after the existing M-py success table;
  `Update (M2.6 task 3)` paragraph after the existing honest-misses table.
  Old tables/paragraphs untouched.
- `README.md`: `Update (M2.6 task 3, 2026-08-25)` paragraph + new table
  after the existing "Benchmarks" table (Python bindings section); M-py
  roadmap bullet gained a pointer sentence (no number duplication, per
  M2.5's own precedent for cross-referencing rather than restating).
- `docs/ROADMAP.md`: M-py section gained an `Update (M2.6 task 3, ...)`
  bullet; the M-pub dim8-investigation bullet gained an inline `Update
  (M2.6 task 3)` sentence with the corrected 1.233–1.271 range.

Caught and fixed during self-review: an early draft of README's combined
"Build vs pynanoflann (100k / 1M)" row incorrectly narrowed the range to
`0.543–0.553` — but the COMBINED old range (0.543–0.561, from the wider
100k side) already contains both new points (0.553, 0.553), so the honest
combined citation is unchanged at `0.543–0.561` with the 1M-specific
nuance noted in parens; only the standalone 1M row (in `docs/benchmarks.md`
and `docs/EXPERIMENTS.md`, where 100k/1M are listed separately) needed the
`0.543–0.553` widening. Verified all four docs' markdown tables are
column-count-consistent with a small Python checker before finalizing.

## Verify commands run (all green)

```
cargo test -p xval --test render_report_test        # 19 passed
cargo test --workspace --all-features                # all green (unchanged pass counts elsewhere)
cargo clippy --workspace --all-targets -- -D warnings # clean
RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust --no-deps  # clean
RUSTDOCFLAGS="-D warnings" cargo doc -p xval --no-deps       # clean
cd crates/flannrust-py && .venv/bin/pytest python/tests -q  # 351 passed, 27 xfailed, 1 xpassed (baseline unchanged)
ruff check crates/flannrust-py/python/bench/bench_py.py      # All checks passed!
```

Also: standalone smoke tests of `rep_count`/`stats_from_samples`/
`scale_stats`/`timed_stats_interleaved` against `xval`'s own unit-test
cases (clamp-at-10, clamp-at-100, `t_est<=0 -> 100`, mid-range exact
floor) before committing to the ~24-minute full run, and a small
`build_workload(2000, ...)` end-to-end smoke run to catch integration
issues cheaply.

## Self-review / concerns

- **The `n=35` batched-knn-workers1 cell's own volatility is the direct
  cause of two of the most surprising verdicts ((a), (g))**: this is
  flagged explicitly and repeatedly in every doc touched (not just once) —
  the risk is a future reader citing "0.493" or "1.009" as the new
  established figure without the outlier caveat. Mitigated by wording every
  citation as "new range X–Y, flagged/not declared resolved" rather than a
  bare new number, and by pointing to the cKDTree-specific evidence
  (698.9ms vs. every prior run's 426–430ms) that makes clear this isn't a
  flannrust improvement.
- **Single run only** (per brief) means this task cannot itself distinguish
  "genuine session-to-session noise" from "a real but transient host
  condition" the way M2.6 task 2's 4-session Rust-side re-verification
  could — this is stated as an open item, not glossed over.
- **`crates/xval/src/report.rs`'s `RenderError` enum gained a new variant**
  (`SchemaVersion`) — technically outside "T1's work," but required by the
  brief's file-allowlist note ("report.rs if the parsed schema needs the
  new fields") and additive-only (no existing variant/match arm changed);
  `cargo clippy --workspace` confirms no exhaustiveness-match fallout
  elsewhere.
- **`fmt_ms` (private helper) became dead code** after refactoring
  `fmt_py_stat_cell` to choose ONE precision (from the row's median
  magnitude) for all three displayed numbers instead of formatting each
  independently (the original per-value-adaptive approach rendered e.g.
  `10.295 (10.338±0.250652, n=100)` — inconsistent decimal counts within
  one cell) — deleted rather than left unused; caught by a real render
  against the live JSON, not just the unit-test fixtures (which happened to
  use only >1ms std values, masking the issue).
- Did not attempt to profile/root-cause the cKDTree/pynanoflann slowdown in
  the `workers=1` batched-knn cell — out of this task's scope (bench
  harness + docs, not a WSL2/host investigation); flagged as a caveat, not
  chased.

## Commit

`feat(bench_py): n>=10 adaptive reps with mean/std; docs re-verified`
