# Task 1 report: Statistical harness — Rust side

Branch: `m2p6-rigor`. Scope: `crates/xval/src/lib.rs`, `crates/xval/tests/perf_gate.rs`,
`crates/xval/examples/report_data.rs`, `crates/xval/src/report.rs`,
`crates/xval/tests/render_report_test.rs`. `crates/flannrust`, `crates/nanoflann-ref`,
`bench_py.py`, and `crates/xval/examples/render_report.rs` (the Python-section CLI wrapper)
were not touched.

## Design

Old methodology (`timed_median_ms`/`median_of`): 1 untimed warmup, 7 timed reps, median.
Called sequentially per side (`rust_ms = timed_median_ms(...)`, then `cpp_ms = timed_median_ms(...)`)
— no interleaving.

New methodology, added to `crates/xval/src/lib.rs` alongside (not replacing) the old helpers:

- `TimingStats { mean_ms, std_ms, median_ms, min_ms, max_ms, n }` — `std_ms` is the SAMPLE
  standard deviation (divisor `n-1`), `0.0` for `n=1` (no degrees of freedom, not `NaN`).
  `TimingStats::from_samples(&[f64]) -> Self` is the pure stats-math entry point, unit-tested
  directly against a hand-checked array (mean 5.0, std `sqrt(32/7)`, even-length median) —
  independent of any real timing.
- `rep_count(budget_s, t_est_ms) -> usize` (private): `clamp(10, 100, floor(budget_s*1000 /
  t_est_ms))`. `t_est_ms <= 0.0` maps to the upper bound (100) rather than dividing by
  zero/negative (guards a per-rep cost too fast for `Instant`'s resolution to distinguish from
  zero).
- `measure<F: FnMut()>(f, budget_s) -> TimingStats`: 2 warmup reps (their timings seed
  `t_est_ms`, then are discarded from the returned stats), then `rep_count(...)` timed reps.
  Times one side only.
- `measure_pair<A, B>(rust_fn, cpp_fn, budget_s) -> (TimingStats, TimingStats)`: interleaves
  rust/cpp PER REPETITION — including the 2 warmup reps (R,C,R,C,... throughout, not "warmup
  both then time both"), so a machine-noise/thermal-drift trend can't bias the comparison
  toward whichever side runs first. The shared `n` is computed once from
  `rep_count(budget_s, max(t_est_rust_ms, t_est_cpp_ms))` — using the SLOWER side's estimate
  guarantees BOTH sides' total measured time stays within `budget_s` (the faster side, by
  definition, uses less than its own budget at that same `n`).

Gate/report criterion is unchanged: **ratio of MEDIANS** vs `MARGIN = 1.25`. Mean/std/min/max/n
are published alongside for statistical honesty, never used for the PASS/FAIL decision.

`ratio_means_std` (report_data.rs, new): standard error of `ratio_means` via first-order
(delta-method) error propagation, using each side's own **standard error of the mean**
(`std_ms / sqrt(n)`, not the raw per-rep `std_ms`) — the correct quantity for "how uncertain is
this average-based ratio estimate." This formula isn't specified verbatim in the brief; I
documented the choice inline (`report_data.rs`, `ratio_means_std`'s doc comment) so it's
auditable.

`timed_median_ms`/`median_of` were **kept, untouched** — `examples/m25_diag.rs` still uses them
and is out of this task's scope.

## RED evidence (TDD, stats math)

Added 9 tests to `crates/xval/src/lib.rs`'s `#[cfg(test)] mod tests` referencing
`TimingStats`, `rep_count`, `measure`, `measure_pair` before any of those existed. `cargo test
-p xval --lib` failed to compile with 11 errors (`E0433`/`E0425`, "cannot find type
`TimingStats`" / "cannot find function `rep_count`/`measure`/`measure_pair`"). Full RED
transcript excerpt:

```
error[E0433]: cannot find type `TimingStats` in this scope
    --> crates/xval/src/lib.rs:2420:17
error[E0425]: cannot find function `rep_count` in this scope
    --> crates/xval/src/lib.rs:2426:20
error[E0425]: cannot find function `measure` in this scope
    --> crates/xval/src/lib.rs:2452:21
error[E0425]: cannot find function `measure_pair` in this scope
    --> crates/xval/src/lib.rs:2470:39
error: could not compile `xval` (lib test) due to 11 previous errors
```

After implementing the four items: `cargo test -p xval --lib` → `test result: ok. 90 passed; 0
failed`. New tests cover: known-array mean/std/median/min/max; `n=1` → `std_ms=0.0`; empty
sample panics; `rep_count` clamped at both the 10 and 100 bounds, an exact-floor mid-range
case, and the `t_est_ms <= 0.0` guard; `measure`'s warmup-exclusion invariant
(`total_calls == 2 + stats.n`, `n` within `[10,100]`); `measure_pair`'s shared-`n` and
warmup-exclusion invariants on both sides; and an explicit interleaving-order test
(`measure_pair_orders_calls_interleaved_not_batched`) that records which side fired on every
call (warmup included) and asserts strict R,C,R,C,... alternation — this specifically catches a
"warmup both sequentially, then time both sequentially" implementation, which would pass the
simpler count-based tests but fail this one.

`render_report_test.rs` got 3 new tests (not strict RED-first — written after the renderer
change, to lock in the new display behavior, since the brief's explicit TDD requirement was
scoped to "the stats math," not the renderer): mean/std/n both display; bar geometry uses
`ratio_medians` even when it's fed an adversarially-contradictory legacy `ratio` field (proves
the renderer isn't silently still keying off the old field); and a pre-M2.6 fixture (no `rust`/
`cpp` stats objects) still renders via the legacy display with no fabricated `n=`. All 16 tests
in that file pass, including the 13 pre-existing ones **completely unmodified** — confirms the
new `rust`/`cpp`/`ratio_means`/`ratio_means_std`/`ratio_medians` fields are `#[serde(default)]
Option<...>` and don't break old-format JSON.

## Files changed

- `crates/xval/src/lib.rs`: `TimingStats`, `rep_count`, `measure`, `measure_pair` + 9 unit
  tests. `timed_median_ms`/`median_of` untouched.
- `crates/xval/tests/perf_gate.rs`: all 6 gates converted to `measure_pair(.., BUDGET_S)`
  (`BUDGET_S = 30.0`, replacing `RUNS = 7`). `report()`/`assert_gate!` rewritten to take
  `&TimingStats` and print the extended `PERF_GATE` line (see below). Module doc comment
  updated.
- `crates/xval/examples/report_data.rs`: all 8 speed workloads converted to `measure_pair`.
  `SpeedRow` now holds `rust: TimingStats, cpp: TimingStats` instead of bare `f64`s. New
  `timing_stats_json`/`ratio_means_std` helpers. JSON emission extended per the brief's schema
  (`rust`, `cpp`, `rust_ms`/`cpp_ms`/`ratio` — from medians — `ratio_means`, `ratio_means_std`,
  `ratio_medians`).
- `crates/xval/src/report.rs`: `SpeedStats` (new, `Option`-wrapped on `SpeedRow`) carries
  `mean_ms`/`std_ms`/`n` (the three fields actually displayed; `median_ms`/`min_ms`/`max_ms` are
  present in the JSON but ignored here — `rust_ms`/`cpp_ms` already carry the median). `ratio_medians:
  Option<f64>` added to `SpeedRow`. `render_speed_rows` now: (a) bar geometry uses
  `r.ratio_medians.unwrap_or(r.ratio)`; (b) tooltip and bar-value cell append `mean ± std (n=N)`
  for both sides when present, falling back to the exact pre-M2.6 display when absent.
- `crates/xval/tests/render_report_test.rs`: +3 tests (see RED section).

## PERF_GATE output line format

Kept the historical `PERF_GATE {name}: rust={..}ms cpp={..}ms ratio={..}` prefix byte-for-byte
(`rust`/`cpp` are now the medians, same field names/positions the docs' pasted blocks grep for),
appended stats after a `|`:

```
PERF_GATE {name}: rust={median}ms cpp={median}ms ratio={ratio_medians} | rust mean={} std={} n={} | cpp mean={} std={} n={} | ratio_medians={}
```

`ratio=` and the trailing `ratio_medians=` are the same value (the brief's own example shows
both), redundant but explicit for anyone grepping either name.

## Worst-case runtime

Full `PERF_GATE` suite (6 gates, `--ignored --test-threads=1`, `RUSTFLAGS="-C
target-cpu=native"`, release): **49.07s** (was ~seconds under median-of-7). Every gate's `n`
landed at the upper clamp bound (100) — none of the current workloads are slow enough (>3s/rep)
to hit the 10-rep floor. `report_data` (8 speed workloads + accuracy/brute-force GT, same
harness): **98s** wall time. Both comfortably within the brief's "grows from seconds to a few
minutes; accepted."

## Verification commands + decisive output

**RED** (pre-implementation): see above — 11 compile errors referencing the four undefined
items.

**GREEN, lib unit tests:**
```
$ cargo test -p xval --lib
test result: ok. 90 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
```

**render_report_test.rs (backward-compat + new fields):**
```
$ cargo test -p xval --release --test render_report_test
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**Full workspace:**
```
$ cargo test --workspace
```
Every `test result:` line across the workspace (flannrust, flannrust-py, nanoflann-ref, xval —
unit + integration + doctests): `ok. ... 0 failed`. Totals include `flannrust`'s own 198-test
lib suite, `xval`'s 90-test lib suite (the stats-math tests above), the 16 `render_report_test`
tests, and all `tests/xval_*.rs`/`bench_sanity.rs`/`native_parity.rs` suites — 0 failures, 0
regressions.

**clippy `-D warnings` (workspace, all targets):**
```
$ cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 11.31s
```
Clean, no warnings.

**rustdoc, zero warnings** (`cargo doc --workspace` errors on a PRE-EXISTING, unrelated name
collision between the `flannrust` and `flannrust-py` crates' output paths — not caused by this
task; documented `-p xval -p flannrust -p nanoflann-ref` instead, which is everything xval
touches or depends on):
```
$ RUSTDOCFLAGS="-D warnings" cargo doc -p xval -p flannrust -p nanoflann-ref --no-deps
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.54s
```
Clean.

**`cargo build -p flannrust --no-default-features`:**
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
```
Clean.

**Full PERF_GATE suite (decisive output, every gate PASSES, all well under MARGIN=1.25):**
```
$ PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --ignored --test-threads=1 --nocapture

PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.352ms cpp=9.679ms ratio=0.966 | rust mean=9.444 std=0.230 n=100 | cpp mean=9.763 std=0.248 n=100 | ratio_medians=0.966
ok
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.956ms cpp=3.497ms ratio=1.131 | rust mean=3.998 std=0.142 n=100 | cpp mean=3.533 std=0.130 n=100 | ratio_medians=1.131
ok
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.995ms cpp=31.777ms ratio=0.944 | rust mean=30.080 std=0.554 n=100 | cpp mean=31.879 std=0.485 n=100 | ratio_medians=0.944
ok
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.343ms cpp=7.078ms ratio=1.038 | rust mean=7.420 std=0.227 n=100 | cpp mean=7.168 std=0.214 n=100 | ratio_medians=1.038
ok
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=175.639ms cpp=188.855ms ratio=0.930 | rust mean=175.822 std=1.581 n=100 | cpp mean=189.163 std=1.899 n=100 | ratio_medians=0.930
ok
PERF_GATE perf_gate_radius_dim3_f32: rust=5.232ms cpp=6.498ms ratio=0.805 | rust mean=5.268 std=0.148 n=100 | cpp mean=6.566 std=0.201 n=100 | ratio_medians=0.805
ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 49.07s
```

**Full `report_data` + `render_report` chain:**
```
$ RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example report_data > report.json
# 98s wall time, 8 speed rows + 3 static + 2 dynamic accuracy rows, no errors.
$ cargo run -p xval --release --example render_report -- report.json > report.html
# succeeds; 453-line self-contained HTML.
```
Spot-checked `report.json`'s first speed row has the full schema:
```json
{
  "workload": "build_100k_dim3_f32_seq",
  "rust": { "mean_ms": 10.089292, "std_ms": 0.285658, "median_ms": 9.97983, "min_ms": 9.764194, "max_ms": 11.133814, "n": 100 },
  "cpp": { "mean_ms": 9.652467, "std_ms": 0.25832, "median_ms": 9.565493, "min_ms": 9.363982, "max_ms": 10.56306, "n": 100 },
  "rust_ms": 9.97983, "cpp_ms": 9.565493, "ratio": 1.043316,
  "ratio_means": 1.045255, "ratio_means_std": 0.004072, "ratio_medians": 1.043316
}
```
Spot-checked `report.html`'s bar-value cell for that row:
```
1.043·9.980 ms / 9.565 ms | rust 10.089±0.286 ms (n=100) · cpp 9.652±0.258 ms (n=100)
```
and its tooltip (`title="..."`, plain Unicode `±`/`·`, correctly not double-escaped):
```
build_100k_dim3_f32_seq: ratio_medians 1.043 (rust median 9.980 ms / cpp median 9.565 ms) | rust 10.089±0.286 ms (n=100) · cpp 9.652±0.258 ms (n=100)
```
Confirmed 0 leftover `{placeholder}` tokens in the rendered HTML (`grep -c` for all 5
placeholder substrings → 0), and all 8 speed rows carry `n=100` on both sides.

## Self-review

- Bit-exactness: untouched — no changes to `crates/flannrust` or `crates/nanoflann-ref`.
- Gate decision rule: unchanged (ratio of medians vs `MARGIN=1.25`), now over ≥10 (here, always
  100) reps instead of 7.
- Backward compatibility: the renderer's new fields are all `Option`, verified by re-running the
  13 pre-existing `render_report_test.rs` tests completely unmodified against the updated
  renderer (all pass) plus a dedicated `render_speed_row_without_stats_still_renders_legacy_display`
  test.
- `crates/xval/examples/render_report.rs` (the Python-section CLI): confirmed untouched via
  `git diff crates/xval/examples/render_report.rs` (empty). Its `PyReport`/`render_python_section`/
  `inject_python_section` data path is Task 3's, not touched here.
- `docs/*.md` pasted `PERF_GATE ...` blocks: not edited (grepped, confirmed only referenced,
  never modified).
- Scope check: `git diff --name-only` shows exactly 5 files, all under `crates/xval/`.

## Concerns / open items for the controller

1. `ratio_means_std`'s exact formula (delta-method propagation using each side's *standard
   error of the mean*, `std_ms/sqrt(n)`) is my own defensible choice, not spelled out in the
   brief — flagged for controller review in case a different convention (e.g. propagating raw
   per-rep `std_ms` instead of SEM) is preferred for M2.6's docs.
2. `crates/xval/src/report.rs`'s `SpeedStats` intentionally does not carry `median_ms`/`min_ms`/
   `max_ms` from the JSON (serde ignores the unused fields) — `rust_ms`/`cpp_ms` on `SpeedRow`
   already carry the median, and no page currently needs the reported min/max. If a future task
   wants min/max on the scorecard, those two fields need adding to `SpeedStats`.
3. Every current perf-gate/report speed workload lands at `n=100` (the upper clamp) given the
   30s-per-side budget — the 10-rep floor path is only exercised by unit tests
   (`rep_count_clamps_at_lower_bound_10`), not by any real workload in this repo today. Not a
   defect, just noting the floor is currently untested end-to-end on real timings.
