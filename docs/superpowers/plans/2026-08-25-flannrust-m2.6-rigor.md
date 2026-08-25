# flannrust — Milestone 2.6: Statistical rigor + implementation-fidelity audit

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Same pipeline as M1–M-py.

**Goal:** (1) Replace the median×7 methodology with a statistically defensible one — ≥10 repetitions per scenario, 100 where runs are cheap, published as mean ± std alongside the median — and re-verify every recorded conclusion under it. (2) Audit the Rust port line-by-line against nanoflann 1.12.1 on the workloads where C++ still wins, and close any gap that comes from a missed implementation detail — no tricks, no deviations: follow the C++ exactly.

**User directives (2026-08-25, binding):** "it shouldnt be just 1. it should be 10 (at least) and if its fast enough it should be 100 and we can get a clear mean and std"; "no extra tricks, follow flann exactly but see if we missed something in the rust implementation of it".

**Spec:** no separate spec — this plan + the M1 plan's parity constraints (`docs/superpowers/plans/2026-08-22-nanoflann-rs-m1.md`) are the authority. Binding constraints carried over from M2.5 verbatim: default-build bit-exactness (full xval suite green after every change), measure-first discipline, provenance rules (every number traces to a pasted run).

## Global Constraints

- Bit-exact parity on the default build: full xval suite (static + dynamic) green after every landed change; any candidate that reorders floating-point arithmetic on the default path is REJECTED (not feature-gated — this milestone allows none).
- Repetition policy (the user's rule, made precise): every timed scenario runs `n = clamp(10, 100, floor(BUDGET / t_est))` repetitions, where `t_est` is a 2-rep warm estimate and BUDGET = 30 s per scenario side (Rust and C++ measured separately but interleaved rep-by-rep). n < 10 is never allowed; scenarios cheaper than 300 ms/rep therefore run 100 reps. Warmup reps (2) are excluded from stats.
- Published statistics per scenario side: mean, sample std (n−1), median, min, max, n — plus the ratio of means with a propagated-uncertainty band (ratio_std ≈ ratio × sqrt((std_r/mean_r)² + (std_c/mean_c)²)) and the ratio of medians. Decisions (gates, docs claims) use the ratio of medians; mean ± std is published so readers see the spread.
- Interleaving: A,B,A,B per repetition within one process, GC/alloc warm on both sides, same buffers reused (existing zero-allocation-per-query discipline unchanged).
- clippy `-D warnings`, zero rustdoc warnings, `--no-default-features` builds, pytest suite stays green at every commit.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01T1ELzxC42AesQeKGCT7Kt1`
- Host facts: `export PATH="$HOME/.cargo/bin:$PATH"`; venv `crates/flannrust-py/.venv`; WSL2 (noise floor recorded in EXPERIMENTS.md §1 — one purpose of this milestone is to characterize it properly with n=100 distributions instead of eyeballed ranges).

---

### Task 1: Statistical harness — Rust side

**Files:**
- Modify: `crates/xval/src/perf_gate.rs` (or wherever the PERF_GATE timing loop lives — locate it), `crates/xval/examples/report_data.rs`, `crates/xval/examples/render_report.rs`
- Test: existing render_report_test.rs extended for the new JSON fields

**Interfaces:**
- Produces: a shared timing helper in `crates/xval/src/lib.rs`:
  ```rust
  pub struct TimingStats { pub mean_ms: f64, pub std_ms: f64, pub median_ms: f64,
                           pub min_ms: f64, pub max_ms: f64, pub n: usize }
  /// 2 warmup reps (discarded), then n = clamp(10, 100, floor(budget_s*1000 / t_est_ms))
  /// timed reps of `f`, interleaved by the CALLER (this times one side only).
  pub fn measure<F: FnMut()>(f: F, budget_s: f64) -> TimingStats;
  ```
  and an interleaved pair driver `measure_pair(rust_fn, cpp_fn, budget_s) -> (TimingStats, TimingStats)` that alternates rust/cpp per repetition (this is what gates and report_data call).
- report.json schema per workload grows: `{rust: TimingStats, cpp: TimingStats, ratio_means, ratio_means_std, ratio_medians}` (keep the old `rust_ms/cpp_ms/ratio` keys populated from the medians for renderer compatibility until render_report is updated in the same task).

- [ ] TDD: unit tests for the stats math (known array → mean/std/median; std uses n−1; clamp logic at the 10 and 100 bounds; warmup excluded) — write failing tests against the new helper first.
- [ ] Implement `measure`/`measure_pair`; convert the perf gate to use it (gate criterion: ratio of medians vs the existing MARGIN — unchanged threshold, now on ≥10 reps; print per-gate `mean±std (n=N) median` for both sides in the PERF_GATE output line).
- [ ] Convert `report_data.rs` to `measure_pair` for every speed workload; extend the JSON as above; update `render_report.rs` to show `mean ± std` and n in the tooltip/value cell and keep the bar on the median ratio. Python-section rendering untouched (Task 3 handles bench_py).
- [ ] Verify: `cargo test --workspace`; run the full gate suite once and paste (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --test-threads=1 --nocapture`); regenerate report.json and confirm the renderer output on a real run; clippy/rustdoc clean.
- [ ] Commit: `feat(xval): n>=10 adaptive repetitions with mean/std/median stats in gates and report`.

### Task 2: Full statistical re-run + conclusion re-verification (docs)

**Files:**
- Modify: `docs/EXPERIMENTS.md` (new "M2.6 statistical baseline" section + provenance rows), `docs/benchmarks.md` (headline tables gain mean ± std columns; a "conclusions re-verified" subsection), `docs/ROADMAP.md`
- No library code changes.

- [ ] Run the full Rust chain under the new harness ×2 sessions (gates + report_data), paste everything into EXPERIMENTS.md with SHAs/dates.
- [ ] Re-verify each RECORDED conclusion against the new distributions and state explicitly CONFIRMED or CORRECTED with numbers: (a) M2.5 T3 dim-32 f32 win; (b) M2.5 T2 trade-off give-backs (radius ~0.83, dim8 f64 back to pre-M2.5 band); (c) fixed-3 residual ~1–4%; (d) dim-16 curse-of-dimensionality attribution; (e) the "noise floor" itself — replace the eyeballed 0.956–1.192 range with the measured std at n=100 for knn_fixed3 and state the new floor as mean ± k·std; (f) dyn_add 1.11–1.24 range; (g) build_100k seq range (the 0.974–1.139 spread is suspiciously wide — n=100 settles whether that is noise or drift).
- [ ] Any conclusion that flips: correct it EVERYWHERE it appears (README/benchmarks/EXPERIMENTS/ROADMAP/nanoflann-notes), with the old claim struck through or superseded-banner'd per house style, and list every flip in the task report.
- [ ] Commit: `docs: M2.6 statistical baseline (n>=10..100, mean+-std) and conclusion re-verification`.

### Task 3: Statistical harness — Python bench

**Files:**
- Modify: `crates/flannrust-py/python/bench/bench_py.py`, `crates/xval/examples/render_report.rs` (Python section shows mean ± std + n)

- [ ] Same repetition policy in Python (`n = clamp(10, 100, floor(30 / t_est))` per cell side, 2 warmup reps, A/B/C interleaved per rep as today); JSON per cell grows `{lib}_stats = {mean_ms, std_ms, median_ms, min_ms, max_ms, n}`; ratios switch to medians with mean±std published; cross-checks unchanged.
- [ ] Renderer Python section updated for the new fields (old-format JSON must fail loudly, not render silently wrong — version key in the JSON).
- [ ] Run the full Python bench once under the new policy, paste all rows; `ruff check` clean; pytest suite untouched and green.
- [ ] Re-verify the M-py conclusions (0.712–0.830 vs cKDTree, 0.543–0.561 build, dim-32 0.878–0.899, the two honest misses) against mean ± std; correct docs where the n≥10 numbers disagree (same everywhere-rule as Task 2); note this run stays WSL2 — bare-metal remains M-pub.
- [ ] Commit: `feat(bench_py): n>=10 adaptive reps with mean/std; docs re-verified`.

### Task 4: Fidelity audit of the losing workloads (diagnosis — NO library code changes)

**Files:** report only (plus optional clearly-labeled diagnostic probes under `crates/xval`, committed as `chore(xval): M2.6 diagnostic probes`).

Audit targets (the rows where C++ wins, from the M2.6 baseline of Task 2 — re-rank by the new numbers): sequential build (100k and 1M dim-3 f32), knn_fixed3 residual, dyn_add_20k, dim-32 f64 (~1.10), dim-64 f32 (~1.16–1.23).

- [ ] For each target, a line-by-line comparison of the Rust implementation against the vendored `crates/nanoflann-ref/cpp/nanoflann.hpp` 1.12.1 — the question is NOT "what optimization could we invent" but "what does the C++ do that we do differently": allocation pattern (C++ PooledAllocator grows in 260KB blocks and never frees per-node; our arena Vec growth/reserve behavior — measure reallocation counts during build), `computeMinMax`/`middleSplit_`/`planeSplit` loop shapes and their codegen, `vAcc_` index type widths and copy behavior at build start, dataset access patterns in the build loop (per-component calls in build were never row-pointered — the M2.5 fix only touched query kernels; measure whether build's `point_component` loops vectorize), dynamic adaptor's rebuild path (C++ reuses the sub-tree object; do we rebuild allocations from scratch per slot?), fixed-dim-3 query wiring differences (anything left after M2.5's iterative conversion — frame size, spill checks on the hot path).
- [ ] Every hypothesis gets a measurement (counting probe, asm excerpt, or A/B of an uncommitted local patch — reverted after measuring, `git status` clean) with the new n≥10 harness; predicted gain stated per candidate.
- [ ] Deliverable: ranked, evidence-backed candidate list for Task 5, each marked FIDELITY (C++ does X, we do Y — change to X) or REJECT (we already match; the gap is compiler/ABI/noise). Anything that would deviate from the C++ algorithm is out of scope by user directive — list it under "not eligible" if found.
- [ ] Explicitly re-examine with n=100 whether knn_fixed3 and build_100k seq are even LOSING (their historical ranges straddle 1.0) — "no real gap" is a valid finding that Task 5 then skips.

### Task 5: Land the fidelity fixes

**Files:** whatever Task 4's winning candidates name inside `crates/flannrust/src/` (+ tests).

- [ ] Per candidate, in Task 4's rank order: TDD where behavior-visible (usually invisible — the xval suite is the judge), implement exactly the C++ behavior, A/B with `measure_pair` n≥10 interleaved (paste both sides' mean ± std), land only if the median ratio improves beyond the propagated uncertainty band and NO other gate regresses beyond its band; revert-with-numbers otherwise (record in the report — reverted candidates are results too).
- [ ] Full suite + heavy `--ignored` release tests + both mutation canaries green after each landed candidate; miri (both aliasing models) re-run if `search.rs` or any unsafe-adjacent code is touched.
- [ ] One commit per landed candidate (`perf: <what> (C++ fidelity: <workload> <before>-><after>)`), one final commit for the docs deltas (benchmarks/EXPERIMENTS/ROADMAP updated with before/after mean ± std, provenance rows).

### Task 6: Docs close-out + regression sweep

- [ ] Re-run ALL gates + report chain + Python bench once post-Task-5, paste; update benchmarks.md M2.6 section with final tables (mean ± std everywhere), EXPERIMENTS.md provenance; ROADMAP: M2.6 complete, M-pub unblocked-by list updated (bare-metal re-run now measures mean ± std too).
- [ ] Verify: workspace tests, pytest, clippy, rustdoc, `--no-default-features` — pasted.
- [ ] Commit: `docs: M2.6 results — statistical baseline and fidelity fixes`.

---

## Self-review notes

- User directive coverage: ≥10/100 reps + mean/std → T1 (Rust), T3 (Python), T2 (re-run + re-verify conclusions); "C++ winning categories — did we miss something, no tricks, follow flann exactly" → T4 (audit) + T5 (land) restricted to FIDELITY candidates only.
- T2 before T3: the Rust baseline is the reference for T4's re-ranking; Python harness can proceed independently after T1's schema exists (render_report shared) — order T1→T2→T3→T4→T5→T6, with T3 movable earlier if idle.
- Placeholders: none; T4 names concrete audit questions (allocator, build-loop vectorization, slot rebuild reuse) rather than "investigate performance".
