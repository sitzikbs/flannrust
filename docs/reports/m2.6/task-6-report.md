# Task 6 report: docs close-out + regression sweep (final M2.6 task)

Date: 2026-08-25. Branch `m2p6-rigor`, started at HEAD `b9335fd`, ended at
`2a95b82`. Environment: WSL2, `export PATH="$HOME/.cargo/bin:$PATH"`,
`RUSTFLAGS="-C target-cpu=native"`, cargo/rustc 1.98.0. `git status
--porcelain` clean at the end. No library code changed — docs
(`README.md`, `docs/EXPERIMENTS.md`, `docs/benchmarks.md`,
`docs/ROADMAP.md`, `docs/nanoflann-notes.md`) plus a meta-only addition to
`crates/flannrust-py/python/bench/bench_py.py` (`loadavg_start`/
`loadavg_end`), exactly matching the allowed file scope.

## Status: COMPLETE

## Commit

`2a95b826d0ae663b7fd717cfc3fc7b8249061d1c` — `docs: M2.6 results —
statistical baseline and fidelity fixes` (single commit; 6 files, +647/-16).

## What was done

1. **Final regression sweep, idle host** (loadavg captured before/after
   every session; host stayed at or below 2.10 1-min average throughout,
   `nproc=8`): full perf-gate suite x2 sessions, report chain
   (`report_data` + `render_report`) x1, full `bench_py.py` run x1
   (`RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release`
   rebuilt first). All six gates land inside or immediately adjacent to
   their M2.6 task 2/5 recorded bands: `build_100k` 0.993–1.009,
   `knn_fixed3` 1.011–1.040, `dim8` 0.927–0.941, `radius` 0.831–0.868,
   `dyn_add` 1.0315–1.039 (confirms the task 5 fidelity fix holds on a
   third independent sweep), `dyn_knn_after_churn` 0.931–0.958. Full
   pasted evidence: `docs/EXPERIMENTS.md` new "M2.6 task 6" subsection;
   tables in `docs/benchmarks.md`'s new "M2.6 task 6" section.
2. **Python outlier resolution**: the flagged
   `knn_batched_dim3_f32_..._workers1` cell (0.493 vs cKDTree, 1.009 vs
   pynanoflann) was re-measured fresh on the idle host. Result: 0.688 /
   1.055 — cKDTree's own absolute median (423.19ms) landed right back in
   its historical 426–430ms band, decisively away from the flagged
   session's 698.9ms. **Verdict: CONFIRMED host-load contamination, not a
   new steady state.** Updated every doc spot carrying the caveat
   (README, benchmarks.md x2, EXPERIMENTS.md x3 incl. inline pointers at
   conclusions (a)/(g), ROADMAP.md) with "Update (M2.6 task 6)" banners —
   historical pasted blocks left untouched, honest combined ranges
   (0.493–0.830 / 1.009–1.216) kept with the outlier point labeled, not
   deleted.
3. **T2 conclusion-(g) re-hedge**: this task's own fresh 3-session sweep
   independently corroborates T4's non-reproduction finding a second way —
   both gate sessions measured `build_100k` at 1.009/1.009 and the
   report-chain session at 0.9926, the **opposite** ordering from T2's
   original 0.967-vs-1.037/1.039 split, all three clustered tightly at
   0.993–1.009. Reworded everywhere the split claim appeared: README.md
   (3 spots), `docs/benchmarks.md` (3 spots + a new dedicated section),
   `docs/EXPERIMENTS.md` (conclusion (g) itself + the M2.6-task-2-update
   paragraph + provenance table), `docs/ROADMAP.md`, and
   `docs/nanoflann-notes.md` — all via appended "Update (M2.6 task 6)"
   banners, never editing the historical numbers/prose in place.
4. **Measurement-conditions protocol**: added to `docs/EXPERIMENTS.md` §1
   (the WSL2 noise-caveat subsection) — idle-host requirement, the
   2026-08-25 game-load incident as motivating example, the
   `LOADAVG_BEFORE`/`LOADAVG_AFTER` convention generalized to a standing
   protocol. Added `_loadavg()` + `meta.loadavg_start`/`loadavg_end` to
   `bench_py.py` (verified via `ruff check` + `ast.parse`; this task's own
   full bench run predates the code addition, so its loadavg was captured
   manually via shell instead — documented as such, not re-run for 23
   minutes just to populate the new JSON field).
5. **dim8 hedge (deferred minor from task 5)**: this task's own fresh dim8
   range (0.927–0.941) straddles the recorded 0.929–0.937 band on both
   sides by margins comparable to or larger than task 5's flagged 0.939 —
   corroborates that single-session ±0.2–0.4pp dim8 excursions are
   ordinary noise. Hedged in both `docs/EXPERIMENTS.md` and
   `docs/benchmarks.md`.
6. **benchmarks.md M2.6 section**: new "M2.6 task 6" section with final
   six-gate table (mean±std+median for every session), the `build_100k`
   re-hedge, the dim8 hedge, the Python outlier resolution, and a
   success-criteria verdict table. `docs/ROADMAP.md` marks M2.6 complete,
   updates the M-pub bare-metal item (now references the new protocol)
   and the dim8-vs-pynanoflann item (now carries T4's finding that
   nanoflann 1.5.5 is ~5–6% slower than 1.12.1 on this exact workload,
   narrowing the open question to the Python binding layer). The M2.5
   future-perf-leads list gets T4's dim-32/64 compiler-codegen REJECT
   rationale (LLVM AVX2 vs GCC AVX-512, asm-verified, not a Rust defect).
7. **Success-criteria verdict**: table in `docs/benchmarks.md`'s "M2.6
   task 6" section — statistical methodology (both Rust and Python sides),
   fidelity audit with two landed fixes, evidence-backed REJECTs, and the
   two resolved outliers, each row citing its evidence.
8. **Final hygiene sweep** (all green):
   - `cargo test --workspace`: 198+23+12+90+2+3+19+5+11+13+12+2 passed, 0
     failed across all binaries.
   - `cargo test -p flannrust --release --lib`: 197 passed, 0 failed, 4
     ignored — the T5 release-profile test-gating fix holds.
   - `cargo clippy --workspace --all-targets -- -D warnings`: clean.
   - `RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust -p xval -p
     nanoflann-ref --no-deps`: clean.
   - `cargo build -p flannrust --no-default-features`: clean.
   - `.venv/bin/python -m pytest python/tests -q`: 351 passed, 27 xfailed,
     1 xpassed.
   - `ruff check python/bench/bench_py.py`: all checks passed (both
     before and after the loadavg-capture addition).

## Self-review / concerns

- **Scope discipline**: only docs + `bench_py.py`'s meta-only addition
  touched (`git diff --stat` confirms: README.md, docs/EXPERIMENTS.md,
  docs/benchmarks.md, docs/ROADMAP.md, docs/nanoflann-notes.md,
  crates/flannrust-py/python/bench/bench_py.py — no `crates/flannrust`,
  `crates/xval`, or `crates/flannrust-py/src` changes). `git diff` grep
  for removed lines confirms no historical pasted data was deleted, only
  extended in place (git's line-diff shows old single lines as "removed"
  only because the appended text made them wrap into new lines).
- **The `bench_py.py` loadavg meta fields were added but not exercised by
  a fresh full run** (the ~23-minute full bench for this task's own
  outlier-resolution evidence ran before the code change). Documented
  honestly in `docs/EXPERIMENTS.md`'s "M2.6 task 6" subsection: this
  run's loadavg was captured manually via shell (pasted), and the new
  meta fields apply starting the next `bench_py.py` invocation. Re-running
  the full bench solely to populate the JSON field was judged not worth
  another 23 minutes given the shell-captured values are already pasted
  and equally load-bearing.
- **`ratio_ckdtree` for the resolved outlier cell landed at 0.688, not
  fully back inside the old 0.712–0.830 band** (3.4% below its low end).
  Reported this honestly rather than rounding it into "back to normal" —
  the resolution verdict rests on cKDTree's *absolute* median returning to
  its historical band (423.19ms vs. 426–430ms) plus flannrust itself
  running a touch faster than its own historical band this session, not
  on the ratio landing exactly inside the old bounds.
- **No open concerns.** Every item in the brief and the ledger's deferred
  minors was addressed; hygiene sweep fully green; working tree clean
  after the commit.

## Return contract

- **Status**: COMPLETE
- **Commit hash(es)**: `2a95b826d0ae663b7fd717cfc3fc7b8249061d1c`
- **One-line test summary**: `cargo test --workspace` all green (0
  failed), `cargo test -p flannrust --release --lib` 197/0/4-ignored
  (T5 gating fix holds), `pytest` 351 passed/27 xfailed/1 xpassed,
  `clippy -D warnings` clean, `cargo doc -D warnings` clean,
  `--no-default-features` build clean, `ruff check bench_py.py` clean.
- **Outlier-resolution verdict**: CONFIRMED host-load contamination — the
  flagged batched-knn-dim3-workers=1 outlier (0.493 vs cKDTree, 1.009 vs
  pynanoflann) does not reproduce on a fresh idle-host run (0.688/1.055,
  cKDTree's absolute median back in its historical band), and
  `build_100k`'s "reproducibly split by configuration" claim likewise does
  not reproduce (this task's 3 fresh sessions cluster at 0.993–1.009
  regardless of configuration) — both re-hedged everywhere they were cited.
- **Concerns**: none blocking; two honest caveats noted above (the
  loadavg meta fields weren't exercised by a fresh run; the resolved
  cKDTree ratio sits just outside, not fully inside, its old band).
