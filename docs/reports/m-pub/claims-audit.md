# M-pub task 5: claims audit

Every performance-number claim (ratio, ms/µs/ns, ×, %) in `README.md` and
`docs/benchmarks.md` was enumerated and checked against `docs/EXPERIMENTS.md`
(source of truth — its "5. Number provenance" section already maps almost
every figure in both files to a reproducible command + output field) and
against the fresh post-merge report at commit `9cac562` (core ratios:
`build_100k` 0.999, `knn_dim3` 1.013, `knn_dyn_dim8_f64` 0.927, `radius`
0.876, `build_1M_seq` 0.994, `build_1M_par` 0.604, `dyn_add` 1.025,
`dyn_churn` 0.953 — Rust/C++ medians, `n=100`).

Numbers are grouped by claim/section rather than by literal digit — most
sections carry a table plus several ratios that all trace to the same
run, and this repo already carries a mature "Update (M2.6 task N)" banner
convention layering fresher measurements onto pasted evidence without
editing it in place. This audit re-uses that convention rather than
fighting it.

## Verdicts

TRACED = the claim points (directly or via a nearby "Update" banner) to a
reproducible command + pasted output in `docs/EXPERIMENTS.md`, and is
consistent with the latest recorded measurement and the fresh `9cac562`
check. STALE = the claim is presented as current but a fresher,
already-recorded measurement contradicts or supersedes it without
disclosure. UNTRACEABLE = no reproducible source exists anywhere in this
repo.

| # | Claim (file:line) | Source run | Verdict |
|---|---|---|---|
| 1 | README "How this was built" (new, this task) | N/A — provenance/attribution text, not a number | N/A |
| 2 | README:346–359 six-gate table, "Post-M2.5" (`build_100k` 0.974–1.079, `knn_fixed3` 1.011–1.040, `dim8` 0.950–0.966, `radius` 0.827–0.828, `dyn_add` 1.121–1.170, `dyn_knn_after_churn` 0.873–0.916) | `docs/EXPERIMENTS.md` "M2.5 task 4"; explicitly labeled "pasted evidence, never edited in place" and superseded inline by claim 3 | TRACED (historical, correctly labeled superseded) |
| 3 | README:363–372 "Update (M2.6...)" six-gate table (task-2 ranges, then task-6-corrected `build_100k` 0.993–1.009 inline) | `docs/EXPERIMENTS.md` "M2.6 task 2" + "M2.6 task 6" | TRACED — matches `docs/benchmarks.md`'s equivalent table exactly; consistent with fresh `9cac562` (`build_100k` 0.999, `knn_dim3` 1.013, `dim8` 0.927, `build_1M_seq`/`_par` 0.994/0.604 all inside or touching their recorded ranges) |
| 4 | README:414–436, `docs/benchmarks.md`:826–847 M2.6 task 6 final six-gate table (`build_100k` 0.993–1.009, `knn_fixed3` 1.011–1.040, `dim8` 0.927–0.941, `radius` 0.831–0.868, `dyn_add` 1.0315–1.039, `dyn_knn_after_churn` 0.931–0.958) | `docs/EXPERIMENTS.md` "M2.6 task 6", full pasted `PERF_GATE`/`report.json` sessions | TRACED. Fresh `9cac562` `radius` (0.876) and `dyn_add` (1.025) sit ~1% and ~0.6% outside these ranges respectively — both workloads have shown session-to-session spread of this size or larger throughout M2.6 (radius alone has ranged 0.767–0.876 across the milestone's sessions); read as ordinary noise consistent with the document's own repeated "session spread, not a regression" framing, not a contradiction requiring rewrite |
| 5 | README:472–526, `docs/benchmarks.md`:595–689 dim-32/64 knn table + "Honest residuals" (dim32 f32 0.986–1.062, dim32 f64 1.080–1.104, dim64 f32 1.082–1.235, dim64 f64 0.862–0.956) | `docs/EXPERIMENTS.md` "M2.6 task 7", cross-referenced against "M2.6 task 2" | TRACED — both files carry the identical M2.6-task-7-corrected figures |
| 6 | README:527–533, `docs/benchmarks.md`:278–316 `leaf_max_size` sweep (single 10-sample criterion spot check, `leaf_max_size ∈ {1,4,...,1024}`) | `docs/EXPERIMENTS.md` "3. Reproduction commands" (`bench_build.rs`'s `leaf_sweep` groups) | TRACED — explicitly labeled a single spot check throughout, never presented as a headline/gated figure, not superseded by anything later |
| 7 | `docs/benchmarks.md`:317–345 M1 "Parallel build" section, "roughly 1.75x faster" headline | `docs/EXPERIMENTS.md`'s "M2.5 task 4"-era pasted run only; never carried an "Update (M2.6...)" banner unlike every other headline figure in this document | **STALE (fixed this task).** `docs/EXPERIMENTS.md` "M2.6 task 6" already has a pasted `n=100` re-measurement of the same `build_1M_dim3_f32_par` workload (0.6151, i.e. ~1.63x) that was never surfaced here, and the fresh `9cac562` run confirms it (0.604, ~1.66x). Both are inside the 1.47–1.58x band the same file's "two-command report chain" section already documents for this workload under its lighter methodology — not a new finding, just never reconciled against this section specifically. Fixed by adding an "Update (M2.6 task 6 / M-pub)" paragraph (kept the original pasted table/text untouched per house style) directing readers to cite 1.6–1.66x, not 1.75x, going forward. |
| 8 | `docs/benchmarks.md`:91–159 M1 headline criterion tables (`knn_fixed3` k=1/10/100, `build_fixed3`, dim-8/32 knn) | `docs/EXPERIMENTS.md` "3. Reproduction commands" (full criterion sweep) | TRACED — historical M1 record, explicitly superseded inline by the M2.5/M2.6 sections that follow (dim-32 gap closure at line 148, fixed-dim-3 residual superseded at line 199–213) |
| 9 | `docs/benchmarks.md`:723–911 "M2.6 — statistical re-verification" (all conclusion-by-conclusion (a)–(g) verdicts) | `docs/EXPERIMENTS.md` "M2.6 task 2"/"task 5"/"task 6" | TRACED — every row cites its session count and links to the pasted evidence |
| 10 | README:729–788, `docs/benchmarks.md`:933–1058 M-py benchmark tables (cKDTree/pynanoflann ratios, per-call overhead, honest misses) | `docs/EXPERIMENTS.md` "M-py"/"M2.6 task 3"/"M2.6 task 6" subsections | TRACED — README and `docs/benchmarks.md` figures match exactly |
| 11 | README "Safety" section, "+4.7%" `unsafe`-necessity A/B (README:167–169) | `docs/EXPERIMENTS.md` "M2.5 task 2" | TRACED (in EXPERIMENTS.md's Number Provenance table) |
| 12 | `docs/benchmarks.md`:1088–1119 report-chain scorecard tile reconciliation ("1.47–1.58x" vs. "roughly 1.75x") | `docs/EXPERIMENTS.md` "The two-command report chain" | TRACED — already an honest reconciliation; claim 7's fix aligns the M1 headline text with this existing reconciliation rather than duplicating it |

## What was fixed

- **`docs/benchmarks.md`'s "Parallel build" section** (claim 7): added an
  "Update (M2.6 task 6 / M-pub)" paragraph after the existing pasted table,
  citing the `n=100` `build_1M_dim3_f32_par` re-measurement (0.6151, M2.6
  task 6) and the fresh `9cac562` confirmation (0.604), and stating plainly
  that 1.6–1.66x, not 1.75x, is the figure to publish going forward. The
  original pasted table/text is unchanged, per this repo's "pasted evidence
  is never edited in place" convention.
- **README.md "How this was built"** (new section, between "Attribution &
  license" and "Quickstart"): states that this codebase was implemented by
  an AI agent (Claude Code) directed and reviewed by Itzik Ben-Shabat, who
  does not write Rust, and that correctness rests on the bit-exact
  cross-validation suite against vendored nanoflann 1.12.1, not on the
  author's Rust expertise. No other prose in either file was rewritten.

## No UNTRACEABLE claims found

Every performance number in both files traces to a command + pasted output
in `docs/EXPERIMENTS.md`, most of them indexed directly in that file's own
"5. Number provenance" table. No number was deleted.

## Verification

- `grep -rn "superpowers/sdd" README.md docs/benchmarks.md` — empty, both
  before and after this task's edits.
- `cargo test --workspace` — run once at the end of this task; see the
  task report for the pasted result.
