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

Citations below are anchored to file + nearest **unique** section heading
(and, where a specific figure matters, a short verbatim quote of the
surrounding text) rather than to raw line numbers — line numbers drift
with every commit (fix round 1 shipped citations that were already stale
by the time they landed, because the edits that introduced them shifted
the very lines they pointed at). Every anchor below was verified with
`grep -c` against the current worktree, one hit per anchor; see fix round
2's note in `docs/agentic-development/reports/m-pub/task-5-report.md` for the pasted evidence.

| # | Claim (file, § heading / quote) | Source run | Verdict |
|---|---|---|---|
| 1 | README § "How this was built" (new, this task) | N/A — provenance/attribution text, not a number | N/A |
| 2 | README § "Perf gate (six gated workloads; pass threshold is ratio ≤ 1.25)", quote `"Post-M2.5** (T4's fresh two-run sweep"` — the "Post-M2.5" six-gate table (`build_100k` 0.974–1.079, `knn_fixed3` 1.011–1.040, `dim8` 0.950–0.966, `radius` 0.827–0.828, `dyn_add` 1.121–1.170, `dyn_knn_after_churn` 0.873–0.916) | `docs/EXPERIMENTS.md` "M2.5 task 4"; explicitly labeled "pasted evidence, never edited in place" and superseded inline by claim 3 | TRACED (historical, correctly labeled superseded) |
| 3 | README § "Perf gate (six gated workloads; pass threshold is ratio ≤ 1.25)", quote `"Update (M2.6, commit \`a30a819\`, 2026-08-25):"` — the "Update (M2.6...)" six-gate table (task-2 ranges, then task-6-corrected `build_100k` 0.993–1.009 inline) | `docs/EXPERIMENTS.md` "M2.6 task 2" + "M2.6 task 6" | TRACED — matches `docs/benchmarks.md`'s equivalent table exactly; consistent with fresh `9cac562` (`build_100k` 0.999, `knn_dim3` 1.013, `dim8` 0.927, `build_1M_seq`/`_par` 0.994/0.604 all inside or touching their recorded ranges) |
| 4 | README § "Perf gate (six gated workloads; pass threshold is ratio ≤ 1.25)", quote `"re-confirms every range above"` (the "Update (M2.6 task 6)" paragraph + its 6-row table); `docs/benchmarks.md` § "M2.6 task 6 — final regression sweep, `build_100k` re-hedge, and success-criteria verdict", quote `"**Final six-gate table**"` — M2.6 task 6 final six-gate table (`build_100k` 0.993–1.009, `knn_fixed3` 1.011–1.040, `dim8` 0.927–0.941, `radius` 0.831–0.868, `dyn_add` 1.0315–1.039, `dyn_knn_after_churn` 0.931–0.958) | `docs/EXPERIMENTS.md` "M2.6 task 6", full pasted `PERF_GATE`/`report.json` sessions | TRACED. Re-verified the README half on fix round 1 (both prior line-number citations — round 0's README:414–436, round 1's README:443–452 — were confirmed loose/off by the round-2 review): README's six ranges match `docs/benchmarks.md`'s table exactly, row for row, both located by heading/quote anchor as above. Fresh `9cac562` `radius` (0.876) and `dyn_add` (1.025) sit ~1% and ~0.6% outside these ranges respectively — both workloads have shown session-to-session spread of this size or larger throughout M2.6 (radius alone has ranged 0.767–0.876 across the milestone's sessions); read as ordinary noise consistent with the document's own repeated "session spread, not a regression" framing, not a contradiction requiring rewrite |
| 5 | README § "dim-32/64 knn: the M1-era gap is closed at f32 (M2.5)"; `docs/benchmarks.md` § "dim-32/64 headline: the M1-era gap is closed at f32" — both anchored additionally at the shared quote `"Update (M2.6 task 7, user-directed idle-host re-measurement):"` (present, verified unique, in each file) — dim-32/64 knn table + "Honest residuals" (dim32 f32 0.986–1.062, dim32 f64 1.080–1.104, dim64 f32 1.082–1.235, dim64 f64 0.862–0.956) | `docs/EXPERIMENTS.md` "M2.6 task 7", cross-referenced against "M2.6 task 2" | TRACED — both files carry the identical M2.6-task-7-corrected figures |
| 6 | README § "`leaf_max_size` sweep"; `docs/benchmarks.md` § "`leaf_max_size` sweep" (same heading text, each unique within its own file) — single 10-sample criterion spot check, `leaf_max_size ∈ {1,4,...,1024}` | `docs/EXPERIMENTS.md` "3. Reproduction commands" (`bench_build.rs`'s `leaf_sweep` groups) | TRACED — explicitly labeled a single spot check throughout, never presented as a headline/gated figure, not superseded by anything later |
| 7 | `docs/benchmarks.md` § "Parallel build" (M1 section), "roughly 1.75x faster" headline | `docs/EXPERIMENTS.md`'s "M2.5 task 4"-era pasted run only; never carried an "Update (M2.6...)" banner unlike every other headline figure in this document | **STALE (fixed this task).** `docs/EXPERIMENTS.md` § "M2.6 task 6" already has a pasted `n=100` re-measurement of the same `build_1M_dim3_f32_par` workload (0.6151, i.e. ~1.63x) that was never surfaced here, and the fresh `9cac562` run confirms it (0.604, ~1.66x). Both are inside the 1.47–1.58x band `docs/EXPERIMENTS.md` § "The two-command report chain" already documents for this workload under its lighter methodology — not a new finding, just never reconciled against this section specifically. Fixed by adding a paragraph anchored at the quote `"Update (M2.6 task 6 / M-pub, \`n=100\`-per-side re-measurement"` right after the existing pasted table (kept untouched per house style) directing readers to cite 1.6–1.66x, not 1.75x, going forward. |
| 8 | `docs/benchmarks.md` § "Headline table (criterion, spot-check methodology, `--quick`)" — M1 headline criterion tables (`knn_fixed3` k=1/10/100, `build_fixed3`, dim-8/32 knn) | `docs/EXPERIMENTS.md` "3. Reproduction commands" (full criterion sweep) | TRACED — historical M1 record, explicitly superseded inline by the M2.5/M2.6 sections that follow (dim-32 gap closure and fixed-dim-3 residual sections, further down the same file) |
| 9 | `docs/benchmarks.md` § "M2.6 — statistical re-verification" (all conclusion-by-conclusion (a)–(g) verdicts) | `docs/EXPERIMENTS.md` "M2.6 task 2"/"task 5"/"task 6" | TRACED — every row cites its session count and links to the pasted evidence |
| 10 | README § "Benchmarks" (under "Python bindings (M-py)"); `docs/benchmarks.md` § "Success criteria vs. spec (honest accounting, both bench runs)" and § "Honest misses (not gated, published alongside per this repo's standing "publish the losses too" convention)" — M-py benchmark tables (cKDTree/pynanoflann ratios, per-call overhead, honest misses) | `docs/EXPERIMENTS.md` "M-py"/"M2.6 task 3"/"M2.6 task 6" subsections | TRACED — README and `docs/benchmarks.md` figures match exactly |
| 11 | README § "Safety (`unsafe` in this crate)", quote `"+4.7%** on the fixed-dim-3 knn gate"` — the `unsafe`-necessity A/B figure | `docs/EXPERIMENTS.md` "M2.5 task 2" | TRACED (in EXPERIMENTS.md's Number Provenance table) |
| 12 | `docs/EXPERIMENTS.md` § "The two-command report chain", quote `"Reconciling this tile's 1.47"` — report-chain scorecard tile reconciliation ("1.47–1.58x" vs. "roughly 1.75x") | Same section — this is the primary source, not a secondary citation | TRACED — already an honest reconciliation; claim 7's fix aligns `docs/benchmarks.md`'s M1 headline text with this existing reconciliation rather than duplicating it. **Correction from fix round 1: this claim's file was previously misattributed as `docs/benchmarks.md`; the reconciliation text has only ever lived in `docs/EXPERIMENTS.md`** — `docs/benchmarks.md` never contained it, in any round. |

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
