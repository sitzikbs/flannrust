# Task 2 report — full statistical baseline + conclusion re-verification

Branch `m2p6-rigor`, starting commit `a30a819` (M2.6 task 1, adaptive n=10..100
timing harness). This task ran that harness for real, across independent
sessions, and re-verified every conclusion (a)–(g) named in
`task-2-brief.md`. No library code was changed; the only diff outside docs
is `crates/xval/examples/m25_diag.rs`'s `RUNS: usize = 7 -> 15` constant
bump (plus its doc comment), as explicitly authorized by the task context.

## Per-conclusion verdict table

| # | Conclusion | Verdict | One-line evidence |
|---|---|---|---|
| (a) | M2.5 T3 dim-32 f32 win | **CORRECTED** | New median-of-15, 2-session range **0.986–1.062** straddles parity (old: 0.966–1.008, "parity-or-better"); gap-closing magnitude (1.423x→~1.0x) unaffected. |
| (b) | M2.5 T2 give-backs (radius ~0.83, dim8 f64 back to pre-M2.5 band) | **CORRECTED** | `radius` widens to **0.807–0.867** (n=100, 4 sessions); `dim8 f64` improves to **0.929–0.937** — better than the entire pre-M2.5 range, not merely back inside it. |
| (c) | Fixed-3 residual "~1–4%" | **CORRECTED** | New 4-session range **1.030–1.067** (3.0–6.7%), upper end exceeds old characterization; still inside the (also newly-characterized) noise floor. |
| (d) | dim-16 curse-of-dimensionality attribution | **CONFIRMED** | Median-of-15 re-run: f32 ratio 0.682/0.682 (both sessions); `frac_points_scanned` re-run byte-identical (0.0176→0.4875, 27.7x) — same mechanism. |
| (e) | The noise floor itself | **CORRECTED (replaced)** | New n=100/side, 4-session floor: session medians 1.030–1.067, per-session per-rep ratio σ≈0.03–0.05 (delta-method), approx. mean±2σ band 0.94–1.16 — supersedes the eyeballed 8-run 0.956–1.192. |
| (f) | `dyn_add` "1.11–1.24" | **CONFIRMED, narrowed** | 4-session range **1.112–1.149**, comfortably inside the old 1.106–1.237; old high-water mark (1.237) reads as small-sample noise in hindsight. |
| (g) | `build_100k` seq "0.974–1.139," suspiciously wide | **CORRECTED/SETTLED** | Genuinely dataset-dependent, not noise/drift: gate's own dataset → 0.967/0.967 both sessions; report_data's independently-seeded dataset → 1.037/1.037 both sessions. |

## Methodology note (deferred item closed)

`measure_pair`'s per-repetition interleave is confirmed fixed-order
(`rust_fn()` then `cpp_fn()`, every rep including warmups, never
alternated) — documented in `docs/EXPERIMENTS.md`'s new §4 methodology
bullet and the M2.6 task-2 subsection, with the accepted-risk framing the
brief requested (residual order bias absorbed by the 1.25 gate margin, no
alternating/randomized variant implemented).

## All pasted runs (source of truth: `docs/EXPERIMENTS.md` "M2.6 task 2" subsection)

### Six perf gates, session 1
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.558ms cpp=9.885ms ratio=0.967 | rust mean=9.596 std=0.170 n=100 | cpp mean=9.928 std=0.197 n=100 | ratio_medians=0.967
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.095ms cpp=3.589ms ratio=1.141 | rust mean=4.125 std=0.148 n=100 | cpp mean=3.636 std=0.128 n=100 | ratio_medians=1.141
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=31.123ms cpp=33.094ms ratio=0.940 | rust mean=32.682 std=9.345 n=100 | cpp mean=34.848 std=8.070 n=100 | ratio_medians=0.940
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.702ms cpp=7.462ms ratio=1.032 | rust mean=7.737 std=0.172 n=100 | cpp mean=7.479 std=0.165 n=100 | ratio_medians=1.032
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=181.210ms cpp=193.603ms ratio=0.936 | rust mean=182.745 std=9.476 n=100 | cpp mean=197.367 std=33.783 n=100 | ratio_medians=0.936
PERF_GATE perf_gate_radius_dim3_f32: rust=5.360ms cpp=6.631ms ratio=0.808 | rust mean=5.378 std=0.131 n=100 | cpp mean=6.658 std=0.128 n=100 | ratio_medians=0.808
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 51.33s
```

### Six perf gates, session 2 (separate invocation, minutes later)
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.639ms cpp=9.971ms ratio=0.967 | rust mean=9.653 std=0.086 n=100 | cpp mean=10.098 std=0.943 n=100 | ratio_medians=0.967
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.327ms cpp=3.766ms ratio=1.149 | rust mean=4.878 std=1.210 n=100 | cpp mean=4.280 std=1.082 n=100 | ratio_medians=1.149
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=30.758ms cpp=32.716ms ratio=0.940 | rust mean=30.810 std=0.475 n=100 | cpp mean=32.803 std=0.514 n=100 | ratio_medians=0.940
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.581ms cpp=7.358ms ratio=1.030 | rust mean=7.654 std=0.243 n=100 | cpp mean=7.419 std=0.229 n=100 | ratio_medians=1.030
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=182.580ms cpp=194.829ms ratio=0.937 | rust mean=183.661 std=4.335 n=100 | cpp mean=195.774 std=3.749 n=100 | ratio_medians=0.937
PERF_GATE perf_gate_radius_dim3_f32: rust=5.350ms cpp=6.633ms ratio=0.807 | rust mean=5.400 std=0.191 n=100 | cpp mean=6.649 std=0.124 n=100 | ratio_medians=0.807
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 51.04s
```

### Report chain, run 1
`meta`: date=2026-08-25T03:57:57Z, git_sha=a30a819-dirty (dirty only via
`m25_diag.rs`'s rep-count bump). `report.json` 9525 bytes, `report.html`
15988 bytes. Tiles: `Accuracy @ eps=0: ✓ 100%`, `Bit-exactness: ✓ (all
rows)`, `Best speed win: ✓ 1.56× faster — build_1M_dim3_f32_par`.

| workload | ratio (medians) |
|---|---|
| build_100k_dim3_f32_seq | 1.0386 |
| knn_dim3_f32_k10 | 1.0673 |
| knn_dyn_dim8_f64_k10 | 0.9291 |
| radius_dim3_f32 | 0.8663 |
| build_1M_dim3_f32_seq | 1.0446 |
| build_1M_dim3_f32_par | 0.6423 |
| dyn_add_20k_dim3_f32 | 1.1119 |
| dyn_knn_after_churn_dim3_f32 | 0.9449 |

Full `TimingStats` (mean/std/median/min/max, n=100 every row) pasted in
`docs/EXPERIMENTS.md`. All 11 accuracy rows `rust_eq_cpp_bitexact: true`.

### Report chain, run 2
`meta`: date=2026-08-25T03:59:43Z, git_sha=a30a819-dirty. `report.json`
9527 bytes, `report.html` 15990 bytes. Tiles: `Accuracy @ eps=0: ✓ 100%`,
`Bit-exactness: ✓ (all rows)`, `Best speed win: ✓ 1.57× faster —
build_1M_dim3_f32_par`.

| workload | ratio (medians) |
|---|---|
| build_100k_dim3_f32_seq | 1.0368 |
| knn_dim3_f32_k10 | 1.0614 |
| knn_dyn_dim8_f64_k10 | 0.9286 |
| radius_dim3_f32 | 0.8667 |
| build_1M_dim3_f32_seq | 1.0469 |
| build_1M_dim3_f32_par | 0.6355 |
| dyn_add_20k_dim3_f32 | 1.1241 |
| dyn_knn_after_churn_dim3_f32 | 0.9470 |

All 11 accuracy rows `rust_eq_cpp_bitexact: true` again. Note:
`dyn_knn_after_churn`'s max=117.220ms (vs. median 30.715ms) is the most
extreme single-rep outlier observed this task — ratio of medians (0.947)
unaffected.

### `m25_diag knn`, session 1 (RUNS bumped 7→15)
```
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.533,3.770,0.937
8,f64,3.590,3.852,0.932
16,f32,83.222,121.954,0.682
16,f64,99.278,105.414,0.942
32,f32,229.336,216.019,1.062
32,f64,391.067,357.525,1.094
64,f32,369.011,340.936,1.082
64,f64,910.355,1056.527,0.862
```

### `m25_diag knn`, session 2
```
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.555,3.742,0.950
8,f64,3.594,3.953,0.909
16,f32,83.430,122.412,0.682
16,f64,99.073,105.445,0.940
32,f32,219.269,222.296,0.986
32,f64,390.048,361.244,1.080
64,f32,508.171,411.479,1.235
64,f64,972.795,1017.689,0.956
```

### `m25_diag count` (deterministic — 2 sessions byte-identical, `diff` confirmed)
```
dim,leaf,scalar,queries,eval_per_query,interior_per_query,frac_points_scanned
3,10,f32,1000,66.0,29.1,0.0007
8,10,f32,1000,1756.7,540.4,0.0176
16,10,f32,1000,48748.2,9515.9,0.4875
32,10,f32,1000,99999.9,14438.7,1.0000
3,1,f32,1000,29.4,65.9,0.0003
3,4,f32,1000,43.3,41.5,0.0004
3,32,f32,1000,120.7,19.0,0.0012
3,128,f32,1000,288.5,11.8,0.0029
3,1024,f32,1000,1249.0,6.2,0.0125
```

## Doc locations touched

- `docs/EXPERIMENTS.md`: new "M2.6 task 2: statistical baseline ... " subsection
  under §3 (all pasted runs, the new noise-floor derivation, the
  conclusion-by-conclusion re-verification); §1's WSL2-noise-floor
  paragraph updated with an Update banner pointing here (old range kept as
  historical); §4 Methodology's "Median-of-7 timing" bullet rewritten to
  "Adaptive-n timing (M2.6, current)" including the interleave-order
  deferred item and the `m25_diag`-still-uses-median-of-15 note; §5
  provenance table gained 6 new rows.
- `docs/benchmarks.md`: noise-floor citation near the top (M1 section
  preamble); the "Update (M2.5-T2)" banner's noise-floor pointer; the
  M2.5 summary table's Update banner (new combined ranges); the dim-32/64
  table's "parity-or-better" correction; "Honest residuals" fixed-dim-3
  bullet's Update banner; the M-py perf-gate re-run's Update banner
  (build_100k dataset-dependence, radius widening); new top-level "## M2.6
  — statistical re-verification" section (combined-ranges table +
  per-conclusion verdicts + new-noise-floor callout), inserted before "##
  M-py — Python bindings".
- `docs/ROADMAP.md`: M2.5 milestone summary gained an Update banner; the
  fixed-dim-3 future-perf-lead bullet's noise-floor citation updated; the
  dim-16 lead bullet gained an M2.6 re-confirmation note; the
  `radius_dim3_f32` watch-item's "pre-announcement re-measure mandate" is
  struck through and marked satisfied-and-superseded with the new
  0.807–0.867 range; the M-pub bare-metal-re-run bullet's noise-floor
  citation updated; the M-pub blog-post-draft bullet's stale ~1-4%/~4.4%
  figures corrected with a pointer to the M2.6-verified ranges.
- `docs/nanoflann-notes.md`: "M2.5 outcome" section gained an "Update
  (M2.6, 2026-08-25)" paragraph after the historical pasted-number
  narrative (never edited in place), correcting dim-32 f32, radius, dim8
  f64, fixed-dim-3, and `dyn_knn_after_churn`, and flagging `build_100k`'s
  new dataset-dependence finding.
- `README.md`: "Perf gate" section gained a full "Update (M2.6...)" block
  with a second ratio-range table, corrected noise-floor footnote, and a
  rewritten T2-trade-off paragraph; "dim-32/64 knn" section gained an
  Update paragraph (parity-straddling correction); "Dynamic adaptor (M2)"
  → "Performance" section gained two Update sentences (`dyn_add` narrowing,
  `dyn_knn_after_churn` narrowing); the bottom "Roadmap" M2.5 bullet
  updated (parity-or-better → near-parity, ~1-4% → ~3-7%, pointer to the
  M2.6 sections).
- `crates/xval/examples/m25_diag.rs`: `RUNS: usize = 7` → `15`, module doc
  comment updated to explain the bump and the file's continued independence
  from the `measure`/`measure_pair` harness.

## Self-review

- Every ratio cited in the doc edits traces to one of the 6 pasted
  transcripts above (2 gate sessions, 2 report-chain runs, 2 `m25_diag
  knn` sessions), all reproduced verbatim in `docs/EXPERIMENTS.md`. No
  number was invented or extrapolated beyond what a session actually
  printed.
- The "noise floor" replacement is derived, not asserted: the delta-method
  formula is shown, its assumption (independence between per-rep rust/cpp
  timings) is stated explicitly and argued to be a conservative
  (upper-bound) assumption given `measure_pair`'s same-rep interleaving,
  and it's clearly distinguished from the JSON's existing
  `ratio_means_std` field (SEM of the mean estimate — a different,
  much-tighter quantity that would have been the wrong thing to publish as
  a "floor").
- `crates/xval/tests/perf_gate.rs` prints no `min_ms`/`max_ms` (only
  mean/std/median/n) — the noise-floor min/max discussion in
  `docs/EXPERIMENTS.md` correctly draws min/max only from the report-chain
  JSON runs, not from gate stdout, and says so.
- (g)'s dataset-dependence finding is honestly bounded: I confirmed the
  gate and report_data binaries use different `cfg_seed` tags for
  `build_100k` (`"perf_gate_build"` vs. `"report_build_100k"`) and that
  the two clusters are internally consistent (<1% spread within a harness,
  ~7pp apart between harnesses) — but I explicitly did NOT claim to have
  isolated "different dataset" from "different process context" (gate =
  standalone single-workload binary; report_data = one process running 8
  workloads sequentially) as the sole cause, since that would require a
  controlled experiment out of this task's scope.
- Historical pasted evidence blocks (M1/M2/M2.5 tables and transcripts)
  were never edited in place anywhere — every correction is a new
  "Update (M2.6...)" paragraph/banner/table placed adjacent to the old
  text, per house style.
- Verified `cargo build --workspace --release` and
  `cargo test --workspace --all-features` both green after the
  `m25_diag.rs` edit (the only code change).
- `git status --porcelain` confirmed clean except `crates/xval/examples/m25_diag.rs`
  before capturing the report-chain runs, so the `git_sha=a30a819-dirty`
  meta field is accurately described (dirty only via the doc-comment +
  RUNS-constant change, not via any timed code path).

## Concerns

- `m25_diag`'s dim-32/dim-64 rows show more session-to-session swing than
  dim-3/dim-8 anywhere else in this document (dim-32 f32: 0.986–1.062;
  dim-64 f32: 1.082–1.235) — plausibly cumulative thermal drift across the
  ~67–70s single-process sweep (dim-64 runs last), not investigated
  further (out of scope; flagged in `docs/EXPERIMENTS.md`). This directly
  drove conclusion (a)'s correction and is worth a closer look in a future
  perf task if dim-32/64 numbers are going into a public announcement.
- The `measure_pair` fixed rust-then-cpp interleave order remains an
  accepted, unmitigated methodological risk (per the brief's framing) —
  no evidence of a resulting bias was found in this task's data, but no
  alternating/randomized-order variant was implemented either, since that
  would be a harness change outside this task's scope.
- (g)'s "dataset-seed vs. process-context" ambiguity (see self-review)
  is flagged but not resolved — a genuinely interesting follow-up if
  `build_100k`'s exact number matters for an announcement.
- `crates/xval/tests/perf_gate.rs`'s stdout format doesn't print
  `min_ms`/`max_ms`, so the noise-floor characterization's min/max
  discussion leans on the report-chain JSON only (2 of the 4 sessions) —
  noted explicitly in the docs, not silently generalized to all 4.

---

## Fix round 1/5 (controller review response)

Four findings addressed, all confirmed fixed by grep sweep + tests:

1. **M2 dynamic-forest section missing M2.6 pointer** (`docs/benchmarks.md`
   "M2 — dynamic forest" section, README's designated canonical source for
   `dyn_add`/`dyn_knn_after_churn`): added a short "Update (M2.6, commit
   `a30a819`, 2026-08-25)" banner right after the historical two-row table
   — `dyn_add` **1.112–1.149**, `dyn_knn_after_churn` **0.940–0.947**,
   pointer to `docs/EXPERIMENTS.md` "M2.6 task 2" and the "M2.6 —
   statistical re-verification" section. Historical table left untouched.

2. **Verdict (g) overclaim** ("settled"/"genuinely dataset-dependent"/"NOT
   primarily measurement noise" stated unhedged in six spots — the review's
   count included `docs/benchmarks.md:522,675,714/715,827`, `README.md:353,374`,
   `docs/ROADMAP.md:57/58`): reworded every one of the six (plus
   `docs/nanoflann-notes.md`'s equivalent instance, same pattern, fixed for
   consistency) to state plainly that the split is reproducible/
   configuration-dependent — NOT noise or drift — while explicitly
   admitting dataset-seed and harness/process-context were never crossed
   against each other, so which one is the actual cause is not isolated.
   `docs/EXPERIMENTS.md` conclusion (g) itself (the source of truth) was
   rewritten with the same hedge stated up front (verdict tag downgraded
   from "CORRECTED/SETTLED" to plain "CORRECTED") rather than only buried
   in its closing sentence. Added an explicit "concrete Task-4-audit
   input" note in `docs/ROADMAP.md` (inline in the M2.5 Update banner,
   ~line 62): a crossing experiment — swap which dataset each binary
   builds — would isolate dataset-seed from process-context.
3. **`1.037/1.037` self-contradiction**: the two report-chain session
   ratios (1.038617, 1.036801) round to **1.039** and **1.037**
   respectively, not two identical `1.037`s. Fixed at all 4 flagged
   locations (`docs/EXPERIMENTS.md`'s conclusion (g) and its §5 provenance
   row, `docs/benchmarks.md`'s "M2.6 — statistical re-verification"
   section, `docs/ROADMAP.md`'s M2.5 Update banner) to read `1.039/1.037`
   (or `1.037–1.039` in prose form).
4. **"mean±2σ band 0.94–1.16" mislabel**: verified the actual arithmetic
   behind the published 0.94–1.16 figure is `min(session medians) -
   2×max(per-session σ)` to `max(session medians) + 2×max(per-session σ)`
   = `1.030 - 2×0.046 = 0.938 ≈ 0.94` to `1.067 + 2×0.046 = 1.159 ≈ 1.16` —
   i.e. exactly the "extreme session medians ± 2×max per-session σ"
   formula the reviewer suggested as option A, not a grand mean±2σ over a
   pooled distribution. Chose option A (relabel, no recompute needed since
   the existing 0.94–1.16 figure already matches that formula exactly) —
   renamed to "conservative envelope (extreme session medians ± 2×max
   per-session σ)" consistently across all 8 occurrences in
   `docs/EXPERIMENTS.md`, `docs/benchmarks.md`, `docs/ROADMAP.md`, and
   `README.md`, each with an explicit "not a grand mean±2σ" disclaimer
   where the old label had been used.

**Verification**: `grep -rn "settled\|genuinely dataset\|SETTLED"` and
`grep -rn "1\.037/1\.037"` across `docs/` + `README.md` both return empty.
`cargo test -p xval --lib`: **90 passed, 0 failed**. `cargo clippy
--workspace --all-targets -- -D warnings`: clean.

Commit: `docs: fix M2.6 review findings — M2 section banner, build_100k
hedge, stat labels` (see `git log` for hash).
