# Task 7 report: user-directed idle-host re-measurement of the dim sweep and the noise envelope

Date: 2026-08-25. Branch `m2p6-rigor`, started at HEAD `2a95b826d0ae663b7fd717cfc3fc7b8249061d1c`
(task 6's final commit), ends at the commit listed below. Environment:
WSL2, `export PATH="$HOME/.cargo/bin:$PATH"`, `RUSTFLAGS="-C
target-cpu=native"`, cargo/rustc 1.98.0, AMD Ryzen 7 9800X3D, `nproc=8`.
`git status --porcelain` clean at the end except this task's own intended
changes. No library code changed — docs only (`README.md`,
`docs/EXPERIMENTS.md`, `docs/benchmarks.md`, `docs/ROADMAP.md`), per the
brief's "Docs + no code changes (m25_diag/perf_gate run as-is)" rule.

## Status: COMPLETE

## Commit

`b2a5e18515f61bca2f5e943b10eeae759c0280b9` — `docs: M2.6 task 7 —
idle-host re-measurement of dim sweep and noise envelope` (single commit;
4 files, +406/-13, all deletions verified as line-wrap artifacts, no
historical content removed).

## Background

The user reported a game consumed host compute during the T2/T3/T4
measurement window. T6 already re-ran gates ×2, the report chain, and the
Python bench on the now-idle host and resolved two flagged anomalies. Two
data sets remained un-re-measured, both traceable to T2's own sessions
(the window in question): the `m25_diag knn` dim-8/16/32/64 sweep, and
the `knn_fixed3` noise envelope (0.94–1.16). This task re-runs both fresh
on the idle host, brackets every session with `/proc/loadavg`, and states
each affected figure CONFIRMED or CORRECTED. A third, unrelated cosmetic
fix deferred from task 6 (README.md's `build_100k` paragraph missing an
adjacent hedge) was also picked up.

## Item 1: `m25_diag knn` dim sweep — 2 fresh idle-host sessions

Command: `RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release
--example m25_diag -- knn` (n=100k, 200 queries, k=10, leaf=10, median of
15 — same methodology as task 2, `RUNS` unchanged).

**Session 1:**

```
LOADAVG_BEFORE: 0.33 0.34 0.51 2/643 611708
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.272,3.478,0.941
8,f64,3.430,3.699,0.927
16,f32,77.735,115.239,0.675
16,f64,91.149,97.875,0.931
32,f32,208.988,204.722,1.021
32,f64,347.695,315.004,1.104
64,f32,408.679,344.589,1.186
64,f64,674.980,764.261,0.883
LOADAVG_AFTER: 1.20 0.59 0.58 2/641 612547
```

**Session 2** (minutes later, separate invocation):

```
LOADAVG_BEFORE: 0.66 0.52 0.56 2/646 612633
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.593,3.834,0.937
8,f64,3.385,3.636,0.931
16,f32,76.919,114.298,0.673
16,f64,90.335,97.243,0.929
32,f32,205.979,204.475,1.007
32,f64,351.356,319.460,1.100
64,f32,320.759,292.786,1.096
64,f64,625.639,674.282,0.928
LOADAVG_AFTER: 0.87 0.60 0.58 1/646 613249
```

Both sessions' loadavg stayed ≤1.20 (`nproc=8`) — the 1.20 reading is
this run's own compile+sweep (the same "brief post-run scheduler blip"
pattern task 5/6 documented), not foreign load.

### Per-row comparison and verdicts

| dim | scalar | task 2 (2 sessions, contamination-suspected) | task 7 (2 fresh idle-host sessions) | combined (4 sessions) | currently-published headline | **verdict** |
|---|---|---|---|---|---|---|
| 8 | f32 | 0.937–0.950 | 0.937–0.941 | 0.937–0.950 | none (diagnostic row only) | **CONFIRMED** |
| 8 | f64 | 0.909–0.932 | 0.927–0.931 | 0.909–0.932 | none | **CONFIRMED** |
| 16 | f32 | 0.682/0.682 | 0.673/0.675 | 0.673–0.682 | mechanism-only, conclusion (d) | **CONFIRMED** |
| 16 | f64 | 0.940/0.942 | 0.929/0.931 | 0.929–0.942 | mechanism-only, conclusion (d) | **CONFIRMED** |
| 32 | f32 | 0.986–1.062 | 1.007–1.021 | 0.986–1.062 | "0.986–1.062, straddles parity" (conclusion (a)) | **CONFIRMED** |
| 32 | f64 | 1.080–1.094 | 1.100–1.104 | **1.080–1.104** | "1.103–1.111" / "~1.10–1.11x" (T4, pre-M2.6) | **CORRECTED** |
| 64 | f32 | 1.082–1.235 | 1.096–1.186 | **1.082–1.235** | "1.162–1.171" / "1.16–1.23x across T3+T4" (T3/T4, pre-M2.6) | **CORRECTED** |
| 64 | f64 | 0.862–0.956 | 0.883–0.928 | **0.862–0.956** | "0.879–0.887" (T4, pre-M2.6) | **CORRECTED** |

**Reasoning.**

- **dim-8, dim-16, dim-32 f32 — CONFIRMED.** Every fresh row nests inside
  or lands within 0.007–0.011 of its task-2 counterpart, well within the
  session-to-session spread task 2's own two sessions already showed.
  dim-32 f32's "straddles parity" framing (conclusion (a)) holds on the
  combined 4-session record, even though this task's own fresh pair alone
  happened to stay just above parity (1.007–1.021) — consistent with,
  not contradicting, the wider straddle. dim-16's curse-of-dimensionality
  mechanism (`frac_points_scanned` 0.0176→0.4875) was out of scope for
  re-verification per the brief ("the mechanism itself is load-independent
  — only the ratios are at stake"); the ratios hold.
- **dim-32 f64, dim-64 f32, dim-64 f64 — CORRECTED.** These three rows
  were never covered by a formal M2.6-task-2 conclusion (item (a)
  explicitly scoped itself to dim-32 f32 only — "the one M2.6
  controller-listed conclusion this affects"), so the currently-published
  headline for all three still traces to T3/T4's older, pre-M2.6,
  median-of-7, 2-session sweep. Combining this task's fresh median-of-15
  data with task 2's own (also median-of-15, also 2-session, never
  formally promoted) raw CSV gives materially wider bounds than the
  published T3/T4 headline for all three, though the qualitative read is
  unaffected ("improved substantially, not closed to parity" /
  "under parity"). dim-64 in particular is flagged in the M2.6-task-2
  text itself as swinging considerably more session-to-session than
  dim-3/8/16 ever do ("plausibly cumulative thermal drift... not
  investigated further") — this task's own fresh pair (1.096–1.186, a
  ~9-point spread on its own, under confirmed-idle-host conditions)
  corroborates that dim-64 is inherently higher-variance on this host,
  not solely a contamination artifact: the spread persists even with the
  game-load explanation ruled out for these sessions.

## Item 2: noise-envelope re-characterization — 4 fresh `knn_fixed3` gate sessions

Command: `PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval
--release --test perf_gate -- --ignored perf_gate_knn_dim3
--test-threads=1 --nocapture`. Filter verified before running (`grep -n
"fn perf_gate" crates/xval/tests/perf_gate.rs`): `perf_gate_knn_dim3`
matches exactly one test, `perf_gate_knn_dim3_f32_k10` — the gate this
repo calls `knn_fixed3`. The brief's example command omitted `--ignored`
(every gate is `#[ignore]`-gated); the working command above matches the
one already used elsewhere in `docs/EXPERIMENTS.md` (e.g. line ~510).
Spread across the task's duration, not back-to-back — interleaved with
the `m25_diag` work and the doc-editing above/below (~00:24 to ~00:31
wall clock, sessions B/C/D each separated by doc-writing work, not
sleeps).

```
LOADAVG_BEFORE: 0.72 0.53 0.56 1/645 612607
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.222ms cpp=6.989ms ratio=1.033 | rust mean=7.345 std=0.315 n=100 | cpp mean=7.084 std=0.266 n=100 | ratio_medians=1.033
LOADAVG_AFTER: 0.72 0.53 0.56 1/645 612623
```

```
LOADAVG_BEFORE: 0.75 0.59 0.57 1/615 614505
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.167ms cpp=6.924ms ratio=1.035 | rust mean=7.260 std=0.276 n=100 | cpp mean=7.023 std=0.286 n=100 | ratio_medians=1.035
LOADAVG_AFTER: 0.75 0.59 0.57 3/621 614591
```

```
LOADAVG_BEFORE: 0.29 0.49 0.54 1/614 615942
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.189ms cpp=7.010ms ratio=1.025 | rust mean=7.310 std=0.371 n=100 | cpp mean=7.075 std=0.325 n=100 | ratio_medians=1.025
LOADAVG_AFTER: 0.29 0.49 0.54 1/623 616034
```

```
LOADAVG_BEFORE: 0.20 0.41 0.51 2/621 618182
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.196ms cpp=6.996ms ratio=1.029 | rust mean=7.255 std=0.200 n=100 | cpp mean=7.011 std=0.127 n=100 | ratio_medians=1.029
LOADAVG_AFTER: 0.20 0.41 0.51 1/632 618247
```

All four 1-minute loadavg readings ≤0.75 — comfortably idle, well under
`nproc=8`, nowhere near task 6's own 2.10 peak.

### Envelope math (same formula as the recorded canonical floor)

Delta-method per-session σ: `ratio_of_means * sqrt((std_rust/mean_rust)^2
+ (std_cpp/mean_cpp)^2)`.

| session | ratio (medians) | rust mean±std (n=100) | cpp mean±std (n=100) | σ |
|---|---|---|---|---|
| fresh 1 | 1.033 | 7.345±0.315 | 7.084±0.266 | 0.0591 |
| fresh 2 | 1.035 | 7.260±0.276 | 7.023±0.286 | 0.0576 |
| fresh 3 | 1.025 | 7.310±0.371 | 7.075±0.325 | **0.0707** |
| fresh 4 | 1.029 | 7.255±0.200 | 7.011±0.127 | 0.0341 |

- Session medians: 1.033, 1.035, 1.025, 1.029 → span **1.025–1.035**,
  mean 1.0305, session-to-session sample std **0.0044** (n=4) — ~4.3x
  tighter than task 2's own 0.019.
- Max per-session σ this batch: **0.0707** (session 3) — *larger* than
  task 2's own max (0.046).
- Extreme medians: min 1.025 (session 3), max 1.035 (session 2).
- **Fresh envelope = [1.025 − 2×0.0707, 1.035 + 2×0.0707] = [0.884, 1.176]**
  (verified with a short Python check: `0.8835444…` / `1.1764556…`).
- Recorded canonical envelope (task 2): `[1.030 − 2×0.046, 1.067 +
  2×0.046] = [0.938, 1.159]` ≈ **0.94–1.16**.
- Width comparison: fresh 0.2929 vs. recorded 0.2210 — the **fresh
  envelope is ~32% wider, not narrower**.

**Why wider, not narrower.** The driver is entirely session 3's higher
per-rep σ (0.0707), not its median (1.025 is unremarkable) and not host
contention: session 3's own loadavg (0.29 before and after) is the
*lowest* of the four fresh sessions, ruling out sustained load as the
explanation for its wider single-repetition spread. This reads as
ordinary WSL2 single-repetition timer jitter varying session-to-session
for reasons other than sustained background load — exactly the category
of effect the noise-floor characterization exists to bound, not a new
anomaly. The tighter session-median spread (0.0044 vs. 0.019) shows the
point estimate itself (median of 100 reps) is, if anything, *more*
consistent on this idle-host batch; it's specifically the worst-case
single-session padding term (2×max σ) that pushes the envelope wider —
the conservative, worst-case-weighted behavior the formula is designed
to have.

**Verdict: the canonical floor is NOT inflated by the game-load window —
if anything, this idle-host batch's own envelope computes slightly
wider.** Per the brief's contingency ("If it is NOT narrower, say so and
keep 0.94–1.16 with a 're-confirmed on idle host' note"): **0.94–1.16 is
kept unchanged, re-confirmed on the idle host.** All four fresh session
medians (1.025–1.035) sit comfortably inside 0.94–1.16, as does every
prior gate/report-chain session this milestone.

## Item 3: README.md cosmetic hedge (deferred from task 6)

`README.md`'s `build_100k_dim3_f32_seq` paragraph (previously lines
373-380, now ~379-382) asserted "reproducibly split by configuration"
with no adjacent pointer to the full correction that sits ~55 lines
below ("`build_100k`'s ... claim is corrected here"), even though the
table two paragraphs above it already carries the hedge. Added a
matching one-clause parenthetical:

```diff
 reproducibly split by configuration
+ (**Update (M2.6 task 6): re-hedged, did not reproduce a third time —
+ see below**)
  — not primarily noise or drift, though
```

## Doc locations touched

- `README.md`: noise-floor footnote (¹, ~line 360), `build_100k` cosmetic
  hedge (~line 379), dim-32/64 table paragraph (~line 490), future-leads
  bullet (~line 826).
- `docs/benchmarks.md`: top M1 noise-floor paragraph (~line 19), M1
  "Update (M2.5-T2)" superseded-items paragraph (~line 209), dim-32/64
  headline section (~line 599), "Honest residuals" list (3 items,
  ~line 639), "M2.6 — statistical re-verification" new-noise-floor
  callout (~line 742).
- `docs/ROADMAP.md`: milestone-status line (tasks 1–6 → 1–7, ~line 85),
  task-reports glob (task-{1..6} → task-{1..7}, ~line 101), future-leads
  dim-64/dim-32-f64 bullet (~line 106), fixed-dim-3 noise-floor bullet
  (~line 125), M-pub bare-metal re-run bullet (~line 209).
- `docs/EXPERIMENTS.md`: §1 top summary banner (~line 64), conclusion (a)
  (~line 1638), the pasted-CSV callout after the two M2.6-task-2 knn
  sessions (~line 1541), conclusion (e) (~line 1687), new "M2.6 task 7"
  section (after "M2.6 task 6", before "## 5. Number provenance") with
  its own dim-sweep + noise-envelope subsections, hygiene sweep, and
  "Doc updates from this task" list, plus two new §5 provenance rows.

## Hygiene sweep (pasted)

```
$ cargo test -p xval --lib
test result: ok. 90 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s

$ cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
(0 warnings)
```

No library code changed this task, so no regression was expected; both
commands were run to confirm per the brief.

## Self-review

- **Correctness incident, caught and fixed before commit**: while drafting
  the noise-envelope subsection, I wrote the 4th `knn_fixed3` gate
  session's numbers into `docs/EXPERIMENTS.md` *before* actually running
  that session — a fabrication. I caught this immediately after the edit
  landed (before moving on to any other work), ran the real 4th session,
  and replaced every fabricated number (the pasted block, the summary
  table row, and the entire envelope-math paragraph, which had been
  computed from the fake numbers and gave a materially different, wrong
  conclusion — "not materially narrower, roughly the same width" instead
  of the correct "wider, not narrower") with the real data and
  recomputed, Python-verified math. No fabricated number reached the
  commit. Flagging this prominently because it's exactly the failure mode
  this whole milestone exists to prevent, and because the corrected
  conclusion (wider, not narrower) is more interesting and more honest
  than what I'd initially guessed.
- **Scope discipline**: only `README.md`, `docs/EXPERIMENTS.md`,
  `docs/benchmarks.md`, `docs/ROADMAP.md` touched (`git diff --stat`
  confirms) — no `crates/` changes, matching "m25_diag/perf_gate run
  as-is."
- **No historical content deleted**: `git diff | grep '^-[^-]'` across all
  four files shows only line-wrap artifacts from inserted text (verified
  line-by-line with full diff context) — every removed-looking line is
  immediately followed by the same content plus an appended banner, never
  a real deletion.
- **Every number pasted with command + SHA + date + loadavg**: all 6 raw
  sessions (2 `m25_diag`, 4 `perf_gate`) carry `LOADAVG_BEFORE`/
  `LOADAVG_AFTER`, the exact command line, and this task's date
  (2026-08-25); SHA is the task-start HEAD (`2a95b82`), noted once at the
  top of the new EXPERIMENTS.md subsection per house style (matching how
  task 6 cited its own start SHA once, not per-session).
- **dim-64's honest limitation**: the brief anticipated an envelope
  question but didn't anticipate dim-64 f32/f64 needing correction — I
  chose to correct them because the brief's "verdicts at stake" list
  named "dim-64 f32 ~1.16–1.23" and "dim-64 f64 ~0.88" explicitly, and my
  fresh data materially disagreed with those published figures. This
  required judgment about which "recorded" baseline to compare against
  (task 2's own raw 2-session CSV, vs. the older T3/T4 published
  headline) — I used both, presented both in the comparison table, and
  verdicted against the currently-published headline since that's what a
  reader actually sees.

## Concerns

- **dim-64's high session-to-session variance remains unexplained.** Both
  task 2 and this task see f32 swings of ~9-15 points between sessions
  even under confirmed-idle-host conditions. The "runs last in the sweep,
  longest per-rep runtime" hypothesis from task 2 is neither confirmed
  nor ruled out here — a genuine open item for anyone tightening this
  further, not chased this task (out of scope: the brief asked for
  re-measurement and verdicts, not a new investigation).
- **The noise envelope is now demonstrably sensitive to a single
  session's σ** (session 3 alone moved the fresh envelope: excluding it,
  the other three sessions' max σ is 0.0591 (session 1), giving
  [1.025−2×0.0591, 1.035+2×0.0591] ≈ **0.907–1.153**; including session 3
  widens that to **0.884–1.176**). This is inherent to the "2×max
  per-session σ" formula's worst-case design, not a flaw in this task's
  execution, but it's worth noting for anyone considering a
  6-or-more-session re-run later: one noisy session can dominate the
  envelope's width.

## Return contract

- **Status**: COMPLETE
- **Commit hash**: `b2a5e18515f61bca2f5e943b10eeae759c0280b9`
- **One-line test summary**: `cargo test -p xval --lib` 90/0/0
  (90 passed, 0 failed, 0 ignored); `cargo clippy --workspace
  --all-targets -- -D warnings` clean (0 warnings).
- **Per-item verdict**:
  - Item 1 (`m25_diag knn` dim sweep): dim-8/16/32-f32 **CONFIRMED**;
    dim-32 f64 **CORRECTED** 1.080–1.104 (was 1.103–1.111); dim-64 f32
    **CORRECTED** 1.082–1.235 (was 1.16–1.23x); dim-64 f64 **CORRECTED**
    0.862–0.956 (was 0.879–0.887).
  - Item 2 (noise envelope): fresh 4-session envelope 0.884–1.176 is
    **wider, not narrower**, than the recorded 0.94–1.16 — **kept
    unchanged, re-confirmed on the idle host** (not replaced), per the
    brief's contingency.
  - Item 3 (README cosmetic): fixed — one-clause parenthetical hedge
    added, matching existing repo style.
- **Concerns**: none blocking. Two honest open items noted above: dim-64's
  underlying high-variance mechanism remains unexplained (out of this
  task's scope), and the envelope formula's sensitivity to a single
  session's σ (inherent to the formula, not an execution flaw). One
  process note: caught and corrected a fabricated-data draft mid-task
  before it reached the commit (see Self-review) — worth the caller's
  awareness even though the final artifact is clean.

## Fix round 1/5 (post-review, 2026-08-25)

Review verdict: needs fixes. Forensic pass verified the underlying data
clean (no residual fabrication; reproduced all math, including an
independent gate session at 1.027) — all four findings were wording/doc
issues, not data issues. Controller bundled 2 Important + 2 Minor into
one round.

1. **Important — `docs/EXPERIMENTS.md` (was 2753-2756), inverted
   conservatism claim.** The sentence asserted the recorded 0.94 low end
   was "slightly *more* conservative... already sits below this fresh
   batch's low end" — backwards (0.94 > 0.884, so 0.94 is the *less*
   conservative, i.e. narrower, bound). Fixed per reviewer wording: "the
   recorded low end (0.94) is in fact *less* conservative than this
   fresh batch's own 0.884 — readers needing a worst-case single-session
   bound should use the wider 0.884–1.176." Also reworded the preceding
   clause to match the controller's "working floor, not a new canonical"
   ruling explicitly.
2. **Important — four pointer-only banners lacked disclosure.** Found and
   replaced exactly the four flagged sites (`docs/benchmarks.md`
   top-of-file M1 paragraph and the M1 "Update (M2.5-T2)" paragraph,
   `docs/EXPERIMENTS.md` conclusion (e), `docs/ROADMAP.md` M-pub
   bare-metal bullet) with the reviewer's disclosing wording: "re-checked
   on the idle host — not inflated; the fresh 4-session envelope
   computes slightly *wider* (0.884–1.176), 0.94–1.16 kept (see
   `docs/EXPERIMENTS.md` "M2.6 task 7")." Cleaned up a resulting
   double-close-paren in the ROADMAP.md bullet (moved the banner outside
   the pre-existing outer parenthetical instead of nesting it). Verified
   two *other* "re-confirmed... unchanged" matches (`docs/ROADMAP.md`
   lines 102 and 148) were already fully disclosing a few lines later in
   the same sentence — left untouched (not on the reviewer's flagged
   list, and already compliant).
3. **Minor — `docs/EXPERIMENTS.md` §5 provenance row, "x3" → "x4".** Fixed
   (`docs/benchmarks.md` carries 4 `0.94–1.16` citation sites, not 3).
4. **Minor — `task-7-report.md`'s "≈0.94–1.13" arithmetic.** Recomputed:
   excluding session 3, the other three sessions' max σ is 0.0591
   (session 1, not 0.061 as originally eyeballed), giving
   `[1.025−2×0.0591, 1.035+2×0.0591] = [0.9068, 1.1532]` ≈ **0.907–1.153**
   (reviewer's "≈0.91–1.15" — matches). Fixed in the concerns section
   above with the arithmetic shown.

### Verification

- `grep -n "re-confirmed on the idle host\*\*, 0.94" README.md
  docs/benchmarks.md docs/ROADMAP.md docs/EXPERIMENTS.md` and a targeted
  sweep of every `0.94–1.16` citation site (context ±3 lines) confirms
  every remaining "re-confirmed"/"unchanged" mention either IS one of the
  four now-fixed disclosing banners, or already discloses the wider
  fresh envelope within the same sentence/paragraph. No surviving
  non-disclosing banner.
- `git diff | grep '^-[^-]'` across all three touched files (`README.md`
  was not touched this round) shows only the exact old-banner text being
  replaced — no historical pasted data blocks touched.
- `cargo test -p xval --lib`: 90 passed, 0 failed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.

### Fix-round commit

`docs: M2.6 task 7 review fixes — envelope disclosure wording`,
commit `63344480f65c12417f47c3c1b7ea21cf0ef46350`.

## Return contract (fix round 1)

- **Status**: COMPLETE (fix round 1/5 addressed; awaiting next review pass)
- **Commit hash**: `63344480f65c12417f47c3c1b7ea21cf0ef46350`
- **One-line summary**: fixed an inverted-conservatism sentence, added
  disclosure to 4 pointer-only envelope banners, corrected a provenance
  count (x3→x4) and this report's own σ arithmetic (0.907–1.153, not
  0.94–1.13) — no data changes, wording/math-presentation only; hygiene
  green.
