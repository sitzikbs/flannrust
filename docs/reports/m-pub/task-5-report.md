# Task 5 report: Claims audit + provenance note

Worktree: /home/sitzikbs/dev/flannrust/.claude/worktrees/agent-ae28aaa5e7bd16ea0
Branch: worktree-agent-ae28aaa5e7bd16ea0
Commit: dd94f2a (parent 1dcc34b, which is 1 commit ahead of the merge base 9cac562)

## What was done

1. **Enumerate.** Grepped README.md and docs/benchmarks.md for every
   performance-number claim (ratios/ms/x/%). Given the extreme existing
   density -- both files already carry a mature, self-consistent "Update
   (M2.6 task N)" banner convention layering fresher measurements onto
   pasted evidence without editing it in place, and docs/EXPERIMENTS.md's
   "5. Number provenance" section already maps nearly every figure in both
   files to a reproducible command + output field -- the audit grouped
   claims by section/workload rather than by literal digit and cross-checked
   each against docs/EXPERIMENTS.md and the fresh post-merge report at
   9cac562 (core ratios supplied by the controller: build_100k 0.999,
   knn_dim3 1.013, knn_dyn_dim8_f64 0.927, radius 0.876, build_1M_seq
   0.994, build_1M_par 0.604, dyn_add 1.025, dyn_churn 0.953).

2. **Fix.** Found exactly one STALE claim: docs/benchmarks.md's M1
   "Parallel build" section headlines "roughly 1.75x faster" for
   build_1M_dim3_f32_par and -- unlike every other headline figure in this
   document -- never carried an "Update (M2.6...)" banner. EXPERIMENTS.md
   already has a pasted M2.6-task-6 n=100 re-measurement of the same
   workload (0.6151, ~1.63x) that was never surfaced into this section, and
   the fresh 9cac562 run confirms it (0.604, ~1.66x) -- both consistent
   with the 1.47-1.58x band the same file's "two-command report chain"
   subsection already documents for this workload under a different
   methodology. Fixed by adding an "Update (M2.6 task 6 / M-pub)" paragraph
   directly after the existing pasted table (table itself untouched, per
   house style -- pasted evidence is never edited in place), stating plainly
   that 1.6-1.66x, not 1.75x, is the figure to publish going forward.
   No other STALE or UNTRACEABLE claims were found -- every other number in
   both files traces to a command + pasted output in docs/EXPERIMENTS.md.
   Two fresh-9cac562 figures (radius 0.876 vs. recorded 0.831-0.868;
   dyn_add 1.025 vs. recorded 1.0315-1.039) sit ~1% and ~0.6% outside the
   currently-published M2.6-task-6 ranges; both are within the
   session-to-session spread these two workloads have shown throughout
   M2.6 (radius alone has ranged 0.767-0.876 across the milestone), so
   these were marked TRACED (consistent with the noise story the docs
   already tell repeatedly), not rewritten.

3. **Provenance note.** Added a "How this was built" section to README.md
   (between "Attribution & license" and "Quickstart"): states plainly that
   this codebase was implemented by an AI agent (Claude Code) directed and
   reviewed by Itzik Ben-Shabat, who does not write Rust, and that
   correctness rests on the bit-exact cross-validation suite against
   vendored nanoflann 1.12.1, not on the author's Rust expertise. No
   marketing language.

4. **Verify.** grep -rn "superpowers/sdd" README.md docs/benchmarks.md
   is empty (checked before and after edits). cargo test --workspace run
   once at the end: **all green, 0 failed** across every binary (workspace
   unit tests, xval integration suites, doctests) -- full pasted summary
   below.

## Files touched

- Modified: README.md (new "How this was built" section)
- Modified: docs/benchmarks.md (one "Update" paragraph added to the
  "Parallel build" section; nothing else changed)
- Created: docs/reports/m-pub/claims-audit.md (the full claim-by-claim
  audit table, verdicts, and what was fixed -- this is the whitelist Task 7
  (blog) should treat as publishable numbers)

## cargo test --workspace result (run once, end of task)

All "test result: ok" lines, zero failures, across every binary:

```
test result: ok. 198 passed; 0 failed; 4 ignored; ...
test result: ok. 0 passed; 0 failed; 0 ignored; ...
test result: ok. 0 passed; 0 failed; 0 ignored; ...
test result: ok. 23 passed; 0 failed; 0 ignored; ...
test result: ok. 12 passed; 0 failed; 0 ignored; ...
test result: ok. 90 passed; 0 failed; 0 ignored; ...
test result: ok. 2 passed; 0 failed; 0 ignored; ...
test result: ok. 0 passed; 0 failed; 1 ignored; ...
test result: ok. 3 passed; 0 failed; 6 ignored; ...
test result: ok. 19 passed; 0 failed; 0 ignored; ...
test result: ok. 5 passed; 0 failed; 0 ignored; ...
test result: ok. 11 passed; 0 failed; 2 ignored; ...
test result: ok. 13 passed; 0 failed; 1 ignored; ...
test result: ok. 12 passed; 0 failed; 1 ignored; ...
test result: ok. 2 passed; 0 failed; 0 ignored; ...   (doctests flannrust)
test result: ok. 0 passed; 0 failed; 0 ignored; ...   (doctests nanoflann-ref)
test result: ok. 0 passed; 0 failed; 0 ignored; ...   (doctests xval)
```

No FAILED or error[ lines anywhere in the run.

## Self-review notes

- Re-read the full diff after committing (git diff dd94f2a~1 dd94f2a):
  both hunks are additive-only (no deletions, no rewording of existing
  pasted evidence), consistent with the "do not rewrite docs wholesale"
  constraint and this repo's own "pasted evidence never edited in place"
  house convention.
- Deliberately did not touch docs/EXPERIMENTS.md (out of scope per the
  brief's file list) even though its "Number provenance" table's
  "Parallel build 1.75x win (M1)" row is technically now slightly
  out-of-date relative to the new benchmarks.md banner -- flagging this for
  the controller/Task 7 rather than expanding this task's scope.
  docs/reports/m2.*/ and docs/reports/m-py/ were not touched.
- The 9cac562 figures I cite in the new benchmarks.md paragraph are
  cited as controller-supplied data (not claimed to be pasted verbatim in
  EXPERIMENTS.md) -- phrasing checked to stay honest about that
  distinction.

## Status

DONE. No blockers, no concerns beyond the EXPERIMENTS.md provenance-table
staleness flagged above (informational only, out of this task's file
scope).

## Fix round 1 (review findings)

Commit: 3b29f50 (parent dd94f2a)

Three Important findings addressed, all citation/wording corrections, no
substantive claims changed:

1. claims-audit.md claim 4 cited docs/benchmarks.md:826-847 for the M2.6
   task 6 six-gate table; that range is actually task 5's dyn_add content.
   Corrected to docs/benchmarks.md:846-865 (section header + full table).
   Re-verified the README half per the reviewer's "adjacent-but-loose"
   flag: the original README:414-436 citation was also wrong (pointed at
   adjacent M2.5-statistical-re-verification / M2.6-task-5 prose, not the
   table) -- the actual README M2.6-task-6 table is at 443-452. Confirmed
   the README and docs/benchmarks.md tables match row for row (all six
   workload ranges identical); citation corrected to README:443-452.

2. claims-audit.md claim 11 cited README.md:167-169 for the Safety
   "+4.7%" unsafe-necessity figure; the Safety section actually starts at
   README.md:174 and the figure itself is at line 187. Corrected.

3. README.md's "How this was built" provenance sentence said every
   performance figure "traces to a reproducible, pasted command and run
   recorded in docs/EXPERIMENTS.md" -- but the new benchmarks.md paragraph
   (from the original task) quotes 0.604 from commit 9cac562, a
   controller-supplied post-merge confirmation run that does not appear in
   EXPERIMENTS.md. Reworded to "...recorded in docs/EXPERIMENTS.md or, for
   post-merge confirmation runs, in the commit-tagged report data cited
   alongside the figure" -- stays strictly true, the 0.604 mention is
   untouched (its hedged wording was already judged correct by the
   reviewer). docs/EXPERIMENTS.md itself was not touched, per scope.

Checks re-run: grep -rn "superpowers/sdd" README.md docs/benchmarks.md is
empty (only files touched this round: README.md and
docs/reports/m-pub/claims-audit.md -- docs/benchmarks.md was not modified
this round, so its own grep result is unchanged from the original task).
cargo test --workspace not re-run: this round is doc-only citation/wording
fixes touching README.md and docs/reports/m-pub/claims-audit.md, neither
of which affects any test or code path.

Status: DONE. Commit 3b29f50. No new concerns.

## Fix round 2 (review findings: line numbers drift)

Commit: 606e40e (parent 3b29f50)

Re-review found fix round 1's corrected line ranges were ALREADY off
(benchmarks.md 846-865 vs actual table at 856-866; README range cut the
table at row 2; and README's own fix-round-1 edit shifted the Safety
section's 174/187 citations to 176/189). Root cause: raw line numbers are
brittle against a repo whose own commits move them, including this task's
own commits.

Structural fix applied: every citation in claims-audit.md now points to
file + nearest UNIQUE section heading, plus a short verbatim quote where a
specific figure/paragraph matters, instead of raw line numbers. Added one
preamble sentence to the table explaining why (line numbers drift with
every commit, including the audit's own). Also caught and fixed a real
file misattribution while restructuring: claim 12's "1.47-1.58x"
report-chain-tile reconciliation text has only ever lived in
docs/EXPERIMENTS.md's "The two-command report chain" section -- it was
cited as docs/benchmarks.md in every prior round, which was simply wrong
(verified: docs/benchmarks.md has no "two-command report chain" heading or
"Reconciling this tile" text anywhere).

### Anchor verification (grep -c, one hit required per anchor)

All 23 anchors used in the rewritten table, checked against the current
worktree after the round-2 edit landed:

```
R1 heading [README.md]: 1        (^## How this was built$)
R2 heading [README.md]: 1        (^### Perf gate (six gated workloads; pass threshold is ratio <= 1.25)$)
R2 quote [README.md]: 1          (Post-M2.5** (T4's fresh two-run sweep)
R3 quote [README.md]: 1          (Update (M2.6, commit `a30a819`, 2026-08-25):)
R4 quote [README.md]: 1          (re-confirms every range above)
R5 heading [README.md]: 1        (^### dim-32/64 knn: the M1-era gap is closed at f32 (M2.5)$)
R5 quote task7 [README.md]: 1    (Update (M2.6 task 7, user-directed idle-host re-measurement):)
R6 heading [README.md]: 1        (^### `leaf_max_size` sweep$)
R10 heading [README.md]: 1       (^### Benchmarks$)
R11 heading [README.md]: 1       (^## Safety)
R11 quote [README.md]: 1         (+4.7%** on the fixed-dim-3 knn gate)

R4 bm heading [docs/benchmarks.md]: 1     (^### M2.6 task 6 -- final regression sweep)
R4 bm quote [docs/benchmarks.md]: 1       (**Final six-gate table**)
R5 bm heading M2.5 [docs/benchmarks.md]: 1 (^### dim-32/64 headline: the M1-era gap is closed at f32$)
R5 bm quote task7 [docs/benchmarks.md]: 1 (Update (M2.6 task 7, user-directed idle-host re-measurement):)
R6 bm heading [docs/benchmarks.md]: 1     (^### `leaf_max_size` sweep$)
R7 bm heading [docs/benchmarks.md]: 1     (^### Parallel build$)
R7 bm my new quote [docs/benchmarks.md]: 1 (Update (M2.6 task 6 / M-pub)
R8 bm heading [docs/benchmarks.md]: 1     (^### Headline table (criterion, spot-check methodology, `--quick`)$)
R9 bm heading [docs/benchmarks.md]: 1     (^## M2.6 -- statistical re-verification$)
R10 bm heading a [docs/benchmarks.md]: 1  (^### Success criteria vs. spec (honest accounting, both bench runs)$)
R10 bm heading b [docs/benchmarks.md]: 1  (^### Honest misses)

R12 EXP heading [docs/EXPERIMENTS.md]: 1  (^### The two-command report chain$)
R12 EXP quote [docs/EXPERIMENTS.md]: 1    (Reconciling this tile's 1.47)
```

Every anchor returned exactly 1 match -- confirmed unique before being
committed. Script used: a small bash file with a `check()` helper running
`grep -c` per anchor (avoided inline complex-quoting bash calls, which the
sandbox refused as "too complex to verify stays inside the worktree").

Checks re-run: `grep -rn "superpowers/sdd" README.md docs/benchmarks.md`
still empty (only docs/reports/m-pub/claims-audit.md was touched this
round; that file's own Verification section quotes the grep command
itself as example text, which is not a dead pointer). cargo test not
re-run: this round only edits a report/docs file
(docs/reports/m-pub/claims-audit.md), no test-relevant path touched.

Status: DONE. Commit 606e40e. No new concerns.
