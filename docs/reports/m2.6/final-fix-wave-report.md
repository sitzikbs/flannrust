# Final review fix wave — report

Branch `m2p6-rigor`, base HEAD `6334448`.

## Finding 1 (Important): task-report citations point at an unreachable path

The branch's new doc text cited task reports two ways a repo reader can't follow:

- `docs/ROADMAP.md:109` (was) cited
  `docs/superpowers/sdd/2026-08-25-flannrust-m2.6-rigor/task-{1..7}-report.md` — that directory
  doesn't exist under `docs/`; the actual files live in the gitignored
  `.superpowers/sdd/2026-08-25-flannrust-m2.6-rigor/`, invisible to anyone who clones the repo.
- Nine more spots (all new in this branch, verified via `git diff ef8467c..HEAD`) cited the bare
  filename `task-4-report.md` with no path at all: `README.md:828`, `docs/benchmarks.md:858,907,909,1052`,
  `docs/EXPERIMENTS.md:2340,2871`, `docs/ROADMAP.md:126,222`.

Fix, per the house convention already used for M2.5 (`docs/reports/m2.5/`) and M-py
(`docs/reports/m-py/`):

1. `mkdir -p docs/reports/m2.6 && cp .superpowers/sdd/2026-08-25-flannrust-m2.6-rigor/task-{1,2,3,4,5,6,7}-report.md docs/reports/m2.6/`.
2. Repointed every one of the ten citations above to `docs/reports/m2.6/task-N-report.md`.
3. Grepped the seven copied reports for `superpowers/sdd` self-references — none found, no further
   edits needed inside them.

Left untouched (confirmed pre-existing, not introduced by this branch, via
`git diff ef8467c..HEAD` — `ef8467c` is `main`'s merge-base): bare `task-2-report.md` /
`task-3-report.md` mentions in `docs/benchmarks.md` (562, 598, 651, 660, 694) and
`docs/EXPERIMENTS.md` (134, 1146, 1157, 1189, 1246, 2856), which are older M2.5/M-py citations
already broken before this branch started — out of this fix wave's scope, but worth a follow-up
pass.

Verified clean:
```
$ grep -rn "superpowers/sdd" docs README.md
(no output)
```

## Finding 2 (Minor): pre-existing ruff F401 — checked, was genuinely unused

`crates/flannrust-py/python/tests/test_query_radius_box.py:9` — `import pytest`. Read the full
134-line file: no `@pytest.mark`, `pytest.raises`, `pytest.fixture`, or any other `pytest.*` use
anywhere in it (dtype parametrization comes from a `dtype` fixture supplied via `conftest.py`, not
referenced through the `pytest` module directly). The finding was correct — removed the import.

```
$ .venv/bin/pytest python/tests/test_query_radius_box.py -q
..............                                                           [100%]
14 passed in 0.03s
```

## Verification (paste)

```
$ ruff check crates/flannrust-py/python
All checks passed!
```

```
$ .venv/bin/pytest python/tests -q      # from crates/flannrust-py
........................................................................ [ 18%]
........................................................................ [ 37%]
........................................................................ [ 56%]
.........x...x...............................xxxxx.......x.x.x.......... [ 75%]
......x.x....x.x.....x.x.........................................xxxXxxx [ 94%]
xxxxx..............                                                      [100%]
351 passed, 27 xfailed, 1 xpassed in 3.64s
```
Matches the 351-passed baseline exactly.

```
$ cargo test -p xval --lib
test result: ok. 90 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
```

```
$ cargo clippy --workspace --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
```
No warnings.

## Commit

`docs: final-review fixes — preserve M2.6 task reports in docs/reports, ruff cleanup`

Files touched: `README.md`, `docs/ROADMAP.md`, `docs/benchmarks.md`, `docs/EXPERIMENTS.md`,
`docs/reports/m2.6/task-{1,2,3,4,5,6,7}-report.md` (new),
`crates/flannrust-py/python/tests/test_query_radius_box.py`, this report (both copies).
