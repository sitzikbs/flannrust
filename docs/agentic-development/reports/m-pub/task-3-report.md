# Task 3 report: Packaging dry-runs

Status: DONE
Worktree branch: `worktree-agent-a308ef801e862fa66`
Commit: `2b1ee18` — chore: packaging dry-run evidence (crates.io + wheel smoke test)

## What was done

1. Applied the controller ruling: removed `publish = false` (and its explanatory
   comment) from `crates/flannrust/Cargo.toml`. This was the only metadata change
   needed anywhere -- `crates/flannrust-py/pyproject.toml` required no changes.
2. Step 1 -- `cargo publish --dry-run -p flannrust --allow-dirty` succeeded on the
   first attempt after the `publish = false` removal (no other metadata issues).
3. Step 2 -- name availability: crates.io API requires a `User-Agent` header (a bare
   `curl` gets HTTP 403 with an empty body); with a UA header it returns HTTP 404
   `{"errors":[{"detail":"crate \`flannrust\` does not exist"}]}`. PyPI returns HTTP
   404 `{"message": "Not Found"}`. Both names are free.
4. Step 3 -- built a release wheel with `maturin build --release`, using the main
   repo's `crates/flannrust-py/.venv` maturin binary (1.14.1) but pointed at this
   worktree's `crates/flannrust-py/Cargo.toml` via `-m`, so the compiled extension is
   this worktree's code. Built:
   `target/wheels/flannrust-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl`.
   Created a brand-new venv at the scratchpad path given in the brief, installed the
   wheel + numpy into it, and ran the smoke-test one-liner from the brief. Output:
   `(5, 3) (5, 3)`.
5. Wrote `docs/agentic-development/reports/m-pub/packaging-dryrun.md` with all four steps' evidence
   pasted verbatim (commands + full output), plus a summary table.
6. Committed both files in a single commit in the worktree.

Full evidence is in `docs/agentic-development/reports/m-pub/packaging-dryrun.md` in the worktree (see
commit `2b1ee18`); every number in that file is command output, not paraphrase.

No `cargo publish` was ever run without `--dry-run`. No PyPI upload was performed.

## Self-review of diff

- `crates/flannrust/Cargo.toml`: 4 lines removed (the `publish = false` line, its
  2-line comment, and the resulting stray blank line was fixed so `[features]` keeps
  a blank line before it, matching the file's original style). No other fields
  touched.
- `docs/agentic-development/reports/m-pub/packaging-dryrun.md`: new file, evidence only, no code.
- No changes to `crates/flannrust-py/pyproject.toml` -- the dry-runs did not demand
  any, so per the brief's "only if dry-runs demand" instruction it was left alone.
- `crates/flannrust-py/Cargo.toml` still has `publish = false`, which is correct and
  out of scope for this task (the brief only calls out `flannrust`'s crates.io
  publishability; the Python binding crate is not itself published to crates.io).

## Concerns

None. All four steps in the brief passed cleanly with no surprises beyond the
crates.io API's User-Agent requirement (worked around with `-A`, does not affect the
crate name being free).
