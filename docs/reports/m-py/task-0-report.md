# Task 0 report: rename `nanoflann-rs` -> `flannrust`

Commit: `a5f5aac` on branch `m-py` (was already checked out; no branch creation needed).

## What was changed

1. **Crate move**: `git mv crates/nanoflann-rs crates/flannrust`.
2. **`Cargo.toml` (workspace)**: `members` updated to `crates/flannrust`; added
   `default-members = ["crates/flannrust", "crates/nanoflann-ref", "crates/xval"]`.
3. **`crates/flannrust/Cargo.toml`**: `name = "flannrust"` (rest unchanged —
   still `publish = false`, same description/license/features).
4. **`crates/xval/Cargo.toml`**: dependency entry `flannrust = { path = "../flannrust" }`.
5. **Mechanical `.rs` replace** (`nanoflann_rs` -> `flannrust`, `nanoflann-rs` -> `flannrust`)
   across every file in `crates/flannrust/**` and `crates/xval/**` (source, tests,
   benches, examples) — `use` paths, doc-comments, panic-message strings
   (`tree.rs`'s `BuildThreads` panic + its `#[should_panic]` test), and the
   `xval` HTML scorecard's `<title>`/`<h1>` text (`report.rs`,
   `render_report_test.rs`'s assertion). `crates/nanoflann-ref` (the C++
   oracle crate, unrelated name) was left untouched — confirmed no
   `nanoflann_rs`/`nanoflann-rs` hits inside it.
6. **README.md**: title, the "derivative port of nanoflann" sentence, the
   Quickstart `use` line, and every current-state `crates/nanoflann-rs/src/...`
   source-file pointer (lib.rs doctest note, Safety section, `point_row`
   precondition note, dynamic.rs pointer), plus the `cargo doc -p nanoflann-rs
   --no-deps` hygiene-criterion command, all renamed to `flannrust`.
   Left unchanged: the two links to
   `docs/superpowers/plans/2026-08-22-nanoflann-rs-{m1,m2-dynamic}.md` —
   those are literal filenames of real, un-renamed historical plan documents
   (out of T0's scope), so renaming the link text would break them.
7. **docs/ROADMAP.md**: title (`# flannrust Roadmap`), the M-py goal sentence
   ("make flannrust usable from Python..."), the `point_row` source pointer
   (now `crates/flannrust/src/data_source.rs`), and the miri reproduction
   command (`cargo +nightly miri test -p flannrust --lib -- search`) all
   renamed. Left unchanged: the two `docs/superpowers/plans/2026-08-2{2,3}-nanoflann-rs-*.md`
   plan-filename references (same reasoning as README).
8. **docs/EXPERIMENTS.md**: added the required rename note at the top of §1
   ("Environment"):
   > The library crate was renamed `nanoflann-rs` → `flannrust` on 2026-08-23
   > (M-py T0); pasted outputs earlier than that show the old name/paths.

   Also updated the two *reproduction-instruction* command blocks that are
   not part of a pasted-output transcript (§3's "Miri" subsection's two bare
   `cargo +nightly miri test -p nanoflann-rs --lib -- search` /
   `MIRIFLAGS=... cargo +nightly miri test -p nanoflann-rs --lib -- search`
   lines, and the inline `cargo test -p nanoflann-rs --lib -- search --list`
   aside) to `-p flannrust`, per the given rule that current-runnable
   reproduction commands should keep working.

## Doc occurrences deliberately left as historical/out-of-scope (with reasoning)

- **`docs/benchmarks.md`** — left entirely untouched (title, prose, quoted
  asm symbol names, `-p nanoflann-rs` command mentions inside historical
  result paragraphs). Explicitly named in the brief's "Do NOT rewrite" list;
  no override context was given for this file the way EXPERIMENTS.md got
  the reproduction-command carve-out.
- **`docs/reports/m2.5/*.md`** — left entirely untouched (explicitly named
  "Do NOT rewrite", historical task reports).
- **`docs/EXPERIMENTS.md`** — left unchanged beyond the two items above:
  - Lines describing/tied to specific past measured results or a specific
    commit's tree state (e.g. "pre-M2.5 tree; `crates/nanoflann-rs` and
    `crates/nanoflann-ref` byte-identical to `41abdb6`", the `359/0/15`
    heavy-suite paragraph, the `179 passed, 0 failed, 3 ignored`
    `--no-default-features` paragraph, the sabotage-test transcript intro
    `cargo test -p nanoflann-rs metric::`, all `$`-prefixed pasted terminal
    blocks, and the mangled-asm-symbol dump) — these are evidentiary,
    provenance-anchored to a specific historical commit/run.
  - Two source-code doc-comment pointers in explanatory prose (§3's
    "`crates/nanoflann-rs/src/search.rs`'s `FrameStack`" intro sentence, and
    the M2-final-review-fix-wave paragraph's "`nanoflann_rs::dynamic::
    DynamicKdTree::add_points`'s doc comment" reference) — judgment call:
    EXPERIMENTS.md wasn't in the brief's "Modify" list (only the note was),
    and these read closer to the file's dominant historical-record register
    than to README/ROADMAP's forward-looking usage prose. Left as-is per
    "when in doubt inside EXPERIMENTS/benchmarks, leave it and note it."
- **`docs/nanoflann-notes.md`** — one occurrence (`crates/nanoflann-rs/src/dynamic.rs`'s
  `add_points` doc comment pointer). This file wasn't named anywhere in the
  brief (neither "Modify" nor "Do NOT rewrite"), so it's technically outside
  T0's declared scope; left untouched and flagged here as a possible
  follow-up (it's a stale path pointer now, same category as the ones fixed
  in README/ROADMAP).
- **`LICENSE`** — not in the brief's Files list. Contains two self-referential
  old-name mentions ("Copyright (c) 2026 nanoflann-rs contributors" and
  "...not part of the `nanoflann-rs` Rust library"). Left unchanged since
  the brief's verification bullet only asks to confirm nanoflann/Blanco-Claraco
  attribution is intact (it is — see below), not to rename LICENSE's own
  self-references. Flagging as a possible housekeeping follow-up.
- **`docs/superpowers/plans/2026-08-2{2,3}-nanoflann-rs-*.md`** — the actual
  plan filenames were not renamed (not in scope); every reference to them
  elsewhere (README, ROADMAP) necessarily still spells the old name since
  it's a real path on disk.

## Verifications run (decisive output)

**`grep -rn "nanoflann_rs" crates/`** — empty (required by the brief):
```
(no output)
```

**`grep -rn "nanoflann-rs" README.md docs/*.md`** — hits are the plan-filename
links (README/ROADMAP, unavoidable per above), the EXPERIMENTS.md rename
note itself, and the historical/out-of-scope occurrences enumerated above
(benchmarks.md, remaining EXPERIMENTS.md provenance text, nanoflann-notes.md).
No stray current-state mentions in README.md or docs/ROADMAP.md remain.

**`cargo build --workspace`**:
```
   Compiling flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust)
   Compiling xval v0.0.0 (/home/sitzikbs/dev/flannrust/crates/xval)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.40s
```

**`cargo test --workspace`** — aggregated across all 13 test binaries + 3 doctest
groups:
```
passed total: 359
ignored total: 15
(no FAILED, no failures:)
```
Matches the brief's "359 tests" exactly, 0 failed, 15 ignored (same as the
pre-rename baseline recorded in docs). Doc-tests confirm the new import
path works: `test crates/flannrust/src/lib.rs - (line 53) ... ok` and
`test crates/flannrust/src/dynamic.rs - dynamic::DynamicKdTree (line 320) ... ok`.

**`cargo clippy --workspace --all-targets -- -D warnings`**:
```
   Compiling nanoflann-ref v0.0.0 (...)
    Checking flannrust v0.1.0 (...)
    Checking xval v0.0.0 (...)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 11.07s
```
Zero warnings, zero errors.

**`cargo build -p flannrust --no-default-features`**:
```
   Compiling flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.18s
```

**`RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust --no-deps`**:
```
 Documenting flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.41s
   Generated /home/sitzikbs/dev/flannrust/target/doc/flannrust/index.html
```
Zero warnings.

**Attribution check** (README.md / LICENSE):
```
README.md:16:credited to Jose Luis Blanco-Claraco et al., which itself builds on FLANN by
LICENSE:31:This project vendors, unmodified, a copy of nanoflann 1.12.1
LICENSE:32:(https://github.com/jlblancoc/nanoflann) at
```
Attribution to nanoflann / Jose Luis Blanco-Claraco et al. is intact in both
files; the vendored C++ header's own license text is untouched (not part of
this rename's scope).

## Commit

`a5f5aac refactor: rename crate nanoflann-rs -> flannrust (published name, user decision)`
— 38 files changed (16 via `git mv` rename detection), working tree clean
after commit. Includes both required trailers.

## Concerns

- None blocking. The three out-of-scope stale pointers noted above
  (`docs/nanoflann-notes.md` line 202, `LICENSE`'s two self-references, and
  the two EXPERIMENTS.md doc-comment pointers left per judgment call) are
  cosmetic and don't affect build/test/doc correctness — worth a cheap
  follow-up sweep in a later task if the controller wants full consistency,
  but T0's own acceptance bar (crates/ clean, workspace green, clippy/doc
  clean, attribution intact) is fully met.
