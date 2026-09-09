# flannrust Shipping Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make flannrust ready to publish to crates.io and PyPI: LICENSE actually packaged (with correct derivative-work attribution), a fresh clone that builds without a C++ compiler, fmt-clean CI, a ~140-line landing-page README with the deep content extracted into `docs/semantics.md` / `docs/testing.md`, agentic-process docs under `docs/agentic-development/`, and the packaging-metadata polish (pyproject, CHANGELOG, example) that a 0.1.0 release expects. No actual publish.

**Architecture:** Docs/metadata-only milestone. The only source-code diffs permitted are the `cargo fmt --all` pass (Task 3) and one new example file (Task 9). Task order is dependency-driven: extraction (Task 5) must read the README *before* the rewrite replaces it (Task 7), and the `docs/agentic-development/` move (Task 6) must exist before the new README links to it.

**Tech Stack:** cargo (workspace, package, fmt, doc), maturin/pyproject (PEP 621/639), GitHub Actions, git mv.

**Spec:** docs/superpowers/specs/m-ship-spec.md

## Global Constraints

- **No behavior changes.** Nothing under `crates/*/src` changes except via `cargo fmt --all` (Task 3, its own commit) and the new `crates/flannrust/examples/knn.rs` (Task 9). No test, benchmark, or number changes.
- **Range honesty preserved.** Never alter a measured number. The README rewrite *deletes* duplicated narrative; every figure it keeps must be copied exactly from the M2.6 task 6 table. `docs/benchmarks.md` / `docs/EXPERIMENTS.md` / `docs/ROADMAP.md` / `docs/nanoflann-notes.md` content is untouched except path-reference fixes (Task 6).
- **Extraction, not rewriting.** `docs/semantics.md` / `docs/testing.md` are verbatim copies of README sections with only the mechanical adjustments each step lists (headings, relative links, "above/below" cross-references). Do not paraphrase, tighten, or "improve" the extracted prose.
- **One commit per task**, message prefix `docs:`, `chore:`, or `style:` as noted per task. Task 3 (fmt) must be a formatting-only commit with no other changes mixed in.
- **Never touch `.claude/worktrees/`** — every `grep`/`sed` in this plan must exclude it (stale worktree copies of the repo live there).
- After every task: `cargo test --workspace` must still pass (Tasks 1, 3, 9, 10 verify explicitly; other tasks touch no code).
- Line numbers cited into `README.md` refer to its current 889-line state (commit `ba62181`); Tasks 1–4 do not modify README, so they stay valid through Task 5. Verify with the quoted heading text before copying.

---

## Task 1: default-members fix (must-fix)

A bare `cargo build` / `cargo test` in a fresh clone currently compiles `nanoflann-ref` (C++17 via `cc`) and fails without g++/clang++. Published users are already safe; this fixes the contributor path. CI keeps full coverage because `ci.yml` already uses `--workspace` explicitly.

**Files:**
- Modify: `/home/sitzikbs/dev/flannrust/Cargo.toml`
- Modify: `/home/sitzikbs/dev/flannrust/CONTRIBUTING.md` (one-line note)

**Interfaces:** consumes nothing; produces a workspace whose default build set is pure Rust.

**Steps:**

- [ ] In `/home/sitzikbs/dev/flannrust/Cargo.toml`, change:

  ```toml
  default-members = ["crates/flannrust", "crates/nanoflann-ref", "crates/xval"]
  ```

  to:

  ```toml
  # Bare `cargo build`/`cargo test` in a fresh clone builds only the pure-Rust
  # library -- no C++17 compiler needed. Cross-validation against the vendored
  # C++ oracle: `cargo test --workspace` (what CI runs), or `cargo test -p xval`.
  default-members = ["crates/flannrust"]
  ```

- [ ] In `CONTRIBUTING.md`, near wherever it describes running tests, add one line (adapt placement to the file's existing structure, do not restructure it):

  ```markdown
  Note: bare `cargo test` builds/tests only the pure-Rust `flannrust` crate.
  Cross-validation against the vendored C++ oracle needs a C++17 compiler:
  `cargo test --workspace`.
  ```

- [ ] Commit: `chore: default-members = flannrust only, so fresh clones build without a C++ compiler`

**Verification:**

```bash
cargo clean && cargo build            # must succeed and NOT compile nanoflann-ref/cc
cargo test                            # flannrust only, green
cargo test --workspace --no-run       # full workspace still compiles (needs g++)
```

Confirm the first command's output contains no `nanoflann-ref` or `cc` compile lines.

---

## Task 2: LICENSE packaging + derivative-work attribution (must-fix)

`cargo package -p flannrust --list` does not currently include LICENSE. Also, the LICENSE frames upstream's notice as covering only the vendored header, but the Rust library is a derivative work of nanoflann — BSD-2-Clause condition 1 requires the upstream copyright notice on the derived source itself.

**Files:**
- Modify: `/home/sitzikbs/dev/flannrust/LICENSE` (copyright block)
- Create: `/home/sitzikbs/dev/flannrust/crates/flannrust/LICENSE` (copy)
- Create: `/home/sitzikbs/dev/flannrust/crates/flannrust-py/LICENSE` (copy)
- Modify: `/home/sitzikbs/dev/flannrust/crates/flannrust-py/pyproject.toml` (license SPDX string + license-files)

**Interfaces:** consumes root LICENSE; produces LICENSE files that cargo/maturin actually package.

**Steps:**

- [ ] In `/home/sitzikbs/dev/flannrust/LICENSE`, replace lines 1–3:

  ```
  BSD 2-Clause License

  Copyright (c) 2026 flannrust contributors
  ```

  with:

  ```
  BSD 2-Clause License

  Copyright (c) 2026 flannrust contributors
  Portions derived from nanoflann (https://github.com/jlblancoc/nanoflann):
  Copyright (c) 2008-2009 Marius Muja (mariusm@cs.ubc.ca)
  Copyright (c) 2008-2009 David G. Lowe (lowe@cs.ubc.ca)
  Copyright (c) 2011-2026 Jose Luis Blanco (joseluisblancoc@gmail.com)
  All rights reserved.
  ```

  Keep the entire rest of the file — conditions, disclaimer, and the verbatim vendored-header section — byte-for-byte unchanged.

- [ ] Copy the (now-updated) root LICENSE into both packaged crates. These are plain copies, not symlinks (symlinks break `cargo package` on some platforms and sdists on Windows):

  ```bash
  cp /home/sitzikbs/dev/flannrust/LICENSE /home/sitzikbs/dev/flannrust/crates/flannrust/LICENSE
  cp /home/sitzikbs/dev/flannrust/LICENSE /home/sitzikbs/dev/flannrust/crates/flannrust-py/LICENSE
  ```

  (cargo auto-includes a `LICENSE` file found in the package directory; no `include`/`license-file` key needed since `license = "BSD-2-Clause"` is already set. The flannrust-py copy is required because PEP 639 `license-files` paths must live inside the project directory — `../../LICENSE` is not allowed.)

- [ ] In `/home/sitzikbs/dev/flannrust/crates/flannrust-py/pyproject.toml`, replace:

  ```toml
  license = { text = "BSD-2-Clause" }
  ```

  with:

  ```toml
  license = "BSD-2-Clause"
  license-files = ["LICENSE"]
  ```

  and delete the now-redundant classifier line (PEP 639 deprecates license classifiers alongside an SPDX expression; maturin warns/errors on the combination):

  ```toml
      "License :: OSI Approved :: BSD License",
  ```

- [ ] Commit: `docs: LICENSE derivative-work attribution + ship LICENSE in crate and wheel`

**Verification:**

```bash
cargo package -p flannrust --list | grep LICENSE          # must print LICENSE
cd crates/flannrust-py && python -m venv /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/lic-venv \
  && . /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/lic-venv/bin/activate \
  && pip -q install maturin && maturin build --release -o /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheels \
  && python -c "import zipfile,glob; w=glob.glob('/tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheels/*.whl')[0]; print([n for n in zipfile.ZipFile(w).namelist() if 'licenses' in n or 'LICENSE' in n])"
```

The wheel listing must show a `*.dist-info/licenses/LICENSE` entry. Also `head -10 LICENSE` at all three locations must show the new Muja/Lowe/Blanco block.

---

## Task 3: cargo fmt pass + re-enable CI check (should-fix)

**Files:**
- Modify: every `.rs` file `cargo fmt --all` touches (formatting only)
- Modify: `/home/sitzikbs/dev/flannrust/.github/workflows/ci.yml`

**Interfaces:** produces an fmt-clean tree gated by CI.

**Steps:**

- [ ] Confirm clean working tree (`git status --porcelain` empty), then run:

  ```bash
  cargo fmt --all
  cargo fmt --all -- --check   # must now exit 0
  cargo test --workspace       # formatting must not have broken anything
  cargo clippy --workspace --all-targets -- -D warnings
  ```

- [ ] In `ci.yml`, replace the clippy-component step and the disabled-fmt NOTE block:

  ```yaml
        # Explicitly ensure clippy is present on the pinned 1.98.0 toolchain --
        # the dtolnay action above only installs components onto "stable",
        # not onto whatever the rust-toolchain.toml override resolves to.
        - name: Install clippy on the pinned toolchain
          run: rustup component add clippy

        # NOTE: `cargo fmt --check` is intentionally not run here. The
        # repository is not fmt-clean yet (verified locally on 2026-08-30),
        # and reformatting the tree is out of scope for this CI setup task.
        # Add this step back once a formatting pass has landed.
  ```

  with:

  ```yaml
        # Explicitly ensure clippy/rustfmt are present on the pinned 1.98.0
        # toolchain -- the dtolnay action above only installs components onto
        # "stable", not onto whatever the rust-toolchain.toml override
        # resolves to.
        - name: Install clippy and rustfmt on the pinned toolchain
          run: rustup component add clippy rustfmt

        - name: cargo fmt
          run: cargo fmt --all -- --check
  ```

- [ ] Commit everything from this task as ONE commit containing only formatting + the ci.yml change: `style: cargo fmt --all; re-enable fmt check in CI`

**Verification:**

```bash
cargo fmt --all -- --check && echo FMT-CLEAN
cargo test --workspace
git show --stat HEAD | tail -3   # sanity: only .rs files + ci.yml in the commit
```

---

## Task 4: .gitignore cleanup (nice-to-have)

**Files:**
- Modify: `/home/sitzikbs/dev/flannrust/.gitignore`

**Steps:**

- [ ] Append to `.gitignore`:

  ```
  .pytest_cache/
  .ruff_cache/
  .bench-venv/
  .ci-venv/
  *.pyd
  ```

- [ ] Commit: `chore: gitignore pytest/ruff caches, bench/ci venvs, *.pyd`

**Verification:**

```bash
git status --porcelain | grep -E "pytest_cache|ruff_cache" ; echo "exit=$?"   # expect exit=1 (no hits)
git check-ignore -v .pytest_cache .ruff_cache .bench-venv .ci-venv foo.pyd
```

---

## Task 5: docs extraction — docs/semantics.md and docs/testing.md (should-fix)

Extract the deep-reference content out of the current 889-line README into two docs files, **before** Task 7 replaces the README. Content is copied verbatim (Global Constraints: extraction, not rewriting); the only edits are the mechanical ones listed below. Source: `README.md` as on disk (unchanged since commit `ba62181`; cited line numbers refer to that state — verify each section by its quoted heading before copying).

**Files:**
- Create: `/home/sitzikbs/dev/flannrust/docs/semantics.md`
- Create: `/home/sitzikbs/dev/flannrust/docs/testing.md`
- (README itself is NOT modified in this task — Task 7 does that.)

**Interfaces:** consumes README sections; produces the two files the new README links to.

**Steps:**

- [ ] Create `docs/semantics.md` with this exact skeleton, where each `<<...>>` marker means "paste the named README section body verbatim":

  ```markdown
  # flannrust semantics

  Exact behavioral contracts of `flannrust`, and every place it deliberately
  differs from nanoflann 1.12.1's C++. Extracted from the project README;
  how these contracts are *verified* is documented in [testing.md](testing.md).

  ## Behavioral contracts
  <<README lines 82–113: body of "## Behavioral contracts" (intro paragraph + all bullets)>>

  ## Deliberate deviations from C++
  <<README lines 117–125: the deviations table>>

  ## Input domain
  <<README lines 129–145: body of "## Input domain">>

  ## Feature matrix
  <<README lines 149–161: body of "## Feature matrix">>

  ### The `Sync` escape hatch: `build_sequential()`
  <<README lines 164–174: body of that subsection>>

  ## Dynamic adaptor (`DynamicKdTree`)
  <<README lines 538–550: intro paragraph of "## Dynamic adaptor (M2)">>

  ### API overview
  <<README lines 554–605: the full API-overview bullet list>>

  ### Empty-forest `full()` quirk (inherited from C++, unchanged)
  <<README lines 609–617: body of that subsection>>

  ## Python API
  <<README lines 665–674: the bold squared-distance paragraph from "## Python bindings (M-py)">>

  <<README lines 706–722: the API table + the `workers`/`metric` paragraph under "### API">>
  ```

- [ ] Create `docs/testing.md` with this exact skeleton:

  ```markdown
  # flannrust testing & parity methodology

  How flannrust's bit-exact-parity claim is verified. The contracts being
  verified are documented in [semantics.md](semantics.md). Extracted from the
  project README.

  ## Safety (`unsafe` in this crate)
  <<README lines 178–195: body of "## Safety (`unsafe` in this crate)">>

  ## Parity & testing
  <<README lines 199–241: body of "## Parity & testing" (intro + bullets incl. matrix coverage)>>

  ### Dynamic (M2) cross-validation
  <<README lines 245–291: body of that subsection>>

  ## Python parity — scoped, not blanket (binding controller ruling)
  <<README lines 726–742: body of "### Parity — scoped, not blanket (binding controller ruling)">>
  ```

- [ ] Mechanical link fixes inside BOTH new files (they now live in `docs/`, so `docs/`-prefixed relative links must drop the prefix; repo-root files gain `../`). Run and then eyeball the diff:

  ```bash
  cd /home/sitzikbs/dev/flannrust
  sed -i 's|(docs/benchmarks\.md)|(benchmarks.md)|g; s|(docs/EXPERIMENTS\.md)|(EXPERIMENTS.md)|g; s|(docs/ROADMAP\.md)|(ROADMAP.md)|g; s|(docs/nanoflann-notes\.md)|(nanoflann-notes.md)|g; s|(LICENSE)|(../LICENSE)|g' docs/semantics.md docs/testing.md
  sed -i 's|`docs/benchmarks\.md`|`benchmarks.md`|g; s|`docs/EXPERIMENTS\.md`|`EXPERIMENTS.md`|g; s|`docs/ROADMAP\.md`|`ROADMAP.md`|g; s|`docs/nanoflann-notes\.md`|`nanoflann-notes.md`|g' docs/semantics.md docs/testing.md
  ```

- [ ] Fix the "above/below" cross-references that no longer point within the same file (exact replacements):
  - In `docs/testing.md`, the Parity & testing tie-latitude bullet's `(see "Deliberate deviations" above)` → `(see [semantics.md](semantics.md)'s "Deliberate deviations")`.
  - In `docs/testing.md`, the Safety section is otherwise self-contained — leave its EXPERIMENTS/ROADMAP references (already fixed by the sed).
  - In `docs/semantics.md`, the Behavioral-contracts intro's `the "Deliberate deviations" table below` still points within the file — leave it. The dynamic API-overview's `see "Dynamic adaptor (M2)" above` (inside the Python API table, box-search row) → `see the "Dynamic adaptor" section above`. Any remaining `see "Parity & testing" below`-style reference in semantics.md → `see [testing.md](testing.md)`.
  - Sweep for stragglers: `grep -n '" above\|" below\|(M2)\|(M-py)' docs/semantics.md docs/testing.md` and resolve each hit (either an intra-file reference that still works, or point it at the sibling file/section as above; drop milestone tags like "(M2)"/"(M2.5)" from headings only, never from body prose citing evidence).

- [ ] Commit: `docs: extract README semantics + testing content into docs/semantics.md, docs/testing.md`

**Verification:**

```bash
wc -l docs/semantics.md docs/testing.md      # expect roughly 230-280 and 130-180
grep -n "docs/docs\|(docs/" docs/semantics.md docs/testing.md ; echo "exit=$?"   # expect exit=1
# spot-check: every heading present
grep -c "^## \|^### " docs/semantics.md docs/testing.md
```

---

## Task 6: agentic docs reorganization (should-fix)

Move process/agent documentation under `docs/agentic-development/` with `git mv` (history-preserving), add its README, fix every cross-reference. Note this moves the current plan and spec files themselves — that is expected; the sed below fixes their internal references too.

**Files:**
- Move: `docs/superpowers/` → `docs/agentic-development/` (contains `plans/`, `specs/`)
- Move: `docs/reports/` → `docs/agentic-development/reports/`
- Create: `docs/agentic-development/README.md`
- Modify (path references only): `README.md`, `CONTRIBUTING.md`, `docs/ROADMAP.md`, `docs/benchmarks.md`, `docs/EXPERIMENTS.md`, and the moved plans/specs/reports themselves

**Steps:**

- [ ] Move with git:

  ```bash
  cd /home/sitzikbs/dev/flannrust
  git mv docs/superpowers docs/agentic-development
  git mv docs/reports docs/agentic-development/reports
  ```

- [ ] Create `docs/agentic-development/README.md` with exactly this content (the long "How this was built" text relocated from the old README, plus a directory map):

  ```markdown
  # Agentic development record

  This codebase was implemented by an AI agent (Claude Code), directed and
  reviewed by Itzik Ben-Shabat, who does not write Rust. Every line of Rust,
  C++ FFI, and Python-binding code here is agent-written; the author's role
  was specifying requirements, reviewing the generated code and each task's
  report, and directing dedicated rigor/fidelity-audit passes (the plans and
  reports below). Because the author cannot independently vet Rust idioms or
  catch subtle logic errors by reading the code, correctness here does not
  rest on the author's Rust expertise — it rests on the bit-exact
  cross-validation suite run against the vendored nanoflann 1.12.1 C++
  reference implementation (see [../testing.md](../testing.md)), which checks
  every result index, distance, and internal tree permutation against the
  real C++ library, in-process, on every change. Every performance figure
  quoted in the README and [../benchmarks.md](../benchmarks.md) traces to a
  reproducible, pasted command and run recorded in
  [../EXPERIMENTS.md](../EXPERIMENTS.md) or in the commit-tagged report data
  cited alongside the figure; see
  [reports/m-pub/claims-audit.md](reports/m-pub/claims-audit.md) for the
  audit that verified that claim.

  ## Layout

  - [`plans/`](plans/) — per-milestone implementation plans (M1 static tree,
    M2 dynamic forest, M2.5 perf, M2.6 rigor, M-py bindings, M-pub, M-ship).
  - [`specs/`](specs/) — design/requirement specs the plans implement.
  - [`reports/`](reports/) — per-task worker/reviewer reports and audits
    (m2.5, m2.6, m-py, m-pub), the primary evidence chain behind the
    numbers in [../benchmarks.md](../benchmarks.md).
  ```

- [ ] Fix every cross-reference outside the ignored worktrees (the two patterns are disjoint, so a single sed pass is safe):

  ```bash
  cd /home/sitzikbs/dev/flannrust
  grep -rl "docs/superpowers\|docs/reports" --include="*.md" . | grep -v ".claude/worktrees" | \
    xargs sed -i 's|docs/reports/|docs/agentic-development/reports/|g; s|docs/superpowers/|docs/agentic-development/|g'
  ```

- [ ] `grep -rn "docs/superpowers\|docs/reports" . --include="*.md" | grep -v ".claude/worktrees"` must return nothing. Also check non-md references: `grep -rn "docs/superpowers\|docs/reports" --include="*.toml" --include="*.yml" --include="*.rs" . | grep -v ".claude/worktrees"` (expected: none today).

- [ ] Commit: `docs: move superpowers/ and reports/ under docs/agentic-development/, add its README`

**Verification:**

```bash
ls docs/agentic-development           # plans  specs  reports  README.md
git log --oneline -1 --stat | head    # renames, not delete+add
grep -rn "docs/superpowers\|docs/reports" --include="*.md" . | grep -v ".claude/worktrees" ; echo "exit=$?"   # expect exit=1
```

---

## Task 7: README rewrite (should-fix)

Replace the 889-line README with this ~140-line landing page. It becomes the crates.io AND PyPI front page (both `readme` keys point at it). Everything removed already lives in `docs/semantics.md`, `docs/testing.md`, `docs/benchmarks.md`, `docs/EXPERIMENTS.md`, `docs/ROADMAP.md`, or `docs/agentic-development/` — nothing is lost.

**Files:**
- Rewrite: `/home/sitzikbs/dev/flannrust/README.md`

**Steps:**

- [ ] Replace the full contents of `README.md` with exactly:

  ````markdown
  # flannrust

  [![CI](https://github.com/sitzikbs/flannrust/actions/workflows/ci.yml/badge.svg)](https://github.com/sitzikbs/flannrust/actions/workflows/ci.yml)
  [![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
  <!-- At publish time, add: crates.io, docs.rs, and PyPI version badges. -->

  A Rust port of [nanoflann](https://github.com/jlblancoc/nanoflann) (the C++
  kd-tree library), targeting bit-exact result parity with the C++ reference
  and equal-or-better speed. Static and dynamic indexes, with Python bindings.

  ## Features

  - **Bit-exact parity with nanoflann 1.12.1** — every result index, distance,
    and internal tree permutation is cross-validated in-process against the
    vendored C++ reference on every commit ([docs/testing.md](docs/testing.md))
  - **Static kd-tree** (`KdTree`) and **dynamic** Bentley–Saxe forest
    (`DynamicKdTree`) with point add/remove after construction
  - **Matches or beats the C++** on most benchmarked workloads (table below)
  - L1 / L2 / L2-simple / SO2 / SO3 metrics; `f32`/`f64`; compile-time
    (`ConstDim`) or runtime (`DynDim`) dimension; `u32`/`u64`/`usize` indices
  - Optional parallel build via rayon (default feature `parallel`)
  - Python bindings: `flannrust.KDTree` / `DynamicKDTree`, NumPy in/out,
    GIL released during build and query
  - Exactly two `unsafe` blocks, both miri-verified in CI

  ## Installation

  Not yet published to crates.io / PyPI — until then, use git / build from source.

  Rust:

  ```toml
  [dependencies]
  flannrust = { git = "https://github.com/sitzikbs/flannrust" }
  ```

  Python (from a clone, inside a virtualenv):

  ```bash
  pip install maturin numpy
  maturin develop -m crates/flannrust-py/Cargo.toml --release
  ```

  MSRV: Rust 1.98.0 (pinned in `rust-toolchain.toml`).

  ## Quick start

  ### Rust

  ```rust
  use flannrust::{ConstDim, KdTreeBuilder};

  let pts: &[[f64; 3]] = &[
      [0.0, 0.0, 0.0],
      [10.0, 10.0, 10.0],
      [1.0, 1.0, 1.0],
  ];
  let tree = KdTreeBuilder::new(ConstDim::<3>, pts).build();

  let mut indices = [0u32; 2];
  let mut dists = [0.0f64; 2];
  let found = tree.knn_search(&[0.1, 0.1, 0.1], &mut indices, &mut dists);

  assert_eq!(found, 2);
  assert_eq!(indices[0], 0); // nearest point is [0.0, 0.0, 0.0]
  ```

  ### Python

  ```python
  import numpy as np
  import flannrust

  pts = np.random.default_rng(0).uniform(-10, 10, size=(100_000, 3)).astype(np.float32)
  tree = flannrust.KDTree(pts, leaf_size=10, metric="l2", threads=None)

  q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
  dists, idxs = tree.query(q, k=5)               # dists are SQUARED l2
  r_idxs, r_dists = tree.query_radius(q, r=4.0)  # r is SQUARED too, strict `<`

  dyn = flannrust.DynamicKDTree(dim=3, dtype="float32")
  dyn.add_points(pts)
  dyn.remove_point(0)                            # lazy tombstone
  dists, idxs = dyn.query(q, k=5)
  ```

  > **Distances and radii are SQUARED** for `l2`/`l2_simple` — unlike
  > `scipy.spatial.cKDTree`. Square your radius before calling; expect squared
  > values back (`l1` is an unsquared sum of absolute differences). This is
  > the single most common mistake porting code from `cKDTree`.

  ## Performance

  Six gated workloads vs. the vendored C++ oracle, ratio = Rust time / C++
  time (lower is better for Rust). Latest idle-host re-measurement (M2.6,
  3 sessions, statistical harness):

  | Workload | Ratio (Rust / C++) |
  |---|---|
  | build 100k, dim 3, f32, sequential | 0.99–1.01 |
  | knn, fixed dim 3, f32, k=10 | 1.01–1.04 |
  | knn, runtime dim 8, f64, k=10 | 0.93–0.94 |
  | radius, dim 3, f32 | 0.83–0.87 |
  | dynamic add 20k, dim 3, f32 | 1.03–1.04 |
  | dynamic knn after churn, dim 3, f32 | 0.93–0.96 |

  Measured on WSL2, AMD Ryzen 7 9800X3D, `rustc 1.98.0`, `-C target-cpu=native`
  vs. C++ `-O3 -march=native -ffp-contract=off`. Full methodology, history,
  honest residuals, and a portable repro kit:
  [docs/benchmarks.md](docs/benchmarks.md),
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
  [docs/benchkit.md](docs/benchkit.md).

  ## Documentation

  - API docs: `cargo doc -p flannrust --open` (docs.rs after publish);
    Python docstrings on every class/method
  - [docs/semantics.md](docs/semantics.md) — exact behavioral contracts,
    deliberate deviations from C++, input domain, feature flags, dynamic
    adaptor and Python API details
  - [docs/testing.md](docs/testing.md) — how bit-exact parity is verified
    (cross-validation matrix, dynamic op-sequence suite, miri, canaries)
  - [docs/benchmarks.md](docs/benchmarks.md) / [docs/benchkit.md](docs/benchkit.md)
    — the numbers and how to reproduce them on your hardware
  - [docs/ROADMAP.md](docs/ROADMAP.md) — what's next (M3 incremental adaptor,
    M4 multithreaded wrapper, serialization)
  - [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup; note that
    cross-validation against the C++ oracle needs a C++17 compiler
    (`cargo test --workspace`)

  ## How this was built

  Every line of Rust, C++ FFI, and Python-binding code here was written by an
  AI agent (Claude Code), directed and reviewed by Itzik Ben-Shabat.
  Correctness does not rest on human code review — it rests on the bit-exact
  cross-validation suite run against the real C++ library on every change,
  and every performance figure traces to a pasted, reproducible run. The full
  process record — plans, specs, and per-task reports:
  [docs/agentic-development/](docs/agentic-development/).

  ## License

  BSD-2-Clause — see [LICENSE](LICENSE). flannrust is a derivative work of
  [nanoflann](https://github.com/jlblancoc/nanoflann) by Jose Luis
  Blanco-Claraco et al., which builds on FLANN by Marius Muja and David G.
  Lowe; the upstream copyright notices are retained in LICENSE. The vendored
  `nanoflann.hpp` (used only as a test/benchmark oracle, not part of the Rust
  library) keeps its original license header verbatim.
  ````

- [ ] Refresh the two LICENSE copies made in Task 2? **No** — LICENSE didn't change here. But DO re-check that the Rust quickstart block is still byte-identical to the lib.rs doctest (`crates/flannrust/src/lib.rs` crate docs) — it must stay a truthful mirror.
- [ ] Commit: `docs: rewrite README as a ~140-line landing page`

**Verification:**

```bash
wc -l README.md                                  # target 130-160
# every relative link resolves:
grep -oP '\]\(\K[^)#]+' README.md | grep -v '^http' | sort -u | while read -r f; do [ -e "$f" ] || echo "BROKEN: $f"; done
cargo test -p flannrust --doc                    # doctest (source of the quickstart) still green
```

---

## Task 8: pyproject.toml + wheels.yml polish (nice-to-have)

**Files:**
- Modify: `/home/sitzikbs/dev/flannrust/crates/flannrust-py/pyproject.toml`
- Modify: `/home/sitzikbs/dev/flannrust/.github/workflows/wheels.yml`

**Steps:**

- [ ] Rewrite `crates/flannrust-py/pyproject.toml` `[project]` section to (Task 2 already changed the license keys — this builds on that state; the version now comes from `crates/flannrust-py/Cargo.toml`, killing the drift risk):

  ```toml
  [build-system]
  requires = ["maturin>=1.14,<2.0"]
  build-backend = "maturin"

  [project]
  name = "flannrust"
  dynamic = ["version"]
  description = "Bit-exact Rust port of nanoflann's kd-tree, Python bindings"
  requires-python = ">=3.9"
  license = "BSD-2-Clause"
  license-files = ["LICENSE"]
  authors = [{name = "Itzik Ben-Shabat"}]
  readme = "../../README.md"
  classifiers = [
      "Development Status :: 4 - Beta",
      "Programming Language :: Rust",
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "Programming Language :: Python :: 3.12",
      "Programming Language :: Python :: 3.13",
      "Topic :: Scientific/Engineering",
      "Operating System :: OS Independent",
  ]

  [project.urls]
  Repository = "https://github.com/sitzikbs/flannrust"
  Documentation = "https://github.com/sitzikbs/flannrust"

  [tool.maturin]
  module-name = "flannrust"
  features = ["pyo3/extension-module"]
  ```

  Note: maturin reads the dynamic version from `crates/flannrust-py/Cargo.toml` (`version = "0.1.0"`). At release time that version must be bumped in lockstep with `crates/flannrust/Cargo.toml` — add this comment above `version` in `crates/flannrust-py/Cargo.toml`:

  ```toml
  # Single source of the PyPI package version (pyproject.toml declares
  # `dynamic = ["version"]`). Bump together with crates/flannrust's version.
  ```

- [ ] Append an sdist job to `.github/workflows/wheels.yml` (top-level under `jobs:`, sibling of `build`):

  ```yaml
    sdist:
      name: sdist
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4

        # maturin bundles local path dependencies (the flannrust crate) into
        # the sdist, so `pip install flannrust-<ver>.tar.gz` works from a
        # machine with only a Rust toolchain.
        - uses: PyO3/maturin-action@v1
          with:
            command: sdist
            args: -m crates/flannrust-py/Cargo.toml

        - uses: actions/upload-artifact@v4
          with:
            name: sdist
            path: target/wheels/*.tar.gz
  ```

- [ ] Commit: `chore: pyproject dynamic version + classifiers; sdist job in wheels.yml`

**Verification:**

```bash
cd crates/flannrust-py
. /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/lic-venv/bin/activate 2>/dev/null || { python -m venv /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/lic-venv && . /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/lic-venv/bin/activate && pip -q install maturin; }
maturin build --release -o /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheels2   # wheel version must read 0.1.0 (from Cargo.toml)
maturin sdist -o /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/sdist
tar tzf /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/sdist/*.tar.gz | grep -E "LICENSE|flannrust/src/lib.rs|PKG-INFO"
python - <<'EOF'
import glob, zipfile
w = glob.glob('/tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheels2/*.whl')[0]
meta = [n for n in zipfile.ZipFile(w).namelist() if n.endswith('METADATA')][0]
print(zipfile.ZipFile(w).read(meta).decode()[:800])
EOF
```

METADATA must show `Version: 0.1.0`, `License-Expression: BSD-2-Clause`, `License-File: LICENSE`, and the new classifiers. The sdist must contain the bundled `flannrust` crate sources plus LICENSE.

---

## Task 9: nice-to-haves — example, CHANGELOG, MSRV comment, documentation field

**Files:**
- Create: `/home/sitzikbs/dev/flannrust/crates/flannrust/examples/knn.rs`
- Create: `/home/sitzikbs/dev/flannrust/CHANGELOG.md`
- Modify: `/home/sitzikbs/dev/flannrust/crates/flannrust/Cargo.toml`
- Modify: `/home/sitzikbs/dev/flannrust/rust-toolchain.toml`

**Steps:**

- [ ] Create `crates/flannrust/examples/knn.rs` with exactly:

  ```rust
  //! Build a static kd-tree, then run a knn and a radius query.
  //!
  //! Run with: `cargo run --example knn`

  use flannrust::{ConstDim, KdTreeBuilder, ResultItem};

  fn main() {
      // 1000 deterministic pseudo-random 3-D points in [-10, 10)^3
      // (xorshift64 -- no dependencies needed for an example).
      let mut state: u64 = 42;
      let mut next = move || {
          state ^= state << 13;
          state ^= state >> 7;
          state ^= state << 17;
          (state >> 11) as f64 / (1u64 << 53) as f64 * 20.0 - 10.0
      };
      let pts: Vec<[f64; 3]> = (0..1000).map(|_| [next(), next(), next()]).collect();

      let tree = KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();

      // 5 nearest neighbors of the origin. NOTE: distances are SQUARED for L2.
      let query = [0.0, 0.0, 0.0];
      let mut indices = [0u32; 5];
      let mut dists = [0.0f64; 5];
      let found = tree.knn_search(&query, &mut indices, &mut dists);
      println!("knn: {found} nearest neighbors of {query:?}:");
      for (i, d) in indices.iter().zip(&dists) {
          println!("  index {i:4}  squared distance {d:.6}");
      }

      // Every point with SQUARED distance < 4.0 (i.e. within radius 2.0),
      // sorted by distance. Strictly `dist < radius`.
      let mut out: Vec<ResultItem<u32, f64>> = Vec::new();
      let n = tree.radius_search(&query, 4.0, &mut out);
      println!("radius: {n} points within squared distance 4.0");
      for item in out.iter().take(5) {
          println!("  index {:4}  squared distance {:.6}", item.index, item.distance);
      }
  }
  ```

  If `cargo run --example knn` fails to compile because a signature differs from the above (e.g. `ResultItem` field names), fix the example to match the real API — never the library.

- [ ] Create `CHANGELOG.md` with exactly:

  ```markdown
  # Changelog

  All notable changes to flannrust (the Rust crate and the Python package —
  they version in lockstep) are documented here. Format:
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning:
  [SemVer](https://semver.org/spec/v2.0.0.html).

  ## [Unreleased]

  ## [0.1.0] - TBD

  Initial release.

  - Static kd-tree (`KdTree`): bit-exact port of nanoflann 1.12.1's
    `KDTreeSingleIndexAdaptor` — knn / rknn / radius / box search.
  - Dynamic Bentley–Saxe forest (`DynamicKdTree`): port of
    `KDTreeSingleIndexDynamicAdaptor` — point add/remove after construction.
  - L1 / L2 / L2-simple / SO2 / SO3 metrics; `f32`/`f64`; compile-time or
    runtime dimension; `u32`/`u64`/`usize` indices; optional rayon-parallel
    build (feature `parallel`, on by default).
  - Python bindings (PyPI package `flannrust`): `KDTree` / `DynamicKDTree`,
    NumPy in/out, GIL released during build/query. Distances and radii are
    SQUARED for l2 metrics (see README).
  ```

  (Set the date when the tag is actually cut — that is out of this plan's scope.)

- [ ] In `crates/flannrust/Cargo.toml`, add below the `repository` line:

  ```toml
  documentation = "https://docs.rs/flannrust"
  ```

- [ ] In `rust-toolchain.toml`, add the MSRV-coupling comment:

  ```toml
  [toolchain]
  # Pinned to exactly the MSRV declared as `rust-version` in
  # crates/flannrust/Cargo.toml (and flannrust-py's). This makes CI's test job
  # an implicit MSRV check -- when bumping this channel, bump `rust-version`
  # in both crates (or consciously keep MSRV lower and add a real MSRV CI job).
  channel = "1.98.0"
  ```

- [ ] Commit: `chore: knn example, CHANGELOG stub, docs.rs link, MSRV-coupling comment`

**Verification:**

```bash
cargo run --example knn -p flannrust        # runs, prints 5 neighbors + radius hits
cargo package -p flannrust --list | grep -E "examples/knn.rs|LICENSE"   # both present
cargo build                                  # toolchain file still parses
```

---

## Task 10: full verification pass

No new changes — prove the whole milestone holds together. Fix anything that fails, amending the responsible task's commit style (`docs:`/`chore:` follow-up commit is fine).

**Steps:**

- [ ] Run, in order, all must pass:

  ```bash
  cd /home/sitzikbs/dev/flannrust
  cargo clean
  cargo build                                        # default-members: pure Rust, no C++ compile lines
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace                             # full parity suite (needs g++)
  cargo doc -p flannrust --no-deps                   # zero warnings
  cargo run --example knn -p flannrust
  cargo package -p flannrust --list                  # must list: LICENSE, README, examples/knn.rs, src/*
  cargo publish -p flannrust --dry-run
  ```

- [ ] Link check across all touched markdown (README + docs/*.md + docs/agentic-development/README.md):

  ```bash
  for md in README.md CHANGELOG.md docs/semantics.md docs/testing.md docs/agentic-development/README.md; do
    dir=$(dirname "$md")
    grep -oP '\]\(\K[^)#]+' "$md" | grep -v '^http' | sort -u | while read -r f; do
      [ -e "$dir/$f" ] || [ -e "$f" ] || echo "BROKEN in $md: $f"
    done
  done
  ```

  Expect no `BROKEN` lines.

- [ ] Residual-reference sweep: `grep -rn "docs/superpowers\|docs/reports" --include="*.md" --include="*.toml" --include="*.yml" . | grep -v ".claude/worktrees"` → empty.

- [ ] CI dry-run of the two changed workflow bits locally: the fmt step (`cargo fmt --all -- --check`, already run above) and the python job's core (`maturin develop -m crates/flannrust-py/Cargo.toml --release && pytest crates/flannrust-py/python/tests` inside a venv) — pytest must stay green (348 tests at last count; pynanoflann-parity tests may be skipped locally if `pynanoflann` isn't installed — that's acceptable, CI runs them).

- [ ] Confirm the branch's commit list reads as one commit per task (plus this plan/spec's own commit) and push for CI to give the final verdict.

**Done when:** every command above passes, CI is green on the pushed branch, and `cargo package -p flannrust --list` shows LICENSE + example + README.
