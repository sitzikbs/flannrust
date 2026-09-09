# M-ship: Shipping Readiness Spec

**Source audit:** shipping-readiness audit of commit `5f84f4e` (branch `m2p6-rigor`), plus
the README/docs redesign plan. Goal: everything needed so `cargo publish -p flannrust`
and a PyPI upload of `flannrust` ship a correct, well-presented package — without
actually publishing (real publish remains deferred, per M-pub).

## Must-fix (blocks publish)

1. **LICENSE not packaged.** `cargo package -p flannrust --list` does not include the
   LICENSE (only README via the relative `readme` path). Fix: copy the root LICENSE
   (including the nanoflann attribution block) into `crates/flannrust/`. Same gap for
   the Python wheel: `pyproject.toml` uses `license = { text = "BSD-2-Clause" }`, which
   embeds no license file — switch to the SPDX string + `license-files`, with a LICENSE
   copy in `crates/flannrust-py/`.
2. **LICENSE derivative-work attribution.** The Rust library is a deliberate
   line-for-line port of nanoflann's C++ — a derivative work. BSD-2-Clause condition 1
   requires upstream's copyright notice on the derived source, not just on the vendored
   header. Fix: extend the LICENSE copyright block with the Muja/Lowe/Blanco lines
   (exact text in the plan). Keep the verbatim vendored-header section unchanged.
3. **`default-members` requires a C++ compiler.** Workspace `default-members` includes
   `nanoflann-ref` and `xval`, so a bare `cargo build`/`cargo test` in a fresh clone
   fails without g++/clang++. Fix: `default-members = ["crates/flannrust"]`.
   Published users are already safe (nanoflann-ref is `publish = false` and not in
   flannrust's dep graph); this is contributor-facing only. CI is unaffected — it
   already passes `--workspace` explicitly.

## Should-fix

4. **`cargo fmt` pass.** The repo is not fmt-clean; the CI fmt check is disabled with a
   dated comment. Run `cargo fmt --all` as a single formatting-only commit, re-enable
   the check in `ci.yml`.
5. **README trim + docs redesign** (from the README/docs plan):
   - Rewrite `README.md` from 889 lines to ~140: title, badges (CI + license now;
     crates.io/docs.rs/PyPI at publish), features, installation, quickstart (Rust +
     Python), the squared-distance warning, ONE current performance table (M2.6 task 6
     idle-host ranges), documentation links, a brief "How this was built", license
     section with derivative-work attribution.
   - Create `docs/semantics.md` (behavioral contracts, deliberate deviations, input
     domain, feature matrix + Sync escape hatch, dynamic adaptor API, empty-forest
     quirk, Python API table + conventions) and `docs/testing.md` (safety/unsafe/miri,
     parity methodology, xval matrix, dynamic op-sequence suite, canaries, Python
     parity scoping) — content **extracted** from the current README, lightly deduped,
     not rewritten.
   - The "Update (M2.6 task N)" chains, success-criteria bookkeeping, and roadmap prose
     are **cut**, not re-homed — `docs/benchmarks.md`, `docs/EXPERIMENTS.md`, and
     `docs/ROADMAP.md` already contain all of it.
   - Move `docs/superpowers/` → `docs/agentic-development/` and `docs/reports/` →
     `docs/agentic-development/reports/` (`git mv`), add a short
     `docs/agentic-development/README.md`, fix all cross-references.

## Nice-to-have

6. `CHANGELOG.md` stub (Keep a Changelog, `## 0.1.0` unreleased entry).
7. `crates/flannrust/examples/knn.rs` (~35 lines: build, knn, radius) so
   `cargo run --example knn` works.
8. `.gitignore`: add `.pytest_cache/`, `.ruff_cache/`, `.bench-venv/`, `.ci-venv/`,
   `*.pyd`.
9. `pyproject.toml`: `dynamic = ["version"]` (version from Cargo.toml, kills the drift
   risk), `Development Status :: 4 - Beta` + per-version Python classifiers.
10. `wheels.yml`: add an sdist build step alongside the wheel matrix.
11. `Cargo.toml`: `documentation = "https://docs.rs/flannrust"`; MSRV-coupling comment
    (rust-toolchain.toml channel == rust-version — bump together).

## Not in scope

- Actually running `cargo publish` or uploading to PyPI (deferred, as in M-pub).
- Any change to library/binding **code** (src/), benchmarks, tests, or performance
  numbers. Formatting via `cargo fmt` is the only permitted source diff.
- Rewriting docs/benchmarks.md, docs/EXPERIMENTS.md, docs/ROADMAP.md,
  docs/nanoflann-notes.md content (path-reference fixes only).
- A `[features] oracle` split in xval, an MSRV CI job, PyPI trusted-publishing
  automation, README badges for registries the package isn't on yet.
- Deleting any existing doc — the slop is duplication in README, not the docs.
