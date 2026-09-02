# M-pub — Announcement Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make flannrust publicly announceable: green CI on GitHub, verified packaging (dry-run only), a portable bench kit for multi-host numbers, a claims audit so every published number traces to a run, and a blog post for itzikbs.com.

**Architecture:** No library code changes. Everything is infrastructure (GitHub Actions, packaging metadata), tooling (a host-spec-capturing bench runner), and documentation (claims audit, blog post). The blog post lives in the separate `~/dev/my_website` Eleventy repo on a branch; everything else lives in this repo.

**Tech Stack:** GitHub Actions, maturin + maturin-action, cargo publish --dry-run, Python (bench kit), Eleventy markdown + inline-referenced SVG.

**Spec:** This section. User decisions (2026-08-30): GitHub repo **private now, public at announce** (`github.com/sitzikbs/flannrust`); PyPI/crates.io **dry-run only** — no real publish; blog post for **the personal Eleventy site** (`~/dev/my_website`); bench numbers **WSL2 primary (fully disclosed)** with a portable bench kit so MacBook Air / native Windows runs can be added later — blog not blocked on them; blog MUST be honest that **the user does not write Rust and directed an AI agent (Claude Code) that implemented it** — that is the hook, not a footnote.

## Global Constraints

- No changes to library behavior. `cargo test --workspace` green at every commit (390 tests today).
- License is BSD-2-Clause everywhere (matches nanoflann); vendored `nanoflann.hpp` keeps its original header.
- Repo URL everywhere: `https://github.com/sitzikbs/flannrust`.
- Only `crates/flannrust` is publishable; `nanoflann-ref`, `xval`, `flannrust-py` stay `publish = false` (flannrust-py publishes to PyPI via maturin, not crates.io).
- No real publishes to PyPI or crates.io. Dry-run / local-build verification only.
- Benchmark claims: every number in README/docs/blog traces to a pasted run or a report JSON; WSL2 + hardware disclosed wherever numbers appear.
- Commits end with the standard trailers (Co-Authored-By Claude Fable 5 + Claude-Session).
- Blog repo (`~/dev/my_website`) work happens on branch `flannrust-post`, never on its main.
- cargo on PATH via `export PATH="$HOME/.cargo/bin:$PATH"`. Python venv for bindings: `crates/flannrust-py/.venv`.

---

### Task 1: Packaging metadata + repo hygiene

**Files:**
- Modify: `crates/flannrust/Cargo.toml` (fill `repository`, add `keywords`, `categories`, `readme`)
- Modify: `crates/flannrust-py/pyproject.toml` (authors, urls, classifiers, readme)
- Create: `CONTRIBUTING.md`
- Verify: `LICENSE`, `crates/nanoflann-ref/cpp/nanoflann.hpp` header intact

**Interfaces:**
- Produces: metadata that `cargo publish --dry-run` (Task 3) and `gh repo create` (Task 6) rely on. Exact repo URL: `https://github.com/sitzikbs/flannrust`.

- [ ] **Step 1: Fill flannrust Cargo.toml metadata.** In `[package]`: uncomment/set `repository = "https://github.com/sitzikbs/flannrust"`, add `keywords = ["kdtree", "nearest-neighbor", "nanoflann", "spatial", "knn"]`, `categories = ["algorithms", "data-structures", "science"]`, `readme = "../../README.md"` — check `cargo package --list -p flannrust` accepts the readme path; if it complains, copy README.md into the crate dir instead and note it.
- [ ] **Step 2: pyproject.toml metadata.** Add under `[project]`: `authors = [{name = "Itzik Ben-Shabat"}]`, `classifiers` (Programming Language :: Rust, Programming Language :: Python :: 3, License :: OSI Approved :: BSD License, Topic :: Scientific/Engineering, Operating System :: OS Independent), `[project.urls]` with `Repository` and `Documentation` pointing at the GitHub repo. Add `description` and `readme` if missing.
- [ ] **Step 3: Write CONTRIBUTING.md** — short: dev setup (rust-toolchain pinned, venv, maturin develop), test commands (`cargo test --workspace`, pytest), parity rule (bit-exact vs vendored nanoflann 1.12.1 — xval suite is the judge; any behavior change must keep xval green), benchmark protocol pointer to `docs/EXPERIMENTS.md` measurement-conditions section.
- [ ] **Step 4: Verify licensing.** `LICENSE` is BSD-2-Clause with nanoflann attribution for the port; vendored header untouched. Fix only if wrong.
- [ ] **Step 5: Verify** `cargo package --list -p flannrust` succeeds and `cargo test --workspace` still green. Commit `chore: packaging metadata + CONTRIBUTING for public release`.

### Task 2: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/wheels.yml`

**Interfaces:**
- Consumes: nothing from other tasks (metadata from T1 not required for CI).
- Produces: workflows that Task 6 pushes and must turn green.

- [ ] **Step 1: Write `ci.yml`.** Trigger: push + pull_request on main. Jobs:
  - `test` (ubuntu-latest): checkout, dtolnay/rust-toolchain@stable (repo pins via rust-toolchain.toml), `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo build -p flannrust --no-default-features`. Needs a C++ toolchain for nanoflann-ref — ubuntu runners have g++, nothing extra.
  - `miri` (ubuntu-latest, REQUIRED — same workflow so it gates merges): nightly toolchain with miri component, `cargo +nightly miri test -p flannrust --lib`. flannrust is pure Rust (no FFI) so miri works; xval/nanoflann-ref are out of miri scope. Use `MIRIFLAGS="-Zmiri-disable-isolation"` only if tests need it; try without first.
  - `python` (ubuntu-latest): setup-python 3.11, `pip install maturin numpy pytest scipy`, `maturin develop -m crates/flannrust-py/Cargo.toml --release`, `pytest crates/flannrust-py/python/tests`. Skip pynanoflann-dependent tests if that install is flaky: `pip install pynanoflann` and if it fails on CI, mark that step `continue-on-error: false` first and let Task 6 decide based on the actual failure.
- [ ] **Step 2: Write `wheels.yml`.** Trigger: workflow_dispatch + tags `v*`. Matrix: ubuntu-latest, macos-14, windows-latest. Steps: PyO3/maturin-action@v1 with `args: --release -m crates/flannrust-py/Cargo.toml`, abi3 wheel (already abi3-py39 in the crate), upload-artifact per OS. NO publish step — artifacts only.
- [ ] **Step 3: Lint workflows** with `actionlint` if available (`command -v actionlint`), else careful YAML read-through + `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` for both files.
- [ ] **Step 4: Commit** `ci: GitHub Actions — test/clippy/miri/python + wheel matrix`.

### Task 3: Packaging dry-runs

**Files:**
- Modify (only if dry-runs demand): `crates/flannrust/Cargo.toml`, `crates/flannrust-py/pyproject.toml`
- Create: `docs/reports/m-pub/packaging-dryrun.md` (pasted outputs)

**Interfaces:**
- Consumes: Task 1 metadata.
- Produces: evidence report other tasks cite; confirmed-free package names.

- [ ] **Step 1:** `cargo publish --dry-run -p flannrust --allow-dirty` — must succeed. Paste tail of output into the report. Fix metadata if it rejects.
- [ ] **Step 2: Name availability.** `curl -s https://crates.io/api/v1/crates/flannrust | head -c 200` (expect not-found) and `curl -s https://pypi.org/pypi/flannrust/json | head -c 200` (expect 404). Paste both.
- [ ] **Step 3: Local wheel.** `cd crates/flannrust-py && .venv/bin/maturin build --release` (install maturin into venv if absent). Then fresh venv smoke test: `python3 -m venv /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheeltest-venv`, pip install the built wheel + numpy, run `python -c "import flannrust, numpy as np; t=flannrust.KDTree(np.random.rand(100,3).astype(np.float32)); d,i=t.query(np.random.rand(5,3).astype(np.float32), k=3); print(d.shape, i.shape)"`. Paste output.
- [ ] **Step 4:** Write `docs/reports/m-pub/packaging-dryrun.md` with all pasted evidence. Commit `chore: packaging dry-run evidence (crates.io + wheel smoke test)`.

### Task 4: Portable bench kit

**Files:**
- Create: `scripts/benchkit.py`
- Create: `docs/benchkit.md`

**Interfaces:**
- Consumes: existing `cargo run -p xval --release --example report_data` and `crates/flannrust-py/python/bench/bench_py.py`.
- Produces: `benchkit.py` emitting `bench-<hosttag>-<date>.json` bundle `{host: {...specs...}, rust_report: <report_data JSON>, python_report: <bench_py JSON or null>}`. Blog task (7) and future multi-host runs consume this shape.

- [ ] **Step 1: Write `scripts/benchkit.py`.** Python 3.9+ stdlib only. Behavior:
  - Capture specs: `platform.platform()`, `platform.processor()` (plus `/proc/cpuinfo` model name on Linux, `sysctl -n machdep.cpu.brand_string` on macOS, `wmic cpu get name` fallback on Windows), core count, python version, `rustc --version`, `cc/c++/cl` version if found, load average where available, WSL detection (`microsoft` in platform release), timestamp, `git rev-parse --short HEAD`.
  - Refuse to run if 1-min loadavg > 0.5 on Unix (idle-host protocol) unless `--force`.
  - Run `cargo run -q -p xval --release --example report_data` with `RUSTFLAGS="-C target-cpu=native"`, capture stdout JSON.
  - If `--python VENVPY` given, run bench_py with that interpreter, capture JSON; else `python_report: null`.
  - Write bundle to `--out` (default `bench-<hosttag>-<YYYYMMDD>.json`, hosttag from `platform.node()`).
  - Sequential, never concurrent runs.

```python
#!/usr/bin/env python3
"""Portable bench runner: captures host specs + runs the Rust (and optionally
Python) benchmark suites, emitting one self-describing JSON bundle per host."""
import argparse, datetime, json, os, platform, shutil, subprocess, sys

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def cpu_model():
    s = platform.system()
    if s == "Linux":
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if s == "Darwin":
        r = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        if r.returncode == 0:
            return r.stdout.strip()
    if s == "Windows":
        r = sh(["wmic", "cpu", "get", "name"])
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
    return platform.processor() or "unknown"

def tool_version(names):
    for n in names:
        if shutil.which(n):
            r = sh([n, "--version"])
            if r.returncode == 0:
                return r.stdout.splitlines()[0]
    return None

def host_specs():
    la = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": sys.version.split()[0],
        "rustc": tool_version(["rustc"]),
        "cxx": tool_version(["c++", "g++", "clang++", "cl"]),
        "loadavg": la,
        "git_sha": sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="venv python for bench_py (optional)")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true", help="skip idle-host check")
    a = ap.parse_args()
    specs = host_specs()
    if not a.force and specs["loadavg"] and specs["loadavg"][0] > 0.5:
        sys.exit(f"host not idle (loadavg {specs['loadavg'][0]}); close apps or --force")
    env = dict(os.environ, RUSTFLAGS="-C target-cpu=native")
    r = sh(["cargo", "run", "-q", "-p", "xval", "--release", "--example", "report_data"], env=env)
    if r.returncode != 0:
        sys.exit(f"report_data failed:\n{r.stderr[-2000:]}")
    bundle = {"host": specs, "rust_report": json.loads(r.stdout), "python_report": None}
    if a.python:
        rp = sh([a.python, "crates/flannrust-py/python/bench/bench_py.py"])
        if rp.returncode != 0:
            sys.exit(f"bench_py failed:\n{rp.stderr[-2000:]}")
        bundle["python_report"] = json.loads(rp.stdout)
    out = a.out or f"bench-{platform.node()}-{datetime.date.today():%Y%m%d}.json"
    with open(out, "w") as f:
        json.dump(bundle, f, indent=1)
    print(out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on this host.** `python3 scripts/benchkit.py --out /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/benchkit-test.json` (Rust side only — full run takes ~10 min; that IS the test, run it in the background and verify the JSON parses, has all host keys non-null where expected, and rust_report matches report_data schema). Also test the idle-check refusal path by temporarily faking: `python3 -c` unit-style check is overkill — instead verify `--force` flag parses and the loadavg branch reads correctly.
- [ ] **Step 3: Write `docs/benchkit.md`** — prerequisites per OS (Linux/WSL2: build-essential; macOS: Xcode CLT; Windows: MSVC Build Tools + note native Windows is untested by us so far), invocation examples with and without `--python`, idle-host protocol summary (close heavy apps, loadavg gate), and the bundle schema. State plainly which hosts have published numbers today (WSL2 only).
- [ ] **Step 4: Commit** `feat: portable bench kit with host-spec capture (scripts/benchkit.py)`.

### Task 5: Claims audit + provenance note

**Files:**
- Modify: `README.md`, `docs/benchmarks.md` (only where stale/untraceable)
- Create: `docs/reports/m-pub/claims-audit.md`

**Interfaces:**
- Consumes: fresh post-merge report (git sha 9cac562 run exists; regenerate if needed).
- Produces: audit report Task 7 (blog) treats as the whitelist of publishable numbers.

- [ ] **Step 1: Enumerate.** Grep README.md + docs/benchmarks.md for every performance number (ratios, ms, ×, %). Build a table in `docs/reports/m-pub/claims-audit.md`: claim | file:line | source run | verdict (TRACED / STALE / UNTRACEABLE).
- [ ] **Step 2: Fix.** STALE numbers: update from the latest statistical baseline (docs/EXPERIMENTS.md M2.6 sections are the source of truth) or delete. UNTRACEABLE: delete or re-measure. Every kept number must carry (or sit near) its host + conditions disclosure.
- [ ] **Step 3: Provenance note.** Add a short "How this was built" section to README.md: this codebase was implemented by an AI agent (Claude Code) directed and reviewed by Itzik Ben-Shabat, who does not write Rust; correctness rests on the bit-exact cross-validation suite against vendored nanoflann 1.12.1, not on the author's Rust expertise. Honest, plain, no marketing gloss.
- [ ] **Step 4: Verify** no dead links / dead doc pointers introduced (`grep -rn "superpowers/sdd" README.md docs/benchmarks.md` must stay empty). `cargo test --workspace` untouched-green. Commit `docs: claims audit — every published number traced; provenance note`.

### Task 6: GitHub repo + green CI (controller task — main session, not a subagent)

**Files:** none new (fixes only if CI fails)

**Interfaces:**
- Consumes: Tasks 1-5 merged into main locally.
- Produces: `https://github.com/sitzikbs/flannrust` (private), CI green on main. Blog links to it.

- [ ] **Step 1:** `gh repo create sitzikbs/flannrust --private --source . --push` (pushes main). Confirm with user before this step ONLY if anything about the repo name/visibility changed.
- [ ] **Step 2:** `gh run watch` the triggered workflows; on failure, diagnose, fix, push, repeat (fix rounds ≤ 5, then stop and report).
- [ ] **Step 3:** Trigger `wheels.yml` via `gh workflow run wheels.yml`; verify all three OS wheels build and upload. Download the linux artifact and re-run the Task 3 fresh-venv smoke test against it.
- [ ] **Step 4:** Record run URLs in `docs/reports/m-pub/ci-green.md`, commit, push.

### Task 7: Blog post for itzikbs.com

**Files (in `~/dev/my_website`, branch `flannrust-post`):**
- Create: `blog/posts-md/2026-08-31-flannrust-ai-agent-port.md` (date = actual completion date)
- Create: `assets/images/blog/flannrust-*.svg` (2-3 graphs)

**Interfaces:**
- Consumes: claims-audit whitelist (Task 5), repo URL (Task 6), latest bench bundle/report JSONs, `blader/humanizer` SKILL.md (fetch via `gh api repos/blader/humanizer/contents/SKILL.md --jq .content | base64 -d` into the scratchpad and follow it), dataviz skill for graph design.
- Produces: draft post on a branch — NOT merged, NOT published; user reviews.

- [ ] **Step 1: Fetch humanizer SKILL.md** (command above) and read it fully before writing any prose. Read the dataviz skill (Skill tool) before making any graph.
- [ ] **Step 2: Graphs.** From the latest report JSON build 2-3 SVGs (hand-written SVG or matplotlib-to-SVG, then cleaned): (a) headline ratio chart — 8 core workloads, Rust/C++ median ratio with the noise envelope band 0.94-1.16 shaded and disclosed; (b) dyn_add before/after fidelity fix (1.13-1.14 → 1.03-1.04); optionally (c) Python bindings vs scipy/pynanoflann. Dark/light safe colors per dataviz. Every graph footer: hardware + WSL2 + n reps.
- [ ] **Step 3: Write the post.** Frontmatter matches existing posts exactly (layout, title, description, date, author, permalink, categories, tags, image, excerpt, readingTime — copy structure from `blog/posts-md/2026-06-01-cvpr-2026-survival-guide.md`). Content arc: (1) the honest premise up front — "I don't write Rust; I directed an AI agent that does" and what directing meant (specs, reviews, statistical rigor demands, catching contamination); (2) what was built — bit-exact nanoflann port, how parity is proven (vendored C++ oracle in the same binaries, xval suite, ULP/tie-group comparators); (3) the numbers with graphs — wins, ties, honest losses-turned-ties, noise envelope methodology, the game-load contamination story as a lesson; (4) implementation notes worth reading (arena nodes 20B vs 48B, deterministic parallel build, fidelity-only optimization rule); (5) what an AI-built library means for trust — tests as the authority; (6) links: repo, PyPI-when-published note. Apply humanizer guidance throughout; no AI-tell phrasing.
- [ ] **Step 4: Build the site** locally (`npx eleventy` or the repo's build script per its package.json) and verify the post renders, SVGs display, no build errors.
- [ ] **Step 5: Commit on `flannrust-post` branch** in my_website with standard trailers. Do NOT merge or deploy. Report the preview path.

---

Dependencies: T1 → {T2 (independent, can start with T1), T3 (needs T1), T4 (independent), T5 (independent)} → T6 (needs T1-T5) → T7 (needs T5+T6).
Parallel dispatch: wave 1 = T1+T2+T4+T5 concurrently (disjoint files); wave 2 = T3; then T6 (controller), then T7.
