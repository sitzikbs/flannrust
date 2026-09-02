# Packaging dry-run evidence (M-pub, Task 3)

Environment: `cargo 1.98.0 (797e8a9bc 2026-08-05)`, `rustc 1.98.0 (88d9e12ae 2026-08-18)`,
Python 3.12.11 (main-repo venv, maturin 1.14.1), Python 3.12.3 (fresh smoke-test venv).

## Metadata fix required

`crates/flannrust/Cargo.toml` had `publish = false`. Per controller ruling, this line
was removed so the crate can be published to crates.io; no other metadata changes were
needed for the dry-run to pass.

## Step 1: `cargo publish --dry-run`

```
$ cargo publish --dry-run -p flannrust --allow-dirty
    Updating crates.io index
   Packaging flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust)
    Updating crates.io index
    Packaged 20 files, 456.7KiB (127.8KiB compressed)
   Verifying flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust)
   Compiling crossbeam-utils v0.8.22
   Compiling crossbeam-epoch v0.9.20
   Compiling crossbeam-deque v0.8.7
   Compiling rayon-core v1.13.0
   Compiling either v1.18.0
   Compiling rayon v1.12.0
   Compiling flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/target/package/flannrust-0.1.0)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.66s
   Uploading flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust)
warning: aborting upload due to dry run
```

Dry-run succeeded on the first attempt after removing `publish = false`; no further
metadata changes were required. No `cargo publish` (without `--dry-run`) was ever run.

## Step 2: Name availability

crates.io (with a User-Agent header, which the crates.io API requires — a bare `curl`
without one returns HTTP 403 with an empty body):

```
$ curl -s -A "flannrust-packaging-check/0.1 (sitzikbs@gmail.com)" -o /tmp/crates_resp.json -w "HTTP_STATUS:%{http_code}\n" https://crates.io/api/v1/crates/flannrust
HTTP_STATUS:404
$ cat /tmp/crates_resp.json
{"errors":[{"detail":"crate `flannrust` does not exist"}]}
```

PyPI:

```
$ curl -s -o /tmp/pypi_resp.json -w "HTTP_STATUS:%{http_code}\n" https://pypi.org/pypi/flannrust/json
HTTP_STATUS:404
$ cat /tmp/pypi_resp.json
{"message": "Not Found"}
```

Both confirm the `flannrust` name is unclaimed on crates.io and PyPI as of 2026-08-30.

## Step 3: Local wheel build + fresh-venv smoke test

Build (maturin from the main repo's `crates/flannrust-py/.venv`, targeting this
worktree's `Cargo.toml` so the built extension is this worktree's code):

```
$ /home/sitzikbs/dev/flannrust/crates/flannrust-py/.venv/bin/maturin build --release \
    -m /home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust-py/Cargo.toml
🐍 Found CPython 3.12 at /usr/bin/python3
🔗 Found pyo3 bindings with abi3-py3.9 support
📡 Using build options features from pyproject.toml
   ... (dependency compilation) ...
   Compiling flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust)
   Compiling flannrust-py v0.1.0 (/home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/crates/flannrust-py)
    Finished `release` profile [optimized] target(s) in 12.82s
📦 Built wheel for abi3 Python ≥ 3.9 to /home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/target/wheels/flannrust-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl
```

Fresh venv + install:

```
$ python3 -m venv /tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/wheeltest-venv
$ .../wheeltest-venv/bin/pip install --quiet /home/sitzikbs/dev/flannrust/.claude/worktrees/agent-a308ef801e862fa66/target/wheels/flannrust-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl numpy
$ .../wheeltest-venv/bin/pip list | grep -i -E "flannrust|numpy"
flannrust 0.1.0
numpy     2.5.2
```

Smoke test:

```
$ .../wheeltest-venv/bin/python -c "import flannrust, numpy as np; t=flannrust.KDTree(np.random.rand(100,3).astype(np.float32)); d,i=t.query(np.random.rand(5,3).astype(np.float32), k=3); print(d.shape, i.shape)"
(5, 3) (5, 3)
```

Wheel installs cleanly into a fresh venv (no dependency on the build venv) and the
built extension imports and runs a real query correctly.

## Summary

| Check | Result |
|---|---|
| `cargo publish --dry-run -p flannrust` | Pass (after removing `publish = false`) |
| crates.io name `flannrust` | Free (404) |
| PyPI name `flannrust` | Free (404) |
| `maturin build --release` (flannrust-py) | Pass |
| Fresh-venv wheel install + smoke test | Pass, `(5, 3) (5, 3)` |

No real `cargo publish` or PyPI upload was performed at any point.
