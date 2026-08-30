# benchkit.py — portable bench runner

`scripts/benchkit.py` is a stdlib-only (Python 3.9+) wrapper that runs this
repo's benchmark suites on whatever host it's invoked on and packages the
result — plus a snapshot of the host's specs — into one self-describing JSON
bundle. It exists so a second (or third) host's numbers can be collected
without hand-transcribing CPU model, compiler versions, and loadavg into a
doc each time; see `docs/EXPERIMENTS.md` Section 1 for why that provenance
matters here.

It does not change any benchmark's behavior — it shells out to the same two
entry points a human would run by hand:

- `cargo run -p xval --release --example report_data` (Rust vs. C++ speed +
  accuracy; always run)
- `crates/flannrust-py/python/bench/bench_py.py` (Python bindings vs.
  scipy/pynanoflann; only run if `--python` is given)

## Prerequisites, per OS

Only the Rust side (`report_data`) is required; the Python side is optional
and needs a venv with the built `flannrust` wheel plus `scipy`/`pynanoflann`
already set up (see `crates/flannrust-py/python/bench/bench_py.py`'s own
docstring for that venv setup).

- **Linux / WSL2**: `build-essential` (or equivalent: a C/C++ toolchain —
  `cc`/`g++`/`clang++` — for the vendored C++ oracle) and a working `rustc`
  on `PATH` (or under `~/.cargo/bin`).
- **macOS**: Xcode Command Line Tools (`xcode-select --install`) for
  `clang++`, plus `rustc`/`cargo`.
- **Windows**: MSVC Build Tools (for `cl.exe`) plus `rustc`/`cargo`. Native
  Windows is untested by this project so far — `benchkit.py`'s Windows
  branches (`cpu_model()`'s `wmic` fallback, `cl` as a `cxx` candidate) are
  written to the same contract as the Linux/macOS paths but have not been
  run on a real Windows host.

## Invocation

From the repo root, with an idle host (see below):

```sh
export PATH="$HOME/.cargo/bin:$PATH"

# Rust-only bundle (report_data), default output name:
python3 scripts/benchkit.py

# Rust-only bundle, explicit output path:
python3 scripts/benchkit.py --out bench-myhost-20260830.json

# Rust + Python bundle, using a prepared venv interpreter:
python3 scripts/benchkit.py --python crates/flannrust-py/.venv/bin/python
```

The default output filename is `bench-<hosttag>-<YYYYMMDD>.json`, where
`<hosttag>` is `platform.node()` (the machine's hostname) and the date is
today's UTC-agnostic local date. Runs are strictly sequential — Rust first,
then Python if requested — never concurrent, to keep both sides' timings
free of cross-contamination from the other suite running at the same time.

## Idle-host protocol

Benchmark numbers in this repo are only meaningful captured on an otherwise
idle machine (see `docs/EXPERIMENTS.md` Section 1's WSL2 noise caveat and
its "confirmed idle" methodology notes throughout). Before running
`benchkit.py`:

- Close other heavy applications (browsers with many tabs, other builds,
  games, etc.) — anything that competes for CPU with the benchmark process.
- `benchkit.py` checks the 1-minute load average (`os.getloadavg()`, Unix
  only) and refuses to run if it's above `0.5`, printing the observed value
  and exiting non-zero.
- Pass `--force` to skip that check (e.g. when you've confirmed by other
  means that the host is idle enough, or the load-average check itself is
  unavailable/misleading on your system). The bundle still records whatever
  loadavg was observed in `host.loadavg`, so a forced run's provenance is
  visible in the output.
- On non-Unix hosts without `os.getloadavg()` (e.g. Windows), `host.loadavg`
  is `null` and the gate is skipped automatically.

## Bundle schema

```jsonc
{
  "host": {
    "platform": "...",       // platform.platform()
    "cpu_model": "...",      // /proc/cpuinfo model name (Linux), sysctl (macOS), wmic (Windows fallback)
    "cpu_count": 8,          // os.cpu_count()
    "wsl": false,            // "microsoft" in platform.release().lower()
    "python": "3.x.y",
    "rustc": "rustc 1.x.y (...)",   // null if rustc not found
    "cxx": "c++ (...) ...",         // first of c++/g++/clang++/cl found; null if none
    "loadavg": [0.1, 0.2, 0.3],     // os.getloadavg() 1/5/15-min; null off-Unix
    "git_sha": "abcdef0",
    "timestamp": "2026-08-30T12:34:56.789012+00:00"  // UTC, ISO 8601
  },
  "rust_report": { /* report_data's own JSON, verbatim — see crates/xval/examples/report_data.rs */ },
  "python_report": { /* bench_py.py's own JSON, verbatim, or null if --python wasn't given */ }
}
```

`rust_report` and `python_report` are each embedded exactly as their
respective tool emits them (`report_data.rs` and `bench_py.py` own their own
schemas and versioning — `bench_py.py`'s JSON carries its own
`schema_version`); `benchkit.py` does not reshape or reinterpret their
contents, only wraps them alongside `host`.

## Published numbers today: WSL2 only

Every number currently published in this repo (`README.md`,
`docs/benchmarks.md`, `docs/EXPERIMENTS.md`) was captured on a single WSL2
host. `benchkit.py` is written to be portable to native Linux, macOS, and
(untested) Windows, but no bundle from any of those has been captured or
published yet — treat any such numbers as unverified until a bundle from
that host exists in this repo.
