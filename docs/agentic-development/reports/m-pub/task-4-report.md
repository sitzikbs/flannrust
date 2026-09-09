# Task 4 report: Portable bench kit

## Status
DONE

## Worktree / commit
- Branch: `worktree-agent-af8fdd15072a11761`
- Commit: `c9d623b5cfcf229188d8969de34a671a72898ef4` — "feat: portable bench kit with host-spec capture (scripts/benchkit.py)"
- Files added: `scripts/benchkit.py` (75 lines, executable), `docs/benchkit.md` (114 lines). No library files touched.

## What was done

1. **`scripts/benchkit.py`** written verbatim from the brief's code block (Python 3.9+, stdlib only: `argparse, datetime, json, os, platform, shutil, subprocess, sys`). Made executable (`chmod +x`). `python3 -m py_compile` passes.
2. **Test run on this host** (WSL2, AMD Ryzen 7 9800X3D, nproc=8):
   - First verified the idle-check refusal path *without* `--force`: host loadavg was 0.79 (> 0.5 gate) at test time — refused correctly with `host not idle (loadavg 0.79443359375); close apps or --force`, exit code 1. This confirms the gate logic (`specs["loadavg"] and specs["loadavg"][0] > 0.5`) works as written.
   - The controller's brief flagged that other agents run in parallel on this host and loadavg may sit above 0.5 for reasons unrelated to this script — confirmed true (0.57-0.79 observed across checks). Per the brief's explicit instruction, used `--force` **only for the test run itself**; the gate remains un-relaxed in the committed script.
   - Ran the full test in the background (`nohup ... &`, PID 70428) and waited via `tail --pid=70428 -f /dev/null` — took roughly 10 minutes as expected for `report_data`'s full speed+accuracy suite.
   - Output file: `/tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/benchkit-test.json` — written successfully, parses as valid JSON.
3. **`docs/benchkit.md`** written: per-OS prerequisites (Linux/WSL2 build-essential, macOS Xcode CLT, Windows MSVC Build Tools + explicit "untested by us so far" note), invocation examples with and without `--python`, idle-host protocol (close heavy apps, loadavg gate, `--force` escape hatch, behavior on non-Unix), full bundle JSON schema with field-by-field comments, and a closing section stating plainly that only WSL2 numbers are published today.
4. **Commit** made with the exact requested subject line plus a short why-paragraph body and the required trailer lines.

## Verified JSON structure of the test bundle

Top-level: `{"host": {...}, "rust_report": {...}, "python_report": null}` (no `--python` was passed for this test, per the brief — Rust side only).

`host` — all keys present, all non-null where expected on this Linux/WSL2 host:
```json
{
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "cpu_model": "AMD Ryzen 7 9800X3D 8-Core Processor",
  "cpu_count": 8,
  "wsl": true,
  "python": "3.12.3",
  "rustc": "rustc 1.98.0 (88d9e12ae 2026-08-18)",
  "cxx": "c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0",
  "loadavg": [1.53173828125, 0.46630859375, 0.42529296875],
  "git_sha": "1dcc34b",
  "timestamp": "2026-08-30T22:57:10.179255+00:00"
}
```
(`loadavg[0]` was elevated during the actual timed run — expected, other agents active on the shared host concurrently; this is exactly the condition `--force` was used to bypass for this test, and the bundle transparently records it in `host.loadavg` as documented.)

`rust_report` — matches `report_data.rs`'s documented schema exactly:
- `rust_report.meta`: `{cpu_threads: 8, date: "2026-08-30T22:57:29Z", nanoflann_version: "1.12.1", rustc: "rustc 1.98.0 (...)", target_cpu_native: true, ...}`
- `rust_report.speed`: list of 8 workload rows, each `{workload, rust: {mean_ms, std_ms, median_ms, min_ms, max_ms, n}, cpp: {same fields}, rust_ms, cpp_ms, ratio, ratio_means, ratio_means_std, ratio_medians}` — full `TimingStats` objects for both sides plus the pre-M2.6 flat keys, as report_data.rs's doc comment specifies.
- `rust_report.accuracy`: list of 11 workload rows (3 seeded datasets across eps values), each with `workload, n_queries, rust_exact_tie_aware_vs_bruteforce, cpp_exact_tie_aware_vs_bruteforce, rust_eq_cpp_bitexact`.

`python_report`: `null` (as expected — `--python` not passed).

The bundle validated cleanly against everything the brief and `report_data.rs`'s own doc comments promise.

## Concerns

- None on correctness. One minor procedural note: the idle-host gate was bypassed with `--force` for the test run itself, as explicitly pre-authorized by the controller's brief (parallel-agent contention on the shared host), and the gate code itself is untouched/still enforced by default in the committed script.
- The test bundle JSON (`benchkit-test.json`) and the earlier refusal-path invocation live only in the scratchpad, not committed — matches the brief (Step 2 is a verification step, not an artifact to commit).

## Fix round 1 (post-review)

Addressed two findings from the coordinator's review:

1. **IMPORTANT — executable bit not committed.** `git ls-tree HEAD scripts/benchkit.py` showed mode `100644` despite the earlier report claiming `chmod +x` was done. Root cause: this repo has `core.fileMode = false`, so a local `chmod +x` followed by plain `git add` never reaches the index (git ignores the mode bit locally). Fixed with `git update-index --chmod=+x scripts/benchkit.py` (plus a matching `chmod +x` on disk) and re-committed.

   Evidence — `git ls-tree HEAD scripts/benchkit.py` after the fix:
   ```
   100755 blob 2e2594dba8f00f812565cc3b5af558640ac913bf	scripts/benchkit.py
   ```
   (blob hash unchanged — content untouched, mode-only fix.)

2. **MINOR — RUSTFLAGS not documented in bundle schema.** Added a paragraph to `docs/benchkit.md`'s "Bundle schema" section, directly after the JSON schema block, noting that `benchkit.py` always sets `RUSTFLAGS="-C target-cpu=native"` before running `report_data`, and that this means `rust_report`'s speed numbers are native-tuned per capturing host and not bit-for-bit comparable across hosts with different CPUs.

New commit: `eaf1db2` — "fix: mark benchkit.py executable, document target-cpu=native in bundle schema" (2 files changed, 9 insertions, one mode change). No bench re-run needed (mode/doc-only fix).
