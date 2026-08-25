# EXPERIMENTS.md — reproducing nanoflann-rs's numbers

Every performance and accuracy figure in this repo's `README.md` and
`docs/benchmarks.md` traces to a command listed in this file. This
document exists per the project's standing requirement (`docs/ROADMAP.md`):
*"Every number that could appear in an announcement traces to a committed
artifact (JSON + generator + doc), never to a chat log."* Every command
below was actually run at least once while writing this document — see
each section's captured output.

## 1. Environment

The library crate was renamed `nanoflann-rs` → `flannrust` on 2026-08-23
(M-py T0); pasted outputs earlier than that show the old name/paths.

Real values, captured via `report_data`'s own self-describing `meta` block
(`cargo run -p xval --release --example report_data`, meta-capture helpers
in `crates/xval/src/lib.rs`: `cpu_model`/`kernel_version`/
`cxx_compiler_version`/`git_sha`/`is_wsl`, all best-effort — fall back to
`"unknown"`/`false` on any capture failure, never panic) plus a direct
toolchain check, both re-run for this document:

| Field | Value | How captured |
|---|---|---|
| CPU model | AMD Ryzen 7 9800X3D 8-Core Processor | `/proc/cpuinfo` `model name` line (`cpu_model()`) |
| CPU threads visible | 8 | `std::thread::available_parallelism()` |
| Kernel | `Linux 6.6.87.2-microsoft-standard-WSL2` | `uname -sr` (`kernel_version()`) |
| **Running under WSL2** | **true** | `/proc/version` contains `"microsoft"` (`is_wsl()`) — see the noise caveat below |
| rustc | `rustc 1.98.0 (88d9e12ae 2026-08-18)` | `rustc --version` |
| C++ compiler | `c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` | `$CXX --version`, falls back to `c++ --version` (`cxx_compiler_version()`) |
| nanoflann (C++) version | 1.12.1, commit `7812aa0` | vendored tag, `crates/nanoflann-ref/cpp/nanoflann.hpp` |
| git SHA at time of writing | `b90fa57` (branch `m2-dynamic` HEAD immediately before the prior documentation round's three commits, `3e9b8d6`/`3cc1f5b`/`5f671a0`); this document's M2 final-review fix wave starts from `5f671a0` | `git rev-parse --short HEAD` + `git status --porcelain` dirty check (`git_sha()`) |
| git SHA, M2.5 task 4 (regression sweep + docs) | `2d23db4` (branch `m2p5-perf`, HEAD after T2's fix round lands — both T2 and T3 fully landed) — every number in §3's "M2.5 task 4" subsection and in `docs/benchmarks.md`'s M2.5 "Post-T2/final" columns was measured at this exact commit. The other M2.5 §3 subsections each state their own commit: T1's noise sweep (`crates/nanoflann-rs` and `crates/nanoflann-ref` byte-identical to `41abdb6`; T1's diagnostic examples committed as `9b2e85f`), T3's post-T3 A/B (trees landed as `fecbdde` and `549f1ac`), T2's give-back A/B (`549f1ac` vs. the T2 fix-round tree landed as `2d23db4`), the final-review fix wave (`4b9fbc3`), and the asm re-capture (`84c0781`) | `git rev-parse --short HEAD`, working tree clean (`git status --porcelain` empty) |
| git SHA, M2.5 final-review fix wave (miri re-capture + dim-16 CSV + M8 count) | `4b9fbc3` — the tree those runs were captured on. `crates/` differs from `2d23db4` only by doc comments (`git diff --stat 2d23db4 4b9fbc3 -- crates/`: `data_source.rs`, `lib.rs`, `m25_diag.rs`, doc text only), so the timed code is the `2d23db4` code | `git rev-parse --short HEAD` |
| git SHA, asm re-capture (§3 "M2.5 asm re-capture" subsection, 2026-08-23) | `84c0781` — `crates/` vs. `2d23db4`: doc comments plus one `debug_assert!`-gated helper in `metric.rs` (`git diff 2d23db4 84c0781 -- crates/`) that compiles to nothing in `--release` | `git rev-parse --short HEAD`, `git status --porcelain` clean for `crates/` |
| git SHA, M-py task 6 (docs + final sweep, this document's "M-py" subsection) | `0cea30c` (branch `m-py`, HEAD after T5's fix round lands — T0 through T5 fully landed) at measurement time, with T6's own changes staged but not yet committed: one-line doc-comment fixes (`data_source.rs`, this file's two stale-path fixes, `nanoflann-notes.md`), one new `flannrust` unit test (`tree.rs::dataset_returns_the_built_over_data_source`), one new pytest test (`test_query_radius_box.py`'s workers determinism case), and one bench-script line move (`bench_py.py`'s `_pnf_jobs` precompute) — none of which touch any timed code path, so every number below is representative of `0cea30c`'s compiled behavior | `git rev-parse --short HEAD` |

### Vendored C++ header provenance

`crates/nanoflann-ref/cpp/nanoflann.hpp` is byte-identical (after CRLF/LF
line-ending normalization) to upstream nanoflann's tag `1.12.1`, verified
by hashing a fresh clone of that exact tag and comparing against the
vendored file's own hash:

```
git clone --depth 1 --branch 1.12.1 https://github.com/jlblancoc/nanoflann.git /tmp/nanoflann-1.12.1
sed -e 's/\r$//' /tmp/nanoflann-1.12.1/include/nanoflann.hpp | sha256sum
sed -e 's/\r$//' crates/nanoflann-ref/cpp/nanoflann.hpp | sha256sum
```
Both commands print the same digest:
`8918fc2b1492e1f5b790915e1555141125adfe27786434d391cf6f071d42ca7a`. This is
the byte-identity guarantee the whole cross-validation methodology in this
document rests on — the C++ oracle is genuinely upstream nanoflann 1.12.1,
not a modified or hand-edited copy.

### WSL2 measurement-noise caveat (read every number in this repo with this in mind)

**Every number in this repo's README, `docs/benchmarks.md`, and this file
was measured inside WSL2** (Windows Subsystem for Linux 2), a virtualized
environment sharing the host's scheduler with the Windows side. WSL2
introduces real run-to-run scheduler/thermal jitter — M1's own measurement
found roughly **±5-10% on individual runs**.

**Update (M2.6): the noise floor this repo cites everywhere is now the
n=100-per-side, 4-session `knn_fixed3` characterization** in §3's "M2.6
task 2" subsection ("The new noise floor" sub-subsection): session medians
**1.030–1.067** (n=4 independent sessions), per-session per-repetition
ratio σ≈0.03–0.05 (delta-method estimate from `xval::measure_pair`'s
published `mean_ms`/`std_ms` at n=100/side), approximate mean±2σ band
**0.94–1.16** — every "within the noise floor" statement in `README.md`,
`docs/benchmarks.md`, and `docs/ROADMAP.md` now refers to that
characterization. **Historical, superseded, kept for provenance, not
deleted:** the previous canonical floor was the 8-run `knn_fixed3` sweep
pasted in §3's "M2.5 task 1" subsection, ratio **0.956–1.192** (median
1.064) across eight independent runs of one unchanged tree, each run
itself the gate's internal median-of-7; M1's ±5-10% and its 7-run
0.937–1.079 spread (`docs/benchmarks.md` "Remaining gap analysis") are
older, smaller samples of the same phenomenon, also kept as historical
text.

**`dyn_add_20k_dim3_f32`** (workload unchanged by the M2 final-review fix
wave): eight re-runs total across two documentation rounds — four from an
earlier round (§3 has the pasted outputs) plus four fresh runs from this
fix wave (§3, "M2 final-review fix wave" subsection below) — show this
concretely on this exact machine: ratio 1.113 (recorded at commit
`ec9b581`) vs. **1.106**, **1.230**, **1.113**, **1.117** (earlier round)
and **1.108**, **1.237**, **1.134**, **1.137** (this fix wave). Observed
range across all eight: **1.106–1.237** (~11.8% spread) — the 1.237 figure
sitting closer to the 1.25 gate margin than any other recorded or re-run
figure in this repo, purely from run-to-run noise, not a code change
(`dyn_add`'s workload is untouched by this fix wave, and nothing else in
`crates/` changed between any of these eight runs).

**`dyn_knn_after_churn_dim3_f32`**: this workload WAS changed by the M2
final-review fix wave (see the "Corrected `dyn_knn_after_churn` workload"
subsection below) — the OLD (no-op-churn) and NEW (live-tombstone,
real-merge) numbers are not comparable and are kept separate. Old workload,
four re-runs from the earlier round: ratio 0.964 (recorded) vs. **0.945**,
**0.946**, **0.964**, **0.972** — range **0.945–0.972** (~2.9% spread),
kept here only as historical context for the noise-band discussion, not as
a currently-accurate figure (that workload was a provable no-op and is no
longer what `perf_gate.rs`/`bench_dynamic.rs` measure). New (corrected)
workload, four fresh runs from this fix wave: **0.975**, **0.972**,
**0.971**, **0.976** — range **0.971–0.976** (~0.5% spread), noticeably
TIGHTER than the old workload's spread, though that is almost certainly
because these four runs happened close together in one sitting rather than
across two documentation rounds, not evidence the corrected workload is
inherently less noisy.

The perf gates' 1.25 margin and the median-of-7 methodology (below) both
exist specifically to absorb this noise, not to paper over a real
regression — but it is still noise, not a bare-metal-quality measurement,
and `dyn_add`'s swing up to 1.237 is a concrete illustration of why the
margin isn't set tighter. Noise isn't always benign, though: during M2.5
task 4's report-chain regeneration, a `report_data` run executed while a
background `miri` process was still consuming a full CPU core came back
with an implausible `knn_dyn_dim8_f64_k10` ratio of 1.214 (vs. every other
dim8 measurement in that task's own sweep landing at 0.926–0.966) — caught
by cross-checking against the clean gate numbers, discarded, and re-run
only after confirming (`ps aux`) no CPU-heavy background process remained;
see the "M2.5 task 4" subsection below for the clean numbers, and
`task-4-report.md` §2 for the full account — a concrete illustration of
why every timed run in this repo is captured on an otherwise-idle host, not
just why the margin exists.

**Before any public announcement, `docs/ROADMAP.md`'s M-pub milestone
requires a bare-metal Linux re-run** of the full evaluation, with only
those bare-metal numbers published (WSL2 numbers kept only as a secondary
data point). Nothing in this repo has been re-run on bare metal yet — every
number currently published anywhere in this repo, including in this
document, is a WSL2 number.

## 2. Compiler configuration (both sides, exact)

### Rust

- **Release profile** (`Cargo.toml`, workspace root):
  ```toml
  [profile.release]
  codegen-units = 1
  lto = "thin"

  [profile.bench]
  inherits = "release"
  ```
  `codegen-units = 1`: the C++ oracle is always compiled as a single
  translation unit at `-O3` (see below) — Rust's default 16 codegen units
  would be an unforced handicap against that single-TU baseline, so this is
  removed as a variable. `lto = "thin"`: buys back cross-function inlining
  a single codegen unit already gets for free within itself, at a fraction
  of "fat" LTO's build cost. `[profile.bench]` inherits `[profile.release]`
  so `cargo bench` gets identical codegen to `cargo build --release`/
  `cargo run --release` — no separate, uncontrolled bench-profile codegen.
- **`RUSTFLAGS="-C target-cpu=native"`** — required for every timed
  comparison (perf gates, benches, `report_data`). Without it, the Rust
  side is handicapped to a generic-x86-64-baseline instruction set while
  the C++ side (built `-march=native`, below) already gets the host's full
  ISA — an unfair, non-native-vs-native comparison. `cargo test`
  (correctness only, not timed) does not need this flag.

### C++ (the oracle, `crates/nanoflann-ref/build.rs`, via the `cc` crate)

```rust
cc::Build::new()
    .cpp(true)
    .std("c++17")
    .file("cpp/wrapper.cpp")
    .include("cpp")
    .opt_level(3)                          // load-bearing, unconditional
    .flag_if_supported("-march=native")
    .flag_if_supported("-ffp-contract=off") // load-bearing
    .compile("nanoflann_ref");
```

- **`-O3` unconditionally** — the oracle must always be the fast baseline,
  even when the Rust side is built in `cargo test`'s default debug profile;
  this is not gated on Rust's own profile.
- **`-march=native`** — matches Rust's `target-cpu=native`, so neither side
  gets an ISA advantage over the other.
- **`-ffp-contract=off`** (the contraction rationale): at `-O3`, both gcc
  and clang will by default fuse `a*b+c` into a single fused-multiply-add
  (FMA) instruction, which rounds once instead of twice and can produce a
  different bit pattern than the un-fused expression. rustc does **not**
  perform this contraction by default. Without this flag, the C++ oracle's
  distance computations could silently diverge from the Rust
  implementation's bit-for-bit output on identical inputs — breaking the
  entire premise of bit-exact cross-validation. Verified for real, not just
  asserted: `crates/xval/tests/native_parity.rs`'s
  `native_parity_build_knn_dim8_f64` test exists specifically to catch a
  regression here (re-run during this task, see §3 below).

## 3. Reproduction commands

`export PATH="$HOME/.cargo/bin:$PATH"` first if `cargo`/`rustc` aren't
already on `PATH` (this repo's own dev environment needs it). Every command
below was actually run at least once, across the documentation round that
originally wrote this section, the M2 final-review fix wave (which
corrected the `dyn_knn_after_churn` workload, see below), or one of the
M2.5 subsections (tasks 1–4, the M2.5 final-review fix wave, and the asm
re-capture — each states the commit it was measured at); output is
summarized inline, full transcripts are pasted directly in this document.

### Full test suite

```
cargo test --workspace --all-features
```
Expected runtime: a few seconds (debug build, no heavy/perf-gated tests —
those are `#[ignore]`d). **Re-run at the M2 final-review fix wave: 345
passed, 0 failed, 15 ignored**, including doctests (2, both in
`nanoflann-rs`: the crate-doc example and `DynamicKdTree`'s doc example).
This supersedes both M1's own 269-passed/8-ignored count recorded at M1
completion (see `docs/benchmarks.md`'s M1 "Test status" subsection, now
marked superseded) and M2's own 343-passed count at its initial completion
— M2 added `nanoflann-ref`'s dynamic-oracle tests, `xval`'s dynamic
cross-validation suite, the report renderer's own test file, and the two
new dynamic perf gates; the M2 final-review fix wave added two more
`nanoflann-rs` unit tests (`maximum_point_count`'s new capacity bounds
check and its `n == 0` panic), all counted in the new total. `cargo test
--workspace` (default features, no `--all-features`) is an identical
invocation to the command above: `parallel` (on by default) is
`nanoflann-rs`'s only feature.

### Heavy / `--ignored` suite (release; includes canary mutation tests)

```
cargo test --workspace --all-features --release -- --ignored --test-threads=1
```
Expected runtime: **~2-3 minutes** (dominated by the two deepest
degenerate-tree builds — `heavy_exponential_build_1m` depth ~2115,
`heavy_exponential_build_1m_dim8` depth ~16794 — plus a parallel-build heavy
test and a heavy degenerate-tree query test). **Re-run for this document:
0 failed** across all 15 previously-ignored tests, in **162.12s** for the
`nanoflann-rs` unit-test binary alone (the four heavy build/query tests;
the rest of the workspace's ignored tests — `native_parity`'s dim-8 canary,
all six `perf_gate_*` tests self-skipping without `PERF_GATE=1` set, and
five mutation canaries across `xval_dynamic`/`xval_knn`/`xval_radius_box` —
finished in well under a second each). `--test-threads=1` matters here for
the perf-gate tests specifically (see below); it's harmless for the heavy
correctness tests.

### Perf gates (M1's four static + M2's two dynamic — six total)

```
PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release \
  --test perf_gate -- --ignored perf_gate --test-threads=1 --nocapture
```
Expected runtime: a few seconds (each gate is a handful of timed runs on a
fixed-size workload, not a full benchmark sweep). `--test-threads=1` is
load-bearing here, not cosmetic: gate timings must not share the CPU with
a sibling test's load.

### Miri (undefined-behavior check for `search.rs`'s `unsafe`)

`crates/flannrust/src/search.rs`'s `FrameStack` (M2.5 task 2, explicit-
stack iterative `search_level`) is this crate's first and only `unsafe`
code — two small blocks (`assume_init_read`/`assume_init_mut`) implementing
a fixed-capacity inline frame buffer with a heap-`Vec` spill. Requires
`rustup component add miri --toolchain nightly` once; both commands below
must be run — Stacked Borrows (miri's default aliasing model) and Tree
Borrows (`-Zmiri-tree-borrows`, a stricter/different model) can each catch
UB the other misses:

```
cargo +nightly miri test -p flannrust --lib -- search
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p flannrust --lib -- search
```

Expected runtime: **~8 minutes each** (miri interprets every instruction,
~50-100x slower than native; the `-- search` filter selects the 35 lib
tests whose path contains `search` — 28 in `search.rs`'s own test module
plus 7 elsewhere whose names contain `search` (5 in `tree::tests`, 2 in
`dynamic::tests`; `cargo test -p flannrust --lib -- search --list`
enumerates them) — 34 run + 1 `#[ignore]`d heavy test skipped, since miri
interpreting a depth-~16 794 degenerate-tree query would run for hours). Includes
`spill_boundary_deep_tree_knn_and_radius_match_brute_force`, which is the
one test in this filtered set that actually exercises `FrameStack`'s
`overflow` spill path (see that test's doc comment) — i.e. miri validates
the `unsafe` blocks under a real spill, not just the common inline-only
case. **Both re-run for M2.5 task 2's fix round: 34 passed, 0 failed, 1
ignored, clean under both aliasing models** — no UB detected.

**Re-run 1, all six gates** (during this document's original drafting):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.128ms cpp=9.254ms ratio=0.986
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.808ms cpp=3.444ms ratio=1.106
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=30.418ms cpp=32.185ms ratio=0.945
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.220ms cpp=6.968ms ratio=1.036
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=195.461ms cpp=179.159ms ratio=1.091
PERF_GATE perf_gate_radius_dim3_f32: rust=4.750ms cpp=5.995ms ratio=0.792
test result: ok. 6 passed; 0 failed
```

**Re-run 2, all six gates** (this fix round, minutes later, no code changes
in between):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.996ms cpp=10.039ms ratio=0.996
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.160ms cpp=3.383ms ratio=1.230
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=28.971ms cpp=30.638ms ratio=0.946
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.235ms cpp=6.809ms ratio=1.063
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=174.158ms cpp=179.602ms ratio=0.970
PERF_GATE perf_gate_radius_dim3_f32: rust=4.606ms cpp=5.918ms ratio=0.778
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.83s
```

All twelve gate invocations across both re-runs pass the 1.25 margin.
Comparing against `docs/benchmarks.md`'s recorded figures (build 1.018,
`knn_fixed3` 1.042, `knn_dyn_dim8_f64_k10` 0.973, radius 0.767, `dyn_add`
1.113, `dyn_knn_after_churn` 0.964): most ratios across both re-runs sit
within a few points of the recorded figure, consistent with ordinary WSL2
noise (§1) — except `knn_dyn_dim8_f64_k10` in Re-run 1 (0.973 recorded vs.
**1.091**, ~12%) and `dyn_add_20k_dim3_f32` in Re-run 2 (1.113 recorded
vs. **1.230**, ~10.5%, and the closest any recorded run in this repo has
come to the 1.25 gate margin). Neither is a regression — no code in
`crates/` changed between the recorded run and either re-run, this task is
documentation-only — but both are real, larger-than-typical single-run
swings, and are left in this document exactly as measured rather than
re-run until a quieter number appeared, per the same "state the observed
min–max honestly" standard the rest of this section follows.

**Re-run, dynamic gates only** (`--ignored perf_gate_dyn` filter), two
consecutive invocations during this fix round, minutes apart:
```
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.787ms cpp=3.403ms ratio=1.113
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.112ms cpp=30.208ms ratio=0.964
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.56s
```
```
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.229ms cpp=3.787ms ratio=1.117
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.556ms cpp=30.415ms ratio=0.972
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.58s
```
The first of these two matches the recorded commit-time figures (1.113/
0.964) exactly; the second drifts a few points (1.117/0.972). Across all
four dynamic-gate re-runs pasted in this section — these two dedicated
runs, plus Re-run 1's and Re-run 2's dynamic rows above (all four executed
across this task and its fix round, no code changes between any of them)
— the observed range is **`dyn_add` 1.106–1.230** and
**`dyn_knn_after_churn` 0.945–0.972**, stated as the honest min–max
across every pasted run, not averaged or cherry-picked. (This matches
`docs/benchmarks.md`'s "M2 — dynamic forest" section, which cites this
exact same four-run range.)

To run only the two dynamic gates yourself: append `_dyn` to the filter
(`--ignored perf_gate_dyn`, as used for the two dedicated re-runs above).

### M2 final-review fix wave: corrected `dyn_knn_after_churn` workload

The `dyn_knn_after_churn_dim3_f32` figures above (0.964 recorded, 0.945–0.972
re-run range) were measured against a workload that is a **provable no-op**:
it removed 5,000 points, then reactivated ALL 5,000 of them before the timed
knn loop ran. Since every removal was undone, `removed_len() == 0` at query
time — the timed loop never once ran tombstone-filtered search against a
forest with any live tombstones, and no `add_points` call in the setup ever
triggered a merge deep enough to matter (reactivation never touches slot
membership at all, see `flannrust::dynamic::DynamicKdTree::add_points`'s
doc comment). This was fixed in both `crates/xval/tests/perf_gate.rs` and
`crates/xval/benches/bench_dynamic.rs`: the corrected workload builds 95,000
points, removes 5,000, reactivates only half (2,500 — leaving 2,500 LIVE
tombstones), then adds a fresh CONTIGUOUS growth batch of 5,000 more points
(starting exactly at 95,000, the running `point_count`) to reach 100,000
total. That growth batch is deliberately sized past 4,096 = 2^12, which by
the pigeonhole principle GUARANTEES it passes through at least one
`point_count` value whose binary representation ends in 12 or more one-bits
— i.e. `first0bit` returns >= 12 somewhere in the batch, forcing a real
cascading merge across 12+ slots, which also migrates the still-removed
tombstones' recorded slot (the exact mechanism `add_points`'s doc comment
describes). The setup asserts `removed_len() > 0` on the Rust side, plain
`assert!` (not `debug_assert!`, since perf gates run `--release` without
`debug-assertions = true`), right before the timed loop starts, on BOTH the
Rust and C++ sides driven through the identical op sequence.

**RED capture** — proving the assert actually fires, not just decoration:
temporarily set `READD` back to `REMOVE` (5,000, i.e. reactivate every
removed point again, reproducing the OLD no-op shape) and re-run the single
gate test:
```
$ PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release \
    --test perf_gate perf_gate_dyn_knn_after_churn_dim3_f32 -- --ignored --exact --nocapture

thread 'perf_gate_dyn_knn_after_churn_dim3_f32' panicked at crates/xval/tests/perf_gate.rs:389:5:
perf_gate_dyn_knn_after_churn_dim3_f32 setup: removed_len() must be > 0 at query time (workload must leave live tombstones) -- got 0, which means this workload regressed back to the provable-no-op churn shape this gate was fixed to avoid
test perf_gate_dyn_knn_after_churn_dim3_f32 ... FAILED
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 8 filtered out; finished in 0.02s
```
`READD` was then restored to `2_500` and the suite re-verified green before
any of the runs below were captured.

**Re-run 1, all six gates** (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native"
cargo test -p xval --release --test perf_gate -- --ignored perf_gate
--test-threads=1 --nocapture`):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=8.987ms cpp=9.118ms ratio=0.986
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.829ms cpp=3.455ms ratio=1.108
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=30.034ms cpp=30.800ms ratio=0.975
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.170ms cpp=6.770ms ratio=1.059
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=170.867ms cpp=179.698ms ratio=0.951
PERF_GATE perf_gate_radius_dim3_f32: rust=4.595ms cpp=5.932ms ratio=0.775
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.80s
```

**Re-run 2, all six gates** (same command, minutes later, no code changes
in between):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=10.178ms cpp=10.136ms ratio=1.004
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.222ms cpp=3.413ms ratio=1.237
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.655ms cpp=30.523ms ratio=0.972
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.099ms cpp=6.730ms ratio=1.055
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=170.512ms cpp=179.753ms ratio=0.949
PERF_GATE perf_gate_radius_dim3_f32: rust=4.583ms cpp=5.925ms ratio=0.774
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.80s
```

**Re-run, dynamic gates only** (`--ignored perf_gate_dyn` filter), two more
consecutive invocations, minutes apart:
```
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.292ms cpp=3.786ms ratio=1.134
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.589ms cpp=30.485ms ratio=0.971
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.58s
```
```
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.870ms cpp=3.405ms ratio=1.137
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.820ms cpp=30.543ms ratio=0.976
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.57s
```

All eight gate invocations pass the 1.25 margin (six-gate runs run all six
gates each; dyn-only runs run two each — twelve total gate PASS lines
across the four invocations). Across these four fresh runs:
`dyn_knn_after_churn_dim3_f32` (corrected workload) ranged **0.971–0.976**
— tight, and clearly a different population from the old workload's
0.945–0.972 (expected: different workload, not comparable). `dyn_add`
(workload unchanged by this fix wave) ranged **1.108–1.237** in these four
fresh runs; combined with the four runs recorded in the earlier
documentation round (1.106, 1.230, 1.113, 1.117), the honest range across
all eight `dyn_add` measurements taken to date is **1.106–1.237** — wider
than the 1.106–1.230 this repo previously documented, purely because an
additional four runs were taken, not because anything regressed (`dyn_add`
gate code is byte-identical before and after this fix wave). This is
exactly the kind of range-widening honest re-measurement is expected to
produce, and is reported as measured rather than rounded down to match a
previously-published figure.

### M2.5 task 1: the noise floor this repo cites (pre-M2.5 tree; `crates/nanoflann-rs` and `crates/nanoflann-ref` byte-identical to `41abdb6`)

T1 was diagnosis only — no library change; its two diagnostic examples
(`m25_diag.rs`, `m25_asm.rs`) were committed as `9b2e85f`, with
`crates/nanoflann-rs` and `crates/nanoflann-ref` byte-identical to
`41abdb6` throughout (T1 report §10, `git diff` empty on both crates). It
measured the fixed-dim-3 gate eight independent times on that unmodified
tree, each run itself the gate's internal median-of-7. **This is THE noise
floor** every "within the noise floor" statement in this repo refers to
(`README.md` "Perf gate" footnote, `docs/benchmarks.md` "Honest residuals",
`docs/ROADMAP.md` M2.5 leads and M-pub). Copied verbatim from
`docs/reports/m2.5/task-1-report.md` §5 — T1 recorded the eight ratios,
not the raw `PERF_GATE ... rust=/cpp=` lines:

```
$ PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --ignored perf_gate_knn_dim3 --test-threads=1 --nocapture
(perf_gate_knn_dim3_f32_k10 ratio, 8 independent invocations, each an internal median-of-7)
0.956, 1.035, 1.043, 1.063, 1.064, 1.071, 1.104, 1.192
median 1.064, range 0.956-1.192
```

T1's own reading, quoted: "Any dim-3 claim below ~2 % is inside the noise
floor of this machine." The two older characterizations in this repo —
M1's "~±5-10% on individual runs" (§1, `docs/benchmarks.md` M1 preamble)
and M1's 7-run `knn_fixed3` spread 0.937–1.079 (`docs/benchmarks.md`
"Remaining gap analysis", item 2) — are smaller samples of the same
phenomenon, kept as historical text; the 8-run sweep above is the figure
to quote.

### M2.5 task 3: post-T3 gate state, dim-32/64 A/B, and the kernel bit-exactness guard (trees landed as `fecbdde`, fix round `549f1ac`)

Source for `docs/benchmarks.md`'s M2.5 summary table "Post-T3" column and
its dim-32/64 table's "T3 landed" column. Copied verbatim from
`docs/reports/m2.5/task-3-report.md` §4a, §4b, §8.1, §8.2. T3's
methodology: two prebuilt binaries — "before" = T1's `9b2e85f` (pre-T3;
identical library code to the milestone-start tree), "after" = T3's
working tree, landed as `fecbdde` — invoked interleaved
before/after/before/after/before/after with no recompile between runs,
each run the gate's internal median-of-7.

**Six PERF_GATE gates** (`PERF_GATE=1 <binary> --ignored --test-threads=1
--nocapture`, n=3 each side):

| Gate | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|
| `build_100k_dim3_f32_seq` | 0.989, 0.998, 0.987 | **0.989** | 0.974, 0.971, 0.971 | **0.971** |
| `dyn_add_20k_dim3_f32` | 1.210, 1.120, 1.122 | **1.122** | 1.137, 1.139, 1.141 | **1.139** |
| `dyn_knn_after_churn_dim3_f32` | 0.956, 0.965, 0.965 | **0.965** | 0.969, 0.971, 0.968 | **0.969** |
| `knn_dim3_f32_k10` (fixed3) | 1.030, 1.045, 1.039 | **1.039** | 1.065, 1.049, 1.054 | **1.054** |
| `knn_dyn_dim8_f64_k10` | 0.949, 0.944, 0.945 | **0.945** | 0.864, 0.868, 0.867 | **0.867** |
| `radius_dim3_f32` | 0.784, 0.779, 0.778 | **0.779** | 0.776, 0.784, 0.777 | **0.777** |

**Dim-32/64, gate-style real tree** (`RUSTFLAGS="-C target-cpu=native"
cargo run -p xval --release --example m25_diag -- knn`, leaf=10, n=100k,
200 queries, k=10; n=3 each side, interleaved):

| dim | scalar | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|---|
| 32 | f32 | 1.423, 1.420, 1.428 | **1.423** | 0.966, 0.947, 0.972 | **0.966** |
| 32 | f64 | 1.315, 1.315, 1.317 | **1.315** | 1.119, 1.105, 1.113 | **1.113** |
| 64 | f32 | 1.917, 1.923, 1.918 | **1.918** | 1.228, 1.234, 1.228 | **1.228** |
| 64 | f64 | 1.220, 1.213, 1.208 | **1.213** | 0.877, 0.874, 0.874 | **0.874** |

**Dynamic gates, re-measured in T3's fix round** after `&GrowableFlat`
gained a `point_row` override (until then both dynamic gates ran
byte-identical code on both arms, so their movement in the six-gate table
above was provably noise — T3 §8.2). "before" = `fecbdde`, "after" = the
fix-round tree, landed as `549f1ac`; n=3 interleaved:

| Gate | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|
| `dyn_add_20k_dim3_f32` | 1.146, 1.142, 1.150 | **1.146** | 1.123, 1.112, 1.131 | **1.123** |
| `dyn_knn_after_churn_dim3_f32` | 0.980, 0.974, 0.989 | **0.980** | 0.969, 0.980, 0.974 | **0.974** |

So `docs/benchmarks.md`'s "Post-T3" column is: `build` 0.971,
`knn_fixed3` 1.054, `dim8` 0.867, `radius` 0.777 (six-gate "after"
medians, tree landed as `fecbdde`) and `dyn_add` 1.123,
`dyn_knn_after_churn` 0.974 (fix-round "after" medians, tree landed as
`549f1ac`). The six-gate "before" arm is itself a pre-M2.5 measurement and
is folded into the pre-M2.5 pasted range (see the "M2.5 task 4"
subsection below).

**Kernel bit-exactness guard — multi-salt RED captures (T3 §8.1, landed in
`549f1ac`)**: the chunked row walk is proven bit-identical to the untouched
per-component loop by `metric.rs`'s four bit-equality tests, each sweeping
dims `{1..=8, 15, 16, 17, 31, 32, 33, 64}` × 6 salts (`SALTS_F32`/
`SALTS_F64`), for `L1`/`L2` × f32/f64. That guard was shown to actually
discriminate by two deliberate sabotages, each reverted before commit;
`cargo test -p nanoflann-rs metric::` under each (one failing test shown
per sabotage; all four core tests fail independently under each — full
transcripts in T3 §8.1):

```
# Sabotage 1 — ascending remainder order (d, d+1, d+2 instead of the required d+2, d+1, d+0)
---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:694:17:
assertion `left == right` failed: dim=3 salt=-4.2214 row=54666.434 fallback=54666.438
  left: 1196788335
 right: 1196788336
test result: FAILED. 21 passed; 6 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s

# Sabotage 2 — fully left-associative chunk grouping (((d0²+d1²)+d2²)+d3² instead of (d0²+d1²)+(d2²+d3²))
---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:694:17:
assertion `left == right` failed: dim=4 salt=3.5588 row=150733.8 fallback=150733.78
  left: 1209217907
 right: 1209217906
test result: FAILED. 23 passed; 4 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```

Both sabotages reverted; `metric::` back to 27/27 green before commit.

### M2.5 task 2: the give-back A/B, capacity-32 check, `unsafe` necessity A/B, and the fixed-dim-3 win (`549f1ac` vs. the T2 trees landed as `1ecfd46` / `2d23db4`)

Source for `docs/benchmarks.md`'s "The T2 trade-off" paragraph (radius
+4.4–7.4%, dim8 +6.4–6.8%), its capacity-32 statement, the README
"Safety" section's +4.7% figure, and the fixed-dim-3 1.052 → 1.0205
medians. Copied verbatim from `docs/reports/m2.5/task-2-report.md` §6b and
Fix round 1, Items 2 and 5.

**Give-back A/B (Item 2).** Raw `rust_ms` (not the ratio, which conflates
the shared-noise C++ side). Three prebuilt binaries — OLD = `549f1ac`
(recursive `search_level`), NEW = T2's fix-round tree (iterative, inline
capacity 128; landed as `2d23db4`), NEW32 = the same with
`SEARCH_STACK_INLINE_CAPACITY = 32` (never committed) — 8 reps interleaved
OLD→NEW→NEW32 per rep, then 8 more in reversed order NEW32→NEW→OLD. Each
cell is the gate's internal median-of-7.

`radius_dim3_f32` (rust_ms, PERF_GATE, n=100k, 2000 queries):

| rep | OLD (fwd) | NEW (fwd) | NEW32 (fwd) | NEW32 (rev) | NEW (rev) | OLD (rev) |
|---|---|---|---|---|---|---|
| 1 | 5.241 | 5.476 | 4.995 | 5.170 | 5.071 | 4.815 |
| 2 | 4.692 | 4.946 | 4.937 | 5.025 | 4.992 | 4.696 |
| 3 | 4.714 | 4.960 | 4.954 | 4.955 | 4.894 | 4.713 |
| 4 | 4.668 | 4.952 | 5.024 | 4.949 | 5.009 | 4.660 |
| 5 | 4.656 | 4.912 | 5.028 | 4.958 | 4.961 | 4.669 |
| 6 | 4.643 | 5.423 | 5.044 | 4.995 | 4.905 | 4.782 |
| 7 | 4.712 | 5.047 | 5.032 | 4.925 | 4.948 | 5.501 |
| 8 | 4.642 | 5.027 | 5.101 | 5.038 | 4.888 | 4.778 |
| **median** | **4.680** | **4.994** | **5.026** | **4.977** | **4.955** | **4.746** |

Forward: OLD→NEW **+6.7%**, OLD→NEW32 **+7.4%**. Reversed: OLD→NEW
**+4.4%**, OLD→NEW32 **+4.9%**. Direction consistent in 15/16 individual
pairs (the exception, reversed rep 7, is an OLD-side outlier at 5.501 ms).

`knn_dyn_dim8_f64_k10` (rust_ms, PERF_GATE, n=100k, 10000 queries):

| rep | OLD (fwd) | NEW (fwd) | NEW32 (fwd) | NEW32 (rev) | NEW (rev) | OLD (rev) |
|---|---|---|---|---|---|---|
| 1 | 156.754 | 167.848 | 168.031 | 168.513 | 168.331 | 158.480 |
| 2 | 160.195 | 167.394 | 167.612 | 168.471 | 166.230 | 158.325 |
| 3 | 159.282 | 168.400 | 168.965 | 170.116 | 168.092 | 157.577 |
| 4 | 158.681 | 167.322 | 167.604 | 170.638 | 167.346 | 156.880 |
| 5 | 156.598 | 169.259 | 169.283 | 170.479 | 166.783 | 158.021 |
| 6 | 155.918 | 167.263 | 167.356 | 168.459 | 167.889 | 157.990 |
| 7 | 157.411 | 168.707 | 167.434 | 167.244 | 168.235 | 155.655 |
| 8 | 157.957 | 169.687 | 168.912 | 166.764 | 168.387 | 157.918 |
| **median** | **157.684** | **168.124** | **167.822** | **168.492** | **167.991** | **157.954** |

Forward: OLD→NEW **+6.6%**, OLD→NEW32 **+6.4%**. Reversed: OLD→NEW
**+6.4%**, OLD→NEW32 **+6.7%**. Direction 16/16 — OLD is the fastest of
the three in every rep, both orderings.

Reading: both give-backs are real and reproducible, and capacity 32 does
not recover either (NEW32 ≈ NEW in every column), so the inline-capacity
constant is not a lever for this cost. The mechanism stated in
`docs/benchmarks.md` (frame store/reload cost in the fully-inlined
explicit-stack form, plus code-size effects) is the T2 reviewer's
diagnosis, consistent with this data; it has **not** been separately
quantified from asm — no figure in this repo is an asm-measured share of
the fixed-dim-3 residual.

**`unsafe` necessity A/B (Item 5).** `FrameStack`'s `MaybeUninit` inline
array (the crate's two `unsafe` blocks) vs. a zero-`unsafe`
`Vec::with_capacity(SEARCH_STACK_INLINE_CAPACITY)` frame stack with
otherwise identical code, two prebuilt binaries, `perf_gate_knn_dim3_f32_k10`
rust_ms, 8 interleaved reps:

| rep | Unsafe (`MaybeUninit`) | Safe (`Vec::with_capacity`) |
|---|---|---|
| 1 | 6.982 | 7.346 |
| 2 | 7.160 | 7.555 |
| 3 | 6.963 | 7.326 |
| 4 | 7.568 | 7.445 |
| 5 | 7.180 | 7.595 |
| 6 | 7.131 | 7.586 |
| 7 | 6.946 | 7.291 |
| 8 | 6.997 | 7.349 |
| **median** | **7.064** | **7.397** |

Safe is slower in 8/8 reps, median **+4.7%** — the evidence the README's
"Safety" section cites for keeping the `unsafe`.

**Fixed-dim-3 win (T2 §6b).** `perf_gate_knn_dim3_f32_k10` ratio, A =
`549f1ac` (recursive), B = the `MaybeUninit`-backed iterative form as
landed in `1ecfd46`, two prebuilt binaries, 8 interleaved reps:

| rep | A ratio (baseline) | B ratio (fixed iterative) |
|---|---|---|
| 1 | 1.044 | 1.021 |
| 2 | 1.043 | 1.020 |
| 3 | 1.049 | 1.042 |
| 4 | 1.066 | 1.008 |
| 5 | 1.052 | 1.026 |
| 6 | 1.084 | 1.323 (outlier — rust=9.087ms, a one-off scheduling blip) |
| 7 | 1.052 | 0.932 (outlier — cpp=7.506ms, the C++ side blipped slow) |
| 8 | 1.059 | 0.999 |
| **median** | **1.052** | **1.0205** |

Medians 1.052 → 1.0205; B beats A in 6/8 pairs, the two misses each a
one-sided blip visible in the raw ms (reported by T2, kept as measured).

**Why the inline array is `MaybeUninit` (T2 §6a)** — the README "Safety"
section's "zero-filling it on every query was measured as a real
regression". T2's first attempt used a plain, eagerly
`Frame::default()`-filled `[Frame; 128]`; same interleaved methodology,
A = `549f1ac`, B = that first attempt (never committed):

| rep | A ratio | B ratio |
|---|---|---|
| 1 | 1.134 | 1.100 |
| 2 | 1.107 | 1.123 |
| 3 | 1.051 | 1.168 |
| 4 | 1.057 | 1.189 |
| 5 | 1.033 | 1.148 |
| 6 | 1.071 | 1.137 |
| 7 | 1.069 | 1.108 |
| 8 | 1.038 | 1.173 |
| **median** | **1.052** | **1.14** |

B worse in 7/8 pairs; T2's asm of that build showed a per-query ~4 KiB
fill of the frame array (T2 §7 and Fix round 1 Item 6 for the corrected
description of the fill sequence). Switching the storage to `MaybeUninit`
(O(1) `FrameStack::new()`, no fill) produced the 1.0205 median above.

### M2.5 task 4: full regression sweep (commit `2d23db4`, milestone close-out)

Both of T2 (explicit-stack iterative `search_level`) and T3 (bounds-check-free
chunked kernel row walk) are landed at this commit. Every command below was
run fresh for this task; full workspace `cargo build --workspace --release`
(with `RUSTFLAGS="-C target-cpu=native"`) preceded all timed runs, and the
two miri commands (§3's "Miri" subsection above) were also both re-run —
see their results at the end of this subsection.

**Six perf gates, run 1** (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native"
cargo test -p xval --release --test perf_gate -- --ignored perf_gate
--test-threads=1 --nocapture`):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.955ms cpp=9.224ms ratio=1.079
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.793ms cpp=3.383ms ratio=1.121
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=28.514ms cpp=32.650ms ratio=0.873
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.136ms cpp=6.859ms ratio=1.040
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=170.552ms cpp=179.529ms ratio=0.950
PERF_GATE perf_gate_radius_dim3_f32: rust=4.993ms cpp=6.040ms ratio=0.827
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.82s
```

**Six perf gates, run 2** (same command, minutes later, no code changes in
between):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.918ms cpp=10.182ms ratio=0.974
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.939ms cpp=3.366ms ratio=1.170
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=28.443ms cpp=31.053ms ratio=0.916
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.047ms cpp=6.969ms ratio=1.011
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=171.509ms cpp=177.602ms ratio=0.966
PERF_GATE perf_gate_radius_dim3_f32: rust=4.917ms cpp=5.940ms ratio=0.828
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.81s
```

All twelve gate invocations pass the 1.25 margin. Honest ranges across
these two fresh runs: `build_100k` 0.974–1.079, `dyn_add` 1.121–1.170,
`dyn_knn_after_churn` **0.873–0.916** (notably better than the
0.971–0.976 previously recorded — T2's churn win, confirmed), `knn_fixed3`
1.011–1.040 (T2's win, confirmed), `knn_dyn_dim8_f64_k10` 0.950–0.966
(T2's give-back from T3's post-fix 0.867, landing back inside the pre-M2.5
pasted range 0.944–1.091 — 0.944–0.984 excluding the flagged 1.091
outlier), `radius_dim3_f32` 0.827–0.828 (T2's give-back from T3's post-fix
0.777, landing ~4.4% above the top of the pre-M2.5 pasted range
0.767–0.792 — the one figure in this sweep that sits outside its recorded
historical range, honestly reported, not smoothed over).

**The pre-M2.5 pasted range** ("milestone-start" in `docs/benchmarks.md`'s
M2.5 summary table) is the min–max across every pre-M2.5 gate run pasted
in this repo: this document's four pre-M2.5 six-gate runs above (Re-run
1/2 in the original round, Re-run 1/2 in the M2 fix wave), the
`--ignored perf_gate_dyn` runs above, `docs/benchmarks.md`'s M1 table
("After Task 14" and "Final single run" columns), and the "before" arm of
T3's interleaved A/B (the "M2.5 task 3" subsection above, pre-T3 code):
`build_100k` 0.986–1.018, `knn_fixed3` 1.030–1.063, `knn_dyn_dim8_f64_k10`
0.944–1.091 (0.944–0.984 excluding the one flagged outlier), `radius`
0.767–0.792, `dyn_add` 1.106–1.237, `dyn_knn_after_churn` 0.956–0.976
(corrected workload only).

**dim-32/64, run 1** (`RUSTFLAGS="-C target-cpu=native" cargo run -p xval
--release --example m25_diag -- knn`, n=100k, 200 queries, k=10, leaf=10 —
same "gate-style knn" methodology T1/T3 used; dim 8/16 rows omitted here,
not part of this task's scope):
```
dim,scalar,rust_ms,cpp_ms,ratio
32,f32,199.803,206.866,0.966
32,f64,357.606,324.088,1.103
64,f32,425.920,363.746,1.171
64,f64,667.777,759.691,0.879
```

**dim-32/64, run 2** (same command, minutes later):
```
dim,scalar,rust_ms,cpp_ms,ratio
32,f32,202.357,200.815,1.008
32,f64,354.508,319.199,1.111
64,f32,422.310,363.564,1.162
64,f64,676.539,762.898,0.887
```

Both runs also printed dim-8/16 rows (omitted above, out of this task's
dim-32/64 scope) — dim-16 showed an unexplained large jump versus dim-8
(both f32/f64 ~76–88ms vs. dim-8's ~3.2–3.5ms, consistent across both runs,
so not noise) that is plausibly a curse-of-dimensionality traversal cliff
at this dataset/leaf-size combination, not measured or root-caused further
by this task — out of scope (T1/T3 only diagnosed dim-32/64), flagged here
so it isn't silently dropped from the record. (Diagnosed below: the "M2.5
final-review fix wave" subsection pastes the dim-16 CSV rows and the M8
`m25_diag count` run confirming curse-of-dimensionality.)

**Full test suite** (`cargo test --workspace --all-features`): **359
passed, 0 failed, 15 ignored** (up from 345/0/15 recorded at M2's
final-review fix wave).

**Heavy/`--ignored` suite** (`cargo test --workspace --all-features
--release -- --ignored --test-threads=1`): **0 failed** across all 15
previously-ignored tests; the `nanoflann-rs` unit-test binary (the four
heavy build/query tests, including `heavy_query_degenerate_trees` at depth
~16,794, now exercising T2's `FrameStack` spill path) finished in
**151.80s**; all six `perf_gate_*` tests pass (self-skipping without
`PERF_GATE=1`, as before); all five mutation canaries
(`xval_dynamic`/`xval_knn`/`xval_radius_box`) still correctly detect their
injected mutations.

**Clippy** (`cargo clippy --workspace --all-targets -- -D warnings`):
clean, zero warnings.

**Rustdoc** (`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`):
clean, zero warnings.

**`--no-default-features`** (`cargo build -p nanoflann-rs
--no-default-features` then `cargo test -p nanoflann-rs
--no-default-features`): clean build; **179 passed, 0 failed, 3 ignored**.

**Miri** (both commands from §3's "Miri" subsection above, re-run for this
task):
```
$ cargo +nightly miri test -p nanoflann-rs --lib -- search
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s

$ MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p nanoflann-rs --lib -- search
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s
```
Both clean under Stacked Borrows and Tree Borrows — no UB detected in
`FrameStack`'s `unsafe` code, including under the real spill path
(`spill_boundary_deep_tree_knn_and_radius_match_brute_force`).

**Provenance flag on the identical `488.12s` figures above**: both lines
were genuinely pasted from real runs at the time this subsection was
written, but the whole-branch final review correctly flagged that two
independent ~8-minute interpreted runs reporting the exact same duration
to the centisecond is implausible on its face and warranted a fresh,
skeptical re-capture — see the "M2.5 final-review fix wave" subsection
immediately below for that re-capture and the (surprising, but
documented) explanation: the two runs' *real* host wall-clock times do
differ substantially, but miri's own internal elapsed-time report does
not, for a specific, reproducible reason.

### M2.5 final-review fix wave: miri provenance re-capture + dim-16 CSV (tree committed as `4b9fbc3`; timed code identical to `2d23db4` — see §1's SHA table)

Both miri commands (§3's "Miri" subsection) were re-run FRESH for this fix
wave, this time each wrapped in the shell's own `time` to capture genuine
host wall-clock duration independent of whatever the interpreted test
binary itself reports, specifically to settle the "centisecond-identical"
provenance concern raised against the M2.5-task-4 figures above.

```
$ time (cargo +nightly miri test -p nanoflann-rs --lib -- search)
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s
( cargo +nightly miri test -p nanoflann-rs --lib -- search ) 154.14s user 0.18s system 99% cpu 2:34.36 total

$ time (MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p nanoflann-rs --lib -- search)
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s
( MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p nanoflann-rs --lib -- search ) 264.26s user 0.77s system 99% cpu 4:26.15 total
```

**These two runs are genuinely different measurements, and they DO
differ** — just not in the field the earlier documentation round pasted.
Real host wall-clock time: Stacked Borrows **2:34.36 (154.36s)**, Tree
Borrows **4:26.15 (266.15s)** — a 1.7x spread, consistent with Tree
Borrows' stricter aliasing model doing genuinely more interpreter work per
memory access. The `test result: ... finished in 488.12s` line, by
contrast, is identical across both runs (and matches the M2.5-task-4
figures above too) — **not** because the number was copy-pasted or the
run wasn't real, but because that duration is computed *inside* miri's
interpreted sandbox via `Instant`/`SystemTime`, and miri's default
isolation mode (undoing this needs `-Zmiri-disable-isolation`, not used
here or previously) advances the interpreted program's synthetic clock
deterministically from the instruction stream rather than sampling the
real system clock — so two runs that execute the identical 34-test
sequence in the identical order report the identical synthetic elapsed
time, regardless of how long the interpretation actually took on the host.
Net conclusion: no UB in either run (both `ok`, both `34 passed; 0 failed;
1 ignored`), and the identical `488.12s` reported by miri's own harness is
an artifact of miri's clock isolation, not evidence of a stale or reused
number — confirmed here by wrapping both commands in a real, host-side
timer that shows genuinely different elapsed times.

**`m25_diag knn` dim-16 rows (M8/I4b — previously published without their
pasted CSV)**: the README's/`docs/benchmarks.md`'s in-passing mention of
dim-16's "~76–88ms vs. dim-8's ~3.2–3.5ms" cited both M2.5-task-4 runs but
never pasted the underlying rows. Fresh run below (`RUSTFLAGS="-C
target-cpu=native" cargo run -p xval --release --example m25_diag -- knn`,
same n=100k/200-queries/k=10/leaf=10 methodology as the dim-32/64 table),
CPU otherwise idle (no background miri/benchmark process running):

```
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.547,3.933,0.902
8,f64,3.322,3.632,0.915
16,f32,76.148,115.954,0.657
16,f64,87.086,95.623,0.911
32,f32,199.066,200.189,0.994
32,f64,360.144,324.031,1.111
64,f32,384.852,329.154,1.169
64,f64,681.040,773.105,0.881
```

(dim-32/64 rows included here too since this was a fresh full run of the
same command that produces the README's/`docs/benchmarks.md`'s headline
table — consistent with those, modulo ordinary WSL2 run-to-run swing.)

Confirms the previously-cited ranges (dim-8 ~3.3–3.5ms, dim-16 ~76–87ms —
inside the earlier "~76–88ms vs ~3.2–3.5ms" characterization) with the
actual pasted numbers now on record. `ratio` stays a Rust win at every
row above (dim-16 f32's 0.657 is the widest margin of the whole table this
run — still the same shared traversal-cost cliff discussed below, not a
Rust-vs-C++ regression: both sides pay the near-half-dataset scan, Rust
just pays it somewhat faster this run).

**M8 — curse-of-dimensionality hypothesis test (`m25_diag count`, dim 16
added to the existing sweep)**: fresh run, `RUSTFLAGS="-C
target-cpu=native" cargo run -p xval --release --example m25_diag --
count` (n=100k, leaf=10, 1000 queries per dim, `frac_points_scanned` =
mean leaf-scan evaluations per query divided by n):

```
dim,leaf,scalar,queries,eval_per_query,interior_per_query,frac_points_scanned
3,10,f32,1000,66.0,29.1,0.0007
8,10,f32,1000,1756.7,540.4,0.0176
16,10,f32,1000,48748.2,9515.9,0.4875
32,10,f32,1000,99999.9,14438.7,1.0000
```

**Confirmed, not just plausible**: `frac_points_scanned` jumps from 1.76%
at dim 8 to **48.75% at dim 16** (a ~27.7x increase) en route to an
exhaustive 100.00% scan at dim 32. The evaluated-fraction ratio (~27.7x)
lines up closely with the dim-16-vs-dim-8 rust_ms timing ratio measured
just above (76.148/3.547 ≈ 21.5x f32, 87.086/3.322 ≈ 26.2x f64) — close
enough that the evaluated-fraction jump is sufficient on its own to
explain the timing cliff, with no separate code-path anomaly required.
This upgrades the dim-16 lead from "undiagnosed" to a confirmed mechanism:
garden-variety curse-of-dimensionality (the kd-tree's pruning bound
degrades and the search degenerates toward a near-exhaustive scan well
before dim 32's
already-known 100% scan), not a bug or regression. See
`docs/ROADMAP.md`'s M2.5 future-perf-leads list for the updated framing.

### M2.5 asm re-capture: zero `search_level` call targets (commit `84c0781`, 2026-08-23)

Backs the README "Perf gate" paragraph's "0 `search_level` call targets
remain in the compiled asm" and `docs/benchmarks.md`'s "Update (M2.5-T2)"
banner. `crates/xval/examples/m25_asm.rs`'s `#[no_mangle] #[inline(never)]
probe_knn3` forces the `ConstDim<3>`/f32 knn monomorphization. Re-run
fresh for this document (`git rev-parse --short HEAD` = `84c0781`,
`crates/` clean; `crates/` vs. `2d23db4` is doc comments plus one
`debug_assert!`-gated helper, see §1's SHA table):

```
$ RUSTFLAGS="-C target-cpu=native" cargo rustc -p xval --release --example m25_asm -- --emit asm
   Compiling xval v0.0.0 (/home/sitzikbs/dev/flannrust/crates/xval)
    Finished `release` profile [optimized] target(s) in 2.58s
$ ls target/release/examples/m25_asm-*.s | wc -l
27
$ cat target/release/examples/m25_asm-*.s | grep -c 'search_level'
0
$ cat target/release/examples/m25_asm-*.s | grep -cE '^\s*call[q]?\s+.*search_level'
0
```

(Thin LTO splits the example into 27 codegen-unit `.s` files; both greps
run over all of them. `cargo rustc --emit asm` only regenerates the `.s`
files when the example actually recompiles — `touch
crates/xval/examples/m25_asm.rs` first if a cached build reports
`Finished` without a `Compiling` line.) Bounding `probe_knn3`'s own body
with T2's corrected extractor (find the `.Lfunc_beginN:` label after
`probe_knn3:`, stop at the matching `.Lfunc_endN:`):

```
$ f=$(grep -l '^probe_knn3:' target/release/examples/m25_asm-*.s)
$ awk -v fn="probe_knn3" '
    $0 ~ "^"fn":" {grab=1}
    grab && /^\.Lfunc_begin[0-9]+:/ {n=$0; sub(/^\.Lfunc_begin/,"",n); sub(/:$/,"",n); endlabel=".Lfunc_end" n ":"}
    grab {print}
    grab && endlabel != "" && $0==endlabel {exit}
  ' "$f" > probe_knn3.s
$ wc -l < probe_knn3.s
569
$ grep -E '^\s*call' probe_knn3.s | sed 's/^\s*//' | sort | uniq -c
      1 callq	*_RINvNtCsc36rpYXAlPq_4core9panicking13assert_failedjjEB4_@GOTPCREL(%rip)
      2 callq	*_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check@GOTPCREL(%rip)
      1 callq	*_RNvNtCsc36rpYXAlPq_4core9panicking9panic_fmt@GOTPCREL(%rip)
      1 callq	*_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail@GOTPCREL(%rip)
      2 callq	*free@GOTPCREL(%rip)
      1 callq	_RNvMs4_NtCscHgRw1M2fX5_5alloc7raw_vecINtB5_6RawVecINtNtCs4tP4CM7yoJ7_12nanoflann_rs6search5FramefEE8grow_oneCs1wF3btebYy0_7m25_asm
      1 callq	_Unwind_Resume@PLT
```

A 569-line function (the same count T2's fix round recorded, Item 6) whose
only `call` instructions are panic/unwind landing pads plus the single
cold `RawVec::<Frame<f32>>::grow_one` (the heap spill past 128 frames) —
no node visit is a call, and the `search_level` symbol no longer exists in
the binary at all: it is fully inlined into `probe_knn3`. Contrast M1's
recorded state (`docs/benchmarks.md` "Remaining gap analysis"): two `callq`
self-calls per interior node. This establishes only that the call overhead
is gone; it does not, by itself, attribute the remaining 1.011–1.040
residual to any specific instruction sequence.

### Criterion benches (full sweep — ~15 minutes; smoke forms below for a quick check)

```
RUSTFLAGS="-C target-cpu=native" cargo bench -p xval
```
Runs all four bench binaries (`bench_build`, `bench_knn`, `bench_radius`,
`bench_dynamic`) at their full criterion sample sizes/measurement windows.
`bench_build.rs`'s own module doc states the full-suite runtime budget is
deliberately held to **~15 minutes** (its `n = 1_000_000 × dim = 8` arm is
skipped for exactly this reason). This full sweep was **not** re-run in
its entirety for this documentation task (per the brief's allowance) —
instead, each of the four bench binaries was smoke-tested individually
with a filter + reduced sample size, to confirm every command actually
runs to completion on this exact command surface:

```
RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench bench_knn -- --quick knn_fixed3/rust/f32/k10
# -> knn_fixed3/rust/f32/k10 time: [735.86 ns 742.74 ns 744.46 ns]  (~10s)

RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench bench_radius -- --quick sel10
# -> radius/rust/sel10 ~409ns, radius/cpp/sel10 ~414ns  (~10s)

RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench bench_build -- \
  leaf_sweep/rust/10 --sample-size 10 --measurement-time 1 --warm-up-time 1
# -> leaf_sweep/rust/1024/build ~4.52ms, leaf_sweep/rust/1024/knn ~3.52us  (~5s)

RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench bench_dynamic -- \
  dyn_add/rust --sample-size 10 --measurement-time 1 --warm-up-time 1
# -> dyn_add/rust ~24.08ms (10 samples, ~55 iterations)  (~2s)
```
(Note: criterion's `--quick` flag and an explicit `--sample-size`/
`--measurement-time` override are mutually exclusive on this criterion
version — use one style or the other per invocation, not both, as shown
above.) All four smoke commands exited 0 and produced the expected
benchmark-group output; none of the four bench binaries has bit-rotted.
The headline numbers quoted in `docs/benchmarks.md` come from the full,
un-truncated sweep run during M1's and M2's own performance passes, not
from these smoke commands — the smoke commands exist only to prove the *commands*
still work, not to re-derive the headline figures.

### The two-command report chain

```
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example report_data > report.json
cargo run -p xval --release --example render_report -- report.json > report.html
```
Expected runtime: **under a minute** for `report_data` (dominated by the
three static + one dynamic brute-force ground-truth computations, each
O(n_queries × n) — the slowest is 2000 queries × 50,000 points); `report.json`
is regenerated fully every run, not incrementally, so there is no cheaper
partial-update path.  `render_report` itself is near-instant (pure
string/template work over an already-computed JSON).

**Current snapshot — re-run for M2.5 task 4** (commit `2d23db4`, 2026-08-23;
exact byte counts via `wc -c`, not the block-rounded `du -h` figures used
in earlier rounds): both exited 0; `report.json` was **6495 bytes**
(~6.3KB) and `report.html` was **14177 bytes** (~13.9KB); verdict tiles
computed live from the data read:

```
Accuracy @ eps=0:  ✓ 100% exact @ eps=0
Bit-exactness:     ✓ Bit-exact vs C++ (all rows)
Best speed win:    ✓ 1.47× faster — build_1M_dim3_f32_par
```

(Superseded, kept for context: a prior documentation round, pre-M2.5,
recorded `report.json` 6495 bytes / `report.html` **14179** bytes and a
**1.58×** best-speed-win tile — the 2-byte HTML difference and the win-tile
delta both trace to ordinary WSL2 run-to-run timing noise on the SAME
speed row, `build_1M_dim3_f32_par`, not a different row taking over the
"best speed win" title, and not a code or methodology change between the
two snapshots.)

**Reconciling this tile's 1.47–1.58× against the criterion-table "roughly
1.75x faster" claim (both `build_1M_dim3_f32_par`)**: same workload,
different methodology, not a contradiction. The criterion figure is the
statistically-modeled result of `cargo bench` (hundreds of iterations,
outlier rejection, warm-up); the scorecard tile is `report_data`'s own
`timed_median_ms` (one warmup + median of 7 raw wall-clock runs, the same
lighter-weight methodology used throughout this doc's perf-gate sections)
— a smaller, noisier sample that swings a few tenths of an x-factor
run-to-run on this WSL2 host, which is exactly the 1.47–1.58× spread seen
above. Both stay on the same side of "substantially faster"; neither
number is wrong, they're just two different measurement instruments
pointed at the same workload.

`report_data`'s own emitted `meta` block is this run's environment
fingerprint (see §1's table — those values were read directly from this
exact run's JSON).

### M-py: Python bindings environment, parity, and two bench runs (range honesty)

Python environment (`crates/flannrust-py/.venv`, captured live by
`bench_py.py`'s own `build_meta()` in every run's `meta` block — never
hand-typed):

```
python 3.12.11, numpy 2.5.2, scipy 1.18.1, pynanoflann 0.10.0
```

Same machine/methodology as every other section of this document (WSL2,
AMD Ryzen 7 9800X3D, `RUSTFLAGS="-C target-cpu=native"`). Rebuild the
bindings before any timed run:

```
export PATH="$HOME/.cargo/bin:$PATH"
cd crates/flannrust-py
RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release
```

Full test suite (Rust + Python, exact commands used for T6's final
verification, see the "Verify" section of `task-6-report.md`):

```
cargo test --workspace          # compiles+tests flannrust-py too, once the venv's PYO3 env is active
cd crates/flannrust-py && .venv/bin/python -m pytest python/tests -q
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust --no-deps
cargo build -p flannrust --no-default-features
```

Bench (methodology, cross-checks, and full result-row table for the first
of the two runs below: `task-5-report.md`'s "Bench methodology, as run" and
"Full result rows" sections — reproduced here since that report is
gitignored):

```
cd crates/flannrust-py
export RUSTFLAGS="-C target-cpu=native"
.venv/bin/python python/bench/bench_py.py > /tmp/report_py.json
```
~7 minutes wall clock, dominated by the dim-32 knn cell (curse-of-dimensionality
makes kd-tree pruning far less effective at dim 32 for all three engines —
same phenomenon `docs/EXPERIMENTS.md`'s M2.5 dim-16/32 sections document for
the Rust-vs-C++ side).

#### Parity: what "M-py parity" means, scoped per the controller's binding ruling

`pynanoflann` 0.10.0 (the only release on PyPI) vendors its own copy of
`nanoflann.hpp` at version **1.5.5** (`NANOFLANN_VERSION 0x155`), while
flannrust's own C++ xval reference (`crates/nanoflann-ref/cpp/nanoflann.hpp`)
is **1.12.1** — the exact version the Rust kernel is deliberately bit-matched
against (`crates/flannrust/src/metric.rs`'s `l2_eval_row`). Between 1.5.5 and
1.12.1, nanoflann's `L2_Adaptor`/`L1_Adaptor::evalMetric` changed how the
4-wide unrolled loop's partial sums combine (sequential chain in 1.5.5 vs.
pairwise in 1.12.1) and the `dim < 4` remainder loop's iteration order
flipped — floating-point addition is commutative but not associative, so
these two valid summation orders produce squared-distance results differing
by exactly 1 ULP for any `dim >= 3`. This is a version-gap environment
artifact against this one specific `pynanoflann` build, not a flannrust
defect — verified via code diff and empirical measurement (dim=2: 0/3000
mismatches, provably immune 2-term sum; dim=3: 339/3000; dim=8: 626/3000;
dim=32: 1195/3000, every mismatch exactly 1 ULP, never flipping which point
is nearest at ordinary tie-free scale). Full root-cause writeup, code diff,
and the escalation trail: `task-3-report.md`'s "Escalation" section
(gitignored — reproduced here) and
`crates/flannrust-py/python/tests/test_parity_pynanoflann.py`'s module
docstring (checked into the repo, not gitignored — the authoritative live
copy of this finding).

**Controller ruling (binding, accepted in task-3's fix round 1/5):** the
parity claim is scoped to what holds unconditionally against this
`pynanoflann` build: (a) tie-free KNN index-sequence parity across the full
spec'd matrix (dims `{2,3,8,32}` × `{f32,f64}` × leaf `{1,10,64}` × metric
`{l2,l1}` × `k ∈ {1,10}`, 200 seeded queries, 20 for n=20000) — **96/96
nodes pass, bit-exact index sequences**; (b) `dim==2` KNN distance-value
bit-exactness (a pure 2-term, commutative-only sum, provably immune to the
1.5.5-vs-1.12.1 gap) — **4/4 pass**. The 16 parametrized nodes attributable
to the documented 1-ULP boundary-flip mechanism (10 in radius-index parity at
dim ∈ {8,32}/float32, 6 in the tie-multiset comparison at
dim ∈ {3,8,32}/float32) are pinned as
targeted `xfail(strict=True)` — not blanket-marked whole test functions —
so an unexpected pass would show up loudly (XPASS) rather than being
silently absorbed. Combined suite run
(`.venv/bin/pytest python/tests/test_parity_pynanoflann.py -q`):
`0 failed, 244 passed, 27 xfailed, 1 xpassed` (the 1 xpassed is a
pre-existing, documented `strict=False` data-dependent dim=3/float64/l1
flake in the distance-bit-exactness xfail, not a new issue). No tolerance
was ever loosened anywhere in this suite to force a green result.

#### Rust perf gates, re-run for T6 (confirms the T1–T5 rename/additions didn't move anything)

```
$ PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release \
  --test perf_gate -- --ignored perf_gate --test-threads=1 --nocapture

PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=10.848ms cpp=9.520ms ratio=1.139
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.923ms cpp=3.530ms ratio=1.111
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.985ms cpp=33.286ms ratio=0.901
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.334ms cpp=7.171ms ratio=1.023
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=179.656ms cpp=187.852ms ratio=0.956
PERF_GATE perf_gate_radius_dim3_f32: rust=5.240ms cpp=6.204ms ratio=0.845
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 4.06s
```
All six pass the 1.25 margin with wide room. Comparing against
`docs/benchmarks.md`'s M2.5 "Post-T2/final" ranges (`build_100k`
0.974–1.079, `knn_fixed3` 1.011–1.040, `dim8` 0.950–0.966, `radius`
0.827–0.828, `dyn_add` 1.121–1.170, `dyn_knn_after_churn` 0.873–0.916):
`dyn_add` (1.111), `dyn_knn_after_churn` (0.901), `knn_fixed3` (1.023), and
`dim8` (0.956) all land inside or within a couple points of their recorded
ranges — ordinary WSL2 noise (§1). Two widen the previously-recorded range
honestly, not as a regression (M-py's T0–T5 changes never touched
`crates/flannrust`'s query/build/kernel code, only added `flannrust-py` on
top and one rename): `build_100k` at **1.139** sits above the prior
0.974–1.079 band (new honest range **0.974–1.139**), and `radius_dim3_f32`
at **0.845** sits above the prior 0.827–0.828 band (new honest range
**0.827–0.845**) — both comfortably under the 1.25 gate, both single
additional data points widening an already-narrow recorded band, consistent
with this host's documented noise floor (`knn_fixed3` 0.956–1.192 over 8
runs, "M2.5 task 1" above), not a code-driven change.

#### Bench run 1 (from `task-5-report.md`, gitignored — reproduced verbatim here)

`meta`: `git_sha=509cd36`, `date=2026-08-24T08:19:25Z`, `rustflags=-C target-cpu=native`.

| workload | flannrust_ms | ckdtree_ms | pynanoflann_ms | ratio_ckdtree | ratio_pynanoflann |
|---|---|---|---|---|---|
| build_100k_dim3_f32_threads1 | 9.576 | 11.669 | 17.552 | 0.821 | 0.546 |
| build_1M_dim3_f32_threads1 | 130.388 | 132.147 | 239.969 | 0.987 | 0.543 |
| knn_batched_dim3_f32_k10_q200k_workers1 | 306.391 | 430.344 | 289.668 | 0.712 | 1.058 |
| knn_batched_dim3_f32_k10_q200k_workersNeg1 | 45.095 | 63.362 | 45.075 | 0.712 | 1.000 |
| knn_dim8_float64_k10_workers1 | 392.347 | 510.625 | 318.081 | 0.768 | 1.233 |
| knn_dim32_float32_k10_workers1 | 22023.360 | 43101.397 | 25082.533 | 0.511 | 0.878 |
| radius_dim3_f32_sel10 | 2.606 | 4.082 | 4.415 | 0.638 | 0.590 |
| radius_dim3_f32_sel1000 | 66.102 | 129.133 | 230.621 | 0.512 | 0.287 |
| single_query_loop_dim3_f32_k10_percall_ms | 0.001968 | 0.007245 | 0.002273 | 0.272 | 0.866 |

(`threadsNone_parallel_build` rows omitted from this table — same
apples-to-oranges caveat as every other section of this document; both
runs agree they're not a same-config parity claim, see the row `note` in
the raw JSON.)

#### Bench run 2 (fresh, run for this task — range honesty per the brief)

```
$ export PATH="$HOME/.cargo/bin:$PATH"
$ cd crates/flannrust-py
$ RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release   # rebuilt first
$ RUSTFLAGS="-C target-cpu=native" .venv/bin/python python/bench/bench_py.py > /tmp/report_py_run2.json
```
`meta`: `git_sha=0cea30c`, `date=2026-08-24T08:49:29Z`, `rustflags=-C target-cpu=native`
(same `python 3.12.11`/`numpy 2.5.2`/`scipy 1.18.1`/`pynanoflann 0.10.0` as
run 1). All cross-checks passed, including the radius-specific cross-check
added in task 5's fix round (`radius_cross_check`, exercised twice — once
per selectivity — right before `radius_workload`'s timed cells):

```
[bench_py] cross-check OK (build_100k_dim3_f32 dataset): 10/10 index match, max rel dist err 1.09e-07
[bench_py] cross-check OK (build_1M_dim3_f32 dataset): 10/10 index match, max rel dist err 1.18e-07
[bench_py] cross-check OK (knn_batched_dim3_f32 dataset): 10/10 index match, max rel dist err 1.07e-07
[bench_py] cross-check OK (knn_dim8_float64 dataset): 10/10 index match, max rel dist err 2.05e-16
[bench_py] cross-check OK (knn_dim32_float32 dataset): 10/10 index match, max rel dist err 1.40e-07
[bench_py] cross-check OK (radius_dim3_f32 dataset): 10/10 index match, max rel dist err 1.13e-07
[bench_py] radius cross-check OK (radius_dim3_f32_sel10): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)
[bench_py] radius cross-check OK (radius_dim3_f32_sel1000): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)
[bench_py] cross-check OK (single_query_loop dataset): 10/10 index match, max rel dist err 1.18e-07
```

| workload | flannrust_ms | ckdtree_ms | pynanoflann_ms | ratio_ckdtree | ratio_pynanoflann |
|---|---|---|---|---|---|
| build_100k_dim3_f32_threads1 | 9.956 | 12.012 | 17.739 | 0.829 | 0.561 |
| build_1M_dim3_f32_threads1 | 132.574 | 138.466 | 240.421 | 0.957 | 0.551 |
| knn_batched_dim3_f32_k10_q200k_workers1 | 353.626 | 426.143 | 290.905 | 0.830 | 1.216 |
| knn_batched_dim3_f32_k10_q200k_workersNeg1 | 51.310 | 62.042 | 57.285 | 0.827 | 0.896 |
| knn_dim8_float64_k10_workers1 | 395.028 | 520.524 | 316.390 | 0.759 | 1.249 |
| knn_dim32_float32_k10_workers1 | 22234.437 | 43737.826 | 24741.719 | 0.508 | 0.899 |
| radius_dim3_f32_sel10 | 2.550 | 4.154 | 4.392 | 0.614 | 0.581 |
| radius_dim3_f32_sel1000 | 66.663 | 131.109 | 232.162 | 0.508 | 0.287 |
| single_query_loop_dim3_f32_k10_percall_ms | 0.002255 | 0.007501 | 0.002293 | 0.301 | 0.983 |

#### Honest ranges across both runs (the figures `docs/benchmarks.md`'s M-py section and the README Python section cite)

| workload | ratio_ckdtree range | ratio_pynanoflann range |
|---|---|---|
| `build_100k_dim3_f32_threads1` | 0.821–0.829 | 0.546–0.561 |
| `build_1M_dim3_f32_threads1` | 0.957–0.987 | 0.543–0.551 |
| `knn_batched_dim3_f32_k10_q200k_workers1` | **0.712–0.830** | 1.058–1.216 |
| `knn_batched_dim3_f32_k10_q200k_workersNeg1` | **0.712–0.827** | 0.896–1.000 |
| `knn_dim8_float64_k10_workers1` | 0.759–0.768 | 1.233–1.249 |
| `knn_dim32_float32_k10_workers1` | 0.508–0.511 | **0.878–0.899** |
| `radius_dim3_f32_sel10` | 0.614–0.638 | 0.581–0.590 |
| `radius_dim3_f32_sel1000` | 0.508–0.512 | 0.287 (identical both runs) |
| `single_query_loop_dim3_f32_k10_percall_ms` (overhead-bound; ratio, not ms) | 0.272–0.301 | 0.866–0.983 |

(Bold ranges are the three spec'd success-criteria cells, all fully inside
their gates across both runs.)

#### Success criteria vs. the M-py spec (honest accounting, both runs)

1. **Batched knn dim3 f32 ≤ 1.00 vs cKDTree, workers 1 and −1**: **MET**
   both runs, both worker counts — range 0.712–0.830.
2. **Build ≤ 1.00 vs pynanoflann**: **MET** both runs, both sizes — range
   0.543–0.561 (100k), 0.543–0.551 (1M).
3. **dim-32 ≤ 1.10 vs pynanoflann**: **MET** both runs — range 0.878–0.899.
4. **Per-call overhead measured**: **MET** — `single_query_loop_*_percall_ms`
   row, both runs (flannrust ≈2.0–2.3µs/call, cKDTree ≈7.2–7.5µs/call,
   pynanoflann ≈2.3µs/call; ratios in the table above, explicitly labeled
   `overhead-bound` in the rendered report so it isn't read as a
   steady-state throughput number).

**Honest misses, published alongside** (not spec'd success criteria, not
gating, but reported for completeness per this repo's standing "publish the
losses too" convention): `knn_batched_dim3_f32_k10_q200k_workers1` vs.
pynanoflann ranges **1.058–1.216** (flannrust 6–22% slower there across the
two runs); `knn_dim8_float64_k10_workers1` vs. pynanoflann ranges
**1.233–1.249** (flannrust 23–25% slower, the most consistent miss in the
whole matrix — both runs agree closely, so this reads as a genuine gap, not
noise). At `workers=-1` the batched-dim3-vs-pynanoflann ratio drops to
0.896–1.000 (near parity or better) — the miss is specifically a
single-threaded-call-overhead shape, not an algorithmic one. flannrust beats
cKDTree on every single row, both runs, without exception. See
`docs/ROADMAP.md`'s M-pub section for the open investigation this dim8 f64
gap motivates (pynanoflann's 1.5.5 kernel/pybind11 marshalling vs. this
crate's own 1.12.1-oracle cross-check, which independently measures
flannrust at 0.95x against the C++ reference at the same dim/dtype — so the
gap is specifically a pynanoflann-vs-flannrust shape, not a flannrust
regression against its own C++ ground truth).

### M2.6 task 2: statistical baseline (n>=10..100, mean±std) and full conclusion re-verification (commit `a30a819`, 2026-08-25)

M2.6 task 1 (commit `a30a819`) replaced the fixed median-of-7 timing
methodology in `crates/xval/tests/perf_gate.rs` and
`crates/xval/examples/report_data.rs` with `xval::measure`/`measure_pair`:
2 discarded warmup reps, then an **adaptive `n = clamp(10, 100,
floor(budget_s*1000 / t_est_ms))`** timed reps of BOTH sides — `budget_s =
30` per side, so every gate/report speed row below ran with **`n=100`**
(every workload's per-rep cost was fast enough to hit the upper clamp).
`TimingStats` (`mean_ms`/`std_ms`(sample, `n-1`)/`median_ms`/`min_ms`/
`max_ms`/`n`) is published for BOTH sides; gate PASS/FAIL and every ratio
claim in this repo remain the ratio of MEDIANS (unchanged decision rule —
`assert_gate!` in `perf_gate.rs`), never the ratio of means. `xval::median_of`/
`timed_median_ms` (the old fixed-7 helpers) are unchanged and still back
`m25_diag.rs` (see below) — T1 did not touch that file.

**Interleave order (T1 review deferred item, closed here):**
`measure_pair`'s per-repetition interleave is `rust_fn()` then `cpp_fn()`,
always in that fixed order, every rep including the 2 warmups — never
alternated or randomized. This means any systematic per-rep ordering bias
(e.g. the second call in a back-to-back pair running on a slightly warmer
cache/branch-predictor state, or a scheduler quantum boundary landing
consistently after the first call) would land on the C++ side every rep,
not average out across reps. No evidence of such a bias was found in this
task's data (see the noise-floor characterization below — the ratio
distribution is well-behaved, not skewed in a way that points to a fixed
one-sided artifact), but the fixed order is a real, accepted methodological
choice, not an oversight: the 1.25 gate margin and the min/max spreads
documented throughout this file are the safety margin that absorbs
whatever residual order bias exists, the same way they already absorb
WSL2 scheduler/thermal noise (§1). Alternating the order per rep (or per
session) was considered and explicitly not implemented — out of this
task's "re-run and re-verify, no harness rewrites" scope; flagged here as
a real, still-open methodological caveat, not silently accepted.

Every run below was captured fresh for this task, `export PATH="$HOME/.cargo/bin:$PATH"`
first, `RUSTFLAGS="-C target-cpu=native"` set as usual. `git status --porcelain`
was clean for every file except `crates/xval/examples/m25_diag.rs` throughout
this task (`crates/xval/tests/perf_gate.rs`, `crates/xval/examples/report_data.rs`,
`crates/xval/src/lib.rs`, `crates/xval/src/report.rs` are all exactly the
committed `a30a819` code) — `m25_diag.rs`'s only change is this task's own
`RUNS: usize = 7 -> 15` constant bump plus a doc-comment update (see the
"m25_diag" subsection below), not a structural rewrite.

#### Six perf gates, session 1 (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --ignored perf_gate --test-threads=1 --nocapture`)

```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.558ms cpp=9.885ms ratio=0.967 | rust mean=9.596 std=0.170 n=100 | cpp mean=9.928 std=0.197 n=100 | ratio_medians=0.967
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.095ms cpp=3.589ms ratio=1.141 | rust mean=4.125 std=0.148 n=100 | cpp mean=3.636 std=0.128 n=100 | ratio_medians=1.141
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=31.123ms cpp=33.094ms ratio=0.940 | rust mean=32.682 std=9.345 n=100 | cpp mean=34.848 std=8.070 n=100 | ratio_medians=0.940
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.702ms cpp=7.462ms ratio=1.032 | rust mean=7.737 std=0.172 n=100 | cpp mean=7.479 std=0.165 n=100 | ratio_medians=1.032
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=181.210ms cpp=193.603ms ratio=0.936 | rust mean=182.745 std=9.476 n=100 | cpp mean=197.367 std=33.783 n=100 | ratio_medians=0.936
PERF_GATE perf_gate_radius_dim3_f32: rust=5.360ms cpp=6.631ms ratio=0.808 | rust mean=5.378 std=0.131 n=100 | cpp mean=6.658 std=0.128 n=100 | ratio_medians=0.808
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 51.33s
```

#### Six perf gates, session 2 (same command, minutes later, separate invocation, no code changes in between)

```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.639ms cpp=9.971ms ratio=0.967 | rust mean=9.653 std=0.086 n=100 | cpp mean=10.098 std=0.943 n=100 | ratio_medians=0.967
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=4.327ms cpp=3.766ms ratio=1.149 | rust mean=4.878 std=1.210 n=100 | cpp mean=4.280 std=1.082 n=100 | ratio_medians=1.149
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=30.758ms cpp=32.716ms ratio=0.940 | rust mean=30.810 std=0.475 n=100 | cpp mean=32.803 std=0.514 n=100 | ratio_medians=0.940
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.581ms cpp=7.358ms ratio=1.030 | rust mean=7.654 std=0.243 n=100 | cpp mean=7.419 std=0.229 n=100 | ratio_medians=1.030
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=182.580ms cpp=194.829ms ratio=0.937 | rust mean=183.661 std=4.335 n=100 | cpp mean=195.774 std=3.749 n=100 | ratio_medians=0.937
PERF_GATE perf_gate_radius_dim3_f32: rust=5.350ms cpp=6.633ms ratio=0.807 | rust mean=5.400 std=0.191 n=100 | cpp mean=6.649 std=0.124 n=100 | ratio_medians=0.807
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 51.04s
```

All twelve gate invocations across both sessions pass the 1.25 margin, most
with far more room than the old n=7 sweeps suggested (`dyn_add`, the
tightest gate historically, is 1.141/1.149 here — nowhere near its old
1.237 high-water mark; see item (f) below). The huge `dyn_knn_after_churn`/
`radius`/`dyn_add` `std` values flagged with `(!)` in the raw numbers above
(e.g. session 1's `dyn_knn_after_churn` cpp `std=8.070` against a `mean` of
34.848) are real: a handful of the 100 reps in each affected workload hit a
scheduler/thermal outlier well above the typical per-rep cost (confirmed in
the report-chain JSON below, which publishes `min`/`max` — gate stdout does
not). The MEDIAN stays stable through this (compare `dyn_knn_after_churn`'s
ratio 0.940 in both sessions despite very different `std`/outlier
severity) — a concrete, n=100-scale illustration of why this repo's gate
decision and every doc ratio claim use the ratio of medians, never the
ratio of means.

#### Report chain, run 1 (`RUSTFLAGS="-C target-cpu=native" cargo run -q -p xval --release --example report_data > report.json`)

`meta`: `date=2026-08-25T03:57:57Z`, `git_sha=a30a819-dirty` (dirty only via
this task's `m25_diag.rs` rep-count bump, see above — `report_data.rs`
itself is byte-identical to `a30a819`). Exit 0, **`report.json` 9525
bytes**; `render_report -- report.json > report.html` produced **`report.html`
15988 bytes**. Verdict tiles: `Accuracy @ eps=0: ✓ 100% exact @ eps=0`,
`Bit-exactness: ✓ Bit-exact vs C++ (all rows)`, `Best speed win: ✓ 1.56×
faster — build_1M_dim3_f32_par`. All 11 accuracy rows (3 static datasets ×
3 eps + the dynamic churn row × 2 eps) report `rust_eq_cpp_bitexact: true`
— no divergence, same as every prior round.

Speed rows (`rust`/`cpp` are the full `TimingStats`, `ms`; `n=100` every row):

| workload | rust mean±std (median, min–max) | cpp mean±std (median, min–max) | ratio (medians) |
|---|---|---|---|
| `build_100k_dim3_f32_seq` | 10.139±0.173 (10.123, 9.829–10.905) | 9.768±0.159 (9.747, 9.526–10.395) | 1.0386 |
| `knn_dim3_f32_k10` | 7.703±0.159 (7.688, 7.461–8.184) | 7.244±0.163 (7.203, 6.975–7.791) | 1.0673 |
| `knn_dyn_dim8_f64_k10` | 191.872±4.688 (191.198, 184.354–221.540) | 206.534±4.267 (205.778, 200.130–231.851) | 0.9291 |
| `radius_dim3_f32` | 5.773±0.149 (5.753, 5.515–6.379) | 6.659±0.146 (6.641, 6.383–7.049) | 0.8663 |
| `build_1M_dim3_f32_seq` | 135.698±8.077 (134.215, 129.606–208.543) | 132.105±30.264 (128.487, 124.358–429.832) | 1.0446 |
| `build_1M_dim3_f32_par` | 31.780±0.592 (31.703, 30.726–33.875) | 50.500±2.898 (49.359, 47.443–58.727) | 0.6423 |
| `dyn_add_20k_dim3_f32` | 6.310±5.280 (4.403, 4.058–46.285) | 5.543±4.889 (3.959, 3.570–43.879) | 1.1119 |
| `dyn_knn_after_churn_dim3_f32` | 30.864±1.027 (30.760, 29.270–36.346) | 32.688±0.865 (32.553, 31.380–36.795) | 0.9449 |

`build_1M_dim3_f32_seq` cpp's `max=429.832` (vs. a `median` of 128.487) and
`dyn_add`'s `max=46.285`/`43.879` (vs. medians ~4ms) are single-rep outliers
inside this run's n=100 sample — one or two of the 100 timed reps hit a
severe scheduler/thermal stall (WSL2, §1); the median-based ratio is
unaffected, exactly the robustness property flagged above.

#### Report chain, run 2 (same command, minutes later, separate invocation)

`meta`: `date=2026-08-25T03:59:43Z`, `git_sha=a30a819-dirty` (same dirty
state as run 1). Exit 0, **`report.json` 9527 bytes**, **`report.html` 15990
bytes**. Verdict tiles: `Accuracy @ eps=0: ✓ 100% exact @ eps=0`,
`Bit-exactness: ✓ Bit-exact vs C++ (all rows)`, `Best speed win: ✓ 1.57×
faster — build_1M_dim3_f32_par`. All 11 accuracy rows bit-exact again.

| workload | rust mean±std (median, min–max) | cpp mean±std (median, min–max) | ratio (medians) |
|---|---|---|---|
| `build_100k_dim3_f32_seq` | 10.242±0.146 (10.207, 9.991–10.736) | 9.858±0.140 (9.845, 9.574–10.363) | 1.0368 |
| `knn_dim3_f32_k10` | 7.769±0.175 (7.744, 7.359–8.507) | 7.321±0.201 (7.296, 7.025–8.360) | 1.0614 |
| `knn_dyn_dim8_f64_k10` | 191.415±7.362 (189.118, 182.420–222.847) | 205.517±7.027 (203.662, 195.906–234.736) | 0.9286 |
| `radius_dim3_f32` | 7.211±4.238 (5.768, 5.560–36.334) | 8.075±3.725 (6.655, 6.473–29.236) | 0.8667 |
| `build_1M_dim3_f32_seq` | (not re-tabulated; ratio) | | 1.0469 |
| `build_1M_dim3_f32_par` | (not re-tabulated; ratio) | | 0.6355 |
| `dyn_add_20k_dim3_f32` | 4.153±0.099 (4.123, 4.005–4.497) | 3.698±0.104 (3.668, 3.569–4.033) | 1.1241 |
| `dyn_knn_after_churn_dim3_f32` | 32.271±9.439 (30.715, 29.446–117.220) | 33.766±6.595 (32.434, 31.205–84.022) | 0.9470 |

`dyn_knn_after_churn`'s `max=117.220` in run 2 (vs. a `median` of 30.715 —
nearly 4x) is the single most extreme outlier across every session in this
task; the ratio of medians (0.947) is essentially unaffected, in line with
run 1's 0.940–0.945 and the two gate sessions' 0.940/0.940 — four
independent n=100 samples, all landing within 0.007 of each other on the
median despite wildly different tail behavior. `radius_dim3_f32`'s `max=36.334`
(vs. median 5.768) is the same phenomenon on a different workload.

#### `m25_diag knn`/`count`, RUNS bumped 7→15 (`crates/xval/examples/m25_diag.rs`, this task's only code change)

`m25_diag.rs` predates T1 and uses its own `xval::timed_median_ms`
(one-untimed-warmup + median-of-`RUNS`) methodology, not `measure`/
`measure_pair` — T1 explicitly did not touch it, and this task's brief does
not authorize a structural rewrite to adopt the mean/std harness. What IS
authorized and done here: `RUNS` is a single top-level `const usize`
threaded through every subcommand via `timed_median_ms(RUNS, ...)` — a
trivially parameterizable constant, bumped from **7 to 15** (n>=10 per the
M2.6 statistical-rigor directive) plus a doc-comment update; no other line
in the file changed (`git diff crates/xval/examples/m25_diag.rs` — 12 lines
changed, all in the module doc comment and the `const RUNS` line).
`m25_diag` still reports only the median (no mean/std/min/max — that
information genuinely isn't available from this file's methodology without
the disallowed rewrite), so the two sessions below are presented as
independent point-in-time median measurements, not a distribution — the
"~2×" of sample-count honesty this buys is real, but smaller than the
`measure_pair`-backed sections above.

**`knn` subcommand, session 1** (`RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- knn`, n=100k, 200 queries, k=10, leaf=10, median of 15):

```
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.533,3.770,0.937
8,f64,3.590,3.852,0.932
16,f32,83.222,121.954,0.682
16,f64,99.278,105.414,0.942
32,f32,229.336,216.019,1.062
32,f64,391.067,357.525,1.094
64,f32,369.011,340.936,1.082
64,f64,910.355,1056.527,0.862
```

**`knn` subcommand, session 2** (same command, minutes later, separate invocation):

```
dim,scalar,rust_ms,cpp_ms,ratio
8,f32,3.555,3.742,0.950
8,f64,3.594,3.953,0.909
16,f32,83.430,122.412,0.682
16,f64,99.073,105.445,0.940
32,f32,219.269,222.296,0.986
32,f64,390.048,361.244,1.080
64,f32,508.171,411.479,1.235
64,f64,972.795,1017.689,0.956
```

dim-16 is essentially identical across both sessions (f32 0.682/0.682, f64
0.942/0.940) — see item (d) below. dim-32/64 swing considerably more
between sessions than dim-3/8 do anywhere else in this document (dim-32 f32
0.986–1.062, dim-64 f32 1.082–1.235) — plausibly cumulative thermal drift
across the ~67-70s single-process sweep (dim-64 runs last in each session,
after dim-8/16/32 already ran), not investigated further here (out of this
task's scope — flagged, not chased). Item (a) below addresses dim-32 f32
specifically, the one M2.6 controller-listed conclusion this affects.

**`count` subcommand** (`RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- count`; deterministic — a call-counting metric wrapper, not a wall-clock timing, so a second session reproduced this byte-for-byte, confirmed via `diff`):

```
dim,leaf,scalar,queries,eval_per_query,interior_per_query,frac_points_scanned
3,10,f32,1000,66.0,29.1,0.0007
8,10,f32,1000,1756.7,540.4,0.0176
16,10,f32,1000,48748.2,9515.9,0.4875
32,10,f32,1000,99999.9,14438.7,1.0000
3,1,f32,1000,29.4,65.9,0.0003
3,4,f32,1000,43.3,41.5,0.0004
3,32,f32,1000,120.7,19.0,0.0012
3,128,f32,1000,288.5,11.8,0.0029
3,1024,f32,1000,1249.0,6.2,0.0125
```

Byte-identical to the M2.5 final-review fix wave's pasted `count` output
above (this run being deterministic is itself the confirmation the M8
mechanism is a pure function of the dataset/tree, not a timing artifact).

#### The new noise floor (conclusion (e)): replacing the eyeballed 8-run range with an n=100-grounded characterization

**The old floor**, kept here as historical, not deleted: `knn_fixed3`
ratio over 8 independent runs, each itself a median-of-7 (`"M2.5 task 1"`
subsection above): **0.956–1.192** (median 1.064) — a bare min–max over 8
opaque single numbers, no `std` ever computed because the underlying
per-run sample (7 reps) was never itself published, only its median.

**The new floor.** The exact same conceptual workload (`ConstDim<3>`/f32
knn, n=100k, 10,000 queries/rep, k=10, leaf=10) was measured 4 independent
times this task — the two gate sessions above (dataset seeded
`"perf_gate_knn_fixed3"`) and the two report-chain runs above (dataset
seeded `"report_knn_fixed3"` — a genuinely different, independently-drawn
uniform dataset of the same shape, per this repo's `cfg_seed` convention,
§4) — each session `n=100` timed reps per side:

| session | ratio (medians) | rust mean±std ms (n=100) | cpp mean±std ms (n=100) | ratio of means | approx. per-rep ratio σ¹ |
|---|---|---|---|---|---|
| gate 1 | 1.032 | 7.737±0.172 | 7.479±0.165 | 1.0345 | 0.032 |
| gate 2 | 1.030 | 7.654±0.243 | 7.419±0.229 | 1.0317 | 0.046 |
| report 1 | 1.067 | 7.703±0.159 | 7.244±0.163 | 1.0634 | 0.033 |
| report 2 | 1.061 | 7.769±0.175 | 7.321±0.201 | 1.0611 | 0.038 |

¹ Delta-method (first-order error propagation) estimate of the standard
deviation of the PER-REPETITION ratio, `ratio_of_means * sqrt((std_rust /
mean_rust)^2 + (std_cpp / mean_cpp)^2)` — the same formula
`report_data.rs`'s `ratio_means_std` function uses, but applied to each
side's raw per-rep `std_ms` (single-repetition variability) rather than its
standard ERROR of the mean (`std_ms / sqrt(n)`, which is what
`ratio_means_std`/the JSON's `ratio_means_std` field actually reports — the
precision of the session's mean ESTIMATE, not the rep-to-rep spread; at
n=100 that SEM-based figure is tiny, ~0.003–0.007, and is NOT the "noise
floor" number this section is characterizing). This estimate assumes the
per-rep rust/cpp timings are independent; because `measure_pair` interleaves
them (rust then cpp, same rep, same instant — see the interleave-order note
above), any shared host-noise trend would induce positive correlation
between the two sides, which would partially CANCEL in the ratio rather
than add — so this delta-method figure is a plausible upper bound on the
true per-rep ratio σ, not an underestimate.

**Reading**: the four session medians span **1.030–1.067** (mean 1.0475,
session-to-session sample std 0.019, n=4) — already far tighter than the
old 8-run min–max, because "ratio of a median-of-100" is a much
less noisy summary statistic than "ratio of a median-of-7" was (exactly
the point of T1's rewrite). Within any one session, the per-repetition
ratio itself has an estimated σ of roughly **0.03–0.05** (table above) —
so a **mean ± 2σ band of approximately 0.94–1.16** comfortably contains
every session median observed here, sits well inside the 1.25 gate margin,
and nests inside (does not merely re-confirm) the old eyeballed
0.956–1.192 range: the new number is narrower on both ends and now has an
actual statistical basis (a computed σ from n=100 real samples) rather
than being read off 8 point values by eye.

**This is the new canonical noise floor.** Every "within the noise floor"
citation in this repo (`README.md`, `docs/benchmarks.md`,
`docs/ROADMAP.md`) is updated to point here; the old 8-run 0.956–1.192
figure (and the M1-era ±5-10%/0.937–1.079 characterizations before it) are
kept as historical text, marked superseded, never edited in place.

#### Conclusion-by-conclusion re-verification (M2.6 controller's binding list)

Every ratio cited below is drawn from the sessions pasted above in this
subsection (2 gate sessions + 2 report-chain runs, all fresh for this
task, all n=100 per side; `m25_diag` dim-32/64/16 figures are median-of-15,
2 sessions, per the harness-difference note above).

**(a) M2.5 T3 dim-32 f32 win.** Previously recorded (T4's `m25_diag`
median-of-7, 2 sessions, "M2.5 task 4" subsection above): 0.966–1.008 —
read as "parity-or-better," i.e. Rust never loses. **CORRECTED.** This
task's median-of-15, 2-session re-run: **0.986–1.062** — straddles parity,
including one session where C++ is measurably faster (1.062, +6.2%). The
headline mechanism and magnitude are unaffected and remain solid: the fix
still closes a ~1.42x C++ win down to essentially 1.0x (a >25-point ratio
improvement, confirmed again here), but the specific "0.966–1.008,
parity-or-better" framing is superseded — the honest current range is
0.986–1.062, "near parity, occasionally a few percent either way," not a
guaranteed Rust win at every measurement. Not a regression (no kernel code
changed since M2.5-T3); a wider, more honestly-sampled range from more
repetitions per session (15 vs. 7).

**(b) M2.5 T2 give-back A/B (radius ~0.83, dim8 f64 back to the pre-M2.5
band).** **CORRECTED**, in both directions:
- `radius_dim3_f32`: this task's 4 fresh n=100 sessions range **0.807–0.867**
  (gate sessions 0.807/0.808; report-chain sessions 0.866/0.867 — a
  genuinely different, independently-seeded dataset per session, see the
  noise-floor discussion above). "~0.83" undersells the spread: the low
  end (0.807) is BETTER than every previously-recorded T2/M-py figure
  (0.827–0.845), nearly back to the pre-M2.5 pasted top (0.792); the high
  end (0.867) is WORSE than the M-py-era top (0.845). The true current
  range is wider than any single "~0.83" point estimate suggested.
- `knn_dyn_dim8_f64_k10`: 4 fresh n=100 sessions range **0.929–0.937** (gate
  0.936/0.937; report-chain 0.929/0.929) — better (lower) than the ENTIRE
  pre-M2.5 pasted range's own best point (0.944–1.091 / 0.944–0.984
  excluding the flagged outlier). "Back inside the pre-M2.5 band" undersells
  this too: dim8 f64 now measures better than the pre-M2.5 band ever did,
  not merely inside it.

**(c) Fixed-dim-3 residual "~1–4%."** **CORRECTED.** 4 fresh n=100 sessions
(the noise-floor table above) range **1.030–1.067** — 3.0% to 6.7%. The
upper end (6.7%) exceeds the old "~1–4%" characterization. Still not a
regression and still comfortably inside the newly-characterized noise
floor (mean≈1.05, ±2σ≈0.94–1.16, above) — the wider honest range comes
from sampling 2 independently-seeded datasets × n=100 reps instead of 1
dataset's n=7-median point estimates, not from any code change.

**(d) Dim-16 curse-of-dimensionality attribution.** **CONFIRMED.** New
median-of-15 dim-16 CSV (2 sessions): f32 ratio 0.682/0.682 (identical),
f64 0.942/0.940 (near-identical) — rust_ms ~83.2–83.4ms vs. dim-8's
~3.53–3.56ms, a ~23.4–23.6x jump (f32). `frac_points_scanned` (deterministic,
re-run byte-identical) still jumps 0.0176 (dim8) → 0.4875 (dim16), a
27.7x increase — still sufficient on its own to explain the timing cliff,
same mechanism, same conclusion. Absolute `rust_ms` shifted from the old
76–88ms to 83.2–99.3ms (both sessions), inside ordinary WSL2 run-to-run
variation, not a regression or a change in mechanism.

**(e) The noise floor itself.** **CORRECTED** (replaced) — see the
dedicated subsection immediately above. New canonical floor: session
medians 1.030–1.067 (n=4 sessions), per-session per-rep ratio σ≈0.03–0.05
(delta-method, n=100/side), approximate mean±2σ band 0.94–1.16. Old
0.956–1.192 (8 runs, median-of-7 each) kept as historical, superseded.

**(f) `dyn_add` range "1.11–1.24."** **CONFIRMED, narrowed.** 4 fresh n=100
sessions range **1.112–1.149** (gate 1.141/1.149; report-chain 1.112/1.124)
— comfortably inside the old 1.106–1.237 span. The n=100-median data
suggests the true steady-state band is narrower (roughly 1.11–1.15) than
the old n=7-median range implied; the old top end (1.237) reads, in
hindsight, like small-sample noise from the less-robust median-of-7
methodology rather than evidence the true distribution reaches that high.
Every session here still passes the 1.25 gate margin with more room than
the old high-water mark suggested.

**(g) `build_100k` seq range "0.974–1.139," flagged as suspiciously wide.**
**CORRECTED/SETTLED** — genuinely dataset-dependent, not pure noise or
drift. This task's 4 fresh n=100 sessions cluster by WHICH dataset they
ran against, not randomly: both gate sessions (dataset seeded
`"perf_gate_build"`) land at **0.967/0.967** (Rust faster, std <0.01
between sessions); both report-chain sessions (dataset seeded
`"report_build_100k"` — a different, independently-drawn uniform dataset of
the identical shape) land at **1.037/1.037** (C++ faster, std <0.002
between sessions). Combined range: **0.967–1.039** — narrower than the old
0.974–1.139, AND internally structured: each dataset is individually very
stable session-to-session (well under 1% drift), but the two datasets
differ from each other by ~7 percentage points. This directly answers the
brief's question: the old wide spread was not primarily measurement noise
(a single stable workload wouldn't reproduce two tight, ~7pp-apart
clusters like this) and not temporal drift (both gate sessions and both
report sessions were captured minutes apart, immediately before/after each
other, yet each pair agrees with itself far more than across the
gate/report split). The most likely structural explanation is the
different independently-seeded dataset per harness (`cfg_seed`'s
`"perf_gate_build"` vs. `"report_build_100k"` tags produce different
uniform point clouds of the same size/shape) — plausible because kd-tree
build cost is not perfectly invariant to the specific point distribution
even under "the same" generator/shape, though this task did not fully
isolate dataset-seed from harness/process context (gate = a standalone
single-workload test binary; report_data = one process running 8 workloads
sequentially) as the exact cause, and does not claim to. Not chased
further here (out of scope), but no longer honestly describable as "just
noise" — flagged for a future task if the distinction matters.



- **Seeds — the `cfg_seed` scheme**: every xval test derives its data/query
  seed deterministically from the *configuration itself*, not a loop
  counter or a global RNG state, via `xval::cfg_seed(tag: &str, parts:
  &[usize]) -> u64` — an FNV-1a-style mixing hash over a string tag (e.g.
  `"knn_matrix_data"`) and a slice of config-describing integers (dim,
  metric tag, dataset kind, leaf size, ...). This means every distinct
  `(test-purpose, config)` pair gets its own reproducible, collision-
  resistant data/query stream, and — critically — adding a new matrix cell
  or reordering an existing loop never silently reseeds an unrelated
  existing case, since the seed depends on the *semantic* config values,
  not their position in a loop. Data generators (`xval::uniform`/
  `clustered`/`with_duplicates`/etc.) all consume a `u64` seed through
  `ChaCha8Rng::seed_from_u64` — a named, versioned PRNG algorithm, so a
  given seed reproduces the exact same bits forever, on any platform.
- **Adaptive-n timing (M2.6, current for gates/`report_data`)**: every perf
  gate and every `report_data` speed row uses `xval::measure_pair` — 2
  discarded warmup reps, then `n = clamp(10, 100, floor(budget_s*1000 /
  t_est_ms))` timed reps of BOTH sides (`budget_s = 30`/side; every
  workload in this repo hits the upper clamp, `n=100`), reporting
  `TimingStats` (mean/std(sample, `n-1`)/median/min/max/`n`) for both
  sides. **The per-repetition interleave order is fixed: `rust_fn()` then
  `cpp_fn()`, every rep, never alternated** — a residual order-bias risk
  this repo accepts and does not correct for (no alternating/randomized
  variant was implemented — see §3's "M2.6 task 2" subsection for the full
  discussion); the 1.25 gate margin and the published min/max spreads are
  the safety margin that would absorb such a bias if it exists, the same
  way they already absorb WSL2 scheduler/thermal noise. Gate PASS/FAIL and
  every doc ratio claim remain the ratio of MEDIANS, never means — median
  is deliberately chosen over mean specifically to resist the occasional
  high outlier a shared/virtualized host like WSL2 (§1) tends to produce
  (concretely illustrated at n=100 in §3's "M2.6 task 2" subsection: some
  workloads' per-rep `std`/`max` show severe single-rep outliers while
  their ratio-of-medians stays stable across sessions).
  **`crates/xval/examples/m25_diag.rs` still uses the older, unrelated
  `xval::timed_median_ms`/`xval::median_of` pair** (predates T1, not
  rewritten — see §3): one untimed warmup, then `RUNS` (bumped from 7 to
  15 in M2.6 task 2, still n>=10) timed reps via `std::time::Instant`,
  reporting only the median (no mean/std/min/max).
- **Zero-allocation query paths, both sides**: every timed query workload
  (gates, benches, `report_data`'s speed rows) allocates its out-buffers
  exactly once, outside the timed region, and reuses them across every
  query — `knn_search`/`radius_search` writing into caller-owned
  `&mut [Idx]`/`&mut [T]` slices or a caller-owned `Vec` on the Rust side,
  and `knn_into`/`radius_into` (the caller-buffer FFI methods, backed
  directly by the raw out-pointer writes) on the C++ side — never the
  allocating `knn()`/`radius()` convenience wrappers on either side. This
  is stated as the *methodology*, not a caveat: both sides are
  allocation-symmetric, so neither implementation is unfairly penalized
  for allocator overhead the other doesn't pay. (`report_data`'s emitted
  `meta.speed_methodology` field states this same fact verbatim, so it
  travels with every generated report.)
- **Tie-aware, kernel-routed ground-truth scoring**: `report_data`'s
  accuracy rows score both implementations against a brute-force linear
  scan (`xval::brute_force_knn_l2_f32`/`_f64`, or the live-set-restricted
  `_live` variants for the dynamic churned-forest row) whose *selection*
  is a dumb, tree-independent scan (tie-broken by ascending index, a
  total order independent of any tree's own tie rule), but whose *distance
  arithmetic* is **the library's own `L2::eval` kernel** — the exact same
  code path `KdTree::knn_search`'s leaf scan calls internally — not an
  independently-written summation. This is load-bearing, not a stylistic
  choice: a first attempt using a hand-rolled summation for ground truth
  made the bit-exact scorer fail on ~90% of totally-correct, non-tied
  uniform-data queries, purely from 1-ULP drift against the ad-hoc
  scanner's different summation order — despite Rust and C++ trees
  agreeing with EACH OTHER bit-exactly on every one of those same queries.
  Routing ground truth through the shared kernel closes that gap. On top
  of this, `score_exact_tie_aware_f32`/`_f64` (the actual accuracy metric
  reported) is **tie-aware**: it checks the k returned *distances*
  bit-match the ground truth's k smallest distances positionally, and that
  every returned index's recomputed true distance equals its reported
  distance — accepting any valid k-th-boundary tie resolution while still
  catching a wrong point, wrong distance, or a missed closer neighbor. A
  plain order-independent index-SET match (the older, still-available
  `QueryScore::exact`) is too strict on duplicate-heavy datasets, where
  multiple points can be genuinely tied at the k-th distance boundary and
  a candidate legitimately picking a different (equally valid) tie winner
  would otherwise score as wrong.
- **What "bit-exact" means, and where tie latitude applies**: the default
  comparator everywhere is positional, `max_ulps = 0` — indices and
  distances must match exactly, in the same order, between Rust and C++.
  The one legitimate exception is index *order* among a group of results
  that are bit-equal in distance (never across a genuine distance
  difference): C++'s `std::sort` (used by radius search's optional sort)
  is unstable, while this port's radius-result sort is Rust's stable sort
  — a documented, deliberate deviation (README's "Deliberate deviations"
  table) — so equal-distance order is the one place output order may
  legally differ. `ties = true` comparators relax exactly that, and only
  that.
- **The dynamic op-sequence legality model**: `xval::dyn_ops(seed,
  total_capacity, n_ops)` generates a sequence of `GrowAndAdd{count}`/
  `Remove{live_idx}`/`ReAdd{removed_idx}` ops (weights 50/30/20,
  renormalized each step over whichever kinds are currently legal given
  the sequence generated so far — e.g. `Remove`/`ReAdd` are excluded from
  the very first op, since nothing exists yet to remove or reactivate).
  This generator never emits an op that would violate `add_points`'s
  contiguity contract (README's "Dynamic adaptor (M2)" section) or a
  remove/reactivate on a point not in the right state, by construction.
  That claim is not just trusted: `xval::validate_dyn_ops_legal` is an
  *independent*, from-scratch reimplementation of the same legality rules
  (it does not call back into `dyn_ops`'s internal state), run as its own
  property test over many seeds — so a bug shared between generation and
  validation logic could not hide behind both agreeing with each other.
  Every hand-scripted scenario sequence (tombstone migration,
  drain-and-refill, the eps-parity case, both mutation-canary setups) is
  also validated through this same independent checker before use, not
  just the randomly-generated matrix sequences.

## 5. Number provenance

Every figure that appears in `README.md` or `docs/benchmarks.md` traces to
one of the commands in §3 and one field/line in its output, as follows. The
four M1-era single-value gate figures below (`build_100k` 1.018,
`knn_fixed3` 1.042, `knn_dyn_dim8_f64_k10` 0.973, `radius` 0.767) and the
original `dyn_knn_after_churn` range (0.971–0.976) are **pre-M2.5 recorded
values**, kept as inputs to the "Pre-M2.5 pasted range" row further down
(the "Milestone-start" column of `docs/benchmarks.md`'s M2.5 summary table
is a min–max over every pasted pre-M2.5 run, not these single values) —
they are **superseded** as current-state figures by the M2.5 rows
(`README.md` and `docs/benchmarks.md`'s live "Benchmarks"/"M2.5" sections
cite the M2.5 ranges, not these).

| Claim (where it appears) | Regenerating command | JSON field / gate line |
|---|---|---|
| `build_100k_dim3_f32_seq` ratio 1.018 (pre-M2.5 recorded value; input to the "Pre-M2.5 pasted range" row below) | Perf gate command, §3 | `PERF_GATE perf_gate_build_100k_dim3_f32_seq: ... ratio=...` |
| `knn_fixed3_dim3_f32_k10` ratio 1.042 (pre-M2.5 recorded value; input to the "Pre-M2.5 pasted range" row below) | Perf gate command, §3 | `PERF_GATE perf_gate_knn_dim3_f32_k10: ... ratio=...` |
| `knn_dyn_dim8_f64_k10` ratio 0.973 (pre-M2.5 recorded value; input to the "Pre-M2.5 pasted range" row below) | Perf gate command, §3 | `PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: ... ratio=...` |
| `radius_dim3_f32` ratio 0.767 (pre-M2.5 recorded value; input to the "Pre-M2.5 pasted range" row below) | Perf gate command, §3 | `PERF_GATE perf_gate_radius_dim3_f32: ... ratio=...` |
| `dyn_add_20k_dim3_f32` range 1.106–1.237 (still current, unaffected by M2.5, README "Dynamic adaptor (M2)", `docs/benchmarks.md`) | Perf gate command with `perf_gate_dyn` filter, §3 | `PERF_GATE perf_gate_dyn_add_20k_dim3_f32: ... ratio=...` |
| `dyn_knn_after_churn_dim3_f32` range 0.971–0.976 (pre-M2.5 recorded value; input to the "Pre-M2.5 pasted range" row below) | Perf gate command with `perf_gate_dyn` filter, §3, "M2 final-review fix wave" subsection | `PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: ... ratio=...` |
| **M2.5 final six-gate ranges** (`build_100k` 0.974–1.079, `knn_fixed3` 1.011–1.040, `dim8` 0.950–0.966, `radius` 0.827–0.828, `dyn_add` 1.121–1.170, `dyn_knn_after_churn` 0.873–0.916 — README "Perf gate" table, `docs/benchmarks.md` M2.5 summary table's "Post-T2/final" column) | Perf gate command, §3, "M2.5 task 4" subsection | `PERF_GATE perf_gate_*: ... ratio=...`, both pasted runs |
| **M2.5 dim-32/64 final ranges** (dim32 f32 0.966–1.008, dim32 f64 1.103–1.111, dim64 f32 1.162–1.171, dim64 f64 0.879–0.887 — README "dim-32/64 knn" table, `docs/benchmarks.md` M2.5 dim-32/64 table) | `cargo run -p xval --release --example m25_diag -- knn`, §3, "M2.5 task 4" subsection | stdout CSV, `dim,scalar,rust_ms,cpp_ms,ratio` rows for dim 32/64 |
| **Pre-M2.5 pasted range** — the "Milestone-start" column of `docs/benchmarks.md`'s M2.5 summary table (`build_100k` 0.986–1.018, `knn_fixed3` 1.030–1.063, `dim8` 0.944–1.091 / 0.944–0.984 excluding the flagged outlier, `radius` 0.767–0.792, `dyn_add` 1.106–1.237, `dyn_knn_after_churn` 0.956–0.976) | Perf gate command, §3: the four pre-M2.5 six-gate runs (Re-run 1/2, original round and M2 fix wave) and the `perf_gate_dyn` runs, `docs/benchmarks.md`'s M1 table, and the "before" arm in the "M2.5 task 3" subsection; range definition stated in the "M2.5 task 4" subsection | `PERF_GATE perf_gate_*: ... ratio=...` lines, min–max |
| **Post-T3 column** of `docs/benchmarks.md`'s M2.5 summary table (`build` 0.971, `knn_fixed3` 1.054, `dim8` 0.867, `radius` 0.777, `dyn_add` 1.123, `dyn_knn_after_churn` 0.974) and the dim-32/64 table's "T3 landed" column (0.966 / 1.113 / 1.228 / 0.874) | §3, "M2.5 task 3" subsection (interleaved A/B "after" medians, 3 runs each); source `docs/reports/m2.5/task-3-report.md` §4a, §4b, §8.2 | the "after median" column of each pasted table |
| T2 give-back A/B (`radius` +4.4–7.4%, `dim8` +6.4–6.8% raw ms, direction 15/16 and 16/16) and the capacity-32 non-recovery (NEW32 ≈ NEW) — `docs/benchmarks.md` "The T2 trade-off" and "Robustness upgrade", `docs/ROADMAP.md` radius watch-item | §3, "M2.5 task 2" subsection; source `docs/reports/m2.5/task-2-report.md` Fix round 1, Item 2 | the two rust_ms tables' median rows |
| `unsafe` necessity: safe `Vec` frame stack slower in 8/8 reps, median +4.7% (README "Safety") | §3, "M2.5 task 2" subsection; source `docs/reports/m2.5/task-2-report.md` Fix round 1, Item 5 | rust_ms table, median row 7.064 vs 7.397 |
| Fixed-dim-3 A/B medians 1.052 → 1.0205 (`docs/ROADMAP.md`, `docs/nanoflann-notes.md`, `docs/benchmarks.md` M2.5 trade-off paragraph) | §3, "M2.5 task 2" subsection; source `docs/reports/m2.5/task-2-report.md` §6b | ratio table, median row |
| **Noise floor** `knn_fixed3` 0.956–1.192 (median 1.064) over 8 runs — README "Perf gate" footnote, `docs/benchmarks.md` "Honest residuals", `docs/ROADMAP.md` | §3, "M2.5 task 1" subsection; source `docs/reports/m2.5/task-1-report.md` §5 | the eight listed ratios |
| Kernel bit-exactness guard actually discriminates (multi-salt RED: `21 passed; 6 failed` and `23 passed; 4 failed` under the two sabotages) — `docs/benchmarks.md` dim-32/64 mechanism paragraph and "Honest residuals" | §3, "M2.5 task 3" subsection; source `docs/reports/m2.5/task-3-report.md` §8.1 | pasted `test result: FAILED.` lines |
| Zero `search_level` call targets in compiled asm; `probe_knn3` body 569 lines, only cold `grow_one` + panic pads as calls (README "Perf gate" paragraph, `docs/benchmarks.md` "Update (M2.5-T2)" banner) | `cargo rustc -p xval --release --example m25_asm -- --emit asm` + greps, §3, "M2.5 asm re-capture" subsection | pasted `grep -c` outputs (0, 0) and the `uniq -c` call list |
| M2.5 test-status counts (359 passed / 0 failed / 15 ignored; `--no-default-features` 179/0/3; heavy suite 0 failed, 151.80s) | Full test suite + heavy suite + `--no-default-features` commands, §3, "M2.5 task 4" subsection | terminal `test result:` lines |
| Miri clean (Stacked Borrows + Tree Borrows, re-run for M2.5 task 4) | Miri commands, §3 | terminal `test result: ok. 34 passed; 0 failed; 1 ignored` for each aliasing model |
| Miri re-capture, final-review fix wave (real host wall-clock 2:34.36 vs 4:26.15, `test result:` line's `488.12s` identical both times — clock-isolation artifact, not a stale/reused number) | Miri commands each wrapped in `time`, §3, "M2.5 final-review fix wave" subsection | terminal `time` output + `test result:` line, both commands |
| dim-16 knn CSV rows (rust_ms/cpp_ms/ratio, f32+f64 — previously cited only as a range, never pasted) | `cargo run -p xval --release --example m25_diag -- knn`, §3, "M2.5 final-review fix wave" subsection | stdout CSV, `dim,scalar,rust_ms,cpp_ms,ratio` rows for dim 16 |
| M8 curse-of-dimensionality confirmation (`frac_points_scanned` 0.0176 dim8 → 0.4875 dim16 → 1.0000 dim32) | `cargo run -p xval --release --example m25_diag -- count`, §3, "M2.5 final-review fix wave" subsection | stdout CSV, `dim,leaf,scalar,queries,eval_per_query,interior_per_query,frac_points_scanned` rows |
| Churned-forest accuracy row (1.0/1.0/bit-exact at eps=0) | Report chain, §3 | `report.json`'s `accuracy[].workload == "dyn_churn_dim3_f32_k10_eps0"`: `rust_exact_tie_aware_vs_bruteforce`, `cpp_exact_tie_aware_vs_bruteforce`, `rust_eq_cpp_bitexact` |
| Static accuracy rows (uniform/with_duplicates, eps ∈ {0, 0.1, 1}) | Report chain, §3 | `report.json`'s `accuracy[]` array, one row per `{dataset}_dim{d}_f32_k10_eps{e}` workload |
| `knn_fixed3` headline table (k=1/10/100, f32/f64) | Full criterion sweep (§3; smoke form shown covers `k10/f32`) | `bench_knn.rs`'s `knn_fixed3/{lib}/{scalar}/k{k}` groups |
| `build_fixed3` headline table (n=100k/1M) | Full criterion sweep | `bench_build.rs`'s `build_fixed3/{lib}/{n}` groups |
| dim-8/dim-32 knn headline table | Full criterion sweep | `bench_knn.rs`'s `knn/{lib}/{dim}/{scalar}/k{k}` groups |
| `leaf_max_size` sweep table | Full criterion sweep (§3; smoke form shown covers `leaf=10/1024`) | `bench_build.rs`'s `leaf_sweep/{lib}/{leaf}/{phase}` groups |
| Parallel build 1.75× win (M1), `build_1M_dim3_f32_par` (report tile) | Report chain, §3 | `report.json`'s `speed[].workload == "build_1M_dim3_f32_par"`: `ratio` |
| `dyn_add`/`dyn_churn`/`dyn_knn_after_churn` criterion numbers | Full criterion sweep (§3; smoke form shown covers `dyn_add/rust`) | `bench_dynamic.rs`'s three groups |
| Test-status counts (343 passed at M2 initial completion, superseded by 345 at M2's final-review fix wave, superseded by **359** at M2.5 task 4 — current figure) | Full test suite, §3, "M2.5 task 4" subsection | terminal `test result:` line summed across all binaries — see `docs/benchmarks.md`'s "Test status (workspace, current)" section |
| Heavy-suite pass (0 failed, incl. depth ~2115/~16794 builds and mutation canaries) | Heavy/`--ignored` suite, §3 | terminal `test result:` line per binary |
| CPU model / kernel / compiler / git SHA / WSL flag (this doc's §1 table) | Report chain, §3, or `report_data` alone | `report.json`'s `meta.{cpu_model,kernel,cxx_compiler,git_sha,wsl}` |
| `--bad`/`--good` verdict tiles, HTML scorecard | Report chain, §3 | `report.html`'s `<div class="tiles">` block, computed live from `report.json` — never hardcoded (see `crates/xval/tests/render_report_test.rs` for the test suite proving this) |
| M-py parity scope (96/96 tie-free knn index-sequence nodes, 4/4 dim-2 distance bit-exactness, 16 nodes pinned `xfail(strict=True)`, combined suite `0 failed, 244 passed, 27 xfailed, 1 xpassed`) — README Python section, `docs/benchmarks.md` M-py section | `.venv/bin/pytest python/tests/test_parity_pynanoflann.py -q` and `python/tests -q`, §3 "M-py" subsection | pasted `pytest` summary lines; source `task-3-report.md` "Escalation"/"Parity suite" sections and `test_parity_pynanoflann.py`'s module docstring |
| M-py perf-gate re-run (T6, confirms T0–T5 didn't move Rust-side perf; honest range widening on `build_100k` 0.974–1.139 and `radius` 0.827–0.845) | Perf gate command, §3 "M-py" subsection | `PERF_GATE perf_gate_*: ... ratio=...` lines, this task's single run |
| M-py bench success criteria (batched knn dim3 vs cKDTree 0.712–0.830; build vs pynanoflann 0.543–0.561/0.543–0.551; dim32 vs pynanoflann 0.878–0.899; per-call overhead ≈2.0–2.3µs) — README Python section, `docs/benchmarks.md` M-py section, `docs/ROADMAP.md` M-py entry | `bench_py.py`, §3 "M-py" subsection, both pasted runs | `report_py.json`'s `workloads[]`, `ratio_ckdtree`/`ratio_pynanoflann`/`*_ms` fields |
| M-py honest misses (`knn_batched_..._workers1` vs pynanoflann 1.058–1.216; `knn_dim8_float64_..._workers1` vs pynanoflann 1.233–1.249) — README Python section, `docs/benchmarks.md` M-py section, `docs/ROADMAP.md` M-pub dim8 lead | `bench_py.py`, §3 "M-py" subsection, both pasted runs | same `report_py.json` fields as the row above |
| **M2.6 statistical baseline, six-gate ranges** (`build_100k` 0.967–1.039, `knn_fixed3` 1.030–1.067, `dim8` 0.929–0.937, `radius` 0.807–0.867, `dyn_add` 1.112–1.149, `dyn_knn_after_churn` 0.940–0.947 — combined across 2 gate sessions + 2 report-chain runs, n=100/side each) — README "Perf gate" table, `docs/benchmarks.md` M2.6 section | Perf gate command + report chain, §3 "M2.6 task 2" subsection | `PERF_GATE perf_gate_*: ... ratio=...` and `report.json`'s `speed[].ratio`, all 4 pasted sessions |
| **New canonical noise floor** (`knn_fixed3` session medians 1.030–1.067 across n=4 sessions, per-session per-rep ratio σ≈0.03–0.05 at n=100/side, approx. mean±2σ band 0.94–1.16 — supersedes the old 8-run 0.956–1.192 everywhere it was cited: `README.md`, `docs/benchmarks.md`, `docs/ROADMAP.md`) | Perf gate command + report chain, §3 "M2.6 task 2" subsection, "The new noise floor" sub-subsection | `PERF_GATE`/`report.json` `TimingStats` `mean_ms`/`std_ms`/`n` fields for `knn_dim3_f32_k10`/`knn_fixed3`, all 4 pasted sessions |
| **M2.6 dim-32 f32 re-verification** (0.986–1.062, straddles parity — corrects the previously-cited "parity-or-better" 0.966–1.008) — README "dim-32/64 knn" section, `docs/benchmarks.md` M2.6 section | `cargo run -p xval --release --example m25_diag -- knn`, §3 "M2.6 task 2" subsection | stdout CSV, `dim,scalar,rust_ms,cpp_ms,ratio` dim-32 f32 rows, both sessions |
| **M2.6 T2 give-back re-verification** (`radius` 0.807–0.867, wider than the previously-cited "~0.83"; `knn_dyn_dim8_f64_k10` 0.929–0.937, better than the entire pre-M2.5 band, not merely inside it) — `docs/benchmarks.md`/`docs/ROADMAP.md` M2.5 trade-off text | Perf gate command + report chain, §3 "M2.6 task 2" subsection | same `ratio` fields as the six-gate-ranges row above |
| **M2.6 `build_100k` dataset-dependence finding** (0.967/0.967 on the gate's own dataset vs. 1.037/1.037 on report_data's differently-seeded dataset — settles the old "0.974–1.139, suspiciously wide" question as dataset-structured, not pure noise or drift) | Perf gate command + report chain, §3 "M2.6 task 2" subsection, conclusion (g) | `PERF_GATE perf_gate_build_100k_dim3_f32_seq`/`report.json`'s `build_100k_dim3_f32_seq` `ratio`, all 4 pasted sessions |
| `m25_diag` `RUNS` bump 7→15 (n>=10, M2.6 task 2's only code change) | `git diff crates/xval/examples/m25_diag.rs` (12 lines, doc comment + one constant) | the file itself; re-run commands unchanged |

## 6. See also

- [`README.md`](../README.md) — "Reproducing our numbers" pointer (Benchmarks
  section) links back here; "Dynamic adaptor (M2)" and "Parity & testing"
  sections describe what each dynamic figure means.
- [`docs/benchmarks.md`](benchmarks.md) — the actual recorded numbers (M1
  static, M2 dynamic, and M2.5 performance deep-dive sections) this
  document's commands regenerate.
- [`docs/nanoflann-notes.md`](nanoflann-notes.md) — verified C++ source
  facts backing the parity claims this document's methodology section
  refers to.
- [`docs/ROADMAP.md`](ROADMAP.md) — the standing requirement this document
  exists to satisfy, and the M-pub bare-metal re-run this document's §1
  caveat points to.
