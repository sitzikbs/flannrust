# EXPERIMENTS.md — reproducing nanoflann-rs's numbers

Every performance and accuracy figure in this repo's `README.md` and
`docs/benchmarks.md` traces to a command listed in this file. This
document exists per the project's standing requirement (`docs/ROADMAP.md`):
*"Every number that could appear in an announcement traces to a committed
artifact (JSON + generator + doc), never to a chat log."* Every command
below was actually run at least once while writing this document — see
each section's captured output.

## 1. Environment

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
| git SHA at time of writing | `b90fa57` (branch `m2-dynamic` HEAD immediately before this task's two commits) | `git rev-parse --short HEAD` + `git status --porcelain` dirty check (`git_sha()`) |

### WSL2 measurement-noise caveat (read every number in this repo with this in mind)

**Every number in this repo's README, `docs/benchmarks.md`, and this file
was measured inside WSL2** (Windows Subsystem for Linux 2), a virtualized
environment sharing the host's scheduler with the Windows side. WSL2
introduces real run-to-run scheduler/thermal jitter — M1's own measurement
found roughly **±5-10% on individual runs**, and re-running individual perf
gates during this task showed ratios drifting by several points between
back-to-back runs on an otherwise-idle machine (e.g. `dyn_add_20k_dim3_f32`:
1.113 at commit time, 1.106 and 1.156 on two re-runs minutes apart; see
`docs/benchmarks.md`'s "M2 — dynamic forest" section). The perf gates' 1.25
margin and the median-of-7 methodology (below) both exist specifically to
absorb this noise, not to paper over a real regression — but it is still
noise, not a bare-metal-quality measurement.

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
below was run during this documentation task; output is summarized inline,
full transcripts are in this task's report.

### Full test suite

```
cargo test --workspace --all-features
```
Expected runtime: a few seconds (debug build, no heavy/perf-gated tests —
those are `#[ignore]`d). **Re-run for this document: 343 passed, 0 failed,
15 ignored**, including doctests (2, both in `nanoflann-rs`: the crate-doc
example and `DynamicKdTree`'s doc example). This supersedes M1's own
269-passed/8-ignored count recorded at M1 completion (see
`docs/benchmarks.md`'s M1 "Test status" subsection, now marked superseded)
— M2 added `nanoflann-ref`'s dynamic-oracle tests, `xval`'s dynamic
cross-validation suite, the report renderer's own test file, and the two
new dynamic perf gates, all counted in the new total.

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
a sibling test's load. **Re-run for this document** (all six, one
invocation):
```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=9.128ms cpp=9.254ms ratio=0.986
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.808ms cpp=3.444ms ratio=1.106
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=30.418ms cpp=32.185ms ratio=0.945
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.220ms cpp=6.968ms ratio=1.036
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=195.461ms cpp=179.159ms ratio=1.091
PERF_GATE perf_gate_radius_dim3_f32: rust=4.750ms cpp=5.995ms ratio=0.792
test result: ok. 6 passed; 0 failed
```
All six pass the 1.25 margin. Comparing against `docs/benchmarks.md`'s
recorded figures (build 1.018, `knn_fixed3` 1.042, `knn_dyn_dim8_f64_k10`
0.973, radius 0.767, `dyn_add` 1.113, `dyn_knn_after_churn` 0.964): most
re-run ratios above sit within a few points, consistent with ordinary WSL2
noise (§1) — except `knn_dyn_dim8_f64_k10`, which swung further this run
(0.973 recorded vs. **1.091** here, ~12%). Still comfortably inside the
1.25 gate margin and not a regression (no code changed between the
recorded run and this one — this task is documentation-only), but it's a
larger single-run swing than the "~±5-10%" figure quoted above, and is
itself a live illustration of exactly why the WSL2 caveat exists and why
the gate margin is 1.25, not something tighter.

To run only the two dynamic gates: append `_dyn` to the filter
(`--ignored perf_gate_dyn`).

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
un-truncated sweep run during M1's Task 14 / M2's Task 5 (see each task's
own report for the captured full-sweep transcripts), not from these
smoke commands — the smoke commands exist only to prove the *commands*
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
string/template work over an already-computed JSON). **Re-run for this
document**: both exited 0; `report.json` (41 lines of pretty JSON fields,
~8KB compact) and `report.html` (~16KB) were regenerated; verdict tiles
computed live from the data read:

```
Accuracy @ eps=0:  ✓ 100% exact @ eps=0
Bit-exactness:     ✓ Bit-exact vs C++ (all rows)
Best speed win:    ✓ 1.46× faster — build_1M_dim3_f32_par
```

`report_data`'s own emitted `meta` block is this run's environment
fingerprint (see §1's table — those values were read directly from this
exact run's JSON).

## 4. Methodology

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
- **Median-of-7 timing**: every perf gate and every `report_data` speed row
  uses the same `xval::timed_median_ms`/`xval::median_of` pair — run the
  timed closure once, untimed, as a warmup; then run it `RUNS = 7` times,
  timed via `std::time::Instant`; report the median (not mean) of those 7
  wall-clock samples. Median is deliberately chosen over mean specifically
  to resist the occasional high outlier a shared/virtualized host like
  WSL2 (§1) tends to produce.
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
one of the commands in §3 and one field/line in its output, as follows.

| Claim (where it appears) | Regenerating command | JSON field / gate line |
|---|---|---|
| `build_100k_dim3_f32_seq` ratio 1.018 (README, `docs/benchmarks.md`) | Perf gate command, §3 | `PERF_GATE perf_gate_build_100k_dim3_f32_seq: ... ratio=...` |
| `knn_fixed3_dim3_f32_k10` ratio 1.042 | Perf gate command, §3 | `PERF_GATE perf_gate_knn_dim3_f32_k10: ... ratio=...` |
| `knn_dyn_dim8_f64_k10` ratio 0.973 | Perf gate command, §3 | `PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: ... ratio=...` |
| `radius_dim3_f32` ratio 0.767 | Perf gate command, §3 | `PERF_GATE perf_gate_radius_dim3_f32: ... ratio=...` |
| `dyn_add_20k_dim3_f32` ratio 1.113 (README "Dynamic adaptor (M2)", `docs/benchmarks.md`) | Perf gate command with `perf_gate_dyn` filter, §3 | `PERF_GATE perf_gate_dyn_add_20k_dim3_f32: ... ratio=...` |
| `dyn_knn_after_churn_dim3_f32` ratio 0.964 | Perf gate command with `perf_gate_dyn` filter, §3 | `PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: ... ratio=...` |
| Churned-forest accuracy row (1.0/1.0/bit-exact at eps=0) | Report chain, §3 | `report.json`'s `accuracy[].workload == "dyn_churn_dim3_f32_k10_eps0"`: `rust_exact_tie_aware_vs_bruteforce`, `cpp_exact_tie_aware_vs_bruteforce`, `rust_eq_cpp_bitexact` |
| Static accuracy rows (uniform/with_duplicates, eps ∈ {0, 0.1, 1}) | Report chain, §3 | `report.json`'s `accuracy[]` array, one row per `{dataset}_dim{d}_f32_k10_eps{e}` workload |
| `knn_fixed3` headline table (k=1/10/100, f32/f64) | Full criterion sweep (§3; smoke form shown covers `k10/f32`) | `bench_knn.rs`'s `knn_fixed3/{lib}/{scalar}/k{k}` groups |
| `build_fixed3` headline table (n=100k/1M) | Full criterion sweep | `bench_build.rs`'s `build_fixed3/{lib}/{n}` groups |
| dim-8/dim-32 knn headline table | Full criterion sweep | `bench_knn.rs`'s `knn/{lib}/{dim}/{scalar}/k{k}` groups |
| `leaf_max_size` sweep table | Full criterion sweep (§3; smoke form shown covers `leaf=10/1024`) | `bench_build.rs`'s `leaf_sweep/{lib}/{leaf}/{phase}` groups |
| Parallel build 1.75× win (M1), `build_1M_dim3_f32_par` (report tile) | Report chain, §3 | `report.json`'s `speed[].workload == "build_1M_dim3_f32_par"`: `ratio` |
| `dyn_add`/`dyn_churn`/`dyn_knn_after_churn` criterion numbers | Full criterion sweep (§3; smoke form shown covers `dyn_add/rust`) | `bench_dynamic.rs`'s three groups |
| Test-status counts (343 passed / 0 failed / 15 ignored) | Full test suite, §3 | terminal `test result:` line summed across all binaries — see `docs/benchmarks.md`'s "Test status (workspace, current)" section |
| Heavy-suite pass (0 failed, incl. depth ~2115/~16794 builds and mutation canaries) | Heavy/`--ignored` suite, §3 | terminal `test result:` line per binary |
| CPU model / kernel / compiler / git SHA / WSL flag (this doc's §1 table) | Report chain, §3, or `report_data` alone | `report.json`'s `meta.{cpu_model,kernel,cxx_compiler,git_sha,wsl}` |
| `--bad`/`--good` verdict tiles, HTML scorecard | Report chain, §3 | `report.html`'s `<div class="tiles">` block, computed live from `report.json` — never hardcoded (see M2 Task 5b's report for the `render_report` test suite proving this) |

## 6. See also

- [`README.md`](../README.md) — "Reproducing our numbers" pointer (Benchmarks
  section) links back here; "Dynamic adaptor (M2)" and "Parity & testing"
  sections describe what each dynamic figure means.
- [`docs/benchmarks.md`](benchmarks.md) — the actual recorded numbers (M1
  static + M2 dynamic sections) this document's commands regenerate.
- [`docs/nanoflann-notes.md`](nanoflann-notes.md) — verified C++ source
  facts backing the parity claims this document's methodology section
  refers to.
- [`docs/ROADMAP.md`](ROADMAP.md) — the standing requirement this document
  exists to satisfy, and the M-pub bare-metal re-run this document's §1
  caveat points to.
