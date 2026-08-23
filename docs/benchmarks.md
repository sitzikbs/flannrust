# nanoflann-rs benchmarks

Combined benchmark record for the crate: M1's static kd-tree pass and M2's
dynamic (Bentley–Saxe forest) pass. This is a straight relocation of M1's
original `docs/benchmarks-m1.md` (renamed here) plus M2's new dynamic
numbers underneath — the M1 section's text is unchanged content, only its
heading levels were demoted so it nests under this file's own title. See
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md) for the full reproducible
experimental setup (exact commands, environment, methodology,
number-provenance table) backing every figure in both sections below.

## M1 — static kd-tree (Task 14 — performance pass)

Machine: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), AMD Ryzen 7 9800X3D,
8 threads visible, `rustc 1.98.0`. All numbers `RUSTFLAGS="-C
target-cpu=native"`, release profile. C++ oracle built `-O3 -march=native
-ffp-contract=off` (`crates/nanoflann-ref/build.rs`). WSL2 introduces real
run-to-run scheduler/thermal jitter (~±5-10% on individual runs). The perf
gate table below mixes sample sizes across its columns (see the column
headers): the "Before Task 14" column is Step 0's baseline, a **median of
3** timed runs; every "After Task 14" column is a **median of 7** timed runs
(perf-gate methodology throughout: `PERF_GATE=1`, `--test-threads=1`, warmup
pass + median-of-N). The larger post-change sample size doesn't change the
before/after deltas' direction, but the two columns are not directly
apples-to-apples on sample count — flagged explicitly rather than
re-measuring the baseline at N=7, since the pre-change code no longer needs
re-building for this report. Numbers elsewhere are marked "single run" or
"criterion (spot check)" where neither median-of-N methodology applies.

Ratio = rust_ms / cpp_ms; lower is better for Rust; **the milestone success
criterion (dim-3 knn/build) is ratio ≤ 1.0 (Rust ≥ C++), perf-gate pass is
ratio ≤ 1.25.**

### Perf gate (four gated workloads)

| Workload | Before Task 14 (median×3, Step 0 baseline) | After Task 14 (median×7) | Final single run |
|---|---|---|---|
| `build_100k_dim3_f32_seq` | 1.040 | 1.018 | 0.995 |
| `knn_fixed3_dim3_f32_k10` | 1.051 | 1.042 | 1.047 |
| `knn_dyn_dim8_f64_k10` | 0.988 | 0.973 | 0.984 |
| `radius_dim3_f32` | 0.773 | 0.767 | 0.774 |

All four gates pass the 1.25 margin by a wide margin, before and after. The
milestone's stricter bar — dim-3 f32 knn and build ratios ≤ 1.0 — is now
**met for build** (0.995-1.018, effectively at parity) and **nearly met for
knn_fixed3** (1.04-1.05, down from 1.05, a real but partial close — see
"Remaining gap analysis" below).

#### Radius marshalling asymmetry (read `radius_dim3_f32`'s < 1.0 ratio with this in mind)

`radius_dim3_f32`'s ratio (0.767-0.774, Rust *faster* than C++) is genuine,
but the two sides pay a different amount of FFI-wrapper bookkeeping inside
the timed region, and that asymmetry favors Rust — worth reading the number
with this in mind rather than as a pure "faster algorithm" claim.
`nanoflann_ref::RefIndexF32::radius_into` (the C++-side benchmark call,
`crates/nanoflann-ref/src/lib.rs`) uses a two-call `_count`/`_fetch`
protocol: `_count` runs nanoflann's real `radiusSearch` into an
internal `std::vector<ResultItem<uint32_t, T>>` scratch buffer (AoS:
index+distance interleaved), then `_fetch` (`nfr_radius_fetch_impl`,
`crates/nanoflann-ref/cpp/wrapper.cpp`) walks that scratch buffer
element-by-element, splitting each `ResultItem` into the separate
`out_idx[i]`/`out_dist[i]` SoA arrays the Rust FFI caller expects — a plain
loop of `2 * found` scalar stores that a native C++ caller of nanoflann
would never pay (they'd just keep using the AoS `std::vector` nanoflann
itself returns). This split copy runs INSIDE the timed C++ call in the
benchmark. Rust's `radius_search`/`RadiusResultSet::add` has no analogous
step: it pushes directly into the caller's `Vec<ResultItem<Idx, D>>`, which
is already the AoS layout the query needs.

Bounding the likely magnitude: the benchmark's two selectivities return
~10 and ~1000 items per query respectively; at ~10 items the marshalling
loop is negligible next to the tree walk, but at ~1000 items (closer to
"~100 items/query" order of magnitude and up) a `~1000`-iteration scalar
copy loop is a real, if still probably small (low-single-digit-percent),
constant addition to the C++ side's measured time that has nothing to do
with `radiusSearch` itself. This doesn't invalidate Rust's win margin here,
but the margin should be read as "at least this good," not attributed
entirely to algorithmic difference — some of it is this crate's own FFI
wrapper design choice on the C++ side of the harness, not native C++
nanoflann's actual cost.

### Headline table (criterion, spot-check methodology, `--quick`)

#### `knn_fixed3` — `ConstDim<3>` + `&[[T;3]]` (rust) vs `RefIndex3F32`/`RefIndex3F64` (cpp)

| k | scalar | rust | cpp | ratio |
|---|---|---|---|---|
| 1 | f32 | 163.0 ns | 97.6 ns | 1.669 |
| 10 | f32 | 739.0 ns | 690.9 ns | 1.070 |
| 100 | f32 | 5.877 µs | 5.781 µs | 1.017 |
| 1 | f64 | 182.5 ns | 109.2 ns | 1.672 |
| 10 | f64 | 753.9 ns | 713.7 ns | 1.056 |
| 100 | f64 | 5.979 µs | 6.066 µs | 0.986 |

k=1's ~1.67x ratio is a **measurement artifact, not a regression**: at k=1
the whole query (traverse + one comparison) is ~100-180ns, so per-call fixed
overhead (function-call ABI, `black_box`, criterion's own iteration
harness) dominates the ~65ns absolute gap — the perf-gate's own k=10
methodology (10,000 queries accumulated per timed sample, not one
`criterion` iteration each) is deliberately structured to avoid exactly this
sensitivity, which is why it reports ~1.04-1.05 instead. k=10/100 (larger,
more representative workloads) sit at 0.99-1.07 — close to parity.

#### `build_fixed3` — `ConstDim<3>` + `&[[f32;3]]` (rust) vs `RefIndex3F32::build` (cpp)

| n | rust | cpp | ratio |
|---|---|---|---|
| 100,000 | 8.916 ms | 8.865 ms | 1.006 |
| 1,000,000 | 115.74 ms | 113.55 ms | 1.019 |

Both essentially at parity, consistent with the perf-gate's build ratio.

#### `knn` — runtime `DynDim` + `FlatSlice` (rust) vs `DIM=-1` (cpp), dim 8 and dim 32

| dim | k | scalar | rust | cpp | ratio |
|---|---|---|---|---|---|
| 8 | 1 | f32 | 3.920 µs | 3.979 µs | 0.985 |
| 8 | 10 | f32 | 16.84 µs | 17.20 µs | 0.979 |
| 8 | 100 | f32 | 61.26 µs | 61.94 µs | 0.989 |
| 8 | 1 | f64 | 4.011 µs | 4.158 µs | 0.965 |
| 8 | 10 | f64 | 18.71 µs | 18.64 µs | 1.004 |
| 8 | 100 | f64 | 65.54 µs | 66.21 µs | 0.990 |
| 32 | 1 | f32 | 1.492 ms | 1.053 ms | 1.418 |
| 32 | 10 | f32 | 1.479 ms | 1.068 ms | 1.385 |
| 32 | 100 | f32 | 1.576 ms | 1.087 ms | 1.450 |
| 32 | 1 | f64 | 2.159 ms | 1.644 ms | 1.314 |
| 32 | 10 | f64 | 2.164 ms | 1.646 ms | 1.315 |
| 32 | 100 | f64 | 2.211 ms | 1.662 ms | 1.330 |

dim-8 is close to parity (0.97-1.00), consistent with the perf-gate's own
dim-8 workload. **dim-32 shows a real, consistent ~1.3-1.45x gap** — but this
gap **pre-dates Task 14**: verified by checking out the pre-Task-14 commit
(`4e2fec6`) and re-running the identical dim-32 benchmark, which reproduced
the same ~1.4-1.45x ratio (1.476ms rust / 1.018ms cpp at f32/k10, vs.
1.479ms/1.068ms after Task 14 — statistically the same). It is **not**
something this task's changes introduced or regressed, and dim-32 is outside
this task's success criterion ("dim-3 f32/f64 knn and build") and the
perf-gate's covered workloads (dim-3, dim-8 only). Documented here as a known
gap for a future task, not fixed in M1 per this task's explicit scope.

**Update (M2.5-T3):** closing this gap did NOT need SIMD/batching after all.
The root cause (diagnosed in M2.5-T1, fixed in M2.5-T3) was that
`L2::eval`'s per-component bounds checks split the unrolled body into 8
basic blocks, blocking LLVM's SLP vectorizer at runtime-known dim — gcc
compiles the *identical* summation order to AVX2/AVX-512 with no reordering,
proving the order itself was always vectorizable. A bounds-check-free
chunked row walk (`DataSource::point_row` + `as_chunks::<4>()`, bit-exact
with the fallback) closed dim-32 f32 from **1.423× → 0.966×** (gate-style
knn, leaf=10, measured this session) with zero SIMD intrinsics — see
`.superpowers/sdd/2026-08-23-nanoflann-rs-m2.5-perf/task-3-report.md`.
Dim-64 improved substantially but not fully (1.918× → 1.228×); further
dim-64-specific work is out of scope for M2.5-T3, which targeted dim-32.

### Remaining gap analysis: knn_fixed3 (~1.04-1.05x)

Asm inspection (`cargo rustc -p xval --release --example report_data --
--emit asm`, `RUSTFLAGS="-C target-cpu=native"`) of the monomorphized
`search_level::<f32, ConstDim<3>, &[[f32;3]], L2, u32, KnnResultSet, AcceptAll>`
leaf scan confirms the eval kernel itself is now optimal post-Task-14:

- **Before** (Task 14 baseline): a runtime `cmpq $3, %dim` branch selecting
  between the unrolled-loop and remainder paths, PLUS three separate
  `panic_bounds_check` call sites — one per `point_component(idx, d)` call
  (d=0,1,2) — for a single point's distance.
- **After** (post-Candidate-1, ConstDim threading): the `cmpq $3` branch is
  gone entirely (dead-code-eliminated once `d.dim()` constant-folds to `3`
  under `ConstDim<3>` monomorphization), and the three per-component bounds
  checks collapsed to exactly ONE (`idx < len`) via LLVM's own CSE across the
  now straight-line unrolled code — this happened automatically, without
  needing a `DataSource::point_row` fast path (returning a whole point at
  once instead of per-component `point_component` calls) at all — that fast
  path was tried as a candidate optimization, measured as a net performance
  loss once the CSE above already closed most of the gap, and reverted.

  **Update (M2.5-T3, reconciling this record with what actually shipped
  later):** the *bare* `point_row` tried here really was a net loss at
  dim 3/8, exactly as recorded above — but that prior A/B only tested a row
  pointer that was still indexed per-component (`row[d]`), which at
  runtime-known dim can't be bounds-check-hoisted any better than
  `point_component` was. M2.5-T3 re-measured `point_row` at dim 32/64 (never
  tried here) and productionized a different lever on top of it: a
  bounds-check-free **chunked** row walk (`as_chunks::<4>()`, computing the
  IDENTICAL summation order so results stay bit-exact) instead of
  `row[d]`-indexing. That closed dim-32 f32 from **1.423× → 0.966×**
  (gate-style knn, leaf=10) without any SIMD intrinsics or reordering — see
  `.superpowers/sdd/2026-08-23-nanoflann-rs-m2.5-perf/task-3-report.md` for
  the full A/B. The `point_row` fast path from this Task-14 record and
  M2.5-T3's `point_row` are the same trait method; the fix wasn't "add row
  access", it was "make row access bounds-check-free", which needed the
  chunked walk specifically.

What's left in the ~4-5% gap is architectural, not a missed optimization in
the hot kernel:

1. **Recursive `search_level` call overhead.** Both Rust and C++ implement
   the tree walk as native recursion (`search_level`/`searchLevel`), ~14
   levels deep for a balanced tree over 100k points at leaf_max_size=10
   (`log2(100000/10) ≈ 13.3`). Directly observed (not inferred) in the same
   post-Candidate-1 asm dump used above: the monomorphized `search_level`
   body contains two `callq` instructions targeting its OWN mangled symbol
   (one for the "best child" recursive descent, one for the "other child"
   pruned-branch descent) —

   ```
   callq   _RINvNtCs4tP4CM7yoJ7_12nanoflann_rs6search12search_leveldINtNtB4_...
   cmpl    $3, %ebp
   jae     .LBB1_44
   ...
   .LBB1_40:
       movq    %r12, %rdi
       movq    %r14, %rsi
       movq    %rbx, %rdx
       movl    %r13d, %r8d
       movq    %r15, %r9
       vmovss  %xmm2, 8(%rsp)
       callq   _RINvNtCs4tP4CM7yoJ7_12nanoflann_rs6search12search_leveldINtNtB4_...
   ```

   (symbol truncated for width; full mangled name matches the function's own
   `.type` line). Both call sites set up a fresh argument list in registers
   (`movq`/`movl`/`vmovss` into `%rdi`/`%rsi`/`%rdx`/`%r8`/`%r9`) and a real
   `callq` rather than being inlined — confirming LLVM did not inline this
   direct recursion (as expected; LLVM generally declines to inline
   self-recursive calls beyond a shallow, statically-bounded unroll, and tree
   depth here is a runtime value). So each of the ~14 levels pays a full
   function-call ABI cost (argument marshalling, `call`/`ret`, any
   caller-saved register spill/reload) on top of the leaf-kernel work. This
   is structurally present in the C++ source too (`searchLevel` is native
   recursion in nanoflann.hpp as well); any remaining difference is down to
   each compiler's/ABI's call-overhead characteristics, not an algorithmic
   gap. Converting this to an explicit-stack iterative walk (like the
   builder's `SubtreeBuilder`) is a plausible further win, but is a
   materially larger, riskier rewrite of the query hot path that risks the
   bit-exact traversal-order/tie-break contract multiple xval tests pin
   down — out of this task's "measure → change → re-measure, low-risk
   surgical fixes" scope, and arguably borders on the "batch APIs" territory
   explicitly excluded from M1.
2. **Measurement noise.** Across 7 perf-gate runs on this WSL2 host, the
   knn_fixed3 ratio itself ranged 0.937-1.079 (median 1.042) — a ~14-point
   spread attributable to scheduler/thermal jitter on a shared VM, not code
   changes (the build.rs-only Candidate 3 and 5 changes should have zero
   effect on the query path's absolute timing, yet runs before/after those
   commits show similar spread). The gate's own 1.25 margin exists
   specifically to absorb this; the milestone's stricter ≤1.0 bar is a
   secondary aspiration the brief explicitly allows documenting as an
   analyzed residual gap rather than a hard blocker.

Given both factors, closing the remaining ~4-5% would require either (a)
accepting the recursion-to-iteration rewrite risk noted above, or (b)
SIMD/batching — both out of M1 scope per the brief. The perf gate passes
with comfortable margin (1.047 ≤ 1.25) and build is now at parity; this is
recorded as the task's honest final state per the brief's "rigorous
documented analysis" allowance.

### `leaf_max_size` sweep

Measured via `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench
bench_build -- leaf_sweep --sample-size 10 --measurement-time 2
--warm-up-time 1` (a short/quick sweep — 10 samples, 2s measurement window
per point, vs. the headline tables' longer criterion defaults, so treat this
as a spot check, not the high-confidence numbers above). `n = 100_000`, dim
3, f32, uniform data, `leaf_max_size ∈ {1, 4, 10, 16, 32, 50, 128, 1024}`,
both build and k=10 knn time, for both libraries. Point estimates below are
criterion's reported median of the 10 samples; ratio = rust/cpp.

| `leaf_max_size` | build rust | build cpp | build ratio | knn rust | knn cpp | knn ratio |
|---|---|---|---|---|---|---|
| 1 | 13.585 ms | 14.581 ms | 0.932 | 1.4255 µs | 1.2564 µs | 1.135 |
| 4 | 11.642 ms | 11.751 ms | 0.991 | 0.9558 µs | 0.9569 µs | 0.999 |
| 10 (default) | 9.923 ms | 9.722 ms | 1.021 | 0.8192 µs | 0.8168 µs | 1.003 |
| 16 | 9.080 ms | 8.762 ms | 1.036 | 0.7917 µs | 0.8034 µs | 0.985 |
| 32 | 7.950 ms | 7.808 ms | 1.018 | 0.8321 µs | 0.8377 µs | 0.993 |
| 50 | 7.310 ms | 7.118 ms | 1.027 | 0.9445 µs | 0.9435 µs | 1.001 |
| 128 | 6.252 ms | 6.020 ms | 1.039 | 1.2543 µs | 1.2263 µs | 1.023 |
| 1024 | 4.459 ms | 4.257 ms | 1.047 | 3.7129 µs | 3.2894 µs | 1.129 |

Honest conclusion from these numbers: build time falls monotonically as
`leaf_max_size` grows (fewer, bigger leaves means less split/node-allocation
overhead, on both sides) — expected and not particularly informative for
picking a default. The knn column is the one that matters for choosing a
default, and it's a shallow U shape: both very small (`1`) and very large
(`1024`) leaves cost more per query (ratio 1.13-1.14x against C++ at both
extremes — small leaves mean more tree levels/recursive-call overhead per
query, large leaves mean more points to linearly scan per leaf), while
`leaf_max_size` in roughly `4..50` sits close to parity (ratios 0.985-1.003
in this run). `leaf_max_size = 4` measured marginally best on both build and
knn here, but this is a single 10-sample quick run — not enough to
distinguish it from `10` at this noise level, and reading too much into one
short sweep would overfit to this run's jitter. `10` (nanoflann's own
default, and this crate's) remains a good, defensible choice: it sits inside
the near-parity knn band this sweep identifies, and there is no value in
this sweep that clearly dominates it on both axes at once.

### Parallel build

Measured via `xval::timed_median_ms` (the same median-of-7,
one-untimed-warmup methodology `tests/perf_gate.rs` and
`examples/report_data.rs` use), dim 3, f32, uniform data, `leaf_max_size =
10`: `BuildThreads::Auto` (rust) vs C++ `n_thread_build = 0` (both mean
"use the ambient/hardware thread pool" on their respective sides) on this
machine's 8 visible threads. The `n = 1_000_000` sequential/parallel rows
mirror `examples/report_data.rs`'s existing `build_1M_dim3_f32_seq`/`_par`
speed-JSON rows (regenerated here alongside the new `n = 100_000` row, which
`report_data.rs` does not currently emit); `n = 100_000` was added via a
throwaway example reusing the same `xval` helpers, run once, then removed.

| n | seq rust | seq cpp | seq ratio | par (Auto/0) rust | par (Auto/0) cpp | par ratio |
|---|---|---|---|---|---|---|
| 100,000 | 9.201 ms | 9.643 ms | 0.954 | 2.697 ms | 4.731 ms | 0.570 |
| 1,000,000 | 120.969 ms | 122.513 ms | 0.987 | 27.159 ms | 47.310 ms | 0.574 |

The sequential ratios (0.954, 0.987) are consistent with the perf gate's
`build_100k_dim3_f32_seq` figure above. The parallel ratios are the notable
result: Rust's deterministic rayon-based parallel build (`build_parallel.rs`
— partition-then-`rayon::join`, task-local arenas merged at the end) runs
roughly **1.75x faster** than the C++ oracle's own `n_thread_build = 0`
threaded build on this 8-thread machine, at both sizes tested. This crate's
`BuildThreads` docs already deviate from claiming exact parity with C++'s
async per-node thread-gating (see the README's "Deliberate deviations"
table) — this measurement is consistent with that documented difference in
scheduling strategy translating into a real wall-clock win here, not just a
"produces the same tree" equivalence.

### Test status (at M1 completion — superseded)

`cargo test --workspace --all-features` reported 269 passed / 0 failed / 8
ignored at the point M1 was declared done. This count is specific to the
M1-only workspace and is **stale now that M2 added a second crate's worth of
tests** — see the file-level "Test status" section below for the current,
combined M1+M2 total. The M1-era heavy-suite result it stood alongside
still holds unchanged: `cargo test --workspace --all-features --release --
--ignored` (heavy 1M-point stress builds/queries + canary mutation tests)
was 0 failed, including the two deepest degenerate-tree constructions
(`heavy_exponential_build_1m` depth ~2115, `heavy_exponential_build_1m_dim8`
depth ~16794) built with the bbox scratch pool (Candidate 3).

## M2 — dynamic forest (Bentley–Saxe `DynamicKdTree`)

Same machine, same methodology as the M1 section above (WSL2, AMD Ryzen 7
9800X3D, `rustc 1.98.0`, `RUSTFLAGS="-C target-cpu=native"`, release profile
`codegen-units = 1`/`lto = "thin"`, C++ oracle `-O3 -march=native
-ffp-contract=off`) — the dynamic forest reuses the exact same perf-gate/
report-data machinery (`xval::timed_median_ms`, median-of-7, one untimed
warmup) as the four M1 gates, just against two new workloads. Full command
lines and environment/toolchain provenance: `docs/EXPERIMENTS.md`.

### Perf gate (two new gated workloads, same 1.25 margin as M1's four)

`dyn_knn_after_churn_dim3_f32`'s workload was corrected during the M2
final-review fix wave (branch `m2-dynamic`, commit range starting after
`5f671a0`): the original workload removed 5k points and re-added ALL 5k
before querying, which is a provable no-op — every tombstone gets
reactivated and `removed_len() == 0` at query time, so the timed knn loop
never actually exercised tombstone-filtered search against a forest with
live tombstones, and no merge triggered by the churn itself was ever timed.
The corrected workload removes 5k, re-adds only half (2.5k, leaving 2.5k
LIVE tombstones), then adds a fresh contiguous growth batch of 5k more
points — large enough (> 4096 = 2^12) to GUARANTEE at least one real
cascading merge across 12+ slots (pigeonhole: any 4096 consecutive
`point_count` values contain one ending in 12 one-bits). A plain `assert!
(removed_len() > 0)` in the setup, on both the Rust and C++ sides' shared
op sequence, proves the corrected workload actually leaves live tombstones
at query time; inverting it (reactivating all 5k again) reproduces the RED
failure this fix was written against — see `docs/EXPERIMENTS.md`'s
reproduction section for the captured panic. The old (no-op) figure is
kept below as historical, not comparable to the new one (different
workload, not a regression):

| Workload | Ratio (recorded at commit `5f671a0`, pre-fix-wave, historical no-op churn) | Ratio (post-fix, corrected live-tombstone workload) |
|---|---|---|
| `dyn_add_20k_dim3_f32` (20 batches × 1000 contiguous adds, from empty; workload unchanged by this fix) | 1.113 | 1.106–1.237 (range, see below) |
| `dyn_knn_after_churn_dim3_f32` (95k built, 5k removed / 2.5k re-added leaving live tombstones, then a 5k fresh growth batch triggering real merges, then 10,000 k=10 knn queries) | 0.964 (old, no-op-churn workload — not comparable) | 0.971–0.976 |

Both pass the 1.25 gate threshold, but the margin is real, not
"comfortable" — `dyn_add`'s high end (1.237) is only 1.0% below the
threshold. Re-run four times during this fix wave — twice via the full
six-gate invocation (`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo
test -p xval --release --test perf_gate -- --ignored perf_gate
--test-threads=1 --nocapture`) and twice via the dynamic-only filter (same
command with `--ignored perf_gate_dyn`), `crates/xval/tests/perf_gate.rs`;
full pasted output of all four runs is in `docs/EXPERIMENTS.md`'s
reproduction-commands section, not repeated here — `dyn_add_20k_dim3_f32`
ranged **1.106–1.237** (combining this fix wave's four fresh runs with the
four runs recorded in an earlier documentation round; `dyn_add`'s workload
itself is unchanged by this fix wave, so both sets of runs measure the
same thing) and `dyn_knn_after_churn_dim3_f32` (the corrected workload)
ranged **0.971–0.976** across this fix wave's four fresh runs. Every
individual run still passes the 1.25 margin, but `dyn_add`'s high end is a
real, larger-than-typical single-run swing worth knowing about — see
`docs/EXPERIMENTS.md`'s WSL2 caveat. Not a regression: no code affecting
`dyn_add`'s workload changed during this fix wave.

`dyn_add`'s ratio (~1.1-1.2×, Rust slightly slower) and
`dyn_knn_after_churn`'s ratio (~0.97×, Rust slightly faster) both compare
the SAME algorithm implemented twice, not two different designs:
nanoflann's dynamic `addPoints` is inherently O(heavy) per point (a
merge-and-rebuild loop cascading through every slot up to the highest one
touched, not an amortized-O(1) insert — see `crates/nanoflann-rs/src/dynamic.rs`'s
`add_points` doc comment for the exact schedule), and this port matches
that schedule exactly (same `first0bit` slot selection, same "rebuild every
slot up to the highest touched, even a no-op call's slot 0" quirk) — so any
speed difference here reflects implementation quality, not an algorithmic
gap. See `crates/xval/benches/bench_dynamic.rs`'s module doc for the same
point made at length, plus its criterion-only third benchmark group,
`dyn_churn` (self-restoring remove-then-reactivate churn on a pre-built
100k forest, `sample_size(10)` — not gated, criterion spot-check only).

### Churned-forest accuracy row (from `report_data`, reproduced during this task)

A 120-op churn sequence (`xval::dyn_ops`, seeded, capacity 20k, dim 3, f32)
is applied identically to both implementations, then 2000 k=10 knn queries
are scored against a brute-force ground truth restricted to the live set
(`brute_force_knn_l2_live_f32`/`score_exact_tie_aware_live_f32` — see
`docs/EXPERIMENTS.md`'s methodology section for why a live-set-aware
scorer is necessary here, unlike the static M1 accuracy rows). Reproduced
during this task (`RUSTFLAGS="-C target-cpu=native" cargo run -p xval
--release --example report_data`, live/removed counts from this exact run):

| workload | rust exact-tie-aware | cpp exact-tie-aware | rust==cpp bit-exact | live / removed |
|---|---|---|---|---|
| `dyn_churn_dim3_f32_k10_eps0` | 1.0 | 1.0 | true | 2232 / 10 |
| `dyn_churn_dim3_f32_k10_eps0.1` | 0.6645 | 0.6645 | true | 2232 / 10 |

The `eps=0` row is exact (1.0/1.0) and bit-exact on both sides — no
divergence. The `eps=0.1` row's identical (not just close) 0.6645 scores
and identical relative-error statistics between Rust and C++ are the
signature of genuine bit-exact agreement under an APPROXIMATE (eps-pruned)
query, not two implementations independently landing on the same number by
chance — a real eps-pruning divergence would show up as differing scores.
`dyn_evidence` fields present in the underlying JSON row (`grow_and_add_count
62`, `remove_count 34`, `readd_count 24`, `tombstone_migrations 33`) confirm
the churn sequence actually exercised merges/tombstone migrations, not a
trivially-easy all-live degenerate case.

### Empty-forest quirk (inherited from C++, not a Rust bug)

`DynamicKdTree::find_neighbors`'s generic escape hatch, called directly on
a forest with zero occupied slots, returns whatever an untouched result set
naturally reports on `.full()`: `false` for `KnnResultSet`/`RknnResultSet`,
but **`true`** for `RadiusResultSet` (its `full()` is hardwired `true`
regardless of whether anything was ever added, mirroring nanoflann.hpp:433
exactly — see M1's Corrections #4 for the same quirk on the static tree).
The additive `radius_search`/`radius_search_with` wrappers return the found
COUNT (`0` in that case), so the quirk is invisible through the wrapper
API; it is only observable by calling `find_neighbors` directly with a
fresh, empty `RadiusResultSet`. Not benchmarked separately (it's a
correctness/API-surface fact, not a perf number), but documented here
because it's exactly the kind of "why does this return `true` with zero
results" surprise a reader of this benchmarks doc might otherwise hit while
writing their own microbenchmark against an empty forest.

## M2.5 — performance deep-dive

Same machine/methodology as M1/M2 above (WSL2, AMD Ryzen 7 9800X3D, `rustc
1.98.0`, `RUSTFLAGS="-C target-cpu=native"`, release profile
`codegen-units=1`/`lto="thin"`, C++ oracle `-O3 -march=native
-ffp-contract=off`). Two changes landed on top of each other: **T3** (a
bounds-check-free chunked row walk in the `L2`/`L1` kernel, closing the
dim-32 gap) and **T2** (explicit-stack iterative `search_level`, closing the
fixed-dim-3 residual — at the cost of giving back part of T3's win on two
other gates, a deliberate, reviewed trade-off, not a free improvement). Full
task reports:
`.superpowers/sdd/2026-08-23-nanoflann-rs-m2.5-perf/task-{2,3,4}-report.md`.
Every number below was re-measured fresh by this task (T4) at commit
`2d23db4` (2 runs per workload, pasted in full in `docs/EXPERIMENTS.md`),
**except the "Post-T3" column**, which is cited from T3's own report as
measured at commit `549f1ac` (T3's landing commit, before T2 existed) —
not re-run by T4, since T2 had already superseded that state by the time
this task started.

### Summary table: milestone-start → post-T3 → post-T2/final

All ratios `rust_ms / cpp_ms`; lower is better for Rust; gate margin is
`<= 1.25`. "Milestone-start" is the recorded pre-M2.5 range (this file's M1/M2
sections above); "post-T3" is T3's own single measured run with the kernel
fix landed, C++ unchanged (T3's report, commit `549f1ac`); "post-T2/final" is
this task's fresh two-run sweep with both changes landed (commit `2d23db4`).

| Workload | Milestone-start (recorded range) | Post-T3 (kernel fix only) | Post-T2/final (T4, 2 fresh runs) |
|---|---|---|---|
| `build_100k_dim3_f32_seq` | ~1.0 | 0.971 | 0.974–1.079 |
| `knn_fixed3_dim3_f32_k10` | 1.042–1.058 | 1.054 | **1.011–1.040** |
| `knn_dyn_dim8_f64_k10` | 0.95–0.97 | **0.867** | 0.950–0.966 |
| `radius_dim3_f32` | 0.77–0.81 | 0.777 | 0.827–0.828 |
| `dyn_add_20k_dim3_f32` | 1.106–1.237 | 1.123 | 1.121–1.170 |
| `dyn_knn_after_churn_dim3_f32` | 0.971–0.976 | 0.974 | **0.873–0.916** |

**The T2 trade-off, stated plainly — this is not a "no regression" story.**
`knn_dyn_dim8_f64_k10` and `radius_dim3_f32` both moved measurably worse
after T2 landed than they were immediately post-T3: T3's kernel fix took
`dim8` from ~0.95 down to 0.867 (a real win, ~0.083 of ratio), and T2's
iterative-search conversion gave back essentially all of that win (0.867 →
0.950–0.966, a give-back of 0.083–0.099 — 100–119% of the win, landing back
inside the milestone-start band rather than below it);
`radius` moved from 0.777 (matching the milestone-start band) to
0.827–0.828, landing about 2% *outside* the milestone-start band's upper
edge (0.81) — a real, if small, net loss relative to where this crate
started M2.5, not just relative to T3's peak. This is confirmed reproducible
(not noise): the task-2 reviewer independently measured raw-ms interleaved
A/Bs (radius +4.4–7.4%, dim8 +6.4–6.8%, direction-clean across both forward
and reversed run orderings — see `task-2-report.md` Fix-round-1 Item 2) and
this task's fresh sweep lands in the same ranges. **It was ruled to land
anyway**, because `knn_fixed3` improved (1.042–1.058 → 1.011–1.040) and
`dyn_knn_after_churn` improved dramatically (0.971–0.976 → 0.873–0.916, a
bigger win in this fresh sweep than T2's own report captured), and because
the explicit-stack form gives the query path stack-overflow immunity on
degenerate trees that the old native-recursion form did not have (C++
remains exposed — see "Robustness upgrade" below). **Both give-back
workloads remain Rust wins vs C++ in absolute terms** (both ratios stay
comfortably under 1.0) — the loss is entirely against this crate's own
milestone-start baseline, not against the C++ oracle.

### dim-32/64 headline: the M1-era gap is closed at f32

| dim | scalar | milestone-start (T1 diagnostic baseline) | T3 landed (single measured run) | T4 final sweep (2 fresh runs) |
|---|---|---|---|---|
| 32 | f32 | 1.423 | 0.966 | **0.966–1.008** |
| 32 | f64 | 1.315 | 1.113 | 1.103–1.111 |
| 64 | f32 | 1.918 | 1.228 | 1.162–1.171 |
| 64 | f64 | 1.213 | 0.874 | 0.879–0.887 |

Measured via `cargo run -p xval --release --example m25_diag -- knn`
(n=100k, 200 queries, k=10, leaf=10 — the "gate-style knn" methodology T1/T3
used; not one of the six `PERF_GATE`-gated workloads, but reproducible the
same way, see `docs/EXPERIMENTS.md`).

Mechanism: `L2::eval`/`L1::eval`'s per-component bounds checks split the
unrolled kernel body into 8 basic blocks and blocked LLVM's SLP vectorizer
at runtime-known dim; gcc compiles the *identical* summation order to
AVX2/AVX-512 with no reordering, proving the order itself was always
vectorizable. The fix — `DataSource::point_row` plus a bounds-check-free
`as_chunks::<4>()` row walk computing the IDENTICAL summation order — is
proven bit-exact with the fallback via a 6-salt-per-dim discriminating test
sweep (`task-3-report.md` §2, §8.1), **not** a reordering or SIMD intrinsic.
dim-32 f32 flips from a ~1.42x C++ win to parity-or-better (0.97–1.01x);
dim-64 f32 improves substantially (1.92x → 1.16–1.23x) but remains open
(see "Honest residuals" below); dim-32/64 f64 both land close to or under
parity (1.10–1.11x / 0.88x).

### Robustness upgrade: query stack-overflow immunity on degenerate trees

T2 converted `search_level` from native self-recursion to an explicit-stack
iteration (`Frame`/`Phase`/`FrameStack`: a `MaybeUninit`-backed 128-frame
inline array, with a heap-`Vec` spill for depth >= 129). This removes a
structural risk the C++ oracle still has: nanoflann's `searchLevel` is
native recursion with no depth bound, so a sufficiently degenerate tree
queried on a small worker thread can, in principle, overflow the native
call stack; this port's iterative form cannot, by construction, regardless
of tree depth. Verified at real scale: `heavy_query_degenerate_trees` (a
tree of depth ~16,794, this crate's documented degenerate-tree ceiling) now
exercises the spill path directly (frame 129 onward) rather than relying on
worker-thread stack size, and passes.

This upgrade has a real, documented cost, not a free lunch: a query deeper
than 128 levels pays per-query heap reallocation (`overflow: Vec`'s normal
doubling growth) that the old native-recursion form never paid (zero heap
allocation at any depth, at the price of the stack-overflow exposure
above). `SEARCH_STACK_INLINE_CAPACITY = 128` is arbitrary-with-headroom
(comfortably above every depth this crate's realistic gate/xval workloads
produce, ~13-17 levels for a balanced 100k-point/leaf-10 tree) — not
derived from a principled depth-distribution analysis, and **not** a lever
that recovers the `radius`/`dim8` give-back above: a capacity-32 variant
was A/B-tested and did not help (`task-2-report.md` Fix round 1, Items 2
and 7).

### Honest residuals (still open after M2.5)

- **Fixed-dim-3 knn**: ~1.01–1.04x in this task's fresh sweep (asm-evidenced
  ~2% residual attributed to frame store/reload cost in the now-fully-inlined
  explicit-stack form — `task-2-report.md` Fix round 1, Item 2) — within
  this host's documented noise floor (0.956–1.192 over 8 runs, T1's
  report), not chased further.
- **dim-32 f64**: ~1.10–1.11x — closed substantially (from 1.315x) but not
  to parity; not specifically targeted by T3 (which prioritized f32).
- **dim-64 f32**: ~1.16–1.23x — improved substantially (from 1.918x) but
  not closed; T1/T3 both flagged dim-64 as "improved, not eliminated," a
  deliberately smaller-priority residual than dim-32, left open by design
  (T3's target was dim-32).
- **Fast-math / reordered-arithmetic kernels**: remains a roadmap idea only
  — **NOT taken** in M2.5. No arithmetic reordering landed anywhere in the
  default build; T3's chunked row walk (the only kernel change) is proven
  bit-exact with the fallback via multi-salt discriminating tests
  (`task-3-report.md` §8.1) — parity is intact. A fast-math feature flag
  (non-default, its own accuracy docs, never in the parity suites) remains
  a possible future lever, unexplored.
- **Parallel slot rebuilds for the dynamic adaptor**: not touched by M2.5
  (T2/T3 both worked the query/kernel hot paths, not `add_points`'s
  sequential rebuild schedule); C++ rebuilds sequentially too, so this
  remains a documented deviation-not-yet-taken, not a regression.

## Test status (workspace, current — supersedes both crate-specific counts above)

`cargo test --workspace --all-features` (re-run for M2.5 task 4, commit
`2d23db4`): **359 passed, 0 failed, 15 ignored** (up from M2's
final-review-fix-wave count of 345/0/15 — M2.5 added `nanoflann-rs`'s
`search.rs` `FrameStack`/spill tests (T2) and `metric.rs`'s `point_row`
bit-equality + contract tests (T3), all counted here; M1's original
269/0/8 remains superseded). Note: `--all-features` and default features
are identical invocations here — `parallel` (rayon, on by default) is
`nanoflann-rs`'s only feature, so `cargo test --workspace` alone (as used
elsewhere in this repo, e.g. the README) exercises exactly the same code.
`cargo clippy --workspace --all-targets -- -D warnings`: clean (re-run for
M2.5 task 4). `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`:
zero warnings (re-run for M2.5 task 4). `cargo build -p nanoflann-rs
--no-default-features` + `cargo test -p nanoflann-rs --no-default-features`:
clean build, **179 passed, 0 failed, 3 ignored** (re-run for M2.5 task 4).
`cargo test --workspace --all-features --release -- --ignored
--test-threads=1` (heavy suite, including M1's two deepest degenerate-tree
builds, M2's mutation canaries, and M2.5's `heavy_query_degenerate_trees`
spill-path exercise): **0 failed, 151.80s** for the `nanoflann-rs` unit-test
binary (re-run for M2.5 task 4) — see `docs/EXPERIMENTS.md`'s
reproduction-commands table for the exact invocation and this task's fully
captured run, plus both miri commands (Stacked Borrows and Tree Borrows)
validating `search.rs`'s `unsafe` `FrameStack` code.
