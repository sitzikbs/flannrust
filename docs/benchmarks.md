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
run-to-run scheduler/thermal jitter (~±5-10% on individual runs — M1's
characterization; **Update (M2.6):** the repo's canonical noise floor is now
the n=100-per-side, 4-session `knn_fixed3` characterization — session
medians 1.030–1.067, per-session per-rep ratio σ≈0.03–0.05, approx.
conservative envelope of 0.94–1.16 (extreme session medians ± 2×max
per-session σ), `docs/EXPERIMENTS.md` "M2.6 task 2" — superseding
the old 8-run `knn_fixed3` sweep, 0.956–1.192, `docs/EXPERIMENTS.md` "M2.5
task 1" (kept there as historical). **Update (M2.6 task 7): re-checked on the idle host — not inflated; the fresh 4-session envelope computes slightly *wider* (0.884–1.176), 0.94–1.16 kept (see `docs/EXPERIMENTS.md` "M2.6 task 7").** The perf
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
`docs/reports/m2.5/task-3-report.md`.
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
  `docs/reports/m2.5/task-3-report.md` for
  the full A/B. The `point_row` fast path from this Task-14 record and
  M2.5-T3's `point_row` are the same trait method; the fix wasn't "add row
  access", it was "make row access bounds-check-free", which needed the
  chunked walk specifically.

**Update (M2.5-T2): the two numbered items below are superseded.** Item
1's native recursion no longer exists — M2.5 converted `search_level` to
an explicit-stack iteration (commit `1ecfd46`, fix round `2d23db4`); the
compiled asm now contains zero `search_level` call targets (re-captured
at `84c0781`, `docs/EXPERIMENTS.md` "M2.5 asm re-capture" subsection),
and what that conversion won and gave back is accounted for in this
file's "M2.5 — performance deep-dive" section. Item 2's 7-run 0.937–1.079
spread is an older, smaller sample than the repo's canonical noise floor —
**Update (M2.6):** now the n=100-per-side, 4-session characterization
(session medians 1.030–1.067, conservative envelope 0.94–1.16 (extreme
session medians ± 2×max per-session σ),
`docs/EXPERIMENTS.md` "M2.6 task 2"), superseding the 8-run 0.956–1.192
figure (`docs/EXPERIMENTS.md` "M2.5 task 1", kept there as historical).
**Update (M2.6 task 7): re-checked on the idle host — not inflated; the fresh 4-session envelope computes slightly *wider* (0.884–1.176), 0.94–1.16 kept (see `docs/EXPERIMENTS.md` "M2.6 task 7").**
Both retained below as the M1 record, not as the current state.

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

**Update (M2.6, commit `a30a819`, 2026-08-25):** re-measured under the
adaptive-n harness (`xval::measure_pair`, n=100/side, 4 independent
sessions — 2 gate + 2 report-chain) — `dyn_add_20k_dim3_f32` **1.112–1.149**,
`dyn_knn_after_churn_dim3_f32` **0.940–0.947**. Both narrower than the
ranges in the table above (the old high-water marks read as small-sample
median-of-7 noise in hindsight, not the true steady-state ratio); both
still comfortably clear the 1.25 gate. Full account, every pasted run:
`docs/EXPERIMENTS.md` "M2.6 task 2"; `docs/benchmarks.md` "M2.6 —
statistical re-verification" (below). Table above kept as-is (historical
pasted evidence).

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
touched, not an amortized-O(1) insert — see `crates/flannrust/src/dynamic.rs`'s
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
fixed-dim-3 residual — at the cost of giving back all of T3's `dim8` win
and leaving `radius` ~4.4% above its pre-M2.5 range, a deliberate,
reviewed trade-off, not a free improvement). Full
task reports:
`docs/reports/m2.5/task-{2,3,4}-report.md`.
Every number in the "Post-T2/final" columns below was re-measured fresh
by T4 at commit `2d23db4` (2 runs per workload, pasted in full in
`docs/EXPERIMENTS.md`'s "M2.5 task 4" subsection). The "Post-T3" column is
T3's own interleaved-A/B "after" medians (3 runs each), pasted in
`docs/EXPERIMENTS.md`'s "M2.5 task 3" subsection — measured on the T3
trees landed as `fecbdde` (the four static gates) and `549f1ac` (the two
dynamic gates, re-measured in T3's fix round), before T2 existed; not
re-run by T4, since T2 had already superseded that state. The
"Milestone-start" column is a min–max over every pre-M2.5 gate run pasted
in this repo (definition under the table).

### Summary table: milestone-start → post-T3 → post-T2/final

All ratios `rust_ms / cpp_ms`; lower is better for Rust; gate margin is
`<= 1.25`. "Milestone-start" is the min–max across every pre-M2.5 gate run
pasted in this repo: this file's M1 table ("After Task 14" and "Final
single run" columns), the four pre-M2.5 six-gate runs and the
`perf_gate_dyn` runs in `docs/EXPERIMENTS.md` §3, and the "before" arm of
T3's interleaved A/B (`docs/EXPERIMENTS.md` "M2.5 task 3") — no value in
that column is a rounded band; "post-T3" is T3's interleaved-A/B "after"
median with the kernel fix landed, C++ unchanged; "post-T2/final" is T4's
fresh two-run sweep with both changes landed (commit `2d23db4`).

| Workload | Milestone-start (pre-M2.5 pasted range) | Post-T3 (kernel fix only, A/B after-median) | Post-T2/final (T4, 2 fresh runs) |
|---|---|---|---|
| `build_100k_dim3_f32_seq` | 0.986–1.018 | 0.971 | 0.974–1.079 |
| `knn_fixed3_dim3_f32_k10` | 1.030–1.063 | 1.054 | **1.011–1.040** |
| `knn_dyn_dim8_f64_k10` | 0.944–1.091 (0.944–0.984 excluding one flagged outlier) | **0.867** | 0.950–0.966 |
| `radius_dim3_f32` | 0.767–0.792 | 0.777 | 0.827–0.828 |
| `dyn_add_20k_dim3_f32` | 1.106–1.237 | 1.123 | 1.121–1.170 |
| `dyn_knn_after_churn_dim3_f32` | 0.956–0.976 | 0.974 | **0.873–0.916** |

**Update (M2.6):** the "Post-T2/final" column above is T4's 2-run,
median-of-7 sweep (commit `2d23db4`) and is retained as-is (historical
pasted evidence, never edited in place). A fresh n=100-per-side, 4-session
re-verification (2 gate + 2 report-chain sessions, commit `a30a819`,
`docs/EXPERIMENTS.md` "M2.6 task 2") widens/corrects several of these
figures: `build_100k` **0.967–1.039** (reproducibly split by configuration, not
noise — see below; **Update (M2.6 task 6): this did not reproduce a third
time — see the "M2.6 task 6" subsection below, "build_100k re-hedge"**),
`knn_fixed3` **1.030–1.067**, `dim8` **0.929–0.937**
(now BETTER than the entire pre-M2.5 range, not merely back inside it),
`radius` **0.807–0.867** (wider than the single "~0.83" point estimate
this file and `docs/ROADMAP.md` previously used), `dyn_add`
**1.112–1.149** (narrower than the old 1.121–1.170/1.106–1.237 spreads),
`dyn_knn_after_churn` **0.940–0.947**. Full per-conclusion verdicts and
every pasted run: `docs/EXPERIMENTS.md` "M2.6 task 2" subsection; the "M2.6
— statistical re-verification" section below this one.

**The T2 trade-off, stated plainly — this is not a "no regression" story.**
`knn_dyn_dim8_f64_k10` and `radius_dim3_f32` both moved measurably worse
after T2 landed than they were immediately post-T3: T3's kernel fix took
`dim8` from 0.945 (its own interleaved before-median) down to 0.867 (a
real win, 0.078 of ratio), and T2's iterative-search conversion gave back
**all** of that win (0.867 → 0.950–0.966, a give-back of 0.083–0.099
against a 0.078 win, landing back inside the pre-M2.5 range 0.944–0.984
rather than below it); `radius` moved from 0.777 (inside the pre-M2.5
range 0.767–0.792) to 0.827–0.828, about **4.4% above that range's top**
(0.792) — a real net loss relative to where this crate started M2.5, not
just relative to T3's peak. This is confirmed reproducible (not noise):
T2's raw-ms interleaved A/Bs (radius +4.4–7.4%, dim8 +6.4–6.8%,
direction-clean across both forward and reversed run orderings — pasted
in `docs/EXPERIMENTS.md`'s "M2.5 task 2" subsection, source
`task-2-report.md` Fix round 1 Item 2) and this task's fresh sweep land in
the same ranges. **It was ruled to land anyway**, because `knn_fixed3`
improved (1.030–1.063 → 1.011–1.040 in the sweeps; T2's interleaved A/B
medians 1.052 → 1.0205 are the cleaner evidence) and
`dyn_knn_after_churn` improved dramatically (0.956–0.976 → 0.873–0.916, a
bigger win in this fresh sweep than T2's own report captured), and because
the explicit-stack form gives the query path stack-overflow immunity on
degenerate trees that the old native-recursion form did not have (C++
remains exposed — see "Robustness upgrade" below). **Both give-back
workloads remain Rust wins vs C++ in absolute terms** (both ratios stay
comfortably under 1.0) — the loss is entirely against this crate's own
milestone-start baseline, not against the C++ oracle.

### dim-32/64 headline: the M1-era gap is closed at f32

| dim | scalar | milestone-start (T3's interleaved "before" median, pre-T3 code) | T3 landed (interleaved A/B "after" median, 3 runs) | T4 final sweep (2 fresh runs) |
|---|---|---|---|---|
| 32 | f32 | 1.423 | 0.966 | **0.966–1.008** |
| 32 | f64 | 1.315 | 1.113 | 1.103–1.111 |
| 64 | f32 | 1.918 | 1.228 | 1.162–1.171 |
| 64 | f64 | 1.213 | 0.874 | 0.879–0.887 |

Measured via `cargo run -p xval --release --example m25_diag -- knn`
(n=100k, 200 queries, k=10, leaf=10 — the "gate-style knn" methodology T1/T3
used; not one of the six `PERF_GATE`-gated workloads, but reproducible the
same way, see `docs/EXPERIMENTS.md` — the first two columns are pasted in
its "M2.5 task 3" subsection, the last in "M2.5 task 4").

Mechanism: `L2::eval`/`L1::eval`'s per-component bounds checks split the
unrolled kernel body into 8 basic blocks and blocked LLVM's SLP vectorizer
at runtime-known dim; gcc compiles the *identical* summation order to
AVX2/AVX-512 with no reordering, proving the order itself was always
vectorizable. The fix — `DataSource::point_row` plus a bounds-check-free
`as_chunks::<4>()` row walk computing the IDENTICAL summation order — is
proven bit-exact with the fallback via a 6-salt-per-dim discriminating test
sweep (RED captures under two sabotages pasted in `docs/EXPERIMENTS.md`'s
"M2.5 task 3" subsection; full transcripts `task-3-report.md` §8.1),
**not** a reordering or SIMD intrinsic. dim-32 f32 flips from a ~1.42x
C++ win to near-parity (0.97–1.01x in T4's 2-run sweep). **Update (M2.6):**
a fresh median-of-15, 2-session re-run (`docs/EXPERIMENTS.md` "M2.6 task
2", conclusion (a)) widens this to **0.986–1.062** — straddling parity,
including one session where C++ is measurably faster — so "parity-or-better"
is corrected to "near parity, occasionally a few percent either way"; the
headline ~1.42x→~1.0x gap closure itself is unaffected and re-confirmed.
dim-64 f32 improves
substantially (1.918x → 1.16–1.23x across T3+T4 runs: T3's A/B median
1.228, T4's 1.162–1.171, the fix-wave re-run's 1.169) but remains open
(see "Honest residuals" below); dim-32/64 f64 both land close to or under
parity (1.10–1.11x / 0.88x).

**Update (M2.6 task 7, user-directed idle-host re-measurement):**
dim-32 f64, dim-64 f32, and dim-64 f64 above are all **CORRECTED wider** —
a fresh 2-session idle-host re-run of `m25_diag knn`, combined with the
M2.6-task-2 raw sweep data these three rows never previously
incorporated (only dim-32 f32 was formally re-verified in task 2), gives:
dim-32 f64 **1.080–1.104** (was 1.103–1.111 / "~1.10–1.11x"), dim-64 f32
**1.082–1.235** (was 1.162–1.171 / "1.16–1.23x across T3+T4 runs"),
dim-64 f64 **0.862–0.956** (was 0.879–0.887). The qualitative read is
unaffected ("improved substantially, not closed to parity" for the first
two, "under parity" for dim-64 f64); dim-64 in particular turns out to be
an inherently higher-variance workload on this host — the spread persists
even under confirmed-idle-host conditions, so it isn't solely a
contamination artifact. Full evidence: `docs/EXPERIMENTS.md` "M2.6
task 7".

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
was A/B-tested and did not help (NEW32 columns in `docs/EXPERIMENTS.md`'s
"M2.5 task 2" subsection; `task-2-report.md` Fix round 1, Items 2 and 7).

### Honest residuals (still open after M2.5)

- **Fixed-dim-3 knn**: 1.011–1.040x in T4's fresh sweep. What IS
  asm-evidenced: the walk is fully inlined — zero `search_level` call
  targets (`docs/EXPERIMENTS.md` "M2.5 asm re-capture"). What is NOT
  separately measured: the residual's mechanism — the T2 reviewer's
  diagnosis attributes it to frame store/reload cost in the explicit-stack
  form (`task-2-report.md` Fix round 1, Item 2), consistent with the
  give-back A/B data but never quantified from asm, so no "~2%" or any
  other share of the residual should be read as asm-derived. **Update
  (M2.6):** an n=100-per-side, 4-session re-verification widens the
  honestly-observed range to **1.030–1.067** (3.0–6.7%,
  `docs/EXPERIMENTS.md` "M2.6 task 2" conclusion (c)) — the whole residual
  still sits inside this host's documented noise floor, which is now
  itself an n=100-grounded characterization (session medians 1.030–1.067,
  conservative envelope 0.94–1.16 (extreme session medians ± 2×max
  per-session σ)) superseding the old 8-run 0.956–1.192
  figure (`docs/EXPERIMENTS.md` "M2.5 task 1", kept as historical) — not
  chased further. **Update (M2.6 task 7): re-confirmed on the idle
  host** — 4 fresh, loadavg-bracketed `knn_fixed3` sessions compute their
  own conservative envelope at 0.884–1.176, slightly *wider* than
  0.94–1.16, not narrower, so 0.94–1.16 stands unchanged as the canonical
  floor (`docs/EXPERIMENTS.md` "M2.6 task 7").
- **dim-32 f64**: ~1.10–1.11x — closed substantially (from 1.315x) but not
  to parity; not specifically targeted by T3 (which prioritized f32).
  **Update (M2.6 task 7):** CORRECTED wider, **1.080–1.104**
  (`docs/EXPERIMENTS.md` "M2.6 task 7").
- **dim-64 f32**: 1.16–1.23x across T3+T4 runs — improved substantially (from 1.918x) but
  not closed; T1/T3 both flagged dim-64 as "improved, not eliminated," a
  deliberately smaller-priority residual than dim-32, left open by design
  (T3's target was dim-32). **Update (M2.6 task 7):** CORRECTED wider,
  **1.082–1.235** — this workload turns out to be inherently
  higher-variance on this host, independent of host load
  (`docs/EXPERIMENTS.md` "M2.6 task 7").
- **dim-64 f64**: 0.879–0.887x (T4) — lands under parity. **Update (M2.6
  task 7):** CORRECTED wider, **0.862–0.956**, same higher-variance
  pattern as dim-64 f32 (`docs/EXPERIMENTS.md` "M2.6 task 7").
- **Fast-math / reordered-arithmetic kernels**: remains a roadmap idea only
  — **NOT taken** in M2.5. No arithmetic reordering landed anywhere in the
  default build; T3's chunked row walk (the only kernel change) is proven
  bit-exact with the fallback via multi-salt discriminating tests
  (`docs/EXPERIMENTS.md` "M2.5 task 3"; `task-3-report.md` §8.1) — parity
  is intact. A fast-math feature flag
  (non-default, its own accuracy docs, never in the parity suites) remains
  a possible future lever, unexplored.
- **Parallel slot rebuilds for the dynamic adaptor**: not touched by M2.5
  (T2/T3 both worked the query/kernel hot paths, not `add_points`'s
  sequential rebuild schedule); C++ rebuilds sequentially too, so this
  remains a documented deviation-not-yet-taken, not a regression.

## M2.6 — statistical re-verification

M2.6 task 1 (commit `a30a819`) replaced the perf gates' and `report_data`'s
fixed median-of-7 timing with an adaptive `n = clamp(10, 100,
floor(budget_s*1000/t_est_ms))` scheme (`xval::measure_pair`, 30s
budget/side) that publishes mean/std/median/min/max at `n=100` per side for
every workload in this repo. M2.6 task 2 (this section, commit `a30a819`,
2026-08-25) ran the full chain twice (2 gate sessions + 2 report-chain
runs, all n=100/side) plus `m25_diag`'s dim-8/16/32/64 sweep twice at a
bumped `RUNS=15` (up from 7), and re-verified every conclusion the M2.5
docs previously recorded. Full pasted evidence: `docs/EXPERIMENTS.md`
"M2.6 task 2" subsection. Combined ranges across all 4 fresh n=100
sessions (2 gate + 2 report-chain):

| Workload | M2.5 "Post-T2/final" (T4, 2 runs, median-of-7) | M2.6 (4 sessions, median-of-100) |
|---|---|---|
| `build_100k_dim3_f32_seq` | 0.974–1.079 | **0.967–1.039** (reproducibly split by configuration — see below; not noise, mechanism not isolated. **Update (M2.6 task 6): re-hedged, did not reproduce — "M2.6 task 6" subsection below.**) |
| `knn_fixed3_dim3_f32_k10` | 1.011–1.040 | **1.030–1.067** |
| `knn_dyn_dim8_f64_k10` | 0.950–0.966 | **0.929–0.937** (better than the entire pre-M2.5 band) |
| `radius_dim3_f32` | 0.827–0.828 | **0.807–0.867** |
| `dyn_add_20k_dim3_f32` | 1.121–1.170 | **1.112–1.149** (narrower than every prior range) |
| `dyn_knn_after_churn_dim3_f32` | 0.873–0.916 | **0.940–0.947** |

Every M2.6 figure is retained alongside, not instead of, the M2.5 pasted
evidence above (house style: pasted evidence is never edited in place).
Per-conclusion verdicts (each ends CONFIRMED or CORRECTED, per the M2.6
controller's binding list):

- **(a) dim-32 f32 win**: **CORRECTED**. Previously "parity-or-better"
  (0.966–1.008, T4's median-of-7). Fresh median-of-15, 2-session re-run:
  **0.986–1.062** — straddles parity, one session C++-faster by 6.2%. The
  ~1.42x→~1.0x gap-closing mechanism/magnitude is unaffected; only the
  "never loses" framing is corrected.
- **(b) T2 give-backs (radius ~0.83, dim8 f64 back to pre-M2.5 band)**:
  **CORRECTED**. `radius` ranges **0.807–0.867** across 4 sessions — wider
  than any single "~0.83" estimate, spanning from better-than-T2's-own-A/B
  to worse-than-M-py's-peak. `knn_dyn_dim8_f64_k10` ranges **0.929–0.937**
  — better than the entire pre-M2.5 pasted band (0.944–1.091 /
  0.944–0.984 excluding the flagged outlier), not merely "back inside" it.
- **(c) fixed-dim-3 residual "~1–4%"**: **CORRECTED**. 4-session range
  **1.030–1.067** (3.0–6.7%) — upper end exceeds the old characterization,
  though still comfortably inside the newly-characterized noise floor
  (below).
- **(d) dim-16 curse-of-dimensionality attribution**: **CONFIRMED**.
  Median-of-15 re-run: f32 ratio 0.682/0.682 (both sessions), rust_ms
  ~83.2–83.4ms vs. dim-8's ~3.53–3.56ms (~23.5x); `frac_points_scanned`
  (deterministic, byte-identical re-run) still jumps 0.0176→0.4875
  (27.7x), still sufficient alone to explain the timing cliff.
- **(e) the noise floor itself**: **CORRECTED** (replaced). See "the new
  noise floor" callout below.
- **(f) `dyn_add` "1.11–1.24"**: **CONFIRMED, narrowed**. 4-session range
  **1.112–1.149**, comfortably inside the old 1.106–1.237 — the n=100-median
  data suggests the true steady-state band is narrower than the old
  n=7-median range implied; the old 1.237 high-water mark reads as
  small-sample noise in hindsight, not a wider true distribution.
- **(g) `build_100k` "0.974–1.139," suspiciously wide**: **CORRECTED**.
  Reproducibly split by configuration — not primarily measurement noise or
  temporal drift, though dataset seed and process context were never
  crossed against each other, so which one is the actual cause is not yet
  isolated (see `docs/EXPERIMENTS.md` conclusion (g) for the full hedge).
  Both gate sessions
  (dataset seeded `"perf_gate_build"`) land at 0.967/0.967; both
  report-chain sessions (independently-seeded dataset, `"report_build_100k"`)
  land at 1.039/1.037 — each pair internally stable to <1%, ~7 points apart
  from the other pair. The old wide spread reflects which dataset was
  measured, not run-to-run jitter on one dataset.

**The new noise floor.** The old 8-run, median-of-7 `knn_fixed3` sweep
(0.956–1.192, `docs/EXPERIMENTS.md` "M2.5 task 1") is superseded by an
n=100-per-side, 4-session characterization: session medians **1.030–1.067**
(mean 1.0475, session-to-session sd 0.019, n=4), with each session's own
per-repetition ratio estimated (delta-method, from the published
mean/std at n=100/side) at σ≈0.03–0.05 — a **conservative envelope of
0.94–1.16** (extreme session medians ± 2×max per-session σ, not a grand
mean±2σ over a pooled distribution), comfortably inside the 1.25 gate margin and narrower on both
ends than the old eyeballed range, now with an actual computed σ behind
it rather than 8 point values read by eye. Full derivation, the
delta-method formula, and the caveat about `measure_pair`'s fixed
rust-then-cpp interleave order (a residual, accepted, non-zero risk of
one-sided bias): `docs/EXPERIMENTS.md` "M2.6 task 2", "The new noise
floor" sub-subsection. Every "noise floor" citation elsewhere in this
file, `README.md`, and `docs/ROADMAP.md` now points here; the old figure
is kept as historical text, not deleted. **Update (M2.6 task 7):
re-confirmed on the idle host** — 4 fresh, loadavg-bracketed
`knn_fixed3` sessions (run to check whether the game-load window that
motivated task 6's measurement-conditions protocol had inflated this
figure) compute their own conservative envelope at 0.884–1.176, slightly
*wider* than 0.94–1.16, not narrower — no evidence of inflation.
0.94–1.16 stands unchanged as the canonical floor
(`docs/EXPERIMENTS.md` "M2.6 task 7").

**Update (M2.6 task 5, commits `3d64c8b`/`a68df86`, 2026-08-24):** task 4's
line-by-line C++ fidelity audit (diagnosis only, no code changed) found
`dyn_add_20k_dim3_f32` to be the one gate with a real, non-noise gap, and
identified two places `DynamicKdTree::add_points`/`compute_min_max`
diverged from `nanoflann.hpp` 1.12.1. Task 5 (this update) implemented
both, exactly matching the C++: (1) the merge loop's slot `vind` now
mirrors C++'s `vAcc_.clear()` — the allocation is retained across a merge
instead of being dropped and regrown from capacity 0 every time; (2)
`compute_min_max` now uses the same 4-wide unrolled scan `middleSplit_`
actually runs (`UNROLL=4` in `nanoflann.hpp:1510-1530`), not the separate
plain-loop `computeMinMax` the port had used instead. Both preserve
bit-exact parity (min/max and merge order are unaffected; the xval
dynamic-parity suite, incl. both mutation canaries, stayed green
throughout). Full A/B evidence (8 pasted sessions, incl. 4 with
`/proc/loadavg` captured on a user-confirmed-idle host):
`docs/EXPERIMENTS.md` "M2.6 task 5" subsection.

| Workload | M2.6 task 2 (4 sessions) | M2.6 task 5, both fixes landed (4 sessions, 2 idle-host-confirmed) |
|---|---|---|
| `dyn_add_20k_dim3_f32` | 1.112–1.149 | **1.034–1.038** |

No other gate moved outside its M2.6 task 2 recorded band (`build_100k`
stayed ≈1.0 under the unroll fix, as task 4 predicted — the 4-wide scan
only matters for the many small dyn-forest rebuild splits, not the one
wide 100k build). Conclusion (f) above (`dyn_add` "1.11–1.24", previously
CONFIRMED/narrowed at M2.6 task 2) is now **IMPROVED BY A LANDED FIX**:
the residual this repo characterized as noise-adjacent is, for
`dyn_add`, substantially closed by a real C++-fidelity correction, not
just re-measured. `build_100k`'s own configuration-split finding
(conclusion (g)) is unaffected by this update and is left for Task 6 per
the M2.6 controller's scoping.

### M2.6 task 6 — final regression sweep, `build_100k` re-hedge, and success-criteria verdict

Final task of the M2.6 milestone: a fresh regression sweep (full gates x2
sessions + report chain x1, all idle-host with `/proc/loadavg` captured
before/after — the "Measurement-conditions protocol" this task adds to
`docs/EXPERIMENTS.md` §1), a full Python bench re-run resolving the one
outstanding flagged outlier, and a docs close-out. No library code
changed. Full pasted evidence: `docs/EXPERIMENTS.md` "M2.6 task 6"
subsection.

**Final six-gate table** (mean ± std + median, three fresh sessions: 2
gate + 1 report-chain, commit `b9335fd`):

| Workload | Session 1 (gate) | Session 2 (gate) | Session 3 (report chain) | Combined range |
|---|---|---|---|---|
| `build_100k_dim3_f32_seq` | 1.009 (rust 9.849±0.316 / cpp 9.760±0.306, med 9.757/9.666) | 1.009 (rust 9.658±0.195 / cpp 9.561±0.127, med 9.613/9.530) | 0.9926 (rust 9.680±0.365 / cpp 9.734±0.186, med 9.609/9.680) | **0.993–1.009** |
| `knn_fixed3_dim3_f32_k10` | 1.040 (rust 7.505±0.190 / cpp 7.221±0.141, med 7.476/7.189) | 1.040 (rust 7.527±0.264 / cpp 7.210±0.213, med 7.424/7.141) | 1.0113 (rust 7.279±0.260 / cpp 7.201±0.292, med 7.201/7.121) | **1.011–1.040** |
| `knn_dyn_dim8_f64_k10` | 0.938 (rust 174.873±6.634 / cpp 185.849±4.531, med 173.679/185.132) | 0.941 (rust 175.348±8.967 / cpp 185.374±6.122, med 172.707/183.537) | 0.9270 (rust 182.091±9.181 / cpp 195.163±4.858, med 180.027/194.198) | **0.927–0.941** |
| `radius_dim3_f32` | 0.846 (rust 5.367±0.292 / cpp 6.347±0.352, med 5.285/6.249) | 0.831 (rust 5.220±0.130 / cpp 6.289±0.155, med 5.188/6.242) | 0.8680 (rust 5.684±0.374 / cpp 6.524±0.356, med 5.554/6.399) | **0.831–0.868** |
| `dyn_add_20k_dim3_f32` | 1.034 (rust 3.642±0.091 / cpp 3.516±0.086, med 3.621/3.502) | 1.039 (rust 3.625±0.095 / cpp 3.489±0.075, med 3.608/3.474) | 1.0315 (rust 3.870±0.604 / cpp 3.760±0.587, med 3.633/3.523) | **1.0315–1.039** |
| `dyn_knn_after_churn_dim3_f32` | 0.931 (rust 30.203±0.472 / cpp 32.419±0.575, med 30.065/32.292) | 0.937 (rust 29.915±0.890 / cpp 31.934±0.638, med 29.796/31.801) | 0.9584 (rust 29.742±0.507 / cpp 31.005±0.427, med 29.643/30.929) | **0.931–0.958** |

All six pass the ≤1.25 gate margin, every session. `dyn_add`'s task-6
range (1.0315–1.039) confirms the task 5 fidelity fix (1.034–1.038) holds
on a third, independently-measured idle-host sweep — the landed correction
is durable, not a one-session artifact. `build_1M_dim3_f32_seq`/`_par`
(report-chain session only, not gated): **0.9897** (med 122.803/124.082,
rust 124.018±5.147, cpp 124.837±3.238) / **0.6151** (med 27.590/44.857,
rust 27.976±1.624, cpp 45.830±4.122).

**`build_100k` re-hedge (T2 conclusion (g), corrected).** T2's "reproducibly
split by configuration" claim (gate dataset ~0.967 vs. report-chain
dataset ~1.037–1.039) does not hold up: T4 (`docs/reports/m2.6/task-4-report.md` §2)
independently re-ran the identical unpatched code/seeds and got
0.9825/0.984 — no trace of the old split. This task's own three fresh
sessions corroborate that finding a second, independent way: the table
above shows both gate sessions at 1.009/1.009 and the report-chain session
at 0.9926 — **the opposite ordering** from T2's original split (T2's gate
config was the Rust-winning one; here it's marginally the C++-winning
one), all three clustered tightly at 0.993–1.009. **Verdict: `build_100k`
shows no reproducible configuration-dependent gap.** The original
1.037–1.039 report-chain cluster was very likely session/host-load state
— consistent with, though not independently proven to be caused by, the
background game-load incident surfaced mid-M2.6 (docs/EXPERIMENTS.md §1's
"Measurement-conditions protocol"). This task's fresh 0.993–1.009 range is
the current record for this workload; every "reproducibly split by
configuration" citation in this repo (this file, `README.md`,
`docs/ROADMAP.md`, `docs/nanoflann-notes.md`) carries an "Update (M2.6
task 6)" banner pointing here — the original text is kept, not deleted.

**dim8 hedge (deferred minor from task 5).** Task 5 flagged one session
where `knn_dyn_dim8_f64_k10` measured 0.939, just outside the M2.6 task 2
band (0.929–0.937). This task's own dim8 range across its three fresh
sessions, **0.927–0.941**, straddles that band on both sides by comparable
or larger margins — corroborating that a single-session ±0.2–0.4
percentage-point dim8 excursion is ordinary noise, not something the
"no other gate moved outside its band" line needs to claim never happens.

**Python bench outlier resolution (M2.6 task 3's flagged
`knn_batched_dim3_f32_..._workers1` cell).** A full fresh `bench_py.py`
run (idle host, loadavg captured before/after) measures this cell at
**`ratio_ckdtree`=0.688** and **`ratio_pynanoflann`=1.055** — decisively
away from the flagged session's 0.493/1.009 and back near the previously
recorded steady bands (cKDTree's absolute median, 423.19ms, lands right
back in its historical 426–430ms range; the pynanoflann ratio, 1.055,
sits essentially at the old 1.058–1.216 range's low end). **Verdict:
CONFIRMED as host-load contamination, not a new steady state** — full
detail, full result table, and every other cell's re-verification:
`docs/EXPERIMENTS.md` "M2.6 task 6" subsection, "Python bench: full
idle-host re-run and outlier resolution". Honest combined ranges going
forward: `ratio_ckdtree` **0.493–0.830**, `ratio_pynanoflann`
**1.009–1.216** (both flagged outlier points kept in the range, per this
repo's range-honesty convention, not folded in silently or deleted).

**Success-criteria verdict (M2.6 milestone, per the plan's stated goal —
statistical rigor + a fidelity audit of the losing workloads):**

| Goal | Delivered | Evidence |
|---|---|---|
| Statistical methodology (n>=10..100 adaptive, mean±std) for the Rust-vs-C++ gates/report chain | **DELIVERED** — `xval::measure_pair`/`TimingStats` (M2.6 task 1) | `docs/EXPERIMENTS.md` "M2.6 task 2" subsection; every six-gate table in this file since |
| Statistical methodology for the Python bench | **DELIVERED** — `bench_py.py`'s `timed_stats_interleaved` (M2.6 task 3), the same `measure_pair` policy generalized from 2 sides to N interleaved engines | `docs/EXPERIMENTS.md` "M2.6 task 3" subsection |
| Fidelity audit of every losing/underperforming workload, "no extra tricks, follow flann exactly" | **DELIVERED** — line-by-line audit of `build_100k`, `dyn_add`, `knn_fixed3`, dim-32/64, allocation patterns, and the Python dim8 miss against vendored nanoflann 1.12.1 (diagnosis-only, no commits — task 4) | `docs/reports/m2.6/task-4-report.md`; its two FIDELITY candidates and REJECT rationale are reproduced in `docs/EXPERIMENTS.md`'s "M2.6 task 5" subsection (landed fixes) and this file's "Honest residuals" section (REJECTs) |
| Landed fixes for real fidelity gaps found | **DELIVERED, 2 fixes** — dyn merge capacity preservation + 4-wide unrolled `compute_min_max`, `dyn_add` **1.13–1.14 → 1.03–1.04**, confirmed durable on this task's own re-sweep | commits `3d64c8b`, `a68df86`; this section's `dyn_add` row above |
| Evidence-backed REJECTs for non-fidelity residuals | **DELIVERED** — `build_100k`/`knn_fixed3` (no unported behavior, session-noise/iterative-stack cost), dim-32/64 f64 (LLVM-vs-GCC vector-width codegen, asm-verified), arena pre-reserve (measured worse), `removed`-map hasher (unmeasurable) | `docs/reports/m2.6/task-4-report.md` §3, §5, §6, §8 |
| Every number traces to a pasted, reproducible run; no chat-log-only claims | **DELIVERED** | every session in `docs/EXPERIMENTS.md`'s M2.6 subsections carries a pasted command + output |
| Flagged single-session outliers resolved or explicitly re-confirmed, not silently folded in | **DELIVERED** — Python bench `knn_batched_dim3_f32_..._workers1` cell resolved this task (CONFIRMED host-load contamination); `build_100k` re-hedge resolved this task (CONFIRMED no reproducible split) | this section, above |

## M-py — Python bindings

Same machine as every section above (WSL2, AMD Ryzen 7 9800X3D, 8 threads,
`RUSTFLAGS="-C target-cpu=native"`, bindings built
`.venv/bin/maturin develop --release`). Python env: `python 3.12.11, numpy
2.5.2, scipy 1.18.1, pynanoflann 0.10.0`. Full commands, both bench runs'
raw JSON rows, and the perf-gate re-run: `docs/EXPERIMENTS.md`'s "M-py"
subsection (§3).

### Parity — scoped per binding controller ruling (task 3, fix round 1/5)

`pynanoflann` 0.10.0 (the only PyPI release) vendors nanoflann **1.5.5**;
flannrust's own C++ xval oracle, and the kernel the Rust core is bit-matched
against, is nanoflann **1.12.1**. Between those two versions,
`L2_Adaptor`/`L1_Adaptor::evalMetric`'s partial-sum combination order
changed (sequential-chain vs. pairwise), which — floating-point addition
being commutative but not associative — produces exactly-1-ULP
squared-distance differences for any `dim >= 3`. This is a version-gap
environment artifact against this one pynanoflann build, not a flannrust
defect (verified by code diff and empirical measurement — see
`docs/EXPERIMENTS.md`). The parity claim is therefore **scoped to what
holds unconditionally**:

- **Tie-free KNN index-sequence parity**, full spec'd matrix (dims
  `{2,3,8,32}` × `{f32,f64}` × leaf `{1,10,64}` × metric `{l2,l1}` ×
  `k ∈ {1,10}`): **96/96 nodes pass, bit-exact**.
- **dim=2 KNN distance-value bit-exactness** (a pure 2-term sum, provably
  immune to the version gap): **4/4 pass**.
- 16 parametrized nodes (radius-index parity at dim ∈ {8,32}/float32, and
  tie-multiset comparison at dim ∈ {3,8,32}/float32 — the documented 1-ULP
  boundary-flip mechanism) are pinned as targeted `xfail(strict=True)`,
  not blanket-marked whole test functions, so an unexpected pass surfaces
  loudly. No tolerance was ever loosened. Combined suite: `0 failed, 244
  passed, 27 xfailed, 1 xpassed`.

### Success criteria vs. spec (honest accounting, both bench runs)

Ratio = `flannrust_ms / other_ms`; lower is better for flannrust. Two full
bench runs were captured (range honesty, not averaged) — full result-row
tables for each run: `docs/EXPERIMENTS.md`'s "M-py" subsection.

| Criterion | Gate | Range (both runs) | Verdict |
|---|---|---|---|
| Batched knn dim3 f32 vs cKDTree, workers=1 | ≤ 1.00 | 0.712–0.830 | **MET** |
| Batched knn dim3 f32 vs cKDTree, workers=−1 | ≤ 1.00 | 0.712–0.827 | **MET** |
| Build vs pynanoflann, 100k | ≤ 1.00 | 0.546–0.561 | **MET** |
| Build vs pynanoflann, 1M | ≤ 1.00 | 0.543–0.551 | **MET** |
| dim-32 knn vs pynanoflann | ≤ 1.10 | 0.878–0.899 | **MET** |
| Per-call overhead measured | — | flannrust ≈2.0–2.3µs, cKDTree ≈7.2–7.5µs, pynanoflann ≈2.3µs | **MET** (measured, not gated) |

All five gated criteria pass, with margin, in both runs independently — not
just on average.

**Update (M2.6 task 3, single n=10..100-per-cell run, 2026-08-25):** `bench_py.py` gained the same adaptive-repetition
`TimingStats` (mean/std/median/min/max/n) upgrade M2.6 task 1 gave the
Rust-vs-C++ side, generalized from 2 sides to N interleaved engines — one
fresh full run (not two; the `n>=10` adaptive sampling is itself the added
statistical weight this task contributes). Full pasted run, per-cell `n`,
and per-conclusion detail: `docs/EXPERIMENTS.md` "M2.6 task 3" subsection.

| Criterion | Gate | M-py range (2 runs, median-of-7) | M2.6 task 3 (1 run, median-of-n\*) | Verdict |
|---|---|---|---|---|
| Batched knn dim3 f32 vs cKDTree, workers=1 | ≤ 1.00 | 0.712–0.830 | **0.493** (n=35) | **MET**, single-session outlier — see caveat below |
| Batched knn dim3 f32 vs cKDTree, workers=−1 | ≤ 1.00 | 0.712–0.827 | **0.703** (n=100) | **MET** |
| Build vs pynanoflann, 100k | ≤ 1.00 | 0.546–0.561 | **0.553** (n=100) | **MET** |
| Build vs pynanoflann, 1M | ≤ 1.00 | 0.543–0.551 | **0.553** (n=100) | **MET** |
| dim-32 knn vs pynanoflann | ≤ 1.10 | 0.878–0.899 | **0.922** (n=10, clamp floor) | **MET** |
| Per-call overhead measured | — | flannrust ≈2.0–2.3µs, cKDTree ≈7.2–7.5µs, pynanoflann ≈2.3µs | flannrust **2.226µs**, cKDTree **8.361µs**, pynanoflann **2.683µs** (all n=100) | **MET** (measured, not gated) |

\* `n` is per-cell, adaptive (`clamp(10, 100, floor(30000/t_est_ms))`,
`t_est_ms` = slowest engine's own 2-warmup-rep mean) — see the table above
and `docs/EXPERIMENTS.md`'s per-cell `n` breakdown.

All five gated criteria still pass, most with wide margin. Two ranges
(`workers=-1`, `dim-32`) widen only modestly and read as ordinary
session-to-session noise. `workers=1`'s **0.493** is a genuine outlier
worth flagging, not folding silently into the range: this run's cKDTree
median for that one cell (698.9ms) is ~63% slower than any previously
recorded run (426–430ms), while flannrust's own median (344.3ms) sits
right where every prior run put it — a scheduler/thermal stall landing
disproportionately on cKDTree's share of that cell's interleaved reps, not
a flannrust speedup (this task changed only the bench harness and
renderer, not `crates/flannrust`'s query/build/kernel code). New honest
range **0.493–0.830**, published as-is per this repo's range-honesty
convention, flagged for confirmation rather than treated as the new
steady state.

**Update (M2.6 task 6, full fresh run, idle host, 2026-08-25):** the
flagged confirmation re-run happened. `workers=1` vs cKDTree now measures
**0.688** (n=69) — decisively away from the 0.493 outlier, and cKDTree's
own absolute median (423.19ms) is back in its historical 426–430ms band
(vs. the flagged session's 698.9ms). **Verdict: CONFIRMED as host-load
contamination, not a new flannrust-vs-cKDTree steady state.** New honest
range **0.493–0.830** is unchanged in its bounds (0.688 sits inside it),
but the 0.493 point is now understood as a labeled, resolved outlier
rather than an open question. Full detail: `docs/EXPERIMENTS.md` "M2.6
task 6" subsection.

### Honest misses (not gated, published alongside per this repo's standing "publish the losses too" convention)

| Workload | vs pynanoflann range | Reading |
|---|---|---|
| `knn_batched_dim3_f32_..._workers1` | 1.058–1.216 | flannrust 6–22% slower single-threaded; drops to 0.896–1.000 (near parity or better) at `workers=-1` — a single-threaded call-overhead shape, not algorithmic |
| `knn_dim8_float64_..._workers1` | 1.233–1.249 | flannrust 23–25% slower — the most consistent miss in the matrix, both runs agree closely (genuine gap, not noise) |

flannrust beats cKDTree on every single row, both runs, without exception —
these two misses are specifically against pynanoflann. The dim8 f64 gap is
notable because flannrust measures **~0.95–0.97x against its own 1.12.1 C++
oracle** at the same dim/dtype (`knn_dyn_dim8_f64_k10` perf gate, M2.5
section above) — i.e. the miss is a pynanoflann-vs-flannrust shape (likely
pynanoflann's simpler pybind11 marshalling path, or its 1.5.5 kernel), not
evidence of a flannrust regression against C++ ground truth. Flagged as an
M-pub investigation item, not chased further this milestone —
`docs/ROADMAP.md`.

**Update (M2.6 task 3):** the same n>=10 single run above re-measured both
honest misses. `knn_batched_dim3_f32_..._workers1` vs pynanoflann is now
**1.009** — near-parity, well below the old 1.058–1.216 range — but this is
the SAME cell flagged above for its cKDTree outlier, and pynanoflann's
median this session (341.3ms) is likewise somewhat above its own
historical range (289–291ms); read as the same single-session host-noise
episode, not a resolved miss. New range **1.009–1.216**, NOT declared
closed pending a confirming re-run. `knn_dim8_float64_..._workers1` vs
pynanoflann is now **1.271** — worse than the old high end, an ordinary
(non-outlier-shaped) widening; new range **1.233–1.271**, still the most
consistent miss in the matrix. Full detail: `docs/EXPERIMENTS.md` "M2.6
task 3" subsection.

**Update (M2.6 task 6, full fresh run, idle host, 2026-08-25):**
`knn_batched_dim3_f32_..._workers1` vs pynanoflann now measures **1.055**
(n=69) — solidly back inside the old 1.058–1.216 range (0.003 below its
low end, effectively at the boundary), a large step away from the
anomalous near-parity 1.009 reading. **Verdict: CONFIRMED as the same
host-load episode as the cKDTree-side outlier above, not a resolution of
this honest miss.** New range **1.009–1.216** is unchanged in its bounds;
this miss remains open, still an M-pub investigation item.
`knn_dim8_float64_..._workers1` vs pynanoflann this run: **1.227** — 0.006
below the old 1.233 low end, essentially at the boundary, still the most
consistent miss in the matrix, still not root-caused this milestone (see
`docs/ROADMAP.md`'s M-pub item, which now also carries `docs/reports/m2.6/task-4-report.md`
§7's finding that pynanoflann's own vendored nanoflann 1.5.5 kernel is
~5–6% *slower* than the 1.12.1 oracle this repo bit-matches against on
this exact workload — refuting the "1.5.5 autovectorizes faster"
hypothesis and narrowing the open question to the Python/pybind11 binding
layer specifically, not the C++ kernel). Full detail:
`docs/EXPERIMENTS.md` "M2.6 task 6" subsection.

### Rust-side perf gates, re-run for M-py (confirms the rename/additions didn't move anything)

```
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=10.848ms cpp=9.520ms ratio=1.139
PERF_GATE perf_gate_dyn_add_20k_dim3_f32: rust=3.923ms cpp=3.530ms ratio=1.111
PERF_GATE perf_gate_dyn_knn_after_churn_dim3_f32: rust=29.985ms cpp=33.286ms ratio=0.901
PERF_GATE perf_gate_knn_dim3_f32_k10: rust=7.334ms cpp=7.171ms ratio=1.023
PERF_GATE perf_gate_knn_dyn_dim8_f64_k10: rust=179.656ms cpp=187.852ms ratio=0.956
PERF_GATE perf_gate_radius_dim3_f32: rust=5.240ms cpp=6.204ms ratio=0.845
test result: ok. 6 passed; 0 failed
```
All six pass the 1.25 margin; two widen a previously-recorded range
honestly (`build_100k` new range 0.974–1.139, `radius` new range
0.827–0.845 — both ordinary WSL2 noise, not a regression, since no
perf-relevant `crates/flannrust` code changed during M-py). Full comparison
against the M2.5 ranges: `docs/EXPERIMENTS.md`'s "M-py" subsection.

**Update (M2.6):** the M-py-era `build_100k` 0.974–1.139 spread motivated a
dedicated re-verification (`docs/EXPERIMENTS.md` "M2.6 task 2" conclusion
(g)) — reproducibly split by configuration, not noise or drift, though
  dataset seed and process context were never crossed to isolate which one
  is the cause (`docs/EXPERIMENTS.md` conclusion (g)): the
`perf_gate.rs` binary's own (differently-seeded) dataset builds
consistently at ~0.967 across 2 fresh sessions, while `report_data.rs`'s
independently-seeded dataset of the same shape builds consistently at
~1.037 across 2 fresh sessions — each individually stable to <1%, but
~7 points apart from each other. `radius` similarly widens to a fresh
0.807–0.867 (conclusion (b)) rather than narrowing.

**Update (M2.6 task 6):** the "reproducibly split by configuration" finding
directly above did not hold up under further testing — T4's independent
crossing experiment and this task's own fresh 3-session sweep both fail to
reproduce it (T4: 0.9825/0.984 on identical unpatched code+seeds; this
task: 0.993–1.009 across gate AND report-chain sessions, in the opposite
order from the original split). Current verdict: `build_100k` shows no
reproducible configuration-dependent gap; the original 0.967-vs-1.037/1.039
split reads as session/host-load state. Full account: `docs/EXPERIMENTS.md`
"M2.6 task 6" subsection, "`build_100k` re-hedge"; this file's own "M2.6
task 6" section above.

### Test status (M-py, T6 final sweep — commit `0cea30c` + T6's own changes)

`cargo test --workspace` (workspace now includes `crates/flannrust-py` —
compiled and its 0-Rust-unit-test binary run, since `flannrust-py`'s own
correctness suite is entirely pytest-based, not `#[test]`): **374 passed, 0
failed, 15 ignored** across all binaries (up from M2.5's 359/0/15 — the
Rust-side growth is M-py's own `data_source.rs`/`OwnedRows` unit tests plus
T6's one new `tree::tests::dataset_returns_the_built_over_data_source`; the
ignored count is unchanged at 15). `cargo clippy --workspace --all-targets
-- -D warnings`: clean. `RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust
--no-deps`: clean. `cargo build -p flannrust --no-default-features`: clean
build; `cargo test -p flannrust --no-default-features`: **194 passed, 0
failed, 3 ignored** + 2 doctests. `.venv/bin/python -m pytest python/tests
-q` (`crates/flannrust-py`): **348 passed, 0 failed, 27 xfailed, 1 xpassed**
(348 = 346 at T5's close-out + 2 new dtype-parametrized cases from T6's
`query_radius` workers-determinism test; the 27 xfailed / 1 xpassed are
unchanged from T3's fix round — the pynanoflann version-gap parity nodes
discussed above).

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
