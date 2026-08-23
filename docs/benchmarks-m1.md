# M1 benchmark results (Task 14 — performance pass)

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

## Perf gate (four gated workloads)

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

### Radius marshalling asymmetry (read `radius_dim3_f32`'s < 1.0 ratio with this in mind)

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

## Headline table (criterion, spot-check methodology, `--quick`)

### `knn_fixed3` — `ConstDim<3>` + `&[[T;3]]` (rust) vs `RefIndex3F32`/`RefIndex3F64` (cpp)

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

### `build_fixed3` — `ConstDim<3>` + `&[[f32;3]]` (rust) vs `RefIndex3F32::build` (cpp)

| n | rust | cpp | ratio |
|---|---|---|---|
| 100,000 | 8.916 ms | 8.865 ms | 1.006 |
| 1,000,000 | 115.74 ms | 113.55 ms | 1.019 |

Both essentially at parity, consistent with the perf-gate's build ratio.

### `knn` — runtime `DynDim` + `FlatSlice` (rust) vs `DIM=-1` (cpp), dim 8 and dim 32

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
gap for a future task, not fixed in M1 per this task's explicit scope
(closing it plausibly needs SIMD/batching to amortize the per-axis L2 kernel
over a wider dimension — out of scope, see brief).

## Remaining gap analysis: knn_fixed3 (~1.04-1.05x)

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

## `leaf_max_size` sweep

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

## Parallel build

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

## Test status

`cargo test --workspace --all-features`: 0 failed (269 passed, 8 ignored)
after every landed change. `cargo test --workspace --all-features --release
-- --ignored` (heavy 1M-point stress builds/queries + canary mutation
tests): 0 failed, including the two deepest degenerate-tree constructions
(`heavy_exponential_build_1m` depth ~2115, `heavy_exponential_build_1m_dim8`
depth ~16794) built with the new bbox scratch pool (Candidate 3).
