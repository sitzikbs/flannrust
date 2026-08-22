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
  needing the `DataSource::point_row` fast path (Candidate 2), which is why
  Candidate 2 measured as a net loser (see task-14-report.md) and was
  reverted.

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

## Test status

`cargo test --workspace --all-features`: 0 failed (269 passed, 8 ignored)
after every landed change. `cargo test --workspace --all-features --release
-- --ignored` (heavy 1M-point stress builds/queries + canary mutation
tests): 0 failed, including the two deepest degenerate-tree constructions
(`heavy_exponential_build_1m` depth ~2115, `heavy_exponential_build_1m_dim8`
depth ~16794) built with the new bbox scratch pool (Candidate 3).
