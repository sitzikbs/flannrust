# Spec: SIMD L2 distance kernel

## Goal

Close the LLVM-vs-GCC codegen gap on L2 kernels for dim ≥ 8 f32/f64.
LLVM 22 emits scalar/ymm-only code for the exact-order reduction
(`perf-opportunities.md` header: zmm 0→0 under every toolchain knob);
GCC widens it to zmm via `vpermt2pd`. Measured gap: dim-32/64 knn
1.08–1.24 vs C++; dim-8 ~2–3% end-to-end.

## Scope

1. **L2Fma** (`metric.rs:367-403`) — primary. Already documented
   non-bit-exact, so any lane order is legal: plain vertical FMA
   accumulators + one horizontal reduce at the end. Simplest, biggest win.
2. **L2** (`l2_eval_row`, `metric.rs:246-278`) — optional second phase,
   bit-exact required. Lanewise `diff*diff` is order-free; only the
   accumulator chain is order-sensitive. The per-chunk sum
   `(d0²+d1²)+(d2²+d3²)` is a balanced 4-lane tree — reproducible exactly
   with two shuffle+add steps per 4-lane group, then one scalar add into
   `result`. Existing bit-equality tests gate it.

Not in scope: L1, ARM NEON, non-x86, `portable_simd`, nightly, external
SIMD crates.

## Tiers and dispatch

- **AVX2+FMA**: 4×f64 / 8×f32 per vector — wins at dim ≤ 16 (cheaper
  horizontal reduce).
- **AVX-512F**: 8×f64 / 16×f32 — wins at dim ≥ 32.
- **Scalar fallback**: current code, unchanged, always compiled.

Kernels are `#[target_feature(enable = "avx2,fma")]` /
`"avx512f"` unsafe fns over two `&[T]` slices (query, row) + dim —
`std::arch` intrinsics, stable Rust. Dispatch inside the row fast path
only (the `point_row` branch; the `point_component` fallback stays
scalar):

- Native builds: `cfg!(target_feature = "avx512f")` etc. — branch folds
  away at compile time.
- Generic builds: `is_x86_feature_detected!` — std caches the cpuid in an
  atomic, so per-call cost is one predictable load+branch; no fn-pointer
  table needed.
- Dim-gate: SIMD only when `dim >= 8`; below that the scalar path stays
  (prior data: SIMD buys ~nothing at dim ≤ 8 f64, and must not regress
  knn_fixed3/dim-2/3).

## Constraints

- Stable Rust 1.98, no new dependencies, no cargo features for the
  bit-exact L2 path (it is exact by construction).
- All existing tests pass, especially
  `l2_eval_row_path_bit_equals_fallback_path_all_dims_{f32,f64}` and the
  xval crossing-matrix suite.
- `unsafe` confined to the kernel fns; callers prove slice lengths
  (`row.len() >= dim && query.len() >= dim` already checked at the call
  site, `metric.rs:293-297`).

# Plan

## Task 1 — L2Fma AVX-512/AVX2 kernels (TDD)

1. RED: in `metric.rs` tests, property test: SIMD kernel vs scalar
   `mul_add` loop within 4·EPS·max(1,|l2|) over `BIT_EQ_DIMS` ×
   `SALTS_F64/F32` (reuse `gen_random_ish_*`).
2. GREEN: add `crates/flannrust/src/simd.rs` with
   `l2fma_{f32,f64}_{avx2,avx512}` (`_mm512_fmadd_pd` etc., 2
   accumulators to hide FMA latency, masked/scalar tail). Wire into
   `impl_l2_fma!`'s row branch (`metric.rs:381-389`) behind the dim-gate
   + detection above.
3. Verify: `cargo test -p flannrust` (debug + release, and once with
   `RUSTFLAGS="-C target-cpu=x86-64"` to exercise the runtime-detect arm).

## Task 2 — bit-exact L2 kernel (only if Task 1's numbers justify it)

1. RED: the existing bit-equality tests already discriminate (proven
   against two sabotage mutations — see `metric.rs:733-743` comment).
   Add one asm-probe assertion via `crates/xval/examples/m25_asm.rs`
   (zmm count > 0 on `probe_lib_dyndim32_f64`).
2. GREEN: `l2_avx512_{f32,f64}` doing vectorized diff/square + the exact
   shuffle-tree reduction; wire into `l2_eval_row`. If the shuffle tax
   eats the win at dim 8–16, gate to dim ≥ 32 only.

## Task 3 — benchmark A/B

`RUSTFLAGS="-C target-cpu=native" cargo bench -p xval --bench bench_knn`
on idle host; compare `knn/rust/{8,16,32}/{f32,f64}/k*` medians vs cpp
and vs pre-change baseline. Kill-criterion: any regression on
`knn_fixed3` or dim-2/3 rows reverts the dim-gate threshold upward.
Then full xval suite as the correctness gate.

## Other findings from this investigation

- **Opp 5 (hybrid recursion) is already done** — `search_level_hybrid`,
  `search.rs:402`, depth limit 96 at `:157`. `perf-opportunities.md`
  lists it as open; update that doc.
- `point_row` fast path IS taken for all common sources (`FlatSlice`,
  `OwnedRows`, `&[[T;N]]` — `data_source.rs:84,127`); per-eval cost is
  one `Option` + two len compares. Hoisting it out of the leaf loop
  (`search.rs:426-430`) is possible but likely sub-1%.
- **Leaf prefetch** (opp 6) still open and cheap: `_mm_prefetch` the
  next `vind` row in the leaf scans at `search.rs:426` and `:588` —
  natural rider on this branch's A/B harness.
- Per-query `dists_scratch` heap alloc for DynDim (`tree.rs:540`) —
  parity with C++, but a future batch API can hoist it (opp 7).
