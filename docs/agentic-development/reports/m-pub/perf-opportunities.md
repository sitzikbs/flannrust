# Performance-opportunity review (beyond-fidelity round)

Date: 2026-09-02. Branch `m2p6-rigor` (read-only review; nothing committed).
Scope: speedups that PRESERVE bit-exact results (xval is the gate). Changes
that alter float results (summation order, FMA) are listed only as opt-in
features. Baseline ratios (Rust/C++ medians, n=100): build_100k 0.999,
knn_fixed3 1.013, dyn_dim8 0.927, radius 0.876, build_1M_seq 0.994,
build_1M_par 0.604, dyn_add 1.025, dyn_churn 0.953. Known open gaps:
dim-32/64 knn 1.08–1.24 (LLVM ymm vs GCC zmm), Python dim8-f64 vs
pynanoflann ~1.16–1.23.

New measurement made this session (compile-only, no benchmark run):
`RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit"` and
`-C llvm-args=-mprefer-vector-width=512` (rejected by LLVM 22; suggestion
`--force-vector-width` targets the loop vectorizer, not this SLP shape)
were A/B'd on `crates/xval/examples/m25_asm.rs`'s
`probe_lib_dyndim32_f64`: **zmm count 0 → 0, ymm 20 → 20**. Host is a Zen 5
(9800X3D, full-width 512-bit datapath, `avx512f` enabled under
`target-cpu=native`, rustc 1.98.0 / LLVM 22.1.8). Conclusion: **the
toolchain knob alone does not unlock AVX-512 for the exact-order L2
kernel** — LLVM will not do the shuffle-based order-preserving widening
GCC does (task-4-report §6's `vpermt2pd` zmm evidence). Closing the
dim-32/64 gap requires a hand-written kernel (opportunity 1).

---

## Ranked opportunities

### 1. Explicit-SIMD L2/L1 row kernel with order-preserving reduction

- **What/where**: `crates/flannrust/src/metric.rs:233-265` (`l2_eval_row`),
  `:134-167` (`l1_eval_row`). Add an `#[cfg(target_feature = "avx512f")]`
  (and an AVX2 tier) intrinsics or `portable_simd` path selected at build
  time, falling back to the current scalar-shaped code.
- **Mechanism**: the per-chunk `diff*diff` lanes are independent FP ops —
  batching them into 512-bit multiplies changes nothing about any single
  operation's operands or rounding. Only the loop-carried `result +=
  (d0²+d1²)+(d2²+d3²)` chain is order-sensitive; keep that reduction as the
  exact same scalar/extract-lane add sequence. This is precisely what g++ 13
  emits for the identical C++ source (`vpermt2pd`/zmm, task-4-report §6),
  and why GCC wins dims ≥ 32. LLVM verifiably will not do it on its own
  (measurement above), so hand-rolling is the only route.
- **Bit-exact**: yes, by construction (same multiplies, same add order);
  xval + the existing `l2_eval_row_path_bit_equals_fallback_path_all_dims`
  tests gate it.
- **Rows moved**: dim-32/64 knn (1.08–1.24 → target ≤1.05); plausibly part
  of the Python dim8-f64 gap (pynanoflann is GCC-compiled); small or nil
  effect at dim ≤ 8 f32 (guard with A/Bs on knn_fixed3/dyn_dim8 — the
  kernel must not regress low dims; keep the dispatch dim-gated if needed).
- **Risk/effort**: medium — `unsafe` intrinsics (or nightly-free
  `core::arch`), per-arch gating, must A/B all dims. Well-contained: one
  function, one call site.
- **Branch**: `perf/simd-l2-kernel`.

### 2. Python binding: zero-copy query fast path + per-call overhead purge

- **What/where**: `crates/flannrust-py/src/convert.rs:25-49` — `as_rows_2d`
  copies EVERY query array element-by-element through ndarray iterators
  (`row.iter().copied()` per row) on every call, even for C-contiguous
  matching-dtype input. `convert.rs:15-18` — `to_ndarray` runs
  `PyModule::import(py, "numpy")` + a Python-level `asarray` call on every
  query. `static_tree.rs:326` / `:386` — `workers>1` builds a **fresh rayon
  ThreadPool per call** (thread spawn/teardown per query call).
- **Mechanism**: fast path — if the input already casts to
  `PyReadonlyArray{1,2}` and `.as_slice()` succeeds (C-contiguous), borrow
  the buffer for the duration of the (GIL-released) search instead of
  copying; skip `asarray` entirely when the cast succeeds; cache the numpy
  module in a `GILOnceCell` for the fallback; cache/reuse the sized thread
  pool (keyed by worker count) on the `KDTree` object. Non-contiguous or
  list input keeps the copying path.
- **Bit-exact**: yes — identical bytes reach the identical kernel.
  (Borrowing while detached from the GIL is sound via `PyReadonlyArray`'s
  lifetime + numpy's readonly guard; needs the usual pyo3 care.)
- **Rows moved**: `single_query_loop_dim3` (the overhead-bound row —
  largest relative win, plausibly 2×+ on the flannrust side of that row);
  `knn_batched_dim3 workers1` (1.06–1.22 recorded) by a few %;
  `knn_dim8_f64` partially. Note: the raw copy on the dim8 row is ~1–3 ms
  of a 392 ms median, so this CANNOT be the whole 16–23% dim8 story —
  branch should start with a py-spy/perf profile; the remainder likely
  overlaps opportunity 1 (pynanoflann = GCC codegen).
- **Risk/effort**: low-medium; pure binding-layer, parity suite +
  pynanoflann cross-check already gate it.
- **Branch**: `perf/py-zero-copy`.

### 3. Opt-in leaf-contiguous dataset reorder

- **What/where**: new builder option (e.g. `.reorder_points(true)`) in
  `crates/flannrust/src/tree.rs`; after build, copy the dataset rows into
  `vind` order so every leaf scan (`search.rs:489-504`) reads CONTIGUOUS
  memory instead of gathering rows through the permutation; keep the
  original-index map for reporting results.
- **Mechanism**: the leaf loop is the hot loop; today each `accessor` is a
  random row gather (dataset 100k x dim easily exceeds L2, so most leaf
  points are cache misses). Contiguous leaves turn ~leaf_size misses into
  ~1–2 line fetches + hardware prefetch. This is what FLANN's original
  `reorder` and scipy's `compact_nodes` do; nanoflann never did, so it is a
  genuine beyond-fidelity lever with no C++ counterpart to lose to.
- **Bit-exact**: yes — identical coordinate values feed identical
  arithmetic in identical order; only addresses change. Reported indices
  mapped back verbatim. xval-gateable directly (opt-in flag defaults off,
  so the default configuration stays byte-identical in behavior AND
  memory).
- **Rows moved**: all knn/radius query rows, growing with dim and n —
  plausibly 5–20% on dim8–64 queries; nothing on build (build pays a +1
  dataset copy; opt-in mitigates the 2x memory cost).
- **Risk/effort**: medium — touches `DataSource` plumbing (needs an owned,
  reordered copy; simplest for `OwnedRows`-backed trees, i.e. the Python
  path, and an owned-dataset Rust path). Feature-flag/API design is most
  of the work.
- **Branch**: `perf/leaf-reorder`.

### 4. PGO (+ release-profile odds and ends)

- **What/where**: toolchain only. Profile already has `codegen-units = 1`,
  `lto = "thin"` (workspace `Cargo.toml` — no cheap wins left there).
  Remaining knobs: `cargo-pgo` (instrument → run `report_data`-style
  training load → rebuild), an `lto = "fat"` A/B, and `panic = "abort"`
  for the bench/bin profile only (NOT the pyo3 cdylib — it needs unwind
  for PyErr).
- **Mechanism**: `search_level` and the sorted-insert are branch-heavy with
  data-dependent, but heavily-skewed, branches (leaf-vs-interior, prune
  test, phase). PGO gets branch layout + inlining right; typical 5–15% on
  tree-traversal code. Does not touch FP arithmetic.
- **Bit-exact**: yes (PGO/BOLT never change FP semantics); xval still runs
  as proof.
- **Rows moved**: all Rust rows a few %, most visible on knn_fixed3 (1.013
  → sub-1.0 plausibly) and dyn rows. Caveat: benchmarks-vs-C++ fairness —
  the C++ oracle is not PGO'd, so PGO numbers must be reported as a
  separate profile, and shipped only in wheels (crates.io users compile
  themselves).
- **Risk/effort**: low risk, medium harness effort (training-run plumbing,
  CI story for wheels).
- **Branch**: `perf/pgo`.

### 5. Hybrid recursion in `search_level` (win back the M2.5 give-backs)

- **What/where**: `crates/flannrust/src/search.rs:440-577`. The M2.5
  explicit-stack conversion is a recorded trade: knn_fixed3 improved
  (1.052 → 1.021) but radius +5.3% and dyn-dim8 +6.8% regressed, and
  m2.5/task-2-report §6 proved the loss is intrinsic to the store/reload
  explicit-stack shape (capacity tuning does NOT recover it).
- **Mechanism**: native recursion keeps `dst`/`cut_dist`/`mindist` in
  registers. Hybrid: recurse natively with a depth counter; at depth ≥ D
  (e.g. 96) switch that subtree to the existing explicit-stack walker —
  degenerate-tree safety preserved, realistic trees (depth ~13–17) pay
  zero explicit-stack cost. Alternative: `stacker::maybe_grow` (adds a
  dependency; the ladder says prefer the in-tree hybrid).
- **Bit-exact**: yes — identical traversal order and arithmetic either way.
- **Rows moved**: radius (0.876 → ~0.83), dyn_dim8 (0.927 → ~0.87) per the
  measured M2.5 deltas; MUST A/B knn_fixed3, which the explicit stack
  helped — if recursion costs knn back its 3pp, the hybrid may be a wash;
  this is an experiment with a known kill-criterion, not a sure win.
- **Risk/effort**: medium — two search-body variants to keep provably
  identical (share the leaf/prune code), degenerate-tree tests already
  exist to gate the spill handoff.
- **Branch**: `perf/search-recursion-hybrid`.

### 6. Software prefetch in the leaf scan

- **What/where**: `search.rs:489-504` — before computing `eval` for
  `vind[i]`, issue `_mm_prefetch` for the row of `vind[i+1]` (and possibly
  the `other_child` node at `search.rs:514`).
- **Mechanism**: overlaps the gather-miss latency of the next leaf point
  with the current distance computation. Cheap-shot version of opportunity
  3; worthwhile on its own only if 3 is not taken (they solve the same
  misses — measure 3 first, or this first as the 1-hour probe).
- **Bit-exact**: yes (prefetch is a hint, no semantic effect).
- **Rows moved**: knn dim ≥ 8 rows, low-single-digit %.
- **Risk/effort**: very low; one intrinsic + A/B.
- **Branch**: `perf/leaf-prefetch` (or fold into `perf/leaf-reorder` as
  its control arm).

### 7. Batch / parallel query API in the core crate

- **What/where**: `tree.rs` — the crate has NO batch or rayon query API;
  the Python layer hand-rolls one (`static_tree.rs:311-333`), Rust users
  cannot. Add `knn_search_batch(&self, queries, k, out_i, out_d, workers)`
  writing disjoint output chunks (rayon behind the existing `parallel`
  feature); reuse one `dists_scratch` per worker while at it — DynDim
  currently heap-allocates scratch **per query** (`tree.rs:540` via
  `dim.rs:46`; C++ does the same, so it's parity today, but a batch API can
  legitimately hoist it).
- **Bit-exact**: yes — per-query results independent; output placement
  fixed by chunk index.
- **Rows moved**: none of the current gates (they're single-thread by
  design) — this is API/user-facing value plus enabling a fair
  parallel-query benchmark row later.
- **Risk/effort**: low.
- **Branch**: `feat/parallel-query-api`.

### 8. Opt-in fast kernels feature (results-changing — opt-in ONLY)

- **What/where**: `metric.rs` behind a `fast-math`-style cargo feature
  (default off): allow FMA contraction and/or a wider reassociated
  reduction in `l2_eval_row`.
- **Mechanism**: task-4-report §7 measured FMA alone buying the C++ side
  ~4% on dim8-f64; reassociation unlocks free 512-bit reductions (no
  shuffle dance needed, subsuming opportunity 1's hard part).
- **Bit-exact**: **NO — results-changing, opt-in only**, per the review
  constraint. Must never be default; xval runs only against the default
  feature set; document the deviation loudly.
- **Rows moved**: dim ≥ 8 kernels 5–15% for users who opt in; no default
  benchmark row moves.
- **Risk/effort**: low code risk, medium docs/positioning risk (the
  project's headline claim is bit-exactness).
- **Branch**: `perf/opt-in-fast-kernels`.

---

## Anti-opportunities (tempting, but tested-and-rejected or barred)

| Idea | Why not | Evidence |
|---|---|---|
| RUSTFLAGS 512-bit knobs (`-prefer-256-bit` off, `llvm-args` width prefs) | Measured this session: zmm 0→0 on the dim32-f64 probe; LLVM 22 won't widen the exact-order kernel | this report, header |
| Arena `Vec::with_capacity` pre-reserve in build | Measured worse twice (+3–6% rust median) | m2.6/task-4-report §5 |
| `get_unchecked` in `KnnResultSet::worst_dist` / sorted insert | Bounded at ≤0.5%, not where time goes | m2.5/task-1-report §6a |
| Plain-array `FrameStack` (no `MaybeUninit`), or tuning `SEARCH_STACK_INLINE_CAPACITY` | 4KiB memset/query regression (1.06→1.14); capacity 32 does NOT recover the radius/dyn-dim8 losses | m2.5/task-2-report §6a, search.rs:137-185 doc |
| Chasing knn_fixed3's residual 1.01–1.07 | No unported behavior; residual is session noise + the recorded stack trade (see opp. 5 for the only real lever) | m2.6/task-4-report §3 |
| build_100k / build_1M seq work | No real gap — Rust won all five crossing-matrix cells | m2.6/task-4-report §2 |
| `removed`-map hasher swap in dynamic tree | Unmeasurable (empty-map short-circuit); tombstone gate already a Rust win | m2.6/task-4-report §4 |
| Adopting nanoflann 1.5.5 kernel shapes | 1.5.5 measured 5–6% SLOWER than 1.12.1, and breaks parity | m2.6/task-4-report §7 |
| Changing default `leaf_max_size` / split rule | Different tree ⇒ different result ordering vs the C++ oracle ⇒ xval breaks; not a code-speed lever | parity contract |
| FMA / reassociation by default | Results-changing; only as opportunity 8's opt-in feature | review constraint |
| Release-profile knobs (`codegen-units`, thin LTO) | Already set optimally in workspace `Cargo.toml` (codegen-units=1, lto="thin"); only the fat-LTO/PGO A/B of opp. 4 remains | Cargo.toml |

## Suggested execution order

1 and 2 first (independent, both attack the two known open gaps, both
low-regression-risk). 4 alongside (pure toolchain). Then 3 (biggest
potential, most design work), with 6 as its cheap control probe. 5 only
with its kill-criterion enforced. 7 anytime (feature work). 8 last, as a
positioning decision more than an engineering one.
