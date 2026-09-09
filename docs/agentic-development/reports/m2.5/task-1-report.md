# M2.5 Task 1 — Performance diagnosis (measurement report)

Branch `m2p5-perf`. **No library changes landed.** `crates/nanoflann-rs/src/*`
and `crates/nanoflann-ref/*` are byte-identical to `41abdb6` at the end of this
task (`git status` shows only the two new xval diagnostic examples, committed as
`chore(xval): M2.5 diagnostic probes`).

Environment: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), AMD Ryzen 7 9800X3D
(Zen 5, AVX-512), rustc 1.98.0, g++ 13.3.0. Everything below is
`RUSTFLAGS="-C target-cpu=native"` + `--release` (workspace profile
`codegen-units=1`, `lto="thin"`), C++ oracle `-O3 -march=native
-ffp-contract=off`. Every timing uses the perf-gate methodology
(`xval::timed_median_ms`: one untimed warmup + **median of 7** timed runs).

## Diagnostic tooling added (committed)

- `crates/xval/examples/m25_diag.rs` — subcommands `kernel`, `leafn`, `count`,
  `sweep`, `knn`, `dump`.
- `crates/xval/examples/m25_asm.rs` — `#[no_mangle] #[inline(never)]` probes
  that force the monomorphizations whose asm is quoted below.

Scratchpad-only (not in the repo, not committed): a **copy** of
`nanoflann.hpp` with visit counters, plus three standalone C++ probes
(`kernel_probe.cpp`, `count_probe.cpp`, `sweep_count.cpp`, `sl3_probe.cpp`) in
`/tmp/claude-1000/.../scratchpad`. The vendored header itself was never
modified.

---

## 1. Answer table

| # | Question | Command | Measurement / asm | Conclusion |
|---|---|---|---|---|
| 1 | Dim-32: kernel or traversal? | `cargo run -p xval --release --example m25_diag -- leafn` | `leaf_max_size = n` (single-leaf tree ⇒ the query is a pure linear kernel sweep), n=100k, 200 queries, k=10: **dim32 f32 rust 164.203 ms / cpp 84.260 ms = 1.949**; dim32 f64 166.156/109.463 = 1.518; dim64 f32 309.076/136.976 = **2.256**; dim16 f32 89.048/79.371 = 1.122; dim8 f32 54.478/45.917 = 1.186 | **The gap is the L2 kernel, not traversal.** With traversal removed entirely the Rust deficit is *larger* (1.95×) than the end-to-end dim-32 gap (1.48×). Corroborated by Q4: at dim 32 the search evaluates **100.0 %** of all points, i.e. the dim-32 workload already *is* a linear kernel sweep. Note Rust f32 ≈ Rust f64 (164 vs 166 ms) — the signature of a non-vectorised kernel; C++ f32 is 1.30× faster than C++ f64, i.e. C++ *is* getting vector width out of f32. |
| 2 | Vectorization state (Rust) | `cargo rustc -p xval --release --example m25_asm -- --emit asm`, symbol `probe_lib_dyndim32_f32` | See §2a. Hot loop `.LBB6_7` = **8 bounds-check `cmp`/`jae` pairs per 4-component step** (4 on `query[d..d+3]`, 4 on `FlatSlice::point_component`), then mostly *scalar* `vsubss`/`vmulss`/`vaddss` with only components 2–3 fused into a 64-bit `vmovsd`/`vsubps`/`vmulps`. No ymm, no zmm. 7 `panic_bounds_check` landing pads. | Rust's `L2::eval` is **not vectorized** at runtime dim. The per-component bounds checks split the unrolled body into 8 basic blocks, which blocks SLP vectorization. |
| 2b | Vectorization state (C++) | `g++ -std=c++17 -O3 -march=native -ffp-contract=off -I crates/nanoflann-ref/cpp -S kernel_probe.cpp`, symbol `cpp_l2_f32` | See §2b. gcc emits a **zmm (AVX-512, 64 components/iteration)** main loop and a **ymm (AVX2, 32 components)** path (`.L3`, taken for `size < 60`, i.e. the dim-32 case), using `vpermt2ps` de-interleaving to preserve the exact `(d0²+d1²)+(d2²+d3²)` order, followed by an in-order `vaddss` horizontal chain. Zero bounds checks. | C++ vectorizes the *same* summation order to 256/512-bit. **Bit-exact vectorization of this kernel is provably possible** — gcc already does it — so the order constraint is not what blocks Rust. |
| 3 | Access-path pricing at 8/16/32/64 | `cargo run -p xval --release --example m25_diag -- kernel` | ns/point, n=100k, f32, permuted (leaf-scan-like) access — see §3 for the full table. dim32: `lib_dyndim` 12.839 → `hand_row` (row slice, still indexed) 11.074 → `hand_row_chunks` (`chunks_exact(4)`) **7.136** → `hand_component_unchecked` (`get_unchecked`, still per-component) **6.954**. dim64: 30.5 → 28.7 → 18.0 → 17.8. dim8: 4.261 → 3.231 → 2.754 → 2.084. All variants asserted **bit-identical to `L2::eval` on all 100 000 points at every dim** (assertion inside the probe). | **A row pointer is not the lever — bounds-check elimination is.** `hand_component_unchecked` keeps the *identical* `data[idx*stride+d]` per-component access and recovers essentially the whole win; `hand_row` (row slice but still `row[d]`-indexed) recovers only ~15 %. `chunks_exact(4)`/`as_chunks::<4>()` on a row slice is the *safe* way to get there (within 3 % of the unsafe variant) and lands on C++'s number: dim32 f32 sequential — Rust chunks 4.29 ns/pt vs C++ 4.21 ns/pt (from `leafn`: 84.260 ms / 200 / 100 000). |
| 4 | Leaf/point-visit equality at dim 32 | `cargo run -p xval --release --example m25_diag -- dump $S` and `$S/count_probe $S` | n=100k, 1000 queries, k=10, leaf=10. **dim 32:** Rust `eval_calls=99 999 862`, interior=`14 460 000`; C++ `eval_calls=99 999 862`, interior=`14 460 000`, leaf nodes=`14 460 959`, `searchLevel` calls=`28 920 959`. **dim 3:** Rust eval=`65 951`, interior=`31 064`; C++ eval=`65 951`, interior=`31 064`, leaf=`9 416`, `searchLevel`=`40 480`. Result **digests bit-identical** (dim3 `0xcf55546b402f530a`, dim32 `0xb8fc714ade6bf1bc`). | **Exact equality — no contradiction.** Rust and C++ visit the identical node set and evaluate the identical point set, and return bit-identical `(index, distance)` streams. Nothing to escalate. (Rust interior counts are derived as `accum_dist` calls minus the `compute_initial_distances` contribution, counted independently from the root bbox; the two numbers match to the unit.) |
| 5 | `search_level` calls/query + per-call cost | `$S/sweep_count $S`; `cargo run -p xval --release --example m25_diag -- sweep` | Gate workload (dim 3, n=100k, leaf=10, k=10): **40.35 `search_level` calls per query** (30.91 interior + 9.44 leaf). Sweep of `rust−cpp` delta vs leaf size, least-squares on `delta = a·searchLevel + b·eval` (no intercept): **a = 0.658 ns per `search_level` call**, b = 0.073 ns per eval. At leaf=10 that attributes **26.6 ns of the 31.7 ns/query gap to call overhead**. | Consistent with "the residual is per-call overhead", but the fit is weak in the middle of the sweep (residuals up to 40 ns; the leaf 64–128 rows flip sign between repeat runs — see §5 for the noise bound). Treat 0.66 ns/call as an order-of-magnitude, not a precision figure. |
| 5b | Why is Rust's call more expensive? | asm of `search_level::<f32,ConstDim<3>,&[[f32;3]],L2,u32,KnnResultSet,AcceptAll>` vs gcc's `searchLevel<…,DIM=3>` | §5a/§5b. (i) **gcc self-inlines one level of the recursion** — its `searchLevel` body contains *three* `call` sites to itself and re-does the split test on the best child inline, so a depth-D descent costs ≈D/2 frames; LLVM emits exactly two self-calls and no inlining, so Rust pays D frames. (ii) Rust re-checks `ctx.nodes[node_idx]` (`cmpq %rdi,%rsi; jbe .LBB0_44`) and `query[idx]` (`cmpq %rbp,%rcx; jbe .LBB0_33`) on **every** call; gcc walks raw `NodePtr`/`const T*`. (iii) `dists[idx]` costs a `cmpl $3,%ebp; jae` check. Both sides push 6 callee-saved regs and spill ~5 live values around the call, so the frame cost itself is symmetric. | Root cause is **call count × (ABI frame + Rust-only bounds checks)**, not the leaf kernel (which post-M1-T14 is one bounds check + 3 `vsubss`/`vmulss`, as `benchmarks.md` records). |
| 6a | Could `worst_dist` re-read be it? | local reverted patch **P1b** (`get_unchecked` in `KnnResultSet::worst_dist` + `add_point_to_sorted`) | `worst_dist()` *is* recomputed inside the leaf loop on both sides (branch on `count<capacity`, then `dists[count-1]`) — nanoflann.hpp:334-336 does the same, so this is a faithful port, not an asymmetry. Rust adds one bounds check on `dists[count-1]`. P1b on top of P1a: gate ratios 1.032/1.000/1.020/1.035 (median 1.026) vs P1a alone 1.007/1.028/1.023/1.047 (median 1.026). | **Not where the 4 % hides — bound it at ≤0.5 %.** T2 should not chase the result set. |
| 6b | Per-query scratch init | `crates/nanoflann-rs/src/tree.rs:530` (`self.dim.filled(ZERO)`) vs nanoflann.hpp:2002-2005 | `ConstDim<N>` ⇒ `[T; N]` on the stack, zero allocation — nothing to win at dim 3. `DynDim` ⇒ `vec![ZERO; dim]`, **one heap allocation per query**; but C++'s `distance_vector_t` for `DIM=-1` is a `std::vector` `assign`-ed per query too (nanoflann.hpp:2002). | **Symmetric — not a gap.** It is a shared absolute cost worth ~30-50 ns/query on the DynDim path (≈0.3 % at dim 8, ≈0.005 % at dim 32); optional T2/T3 polish, not a parity item. |
| 6c | Leaf-loop bounds checks | asm §5a, block `.LBB0_6` | One `cmpq %rdi,%rax; jbe .LBB0_46` per point (the M1-T14 CSE result, confirmed still present), plus one on `dists[count-1]` inside `worst_dist`. | Already optimal for dim 3; nothing left here. |

---

## 2. Asm evidence

### 2a. Rust `L2::eval` at runtime dim 32 (`probe_lib_dyndim32_f32`, main loop)

```asm
.LBB6_7:
	cmpq	%rsi, %r15          # query[d]   bounds check
	jae	.LBB6_20
	leaq	(%rbx,%r15), %r10
	cmpq	%r9, %r10           # data[idx*dim+d] bounds check
	jae	.LBB6_9
	leaq	1(%r15), %r10
	cmpq	%rsi, %r10          # query[d+1]
	jae	.LBB6_28
	leaq	1(%rbx,%r15), %r10
	cmpq	%r9, %r10           # data[..+1]
	jae	.LBB6_9
	leaq	2(%r15), %r10
	cmpq	%rsi, %r10          # query[d+2]
	jae	.LBB6_28
	leaq	2(%rbx,%r15), %r10
	cmpq	%r9, %r10           # data[..+2]
	jae	.LBB6_9
	leaq	3(%r15), %r10
	cmpq	%rsi, %r10          # query[d+3]
	jae	.LBB6_28
	leaq	3(%rbx,%r15), %r10
	cmpq	%r9, %r10           # data[..+3]
	jae	.LBB6_9
	vmovss	(%rdi,%r15,4), %xmm1      # scalar loads
	vmovss	4(%rdi,%r15,4), %xmm2
	vmovss	8(%rdi,%r15,4), %xmm3
	vsubss	-8(%r14,%r15,4), %xmm1, %xmm1
	vsubss	-4(%r14,%r15,4), %xmm2, %xmm2
	vinsertps $16, 12(%rdi,%r15,4), %xmm3, %xmm3
	vmulss	%xmm1, %xmm1, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm1
	vmovsd	(%r14,%r15,4), %xmm2      # only lanes 2,3 vectorised (64-bit)
	addq	$4, %r15
	vsubps	%xmm2, %xmm3, %xmm2
	vmulps	%xmm2, %xmm2, %xmm2
	vmovshdup %xmm2, %xmm3
	vaddss	%xmm3, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%r11, %r15
	jb	.LBB6_7
```

8 branches + ~13 FP ops for 4 components. At dim 32 that is **64 bounds-check
branches per distance evaluation**.

### 2b. C++ `L2_Adaptor<float>::evalMetric` (`cpp_l2_f32`, `-O3 -march=native -ffp-contract=off`)

```asm
.L4:                                    # 64 components per iteration
	vmovups	(%rdx), %zmm1
	vmovups	(%rax), %zmm10
	...
	vpermt2ps -192(%rdx), %zmm5, %zmm1  # de-interleave d0/d1/d2/d3 lanes
	vpermt2ps -192(%rax), %zmm5, %zmm10 # so the (d0²+d1²)+(d2²+d3²) order
	...                                 # is preserved exactly
	vsubps	%zmm3, %zmm7, %zmm7
	vmulps	%zmm7, %zmm7, %zmm7
	vaddps	%zmm7, %zmm3, %zmm3
	vaddps	%zmm1, %zmm0, %zmm0
	vaddps	%zmm3, %zmm0, %zmm0
	vaddss	%xmm0, %xmm2, %xmm2         # in-order horizontal accumulate
	...
```
plus a `ymm` (AVX2) path `.L3` reached when `size < 60` — **this is the path the
dim-32 workload takes** — and a scalar 4-wide `.L9` tail. No bounds checks.

### 5a. Rust `search_level` (ConstDim<3>) — per-call overheads

```asm
_R…search_level…ConstDim…:
	pushq	%rbp ; pushq %r15 ; pushq %r14 ; pushq %r13 ; pushq %r12 ; pushq %rbx
	subq	$24, %rsp
	movq	24(%r15), %rsi
	movl	%r8d, %edi
	cmpq	%rdi, %rsi          # <-- ctx.nodes[node_idx] bounds check, EVERY call
	jbe	.LBB0_44
	...
	cmpq	%rbp, %rcx          # <-- query[idx] bounds check, every interior node
	jbe	.LBB0_33
	...
	movq	%rcx, (%rsp)        # spill query.len
	vmovss	%xmm0, 8(%rsp)      # spill mindist
	vmovss	%xmm1, 16(%rsp)
	vmovss	%xmm2, 20(%rsp)
	vmovss	%xmm1, 12(%rsp)
	callq	_R…search_level…    # best-child descent (self-call, not inlined)
	...
	cmpl	$3, %ebp            # <-- dists[idx] bounds check
	jae	.LBB0_43
.LBB0_39:
	movq	%r15, %rdi ; movq %r14, %rsi ; movq %rbx, %rdx
	movl	%r13d, %r8d ; movq %r12, %r9
	vmovss	%xmm2, (%rsp)
	callq	_R…search_level…    # other-child descent (self-call, not inlined)
```

Exactly two self-`call`s, no inlined level. Note `eps_error` was
constant-folded away by IPSCCP (`.LCPI0_0`), and `AcceptAll` is a ZST, so all
arguments fit in registers — there are **no stack-passed arguments**; the
`(%rsp)` traffic is caller-save spilling, which gcc also does.

### 5b. C++ `searchLevel<KNNResultSet>, DIM=3` — same prologue, one level self-inlined

```asm
_ZNK9nanoflann15KDTreeBaseClass…searchLevel…:
	pushq %r15 ; pushq %r14 ; pushq %r13 ; pushq %r12 ; pushq %rbp ; pushq %rbx
	subq $56, %rsp
	movq	16(%rcx), %r14
	testq	%r14, %r14          # node->child1 (raw pointer, no bounds check)
	je	.L99
	movslq	(%rcx), %r12        # divfeat
	vmovss	(%rdx,%r12), %xmm0  # vec[idx] — raw pointer, no bounds check
	...
.L24:                               # <-- SECOND level processed INLINE
	movslq	(%r11), %r9
	vmovss	(%rbx,%r9), %xmm0
	...
.L44:
	call	_ZNK…searchLevel…   # only then a real recursive call
```
`grep -c 'call .*searchLevel' sl3_probe.s` inside the function body → **3 call
sites** (vs Rust's 2), which is gcc's signature for having peeled one level of
self-recursion.

---

## 3. Kernel isolation table (excerpt; full output in `m25_diag -- kernel`)

`n = 100 000`, one query swept over every point, **ns per point**, median×7.
"perm" = points visited in a scattered order (models a real leaf scan); "seq" =
ascending index (models `leaf_max_size = n`). Every variant is asserted
**bit-identical to `L2::eval`** on all 100 000 points.

| dim | scalar | order | `lib_dyndim` | `lib_constdim` | `hand_component` | `hand_row` | `hand_row_chunks` | `hand_row_arraychunks` | `hand_component_unchecked` |
|---|---|---|---|---|---|---|---|---|---|
| 8 | f32 | seq | 2.628 | 1.263 | 2.255 | 1.977 | 1.981 | — | 1.500 |
| 8 | f32 | perm | 4.261 | 2.440 | 3.619 | 3.231 | 2.754 | — | 2.084 |
| 16 | f32 | seq | 4.568 | 2.730 | 4.152 | 3.553 | 3.595 | — | 2.255 |
| 16 | f32 | perm | 7.398 | 4.739 | 6.737 | 5.935 | 3.775 | — | 3.646 |
| 32 | f32 | seq | 8.055 | 6.097 | 7.740 | 6.683 | **4.290** | 4.435 | **4.132** |
| 32 | f32 | perm | 12.839 | 9.892 | 11.899 | 11.074 | **7.136** | 7.260 | **6.954** |
| 64 | f32 | seq | 15.395 | 15.296 | 14.959 | 12.943 | **8.441** | 8.426 | **8.373** |
| 64 | f32 | perm | 30.532 | 30.489 | 30.551 | 28.749 | **18.131** | 19.174 | **17.974** |
| 32 | f64 | seq | 8.492 | — | 8.045 | 6.883 | 5.199 | — | — |
| 32 | f64 | perm | 18.892 | — | 18.815 | 18.020 | 15.451 | — | — |

C++ reference point for the same kernel: dim 32 f32 sequential = **4.21 ns/pt**
(84.260 ms / 200 queries / 100 000 pts, from `leafn`); dim 32 f64 = 5.47 ns/pt.
Rust `hand_row_chunks` = 4.29 / 5.20 ns/pt. **Parity.**

`as_chunks::<4>()` (clippy's preferred spelling) measures within 1-6 % of
`chunks_exact(4)` — either form is fine for T3.

---

## 4. Root-cause statements

### Gap A — dim-32 (and worse at dim 64) runtime-dim knn, ~1.3-1.45× (f32 1.48×, f64 1.32× re-measured)

> **Root cause (confidence: HIGH).** `L2::eval`'s per-component
> `DataSource::point_component` access emits two bounds checks per component
> (one on `query[d]`, one on the `FlatSlice` index). At runtime dim these
> cannot be hoisted, so the 4-wide unrolled body becomes 8 basic blocks joined
> by conditional branches, which prevents LLVM from SLP-vectorising it. gcc
> compiles the *identical* summation order to AVX2/AVX-512 with `vpermt2ps`
> lane de-interleaving. The dim-32 workload evaluates 100 % of the dataset per
> query (measured), so this kernel deficit *is* the whole gap.

Falsifiers:
- A bounds-check-free variant with the identical summation order failing to
  reach C++'s ns/point at dim 32 (**tested: it reaches it — 4.29 vs 4.21**).
- The gap persisting when the kernel is measured in isolation with the tree
  removed (**tested: it *grows* to 1.95×, which is confirmatory not falsifying**).
- Leaf/point visit counts differing between the two implementations
  (**tested: exactly equal**).

Direct end-to-end confirmation (local, reverted prototype — see §6): adding
`DataSource::point_row` + a `chunks_exact(4)` row walk to `L2::eval` moved
**dim-32 f32 from 1.483 → 0.992** and **dim-64 f32 from 1.944 → 1.142**, with
the knn result digest bit-identical and all 345 workspace tests green.

### Gap B — fixed-dim-3 knn, ~4-6 %

> **Root cause (confidence: MEDIUM-HIGH).** The residual is per-`search_level`-
> call overhead, and it is *twice* what C++ pays for two compounding reasons:
> (a) gcc peels one level of `searchLevel`'s self-recursion, so a depth-D
> descent costs ≈D/2 frames, while LLVM inlines none and costs D frames; and
> (b) every Rust frame additionally re-validates `ctx.nodes[node_idx]`,
> `query[idx]` and `dists[idx]`, which C++ does not (raw `NodePtr` / `const
> T*`). The gate workload makes 40.35 such calls per query; the measured delta
> is ~32 ns/query, i.e. ~0.66-0.79 ns per call. The leaf kernel and the result
> set are NOT contributors (measured, §Q6a/§6c).

Falsifiers:
- Removing the per-call bounds checks alone leaving the ratio unchanged
  (**tested: gate median moved 1.064 → 1.026**).
- Replacing the recursion with an explicit stack leaving the ratio unchanged
  (**tested: gate median moved 1.064 → 1.018**).
- The result-set path mattering (**tested: it does not**).
- **Weakest link:** the per-run spread on this host is ±0.02-0.11 on the ratio,
  so no single A/B run is conclusive; the conclusion rests on 4-6 repeat
  medians per variant plus the asm structural difference.

---

## 5. Noise bound (must be quoted with any dim-3 number)

`perf_gate_knn_dim3_f32_k10` on the **unmodified** tree, 8 independent runs
across this session (each already an internal median-of-7):
`0.956, 1.035, 1.043, 1.063, 1.064, 1.071, 1.104, 1.192` → **median 1.064,
range 0.956-1.192**. The `sweep` probe's mid-range leaf sizes (64-512) flipped
sign between two clean-tree repeats (`-3.76` vs `+31.79` ns/query at leaf 64).
Any dim-3 claim below ~2 % is inside the noise floor of this machine.

Command: `PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --ignored perf_gate_knn_dim3 --test-threads=1 --nocapture`

---

## 6. Prototype A/Bs (all applied as LOCAL patches to `crates/nanoflann-rs/src`, measured, then `git checkout --` reverted; tree is clean)

| Prototype | What it changed | knn_fixed3 gate ratio (repeats) | dim-32 f32 knn ratio | digest bit-exact | `cargo test --workspace` |
|---|---|---|---|---|---|
| baseline | — | 1.043, 1.064, (+6 more, median **1.064**) | **1.483** | (reference) | 345/0 |
| **T3** `point_row` + `chunks_exact(4)` in `L2::eval` | `DataSource::point_row(idx) -> Option<&[T]>` (default `None`; `FlatSlice` and `&[[T;N]]` override), `L2::eval` takes the row path when available | 1.032 (n=1) | **0.992** | ✅ | **345 passed / 0 failed** |
| **T2a** `get_unchecked` in `search_level` only | `ctx.nodes`, `ctx.vind[left..right]`, `query[idx]`, `dists[idx]` | 1.007, 1.028, 1.023, 1.047 (median **1.026**) | — | ✅ | — |
| **T2b** T2a + `get_unchecked` in result set | `KnnResultSet::worst_dist`, `add_point_to_sorted` | 1.032, 1.000, 1.020, 1.035 (median **1.026**) | — | ✅ | — |
| **T2c** explicit-stack iterative `search_level` (safe, `[Frame;128]`, no heap) | recursion → descend/unwind loops with a 2-phase frame | 1.023, 1.013, 1.043, 1.009 (median **1.018**) | — | ✅ | **345 passed / 0 failed** |
| **T2c + T2a** | both | 1.028, 1.022, 1.003, 1.029, 1.050 (median **1.028**) | — | ✅ | — |

Full-gate snapshot under the **T3** prototype (all six gates, one run):
`build 1.003 · dyn_add 1.117 · dyn_knn_after_churn 0.962 · knn_fixed3 1.032 ·
knn_dyn_dim8_f64 0.893 · radius 0.766` — every gate passes, and **dim-8
improved from 0.947 → 0.893**.

T2c also showed a clear large-leaf win that the gate config does not expose:
`sweep` leaf=512 rust 14.852 ms vs baseline 16.39-16.48 ms; leaf=1024 20.844 ms
vs baseline 23.56-23.91 ms (**-10 to -13 %**).

---

## 7. Ranked fix candidates

### For T3 (kernel / dim-32)

**T3-1 — row-slice + fixed-size-chunk walk in `L2::eval` (and `L1`, `L2Simple`). Predicted gain: dim-32 f32 1.48 → ~0.99, dim-64 f32 1.94 → ~1.14, dim-8 f64 gate 0.947 → ~0.89. Parity risk: NONE (measured).**

```rust
// data_source.rs
pub trait DataSource<T: Scalar> {
    /// Whole row, when the layout allows it. Default `None` keeps every
    /// existing impl source-compatible.
    #[inline] fn point_row(&self, _idx: usize) -> Option<&[T]> { None }
}
impl FlatSlice  { fn point_row(&self, i) -> Option<&[T]> { self.data.get(i*self.dim .. i*self.dim + self.dim) } }
impl &[[T; N]]  { fn point_row(&self, i) -> Option<&[T]> { Some(&self[i][..]) } }

// metric.rs, inside impl_l2!, before the existing loop:
if let Some(row) = ds.point_row(idx) {
    if row.len() >= dim && query.len() >= dim {
        let (qc, _) = query[..multof4].as_chunks::<4>();
        let (rc, _) = row[..multof4].as_chunks::<4>();
        for (a, b) in qc.iter().zip(rc) {
            let (d0, d1, d2, d3) = (a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
            result += (d0*d0 + d1*d1) + (d2*d2 + d3*d3);   // ORDER UNCHANGED
        }
        // descending remainder from `row`, then `return result;`
    }
}
```
Parity argument: the arithmetic expression, its parenthesisation and the
descending remainder order are copied verbatim; only the *loads* move. Verified
bit-identical on 100 000 points × 4 dims × 2 scalars in the probe, on the
1000-query knn digest at dim 3 and dim 32, and by the full 345-test workspace
suite including the xval cross-validation and canary suites.
**Note the trait addition is a (source-compatible) public-API change** — it
needs a doc entry and a default-impl note.

**T3-2 — same treatment for `L1` and `L2Simple`. Predicted gain: proportional
(not measured). Parity risk: NONE by the same argument; must re-run xval.**

**T3-3 — hoist the `DynDim` per-query `vec![ZERO; dim]` scratch (tree.rs:530)
into a caller-supplied / reusable buffer. Predicted gain: ~0.3 % at dim 8,
negligible at dim ≥ 32. Parity risk: NONE.** Low priority; C++ pays the same
cost, so it does not move the ratio, only absolute time.

**Explicitly NOT recommended:** a bare `point_row` that keeps `row[d]` indexing
(that is what M1-T14 measured and reverted). It recovers only ~15 % of the gap
at dim 32 (11.07 vs 11.90 ns/pt) and nothing at dim 3. The win is in the
chunked, bounds-check-free walk, not in the row pointer per se.

### For T2 (traversal / dim-3)

**T2-1 — explicit-stack iterative `search_level` (safe, fixed `[Frame; N]` +
heap-spill fallback for pathological depth). Predicted gain: knn_fixed3 gate
1.064 → ~1.02 (median of 4 repeats); large-leaf configs -10 to -13 %. Parity
risk: NONE (measured: digest identical, 345/0).** The prototype is 100 lines;
frame = `{ split_dim, saved_dst, cut_dist, parent_mindist, other_child, phase }`
with `phase 0 = "best child done, consider other"`, `phase 1 = "other done,
restore dists"`; abort propagates by returning `false` straight out of the leaf
scan. **Caveat:** the prototype used a fixed `[Frame; 128]` and `assert!`s on
overflow — the real change needs a spill path, because M1's degenerate-tree
tests build depth-2115 and depth-16794 trees (search depth is bounded by tree
depth, so 128 is *not* safe in general).

**T2-2 — remove the per-call bounds checks in `search_level`
(`ctx.nodes[node_idx]`, `ctx.vind[left..right]`, `query[idx]`, `dists[idx]`).
Predicted gain: knn_fixed3 1.064 → ~1.026. Parity risk: NONE (measured).**
Requires `unsafe` + a documented invariant argument (node indices come from the
builder's own arena; `idx < dim` from `split_dim`; `query.len() >= dim` is
already asserted once per query in `find_neighbors`). Prefer *safe*
reformulations where they generate the same code, e.g. binding
`let node = ctx.nodes.get(i).expect(...)` does not help, but restructuring
`dists` as `D::Array` (fixed-size for `ConstDim`) removes the `dists[idx]`
check for free.

**T2-3 — do NOT chase the result set (`worst_dist` re-read, insert shift loop)
or the leaf-scan bounds checks: both measured at ≤0.5 %.**

**Stacking note:** T2-1 and T2-2 did **not** stack in this measurement
(combined median 1.028 vs 1.018 / 1.026 individually) — all three sit inside
the noise band of each other. Recommend landing **T2-1 alone** first and
re-measuring before adding `unsafe`.

---

## 8. Contradictions with previously recorded conclusions

1. **`docs/benchmarks.md` attributes the dim-32 gap to "the per-axis L2 kernel
   … plausibly needs SIMD/batching to amortize" and calls it out of scope.**
   That diagnosis is *directionally right but understates what is available*: no
   SIMD intrinsics, no batching and no summation reordering are needed. A
   bounds-check-free walk over the identical expression closes it entirely
   (dim-32 f32 1.483 → 0.992) because gcc's own vectorisation of that same
   expression proves the order is vectorisable. This is a "the gap is
   *cheaply* closable" contradiction, not a factual one.

2. **M1-T14's recorded conclusion that `point_row` "was measured as a net
   performance loss … and reverted" generalises incorrectly to dim 32.** The
   present measurement shows *why*: a row pointer that is still indexed
   per-component (`row[d]`) buys only ~15 % at dim 32 and ~0 % at dim 3 — so
   M1-T14's revert was correct **for what it measured**. The lever it did not
   test is the chunked walk. T3 must not read the M1-T14 revert as a blanket
   veto on row access.

3. **`docs/benchmarks.md` says the dim-3 recursion difference is "structurally
   present in the C++ source too … any remaining difference is down to each
   compiler's/ABI's call-overhead characteristics".** True as far as it goes,
   but the asm shows a *specific, nameable* asymmetry rather than a diffuse
   ABI difference: **gcc peels one level of `searchLevel`'s self-recursion
   (3 self-call sites, an inlined second level at `.L24`), LLVM peels none.**
   That roughly halves C++'s frame count for the same tree, and it is the
   mechanism T2-1 neutralises.

4. **`docs/benchmarks.md`'s quoted asm excerpt reads the pre-call
   `vmovss %xmm2, 8(%rsp)` as evidence of "argument marshalling".** In the
   current build it is caller-save spilling, not an argument: `eps_error` is
   constant-folded (`.LCPI0_0`), `AcceptAll` is a ZST, and all remaining
   arguments fit in `rdi/rsi/rdx/rcx/r8/r9 + xmm0`. gcc spills a comparable
   number of live values around its own call. **No argument is passed on the
   stack** — a T2 that tries to "shrink the argument list" is chasing a ghost.

5. **No contradiction on leaf/point visits.** The brief flagged this as the
   escalation trigger; the counts are *exactly* equal at both dim 3 and dim 32
   (65 951 / 99 999 862 evaluations, 31 064 / 14 460 000 interior visits) and
   the result digests are bit-identical. Nothing to escalate.

---

## 9. Reproduction

```bash
export PATH="$HOME/.cargo/bin:$PATH"
S=/tmp/scratch && mkdir -p $S

RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- kernel
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- leafn
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- knn
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- count
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- sweep
RUSTFLAGS="-C target-cpu=native" cargo run -p xval --release --example m25_diag -- dump $S

RUSTFLAGS="-C target-cpu=native" cargo rustc -p xval --release --example m25_asm -- --emit asm
grep -l 'probe_lib_dyndim32_f32:' target/release/examples/*.s

# C++ side (scratchpad only; vendored header is copied, never edited in-tree)
g++ -std=c++17 -O3 -march=native -ffp-contract=off \
    -I crates/nanoflann-ref/cpp -S -o $S/kernel_probe.s $S/kernel_probe.cpp
g++ -std=c++17 -O2 -march=native -ffp-contract=off -I $S -o $S/count_probe $S/count_probe.cpp
$S/count_probe $S
g++ -std=c++17 -O2 -march=native -ffp-contract=off -I $S -o $S/sweep_count $S/sweep_count.cpp
$S/sweep_count $S
```

## 10. Verification at task end

- `git status --short` → only the two new committed xval examples; `git diff` on
  `crates/nanoflann-rs` and `crates/nanoflann-ref` is empty.
- `cargo test --workspace` → **345 passed, 0 failed** (matches the recorded
  M2 total).
- `cargo clippy --workspace --all-targets` → clean.
