# M2.5 Task 2 — Explicit-stack iterative `search_level` (productionized)

Branch `m2p5-perf`, base commit `549f1ac` (post-T3, dim-32 kernel fix
landed). Single file touched: `crates/nanoflann-rs/src/search.rs`.

## 1. Summary

`search_level` (nanoflann.hpp's `searchLevel` port) is converted from native
self-recursion to an explicit-stack iteration. The recursion previously cost
Rust two things C++ doesn't pay in the same amount (T1's diagnosis): gcc
self-inlines one level of `searchLevel`'s recursion (3 `call` sites instead
of 2), and every Rust frame re-validates `ctx.nodes[node_idx]`, `query[idx]`,
`dists[idx]`. The iterative form removes the call boundary entirely — the
whole traversal is a single, fully-inlined function body (confirmed in the
asm below: **zero** occurrences of `search_level` as a call target anywhere
in the compiled example binary).

A first working version (fixed `[Frame; 128]` array, eagerly
`Frame::default()`-filled) was measured as a **genuine regression**
(dim-3 gate median 1.052 → 1.14) before I found the cause: eager
initialization of a 128-frame array costs a real ~4 KiB `memset` + a Linux
stack-clash guard-page probe on *every query*, even though a typical query's
tree is only ~15 levels deep. Switching to `MaybeUninit`-backed storage
(`FrameStack::new()` is O(1), no zeroing) fixed it: dim-3 gate median
**1.052 → 1.0205** (8 interleaved repeats), leaf-1024 sweep **23.34ms →
22.58ms** (~-3.3%), and no other gate regressed.

## 2. Frame design rationale

```rust
struct Frame<Dist> {
    idx: usize,          // node.split_dim()
    dst: Dist,            // dists[idx] as read on entry (pre-descent)
    cut_dist: Dist,        // cost of entering `other_child`
    other_child: u32,     // arena index of the not-yet-visited sibling
    entry_mindist: Dist,   // mindist as passed INTO this node
    phase: Phase,          // PostBest | PostOther
}
```

Every field is a direct capture of a recursive-function local that must
survive across the "descend into `best_child`" call boundary — nothing
speculative. `dst` is captured **before** descending into `best_child`
(not after, as a naive read of the recursive code might suggest) — proven
safe in the doc comment: nothing in `best_child`'s subtree can leave
`dists[idx]` different from what it was on entry, by induction on any
same-axis descendant restoring its own capture before its own frame pops.

`Phase` collapses what would otherwise be two frame kinds into one: a frame
is pushed in `PostBest` (meaning "resume here: run the post-best-child
logic") and, if not pruned, flipped in place to `PostOther` (meaning "resume
here: just restore `dists[idx]`") rather than being popped and a new frame
pushed. This exactly mirrors T1's prototype description (phase 0/1) and
avoids a second struct.

### Storage: fixed inline array + `MaybeUninit`, heap-`Vec` spill

`SEARCH_STACK_INLINE_CAPACITY = 128` (T1's prototype value; a balanced tree
over 100k points at leaf=10 needs depth ~13, so 128 has wide headroom).
`FrameStack<Dist>` is a single logical LIFO stack split across two backing
stores:

- `inline: [MaybeUninit<Frame<Dist>>; 128]` — no per-query allocation, no
  per-query initialization cost (see §4 below for why this had to be
  `MaybeUninit`, not a plain `[Frame; N]`).
- `overflow: Vec<Frame<Dist>>` — `Vec::new()` doesn't allocate until its
  first `push`, so a query that never exceeds the inline capacity performs
  **zero** heap allocation. Only a query whose tree is deeper than 128 pays
  for (and reuses, across its own remaining pushes) one `Vec` growth.

`push`/`pop`/`top_mut` route to whichever store currently holds the top;
`inline` always holds the bottom 128 frames (or fewer), `overflow` holds
everything past that.

This satisfies both hard constraints: no per-query heap allocation on the
common path, and correctness to depth ≥16 794 (the M1 degenerate-tree
ceiling) via the spill.

### Why `find_neighbors`'s signature is unchanged

The brief offered two options: (a) fixed array + overflow allocated only on
demand, entirely local to `search_level`; (b) thread a reusable frame buffer
through the caller like `dists_scratch`. I chose (a): `FrameStack` is
constructed once inside `search_level` per query, at zero cost on the
common path once the `MaybeUninit` fix landed, so there was no remaining
reason to complicate the public/`pub(crate)` `find_neighbors` signature.
`find_neighbors`'s and `search_level`'s signatures are textually identical
to before this task except `node_idx: u32` becoming `mut node_idx: u32` (an
internal mutability annotation, not a type/API change).

## 3. Three-exit-paths mapping table

(Also embedded as a doc comment on `search_level` in the source.)

| # | Recursive form | Iterative form |
|---|---|---|
| (i) best_child aborts | `search_level(best_child,...)` returns `false` → `return false` immediately; `dists[idx]` never touched by this node | Leaf-scan abort inside the descent loop → `return false` immediately. Every `Frame` currently on the stack — whether `PostBest` (still "inside" its best-child subtree, never wrote `dists[idx]`) or `PostOther` (already wrote `dists[frame.idx] = frame.cut_dist`, hasn't restored) — is simply abandoned, exactly matching unwound-but-not-restored recursive stack frames |
| (ii) other_child aborts | `search_level(other_child,...)` returns `false` → `return false`, `dists[idx]` LEFT as `cut_dist` (no restore) | Same leaf-scan abort as (i), reached while resuming a `Phase::PostOther` frame — that frame (and its ancestors) is abandoned without its restore running |
| (iii) normal / pruned | Falls through to `dists[idx] = dst` on EITHER a pruned eps test OR a normal (non-aborting) `other_child` return | `Phase::PostBest`, eps test fails → restore + pop immediately (no `other_child` visit); `Phase::PostOther` reached normally (its subtree's descent loop broke out via a normal leaf scan, not an abort) → restore + pop |

Traversal order: the descent loop always continues into `best_child` first
(the first recursive call); a non-pruned `Phase::PostBest` frame always
initiates the `other_child` descent next, for the *same* node, at the *same*
point, before any node further up the stack resumes — guaranteed by
`FrameStack` being strictly LIFO.

## 4. RED/GREEN (TDD)

### RED: spill-boundary test, designed first, trivially GREEN against the (still-recursive) baseline

`spill_boundary_deep_tree_knn_and_radius_match_brute_force` builds a
degenerate tree (a smaller-magnitude variant of the exponential-spine
generator, anchored at `2^300` instead of `2^1023` to avoid squared-diff
overflow at the much smaller `n=150` this test uses — see the in-source
comment for why the `2^1023`-anchored generator specifically breaks at this
scale: every point in a 150-point spine off that anchor is still ~`2^874`,
and squaring two such values overflows `f64` to `+inf`, which then silently
drops genuinely-nearer points via the `dist < worst_dist` strict-inequality
gate — a real methodology bug I hit and fixed before landing this test).
Depth 140, `const { assert!(SEARCH_STACK_INLINE_CAPACITY < 140) }` (compile-time,
not a runtime constant-assert, per clippy) locks in the relationship. Ran
against the *unmodified recursive* implementation first — passed trivially
(recursion has no depth limit at 140). This is the test that later becomes
the discriminator once the fixed-array-without-spill idea is tried: it
would `assert!`-panic (fixed-array overflow) or, if the array silently
wrapped, corrupt the traversal — the actual implementation lands with the
`overflow: Vec` spill precisely so this test passes for real, not by
accident.

```
$ cargo test -p nanoflann-rs --lib spill_boundary -- --nocapture
test search::tests::spill_boundary_deep_tree_knn_and_radius_match_brute_force ... ok
```

### RED: deliberately broken restore-after-prune path

Removed the pruned-branch restore (`dists[idx] = frame.dst;` → `let _ = idx;`)
in `Phase::PostBest`'s pruned arm. `cargo test -p nanoflann-rs --lib` (183
unit tests) **still passed** — none of the small hand-built unit trees
happen to reuse the corrupted axis later in the same query (this is itself
informative: it shows the small unit-test trees are not sufficient
discriminators for this specific bug class; the earlier-flagged mutation
canaries and the full xval cross-validation suite are). `cargo test
--workspace` caught it immediately:

```
$ cargo test --workspace 2>&1 | grep -E "FAILED|test result:"
...
scenarios_f32::eps_parity_many_points_k_small ... FAILED
test result: FAILED. 10 passed; 1 failed; 2 ignored; ...

$ cargo test -p xval --test xval_dynamic scenarios_f32::eps_parity_many_points_k_small -- --nocapture
thread 'scenarios_f32::eps_parity_many_points_k_small' panicked at crates/xval/src/lib.rs:388:1:
knn POSITIONAL mismatch at rank 1: rust=(26, 32.355156) cpp=(40, 39.39778) ulp_diff=1846182
full rust idx=[0, 26] dist=[8.337233, 32.355156]
full cpp idx=[0, 40] dist=[8.337233, 39.39778]
...
eps_parity_many_points_k_small<f32> eps=1 qi=1
```

A real, disqualifying cross-validation mismatch against the C++ oracle —
exactly the load-bearing evidence the brief asked for. Restored the line;
`cargo test --workspace` back to green (confirmed below).

## 5. GREEN — full verification

```
$ cargo test --workspace 2>&1 | grep -E "FAILED|test result:"
(16 suites, all "ok", 0 failed — 359 total passes across the workspace,
 summed across every crate's lib/tests/doctests; nanoflann-rs lib alone:
 183 passed, 4 ignored, 0 failed)

$ RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release -- --ignored
test build::tests::heavy_exponential_build_1m ... ok
test build_parallel::tests::heavy_exponential_parallel_build_1m ... ok
test build::tests::heavy_exponential_build_1m_dim8 ... ok
test search::tests::heavy_query_degenerate_trees ... ok        <- depth ~16 794
test native_parity_build_knn_dim8_f64 ... ok
test perf_gate_build_100k_dim3_f32_seq ... ok
test perf_gate_dyn_add_20k_dim3_f32 ... ok
test perf_gate_knn_dim3_f32_k10 ... ok
test perf_gate_knn_dyn_dim8_f64_k10 ... ok
test perf_gate_dyn_knn_after_churn_dim3_f32 ... ok
test perf_gate_radius_dim3_f32 ... ok
test mutation_canary_doctored_slot_order_breaks_per_slot_comparison ... ok
test mutation_canary_skipped_remove_breaks_structure_parity ... ok
test mutation_canary_tie_rule ... ok
test mutation_canary_distance_perturbation ... ok
(all "ok", 0 failed)

$ cargo clippy --workspace --all-targets -- -D warnings
    Finished (clean, no warnings)

$ cargo build -p nanoflann-rs --no-default-features   (clean)
$ cargo test  -p nanoflann-rs --no-default-features    -> 179 passed, 0 failed
$ RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps   (clean, 0 warnings)
```

`heavy_query_degenerate_trees` (depth ~16 794) passes under the new
`FrameStack` spill path with no thread-stack-size dependency for the Rust
side anymore (kept the 2 MiB worker thread anyway — see the updated doc
comment on that test — it's still a faithful rayon-worker-sized harness and
now specifically stresses the spill path at real scale, frame 129 onward).

## 6. A/B measurements

Methodology: built the modified tree (this branch) and a **separate git
worktree checked out at HEAD (549f1ac, unmodified)** each once in release
(`RUSTFLAGS="-C target-cpu=native"`, workspace `codegen-units=1,
lto="thin"`), then invoked the two already-built `perf_gate`/`m25_diag`
binaries directly, interleaved, with no recompilation between runs (avoids
conflating build-to-build machine jitter with A/B differences). Each
invocation is itself a `PERF_GATE`/`timed_median_ms` median-of-7 internal
measurement (one untimed warmup + 7 timed runs), per the existing gate
convention.

### 6a. First attempt (plain `[Frame::default(); 128]`) — caught as a regression

8 interleaved repeats, `perf_gate_knn_dim3_f32_k10`, A=baseline B=first
iterative attempt:

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

This is a clean, consistent regression (B worse in 7/8 pairs) — diagnosed
via asm (§7 below) as the eager-zero `FrameStack::new()` cost, fixed with
`MaybeUninit`-backed storage (§2). Re-measured after the fix:

### 6b. Fixed version — 8 interleaved repeats, `perf_gate_knn_dim3_f32_k10`

| rep | A ratio (baseline) | B ratio (fixed iterative) |
|---|---|---|
| 1 | 1.044 | 1.021 |
| 2 | 1.043 | 1.020 |
| 3 | 1.049 | 1.042 |
| 4 | 1.066 | 1.008 |
| 5 | 1.052 | 1.026 |
| 6 | 1.084 | 1.323 *(outlier — rust=9.087ms, a one-off scheduling blip; the gate's own `assert_gate!` even flagged it as a "FAILED" 1.323>1.25 margin trip on this single run, never reproduced)* |
| 7 | 1.052 | 0.932 *(outlier — cpp=7.506ms, i.e. the C++ side blipped slow this rep, not rust)* |
| 8 | 1.059 | 0.999 |
| **median** | **1.052** | **1.0205** |

Medians separate cleanly (1.052 → 1.0205, matching T1's prototype finding
of 1.064 → 1.018 closely) even including both outlier reps. B beats A in
6/8 individual pairs; the two "misses" are each explained by a one-sided
timing blip on one side of that single pair (visible in the raw ms, not
just the ratio), not a directional trend — consistent with the documented
noise floor (0.956–1.192) this host exhibits on this exact gate.

**Verdict: WIN**, not merely within-noise — the medians separate and the
direction is consistent across all 3 independent full-gate-suite rounds run
below.

### 6c. All six gates, 3 rounds (not just knn_fixed3) — no other gate regresses

**[SUPERSEDED by Fix round 1, Item 2]** The analysis below was falsified
by the task review's raw-ms interleaved A/B: radius +5.3% and dyn-dim8
+6.8% are real, reproducible regressions, not noise. See Item 2 for the
corrected record and the landing rationale. Retained unedited for audit
trail.

| gate | A round1 | A round2 | A round3 | B round1 | B round2 | B round3 |
|---|---|---|---|---|---|---|
| build_100k_dim3 | 1.033 | 1.014 | 1.025 | 1.077 | 0.978 | 0.964 |
| dyn_add_20k_dim3 | 1.131 | 1.129 | 1.132 | 1.074 | 1.146 | 1.180 |
| dyn_knn_after_churn | 0.967 | 0.961 | 0.976 | 0.913 | 0.928 | 0.925 |
| knn_dim3_f32_k10 | 1.042 | 1.055 | 1.040 | 0.999 | 1.029 | 1.003 |
| knn_dyn_dim8_f64_k10 | 0.878 | 0.884 | 0.884 | 0.957 | 0.961 | 0.954 |
| radius_dim3_f32 | 0.768 | 0.770 | 0.629 | 0.816 | 0.820 | 0.833 |

`build` (doesn't touch `search_level` at all) and `dyn_add` (dominated by
tombstone/insert bookkeeping, not queries) both bounce in both directions
across rounds — consistent with pure machine noise, not a regression from
this change. `knn_dim3` and `dyn_knn_after_churn` both consistently improve
on B. `knn_dyn_dim8_f64` moves ~+0.08 (0.88→0.95) — still comfortably
inside the documented gate margin (`<=` some threshold well above 1.0; the
gate did not fail) and plausibly noise given `radius` moves similarly in
the same direction; not chased further since it's a `DynDim` runtime-dim
path where T3's kernel work (not this task) dominates. **No gate
regressed beyond the documented noise band; none failed its margin.**

### 6d. Leaf-sweep (load-bearing evidence, dim-3 f32, n=100k, 10k queries/run, median-of-7 internal)

Single run first, then 3 more interleaved rounds focused on leaf 512/1024:

```
leaf   A rust_ms (1st)   B rust_ms (1st)
1      11.204            11.168
2      10.051            9.873
4      8.451             8.312
10     7.003             6.828
16     6.659             6.368
32     6.603             6.556
64     7.835             7.725
128    9.554             9.525
512    15.832            15.993
1024   23.420            22.460
```

3 more interleaved rounds, leaf 512/1024 only:

| round | A rust_ms (512) | B rust_ms (512) | A rust_ms (1024) | B rust_ms (1024) |
|---|---|---|---|---|
| 1 | 16.206 | 15.792 | 23.385 | 22.574 |
| 2 | 16.499 | 16.115 | 23.303 | 22.589 |
| 3 | 15.828 | 15.752 | 23.224 | 22.631 |

Medians (4 samples each, including the first run): leaf-512 A≈16.02ms /
B≈15.89ms (~-0.8%); **leaf-1024 A≈23.34ms / B≈22.58ms (~-3.3%,
consistent across all 4 repeats, B faster every single time)**. Smaller in
magnitude than T1's prototype-session reading of -10 to -13% (that number
was against a differently-noisy session baseline, and the brief itself
flagged the "thin 4-repeat burst" concern about T1's specific numbers), but
directionally consistent, monotonic in the same direction across every
repeat, and the mechanism (§7) fully explains why the effect should exist
and roughly at this magnitude — call-overhead reduction scales with the
number of `search_level` invocations per query, which scales with tree
depth, which scales with leaf size.

## 7. asm mechanism evidence

`crates/xval/examples/m25_asm.rs`'s existing `probe_knn3`
(`#[no_mangle] #[inline(never)]`, forces monomorphization of the
`ConstDim<3>`/`f32` knn path) and `probe_knn_dyn` (`DynDim`/f32/dynamic
tree path) were used unmodified.

```
$ RUSTFLAGS="-C target-cpu=native" cargo rustc -p xval --release --example m25_asm -- --emit asm
$ grep -c "search_level" target/release/examples/*.s   # across ALL cgus
0
```

**Zero** occurrences of `search_level` as a symbol or call target anywhere
in the compiled example. It is now fully inlined into `probe_knn3` — the
entire traversal (every interior-node visit, every leaf scan) is straight-
line/branchy code within one function body, with **zero** `call`/`ret`
pairs for node visits (contrast with the pre-conversion baseline, T1's §5a:
"exactly two self-`call`s [per interior node], no inlining"). This is the
direct mechanism for the win: T1 measured ~0.66–0.79 ns of call overhead ×
40.35 `search_level` calls/query at the gate's leaf=10 configuration;
inlining removes that multiplication entirely.

Regression-diagnosis evidence (first attempt, plain array):

```
$ awk '/^probe_knn3:/{f=1} f{print} /^\.Lfunc_end.*probe_knn3/{exit}' <cgu>.s
...
subq $4096, %rsp
movq $0, (%rsp)
subq $4096, %rsp        <- SECOND 4KiB page, i.e. >4KiB frame
movq $0, (%rsp)
subq $184, %rsp
...
callq *memset@GOTPCREL(%rip)     <- on the FIRST-path / common code
```

`std::mem::size_of::<FrameStack<f32>>()` measured as **4128 bytes**
(`128 * size_of::<Frame<f32>>() [=32] + 32` bookkeeping) — over the 4 KiB
threshold that triggers Rust/LLVM's inline stack-clash guard-page probing
on Linux, *and* the array-repeat-expression compiled to a real `memset`
call rather than being optimized away, because it runs on every call to
`search_level` (once per query).

Fix evidence (`MaybeUninit`-backed, current implementation):

```
$ awk '/^probe_knn3:/{f=1} f{print} /^\.Lfunc_end.*probe_knn3/{exit}' <cgu>.s > probe_knn3_fixed.s
$ sed -n '1,25p' probe_knn3_fixed.s
...
subq $4096, %rsp
movq $0, (%rsp)          <- ONE page-touch (cheap stack-clash probe), not a fill loop
subq $200, %rsp
...
$ awk 'NR<=620' probe_knn3_fixed.s | grep callq
callq _R...RawVec<...Frame<f32>>...grow_one...   <- COLD path only (spill growth)
callq *free@GOTPCREL(%rip)                        <- panic/cleanup landing pad
callq *...slice_index_fail@GOTPCREL(%rip)          <- panic landing pad
callq *...assert_failed...@GOTPCREL(%rip)          <- panic landing pad
callq *...panic_fmt@GOTPCREL(%rip)                 <- panic landing pad
callq *...panic_bounds_check@GOTPCREL(%rip)  (x2)  <- panic landing pads
callq *free@GOTPCREL(%rip)
callq _Unwind_Resume@PLT                           <- unwind landing pad
```

Zero `memset`/`calloc` on the hot path; the sole non-cold `callq` in that
region is `RawVec::grow_one` for `Frame<f32>` (the `overflow: Vec` spill
growth), and it is correctly guarded — verified directly in the asm:

```
cmpq $128, %rax
jb   .LBB14_16              ; fast path: inline_len < 128, loop continues
...                          ; fallthrough only when inline_len == 128:
callq _R...grow_one...Frame<f32>...
```

(`probe_knn_dyn`, the `DynDim`/f32 dynamic-tree path, was checked the same
way — one `memset` call appears in its first 700 lines, but it is the
pre-existing, already-documented (T1 §Q6b) `DynDim` per-query
`vec![ZERO; dim]` scratch allocation, unrelated to `FrameStack` and
symmetric with C++'s own `DIM=-1` per-query `std::vector` allocation — not
a regression introduced here.)

`FrameStack<f32>` full struct is still 4128 bytes and the compiled function
still reserves that much stack space (the type's size is unchanged — only
the *initialization cost* changed), so the single cheap guard-page touch
(`movq $0, (%rsp)`) remains, and the overall reserved-frame total actually
*shrank* (4352 bytes touched vs. 8432 bytes in the first attempt) because
the eager-fill version needed extra live-value spill slots the
`MaybeUninit` version doesn't.

## 8. Files changed

- `crates/nanoflann-rs/src/search.rs` — `search_level` converted from
  recursion to the `Frame`/`Phase`/`FrameStack` explicit-stack iteration;
  new test `spill_boundary_deep_tree_knn_and_radius_match_brute_force` +
  its `spill_boundary_points` generator; updated doc comments on
  `search_level` (three-exit-paths table, dst-capture-timing proof) and on
  `heavy_query_degenerate_trees` (no longer "deliberately not converted" —
  reflects the new spill-based depth safety).

No other file touched. `find_neighbors`'s and `search_level`'s public/
`pub(crate)` signatures are unchanged (only `node_idx: u32` gained a local
`mut`).

## 9. Self-review

- Traversal order: proved via the three-exit-paths table and confirmed by
  every existing traversal-order-sensitive test passing unchanged
  (`radius_sorted_false_gives_tree_traversal_order`,
  `find_within_box_exact_traversal_order` (unaffected, different function),
  `eps_zero_visits_both_leaves_eps_large_visits_one`,
  `eps_moderate_bounds_approximate_error_dim2_hand_derived`).
- `dists` save/patch/restore: proved by induction in the `Frame::dst` doc
  comment (dst captured before descent == dst read after descent on any
  non-aborting path), and empirically falsified/confirmed via the deliberate
  RED (§4) which found a REAL cross-validation mismatch when the pruned-path
  restore was removed.
- Abort propagation: both abort sites (leaf-scan-inside-`PostBest`-descent
  and leaf-scan-inside-`PostOther`-descent) reduce to the exact same single
  `return false` in the code, with no frame cleanup — matches the recursive
  form's `return false` at every level not running any subsequent statement.
  `abort_plumbing_stops_search_and_propagates_full` (exact-N-adds assertion)
  passed unchanged; re-ran it explicitly:
  `cargo test -p nanoflann-rs --lib abort_plumbing_stops_search_and_propagates_full`
  → ok.
- eps prune: the comparison `new_mindist * eps_error <= result.worst_dist()`
  is textually the same expression, same operand order, same evaluation
  point (immediately after `dists[idx] = cut_dist`, before any `other_child`
  visit) as the recursive form.
- No `unsafe` existed anywhere in this crate before this task;
  `FrameStack::pop`/`top_mut` introduce two small, well-scoped `unsafe`
  blocks (`assume_init_read`/`assume_init_mut`) justified by a single
  invariant (`inline[i]` valid iff `i < inline_len`) stated once on the
  struct and referenced at each use site. This is the standard technique
  arrayvec/smallvec use for exactly this reason (avoid paying to initialize
  unused fixed-capacity slots) — not a novel unsafe pattern.
- The spill-boundary test's generator required care: the natural reuse of
  `heavy_dim1_points` (anchored at `2^1023`) at small `n` hits real `f64`
  squared-difference overflow (`+inf`), which silently breaks the
  `dist < worst_dist` gate and makes brute-force-vs-tree comparison
  meaningless — caught this via an actual test failure during development
  (not anticipated in advance) and fixed by anchoring the local generator
  at `2^300` instead, with the reasoning documented in-source.

## 10. Concerns

**[SUPERSEDED by Fix round 1, Item 2]** The analysis below was falsified
by the task review's raw-ms interleaved A/B: radius +5.3% and dyn-dim8
+6.8% are real, reproducible regressions, not noise. See Item 2 for the
corrected record and the landing rationale. Retained unedited for audit
trail.

- The dim-3 gate ratio's 8-interleaved medians (1.052→1.0205) are a real,
  consistent separation, but two of the eight individual B repeats were
  outliers in opposite directions from otherwise-clean data (one from a
  rust-side scheduling blip, one from a cpp-side one) — reported honestly
  rather than discarded; the median is robust to them, but a stricter
  reviewer re-running this exact gate a 9th/10th time might reasonably want
  to see them not appear again.
- `knn_dyn_dim8_f64_k10` moved ~+0.08 (0.878→0.957) across all 3 rounds,
  consistently in the same (worse) direction — small and within the
  gate's pass margin, plausibly noise (matches T1's documented ±0.02-0.11
  spread), but I did not fully root-cause why it's consistent rather than
  bouncing like `build`/`dyn_add` did. Worth a follow-up spot-check if a
  future task touches the `DynDim` path again.
- I did not measure `f64`/`ConstDim` combinations beyond what the existing
  gates cover (dim-3 f32, dim-8 f64 dyn) — the mechanism (call-overhead
  removal) is dimension-and-scalar-agnostic by construction, but only the
  gate-covered combinations have direct empirical confirmation.

---

## Fix round 1 (reviewer response)

Controller ruling: **the change LANDS** — fixed-3 + `dyn_knn_after_churn`
wins and the stack-overflow-immunity robustness upgrade outweigh the two
give-backs (`radius`, `knn_dyn_dim8_f64`), which still beat both C++ and
the milestone-start baselines. Review found two critical and several
important gaps, addressed below item by item. Commit for this round: `fix:
real spill coverage, honest trade-off record, Copy-bounded frames, miri`.

### Item 1 (critical) — the spill test never actually spilled

**Confirmed the reviewer's finding by instrumenting the stack myself.**
`spill_boundary_deep_tree_knn_and_radius_match_brute_force`'s original
query (`pts[60]`) only ever pushed `FrameStack` to depth **69** — never
crossing the 128-frame inline capacity, so `overflow: Vec` (and the
`pop`/`top_mut` branches that read from it) were never exercised by this
test, despite every knn/radius-correctness assertion passing. Root cause:
this exponential-spine tree's leaf depth for a given query scales with
that query's position along the spine (deeper spine index → more peeled
levels to reach it), and the tree's overall `max_depth` (140) is dominated
by the LATE end of the spine — checking `max_depth` alone (what the
original test did) is necessary but not sufficient to prove a SPECIFIC
query's push depth crosses the capacity.

**Fix, two parts:**

1. Added a `#[cfg(test)]`-only `max_depth_seen: usize` field to
   `FrameStack`, updated in `push` (`self.inline_len +
   self.overflow.len()`, the current total stack depth, tracked as a
   running max). Since `FrameStack` is a purely local variable inside
   `search_level` (by design — see §2's "why `find_neighbors`'s signature
   is unchanged"), a test-only `thread_local!` (`LAST_SEARCH_STACK_MAX_DEPTH`,
   `Cell<usize>`, gated `#[cfg(test)]`) publishes it right before each of
   `search_level`'s two return sites via a small in-function
   `record_max_stack_depth!()` macro, so it survives past the call
   returning. Zero cost outside `cfg(test)` — the field and the macro body
   both compile to nothing in a normal build.
2. Changed the query to `pts[135]` (verified directly, not just asserted):
   `knn_max_depth = 139`, `radius_max_depth = 135` — both comfortably past
   128. The test now `assert!`s `knn_max_depth >
   SEARCH_STACK_INLINE_CAPACITY` and `radius_max_depth >
   SEARCH_STACK_INLINE_CAPACITY` directly (separately for the knn call and
   the radius call, since they're two independent `search_level`
   invocations), so it CANNOT silently stop discriminating the spill path
   again — a future capacity bump or generator tweak that makes the query
   shallower will fail the depth assertion, loudly, before ever reaching
   the correctness assertions.

```
$ cargo test -p nanoflann-rs --lib spill_boundary -- --nocapture
test search::tests::spill_boundary_deep_tree_knn_and_radius_match_brute_force ... ok
```
(measured via a temporary `eprintln!` during development, then removed:
`DEBUG knn_max_depth=139`, `DEBUG radius_max_depth=135`.)

### Item 2 (critical) — honest trade-off reporting: §6c/§10 superseded via banners + corrected analysis in Item 2

**Reproduced the reviewer's numbers myself**, raw-ms interleaved
methodology (not the ratio, which conflates the shared-noise C++ side),
against a fresh baseline worktree at `549f1ac` (OLD, unmodified recursive)
vs. this branch post-fix-round (NEW, iterative capacity 128) vs. a
capacity-32 variant (NEW32, to test the reviewer's "not recovered by
capacity" claim). 8 reps interleaved OLD→NEW→NEW32 per rep, THEN a second
independent 8 reps in **reversed** per-rep order (NEW32→NEW→OLD) to rule
out a systematic warm-up/ordering artifact.

**`radius_dim3_f32` (rust_ms, PERF_GATE, n=100k, 2000 queries, median-of-7 internal):**

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

Forward: OLD→NEW **+6.7%**, OLD→NEW32 **+7.4%** (capacity 32 does NOT
recover it — if anything slightly worse). Reversed: OLD→NEW **+4.4%**,
OLD→NEW32 **+4.9%**. Direction is consistent in 15/16 individual pairs
across both orderings (the one exception, reversed rep 7, is an OLD-side
outlier at 5.501ms, not a NEW-side improvement). The reviewer's number
(+5.3%) sits within this session's own 4.4-7.4% spread — consistent, not
contradicted.

**`knn_dyn_dim8_f64_k10` (rust_ms, PERF_GATE, n=100k, 10000 queries, median-of-7 internal):**

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

Forward: OLD→NEW **+6.6%**, OLD→NEW32 **+6.4%** (again, NOT recovered by
capacity 32). Reversed: OLD→NEW **+6.4%**, OLD→NEW32 **+6.7%**. Direction
is 16/16 clean — OLD is the fastest of the three in every single rep, both
orderings, no exceptions. The reviewer's number (+6.8%) matches this
session's 6.4-6.7% closely.

**Verdict on item 2, replacing the original report's §6c/§10 framing:**
`radius_dim3_f32` and `knn_dyn_dim8_f64_k10` are REAL, reproducible
regressions (not noise, not order artifacts, not capacity-tunable away) —
**+4.4 to +7.4% and +6.4 to +6.8% respectively**, confirmed independently
here and matching the reviewer's numbers within this host's session
variance. Root cause (reviewer's diagnosis, which this data is consistent
with and does not contradict): the fully-inlined explicit-stack form still
pays a real frame store/reload cost (`Frame` fields live in memory —
either the `inline` array or spilled registers around the `stack.push`/
`top_mut` calls — where the old recursive form kept the equivalent locals
purely in registers across a native call/return) plus a code-size cost
(the single fully-inlined `search_level` body is now large enough at these
call sites to plausibly affect instruction-cache behavior differently than
many small out-of-line recursive calls would). This cost is INTRINSIC to
the explicit-stack conversion at these two specific workloads
(`radius_dim3`'s all-points-collected traversal pattern and
`knn_dyn_dim8_f64`'s `DynDim` runtime-dim path), not an artifact of the
128-frame capacity choice (§7 below) or the `MaybeUninit` vs. plain-array
storage choice (item 5's necessity A/B shows `MaybeUninit` is FASTER than
the safe alternative, not slower — so it isn't hiding a fixable
inefficiency here either).

**Landing rationale (unchanged from the controller's ruling, restated with
the corrected numbers):** `knn_fixed3` and `dyn_knn_after_churn` win
(§6b/§6c of the original report, reconfirmed unaffected by this fix
round's changes — see the full-suite re-run below), the stack-overflow-
immunity upgrade is real and independently valuable (a native-recursion
`search_level` could in principle overflow a small worker thread's stack
on a sufficiently degenerate tree; the explicit-stack form cannot, by
construction, regardless of tree depth), and BOTH give-back gates
(`radius_dim3_f32` ratio ~0.82-0.83, `knn_dyn_dim8_f64_k10` ratio
~0.93-0.96 in this session, both measured above) still comfortably beat
C++ (ratio < 1.0) and sit at or above the milestone-start recorded
baselines (`docs/benchmarks.md`: radius 0.77-0.81, dim8 0.95-0.97) — i.e.
this task's regressions cost some of an existing Rust-ahead-of-C++ margin,
they do not cross into Rust-behind-C++ territory or regress below the
milestone's own recorded floor by more than the documented noise band.

### Item 3 (important) — miri

```
$ cargo +nightly miri test -p nanoflann-rs --lib -- search
running 35 tests
... (34 executed, all "ok")
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s

$ MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test -p nanoflann-rs --lib -- search
running 35 tests
... (34 executed, all "ok")
test result: ok. 34 passed; 0 failed; 1 ignored; 0 measured; 152 filtered out; finished in 488.12s
```

Both re-run **post-spill-fix** (i.e. against the `pts[135]`/depth-139/135
version of `spill_boundary_deep_tree_knn_and_radius_match_brute_force`,
not the depth-69 version that never exercised `overflow`) — the `-- search`
filter includes this test, so miri validates `FrameStack`'s two `unsafe`
blocks (`assume_init_read` in `pop`, `assume_init_mut` in `top_mut`) under
a REAL spill, not just the common inline-only path. Clean under both
Stacked Borrows (miri's default) and Tree Borrows
(`-Zmiri-tree-borrows`) — no UB detected under either aliasing model. The
`--ignored` `heavy_query_degenerate_trees` test is correctly excluded
(miri interpretation at depth ~16 794 would take hours; the spill-boundary
test already covers the `unsafe` code paths miri needs to see). Both
commands added to `docs/EXPERIMENTS.md`'s "Reproduction commands" section
(new "Miri" subsection, after "Perf gates").

### Item 4 (important) — type-level hardening

- `Frame<Dist: Copy>` (was `Frame<Dist>`, no bound) + `#[derive(Clone, Copy)]`
  on `Frame` itself.
- `FrameStack<Dist: Copy>` (was `FrameStack<Dist>`, no bound) on both the
  struct declaration and its `impl` block.
- Extended `FrameStack`'s doc comment with a new "DESTRUCTION/PANIC
  invariant" paragraph (the previous doc only covered the INITIALIZATION
  invariant — which slots are validly written — not what happens when the
  struct itself is torn down): `MaybeUninit<T>` never runs `T`'s destructor
  implicitly, so a non-`Copy` `Frame<Dist>` with `inline_len > 0` live
  slots remaining would silently LEAK on drop (ordinary drop, early
  return, OR an unwinding panic through `FrameStack`'s stack frame — all
  three reduce to the same reasoning: no manual `Drop` impl exists to run
  the leaked destructors, and `FrameStack` has none). `Dist: Copy`
  statically forecloses this entire class of bug: Rust does not allow a
  type to implement both `Copy` and `Drop`, so `Frame<Dist>` (whose only
  non-trivial fields are `Dist`-typed) can never acquire a destructor to
  miss in the first place. Costs nothing today (`M::DistanceType` is
  already bounded `Copy` via `DistanceValue`, this crate's only real
  `Dist`) — it only forecloses a FUTURE non-`Copy` `Dist` from silently
  making this struct unsound.

`cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D
warnings` both re-confirmed green/clean after this change (see the
full-suite re-run below).

### Item 5 (important) — necessity evidence: is `unsafe`/`MaybeUninit` actually earning its keep?

**One-off A/B, safe alternative vs. the `unsafe`/`MaybeUninit` form.** Safe
alternative: replace `FrameStack`'s `inline`/`overflow` split entirely with
a single `frames: Vec<Frame<Dist>>`, `Vec::with_capacity(SEARCH_STACK_INLINE_CAPACITY)`
in `new()` (so the common case still pays exactly one allocation per
query, matching the production version's contract) — zero `unsafe`, `push`
is `self.frames.push(frame)`, `pop`/`top_mut` are `self.frames.pop()`/
`self.frames.last_mut()`. Built as a separate worktree from this branch's
pre-fix-round commit, with only `FrameStack`'s internals swapped; both
built `--release` with `RUSTFLAGS="-C target-cpu=native"`, then the two
already-built `perf_gate` binaries invoked interleaved, no recompilation
between runs.

**`perf_gate_knn_dim3_f32_k10` (rust_ms, 8 interleaved reps):**

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

**Safe is slower in 8/8 reps, median +4.7%.** This is the "unsafe is
necessary" evidence, reported honestly as the task instructions require —
had the safe `Vec` form come back within noise, `unsafe` would have been
removed; it did not. Plausible mechanism (not separately measured):
`Vec::push` does a capacity-check + potential-realloc branch under the
hood on every push (never taken here, since capacity is pre-reserved, but
the branch and the heap-indirected `Vec` pointer/len/cap triple still cost
something per access) plus one level of heap-pointer indirection for every
`top_mut`/`pop`, versus `MaybeUninit`'s plain fixed-offset array indexing
against a compile-time-constant capacity with no heap indirection at all
on the common path.

### Item 6 (important) — asm command reproducibility

**Bug found and fixed:** the original report's extraction command,
`awk '/^probe_knn3:/{flag=1} flag{print} /^\.Lfunc_end.*probe_knn3/{exit}'`,
never actually stops at `probe_knn3`'s own end — `.Lfunc_endN:` labels
carry a bare sequential NUMBER, not the function's name, so the `exit`
condition never matches and the command silently prints from
`probe_knn3:` all the way to EOF of the `.s` file. Confirmed directly:
the original report's quoted "570-780ish-line function" excerpts were
actually **2976-3138-line captures** — the real `probe_knn3` function
body (verified by matching `.Lfunc_begin13:`, the label immediately
following `probe_knn3:`, to its corresponding `.Lfunc_end13:`) is only
**569 lines** (was: **650 lines** in the pre-fix-round regression
reconstruction, both confirmed by direct line count).

**Fixed extractor** (added as `extract_fn.sh` in the scratchpad, logic
summarized): find the `.Lfunc_beginN:` label immediately after the named
function's own label, compute the matching `.Lfunc_endN:`, and stop there:

```bash
awk -v fn="probe_knn3" '
  $0 ~ "^"fn":" {grab=1}
  grab && /^\.Lfunc_begin[0-9]+:/ {n=$0; sub(/^\.Lfunc_begin/,"",n); sub(/:$/,"",n); endlabel=".Lfunc_end" n ":"}
  grab {print}
  grab && endlabel != "" && $0==endlabel {exit}
' target/release/examples/<cgu-name>.s
```

**Re-verified all substantive claims against the correctly-bounded
extraction — all held up:**
- `grep -c "search_level"` inside the corrected 569-line `probe_knn3` body:
  **0** (matches the original file-wide grep, which was never affected by
  the extraction bug since it ran over the whole `.s` file, not the
  mis-bounded excerpt).
- The `RawVec::grow_one::<Frame<f32>>` spill-growth call is still present
  exactly once, still correctly guarded by `cmpq $128, %rax; jb <fast-path>`
  immediately before it (confirmed at the corrected extraction's line
  170-179).
- `probe_knn_dyn`'s corrected extraction (633 real lines, was captured as
  ~2365) still shows the pre-existing `calloc`+`memset` pair as the
  `DynDim` per-query `vec![ZERO; dim]` scratch allocation (T1 §Q6b,
  unrelated to `FrameStack`, symmetric with C++), NOT the `FrameStack`
  spill path, confirmed by inspecting the surrounding instructions (a
  `testq %rbx,%rbx` / `calloc` / size check / `memset` sequence with no
  `128`-capacity comparison anywhere near it).

**One correction found and fixed:** the original report described the
FIRST-ATTEMPT (`[Frame::default(); 128]`) regression's fill mechanism as
"a ~4 KiB `memset` call". Regenerated that build (hand-reconstructed in a
throwaway worktree, `Frame`/`FrameStack` reverted to the eager
`Default`-array form, parity-checked — 183/0 lib tests still pass on that
reconstruction before extracting its asm) and re-extracted with the fixed
tool: the actual mechanism is a **vectorized zero-fill loop** (`vmovups`
stores into a 5 KiB stack scratch region) **followed by a 4096-byte
`memcpy`** from that zeroed region into the array — not a bare `memset`
call. This is materially the SAME finding (a real, per-query, ~4 KiB bulk
data-movement cost triggered by the eager array fill) and if anything
STRONGER evidence for the diagnosed regression (a zero-fill loop plus a
memcpy is more expensive than a bare memset would have been), but the
specific libc function name in the original report was wrong and is
corrected here. The two-`subq $4096`/one-`movq $0,(%rsp)`-probe-each stack
layout and the 8432-byte total reserved frame are both independently
reconfirmed unchanged.

### Item 7 — degenerate-tree spill heap-churn + capacity justification

Both added as an extended doc comment on `SEARCH_STACK_INLINE_CAPACITY`
in `search.rs` (quoted here, condensed):

- **Capacity 128 is ARBITRARY-WITH-HEADROOM, not derived from a
  principled depth-distribution analysis.** It was chosen (M2.5-T1's
  prototype) only as "comfortably above every depth this crate's own
  gate/xval workloads produce" (~13-17 for the standard 100k-point/leaf-10
  configuration). Task 6's own deliberately-adversarial degenerate-tree
  stress tests reach depth ~2 115 and ~16 794 — far beyond 128 regardless
  of what this constant is set to, so no "reasonable" capacity choice
  avoids the spill path for those trees; 128 was never meant to cover
  them, only the realistic case. No sweep over alternative capacities
  against realistic (non-adversarial) depth distributions has been done.
  Item 5's necessity A/B and item 2's capacity-32 test are both now
  cross-referenced from this doc comment as the closest available
  evidence (lowering to 32 does not recover the `radius`/`dim8` losses,
  so this constant is not a lever for that problem).
- **Spill heap-churn, documented next to the no-stack-overflow upgrade it
  trades against:** a query deeper than 128 pays REAL per-query heap
  reallocation cost (`overflow: Vec`'s normal doubling growth, repeated as
  depth grows past 128/256/512/...) — cost that did NOT exist under the
  old native-recursion form (zero heap allocation at any depth, only
  native call-stack frames, at the price of being vulnerable to a native
  stack overflow on a sufficiently degenerate tree on a small worker
  thread). This is judged a good trade — but it is explicitly flagged as a
  genuine give-back, not a free upgrade, and a workload running MANY
  queries against a tree deeper than 128 EVERY query would pay this
  repeatedly (`FrameStack` is not reused across queries).

### Full re-verification after all seven items

```
$ cargo test --workspace                              -> all "ok", 0 failed (359 total passes, unchanged)
$ RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release -- --ignored
  -> heavy_exponential_build_1m, heavy_exponential_build_1m_dim8,
     heavy_exponential_parallel_build_1m, heavy_query_degenerate_trees: all "ok"
  -> all 6 perf_gate_* tests: "ok" (thresholds pass -- radius/dim8 still
     comfortably under their gate margins, see item 2's ratios above)
  -> all 4 mutation canaries: "ok"
$ cargo clippy --workspace --all-targets -- -D warnings   -> clean
$ cargo build -p nanoflann-rs --no-default-features       -> clean
$ cargo test  -p nanoflann-rs --no-default-features       -> 179 passed, 0 failed
$ RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps  -> clean, 0 warnings
$ cargo +nightly miri test -p nanoflann-rs --lib -- search               -> 34/0/1, clean (Stacked Borrows)
$ MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test ... -- search -> 34/0/1, clean (Tree Borrows)
```

### Files changed this round

- `crates/nanoflann-rs/src/search.rs` — spill test query + instrumentation
  (`max_depth_seen`, `LAST_SEARCH_STACK_MAX_DEPTH`, `record_max_stack_depth!`),
  `Dist: Copy` bounds + `Frame` derive, destruction/panic invariant doc,
  `SEARCH_STACK_INLINE_CAPACITY` doc extended (arbitrary-with-headroom +
  heap-churn).
- `docs/EXPERIMENTS.md` — new "Miri" reproduction-commands subsection.
- `docs/agentic-development/reports/m2.5/task-2-report.md` —
  this section.
