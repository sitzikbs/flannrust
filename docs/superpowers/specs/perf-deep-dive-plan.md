# Performance Deep-Dive Plan

Synthesized from three independent expert analyses (Rust perf, nanoflann C++
comparison, repo-level context), cross-validated against the actual source on
branch `m2p6-rigor` and the existing `perf-opportunities.md`.

**Date:** 2026-09-09.  
**Tests baseline:** 203 passed, 0 failed (`cargo test -p flannrust --release`).  
**Perf baseline (Rust/C++ medians):** knn_fixed3 1.013, dyn_dim8 0.927,
radius 0.876, dim-32/64 knn 1.08-1.24.

---

## Executive summary

| Phase | Items | Expected combined gain | Effort |
|-------|-------|------------------------|--------|
| 1 -- Quick wins | 4 items | 5-15% on dim-3 knn; marginal elsewhere | < 1 hour each |
| 2 -- Medium effort | 4 items | 15-40% dim>=32; 5-20% high-dim; 3-5% all knn | 1-4 hours each |
| 3 -- Hard / infrastructure | 3 items | 5-15% all queries (PGO); 5-20% large-n (reorder); up to 30% k>=50 | 4+ hours each |

**Total realistic end-to-end expectation (non-additive):**
- dim-3 knn: 5-12% improvement (closes 1.013 gap and then some)
- dim-8 knn: 5-10% improvement
- dim-32/64 knn: 20-40% improvement (closes the 1.08-1.24 gap)
- Large-k (k>=50): up to 30% additional improvement
- All queries under PGO: additional 5-15%

**Hard constraint:** No change may alter L2/L1 search results (indices or
distances). L2Fma is already documented non-bit-exact; FMA-order changes
there are fine. All changes must work on stable Rust 1.98+. Gate:
`cargo test -p flannrust --release`.

---

## Phase 1: Quick wins (< 1 hour each, safe, high confidence)

### 1.1 `#[inline(always)]` on `l2_eval_row` and `l1_eval_row`

**Impact:** 3-8% on dim=3 knn; 1-3% at higher dims.  
**Confidence:** High -- standard LLVM inlining behavior, trivial change.  
**Difficulty:** Trivial (two-line change).  
**Risk:** Minimal. Slightly larger binary from monomorphization.

**Rationale:** `l2_eval_row` (metric.rs:245) is `#[inline]` (hint only). For
`ConstDim<3>`, multof4=0 and the ENTIRE computation is the remainder path
(6 indexed accesses). Without forced inlining, LLVM may not propagate the
constant `dim=3` through to eliminate bounds checks and dead-code the chunk
loop. With `#[inline(always)]`, the full call chain from
`search_level_hybrid -> L2::eval -> l2_eval_row` collapses and LLVM sees
dim=3 as constant.

**Files and exact changes:**

`crates/flannrust/src/metric.rs`:
```
Line 245: #[inline]        -->  #[inline(always)]
Line 146: #[inline]        -->  #[inline(always)]
```

Specifically:
- Line 245 (before `fn l2_eval_row`): change `#[inline]` to `#[inline(always)]`
- Line 146 (before `fn l1_eval_row`): change `#[inline]` to `#[inline(always)]`

**Test plan:**
- `cargo test -p flannrust --release` (all 203 tests pass)
- Verify with `cargo asm` that bounds checks are eliminated for `ConstDim<3>` path
- Run knn_fixed3 benchmark: expect 3-8% improvement
- Run dyn_dim8 benchmark: expect 1-3% improvement
- Run radius benchmark: verify no regression

**Dependencies:** None. Do this first.

---

### 1.2 Bounds check elimination in `l2_eval_row`/`l1_eval_row` remainder path

**Impact:** 2-5% on dim=3, defensive/robustness measure on top of 1.1.  
**Confidence:** High -- proven technique, makes the optimization non-fragile.  
**Difficulty:** Easy (sub-slicing before the remainder).  
**Risk:** Minimal. Identical computation, identical summation order.

**Rationale:** The remainder path (metric.rs:263-277) does individual
indexing: `query[d + 2]`, `row[d + 2]`, etc. For `ConstDim<3>` the ENTIRE
computation is this path. Even if 1.1's `#[inline(always)]` lets LLVM
eliminate the checks, this fix makes the optimization non-fragile across
compiler versions. For `DynDim(3)`, the bounds checks persist regardless
of inlining because dim is a runtime value -- this fix eliminates them.

**Files and exact changes:**

`crates/flannrust/src/metric.rs`, in `l2_eval_row` (around line 263-277):

Before:
```rust
    let d = multof4;
    let rem = dim - multof4;
    if rem >= 3 {
        let diff = query[d + 2] - row[d + 2];
        result = result + diff * diff;
    }
    if rem >= 2 {
        let diff = query[d + 1] - row[d + 1];
        result = result + diff * diff;
    }
    if rem >= 1 {
        let diff = query[d] - row[d];
        result = result + diff * diff;
    }
```

After:
```rust
    let d = multof4;
    let rem = dim - multof4;
    let qt = &query[d..d + rem];
    let rt = &row[d..d + rem];
    if rem >= 3 {
        let diff = qt[2] - rt[2];
        result = result + diff * diff;
    }
    if rem >= 2 {
        let diff = qt[1] - rt[1];
        result = result + diff * diff;
    }
    if rem >= 1 {
        let diff = qt[0] - rt[0];
        result = result + diff * diff;
    }
```

Same change for `l1_eval_row` (around line 168-179):

Before:
```rust
    let d = multof4;
    let rem = dim - multof4;
    if rem >= 3 {
        result = result + abs_t(query[d + 2] - row[d + 2], zero);
    }
    if rem >= 2 {
        result = result + abs_t(query[d + 1] - row[d + 1], zero);
    }
    if rem >= 1 {
        result = result + abs_t(query[d] - row[d], zero);
    }
```

After:
```rust
    let d = multof4;
    let rem = dim - multof4;
    let qt = &query[d..d + rem];
    let rt = &row[d..d + rem];
    if rem >= 3 {
        result = result + abs_t(qt[2] - rt[2], zero);
    }
    if rem >= 2 {
        result = result + abs_t(qt[1] - rt[1], zero);
    }
    if rem >= 1 {
        result = result + abs_t(qt[0] - rt[0], zero);
    }
```

**Test plan:**
- `cargo test -p flannrust --release` -- all bit-equality tests
  (`l2_eval_row_path_bit_equals_fallback_path_all_dims_*`) are the
  load-bearing gate
- Benchmark dim-3 and dim-8 knn
- Verify identical results with the existing multi-salt sweep tests

**Dependencies:** Best done immediately after 1.1 (combined effect).

---

### 1.3 Bounds check elimination in `compute_initial_distances`

**Impact:** 0.5-1% per query (runs once per query, not per leaf point).  
**Confidence:** High -- mechanical change.  
**Difficulty:** Trivial.  
**Risk:** Minimal.

**Rationale:** `compute_initial_distances` (search.rs:127-149) indexes
`query[i]`, `ctx.root_bbox[i]`, and `dists[i]` on every iteration. For
`DynDim`, LLVM cannot prove the slices are long enough from the assert at
search.rs:68. Sub-slicing at function entry establishes the bounds once.

**Files and exact changes:**

`crates/flannrust/src/search.rs`, in `compute_initial_distances` (line ~138):

Before:
```rust
    let mut dist = M::DistanceType::ZERO;
    for i in 0..ctx.dim.dim() {
        if query[i] < ctx.root_bbox[i].low {
```

After:
```rust
    let dim = ctx.dim.dim();
    let query = &query[..dim];
    let dists = &mut dists[..dim];
    let mut dist = M::DistanceType::ZERO;
    for i in 0..dim {
        if query[i] < ctx.root_bbox[i].low {
```

And replace `ctx.dim.dim()` in the loop bound with `dim` (already done by
the sub-slicing).

**Test plan:**
- `cargo test -p flannrust --release`
- Benchmark: marginal improvement, mainly code hygiene

**Dependencies:** None.

---

### 1.4 Use `point_row` in `contains_point` for `find_within_box`

**Impact:** 15-20% of box query time; zero impact on knn/radius.  
**Confidence:** High -- follows exact pattern already in `L2::eval`.  
**Difficulty:** Easy.  
**Risk:** Minimal.

**Rationale:** `contains_point` (search.rs:776-783) calls
`ds.point_component(idx, i)` per axis. Each call recomputes `idx * dim + i`
for FlatSlice (a multiply + bounds check per component). `point_row` computes
the base offset once.

**Files and exact changes:**

`crates/flannrust/src/search.rs`, `contains_point` function (lines 764-783):

Before:
```rust
fn contains_point<T, D, DS, M, Idx>(
    ctx: &SearchCtx<'_, T, D, DS, M, Idx>,
    bounds: &[Interval<T>],
    idx: Idx,
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
{
    for i in 0..ctx.dim.dim() {
        let point = ctx.ds.point_component(idx.to_usize(), i);
        if !bounds[i].contains(point) {
            return false;
        }
    }
    true
}
```

After:
```rust
fn contains_point<T, D, DS, M, Idx>(
    ctx: &SearchCtx<'_, T, D, DS, M, Idx>,
    bounds: &[Interval<T>],
    idx: Idx,
) -> bool
where
    T: Scalar,
    D: Dim,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexType,
{
    let dim = ctx.dim.dim();
    if let Some(row) = ctx.ds.point_row(idx.to_usize()) {
        if row.len() >= dim {
            for i in 0..dim {
                if !bounds[i].contains(row[i]) {
                    return false;
                }
            }
            return true;
        }
    }
    // Fallback for DataSources that don't implement point_row
    for i in 0..dim {
        let point = ctx.ds.point_component(idx.to_usize(), i);
        if !bounds[i].contains(point) {
            return false;
        }
    }
    true
}
```

**Test plan:**
- `cargo test -p flannrust --release`
- Benchmark `find_within_box` if a bench exists; otherwise verify via
  the existing `find_within_box_exact_traversal_order` and brute-force
  property tests in search.rs

**Dependencies:** None.

---

## Phase 2: Medium effort (1-4 hours, moderate risk)

### 2.1 SIMD L2 kernel (non-FMA) for the default L2 metric

**Impact:** 15-40% on dim>=32 knn queries; closes the 1.08-1.24x gap.  
**Confidence:** High -- the L2Fma SIMD infrastructure is already proven.  
**Difficulty:** Medium (unsafe intrinsics, per-arch gating).  
**Risk:** Low-medium. Well-contained: one function, one dispatch site.

**Rationale:** The default L2 metric has NO SIMD path. Only L2Fma dispatches
to simd.rs. LLVM verifiably will NOT auto-vectorize `l2_eval_row` (measured:
zmm count 0 even with `target-cpu=native` on Zen 5, per perf-opportunities.md
header). GCC does this widening automatically for the C++ equivalent. This is
the primary cause of the 1.08-1.24x regression at dim 32/64.

An AVX2 kernel using `_mm256_mul_pd` + `_mm256_add_pd` (no FMA) preserves
bit-exact results because the same mul+add operations are used with the same
summation order.

**Files and exact changes:**

1. `crates/flannrust/src/simd.rs`: Add new kernels and dispatch trait:
   - `l2_f64_avx2`, `l2_f32_avx2`: mirror `l2fma_f64_avx2`/`l2fma_f32_avx2`
     but replace `_mm256_fmadd_pd` with `_mm256_mul_pd` + `_mm256_add_pd`
   - `l2_f64_avx512`, `l2_f32_avx512`: same for AVX-512
   - New trait `L2Simd` (like `L2FmaSimd`) with dim>=16 gate for AVX2,
     dim>=32 for AVX-512
   - **Critical constraint:** horizontal reduction must sum lanes in the
     same order as the scalar path's `(d0*d0+d1*d1)+(d2*d2+d3*d3)` grouping

2. `crates/flannrust/src/metric.rs`, in `l2_eval_row` (line 246):
   After the existing function signature, before the scalar loop, add SIMD
   dispatch:
   ```rust
   #[cfg(target_arch = "x86_64")]
   {
       if let Some(d) = <T as crate::simd::L2Simd>::dispatch(query, row, dim) {
           return d;
       }
   }
   ```
   (Same pattern as L2Fma in metric.rs:383-389, but inside `l2_eval_row`
   instead of `L2Fma::eval`.)

   Alternatively, dispatch from `impl_l2` macro (metric.rs:293-297) after
   the `point_row` check, before calling `l2_eval_row`. This matches the
   L2Fma pattern more closely.

**Note:** A detailed spec already exists at
`docs/superpowers/specs/simd-l2-kernel-spec.md`.

**Test plan:**
- Existing `l2_eval_row_path_bit_equals_fallback_path_all_dims_*` tests
  (multi-salt, all dims 1-64) gate bit-exactness
- Add SIMD-specific property test: SIMD dispatch matches scalar for all
  dims in BIT_EQ_DIMS x SALTS (like the existing `l2fma_simd_matches_scalar`
  tests)
- A/B benchmark all dims: dim-3/8 must NOT regress (dim-gate prevents SIMD
  dispatch at low dims); dim-32/64 must show 15-40% improvement

**Dependencies:** None, but benefits from 1.1+1.2 first (so the scalar
fallback path at low dims is also optimized).

---

### 2.2 Early-exit distance computation (`eval_bounded`) for high-dim

**Impact:** 5-20% at dim>=32 knn; near-zero at dim<=8.  
**Confidence:** Medium-high -- FAISS uses this technique; never been in nanoflann.  
**Difficulty:** Medium (trait change, two call site changes).  
**Risk:** Low-medium. Non-breaking API change (default method).

**Rationale:** Neither nanoflann C++ nor flannrust checks `worst_dist`
mid-distance-computation. For dim>=32, many candidate points exceed
`worst_dist` after only 2-3 chunks (8-12 dimensions). Adding a check after
each 4-element chunk enables early termination, skipping 75-87% of work for
distant points.

**Files and exact changes:**

1. `crates/flannrust/src/metric.rs`, Distance trait (line 23-47):
   Add default method:
   ```rust
   /// Early-exit distance: returns `Some(dist)` if `dist < worst_dist`,
   /// `None` if the partial sum already exceeds `worst_dist`. Default
   /// impl calls `eval` then checks -- custom metrics get the benefit
   /// for free. Override for chunked kernels (L2/L1) to check mid-loop.
   fn eval_bounded<DS: DataSource<T> + ?Sized, D: Dim>(
       &self,
       query: &[T],
       ds: &DS,
       idx: usize,
       d: D,
       worst_dist: Self::DistanceType,
   ) -> Option<Self::DistanceType> {
       let dist = self.eval(query, ds, idx, d);
       if dist < worst_dist { Some(dist) } else { None }
   }
   ```

2. `crates/flannrust/src/metric.rs`: Add `l2_eval_row_bounded` variant:
   Same as `l2_eval_row` but checks `if result > worst { return None; }`
   after each 4-wide chunk in the main loop. The remainder path has no
   check (too few elements to matter).

3. `crates/flannrust/src/metric.rs`, `impl_l2` macro: Override
   `eval_bounded` to use `l2_eval_row_bounded` when `point_row` is
   available.

4. `crates/flannrust/src/search.rs`, leaf scan (line 430-431):
   Before:
   ```rust
   let dist = ctx.metric.eval(query, ctx.ds, accessor.to_usize(), ctx.dim);
   if dist < result.worst_dist() && !result.add_point(dist, accessor) {
   ```
   After:
   ```rust
   if let Some(dist) = ctx.metric.eval_bounded(
       query, ctx.ds, accessor.to_usize(), ctx.dim, result.worst_dist()
   ) {
       if !result.add_point(dist, accessor) {
           return false;
       }
   ```

   Same change at search.rs:592-597 (explicit-stack leaf scan).

**Test plan:**
- `cargo test -p flannrust --release` -- all existing brute-force property
  tests gate correctness (the pruned points would have been rejected anyway)
- Add a targeted test: verify that for dim=64, eval_bounded returns
  `Some(d)` iff `d < worst_dist`, with a known worst_dist threshold
- Benchmark dim-32 and dim-64 knn: expect 5-20% improvement
- Benchmark dim-3 knn: expect no change (remainder-only path, no checks)

**Dependencies:** Best done after 2.1 (SIMD kernels should also get bounded
variants for maximum effect at high dim).

---

### 2.3 Cache `worst_dist` as a field in `KnnResultSet`

**Impact:** 3-5% on all knn queries.  
**Confidence:** Medium-high -- removes a branch + array load per leaf point.  
**Difficulty:** Easy.  
**Risk:** Very low.

**Rationale:** `KnnResultSet::worst_dist()` (result_set.rs:143-150) has a
branch on every call: `if self.count < self.dists.len() || self.count == 0
{ D::MAX } else { self.dists[self.count - 1] }`. Called for EVERY point in
EVERY leaf (search.rs:430/597). Caching as a field eliminates the branch,
the `.len()` load, and the `count-1` index computation.

**Files and exact changes:**

`crates/flannrust/src/result_set.rs`:

1. Add field to `KnnResultSet` struct (line 120-125):
   ```rust
   pub struct KnnResultSet<'a, D, Idx, TB = KeepInsertionOrder> {
       indices: &'a mut [Idx],
       dists: &'a mut [D],
       count: usize,
       cached_worst: D,  // <-- NEW
       _tb: PhantomData<TB>,
   }
   ```

2. Initialize in `new()` (line 127-137):
   ```rust
   pub fn new(indices: &'a mut [Idx], dists: &'a mut [D]) -> Self {
       assert_eq!(indices.len(), dists.len(), "indices/dists length mismatch");
       Self {
           indices,
           dists,
           count: 0,
           cached_worst: D::MAX,  // <-- NEW
           _tb: PhantomData,
       }
   }
   ```

3. Simplify `worst_dist()` (line 143-150):
   ```rust
   #[inline]
   fn worst_dist(&self) -> D {
       self.cached_worst
   }
   ```

4. Update `cached_worst` in `add_point()` (after line 156):
   ```rust
   #[inline]
   fn add_point(&mut self, dist: D, index: Idx) -> bool {
       self.count =
           add_point_to_sorted::<D, Idx, TB>(self.indices, self.dists, self.count, dist, index);
       self.cached_worst = if self.count < self.dists.len() || self.count == 0 {
           D::MAX
       } else {
           self.dists[self.count - 1]
       };
       true
   }
   ```

5. Same change for `RknnResultSet` (lines 174-228):
   - Add `cached_worst: D` field, initialize to `max_radius` in `new()`
   - Simplify `worst_dist()` to `self.cached_worst`
   - Update in `add_point()`: `if self.count < self.dists.len() || self.count == 0 { self.max_radius } else { self.dists[self.count - 1] }`

**Test plan:**
- `cargo test -p flannrust --release` -- existing
  `test_knn_worst_dist_transition` and `test_rknn_worst_dist_transition`
  directly gate the cached value's correctness
- All brute-force property tests pass
- Benchmark all knn queries: expect 3-5% improvement

**Dependencies:** None.

---

### 2.4 Bounds check elimination for `dists[idx]` in traversal

**Impact:** 0.5-2% on DynDim search paths; zero for ConstDim.  
**Confidence:** Medium -- small per-node savings, but many nodes per query.  
**Difficulty:** Easy (one assert or sub-slice).  
**Risk:** Minimal.

**Rationale:** In `search_level_hybrid` (search.rs:458-468), `dists[idx]` is
indexed 3 times where `idx = node.split_dim()`. For DynDim, `idx < dim` is
an invariant but LLVM cannot prove it. Adding one assert before the first
access lets LLVM eliminate all subsequent checks.

**Files and exact changes:**

`crates/flannrust/src/search.rs`, in `search_level_hybrid` (before line 458):

Add:
```rust
    debug_assert!(idx < dists.len(), "split_dim >= dim: invariant violated");
    // Help LLVM: sub-slice dists to dim length once.
```

Or preferably, at the function entry after `let idx = node.split_dim();`
(line 442), add:
```rust
    assert!(idx < dists.len());
```

This one assert lets LLVM prove all `dists[idx]` accesses in lines 458-468
are in-bounds.

Same pattern in `search_level_explicit` (line 608 area, where `idx` is read
and `dists[idx]` is accessed).

**Test plan:**
- `cargo test -p flannrust --release`
- Benchmark DynDim knn: expect marginal improvement

**Dependencies:** None.

---

## Phase 3: Hard (> 4 hours, needs careful testing)

### 3.1 PGO (Profile-Guided Optimization) build configuration

**Impact:** 5-15% with zero code changes.  
**Confidence:** High -- well-established for branch-heavy tree traversal.  
**Difficulty:** Medium (toolchain/CI plumbing, not code).  
**Risk:** Low code risk, medium infrastructure effort.

**Rationale:** `search_level_hybrid` and sorted insert are branch-heavy with
data-dependent but skewed branches. PGO teaches LLVM actual branch weights,
enabling optimal branch layout and inlining decisions. Already identified in
perf-opportunities.md as opportunity #4.

**Files and exact changes:**

No source code changes. Build pipeline steps:
```bash
# Step 1: Instrumented build
RUSTFLAGS='-Cprofile-generate=/tmp/pgo-data' cargo build --release

# Step 2: Training run (use the benchmark suite)
cargo bench --release  # or: run the xval benchmark suite

# Step 3: Merge profiles
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/

# Step 4: PGO-optimized rebuild
RUSTFLAGS='-Cprofile-use=/tmp/pgo-data/merged.profdata' cargo build --release
```

Document in a `Makefile` or CI script. Ship PGO'd binaries only in Python
wheels (crates.io users compile themselves).

**Training profile must be representative:** run ALL bench_knn groups (dim-3,
dim-8, dim-16, dim-32, plus knn_fixed3, radius, box queries) and both f32/f64,
with varied k (10, 50, 100). Training on a single dim/k biases branch weights
toward that shape and may hurt other workloads.

**Also A/B:** `lto = "fat"` vs current `lto = "thin"` (0-3% expected;
measure build time tradeoff).

**Test plan:**
- `cargo test -p flannrust --release` after PGO build
- Benchmark all queries: expect 5-15% across the board
- **Fairness note:** C++ oracle is not PGO'd, so PGO numbers should be
  reported as a separate profile

**Dependencies:** None, but most impactful after Phases 1+2 (PGO optimizes
the final code layout).

---

### 3.2 Opt-in leaf-contiguous dataset reorder

**Impact:** 5-20% on dim>=8 knn queries with large datasets (>100k).  
**Confidence:** Medium -- well-known technique (FLANN, scipy), but needs
API design.  
**Difficulty:** Medium-hard (touches DataSource plumbing).  
**Risk:** Medium. 2x memory cost mitigated by opt-in.

**Rationale:** The leaf scan accesses points through `vind[]`, a random
permutation. For datasets exceeding L2 cache, each leaf point is a potential
cache miss. Reordering into `vind` order makes every leaf scan read
contiguous memory. Already identified in perf-opportunities.md as
opportunity #3.

**Files and exact changes:**

1. `crates/flannrust/src/tree.rs`: Add `.reorder_points(true)` option to
   builder. After build completes, if enabled:
   - Allocate new data buffer of size `n * dim`
   - Iterate `vind`, copy each point's row into new buffer in vind order
   - Replace DataSource with reordered copy (needs `OwnedRows` or similar)
   - Update vind to identity (0,1,2,...,n-1)
   - Keep original-index map for translating result indices back

2. `crates/flannrust/src/data_source.rs`: May need a new
   `ReorderedDataSource` wrapper or extend `OwnedRows`.

**Test plan:**
- All existing brute-force property tests with `reorder_points(true)`
- Verify result indices map back correctly to original dataset indices
- Benchmark dim-8/32/64 knn with n=100k: expect 5-20% improvement
- Benchmark dim-3 with n=10k: verify no regression (data fits in cache)

**Dependencies:** None, but should be done after 2.1 (SIMD kernel) to
measure the combined effect.

---

### 3.3 Binary search insertion for large k in `add_point_to_sorted`

**Impact:** Negligible for k<=10; up to 30% for k>=50.  
**Confidence:** Medium -- theoretical analysis is sound, but the common
benchmark gate uses k=10 where this has zero effect.  
**Difficulty:** Easy code, moderate testing (TieBreak policy integration).  
**Risk:** Low-medium.

**Rationale:** `add_point_to_sorted` (result_set.rs:87-116) does O(k) linear
scan per insertion. For k>=50, branch misprediction dominates: ~50 iterations
with ~50% misprediction rate. Binary search reduces to O(log k) comparisons
+ one `copy_within` (memmove).

**Files and exact changes:**

`crates/flannrust/src/result_set.rs`, `add_point_to_sorted` (line 87-116):

Add a threshold branch:
```rust
fn add_point_to_sorted<D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak>(
    indices: &mut [Idx],
    dists: &mut [D],
    count: usize,
    dist: D,
    index: Idx,
) -> usize {
    let capacity = dists.len();

    // ponytail: linear scan for small k (cache-friendly, no branch overhead);
    // binary search + copy_within for large k (O(log k) vs O(k) compares)
    if capacity >= 32 && count > 0 {
        // Binary search for insertion point
        let search_end = count.min(capacity);
        let pos = dists[..search_end].partition_point(|&d| !TB::shift(d, /* ... */));
        // ... shift via copy_within, insert at pos
    } else {
        // Existing linear scan (unchanged)
        let mut i = count;
        while i > 0 {
            // ...existing code...
        }
    }
    // ...existing count update...
}
```

**Complication:** `TB::shift` takes both `(prev_d, prev_i, d, i)`, not just
distances. For `KeepInsertionOrder` (distance-only), `partition_point` works
directly. For `SmallestIndexWins`, the comparison is not monotonic on
distance alone -- need to binary search on distance, then linear scan within
the equal-distance range.

**Test plan:**
- Existing tests: `test_sorted_insert`, `test_ties_keep_insertion_order`,
  `test_ties_smallest_index_wins`
- Add k=100 test with tied distances for both TieBreak policies
- Benchmark with k=50 and k=100: expect 15-30% improvement
- Benchmark with k=10: verify zero regression

**Dependencies:** None.

---

## Not recommended (considered and rejected)

### Arena `Vec::with_capacity` pre-sizing in build

**Reason:** Already measured WORSE (+3-6% Rust median).  
**Evidence:** m2.6/task-4-report section 5; listed in perf-opportunities.md
anti-opportunities table.

### Fat LTO (`lto = "fat"`)

**Reason:** Expected 0-3% gain for potentially 2-5x build time increase. All
hot-path code is in the same crate, so thin LTO already handles cross-module
inlining. Worth A/B-ing as part of PGO work (3.1) but not as a standalone
change.

### `get_unchecked` in `KnnResultSet::worst_dist` / sorted insert

**Reason:** Bounded at <=0.5%, not where time goes.  
**Evidence:** m2.5/task-1-report section 6a.

### Plain-array `FrameStack` / tuning `SEARCH_STACK_INLINE_CAPACITY`

**Reason:** 4KiB memset/query regression (1.06->1.14). Capacity 32 does NOT
recover the radius/dyn-dim8 losses.  
**Evidence:** m2.5/task-2-report section 6a; documented in search.rs lines
137-185.

### Redundant `dist < radius` check in `RadiusResultSet::add_point`

**Reason:** The branch predictor handles always-true branches with near-zero
cost (~1 cycle). Removing it saves <0.5% on radius search and risks
correctness for direct API callers.

### Specialize `eps_error == 1.0` to skip multiply in prune test

**Reason:** One FP multiply per interior node (~50-500 per query) saves
~0.03-1%. The branch to check `eps_error == 1.0` costs nearly as much as it
saves. Too clever for too little gain.

### DynDim per-query scratch allocation elimination (standalone)

**Reason:** 0.5-2% per DynDim query. Subsumed by the batch query API (3.2
or a future `knn_search_batch` that reuses scratch). Not worth a standalone
API method (`knn_search_into`) for such marginal gain.

### Software prefetch in leaf scan (standalone)

**Reason:** 1-3% on large datasets. Subsumed by opportunity 3.2 (leaf
reorder), which solves the same cache-miss problem more completely. Worth
doing as a cheap control probe if 3.2 is deferred, but not worth the
`DataSource` trait surface area increase on its own. If 3.2 is taken,
prefetch is redundant (contiguous data + hardware prefetcher).

### Hybrid recursion recovery

**Reason:** Already implemented. The current code (search.rs:402-470) uses
`search_level_hybrid` with native recursion for the first 96 levels, falling
back to explicit-stack `search_level_explicit` for deeper trees. This was
perf-opportunities.md opportunity #5, and it is DONE.

---

## Implementation order

```
Phase 1 (do first, one PR each, immediate):
  1.1  #[inline(always)] on l2_eval_row/l1_eval_row
  1.2  Bounds check elimination in remainder path    (can combine with 1.1)
  1.3  Bounds check elimination in compute_initial_distances
  1.4  point_row in contains_point

Phase 2 (after Phase 1 lands and benchmarks confirm):
  2.1  SIMD L2 kernel (non-FMA)                     -- biggest impact item
  2.2  Early-exit distance (eval_bounded)            -- after 2.1
  2.3  Cache worst_dist in KnnResultSet              -- independent
  2.4  Bounds check elimination in traversal dists   -- independent

Phase 3 (after Phase 2, separate feature branches):
  3.1  PGO build configuration                       -- independent
  3.2  Leaf-contiguous dataset reorder                -- biggest design work
  3.3  Binary search for large k                     -- independent
```

Items within a phase are independent unless noted. Phase 1 items 1.1 and 1.2
can and should be a single PR (they target the same lines and their combined
effect is what matters). Phase 2 item 2.1 is the single highest-impact item
for closing the known dim-32/64 gap.
