# nanoflann-rs — Milestone 1 Implementation Plan (static kd-tree + NN search)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust port of nanoflann v1.12.1's static kd-tree (`KDTreeSingleIndexAdaptor`) that produces identical search results (cross-validated against the C++ compiled into the same test/bench binaries) and measurably beats it on build and query speed.

**Architecture:** One library crate with a monomorphized, zero-dynamic-dispatch search core (generic over metric, dim, result set, and an inlined point-filter seam for later dynamic adaptors); arena-based `Vec<Node>` with u32 child indices instead of pooled pointers; explicit-stack sequential build and a deterministic rayon parallel build; a `cc`-compiled extern-C wrapper over vendored `nanoflann.hpp` as in-process oracle and benchmark baseline.

**Tech Stack:** Rust stable, rayon (feature-gated), criterion, proptest, rand+rand_chacha (seeded), cc + C++17 for the reference wrapper.

**Spec:** The user's "Rev 2 parity review" (in conversation) **as corrected by the Corrections section below** — corrections override Rev 2 wherever they conflict. C++ reference: nanoflann tag `1.12.1` (commit `7812aa0`), cloned at `/tmp/claude-1000/-home-sitzikbs-dev-flannrust/f431bc5b-728e-4e37-bb64-07067ac0e803/scratchpad/nanoflann` (vendor `include/nanoflann.hpp` into the repo in T0 — the scratchpad is session-temporary).

## Global Constraints

- Match nanoflann 1.12.1 behavior exactly unless listed as a deliberate deviation; deviations are documented in code docs and README.
- L2/L2_Simple distances and radii are SQUARED everywhere; SO2 is an unsquared angle.
- Radius search strictly `dist < radius`; box search inclusive on boundaries; box search never sorts and takes no search params.
- Default `leaf_max_size = 10`; default tree index type `u32`; default tie-break = insertion order (C++ default, no `NANOFLANN_FIRST_MATCH`).
- No batch APIs, no Morton ordering, no SIMD intrinsics in M1.
- `cargo build --no-default-features` (no rayon) must always compile; rayon usage confined to `build_parallel.rs`.
- Never pass fast-math flags to either compiler; benches use `target-cpu=native` on both sides.
- TDD throughout; commit per task; `cargo test --workspace` green at every commit.

## Context

We want to know whether a Rust nanoflann can be faster. A prior session produced a plan + "Rev 2" parity addendum for full parity with nanoflann v1.12.1. This session (a) fact-checked every Rev 2 claim against the actual 1.12.1 source, and (b) narrowed scope with the user:

- **Scope (user decision):** Static kd-tree only — build (sequential + parallel), `knn_search`, `rknn_search`, `radius_search`, `find_within_box`, metrics L1/L2/L2_Simple/SO2/SO3. Dynamic/incremental/MT adaptors and serialization are later milestones; M1 must not preclude them (search core takes an inlined point-filter hook = C++ CRTP `isActive()`; `Distance` keeps per-axis `accum_dist`).
- **Benchmark harness (user decision):** vendored `nanoflann.hpp` + thin extern-C wrapper compiled via `cc` into the same binaries. Criterion benches and cross-validation tests share it.

## Corrections to the Rev 2 document (verified against 1.12.1 source)

Rev 2 was largely accurate — including items that looked fabricated (the incremental adaptor, MT adaptor, `NANOFLANN_NODE_ALIGNMENT`, `NFLI` magic all exist in 1.12.1). The real errors, superseded by this plan:

1. **A3 (box sort) — wrong.** `findWithinBox` takes **no** `SearchParameters`, **never** calls `sort()`, and returns `result.size()` (a count), not `full()`. `BoxResultSet::sort()` (by index) fires only if called manually. `RKNNResultSet::sort()` is an empty no-op like KNN's. → Our `find_within_box(bbox, out) -> usize` has no params and returns indices in traversal order.
2. **A4 (RKNN seed) — vestigial in 1.12.1.** `worstDist()` returns the `maximumSearchDistanceSquared` member directly when not full; the `dists[capacity-1]` seeding no longer carries the radius. Port the semantics (worst = radius² until full, then `dists[count-1]`), not the seeding trick. No squaring happens inside — callers pass an already-squared radius for L2.
3. **A5 — box path never consults `full()`**; returns count, incl. mid-loop early-out.
4. **A6 (empty dynamic index) — wrong.** The user-facing forest performs **no** emptiness checks; with radius/box result sets it returns **true** on an empty index. Only the internal `_`-suffixed sub-tree returns false in both cases. (M2 migration-table note.)
5. **A8 (`size_at_index_build_`) — dead code.** Written in 6 places, read nowhere, no accessor. Stale-queries-after-growth behavior comes from `vAcc_`/`size_` snapshotted at build. → Do **not** port the field.
6. **A10 addendum:** the dynamic forest has **no** `knnSearch`/`radiusSearch`/`rknnSearch` — only `findNeighbors`/`addPoints`/`removePoint`/`getAllIndices`. (M2 note.)
7. **B4 (tombstone merge) — conditional:** `removedPoints_[e] = pos` is the `else` of `if (treeIndex_[e] != -1) treeIndex_[e] = pos;`. (M2 note.)
8. **C2 resolved:** MT adaptor API fully enumerated (persistent worker thread, op-log + snapshot + atomic swap in `integrateIfReady()`, two condition variables, `std::exception_ptr`, defaults `rebuild_growth=1.3`, `min_rebuild_size=10000`). Findings go into `docs/nanoflann-notes.md` (T0).
9. **C4 nuance:** under `NANOFLANN_NO_THREADS` the throw fires whenever normalized `n_thread_build_ != 1`; `0` on a single-core box normalizes to 1 and does not throw.
10. **C5 nuance:** `ResultItem` is `{first: index, second: distance}`; standard-layout is guaranteed by docs + an example static_assert, not a library assert. Ours: `#[repr(C)] ResultItem { index, distance }` + layout test.
11. **Load-bearing facts Rev 2 missed** (verified line refs into 1.12.1):
    - `accum_dist` is used by `searchLevel` (1264/1270) and `computeInitialDistances` (1595) for per-axis bounds → metrics **must** be per-axis decomposable (hard `Distance`-trait constraint).
    - Interior nodes store **two** split bounds `divlow = left_bbox[cutfeat].high`, `divhigh = right_bbox[cutfeat].low` (gap representation); descent picks child via `(q[f]-divlow + q[f]-divhigh) < 0`; incremental mindist update `mindist + cut_dist - dists[idx]` with save/restore.
    - `planeSplit` (1565) is a Dutch-national-flag 3-way partition → `lim1/lim2`; split index = `lim1 > n/2 ? lim1 : lim2 < n/2 ? lim2 : n/2`. `middleSplit_` (1481): candidate dims within `(1−1e−5)·max_span`, `cutfeat` = first with strictly greater spread, cutval = midpoint clamped to `[min_elem, max_elem]`.
    - eps: `epsError = 1 + eps`, prune test `mindist * epsError <= worstDist()` (multiplicative on node bound).
    - Leaf loop tests `dist < worstDist()` **strictly**; ties keep traversal order via strict `>` shift in `addPointToSortedResultSet` (252).
    - Tree `IndexType` defaults `u32` in 1.12.1; result sets default `size_t`; `searchLevel` casts silently.
    - C++ parallel build (1423) offloads only the **right** subtree (`std::async` when `++thread_count < n_thread_build_`), mutex only around pool allocation; partitioning precedes spawning, so parallel and sequential builds produce **identical trees** → exact cross-validation is possible.
    - L2/L1 kernels: `restrict` pointers, `(d0²+d1²)+(d2²+d3²)` + descending remainder switch (d+2, d+1, d+0). L2_Simple plain loop. SO2 = unsquared angle of last dim only, single-shot ±π wrap, result in [0, π]. SO3 delegates to L2_Simple.
    - `findWithinBox` (2030) is the only iterative walk (explicit stack, pushes child1 then child2 → **child2 visited first**); descent pruning inclusive (`low <= divlow` / `high >= divhigh`). `searchLevel`/`divideTree` are unbounded recursion → Rev 2's C1 stack concern confirmed.
    - All-identical points are benign (DNF partition → `index = count/2`, balanced). The degenerate-depth killer is **exponentially spaced** coordinates.

---

## 1. Workspace layout

```
flannrust/
├── Cargo.toml                        # [workspace] resolver="2", members = crates/*
├── rust-toolchain.toml               # pin current stable
├── .cargo/config.toml                # bench profile notes; RUSTFLAGS target-cpu=native documented
├── docs/nanoflann-notes.md           # verification findings (MT/incremental API, stale spots) for M2+
├── crates/
│   ├── nanoflann-rs/                 # THE library
│   │   ├── Cargo.toml                # features: default=["parallel"]; parallel=["dep:rayon"]
│   │   └── src/
│   │       ├── lib.rs                # docs, re-exports, feature gates
│   │       ├── scalar.rs             # Scalar + DistanceValue traits (f32/f64)
│   │       ├── dim.rs                # Dim trait (GAT array-or-vec), ConstDim<N>, DynDim
│   │       ├── data_source.rs        # DataSource trait + &[[T;N]] and flat-slice adaptors
│   │       ├── metric.rs             # Distance trait; L1, L2, L2Simple, SO2, SO3
│   │       ├── result_set.rs         # ResultSet trait; Knn/Rknn/Radius/Box sets; ResultItem #[repr(C)]; TieBreak
│   │       ├── node.rs               # arena Node + leaf sentinel
│   │       ├── bbox.rs               # Interval<T>, compute_bounding_box
│   │       ├── build.rs              # sequential explicit-stack builder (middle_split, plane_split, finalize)
│   │       ├── build_parallel.rs     # cfg(feature="parallel"): rayon::join builder, local-arena merge
│   │       ├── search.rs             # search_level, compute_initial_distances, box-walk core
│   │       ├── filter.rs             # PointFilter + AcceptAll (future isActive seam)
│   │       ├── params.rs             # SearchParams, BuildThreads
│   │       └── tree.rs               # KdTree, KdTreeBuilder, public search methods
│   ├── nanoflann-ref/                # C++ oracle behind FFI (publish=false)
│   │   ├── Cargo.toml                # [build-dependencies] cc = "1"
│   │   ├── build.rs                  # cc: cpp(true), c++17, -O3, -march=native
│   │   ├── cpp/nanoflann.hpp         # vendored verbatim from tag 1.12.1
│   │   ├── cpp/wrapper.cpp           # extern "C" surface (§3)
│   │   └── src/lib.rs                # unsafe decls + safe RAII RefIndex<f32/f64> wrappers
│   └── xval/                         # cross-validation + benches (publish=false)
│       ├── src/lib.rs                # seeded data generators, ULP/tie-group comparators
│       ├── tests/{xval_knn,xval_radius_box,xval_build}.rs
│       └── benches/{bench_build,bench_knn,bench_radius}.rs
```

Unit tests live inline per module plus `crates/nanoflann-rs/tests/behavior.rs` for contract tests that don't need the C++ oracle.

## 2. Public API (exact sketch)

```rust
// scalar.rs
pub trait Scalar: Copy + PartialOrd + Sub<Output=Self> + Add<Output=Self> + Zero + 'static {}
pub trait DistanceValue: Copy + PartialOrd + Add<Output=Self> + Sub<Output=Self> + Zero + 'static { const MAX: Self; }
// impls for f32/f64

// dim.rs — mirrors C++ array_or_vector / if constexpr(DIM>0)
pub trait Dim: Copy + Send + Sync + 'static {
    type Array<T: Copy + Default + 'static>: AsRef<[T]> + AsMut<[T]> + Clone;
    fn dim(self) -> usize;                    // #[inline]; constant-folds for ConstDim
    fn filled<T: Copy + Default + 'static>(self, v: T) -> Self::Array<T>;
}
pub struct ConstDim<const N: usize>;          // Array<T> = [T; N]
pub struct DynDim(pub usize);                 // Array<T> = Vec<T>

// data_source.rs — zero-copy, = kdtree_get_point_count / kdtree_get_pt / kdtree_get_bbox
pub trait DataSource<T: Scalar> {
    fn point_count(&self) -> usize;
    fn point_component(&self, idx: usize, dim: usize) -> T;
    fn fill_bbox(&self, _bbox: &mut [Interval<T>]) -> bool { false }
}
impl<T: Scalar, const N: usize> DataSource<T> for &[[T; N]] { ... }
pub struct FlatSlice<'a, T> { pub data: &'a [T], pub dim: usize }  // row-major

// metric.rs — instances are values (stateful metrics OK; builder takes the instance)
pub trait Distance<T: Scalar>: Send + Sync {
    type DistanceType: DistanceValue;
    fn eval<DS: DataSource<T>>(&self, query: &[T], ds: &DS, idx: usize, dim: usize) -> Self::DistanceType;
    fn accum_dist(&self, a: T, b: T, axis: usize) -> Self::DistanceType;   // per-axis, used for bounds
}
pub struct L2; pub struct L2Simple; pub struct L1; pub struct SO2; pub struct SO3;
// L2/L1 replicate the C++ unroll: (d0²+d1²)+(d2²+d3²) + remainder added in order d+2, d+1, d+0.

// result_set.rs
#[repr(C)] pub struct ResultItem<Idx, D> { pub index: Idx, pub distance: D }  // = first/second
pub trait ResultSet<D: DistanceValue, Idx: Copy> {
    fn worst_dist(&self) -> D;
    fn add_point(&mut self, dist: D, index: Idx) -> bool;  // built-ins always return true (matches C++)
    fn full(&self) -> bool; fn size(&self) -> usize;
    fn sort(&mut self) {}                                  // gated on SearchParams::sorted in find_neighbors
}
pub trait TieBreak: 'static {}            // = NANOFLANN_FIRST_MATCH as a type
pub struct KeepInsertionOrder;            // default: shift only while prev_d > d (strict)
pub struct SmallestIndexWins;             // also shift on == with prev_i > i
pub struct KnnResultSet<'a, D, Idx, TB = KeepInsertionOrder> { /* &mut slices, count */ }
pub struct RknnResultSet<'a, D, Idx, TB = KeepInsertionOrder> { /* + max_radius (already squared for L2) */ }
pub struct RadiusResultSet<'a, D, Idx> { /* radius, &mut Vec<ResultItem>; add: strict dist < radius; full()=true */ }
// worst_dist: Knn → D::MAX until full; Rknn → max_radius until full; then dists[count-1].

// filter.rs — the M2 dynamic-adaptor seam (= CRTP isActive)
pub trait PointFilter<Idx: Copy> { fn is_active(&self, idx: Idx) -> bool; }
pub struct AcceptAll;   // #[inline(always)] true — erased in the static path

// params.rs
pub struct SearchParams { pub eps: f32, pub sorted: bool }   // Default: eps 0.0, sorted true
pub enum BuildThreads { Sequential, Auto, Threads(NonZeroU32) }  // non-Sequential needs feature "parallel"

// tree.rs
pub struct KdTreeBuilder<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder> { ... }
impl KdTreeBuilder {
    pub fn new(dim: D, dataset: DS) -> ...;               // defaults: L2, leaf 10, Sequential
    pub fn with_metric<M2>(self, metric: M2) -> ...;      // takes the INSTANCE (stateful metrics)
    pub fn leaf_max_size(self, n: usize) -> Self;
    pub fn threads(self, t: BuildThreads) -> Self;
    pub fn tie_break<TB2: TieBreak>(self) -> ...;
    pub fn build(self) -> KdTree<...>;                    // infallible; empty dataset → empty index
}
pub struct KdTree<T, D, DS, M, Idx = u32, TB = KeepInsertionOrder> {
    dataset: DS, metric: M, dim: D,
    vind: Vec<Idx>,                       // = vAcc_, permuted; snapshotted at build (stale-growth behavior falls out)
    nodes: Vec<Node<M::DistanceType>>,    // arena, root = 0 when non-empty
    root_bbox: D::Array<Interval<T>>, leaf_max_size: usize, size: usize,
}
impl KdTree {
    pub fn size(&self) -> usize; pub fn dim(&self) -> usize; pub fn used_memory_bytes(&self) -> usize;
    pub fn knn_search(&self, q: &[T], out_i: &mut [Idx], out_d: &mut [M::DistanceType]) -> usize;
    pub fn knn_search_with(&self, q, out_i, out_d, params: &SearchParams) -> usize;  // additive: C++ static has no params
    pub fn rknn_search(&self, q: &[T], radius: M::DistanceType, out_i, out_d) -> usize;
    pub fn radius_search(&self, q: &[T], radius: M::DistanceType,
                         out: &mut Vec<ResultItem<Idx, M::DistanceType>>, params: &SearchParams) -> usize;
    pub fn find_within_box(&self, bbox: &[Interval<T>], out: &mut Vec<Idx>) -> usize;  // no params, never sorts
    pub fn find_neighbors<R: ResultSet<...>>(&self, r: &mut R, q: &[T], params: &SearchParams) -> bool; // returns r.full()
}

// node.rs — 20 B (f32) / 32 B (f64) vs C++ ~48 B
#[repr(C)] pub(crate) struct Node<TD> {
    a: u32, b: u32,        // leaf: [left,right) into vind; interior: child arena indices
    divfeat: u32,          // leaf sentinel = u32::MAX
    divlow: TD, divhigh: TD,
}
```

## 3. FFI oracle (`nanoflann-ref`)

`cpp/wrapper.cpp` — runtime-dim (`DIM=-1`, `IndexType=uint32_t`, `NANOFLANN_FIRST_MATCH` undefined) instantiations for all 5 metrics via a tagged variant, plus fixed-DIM=3 L2 f32/f64 entry points for "beat their best config" benches:

```c
enum nfr_metric { NFR_L1, NFR_L2, NFR_L2_SIMPLE, NFR_SO2, NFR_SO3 };
nfr_index_f* nfr_build_f(const float* pts, size_t n, int dim, int metric,
                         size_t leaf_max_size, unsigned n_thread_build);   // pts borrowed, not copied
void   nfr_free_f(nfr_index_f*);
size_t nfr_knn_f (const nfr_index_f*, const float* q, size_t k, float eps, uint32_t* oi, float* od);
size_t nfr_rknn_f(const nfr_index_f*, const float* q, size_t k, float radius_sq, uint32_t* oi, float* od);
size_t nfr_radius_count_f(nfr_index_f*, const float* q, float radius_sq, int sorted, float eps); // caches in scratch
size_t nfr_radius_fetch_f(const nfr_index_f*, uint32_t* oi, float* od, size_t cap);
size_t nfr_box_count_f(nfr_index_f*, const float* lo, const float* hi);
size_t nfr_box_fetch_f(const nfr_index_f*, uint32_t* oi, size_t cap);
// + _d twins; + nfr_build_f_dim3_l2 / nfr_knn_f_dim3_l2 fast paths
```

`build.rs`: `cc::Build::new().cpp(true).std("c++17").file("cpp/wrapper.cpp").include("cpp").flag_if_supported("-O3").flag_if_supported("-march=native").compile("nanoflann_ref")`. `src/lib.rs`: safe `RefIndex` RAII wrappers (Drop → free, lifetime tied to point slice); tests and benches use only these.

`xval/src/lib.rs`: seeded generators — `uniform(n, dim)`, `clustered`, `duplicates(frac)`, `exponential_spacing(n)` (stack-killer), `on_circle_so2` — and comparators `assert_knn_equal(rust, cpp, ulps)` (exact indices; ULP distances; tie-groups as multisets) and `assert_radius_equal` (sorted: sequence; unsorted: set + expect traversal-order match first).

## 4. Tasks (dependency order; each: test first → fail → implement → pass → commit)

**T0 — Scaffolding.** `git init`; workspace + empty crates; `rust-toolchain.toml`; vendor `nanoflann.hpp` from the 1.12.1 checkout; copy this plan into `docs/superpowers/plans/`; write `docs/nanoflann-notes.md` (MT/incremental API findings + "stale spots not to port": `size_at_index_build_`, RKNN seeding, forest empty-index true, README `checks` param). Verify `cargo test --workspace` and `cargo build --no-default-features -p nanoflann-rs` pass. Commit.

**T1 — scalar.rs + dim.rs.** Tests: `ConstDim::<3>.filled(0.0f32)` is `[f32;3]`; `DynDim(5)` yields len-5 Vec; `dim()` values.

**T2 — data_source.rs.** Tests: `&[[f32;3]]` count/component; `FlatSlice` indexing; default `fill_bbox` false.

**T3 — metric.rs.** Hand-computed-value tests: L2/L1 dims {2,3,8,16,32} (exercises remainder 0–3), squared; L2Simple ULP-close to L2; **SO2 last-dim-only** ((10,0.1) vs (−10,−0.1) → 0.2) and **±π wrap both directions**, result in [0,π], unsquared; SO3 == L2Simple bit-for-bit; `accum_dist` per metric.

**T4 — result_set.rs.** Tests: sorted insert; capacity eviction; **duplicate-distance ties** under both `TieBreak`s; Knn `worst_dist` MAX→`dists[k-1]` transition; **Rknn `worst_dist` = radius until full**, no seed slot; Radius strict `<`, `full()` true, `sort()` by distance stable; `ResultItem` `#[repr(C)]` size/offset test.

**T5 — node.rs + bbox.rs.** Tests: `size_of::<Node<f32>>() == 20`, `<f64>() == 32`; leaf/interior round-trip; `Interval` inclusive-contains; `compute_bounding_box` incl. `fill_bbox == true` short-circuit; empty-dataset bbox error path handled by builder.

**T6 — build.rs (sequential, explicit work stack — no recursion).** Frames `{left, right, bbox, parent_slot}` + second-phase finalize entries. Tests: `plane_split` DNF permutation with exact `lim1/lim2` and the exact final `vind` on a hand-worked 8-element example (must match C++ swap sequence: `<` swap-advance-both, `>` swap-decrement-right, `==` advance mid); `middle_split` candidate threshold `(1−1e−5)·max_span`, cutval clamping, `lim1>n/2 / lim2<n/2 / n/2` rule; **all-identical 1000 points** → balanced, leaves ≤ leaf_max_size, `vind` a permutation; `leaf_max_size ∈ {1, n}`; structural-invariant checker helper (every leaf point satisfies all ancestor gap constraints; nodes ≤ 2n); empty → no nodes; n=1 → single leaf; **1M exponentially-spaced points build (release, `#[ignore]` heavy)** — no stack overflow.

**T7 — nanoflann-ref (FFI)** — parallel with T3–T6. Tests: tiny fixed dim-3 knn matches hand-computed; radius two-call round-trip; repeated build/free loop.

**T8 — search.rs: `search_level` + `compute_initial_distances` + `find_neighbors`.** Replicate: strict `dist < worst_dist()` leaf test, `(diff1+diff2) < 0` branch pick, `mindist + cut_dist − dists[idx]` save/restore, `mindist · (1+eps) <= worst_dist()` prune. Keep search natively recursive (depth = tree depth; build survives degenerate data, so search does too — escalate to explicit stack only if T11's exponential dataset shows otherwise). Tests: 200 seeded random sets — knn exact vs brute force with identical tie rule; query outside root bbox (exercises initial distances); **eps=0 equals brute force; eps=0.5 within (1+eps)·true on crafted set**; **radius boundary: `dist² == radius` excluded, `nextafter(radius)` includes**; **RKNN: k=5, 3-in-radius → 3; 10-in-radius → closest 5**; empty tree → false/0; `leaf_max_size ∈ {1, n}` agree with default.

**T9 — find_within_box (iterative).** Push child1 then child2 (pop = child2 first) to preserve C++ output order. Tests: **point exactly on each box face included**; prune boundaries `low == divlow` / `high == divhigh` traverse; output never sorted (raw traversal order); brute-force set-equality property test; empty tree → 0.

**T10 — tree.rs builder + public API.** Wire T6+T8+T9. Tests: builder defaults; `radius_search` clears `out` and honors `sorted=false`; `compile_fail` doc-test for unbuilt misuse where expressible; `used_memory_bytes`.

**T11 — Cross-validation (xval).** Matrix: {f32,f64} × dims {2,3,8,16,32} runtime + `ConstDim<3>` × metrics {L1,L2,L2Simple,SO3} (+SO2 dim-2) × datasets {uniform, clustered, 30% duplicates, all-identical, exponential} × leaf {1,10,64}: knn k ∈ {1,10,101} incl. k>n, 100 seeded queries incl. on-dataset and far-outside points — exact index equality, ULP-tolerant distances, tie-groups as multisets; rknn straddling fill/not-fill; radius sorted (sequence) and unsorted (set, expect traversal order too); box exact sequence; eps ∈ {0, 0.1, 1.0} both sides; empty-tree parity.

**T12 — build_parallel.rs (feature "parallel").** `plane_split` first, then `rayon::join` on halves when `count ≥ CUTOFF && spawn_depth < MAX_SPAWN_DEPTH (≈ 2·log2(n/cutoff)+8)`; disjoint `&mut vind` via `split_at_mut`; **task-local `Vec<Node>` arenas merged by parent with index offsetting** (no mutex, no contention); below cutoff/depth → T6 iterative builder (bounds native stack on degenerate trees). `Threads(n)` = scoped `rayon::ThreadPoolBuilder` pool; `Auto` = ambient pool (documented deviation: "≤ n workers", not C++ async-gating). Tests: parallel `vind` **bit-identical** to sequential + identical query results over 1k queries (determinism holds because partition precedes spawn); **1M exponential-spacing parallel build completes** (`#[ignore]` heavy); `--no-default-features` still compiles; xval parallel-Rust vs `n_thread_build=4` C++.

**T13 — Criterion benches.** `bench_build` (n {10k,100k,1M} × dim {3,8} × seq/par × Rust/C++), `bench_knn` (k {1,10,100} × dim {2,3,8,16,32} × f32/f64, incl. C++ fixed-DIM-3 fast path vs `ConstDim<3>`), `bench_radius` (selectivity ~10 / ~1000). Shared seeded data from xval; `target-cpu=native` on both sides; also emit a `leaf_max_size` sweep {1,4,10,16,32,50,128,1024} for both libraries (validates default 10 for our node layout).

**T14 — Perf pass (bench-gated changes only).** `#[repr(align(16))]` node A/B; eliminate bounds checks in leaf loop (restructure or audited `get_unchecked`); confirm `ConstDim<3>` L2 fully unrolls (inspect asm); caller-provided scratch only if benches demand. Success criterion: Rust ≥ C++ on dim-3 f32/f64 knn and build, or a documented analysis of any gap.

**T15 — Docs.** README: contract documentation (squared distances, strict radius, inclusive box, SO2 semantics, tie rules, `BuildThreads` deviation, feature matrix), benchmark results, M2 roadmap pointer to `docs/nanoflann-notes.md`.

Dependencies: T0 → T1 → T2 → {T3, T5} → T4 → T6 → T8 → T9 → T10 → T11; T7 ∥ after T0; T11 needs T7+T10; T12 after T6/T11; T13 after T11+T12; T14 after T13.

## 5. Key decisions (recommendation only)

- **DIM:** `Dim` trait with GAT array (`ConstDim<N>` → `[T;N]`, `DynDim` → `Vec<T>`) — one tree type, both monomorphized; exact analog of C++ `array_or_vector` + `if constexpr(DIM>0)`. Rejected: DIM=0 sentinel (needs unstable const-exprs), two tree types (API duplication for M2).
- **Nodes:** arena `Vec<Node>` + u32 children + `divfeat == u32::MAX` leaf sentinel; 20/32 B vs C++ ~48 B. Alignment-16 is a T14 A/B, not a default. Implicit-child pre-order layout deferred (conflicts with parallel arena merge).
- **Search core:** fully monomorphized over (metric, dim, result set, filter); `AcceptAll` ZST filter is the M2 `isActive()` seam at zero static-path cost.
- **Parallel build:** deterministic by construction (partition-before-spawn), contention-free via local arenas — both a correctness lever (exact cross-validation) and a perf lever vs C++'s mutexed pool + async-per-node.
- **Metrics take `&DS` per call** instead of holding a dataset reference (C++ style) — keeps metrics plain stateful values, no lifetime infection; same codegen after inlining.
- **Tie-break as type param** (`TieBreak`), mirroring the compile-time `NANOFLANN_FIRST_MATCH`.
- **Fidelity strategy:** replicate arithmetic order (unroll pairs, descending remainder, mindist update) so easy data matches bit-for-bit, but assert with ULP tolerance + tie-group multisets; no fast-math on either side.

## 5b. Success criteria (binding)

M1 is complete only when all four hold:
1. **Parity:** full xval suite green at bit-exact defaults (max_ulps=0): tree-permutation (vind) equality, knn/rknn/radius/box equality across the config matrix, boundary conventions locked.
2. **Speed:** automated perf gate (`xval/tests/perf_gate.rs`): median Rust <= 1.25x median C++ on fixed build/knn/radius workloads (hard gate); headline criterion numbers show Rust >= C++ or T14 documents the gap with analysis (goal).
3. **Robustness:** heavy degenerate builds (dim-1 depth ~2115, dim-8 depth ~16.8k) pass in release; empty/k>n/duplicate cases covered.
4. **Hygiene:** workspace tests green with zero warnings (forced rebuild), `--no-default-features` builds, doctests pass.

## 6. Verification (end-to-end)

1. `cargo test --workspace` — unit + behavior + xval suites green.
2. `cargo test --workspace --release -- --ignored` — heavy tests (1M exponential builds, seq + parallel).
3. `cargo build -p nanoflann-rs --no-default-features` — no-rayon build compiles.
4. `cargo bench -p xval` (with `RUSTFLAGS="-C target-cpu=native"`) — produces Rust-vs-C++ build/knn/radius numbers + leaf sweep; this answers the project's core question.
5. Spot-check: run the xval knn test with a deliberately broken tie rule and confirm it fails (validates the oracle actually constrains us).
