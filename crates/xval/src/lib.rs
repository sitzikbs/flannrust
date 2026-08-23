//! Cross-validation suite: same data, same queries, Rust `nanoflann-rs`
//! `KdTree` vs the in-process C++ nanoflann 1.12.1 oracle (`nanoflann-ref`),
//! compared BIT-EXACTLY (`ties = false`, positional, is the default
//! everywhere; `ties = true` relaxes ONLY index order within EXACT
//! (bit-equal) tie groups -- see `assert_knn_equal_f64`'s doc comment). This
//! crate is the judge, not a library: deterministic data generators, ULP-tier
//! comparators, and thin build helpers so `tests/*.rs` read cleanly. See
//! `tests/xval_knn.rs`, `tests/xval_radius_box.rs`, `tests/xval_build.rs`.
//! M2's dynamic (add/remove) op-sequence cross-validation lives in
//! `tests/xval_dynamic.rs`, backed by this file's `GrowableFlat`/`DynOp`/
//! `dyn_ops`/`apply_dyn_op_*`/`assert_dyn_structure_equal_*` section below.
//!
//! This crate also hosts the project's speed story: `benches/bench_build.rs`,
//! `benches/bench_knn.rs`, `benches/bench_radius.rs` (criterion, `harness =
//! false`), the automated pass/fail perf gate (`tests/perf_gate.rs`), the
//! native-codegen bit-parity canary (`tests/native_parity.rs`), and the
//! report-data collector (`examples/report_data.rs`). Run the benches with
//! `RUSTFLAGS="-C target-cpu=native" cargo bench -p xval` for a fair fight --
//! the C++ oracle is always built `-O3 -march=native -ffp-contract=off`
//! (see `crates/nanoflann-ref/build.rs`), so without `target-cpu=native` the
//! Rust side would be handicapped to a generic-x86-64 baseline instruction
//! set while the C++ side already gets the host's full ISA.

use nanoflann_rs::{
    BuildThreads as RustBuildThreads, Distance, DynDim, FlatSlice, Interval, KdTree, KdTreeBuilder,
    ResultItem, SearchParams, L1, L2, L2Simple, SO2, SO3,
};
use rand::Rng;
use rand::SeedableRng;

/// HTML scorecard renderer for the `report_data` JSON -- see
/// `examples/render_report.rs` for the CLI wrapper. `render` (this
/// module's entry point) is re-exported at the crate root for convenience.
pub mod report;
pub use report::{render, RenderError};
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Data generators — all produce ROW-MAJOR flat `Vec<f64>` (n*dim) from a
// `u64` seed via `ChaCha8Rng::seed_from_u64`. Deterministic forever: the same
// seed always reproduces the same bits, on any platform, for as long as
// ChaCha8Rng's algorithm doesn't change (it won't -- that's the whole point
// of picking a named, versioned PRNG instead of the platform default).
// ============================================================================

const RANGE: f64 = 10.0;

/// `n*dim` coordinates, each uniform in `[-10, 10)`.
pub fn uniform(seed: u64, n: usize, dim: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen_range(-RANGE..RANGE)).collect()
}

/// `clusters` cluster centers, each uniform in `[-10, 10)`; points are
/// `center + noise`, `noise` uniform in `[-0.1, 0.1)` per axis ("N(0, 0.1)-ish
/// uniform noise is fine" per the brief). Points are assigned to clusters
/// round-robin (`i % clusters`), so cluster membership is balanced and
/// entirely determined by the seed.
pub fn clustered(seed: u64, n: usize, dim: usize, clusters: usize) -> Vec<f64> {
    assert!(clusters > 0, "clustered: clusters must be > 0");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centers: Vec<Vec<f64>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-RANGE..RANGE)).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % clusters];
        for &cv in c {
            let noise = rng.gen_range(-0.1f64..0.1);
            out.push(cv + noise);
        }
    }
    out
}

/// `uniform(seed, n, dim)`, then the LAST `round(n * dup_frac)` points are
/// overwritten with exact copies of earlier (non-duplicate) points, chosen
/// deterministically from the same rng stream. `dup_frac` in `[0, 1]`.
pub fn with_duplicates(seed: u64, n: usize, dim: usize, dup_frac: f64) -> Vec<f64> {
    assert!((0.0..=1.0).contains(&dup_frac), "with_duplicates: dup_frac must be in [0,1]");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pts: Vec<f64> = (0..n * dim).map(|_| rng.gen_range(-RANGE..RANGE)).collect();
    let dup_count = ((n as f64) * dup_frac).round() as usize;
    let dup_count = dup_count.min(n);
    let original_count = n - dup_count;
    if original_count > 0 {
        for i in original_count..n {
            // Source is always in [0, original_count) -- disjoint from the
            // destination range [original_count, n) -- so `copy_within` never
            // aliases source and destination.
            let src = rng.gen_range(0..original_count);
            pts.copy_within(src * dim..(src + 1) * dim, i * dim);
        }
    }
    pts
}

/// Every point is exactly `[1.25, 1.25, ...]` (exactly representable in both
/// f32 and f64, so `to_f32` is lossless on this dataset). No seed needed --
/// fully deterministic already.
pub fn all_identical(n: usize, dim: usize) -> Vec<f64> {
    vec![1.25f64; n * dim]
}

/// Dim-1 dataset with exponentially shrinking gaps: `2^1023, 2^1022, ...`
/// down to (near) the smallest positive value f64 can hold, then padded with
/// `0.0` to reach `n` points. This is the EXACT spine construction
/// `nanoflann-rs`'s `heavy_exponential_build_1m` test (crates/nanoflann-rs/
/// src/build.rs) proved produces a maximally degenerate `middle_split` tree
/// (each split peels ~1-2 points, so depth grows ~linearly with `n` up to
/// f64's ~2098-halving dynamic-range ceiling) -- reused verbatim here so the
/// xval build-parity suite exercises the same worst case cross-language.
pub fn exponential_spacing(n: usize) -> Vec<f64> {
    let mut spine: Vec<f64> = Vec::new();
    let mut v = 2f64.powi(1023);
    while v > 0.0 && spine.len() < n.saturating_sub(1) {
        spine.push(v);
        v /= 2.0;
    }
    spine.resize(n, 0.0);
    spine
}

/// Dim-2 dataset for `SO2` cross-validation: coordinate 0 is uniform "junk"
/// (SO2's `eval` ignores everything but the last axis, so this coordinate
/// must never affect results); coordinate 1 is a uniform angle in
/// `[-pi, pi]` (already wrapped, as SO2's single-shot wrap assumes).
pub fn on_circle_so2(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let junk = rng.gen_range(-RANGE..RANGE);
        let angle = rng.gen_range(-core::f64::consts::PI..core::f64::consts::PI);
        out.push(junk);
        out.push(angle);
    }
    out
}

/// `n_queries` query points (row-major, `n_queries*dim`), a roughly-1/3/1/3/1/3
/// mix: uniform in `[-10, 10)`; EXACT copies of existing `data` points (tests
/// duplicate/tie handling against the real dataset, not synthetic dupes); and
/// "far outside" points (a uniform base +-1000 offset per axis) that exercise
/// bbox-miss / no-prune-possible traversal paths.
pub fn queries(seed: u64, data: &[f64], dim: usize, n_queries: usize) -> Vec<f64> {
    assert!(dim > 0, "queries: dim must be > 0");
    assert_eq!(data.len() % dim, 0, "queries: data.len() must be a multiple of dim");
    let n_data = data.len() / dim;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let third = n_queries / 3;
    let mut out = Vec::with_capacity(n_queries * dim);
    for i in 0..n_queries {
        if i < third {
            for _ in 0..dim {
                out.push(rng.gen_range(-RANGE..RANGE));
            }
        } else if i < 2 * third {
            if n_data > 0 {
                let src = rng.gen_range(0..n_data);
                out.extend_from_slice(&data[src * dim..(src + 1) * dim]);
            } else {
                for _ in 0..dim {
                    out.push(rng.gen_range(-RANGE..RANGE));
                }
            }
        } else {
            for _ in 0..dim {
                let base = rng.gen_range(-RANGE..RANGE);
                let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                out.push(base + sign * 1000.0);
            }
        }
    }
    out
}

/// Casts every element `as f32`. Both sides of an f32 cross-validation get
/// this SAME cast data, so the comparison stays fair (neither side is
/// disadvantaged by an independent f64->f32 rounding).
pub fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Reshapes a row-major flat buffer (`n*3` elements) into `n` fixed-size
/// `[T; 3]` points, for the `ConstDim::<3>` benchmark/report arms (the
/// `&[[T; 3]]` `DataSource` impl in `nanoflann_rs::data_source`). Panics if
/// `flat.len()` is not a multiple of 3.
pub fn to_array3<T: Copy>(flat: &[T]) -> Vec<[T; 3]> {
    assert_eq!(flat.len() % 3, 0, "to_array3: flat.len() ({}) must be a multiple of 3", flat.len());
    flat.as_chunks::<3>().0.to_vec()
}

/// Round-robin iterator over a pre-generated row-major query set (`n*dim`
/// elements), used by `benches/*.rs`, `tests/perf_gate.rs`, and
/// `examples/report_data.rs` so every timed/scored query loop walks the SAME
/// "index counter mod len" pattern instead of each call site reimplementing
/// it slightly differently. Panics (via the `dim == 0` / non-multiple check
/// in `next`'s slicing) only if constructed with `data.len()` not a multiple
/// of `dim`, or `dim == 0`.
pub struct RoundRobin<'a, T> {
    data: &'a [T],
    dim: usize,
    n: usize,
    i: usize,
}

impl<'a, T> RoundRobin<'a, T> {
    pub fn new(data: &'a [T], dim: usize) -> Self {
        assert!(dim > 0, "RoundRobin: dim must be > 0");
        assert_eq!(data.len() % dim, 0, "RoundRobin: data.len() ({}) must be a multiple of dim ({})", data.len(), dim);
        let n = data.len() / dim;
        assert!(n > 0, "RoundRobin: data must contain at least one point");
        Self { data, dim, n, i: 0 }
    }

    /// The next query slice (length `dim`), advancing the internal counter
    /// modulo `n` (wraps back to index 0 after the last query).
    // Deliberately not `Iterator::next` (returns `&'a [T]`, not
    // `Option<&'a [T]>`) -- an infinite wrapping cursor, not a terminating
    // iterator; renaming would ripple across every bench/perf_gate/
    // report_data call site for no behavioral benefit.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> &'a [T] {
        let q = &self.data[self.i * self.dim..(self.i + 1) * self.dim];
        self.i = (self.i + 1) % self.n;
        q
    }
}

// ============================================================================
// ULP-tier comparators
// ============================================================================

macro_rules! impl_ulp_diff {
    ($name:ident, $t:ty) => {
        /// Bit-distance between `a` and `b`. `u64::MAX` ("infinite") if
        /// either is NaN, or if they have different signs and are not BOTH
        /// zero (`0.0`/`-0.0` compare equal first and short-circuit to 0).
        /// Otherwise the plain unsigned difference of their bit patterns --
        /// correct because same-signed IEEE-754 floats' bit patterns
        /// (interpreted as same-width unsigned integers) are monotonic with
        /// magnitude.
        pub fn $name(a: $t, b: $t) -> u64 {
            if a.is_nan() || b.is_nan() {
                return u64::MAX;
            }
            if a == b {
                return 0;
            }
            if a.is_sign_negative() != b.is_sign_negative() {
                return u64::MAX;
            }
            let ab = a.to_bits() as u64;
            let bb = b.to_bits() as u64;
            ab.abs_diff(bb)
        }
    };
}

impl_ulp_diff!(ulp_diff_f64, f64);
impl_ulp_diff!(ulp_diff_f32, f32);

macro_rules! impl_knn_comparator {
    ($assert_fn:ident, $ulp_fn:ident, $t:ty) => {
        /// `ties == false` (the default everywhere): POSITIONAL equality --
        /// indices must match exactly at every rank AND distances must be
        /// bit-equal at every rank. This is what locks cross-language tie
        /// ORDER (`KeepInsertionOrder` vs the C++ default): an index-only
        /// multiset comparison would silently never test it on
        /// duplicate-heavy data.
        ///
        /// `ties == true` ("exact-tie-group" mode, for the one place C++'s
        /// OWN contract leaves order unspecified -- radius search's final
        /// `std::sort`, nanoflann.hpp:463, which is UNSTABLE): the ONLY
        /// thing relaxed is INDEX ORDER, and ONLY within a maximal run of
        /// entries whose distance is EXACTLY equal (`ulp_diff == 0`) to
        /// every other entry in that run -- computed independently on each
        /// side via each side's OWN self-comparison. Two things are still
        /// checked unconditionally, at every rank, exactly as strictly as
        /// the positional mode:
        /// - the exact-tie run boundaries themselves must match between
        ///   rust and cpp (a length mismatch there is a distinct "group
        ///   BOUNDARY mismatch", not silently absorbed into anything else);
        /// - EVERY rank's distance must still be bit-equal cross-side
        ///   (`ulp_diff(r_dist[k], c_dist[k]) == 0` for every `k` in the
        ///   matched run, not just checked once per group) -- this is what
        ///   makes a singleton run (no tie at all) degenerate EXACTLY to the
        ///   positional check, and what catches a same-index,
        ///   different-distance divergence that an index-only multiset
        ///   comparison would silently pass.
        ///
        /// Only the INDEX SEQUENCE within a matched run is compared as a
        /// multiset instead of positionally.
        ///
        /// Panics (`assert!`/`panic!`) with a full dump of both lists on any
        /// mismatch.
        pub fn $assert_fn(rust: (&[u32], &[$t]), cpp: (&[u32], &[$t]), ties: bool) {
            let (r_idx, r_dist) = rust;
            let (c_idx, c_dist) = cpp;
            assert_eq!(
                r_idx.len(),
                r_dist.len(),
                "internal: rust idx/dist length mismatch: {} vs {}",
                r_idx.len(),
                r_dist.len()
            );
            assert_eq!(
                c_idx.len(),
                c_dist.len(),
                "internal: cpp idx/dist length mismatch: {} vs {}",
                c_idx.len(),
                c_dist.len()
            );
            assert_eq!(
                r_idx.len(),
                c_idx.len(),
                "knn result COUNT mismatch: rust={} cpp={}\nrust idx={:?} dist={:?}\ncpp idx={:?} dist={:?}",
                r_idx.len(),
                c_idx.len(),
                r_idx,
                r_dist,
                c_idx,
                c_dist
            );
            let n = r_idx.len();
            if !ties {
                for i in 0..n {
                    let ud = $ulp_fn(r_dist[i], c_dist[i]);
                    if r_idx[i] != c_idx[i] || ud != 0 {
                        panic!(
                            "knn POSITIONAL mismatch at rank {i}: rust=({}, {:?}) cpp=({}, {:?}) ulp_diff={ud}\nfull rust idx={:?} dist={:?}\nfull cpp idx={:?} dist={:?}",
                            r_idx[i], r_dist[i], c_idx[i], c_dist[i], r_idx, r_dist, c_idx, c_dist
                        );
                    }
                }
            } else {
                let mut i = 0;
                while i < n {
                    // Exact-tie run end on EACH side, independently, via
                    // that side's OWN self-comparison (ulp_diff == 0, not
                    // "within some tolerance").
                    let mut end_r = i;
                    while end_r + 1 < n && $ulp_fn(r_dist[end_r + 1], r_dist[i]) == 0 {
                        end_r += 1;
                    }
                    let mut end_c = i;
                    while end_c + 1 < n && $ulp_fn(c_dist[end_c + 1], c_dist[i]) == 0 {
                        end_c += 1;
                    }
                    if end_r != end_c {
                        panic!(
                            "knn tie-GROUP BOUNDARY mismatch starting at rank {i}: rust group end={end_r} cpp group end={end_c}\nfull rust idx={:?} dist={:?}\nfull cpp idx={:?} dist={:?}",
                            r_idx, r_dist, c_idx, c_dist
                        );
                    }
                    let end = end_r;
                    // Distances must be BIT-EQUAL cross-side at EVERY rank
                    // in the matched run, not just once per group -- this is
                    // what a pure index-multiset check would miss entirely
                    // (see mutation_canary_distance_perturbation).
                    for k in i..=end {
                        let ud = $ulp_fn(r_dist[k], c_dist[k]);
                        if ud != 0 {
                            panic!(
                                "knn tie-group DISTANCE mismatch at rank {k} (within run [{i},{end}]): rust=({}, {:?}) cpp=({}, {:?}) ulp_diff={ud}\nfull rust idx={:?} dist={:?}\nfull cpp idx={:?} dist={:?}",
                                r_idx[k], r_dist[k], c_idx[k], c_dist[k], r_idx, r_dist, c_idx, c_dist
                            );
                        }
                    }
                    let mut r_ms: Vec<u32> = r_idx[i..=end].to_vec();
                    let mut c_ms: Vec<u32> = c_idx[i..=end].to_vec();
                    r_ms.sort_unstable();
                    c_ms.sort_unstable();
                    if r_ms != c_ms {
                        panic!(
                            "knn tie-group INDEX MULTISET mismatch in rank range [{i}, {end}]: rust={:?} cpp={:?}\nfull rust idx={:?} dist={:?}\nfull cpp idx={:?} dist={:?}",
                            r_ms, c_ms, r_idx, r_dist, c_idx, c_dist
                        );
                    }
                    i = end + 1;
                }
            }
        }
    };
}

impl_knn_comparator!(assert_knn_equal_f64, ulp_diff_f64, f64);
impl_knn_comparator!(assert_knn_equal_f32, ulp_diff_f32, f32);

macro_rules! impl_radius_comparator {
    ($assert_fn:ident, $knn_fn:ident, $ulp_fn:ident, $t:ty) => {
        /// `sorted == true`: same walk as `$knn_fn` -- `ties` is forwarded
        /// unchanged (ascending order is a contract on both sides, so
        /// positional/exact-tie-group comparison applies identically; see
        /// `$knn_fn`'s doc comment for exactly what `ties` does and does
        /// not relax).
        ///
        /// `sorted == false`: `ties` is UNUSED -- raw traversal order is
        /// compared as a multiset of `(index, distance)` pairs by sorting
        /// BOTH sides by index (indices are unique per point) and then
        /// requiring each matched pair's distance to be BIT-EQUAL
        /// (`ulp_diff == 0`); index order is already irrelevant here by
        /// construction (both sides are re-sorted by index before
        /// comparing), so there is no separate "tie order" concept to
        /// relax in this branch.
        pub fn $assert_fn(rust: &[(u32, $t)], cpp: &[(u32, $t)], sorted: bool, ties: bool) {
            assert_eq!(
                rust.len(),
                cpp.len(),
                "radius result COUNT mismatch: rust={} cpp={}\nrust={:?}\ncpp={:?}",
                rust.len(),
                cpp.len(),
                rust,
                cpp
            );
            if sorted {
                let r_idx: Vec<u32> = rust.iter().map(|x| x.0).collect();
                let r_dist: Vec<$t> = rust.iter().map(|x| x.1).collect();
                let c_idx: Vec<u32> = cpp.iter().map(|x| x.0).collect();
                let c_dist: Vec<$t> = cpp.iter().map(|x| x.1).collect();
                $knn_fn((&r_idx, &r_dist), (&c_idx, &c_dist), ties);
            } else {
                let mut r_sorted = rust.to_vec();
                let mut c_sorted = cpp.to_vec();
                r_sorted.sort_unstable_by_key(|x| x.0);
                c_sorted.sort_unstable_by_key(|x| x.0);
                for i in 0..r_sorted.len() {
                    if r_sorted[i].0 != c_sorted[i].0 {
                        panic!(
                            "radius multiset INDEX SET mismatch (both sides sorted by index for comparison) at position {i}: rust idx={} cpp idx={}\nfull rust={:?}\nfull cpp={:?}",
                            r_sorted[i].0, c_sorted[i].0, rust, cpp
                        );
                    }
                    let ud = $ulp_fn(r_sorted[i].1, c_sorted[i].1);
                    if ud != 0 {
                        panic!(
                            "radius DISTANCE mismatch for index {}: rust={:?} cpp={:?} ulp_diff={ud} (bit-exact required)\nfull rust={:?}\nfull cpp={:?}",
                            r_sorted[i].0, r_sorted[i].1, c_sorted[i].1, rust, cpp
                        );
                    }
                }
            }
        }
    };
}

impl_radius_comparator!(assert_radius_equal_f64, assert_knn_equal_f64, ulp_diff_f64, f64);
impl_radius_comparator!(assert_radius_equal_f32, assert_knn_equal_f32, ulp_diff_f32, f32);

// ============================================================================
// Build helpers
// ============================================================================

/// Distance metric selector for the Rust-side build helpers, mirrored 1:1
/// with `nanoflann_ref::Metric` via `XMetric::to_ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XMetric {
    L1,
    L2,
    L2Simple,
    SO2,
    SO3,
}

impl XMetric {
    /// Maps to the corresponding `nanoflann_ref::Metric`, so tests can build
    /// BOTH sides' indices from one `XMetric` value over the same data/leaf.
    pub fn to_ref(self) -> nanoflann_ref::Metric {
        match self {
            XMetric::L1 => nanoflann_ref::Metric::L1,
            XMetric::L2 => nanoflann_ref::Metric::L2,
            XMetric::L2Simple => nanoflann_ref::Metric::L2Simple,
            XMetric::SO2 => nanoflann_ref::Metric::SO2,
            XMetric::SO3 => nanoflann_ref::Metric::SO3,
        }
    }
}

/// Build-thread policy for `build_rust_*`. Mirrors `nanoflann_rs::BuildThreads`
/// 1:1 (kept as xval's own type rather than a re-export so `tests/*.rs` don't
/// need to depend on `nanoflann-rs`'s feature flags to name a variant) --
/// `to_rust` maps it onto the real thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildThreads {
    #[default]
    Sequential,
    Auto,
    Threads(core::num::NonZeroU32),
}

impl BuildThreads {
    fn to_rust(self) -> RustBuildThreads {
        match self {
            BuildThreads::Sequential => RustBuildThreads::Sequential,
            BuildThreads::Auto => RustBuildThreads::Auto,
            BuildThreads::Threads(n) => RustBuildThreads::Threads(n),
        }
    }
}

/// Defines the `$Name<'a>` opaque Rust index wrapper (an enum over the 5
/// monomorphized `KdTree` metric instantiations, `DynDim` + `FlatSlice<$t>`,
/// `Idx = u32`, `TB = KeepInsertionOrder` -- nanoflann's own default tie
/// rule, matching the C++ oracle's `NANOFLANN_FIRST_MATCH`-undefined
/// default) and the `$build_fn` constructor.
macro_rules! define_rust_index {
    ($Name:ident, $build_fn:ident, $t:ty) => {
        pub enum $Name<'a> {
            L1(KdTree<$t, DynDim, FlatSlice<'a, $t>, L1>),
            L2(KdTree<$t, DynDim, FlatSlice<'a, $t>, L2>),
            L2Simple(KdTree<$t, DynDim, FlatSlice<'a, $t>, L2Simple>),
            SO2(KdTree<$t, DynDim, FlatSlice<'a, $t>, SO2>),
            SO3(KdTree<$t, DynDim, FlatSlice<'a, $t>, SO3>),
        }

        impl<'a> $Name<'a> {
            /// `eps = 0` for exact search, sorted ascending. Returns `(indices,
            /// distances)`, truncated to the number actually found.
            pub fn knn(&self, q: &[$t], k: usize, eps: f32) -> (Vec<u32>, Vec<$t>) {
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                let params = SearchParams { eps, sorted: true };
                let found = match self {
                    $Name::L1(t) => t.knn_search_with(q, &mut idx, &mut dist, &params),
                    $Name::L2(t) => t.knn_search_with(q, &mut idx, &mut dist, &params),
                    $Name::L2Simple(t) => t.knn_search_with(q, &mut idx, &mut dist, &params),
                    $Name::SO2(t) => t.knn_search_with(q, &mut idx, &mut dist, &params),
                    $Name::SO3(t) => t.knn_search_with(q, &mut idx, &mut dist, &params),
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// Zero-allocation `knn`: writes into caller-owned `out_idx`/
            /// `out_dist` (both MUST have length exactly `k` -- asserted)
            /// instead of allocating fresh `Vec`s, via the underlying
            /// `KdTree::knn_search_with` directly. Returns the found count
            /// (`<= k`; valid entries are `out_idx[..found]`/
            /// `out_dist[..found]`). Mirrors `nanoflann_ref::RefIndexF32::
            /// knn_into`'s contract exactly, for allocation-symmetric
            /// benchmark/perf-gate query loops.
            pub fn knn_into(&self, q: &[$t], k: usize, eps: f32, out_idx: &mut [u32], out_dist: &mut [$t]) -> usize {
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                let params = SearchParams { eps, sorted: true };
                match self {
                    $Name::L1(t) => t.knn_search_with(q, out_idx, out_dist, &params),
                    $Name::L2(t) => t.knn_search_with(q, out_idx, out_dist, &params),
                    $Name::L2Simple(t) => t.knn_search_with(q, out_idx, out_dist, &params),
                    $Name::SO2(t) => t.knn_search_with(q, out_idx, out_dist, &params),
                    $Name::SO3(t) => t.knn_search_with(q, out_idx, out_dist, &params),
                }
            }

            /// `radius` is in the metric's native scale (SQUARED for
            /// L2-family, matching the C++ oracle's convention).
            pub fn rknn(&self, q: &[$t], k: usize, radius: $t, eps: f32) -> (Vec<u32>, Vec<$t>) {
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                let params = SearchParams { eps, sorted: true };
                let found = match self {
                    $Name::L1(t) => t.rknn_search_with(q, radius, &mut idx, &mut dist, &params),
                    $Name::L2(t) => t.rknn_search_with(q, radius, &mut idx, &mut dist, &params),
                    $Name::L2Simple(t) => {
                        t.rknn_search_with(q, radius, &mut idx, &mut dist, &params)
                    }
                    $Name::SO2(t) => t.rknn_search_with(q, radius, &mut idx, &mut dist, &params),
                    $Name::SO3(t) => t.rknn_search_with(q, radius, &mut idx, &mut dist, &params),
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// STRICTLY `dist < radius`. Ascending order iff `sorted`.
            pub fn radius(&self, q: &[$t], radius: $t, sorted: bool, eps: f32) -> Vec<(u32, $t)> {
                let mut out: Vec<ResultItem<u32, $t>> = Vec::new();
                let params = SearchParams { eps, sorted };
                match self {
                    $Name::L1(t) => {
                        t.radius_search_with(q, radius, &mut out, &params);
                    }
                    $Name::L2(t) => {
                        t.radius_search_with(q, radius, &mut out, &params);
                    }
                    $Name::L2Simple(t) => {
                        t.radius_search_with(q, radius, &mut out, &params);
                    }
                    $Name::SO2(t) => {
                        t.radius_search_with(q, radius, &mut out, &params);
                    }
                    $Name::SO3(t) => {
                        t.radius_search_with(q, radius, &mut out, &params);
                    }
                }
                out.into_iter().map(|ri| (ri.index, ri.distance)).collect()
            }

            /// Zero-(re)allocation `radius`: writes into a caller-owned
            /// `out: &mut Vec<ResultItem<u32, T>>` instead of allocating a
            /// fresh `Vec` (and a second `Vec` for the tuple conversion)
            /// per call. `KdTree::radius_search_with` itself `clear()`s `out`
            /// and refills it, so reusing the SAME `Vec` across many calls
            /// with similar result sizes reallocates only while growing to
            /// the largest size seen (mirrors
            /// `nanoflann_ref::RefIndexF32::radius_into`'s resize-based
            /// contract). Returns the found count (== `out.len()` after the
            /// call).
            pub fn radius_into(
                &self,
                q: &[$t],
                radius: $t,
                sorted: bool,
                eps: f32,
                out: &mut Vec<ResultItem<u32, $t>>,
            ) -> usize {
                let params = SearchParams { eps, sorted };
                match self {
                    $Name::L1(t) => t.radius_search_with(q, radius, out, &params),
                    $Name::L2(t) => t.radius_search_with(q, radius, out, &params),
                    $Name::L2Simple(t) => t.radius_search_with(q, radius, out, &params),
                    $Name::SO2(t) => t.radius_search_with(q, radius, out, &params),
                    $Name::SO3(t) => t.radius_search_with(q, radius, out, &params),
                }
            }

            /// Inclusive faces on every axis, indices only, never sorted
            /// (raw traversal order).
            pub fn find_within_box(&self, lo: &[$t], hi: &[$t]) -> Vec<u32> {
                assert_eq!(lo.len(), hi.len(), "find_within_box: lo/hi length mismatch");
                let bounds: Vec<Interval<$t>> =
                    lo.iter().zip(hi.iter()).map(|(&low, &high)| Interval { low, high }).collect();
                let mut out = Vec::new();
                match self {
                    $Name::L1(t) => {
                        t.find_within_box(&bounds, &mut out);
                    }
                    $Name::L2(t) => {
                        t.find_within_box(&bounds, &mut out);
                    }
                    $Name::L2Simple(t) => {
                        t.find_within_box(&bounds, &mut out);
                    }
                    $Name::SO2(t) => {
                        t.find_within_box(&bounds, &mut out);
                    }
                    $Name::SO3(t) => {
                        t.find_within_box(&bounds, &mut out);
                    }
                }
                out
            }

            /// The build-time permutation of point indices (= nanoflann's
            /// `vAcc_`), for build-parity cross-validation.
            pub fn vind(&self) -> Vec<u32> {
                match self {
                    $Name::L1(t) => t.point_indices().to_vec(),
                    $Name::L2(t) => t.point_indices().to_vec(),
                    $Name::L2Simple(t) => t.point_indices().to_vec(),
                    $Name::SO2(t) => t.point_indices().to_vec(),
                    $Name::SO3(t) => t.point_indices().to_vec(),
                }
            }

            pub fn used_memory_bytes(&self) -> usize {
                match self {
                    $Name::L1(t) => t.used_memory_bytes(),
                    $Name::L2(t) => t.used_memory_bytes(),
                    $Name::L2Simple(t) => t.used_memory_bytes(),
                    $Name::SO2(t) => t.used_memory_bytes(),
                    $Name::SO3(t) => t.used_memory_bytes(),
                }
            }

            pub fn size(&self) -> usize {
                match self {
                    $Name::L1(t) => t.size(),
                    $Name::L2(t) => t.size(),
                    $Name::L2Simple(t) => t.size(),
                    $Name::SO2(t) => t.size(),
                    $Name::SO3(t) => t.size(),
                }
            }
        }

        /// Builds a `$Name` over flat row-major `data` (`n*dim` elements)
        /// via `DynDim` + `FlatSlice`, the given `metric`/`leaf`/`threads`
        /// (mapped onto `nanoflann_rs::BuildThreads` via `to_rust`).
        pub fn $build_fn<'a>(
            data: &'a [$t],
            dim: usize,
            metric: XMetric,
            leaf: usize,
            threads: BuildThreads,
        ) -> $Name<'a> {
            let ds = FlatSlice::new(data, dim);
            let threads = threads.to_rust();
            match metric {
                XMetric::L1 => $Name::L1(
                    KdTreeBuilder::new(DynDim(dim), ds).with_metric(L1).leaf_max_size(leaf).threads(threads).build(),
                ),
                XMetric::L2 => $Name::L2(
                    KdTreeBuilder::new(DynDim(dim), ds).with_metric(L2).leaf_max_size(leaf).threads(threads).build(),
                ),
                XMetric::L2Simple => $Name::L2Simple(
                    KdTreeBuilder::new(DynDim(dim), ds)
                        .with_metric(L2Simple)
                        .leaf_max_size(leaf)
                        .threads(threads)
                        .build(),
                ),
                XMetric::SO2 => $Name::SO2(
                    KdTreeBuilder::new(DynDim(dim), ds).with_metric(SO2).leaf_max_size(leaf).threads(threads).build(),
                ),
                XMetric::SO3 => $Name::SO3(
                    KdTreeBuilder::new(DynDim(dim), ds).with_metric(SO3).leaf_max_size(leaf).threads(threads).build(),
                ),
            }
        }
    };
}

define_rust_index!(RustIndexF64, build_rust_f64, f64);
define_rust_index!(RustIndexF32, build_rust_f32, f32);

// ============================================================================
// Brute-force ground truth + accuracy scoring -- used by
// `examples/report_data.rs` to score both implementations' *approximate*
// (eps > 0) and exact (eps == 0) k-NN results against a tree-independent
// reference. O(n_queries * n); intentionally the slow, dumb, trivially-
// correct baseline -- see each function's doc comment for the exact tie
// rule and error definition (both hand-checked by the unit tests below).
//
// DISTANCE ARITHMETIC: distances here are computed
// via `L2.eval(query, &ds, idx, dim)` -- the LIBRARY'S OWN metric kernel,
// the exact same trait method `KdTree::knn_search`'s leaf-scan calls
// internally (`search.rs`'s `ctx.metric.eval(...)`) -- NOT a hand-rolled
// `iter().zip().map().sum()`. Selection (which points are scanned, and in
// what order) stays a dumb, tree-independent linear scan over every index;
// only the ARITHMETIC is shared with the trees. This is load-bearing: an
// independently-written summation (even mathematically identical) is not
// guaranteed bit-identical to either tree's internal computation, because
// IEEE-754 addition is not associative and a different code path can round
// differently in the last ULP -- confirmed by direct measurement (a first
// attempt using a hand-rolled sum here made a bit-exact accuracy scorer
// FAIL on ~90% of totally-correct, non-tied uniform-data queries, purely
// from 1-ULP drift against the ad-hoc scanner, despite Rust and C++ trees
// agreeing with EACH OTHER bit-exactly on every one of those same queries).
// Routing GT through the same kernel the trees use closes that gap, because
// the two real implementations were specifically engineered (identical
// unroll/summation order + `-ffp-contract=off` + no rustc contraction --
// see `tests/native_parity.rs`) to bit-match THAT kernel already.
// ============================================================================

macro_rules! impl_brute_force_knn {
    ($name:ident, $t:ty) => {
        /// Linear-scan k-NN ground truth under squared L2, computed via the
        /// LIBRARY's own `L2` metric kernel (`nanoflann_rs::L2::eval`, via a
        /// `FlatSlice` over `data`) -- NOT an independently-written
        /// summation; see this module's doc comment above for why that
        /// distinction is load-bearing for bit-exact scoring. Tie-broken by
        /// ASCENDING INDEX -- a total, deterministic order that does not
        /// depend on any tree's build/traversal order, chosen so the ground
        /// truth's SELECTION is reproducible independent of which
        /// implementation (or which insertion-order tie rule) produced a
        /// candidate result being scored against it (only the arithmetic is
        /// shared with the trees; selection is a plain index-order scan).
        /// Returns up to `k` `(index, squared_distance)` pairs (fewer iff
        /// `data` holds fewer than `k` points), sorted ascending primarily
        /// by distance, secondarily by index.
        ///
        /// Panics if `dim == 0`, `query.len() != dim`, or `data.len()` is
        /// not a multiple of `dim`.
        pub fn $name(data: &[$t], dim: usize, query: &[$t], k: usize) -> Vec<(u32, $t)> {
            assert!(dim > 0, "{}: dim must be > 0", stringify!($name));
            assert_eq!(query.len(), dim, "{}: query.len() must equal dim", stringify!($name));
            assert_eq!(data.len() % dim, 0, "{}: data.len() must be a multiple of dim", stringify!($name));
            let n = data.len() / dim;
            let ds = FlatSlice::new(data, dim);
            let mut all: Vec<(u32, $t)> =
                (0..n).map(|i| (i as u32, L2.eval(query, &ds, i, DynDim(dim)))).collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("brute_force_knn: NaN distance").then(a.0.cmp(&b.0)));
            all.truncate(k.min(n));
            all
        }
    };
}

impl_brute_force_knn!(brute_force_knn_l2_f32, f32);
impl_brute_force_knn!(brute_force_knn_l2_f64, f64);

/// One query's accuracy score against a ground-truth result list.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryScore {
    /// `true` iff the candidate's INDEX SET equals the ground truth's index
    /// SET exactly (order-independent -- deliberately not positional, since
    /// a candidate may legitimately return the same k points in a different
    /// order under a different tie rule and still be "exact").
    ///
    /// NOTE (see `score_exact_tie_aware_f32`/`_f64`'s doc comment below):
    /// this definition is STILL too strict on
    /// duplicate-heavy datasets -- when multiple points are exactly tied at
    /// the k-th distance boundary, "the" index SET is not uniquely defined,
    /// so a candidate that legitimately picks a DIFFERENT (but equally
    /// valid) winner of that tie scores `exact = false` here even though it
    /// made no error. `examples/report_data.rs`'s accuracy report uses
    /// `score_exact_tie_aware_f32`/`_f64` for its `exact_tie_aware_vs_bruteforce`
    /// field instead of this one for exactly that reason. This field (and
    /// `rel_dist_errors` below) are kept as-is for their existing callers/
    /// tests and because `rel_dist_errors` is still the right per-rank error
    /// metric regardless of tie handling.
    pub exact: bool,
    /// Per-rank relative distance error, ONLY populated when
    /// `gt.len() == got.len()` (a length mismatch is itself evidence of an
    /// incomplete/approximate result, not something to paper over with a
    /// partial rank-wise comparison -- left empty in that case). Rank `i`'s
    /// entry is `|got[i].1 - gt[i].1| / gt[i].1`, or `0.0` if
    /// `gt[i].1 == 0.0` and `got[i].1 == 0.0` too (both exactly at the query
    /// point), or `|got[i].1|` (an absolute, not relative, fallback) if
    /// `gt[i].1 == 0.0` but `got[i].1 != 0.0` (a relative error against a
    /// zero denominator is undefined; falling back to the raw magnitude
    /// keeps the number finite and still informative instead of `inf`/`NaN`
    /// poisoning the aggregate mean/max downstream).
    pub rel_dist_errors: Vec<f64>,
}

macro_rules! impl_score_query {
    ($name:ident, $t:ty) => {
        /// Scores one candidate k-NN result (`got`) against ground truth
        /// (`gt`) -- see `QueryScore`'s doc comment for exactly what `exact`
        /// and `rel_dist_errors` mean.
        pub fn $name(gt: &[(u32, $t)], got: &[(u32, $t)]) -> QueryScore {
            let gt_set: std::collections::BTreeSet<u32> = gt.iter().map(|x| x.0).collect();
            let got_set: std::collections::BTreeSet<u32> = got.iter().map(|x| x.0).collect();
            let exact = gt_set == got_set;
            let mut rel_dist_errors = Vec::new();
            if gt.len() == got.len() {
                for i in 0..gt.len() {
                    let g = gt[i].1 as f64;
                    let c = got[i].1 as f64;
                    let err = if g == 0.0 {
                        if c == 0.0 {
                            0.0
                        } else {
                            c.abs()
                        }
                    } else {
                        (c - g).abs() / g
                    };
                    rel_dist_errors.push(err);
                }
            }
            QueryScore { exact, rel_dist_errors }
        }
    };
}

impl_score_query!(score_query_f32, f32);
impl_score_query!(score_query_f64, f64);

macro_rules! impl_score_exact_tie_aware {
    ($name:ident, $t:ty) => {
        /// Tie-aware ground-truth exactness check, used by
        /// `examples/report_data.rs`'s accuracy report -- see
        /// `QueryScore::exact`'s doc comment above for why the plain
        /// order-independent set match is too strict on duplicate-heavy
        /// datasets. A query is EXACT iff BOTH hold:
        ///
        /// 1. `got`'s distances, taken IN ORDER, are bit-equal to `gt`'s
        ///    distances in order (both are assumed already sorted ascending
        ///    by distance, which is how every `gt`/`got` pair in this crate
        ///    is actually produced -- `brute_force_knn_l2_*` and every
        ///    `.knn(...)` wrapper here always return ascending-sorted
        ///    results). This pins the DISTANCE MULTISET (including
        ///    duplicates) without requiring any particular index to win a
        ///    boundary tie, and a length mismatch fails this outright.
        /// 2. Every returned index is a LEGITIMATE neighbor at its rank:
        ///    its OWN true squared-L2 distance to `query` -- recomputed
        ///    directly from `data` via the SAME library `L2` metric kernel
        ///    used for `gt` (see the module doc comment above for why this
        ///    is load-bearing: an independently-written summation is not
        ///    reliably bit-identical to the trees' internal arithmetic,
        ///    even for a mathematically-correct result) -- is bit-equal to
        ///    the distance `got` reported for it. This is what actually
        ///    catches errors condition 1 alone would miss -- e.g. a wrong
        ///    index whose reported distance happens to match the GT
        ///    distance at that rank.
        ///
        /// Together these accept ANY valid resolution of a k-th-boundary
        /// tie (any index whose true distance equals the tied value) while
        /// still rejecting a genuinely wrong neighbor, a wrong distance, or
        /// a missed closer point. Panics if `dim == 0` or `query.len() !=
        /// dim` (same preconditions as `brute_force_knn_l2_*`).
        pub fn $name(data: &[$t], dim: usize, query: &[$t], gt: &[(u32, $t)], got: &[(u32, $t)]) -> bool {
            assert!(dim > 0, "{}: dim must be > 0", stringify!($name));
            assert_eq!(query.len(), dim, "{}: query.len() must equal dim", stringify!($name));
            if gt.len() != got.len() {
                return false;
            }
            for i in 0..gt.len() {
                if got[i].1.to_bits() != gt[i].1.to_bits() {
                    return false;
                }
            }
            let ds = FlatSlice::new(data, dim);
            for &(idx, reported_dist) in got {
                let true_dist: $t = L2.eval(query, &ds, idx as usize, DynDim(dim));
                if true_dist.to_bits() != reported_dist.to_bits() {
                    return false;
                }
            }
            true
        }
    };
}

impl_score_exact_tie_aware!(score_exact_tie_aware_f32, f32);
impl_score_exact_tie_aware!(score_exact_tie_aware_f64, f64);

// ============================================================================
// M2 Task 5 -- live-set-filter extension. `brute_force_knn_l2_*` and
// `score_exact_tie_aware_*` above are the RIGHT ground truth / exactness
// definitions when every dataset index is queryable, but M2's dynamic
// accuracy report scores knn results after a churn (add/remove/re-add)
// sequence: a removed (tombstoned) index's coordinates are still physically
// present in the backing buffer (lazy deletion never touches storage -- see
// `dynamic.rs`'s module doc), so an UNFILTERED brute-force scan would treat
// dead points as valid ground-truth neighbors, and an UNFILTERED tie-aware
// scorer would happily accept a candidate that (erroneously) returns one, as
// long as its recomputed coordinate-distance happens to be numerically
// correct (`tie_aware_live_rejects_a_removed_index_even_with_correct_distance`,
// this module's `#[cfg(test)]`, pins exactly this gap). `live` (length `n =
// data.len()/dim`, `live[i]` true iff dataset index `i` is currently live)
// closes it on both ends: ground truth never considers a dead index a
// candidate neighbor, and the scorer independently rejects any `got` entry
// whose index is dead, regardless of how "correct" its reported distance
// looks.
// ============================================================================

macro_rules! impl_brute_force_knn_live {
    ($name:ident, $t:ty) => {
        /// Live-set-restricted variant of `brute_force_knn_l2_*` (see this
        /// section's module doc) -- identical selection/arithmetic contract
        /// (same library `L2::eval` kernel, same ascending-distance-then-
        /// index tie break) except any index `i` with `live[i] == false` is
        /// excluded from consideration entirely, as if it were not part of
        /// the dataset. Panics if `dim == 0`, `query.len() != dim`,
        /// `data.len()` is not a multiple of `dim`, or `live.len() != n`.
        /// Returns up to `k` pairs (fewer iff fewer than `k` indices are
        /// live).
        pub fn $name(data: &[$t], dim: usize, query: &[$t], k: usize, live: &[bool]) -> Vec<(u32, $t)> {
            assert!(dim > 0, "{}: dim must be > 0", stringify!($name));
            assert_eq!(query.len(), dim, "{}: query.len() must equal dim", stringify!($name));
            assert_eq!(data.len() % dim, 0, "{}: data.len() must be a multiple of dim", stringify!($name));
            let n = data.len() / dim;
            assert_eq!(live.len(), n, "{}: live.len() ({}) must equal n ({n})", stringify!($name), live.len());
            let ds = FlatSlice::new(data, dim);
            let mut all: Vec<(u32, $t)> = (0..n)
                .filter(|&i| live[i])
                .map(|i| (i as u32, L2.eval(query, &ds, i, DynDim(dim))))
                .collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("brute_force_knn_live: NaN distance").then(a.0.cmp(&b.0)));
            all.truncate(k.min(all.len()));
            all
        }
    };
}

impl_brute_force_knn_live!(brute_force_knn_l2_live_f32, f32);
impl_brute_force_knn_live!(brute_force_knn_l2_live_f64, f64);

macro_rules! impl_score_exact_tie_aware_live {
    ($name:ident, $t:ty) => {
        /// Live-set-aware variant of `score_exact_tie_aware_*` (see this
        /// section's module doc) -- the SAME two conditions (positional
        /// bit-equal distance list vs `gt`; every returned index's
        /// recomputed true distance bit-matches its reported distance) PLUS
        /// a third, independent check: every index in `got` must satisfy
        /// `live[idx] == true`. This is load-bearing on its own -- lazy
        /// deletion never touches a tombstoned point's physical storage, so
        /// a dynamic-forest bug that returns a removed index can still
        /// report a numerically "correct" distance for it (see
        /// `tie_aware_live_rejects_a_removed_index_even_with_correct_distance`);
        /// only the liveness check catches that class of bug. `gt` is
        /// expected to already be restricted to the live set (e.g. via
        /// `brute_force_knn_l2_live_*`) -- this function does not itself
        /// filter `gt`, only `got`. Panics if `dim == 0` or `query.len() !=
        /// dim` (same preconditions as `score_exact_tie_aware_*`); an
        /// out-of-range `got` index is treated as dead (`live.get(idx)`
        /// defaults to `false` via `unwrap_or(false)`) rather than panicking.
        pub fn $name(
            data: &[$t],
            dim: usize,
            query: &[$t],
            gt: &[(u32, $t)],
            got: &[(u32, $t)],
            live: &[bool],
        ) -> bool {
            assert!(dim > 0, "{}: dim must be > 0", stringify!($name));
            assert_eq!(query.len(), dim, "{}: query.len() must equal dim", stringify!($name));
            if gt.len() != got.len() {
                return false;
            }
            for i in 0..gt.len() {
                if got[i].1.to_bits() != gt[i].1.to_bits() {
                    return false;
                }
            }
            let ds = FlatSlice::new(data, dim);
            for &(idx, reported_dist) in got {
                if !live.get(idx as usize).copied().unwrap_or(false) {
                    return false;
                }
                let true_dist: $t = L2.eval(query, &ds, idx as usize, DynDim(dim));
                if true_dist.to_bits() != reported_dist.to_bits() {
                    return false;
                }
            }
            true
        }
    };
}

impl_score_exact_tie_aware_live!(score_exact_tie_aware_live_f32, f32);
impl_score_exact_tie_aware_live!(score_exact_tie_aware_live_f64, f64);

// ============================================================================
// Median-of-N timing -- shared by `tests/perf_gate.rs` and
// `examples/report_data.rs` (both use the SAME "median of 7 timed runs,
// after one untimed warmup" methodology, per the brief).
// ============================================================================

/// Median of a non-empty sample. Odd length -> middle element after
/// sorting; even length -> average of the two middle elements (this crate
/// always calls it with an odd `runs` count, but the even-length branch
/// keeps the function total rather than partial).
pub fn median_of(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty(), "median_of: empty sample");
    v.sort_by(|a, b| a.partial_cmp(b).expect("median_of: NaN sample"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Runs `f` once untimed (warmup), then `runs` times timed via
/// `std::time::Instant`; returns the median wall time in milliseconds.
pub fn timed_median_ms<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    assert!(runs > 0, "timed_median_ms: runs must be > 0");
    f();
    let mut samples_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = std::time::Instant::now();
        f();
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    median_of(samples_ms)
}

/// Deterministic, order-sensitive mixing hash over `tag` and `parts` (an
/// FNV-1a variant) -- used by `tests/xval_*.rs` to derive a `u64` seed from
/// loop indices for each config, so every config gets its own reproducible
/// data/query stream instead of every config sharing one seed.
pub fn cfg_seed(tag: &str, parts: &[usize]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tag.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &p in parts {
        h ^= p as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic (seeded) sample of `count` DISTINCT indices from `0..n`, in
/// shuffled (not sorted) order -- a full Fisher-Yates shuffle of `0..n`
/// truncated to `count` (`O(n)`, fine for the few-times-per-run call sites
/// this exists for: `benches/bench_dynamic.rs`'s churn benchmarks and M2
/// Task 5's `perf_gate.rs`/`report_data.rs` churn workloads, none of which
/// call this from inside a timed loop). Used to pick a fixed, reproducible
/// set of dataset indices to remove-then-re-add in a dynamic-forest churn
/// workload. Panics if `count > n`.
pub fn sample_distinct_indices(seed: u64, n: usize, count: usize) -> Vec<usize> {
    assert!(count <= n, "sample_distinct_indices: count ({count}) must be <= n ({n})");
    use rand::seq::SliceRandom;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut all: Vec<usize> = (0..n).collect();
    all.shuffle(&mut rng);
    all.truncate(count);
    all
}

/// Runs `f`; if it panics, re-panics with `ctx` prepended to the original
/// panic message. The comparators (`assert_knn_equal_*`/`assert_radius_equal_*`)
/// have no idea what config/seed produced their inputs -- this is how the
/// xval test suites get seeds and loop-index context INTO the failing
/// assertion's message without threading a context string through every
/// comparator call site. Does not touch the global panic hook (safe under
/// cargo test's default parallel test execution) -- the tradeoff is that the
/// ORIGINAL panic still prints once via the default hook before this
/// re-panics with the contextualized message the test harness ultimately
/// reports.
///
/// Takes plain `FnOnce()` -- no `UnwindSafe` bound -- and wraps `f` in
/// `std::panic::AssertUnwindSafe` internally. This is a deliberate, sound
/// relaxation: `with_ctx` NEVER resumes using `f`'s captured state after a
/// caught panic, it only extracts the panic payload's text and immediately
/// re-panics (aborting the caller's control flow either way), so the usual
/// hazard `UnwindSafe` guards against -- observing a torn/inconsistent
/// value through a captured `&`/`&mut` reference after unwinding past it --
/// can't happen here. Needed for M2's dynamic cross-validation callers
/// (`tests/xval_dynamic.rs`), which capture `&DynamicKdTree<..>` values
/// backed by `GrowableFlat`'s interior-mutable `Cell<usize>` -- `Cell` is
/// never `RefUnwindSafe` by design, so those captures could never satisfy
/// the old bound even though nothing here actually re-reads them
/// post-unwind.
pub fn with_ctx<F: FnOnce()>(ctx: impl std::fmt::Display, f: F) {
    if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        let msg = if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else {
            "<non-string panic payload>".to_string()
        };
        panic!("{ctx}\n{msg}");
    }
}

// ============================================================================
// M2 Task 4 -- dynamic (add/remove) cross-validation helpers. Everything
// below drives IDENTICAL op sequences through `nanoflann_rs::dynamic::
// DynamicKdTree` and `nanoflann_ref::RefDynIndexF32/F64` so `tests/
// xval_dynamic.rs` can compare structure + query results bit-exact after
// EVERY step. See `nanoflann_ref::RefDynIndexF32`'s doc comment for the
// contiguous-append contract this module's op generator exists to never
// violate (the Rust side PANICS on a violation; the C++ side SILENTLY
// CORRUPTS -- see `nanoflann-ref/tests/oracle_dynamic.rs`'s
// `add_points_misaligned_start_documents_silent_corruption_f32`).
// ============================================================================

use nanoflann_ref::{RefDynIndexF32, RefDynIndexF64};
use nanoflann_rs::{DataSource, DynamicKdTree, Scalar};

/// Fixed backing buffer (`data`, `n_capacity = data.len() / dim` points)
/// with a growable LOGICAL size -- the Rust-side twin of the oracle's own
/// mutable-`current_n` adaptor (`RefDynIndexF32`/`RefDynIndexF64`'s
/// `RowMajorDynAdaptor`, see that type's doc comment), so both sides see
/// the IDENTICAL dataset view at every step of an op sequence. `data` is
/// never resized or copied -- only the logical size `DataSource::
/// point_count()` reports changes, via `set_current_n`.
///
/// `current_n` is a `Cell`, not a plain field, so `set_current_n` takes
/// `&self` rather than `&mut self`: `DataSource<T>` is implemented for
/// `&GrowableFlat<'_, T>` (a SHARED reference), not for `GrowableFlat` by
/// value, specifically so a test can build one owned `GrowableFlat`, pass a
/// shared `&growable` into `DynamicKdTreeBuilder` (which then owns that
/// reference as its `DS`), and STILL call `growable.set_current_n(n)` on
/// its own local binding afterward -- both the tree's internal copy of the
/// reference and the external handle read/write the exact same `Cell`.
pub struct GrowableFlat<'a, T> {
    data: &'a [T],
    dim: usize,
    current_n: std::cell::Cell<usize>,
}

impl<'a, T: Scalar> GrowableFlat<'a, T> {
    /// Starts at logical size 0 (mirrors `RefDynIndexF32::build`/
    /// `RefDynIndexF64::build`'s starting state exactly -- no points are
    /// queryable until `set_current_n` grows the logical size AND the
    /// caller separately drives `DynamicKdTree::add_points`/
    /// `RefDynIndex*::add_points` to actually insert the newly-visible
    /// range). Panics if `dim == 0` or `data.len()` is not a multiple of
    /// `dim`.
    pub fn new(data: &'a [T], dim: usize) -> Self {
        assert!(dim > 0, "GrowableFlat::new: dim must be > 0");
        assert!(
            data.len().is_multiple_of(dim),
            "GrowableFlat::new: data.len() ({}) must be a multiple of dim ({dim})",
            data.len()
        );
        Self { data, dim, current_n: std::cell::Cell::new(0) }
    }

    /// Grows (or shrinks) the logical size `DataSource::point_count()`
    /// reports (via the `&GrowableFlat` impl below). Panics if `n * dim`
    /// would exceed the backing buffer's capacity. Mirrors
    /// `RefDynIndexF32::set_current_n`'s contract exactly, including that
    /// shrinking has no effect on points already added to a tree built over
    /// this dataset -- the caller owns keeping subsequent `add_points`/
    /// `remove_point` calls consistent with whatever `n` it sets (same
    /// caveat as the oracle side).
    pub fn set_current_n(&self, n: usize) {
        assert!(
            n * self.dim <= self.data.len(),
            "GrowableFlat::set_current_n: n*dim ({}) exceeds buffer capacity ({})",
            n * self.dim,
            self.data.len()
        );
        self.current_n.set(n);
    }

    /// Current logical size -- exposed so op-driving test code can compute
    /// the next `GrowAndAdd`'s contiguous `start` (`= current_n()`) without
    /// threading its own separate counter alongside the tree's.
    pub fn current_n(&self) -> usize {
        self.current_n.get()
    }
}

impl<'a, T: Scalar> DataSource<T> for &GrowableFlat<'a, T> {
    #[inline]
    fn point_count(&self) -> usize {
        self.current_n.get()
    }

    #[inline]
    fn point_component(&self, idx: usize, dim: usize) -> T {
        self.data[idx * self.dim + dim]
    }
}

// ----------------------------------------------------------------------
// Legal op-sequence generator
// ----------------------------------------------------------------------

/// One legal mutation of a dynamic forest's dataset membership. See
/// [`dyn_ops`]'s doc comment for the legality model that guarantees every
/// generated sequence is safe to replay, in order, against BOTH
/// `DynamicKdTree::add_points`/`remove_point` (which panics on a
/// contiguous-append violation) and the C++ oracle's `RefDynIndexF32/
/// F64::add_points` (which silently corrupts on the same violation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynOp {
    /// Grow the logical dataset size by `count` (`1..=64`, capped by
    /// remaining capacity) and add every one of those brand-new indices in
    /// one contiguous `add_points` call, starting exactly at the running
    /// point-count -- the only legal way to introduce a genuinely-new index
    /// (see `DynamicKdTree::add_points`'s contiguous-append contract doc
    /// comment).
    GrowAndAdd { count: usize },
    /// Lazily remove a currently-LIVE dataset index.
    Remove { live_idx: usize },
    /// Reactivate a currently-REMOVED dataset index (`add_points(idx, idx)`
    /// on both sides -- exempt from the contiguous-append contract).
    ReAdd { removed_idx: usize },
}

/// Deterministic (seeded) generator of a LEGAL op sequence: `n_ops`
/// entries, replayable in order against a fresh forest over a buffer of
/// capacity `total_capacity` (points) without ever violating either side's
/// add/remove legality contract. Maintains its own model of forest state
/// (running point-count, the live index set, the removed index set) purely
/// to decide what is legal next -- it never touches an actual tree.
///
/// Weights ~50/30/20 among `GrowAndAdd`/`Remove`/`ReAdd`, RESTRICTED each
/// step to whichever op kinds are currently legal (e.g. `Remove` is
/// excluded entirely once the live set is empty, `GrowAndAdd` once capacity
/// is exhausted) and renormalized over just those -- so, e.g., the very
/// first op is always `GrowAndAdd` (nothing exists yet to remove or
/// re-add). Given enough `n_ops` against a `total_capacity` that comfortably
/// exceeds 8 (the smallest point-count that forces a slot-2+ merge), the
/// output naturally contains: multiple simultaneously non-empty slots and
/// merges (a structural consequence of `first0bit`'s binary-counter pattern
/// once point-count exceeds a few points -- see `dynamic.rs`'s module doc),
/// tombstone migrations (any `Remove` followed later by enough `GrowAndAdd`
/// growth to cross a further merge boundary), and drain-heavy stretches
/// (runs of `Remove` once the live set is large). See
/// `dyn_ops_legal_property_many_seeds` and
/// `dyn_ops_can_produce_merges_tombstone_migrations_and_drain_stretches`
/// (this module's `#[cfg(test)]`) for the property tests confirming both
/// claims.
pub fn dyn_ops(seed: u64, total_capacity: usize, n_ops: usize) -> Vec<DynOp> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut point_count = 0usize;
    let mut live: Vec<usize> = Vec::new();
    let mut removed: Vec<usize> = Vec::new();
    let mut ops = Vec::with_capacity(n_ops);

    for _ in 0..n_ops {
        let can_grow = point_count < total_capacity;
        let can_remove = !live.is_empty();
        let can_readd = !removed.is_empty();

        // (weight, kind) pairs restricted to currently-legal kinds only.
        let mut choices: Vec<(u32, u8)> = Vec::new();
        if can_grow {
            choices.push((50, 0));
        }
        if can_remove {
            choices.push((30, 1));
        }
        if can_readd {
            choices.push((20, 2));
        }
        assert!(
            !choices.is_empty(),
            "dyn_ops: no legal op available (point_count={point_count}, \
             total_capacity={total_capacity}, live={}, removed={})",
            live.len(),
            removed.len()
        );

        let total_weight: u32 = choices.iter().map(|&(w, _)| w).sum();
        let mut pick = rng.gen_range(0..total_weight);
        let mut kind = choices.last().expect("choices is non-empty (asserted above)").1;
        for &(w, k) in &choices {
            if pick < w {
                kind = k;
                break;
            }
            pick -= w;
        }

        match kind {
            0 => {
                let remaining = total_capacity - point_count;
                let cap = remaining.min(64);
                let count = rng.gen_range(1..=cap);
                for i in point_count..point_count + count {
                    live.push(i);
                }
                point_count += count;
                ops.push(DynOp::GrowAndAdd { count });
            }
            1 => {
                let pick_pos = rng.gen_range(0..live.len());
                let idx = live.swap_remove(pick_pos);
                removed.push(idx);
                ops.push(DynOp::Remove { live_idx: idx });
            }
            _ => {
                let pick_pos = rng.gen_range(0..removed.len());
                let idx = removed.swap_remove(pick_pos);
                live.push(idx);
                ops.push(DynOp::ReAdd { removed_idx: idx });
            }
        }
    }

    ops
}

/// Independent legality validator for a `DynOp` sequence -- used by
/// [`dyn_ops`]'s own property tests AND available to any test driving a
/// hand-written op sequence (e.g. the dedicated scenario tests in
/// `tests/xval_dynamic.rs`). Walks `ops` with its OWN forest-membership
/// model (built without reusing `dyn_ops`' internal generator state, so a
/// bug shared between generation and validation can't hide from this
/// check) and asserts:
/// - every `GrowAndAdd`'s new-index range starts exactly at the running
///   point-count (contiguous-append) and never exceeds `total_capacity`,
///   and `count` is within `1..=64`;
/// - `Remove` never targets a dead (already-removed, or never-added) index;
/// - `ReAdd` never targets a live (or never-added) index.
///
/// Panics with the first violation found (including its `ops` index), else
/// returns quietly.
pub fn validate_dyn_ops_legal(ops: &[DynOp], total_capacity: usize) {
    let mut point_count = 0usize;
    let mut live: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut removed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for (i, op) in ops.iter().enumerate() {
        match *op {
            DynOp::GrowAndAdd { count } => {
                assert!(
                    (1..=64).contains(&count),
                    "op {i}: GrowAndAdd count {count} out of legal range [1,64]"
                );
                assert!(
                    point_count + count <= total_capacity,
                    "op {i}: GrowAndAdd would exceed capacity: point_count={point_count} \
                     count={count} total_capacity={total_capacity}"
                );
                for j in point_count..point_count + count {
                    assert!(live.insert(j), "op {i}: GrowAndAdd re-introduces already-live index {j}");
                }
                point_count += count;
            }
            DynOp::Remove { live_idx } => {
                assert!(
                    live.remove(&live_idx),
                    "op {i}: Remove targets a non-live index {live_idx} (either already \
                     removed or never added)"
                );
                assert!(removed.insert(live_idx), "op {i}: internal validator bug: {live_idx} already in removed set");
            }
            DynOp::ReAdd { removed_idx } => {
                assert!(
                    removed.remove(&removed_idx),
                    "op {i}: ReAdd targets a non-removed index {removed_idx} (either still \
                     live or never added)"
                );
                assert!(live.insert(removed_idx), "op {i}: internal validator bug: {removed_idx} already live");
            }
        }
    }
}

/// Coverage statistics over one `DynOp` sequence: counts of each op kind,
/// plus `tombstone_migrations` -- the number of times a currently-removed
/// index's PHYSICAL slot residency changed because a later `GrowAndAdd`'s
/// merge (`pos = first0bit(point_count)`, same arithmetic as
/// `DynamicKdTree::add_points`) swallowed the slot it still physically
/// occupied. ONE shared implementation for two callers that both need this:
/// `dyn_ops`'s own generic capability property test (below, this module's
/// `#[cfg(test)]`) and `tests/xval_dynamic.rs`'s per-matrix-sequence
/// coverage assertions (fix-round-1 item 2) -- keeping this in one place
/// means the two checks can't silently drift into simulating the
/// first0bit/slot-migration rule differently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynOpStats {
    pub grow_and_add_count: usize,
    pub remove_count: usize,
    pub readd_count: usize,
    pub tombstone_migrations: usize,
}

/// Computes [`DynOpStats`] by replaying `ops` against a hand-simulated
/// slot-residency model (see the struct's doc comment). `first0bit` is
/// `dynamic.rs`'s private `First0Bit` duplicated here -- it's `pub(crate)`
/// to `nanoflann-rs`, not part of xval's dependency surface -- purely so
/// this simulation can determine, from a running `point_count` alone, which
/// slot a genuinely-new index lands in and which lower slots a merge
/// swallows.
pub fn dyn_ops_stats(ops: &[DynOp]) -> DynOpStats {
    fn first0bit(n: usize) -> usize {
        let mut num = n;
        let mut pos = 0usize;
        while num & 1 == 1 {
            num >>= 1;
            pos += 1;
        }
        pos
    }

    let mut stats = DynOpStats::default();
    let mut point_count = 0usize;
    // dataset-index -> slot it's physically resident in right now (mirrors
    // DynamicKdTree::add_points' bookkeeping, ignoring reactivation for
    // SLOT purposes since lazy deletion never touches physical residency --
    // only whether an index counts as "removed" for the migration count).
    let mut slot_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut currently_removed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for op in ops {
        match *op {
            DynOp::GrowAndAdd { count } => {
                stats.grow_and_add_count += 1;
                for _ in 0..count {
                    let pos = first0bit(point_count);
                    if pos > 0 {
                        // Every index physically resident in slots 0..pos
                        // migrates to slot `pos`. If any such index is
                        // currently a tombstone, this is a genuine
                        // tombstone-slot migration.
                        for (&i, s) in slot_of.iter_mut() {
                            if *s < pos {
                                *s = pos;
                                if currently_removed.contains(&i) {
                                    stats.tombstone_migrations += 1;
                                }
                            }
                        }
                    }
                    slot_of.insert(point_count, pos);
                    point_count += 1;
                }
            }
            DynOp::Remove { live_idx } => {
                stats.remove_count += 1;
                currently_removed.insert(live_idx);
            }
            DynOp::ReAdd { removed_idx } => {
                stats.readd_count += 1;
                currently_removed.remove(&removed_idx);
            }
        }
    }

    stats
}

// ============================================================================
// Report meta capture -- best-effort machine/toolchain fingerprinting for
// `examples/report_data.rs`'s emitted `meta` object (see that file's module
// doc and Task 5b's brief). Every helper here is best-effort: any failure
// (command not found, file unreadable, non-UTF8 output) falls back to
// `"unknown"` (or `false` for the WSL flag) rather than panicking -- a
// report run on an unusual/minimal machine should still emit valid JSON,
// just with less detail. Factored out of `report_data.rs` (rather than
// living as private fns there) specifically so they're unit-testable here
// (see this module's `#[cfg(test)]` smoke test) and reusable by
// `xval::report::render`'s own tests without spinning up a whole binary.
// ============================================================================

/// First `model name` line from `/proc/cpuinfo` (the substring after the
/// first `:`, trimmed). `"unknown"` if the file is missing/unreadable or no
/// such line exists (e.g. a non-x86 `/proc/cpuinfo` layout, or a non-Linux
/// host where the file doesn't exist at all).
pub fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents
                .lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// `uname -sr` (kernel name + release), trimmed. `"unknown"` on failure
/// (e.g. `uname` not on `PATH`).
pub fn kernel_version() -> String {
    std::process::Command::new("uname")
        .args(["-sr"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// First line of `$CXX --version` if the `CXX` env var is set, else `c++
/// --version` -- best-effort proxy for the compiler `nanoflann-ref`'s
/// `build.rs` (via the `cc` crate) actually invokes to build the C++
/// oracle. `"unknown"` on failure.
pub fn cxx_compiler_version() -> String {
    let bin = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    std::process::Command::new(&bin)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// `git rev-parse --short HEAD`, with a `-dirty` suffix appended when `git
/// status --porcelain` reports a non-empty working tree. `"unknown"` if
/// `rev-parse` fails (e.g. running outside a git checkout, such as from a
/// packaged crate) -- the dirty check is skipped in that case too, since
/// there is no meaningful SHA to suffix.
pub fn git_sha() -> String {
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let Some(sha) = sha else {
        return "unknown".to_string();
    };

    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    if dirty {
        format!("{sha}-dirty")
    } else {
        sha
    }
}

/// `true` iff `/proc/version` contains "microsoft" (case-insensitive) --
/// the standard WSL kernel-version marker (both WSL1 and WSL2 stamp a
/// "-microsoft" or "microsoft-standard" tag into `uname -r`/`/proc/
/// version`). `false` if the file is missing/unreadable (macOS, a
/// container image without `/proc/version`, etc.) -- absence of evidence
/// is not evidence of WSL, so this never panics or errors, only reports
/// `false`.
pub fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version").map(|s| s.to_lowercase().contains("microsoft")).unwrap_or(false)
}

// ----------------------------------------------------------------------
// Apply / structure-parity helpers -- one instantiation per scalar type
// (the dynamic forest is L2-only, see `RefDynIndexF32`'s doc comment, so
// there is no metric axis to abstract over here, unlike the static
// `define_rust_index!` macro above).
// ----------------------------------------------------------------------

macro_rules! define_dyn_apply {
    ($t:ty, $RefDyn:ident, $apply_fn:ident, $assert_struct_fn:ident) => {
        /// Applies one `DynOp` identically to the Rust forest (`rust_tree`,
        /// backed by `growable`) and the oracle (`oracle`) -- SAME
        /// underlying coordinate buffer on both sides (the caller must have
        /// built `growable`/`oracle` from the exact same flat buffer), SAME
        /// op, so a LEGAL op sequence (see [`dyn_ops`]) keeps both sides'
        /// membership bookkeeping in lockstep at every step.
        pub fn $apply_fn<'a, 'b>(
            op: &DynOp,
            growable: &'b GrowableFlat<'a, $t>,
            rust_tree: &mut DynamicKdTree<$t, nanoflann_rs::DynDim, &'b GrowableFlat<'a, $t>>,
            oracle: &mut $RefDyn<'a>,
        ) {
            match *op {
                DynOp::GrowAndAdd { count } => {
                    let start = growable.current_n();
                    let new_n = start + count;
                    growable.set_current_n(new_n);
                    rust_tree.add_points(start, new_n - 1);
                    oracle.set_current_n(new_n);
                    oracle.add_points(start as u32, (new_n - 1) as u32);
                }
                DynOp::Remove { live_idx } => {
                    rust_tree.remove_point(live_idx);
                    oracle.remove_point(live_idx);
                }
                DynOp::ReAdd { removed_idx } => {
                    rust_tree.add_points(removed_idx, removed_idx);
                    oracle.add_points(removed_idx as u32, removed_idx as u32);
                }
            }
        }

        /// Structure parity: `tree_count`, every slot's point list
        /// (element-for-element, EXACT order -- not membership-only), the
        /// `tree_index()` vector, and `removed_len()`/`removed_count()`.
        /// Panics with a full dump of both sides on the first mismatch
        /// found (call through `with_ctx` at the call site to attach
        /// seed/op-index context).
        pub fn $assert_struct_fn<'a>(
            rust_tree: &DynamicKdTree<$t, nanoflann_rs::DynDim, &GrowableFlat<'a, $t>>,
            oracle: &$RefDyn<'a>,
        ) {
            let r_tc = rust_tree.tree_count();
            let c_tc = oracle.tree_count();
            assert_eq!(r_tc, c_tc, "tree_count mismatch: rust={r_tc} cpp={c_tc}");

            for slot in 0..r_tc {
                let r = rust_tree.point_indices_of_slot(slot);
                let c = oracle.slot_vacc(slot);
                assert_eq!(
                    r,
                    c.as_slice(),
                    "slot {slot} vind (point list) mismatch: rust={:?} cpp={:?}",
                    r,
                    c
                );
            }

            let r_ti = rust_tree.tree_index();
            let c_ti = oracle.tree_index();
            assert_eq!(
                r_ti,
                c_ti.as_slice(),
                "tree_index mismatch: rust={:?} cpp={:?}",
                r_ti,
                c_ti
            );

            let r_rm = rust_tree.removed_len();
            let c_rm = oracle.removed_count();
            assert_eq!(r_rm, c_rm, "removed count mismatch: rust removed_len={r_rm} cpp removed_count={c_rm}");
        }
    };
}

define_dyn_apply!(f32, RefDynIndexF32, apply_dyn_op_f32, assert_dyn_structure_equal_f32);
define_dyn_apply!(f64, RefDynIndexF64, apply_dyn_op_f64, assert_dyn_structure_equal_f64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{self, AssertUnwindSafe};

    // Deliberately does NOT touch the global panic hook (`panic::set_hook`):
    // cargo test runs tests in parallel by default, and the hook is
    // process-global, so mutating it here could race with another thread's
    // panic. The tradeoff is that a genuine panic caught below still prints
    // once via the default hook -- harmless noise, not a correctness issue.
    fn assert_panics<F: FnOnce() + std::panic::UnwindSafe>(f: F, what: &str) {
        let result = panic::catch_unwind(AssertUnwindSafe(f));
        assert!(result.is_err(), "{what}: expected a panic, but it did not panic");
    }

    // ------------------------------------------------------------------
    // ulp_diff
    // ------------------------------------------------------------------

    #[test]
    fn ulp_diff_identical_is_zero() {
        assert_eq!(ulp_diff_f64(1.0, 1.0), 0);
        assert_eq!(ulp_diff_f32(1.0, 1.0), 0);
    }

    #[test]
    fn ulp_diff_zero_and_negative_zero_is_zero() {
        assert_eq!(ulp_diff_f64(0.0, -0.0), 0);
    }

    #[test]
    fn ulp_diff_nan_is_max() {
        assert_eq!(ulp_diff_f64(f64::NAN, 1.0), u64::MAX);
        assert_eq!(ulp_diff_f64(1.0, f64::NAN), u64::MAX);
    }

    #[test]
    fn ulp_diff_cross_sign_nonzero_is_max() {
        assert_eq!(ulp_diff_f64(1e-300, -1e-300), u64::MAX);
        assert_eq!(ulp_diff_f64(0.0, -1.0), u64::MAX);
    }

    #[test]
    fn ulp_diff_adjacent_floats_is_one() {
        let a = 1.0f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_diff_f64(a, b), 1);
        assert_eq!(ulp_diff_f64(b, a), 1); // symmetric
    }

    #[test]
    fn ulp_diff_adjacent_negative_floats_is_one() {
        let a = -1.0f64;
        let b = f64::from_bits(a.to_bits() + 1); // more negative by one ULP
        assert_eq!(ulp_diff_f64(a, b), 1);
    }

    // ------------------------------------------------------------------
    // assert_knn_equal_f64 — ties == false (positional, bit-exact)
    // ------------------------------------------------------------------

    #[test]
    fn knn_equal_passes_on_identical_input() {
        let idx = [1u32, 0, 2];
        let dist = [0.5f64, 1.5, 3.0];
        assert_knn_equal_f64((&idx, &dist), (&idx, &dist), false);
    }

    #[test]
    fn knn_equal_panics_on_index_mismatch_same_rank() {
        let r_idx = [1u32, 0, 2];
        let r_dist = [0.5f64, 1.5, 3.0];
        let c_idx = [1u32, 2, 0]; // rank 1/2 swapped
        let c_dist = [0.5f64, 1.5, 3.0];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), false),
            "index mismatch at same rank must panic",
        );
    }

    #[test]
    fn knn_equal_panics_on_distance_bit_mismatch() {
        let idx = [1u32, 0];
        let r_dist = [0.5f64, 1.5];
        let c_dist = [0.5f64, f64::from_bits(1.5f64.to_bits() + 1)]; // 1 ULP off
        assert_panics(
            || assert_knn_equal_f64((&idx, &r_dist), (&idx, &c_dist), false),
            "1-ULP distance mismatch at ties=false must panic",
        );
    }

    #[test]
    fn knn_equal_panics_on_count_mismatch() {
        let r_idx = [1u32, 0];
        let r_dist = [0.5f64, 1.5];
        let c_idx = [1u32];
        let c_dist = [0.5f64];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), false),
            "found-count mismatch must panic",
        );
    }

    #[test]
    fn knn_equal_panics_on_cpp_extra_entry() {
        // The reverse direction of the count-mismatch check above: cpp has
        // MORE entries than rust.
        let r_idx = [1u32];
        let r_dist = [0.5f64];
        let c_idx = [1u32, 0];
        let c_dist = [0.5f64, 1.5];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), false),
            "cpp-extra-entry count mismatch must panic",
        );
    }

    // ------------------------------------------------------------------
    // assert_knn_equal_f64 — ties == true (exact-tie-group mode)
    // ------------------------------------------------------------------

    #[test]
    fn knn_equal_ties_passes_when_exact_tie_group_order_differs_but_multiset_matches() {
        // Two entries tied at distance 1.0 (BIT-EQUAL to each other), swapped
        // between rust/cpp order within the group -- must pass under
        // ties=true (positional at ties=false would fail this).
        let r_idx = [0u32, 1, 2];
        let r_dist = [1.0f64, 1.0, 5.0];
        let c_idx = [1u32, 0, 2];
        let c_dist = [1.0f64, 1.0, 5.0];
        assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), true);
    }

    #[test]
    fn knn_equal_ties_panics_when_multiset_disagrees() {
        let r_idx = [0u32, 1, 2];
        let r_dist = [1.0f64, 1.0, 5.0];
        let c_idx = [0u32, 3, 2]; // 3 not in rust's tie group at all
        let c_dist = [1.0f64, 1.0, 5.0];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), true),
            "tie-group index multiset disagreement must panic",
        );
    }

    /// Regression test for the exact bug flagged in code review round 1:
    /// the original `ties=true` implementation compared ONLY index
    /// multisets within a matched group and NEVER compared distances
    /// cross-side at all, so two entries with the SAME index (even in the
    /// SAME order) but DIFFERENT distances would silently pass. This is
    /// precisely the scenario `mutation_canary_distance_perturbation`
    /// (tests/xval_radius_box.rs) also exercises end-to-end against the
    /// real trees.
    #[test]
    fn knn_equal_ties_panics_on_matching_indices_diverging_distances() {
        // Both sides see indices [0, 1] in the SAME order, each internally
        // tied (rust: 5.0/5.0; cpp: 6.0/6.0) -- group BOUNDARIES match
        // (both are one 2-element run), and the INDEX multiset matches
        // exactly, but the two sides' shared tie VALUE differs (5.0 vs
        // 6.0). A pure index-multiset check would wrongly PASS this; the
        // fixed comparator must panic on the cross-side distance mismatch.
        let r_idx = [0u32, 1];
        let r_dist = [5.0f64, 5.0];
        let c_idx = [0u32, 1];
        let c_dist = [6.0f64, 6.0];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), true),
            "matching indices with diverging cross-side distances must panic under ties=true",
        );
    }

    #[test]
    fn knn_equal_ties_panics_on_group_boundary_mismatch_rust_longer() {
        // rust: all 3 entries exactly tied at 1.0 (one 3-element run). cpp:
        // only the first two are tied; the third is a different value (a
        // shorter run). Same overall distance VALUES appear on each side at
        // each rank pairwise except this shape difference is what should be
        // caught as a group-boundary mismatch (rust's run is longer).
        let r_idx = [0u32, 1, 2];
        let r_dist = [1.0f64, 1.0, 1.0];
        let c_idx = [0u32, 1, 2];
        let c_dist = [1.0f64, 1.0, 2.0];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), true),
            "group-boundary mismatch (rust's exact-tie run longer than cpp's) must panic",
        );
    }

    #[test]
    fn knn_equal_ties_panics_on_group_boundary_mismatch_cpp_longer() {
        // The reverse direction: rust's first two entries are distinct (no
        // tie at rank 0), cpp's first two entries ARE tied -- so rust's run
        // starting at rank 0 is length 1, cpp's is length 2.
        let r_idx = [0u32, 1, 2];
        let r_dist = [1.0f64, 2.0, 2.0];
        let c_idx = [0u32, 1, 2];
        let c_dist = [1.0f64, 1.0, 2.0];
        assert_panics(
            || assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), true),
            "group-boundary mismatch (cpp's exact-tie run longer than rust's) must panic",
        );
    }

    // ------------------------------------------------------------------
    // assert_radius_equal_f64
    // ------------------------------------------------------------------

    #[test]
    fn radius_equal_sorted_passes_on_identical_input() {
        let r = [(1u32, 0.5f64), (0u32, 1.5)];
        assert_radius_equal_f64(&r, &r, true, false);
    }

    #[test]
    fn radius_equal_sorted_panics_on_order_mismatch() {
        let rust = [(1u32, 0.5f64), (0u32, 1.5)];
        let cpp = [(0u32, 1.5f64), (1u32, 0.5)];
        assert_panics(
            || assert_radius_equal_f64(&rust, &cpp, true, false),
            "sorted radius order mismatch must panic",
        );
    }

    #[test]
    fn radius_equal_sorted_ties_panics_on_matching_indices_diverging_distances() {
        // Same regression shape as knn_equal_ties_panics_on_matching_indices_diverging_distances,
        // routed through the radius comparator's sorted=true path (which
        // delegates to assert_knn_equal_f64 with `ties` forwarded).
        let rust = [(0u32, 5.0f64), (1u32, 5.0)];
        let cpp = [(0u32, 6.0f64), (1u32, 6.0)];
        assert_panics(
            || assert_radius_equal_f64(&rust, &cpp, true, true),
            "radius sorted=true ties=true must still catch a cross-side distance divergence",
        );
    }

    #[test]
    fn radius_equal_unsorted_passes_regardless_of_order() {
        let rust = [(1u32, 0.5f64), (0u32, 1.5)];
        let cpp = [(0u32, 1.5f64), (1u32, 0.5)];
        assert_radius_equal_f64(&rust, &cpp, false, false);
    }

    #[test]
    fn radius_equal_unsorted_panics_on_index_set_mismatch() {
        let rust = [(1u32, 0.5f64), (0u32, 1.5)];
        let cpp = [(1u32, 0.5f64), (2u32, 1.5)]; // index 0 vs 2
        assert_panics(
            || assert_radius_equal_f64(&rust, &cpp, false, false),
            "unsorted radius index-set mismatch must panic",
        );
    }

    #[test]
    fn radius_equal_unsorted_panics_on_any_distance_divergence() {
        // The unsorted branch requires distances BIT-EQUAL per matched
        // index unconditionally (no numeric tolerance surface at all --
        // `ties` is not consulted in this branch, see the doc comment).
        let rust = [(0u32, 1.0f64)];
        let cpp = [(0u32, f64::from_bits(1.0f64.to_bits() + 1))]; // 1 ULP off
        assert_panics(
            || assert_radius_equal_f64(&rust, &cpp, false, false),
            "unsorted radius distance divergence (even 1 ULP) must panic",
        );
        assert_panics(
            || assert_radius_equal_f64(&rust, &cpp, false, true),
            "unsorted radius distance divergence must panic regardless of `ties`",
        );
    }

    // ------------------------------------------------------------------
    // Generators — determinism + basic shape sanity (not the TDD focus of
    // this task, but cheap and catches obvious bugs).
    // ------------------------------------------------------------------

    #[test]
    fn uniform_is_deterministic_and_in_range() {
        let a = uniform(42, 50, 3);
        let b = uniform(42, 50, 3);
        assert_eq!(a, b);
        assert_eq!(a.len(), 150);
        assert!(a.iter().all(|&x| (-10.0..10.0).contains(&x)));
    }

    #[test]
    fn uniform_different_seeds_differ() {
        let a = uniform(1, 20, 2);
        let b = uniform(2, 20, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn all_identical_is_exact() {
        let a = all_identical(10, 4);
        assert_eq!(a.len(), 40);
        assert!(a.iter().all(|&x| x == 1.25));
    }

    #[test]
    fn with_duplicates_produces_exact_copies() {
        let data = with_duplicates(7, 20, 3, 0.5);
        assert_eq!(data.len(), 60);
        // At least one duplicate row must equal some earlier row exactly.
        let rows: Vec<&[f64]> = data.chunks(3).collect();
        let mut found_dup = false;
        for i in 1..rows.len() {
            if rows[..i].contains(&rows[i]) {
                found_dup = true;
                break;
            }
        }
        assert!(found_dup, "expected at least one exact duplicate row");
    }

    #[test]
    fn exponential_spacing_is_strictly_decreasing_then_zero() {
        let v = exponential_spacing(20);
        assert_eq!(v.len(), 20);
        for w in v.windows(2) {
            assert!(w[0] >= w[1], "expected non-increasing spine: {:?}", w);
        }
    }

    #[test]
    fn on_circle_so2_angle_is_wrapped() {
        let v = on_circle_so2(3, 30);
        assert_eq!(v.len(), 60);
        for chunk in v.chunks(2) {
            let angle = chunk[1];
            assert!(
                (-core::f64::consts::PI..=core::f64::consts::PI).contains(&angle),
                "angle out of range: {angle}"
            );
        }
    }

    #[test]
    fn queries_includes_exact_dataset_copies() {
        let data = uniform(11, 30, 2);
        let q = queries(99, &data, 2, 30);
        assert_eq!(q.len(), 60);
        let data_rows: Vec<&[f64]> = data.chunks(2).collect();
        let q_rows: Vec<&[f64]> = q.chunks(2).collect();
        let has_exact_copy = q_rows.iter().any(|qr| data_rows.contains(qr));
        assert!(has_exact_copy, "expected at least one exact dataset-point query");
    }

    #[test]
    fn to_f32_casts_every_element() {
        let v = vec![1.5f64, -2.25, 0.0];
        let got = to_f32(&v);
        assert_eq!(got, vec![1.5f32, -2.25, 0.0]);
    }

    #[test]
    fn to_array3_reshapes_row_major_triples() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let got = to_array3(&flat);
        assert_eq!(got, vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    #[should_panic]
    fn to_array3_non_multiple_of_3_panics() {
        let flat = vec![1.0f32, 2.0];
        let _ = to_array3(&flat);
    }

    #[test]
    fn round_robin_wraps_after_last_query() {
        let data = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0]; // 3 points, dim 2
        let mut rr = RoundRobin::new(&data, 2);
        assert_eq!(rr.next(), &[0.0, 0.0]);
        assert_eq!(rr.next(), &[1.0, 1.0]);
        assert_eq!(rr.next(), &[2.0, 2.0]);
        assert_eq!(rr.next(), &[0.0, 0.0], "must wrap back to index 0");
        assert_eq!(rr.next(), &[1.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn round_robin_zero_dim_panics() {
        let data = [1.0f32];
        let _ = RoundRobin::new(&data, 0);
    }

    // ------------------------------------------------------------------
    // brute_force_knn_l2_* / score_query_* -- tiny hand-checked cases
    // ------------------------------------------------------------------

    #[test]
    fn brute_force_knn_returns_nearest_k_sorted_ascending() {
        // dim 1: points at 0, 1, 2, 3. Query at 0.0 -> squared distances
        // 0, 1, 4, 9 for indices 0, 1, 2, 3 respectively. k=2 -> [(0,0),(1,1)].
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let got = brute_force_knn_l2_f32(&data, 1, &[0.0], 2);
        assert_eq!(got, vec![(0u32, 0.0f32), (1u32, 1.0f32)]);
    }

    #[test]
    fn brute_force_knn_ties_broken_by_ascending_index() {
        // dim 1: points at -1 and 1. Query at 0.0 -> both squared distance
        // 1.0 -- a genuine tie, must resolve to index order 0 then 1.
        let data = [-1.0f32, 1.0];
        let got = brute_force_knn_l2_f32(&data, 1, &[0.0], 2);
        assert_eq!(got, vec![(0u32, 1.0f32), (1u32, 1.0f32)]);
    }

    #[test]
    fn brute_force_knn_k_greater_than_n_truncates() {
        let data = [0.0f32, 5.0];
        let got = brute_force_knn_l2_f32(&data, 1, &[0.0], 10);
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn brute_force_knn_f64_matches_hand_calc_dim2() {
        // Two points at (0,0) and (3,4) (a 3-4-5 triangle), query at
        // (0,0) -> squared distances 0 and 25.
        let data = [0.0f64, 0.0, 3.0, 4.0];
        let got = brute_force_knn_l2_f64(&data, 2, &[0.0, 0.0], 2);
        assert_eq!(got, vec![(0u32, 0.0f64), (1u32, 25.0f64)]);
    }

    #[test]
    fn score_query_exact_match_same_order_zero_error() {
        let gt = [(0u32, 1.0f32), (1u32, 4.0)];
        let got = [(0u32, 1.0f32), (1u32, 4.0)];
        let s = score_query_f32(&gt, &got);
        assert!(s.exact);
        assert_eq!(s.rel_dist_errors, vec![0.0, 0.0]);
    }

    #[test]
    fn score_query_exact_match_different_order_still_exact() {
        // Same index SET, different order (e.g. a different tie rule) --
        // `exact` is order-independent by design.
        let gt = [(0u32, 1.0f32), (1u32, 1.0)];
        let got = [(1u32, 1.0f32), (0u32, 1.0)];
        let s = score_query_f32(&gt, &got);
        assert!(s.exact);
    }

    #[test]
    fn score_query_different_index_set_is_not_exact() {
        let gt = [(0u32, 1.0f32)];
        let got = [(2u32, 1.0f32)];
        let s = score_query_f32(&gt, &got);
        assert!(!s.exact);
    }

    #[test]
    fn score_query_rel_dist_error_hand_calc() {
        // gt distance 4.0, got distance 5.0 -> |5-4|/4 = 0.25 exactly.
        let gt = [(0u32, 4.0f32)];
        let got = [(0u32, 5.0f32)];
        let s = score_query_f32(&gt, &got);
        assert_eq!(s.rel_dist_errors, vec![0.25]);
    }

    #[test]
    fn score_query_length_mismatch_leaves_rel_dist_errors_empty() {
        let gt = [(0u32, 1.0f32), (1u32, 2.0)];
        let got = [(0u32, 1.0f32)];
        let s = score_query_f32(&gt, &got);
        assert!(s.rel_dist_errors.is_empty());
    }

    #[test]
    fn score_query_zero_gt_distance_falls_back_to_absolute_error() {
        // gt distance 0.0 (exact dataset-point query), got distance 0.5 --
        // relative error against a zero denominator is undefined, so the
        // fallback is the raw magnitude (0.5), not inf/NaN.
        let gt = [(0u32, 0.0f32)];
        let got = [(0u32, 0.5f32)];
        let s = score_query_f32(&gt, &got);
        assert_eq!(s.rel_dist_errors, vec![0.5]);
    }

    #[test]
    fn score_query_zero_gt_and_zero_got_distance_is_zero_error() {
        let gt = [(0u32, 0.0f32)];
        let got = [(0u32, 0.0f32)];
        let s = score_query_f32(&gt, &got);
        assert_eq!(s.rel_dist_errors, vec![0.0]);
    }

    // ------------------------------------------------------------------
    // score_exact_tie_aware_f32 -- controller-mandated tie-aware exactness
    // scorer (replaces the order-independent-set-match `.exact` field for
    // report_data.rs's ground-truth exactness metric, which was too strict
    // on duplicate-heavy datasets: see `QueryScore::exact`'s doc comment).
    //
    // Hand-built dim-1 dataset: p0=0.0 (query-exact, dist 0), p1=1.0 (dist
    // 1), p2=-1.0 (dist 1 -- GENUINELY TIED with p1), p3=10.0 (dist 100,
    // clearly excluded from any k=2 top-2). query = [0.0]. GT for k=2, tied
    // at rank 1 broken by ascending index -> [(0,0.0),(1,1.0)].
    // ------------------------------------------------------------------

    const TIE_DATA: [f32; 4] = [0.0, 1.0, -1.0, 10.0];
    const TIE_QUERY: [f32; 1] = [0.0];

    #[test]
    fn tie_aware_accepts_a_different_but_legitimate_tie_winner() {
        let gt = [(0u32, 0.0f32), (1u32, 1.0)];
        // A valid alternate tie resolution: index 2 (p2, ALSO true-distance
        // 1.0) instead of index 1 -- e.g. a tree's traversal-order tie rule
        // picking a different, equally-correct, winner of the rank-1 tie.
        let got = [(0u32, 0.0f32), (2u32, 1.0)];

        // OLD (order-independent set match) scores this a MISS: {0,1} != {0,2}.
        let old = score_query_f32(&gt, &got);
        assert!(!old.exact, "old set-match scorer is expected to reject a different tie winner");

        // NEW (tie-aware) scores this EXACT: both conditions hold (distance
        // list bit-equal positionally; index 2's recomputed true distance
        // -- (-1.0)^2 == 1.0 -- matches its reported distance).
        assert!(
            score_exact_tie_aware_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got),
            "tie-aware scorer must accept a legitimate alternate tie resolution"
        );
    }

    #[test]
    fn tie_aware_rejects_a_genuinely_wrong_neighbor() {
        let gt = [(0u32, 0.0f32), (1u32, 1.0)];
        // Index 3 (p3, TRUE distance 100.0) reported with a fabricated
        // distance of 1.0 -- not a tie, a genuinely wrong result.
        let got = [(0u32, 0.0f32), (3u32, 1.0)];

        let old = score_query_f32(&gt, &got);
        assert!(!old.exact, "old set-match scorer must reject this too: {{0,1}} != {{0,3}}");

        assert!(
            !score_exact_tie_aware_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got),
            "tie-aware scorer must reject a fabricated/mismatched distance-to-index pairing"
        );
    }

    // ------------------------------------------------------------------
    // brute_force_knn_l2_live_f32 / score_exact_tie_aware_live_f32 -- M2
    // Task 5's live-set-filter extension (TDD, per the task brief: written
    // BEFORE the implementation exists). Reuses TIE_DATA/TIE_QUERY from the
    // non-live tests above.
    // ------------------------------------------------------------------

    #[test]
    fn brute_force_knn_l2_live_excludes_dead_indices_hand_calc() {
        // dim 1: points at 0, 1, 2, 3 -> squared distances to query [0.0]
        // are 0, 1, 4, 9 for indices 0, 1, 2, 3. live[1] = false (removed):
        // must be excluded even though it would otherwise be the 2nd
        // nearest, so k=2 over the live set is [(0,0),(2,4)], NOT [(0,0),(1,1)].
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let live = [true, false, true, true];
        let got = brute_force_knn_l2_live_f32(&data, 1, &[0.0], 2, &live);
        assert_eq!(got, vec![(0u32, 0.0f32), (2u32, 4.0f32)]);
    }

    #[test]
    fn brute_force_knn_l2_live_all_dead_returns_empty() {
        let data = [0.0f32, 1.0];
        let live = [false, false];
        let got = brute_force_knn_l2_live_f32(&data, 1, &[0.0], 2, &live);
        assert!(got.is_empty());
    }

    #[test]
    fn tie_aware_live_rejects_a_removed_index_even_with_correct_distance() {
        // p1 (index 1) is REMOVED (live[1] = false). Ground truth over the
        // live set (via brute_force_knn_l2_live_f32) therefore excludes it:
        // k=2 -> [(0,0.0),(2,1.0)] (p2, also true-distance 1.0, takes the
        // rank-1 slot instead).
        let live = [true, false, true, true];
        let gt = brute_force_knn_l2_live_f32(&TIE_DATA, 1, &TIE_QUERY, 2, &live);
        assert_eq!(gt, vec![(0u32, 0.0f32), (2u32, 1.0f32)]);

        // A candidate reports index 1 (removed) with its numerically
        // CORRECT true distance (1.0) at rank 1 -- the distance LIST is
        // bit-equal to gt positionally (0.0, 1.0), and index 1's recomputed
        // true distance does equal 1.0, so the OLD non-live-aware scorer
        // (sanity-checked below) wrongly accepts this. The live-aware
        // scorer must reject it because index 1 is dead.
        let got = [(0u32, 0.0f32), (1u32, 1.0)];
        assert!(
            score_exact_tie_aware_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got),
            "sanity: the OLD non-live-aware scorer has no liveness concept and accepts this"
        );
        assert!(
            !score_exact_tie_aware_live_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got, &live),
            "live-aware scorer must reject a removed index even with a numerically correct distance"
        );
    }

    #[test]
    fn tie_aware_live_accepts_a_live_alternate_tie_winner() {
        // index 2 is LIVE and a legitimate alternate tie winner (mirrors
        // tie_aware_accepts_a_different_but_legitimate_tie_winner above) --
        // must still be accepted once a (trivial, nothing-removed) live
        // filter is threaded through.
        let live = [true, true, true, true];
        let gt = [(0u32, 0.0f32), (1u32, 1.0)];
        let got = [(0u32, 0.0f32), (2u32, 1.0)];
        assert!(score_exact_tie_aware_live_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got, &live));
    }

    #[test]
    fn tie_aware_live_rejects_a_genuinely_wrong_neighbor() {
        // Mirrors tie_aware_rejects_a_genuinely_wrong_neighbor above, with a
        // trivial (nothing-removed) live filter threaded through -- the
        // live check must not mask the existing wrong-distance rejection.
        let live = [true, true, true, true];
        let gt = [(0u32, 0.0f32), (1u32, 1.0)];
        let got = [(0u32, 0.0f32), (3u32, 1.0)]; // p3's true distance is 100.0, not 1.0
        assert!(!score_exact_tie_aware_live_f32(&TIE_DATA, 1, &TIE_QUERY, &gt, &got, &live));
    }

    // ------------------------------------------------------------------
    // sample_distinct_indices
    // ------------------------------------------------------------------

    #[test]
    fn sample_distinct_indices_is_deterministic_and_distinct() {
        let a = sample_distinct_indices(7, 100, 20);
        let b = sample_distinct_indices(7, 100, 20);
        assert_eq!(a, b, "same seed must reproduce the identical sample");
        assert_eq!(a.len(), 20);
        let set: std::collections::BTreeSet<usize> = a.iter().copied().collect();
        assert_eq!(set.len(), 20, "every sampled index must be distinct");
        assert!(a.iter().all(|&i| i < 100));
    }

    #[test]
    fn sample_distinct_indices_full_count_is_a_permutation_of_0_n() {
        let a = sample_distinct_indices(3, 10, 10);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<usize>>());
    }

    #[test]
    #[should_panic(expected = "count (5) must be <= n (3)")]
    fn sample_distinct_indices_count_greater_than_n_panics() {
        let _ = sample_distinct_indices(1, 3, 5);
    }

    // ------------------------------------------------------------------
    // median_of / timed_median_ms
    // ------------------------------------------------------------------

    #[test]
    fn median_of_odd_length_is_middle_element() {
        assert_eq!(median_of(vec![7.0, 1.0, 5.0, 3.0, 9.0, 2.0, 4.0]), 4.0);
    }

    #[test]
    fn median_of_single_element_is_itself() {
        assert_eq!(median_of(vec![42.0]), 42.0);
    }

    #[test]
    fn median_of_even_length_averages_middle_two() {
        assert_eq!(median_of(vec![1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_of_is_robust_to_one_outlier() {
        // 7 samples, one wild outlier -- median must ignore it (this is
        // exactly why the perf gate/report collector use median-of-7, not
        // mean, over real timed runs on a possibly-noisy machine).
        let v = vec![1.0, 1.1, 1.05, 1.2, 1.15, 1.1, 1000.0];
        assert_eq!(median_of(v), 1.1);
    }

    #[test]
    fn timed_median_ms_runs_warmup_plus_n_and_returns_nonnegative() {
        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let ms = timed_median_ms(5, || {
            calls.set(calls.get() + 1);
        });
        assert_eq!(calls.get(), 6, "expected 1 warmup + 5 timed calls");
        assert!(ms >= 0.0);
    }

    // ------------------------------------------------------------------
    // Build helpers — smoke test (real xval assertions live in tests/*.rs)
    // ------------------------------------------------------------------

    #[test]
    fn build_rust_f64_l2_knn_matches_hand_expectation() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0]; // 3 pts, dim 2
        let idx = build_rust_f64(&data, 2, XMetric::L2, 10, BuildThreads::Sequential);
        assert_eq!(idx.size(), 3);
        let (got_idx, _got_dist) = idx.knn(&[0.0, 0.0], 1, 0.0);
        assert_eq!(got_idx, vec![0]);
        let vind = idx.vind();
        let mut sorted_vind = vind.clone();
        sorted_vind.sort_unstable();
        assert_eq!(sorted_vind, vec![0, 1, 2]);
    }

    #[test]
    fn build_rust_f64_auto_and_threads2_wire_through_and_match_sequential_vind() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 2.0, -1.0, -1.0, 5.0, 5.0];
        let seq = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Sequential);
        let auto = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Auto);
        let threads2 =
            build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Threads(core::num::NonZeroU32::new(2).unwrap()));

        assert_eq!(seq.vind(), auto.vind(), "Auto vind must match Sequential");
        assert_eq!(seq.vind(), threads2.vind(), "Threads(2) vind must match Sequential");
    }

    // ------------------------------------------------------------------
    // knn_into / radius_into -- must match their allocating counterparts
    // bit-for-bit (Task 13 fix-round-2: allocation-symmetric query paths).
    // ------------------------------------------------------------------

    #[test]
    fn knn_into_matches_allocating_knn() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 2.0, -1.0, -1.0, 5.0, 5.0];
        let idx = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Sequential);
        let q = [0.5, 0.5];
        let (want_idx, want_dist) = idx.knn(&q, 3, 0.0);

        let mut got_idx = vec![0u32; 3];
        let mut got_dist = vec![0.0f64; 3];
        let found = idx.knn_into(&q, 3, 0.0, &mut got_idx, &mut got_dist);

        assert_eq!(found, want_idx.len());
        assert_eq!(&got_idx[..found], want_idx.as_slice());
        assert_eq!(&got_dist[..found], want_dist.as_slice(), "distances must be bit-identical");
    }

    #[test]
    #[should_panic]
    fn knn_into_panics_on_wrong_buffer_length() {
        let data = vec![0.0, 0.0, 1.0, 0.0];
        let idx = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Sequential);
        let mut short_idx = vec![0u32; 1];
        let mut dist = vec![0.0f64; 2];
        let _ = idx.knn_into(&[0.0, 0.0], 2, 0.0, &mut short_idx, &mut dist);
    }

    #[test]
    fn radius_into_matches_allocating_radius() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 2.0, -1.0, -1.0, 5.0, 5.0];
        let idx = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Sequential);
        let q = [0.0, 0.0];
        let want = idx.radius(&q, 10.0, true, 0.0);

        let mut out: Vec<ResultItem<u32, f64>> = Vec::new();
        let count = idx.radius_into(&q, 10.0, true, 0.0, &mut out);

        assert_eq!(count, want.len());
        assert_eq!(out.len(), want.len());
        for i in 0..want.len() {
            assert_eq!(out[i].index, want[i].0);
            assert_eq!(out[i].distance, want[i].1, "distances must be bit-identical at rank {i}");
        }
    }

    #[test]
    fn radius_into_reuses_and_resizes_across_calls_with_different_result_sizes() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0, 2.0, -1.0, -1.0, 5.0, 5.0];
        let idx = build_rust_f64(&data, 2, XMetric::L2, 2, BuildThreads::Sequential);
        let q = [0.0, 0.0];
        let mut out: Vec<ResultItem<u32, f64>> = Vec::new();

        let c1 = idx.radius_into(&q, 0.5, true, 0.0, &mut out);
        assert_eq!(c1, 1, "only the origin point itself is within radius 0.5");
        assert_eq!(out.len(), 1);

        let c2 = idx.radius_into(&q, 100.0, true, 0.0, &mut out);
        assert_eq!(c2, 5, "radius 100.0 should capture all 5 points");
        assert_eq!(out.len(), 5);

        let c3 = idx.radius_into(&q, 0.5, true, 0.0, &mut out);
        assert_eq!(c3, 1);
        assert_eq!(out.len(), 1, "buffer must shrink back down, not leave stale entries");
    }

    // ------------------------------------------------------------------
    // GrowableFlat -- basic Cell-based interior-mutability contract.
    // ------------------------------------------------------------------

    #[test]
    fn growable_flat_starts_at_logical_size_zero() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g = GrowableFlat::new(&data, 2);
        assert_eq!(g.current_n(), 0);
        assert_eq!((&g).point_count(), 0);
    }

    #[test]
    fn growable_flat_set_current_n_grows_point_count() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g = GrowableFlat::new(&data, 2);
        g.set_current_n(3);
        assert_eq!(g.current_n(), 3);
        assert_eq!((&g).point_count(), 3);
        assert_eq!((&g).point_component(2, 1), 6.0);
    }

    #[test]
    fn growable_flat_shared_reference_sees_mutation_through_owned_handle() {
        // The whole point of the Cell: a shared &GrowableFlat (as handed to
        // DynamicKdTreeBuilder) must observe set_current_n calls made
        // through the ORIGINAL owned binding afterward.
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let g = GrowableFlat::new(&data, 2);
        let ds_ref: &GrowableFlat<f64> = &g;
        assert_eq!(ds_ref.point_count(), 0);
        g.set_current_n(2);
        assert_eq!(ds_ref.point_count(), 2, "the reference must see the mutation through the Cell");
    }

    #[test]
    #[should_panic(expected = "exceeds buffer capacity")]
    fn growable_flat_set_current_n_panics_past_capacity() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let g = GrowableFlat::new(&data, 2);
        g.set_current_n(3); // only 2 points fit in a 4-element, dim-2 buffer
    }

    // ------------------------------------------------------------------
    // dyn_ops -- legality-model property tests (written before wiring the
    // suites, per the task brief's TDD note: these pin the generator's
    // contract independent of any tree).
    // ------------------------------------------------------------------

    #[test]
    fn dyn_ops_legal_property_many_seeds() {
        let total_capacity = 500usize;
        let n_ops = 300usize;
        for seed in 0u64..50 {
            let ops = dyn_ops(seed, total_capacity, n_ops);
            assert_eq!(ops.len(), n_ops, "seed {seed}: dyn_ops must return exactly n_ops entries");
            // Panics (with the seed's op index) on the first illegal op --
            // an independent model, not a call back into dyn_ops' own state.
            validate_dyn_ops_legal(&ops, total_capacity);
        }
    }

    #[test]
    fn dyn_ops_is_deterministic_for_a_given_seed() {
        let a = dyn_ops(12345, 200, 80);
        let b = dyn_ops(12345, 200, 80);
        assert_eq!(a, b, "same seed must reproduce the identical op sequence");
    }

    #[test]
    fn dyn_ops_first_op_is_always_grow_and_add() {
        // Nothing exists yet to Remove or ReAdd -- the very first op must
        // be GrowAndAdd regardless of seed.
        for seed in 0u64..20 {
            let ops = dyn_ops(seed, 100, 1);
            assert_eq!(ops.len(), 1);
            assert!(
                matches!(ops[0], DynOp::GrowAndAdd { .. }),
                "seed {seed}: first op must be GrowAndAdd, got {:?}",
                ops[0]
            );
        }
    }

    #[test]
    fn dyn_ops_never_exceeds_total_capacity() {
        let total_capacity = 37usize; // deliberately small & not a power of two
        for seed in 0u64..20 {
            let ops = dyn_ops(seed, total_capacity, 500);
            let mut point_count = 0usize;
            for op in &ops {
                if let DynOp::GrowAndAdd { count } = *op {
                    point_count += count;
                    assert!(
                        point_count <= total_capacity,
                        "seed {seed}: point_count {point_count} exceeded total_capacity {total_capacity}"
                    );
                }
            }
        }
    }

    #[test]
    fn dyn_ops_can_produce_merges_tombstone_migrations_and_drain_stretches() {
        // Statistical capability check across many seeds (not a per-seed
        // guarantee) -- see dyn_ops' doc comment for why these are expected
        // to emerge naturally from the weighted, state-restricted model.
        // Uses `dyn_ops_stats` for the tombstone-migration count -- the SAME
        // shared simulation `tests/xval_dynamic.rs`'s per-matrix-sequence
        // coverage assertions use (fix-round-1 item 2), so this generic
        // capability check and that per-matrix-config check can't silently
        // drift into simulating the first0bit/slot-migration rule
        // differently.
        let mut saw_point_count_past_merge_boundary = false; // > 8 => at least one slot-3+ merge occurred (first0bit structural guarantee)
        let mut saw_tombstone_migration = false;
        let mut saw_drain_heavy_stretch = false; // >= 5 consecutive Removes

        for seed in 0u64..30 {
            let ops = dyn_ops(seed, 200, 150);

            let stats = dyn_ops_stats(&ops);
            if stats.tombstone_migrations > 0 {
                saw_tombstone_migration = true;
            }

            let mut point_count = 0usize;
            let mut consecutive_removes = 0usize;
            for op in &ops {
                match *op {
                    DynOp::GrowAndAdd { count } => {
                        point_count += count;
                        consecutive_removes = 0;
                        if point_count > 8 {
                            saw_point_count_past_merge_boundary = true;
                        }
                    }
                    DynOp::Remove { .. } => {
                        consecutive_removes += 1;
                        if consecutive_removes >= 5 {
                            saw_drain_heavy_stretch = true;
                        }
                    }
                    DynOp::ReAdd { .. } => {
                        consecutive_removes = 0;
                    }
                }
            }
        }

        assert!(
            saw_point_count_past_merge_boundary,
            "no seed's point_count ever exceeded 8 -- needed for a slot-3+ merge (see first0bit)"
        );
        assert!(
            saw_tombstone_migration,
            "no seed's sequence ever had a tombstoned index physically swallowed by a later \
             merge -- tombstone-migration scenarios never occurred"
        );
        assert!(
            saw_drain_heavy_stretch,
            "no seed produced a drain-heavy stretch (>= 5 consecutive Remove ops)"
        );
    }

    #[test]
    fn dyn_ops_stats_counts_op_kinds_correctly_on_a_hand_crafted_sequence() {
        // Independent hand check of dyn_ops_stats' op-kind counters (the
        // tombstone_migrations field is exercised by the test above and by
        // the tombstone_migration scenario's hand-derivation in
        // dynamic.rs/tests/xval_dynamic.rs -- this just pins the three
        // trivial counts).
        let ops = vec![
            DynOp::GrowAndAdd { count: 4 },
            DynOp::Remove { live_idx: 0 },
            DynOp::Remove { live_idx: 1 },
            DynOp::ReAdd { removed_idx: 0 },
        ];
        let stats = dyn_ops_stats(&ops);
        assert_eq!(stats.grow_and_add_count, 1);
        assert_eq!(stats.remove_count, 2);
        assert_eq!(stats.readd_count, 1);
    }

    #[test]
    #[should_panic(expected = "Remove targets a non-live index")]
    fn validate_dyn_ops_legal_catches_a_hand_crafted_illegal_remove() {
        // Removes index 0 twice in a row -- illegal (already dead the
        // second time) -- confirms the validator itself actually detects a
        // violation, not just that legal sequences pass it vacuously.
        let ops = vec![
            DynOp::GrowAndAdd { count: 1 },
            DynOp::Remove { live_idx: 0 },
            DynOp::Remove { live_idx: 0 },
        ];
        validate_dyn_ops_legal(&ops, 10);
    }

    #[test]
    #[should_panic(expected = "ReAdd targets a non-removed index")]
    fn validate_dyn_ops_legal_catches_a_hand_crafted_illegal_readd() {
        // ReAdds index 0 while it is still live (never removed) -- illegal.
        let ops = vec![DynOp::GrowAndAdd { count: 1 }, DynOp::ReAdd { removed_idx: 0 }];
        validate_dyn_ops_legal(&ops, 10);
    }

    #[test]
    #[should_panic(expected = "GrowAndAdd would exceed capacity")]
    fn validate_dyn_ops_legal_catches_a_hand_crafted_capacity_overrun() {
        let ops = vec![DynOp::GrowAndAdd { count: 5 }, DynOp::GrowAndAdd { count: 5 }];
        validate_dyn_ops_legal(&ops, 8); // 5 + 5 = 10 > 8
    }

    // ------------------------------------------------------------------
    // Report meta capture (Task 5b) -- smoke test on THIS machine: every
    // best-effort helper must return non-empty ("unknown" is a valid
    // non-empty fallback), and `is_wsl` must agree with an independent
    // re-read of `/proc/version` performed right here (not just "trust the
    // function's own internal logic").
    // ------------------------------------------------------------------

    #[test]
    fn meta_capture_helpers_return_non_empty_and_wsl_matches_proc_version() {
        assert!(!cpu_model().is_empty(), "cpu_model() must never return an empty string");
        assert!(!kernel_version().is_empty(), "kernel_version() must never return an empty string");
        assert!(!cxx_compiler_version().is_empty(), "cxx_compiler_version() must never return an empty string");
        assert!(!git_sha().is_empty(), "git_sha() must never return an empty string");

        let expected_wsl =
            std::fs::read_to_string("/proc/version").map(|s| s.to_lowercase().contains("microsoft")).unwrap_or(false);
        assert_eq!(is_wsl(), expected_wsl, "is_wsl() must agree with an independent /proc/version check");
    }
}
