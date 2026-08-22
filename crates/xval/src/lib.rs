//! Cross-validation suite: same data, same queries, Rust `nanoflann-rs`
//! `KdTree` vs the in-process C++ nanoflann 1.12.1 oracle (`nanoflann-ref`),
//! compared BIT-EXACTLY (`ties = false`, positional, is the default
//! everywhere; `ties = true` relaxes ONLY index order within EXACT
//! (bit-equal) tie groups -- see `assert_knn_equal_f64`'s doc comment). This
//! crate is the judge, not a library: deterministic data generators, ULP-tier
//! comparators, and thin build helpers so `tests/*.rs` read cleanly. See
//! `tests/xval_knn.rs`, `tests/xval_radius_box.rs`, `tests/xval_build.rs`.
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
    BuildThreads as RustBuildThreads, DynDim, FlatSlice, Interval, KdTree, KdTreeBuilder,
    ResultItem, SearchParams, L1, L2, L2Simple, SO2, SO3,
};
use rand::Rng;
use rand::SeedableRng;
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
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
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
                        t.radius_search(q, radius, &mut out, &params);
                    }
                    $Name::L2(t) => {
                        t.radius_search(q, radius, &mut out, &params);
                    }
                    $Name::L2Simple(t) => {
                        t.radius_search(q, radius, &mut out, &params);
                    }
                    $Name::SO2(t) => {
                        t.radius_search(q, radius, &mut out, &params);
                    }
                    $Name::SO3(t) => {
                        t.radius_search(q, radius, &mut out, &params);
                    }
                }
                out.into_iter().map(|ri| (ri.index, ri.distance)).collect()
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
// ============================================================================

macro_rules! impl_brute_force_knn {
    ($name:ident, $t:ty) => {
        /// Linear-scan k-NN ground truth under squared L2 (matching this
        /// crate's convention everywhere else: L2-family distances are
        /// SQUARED, never square-rooted). Tie-broken by ASCENDING INDEX --
        /// a total, deterministic order that does not depend on any tree's
        /// build/traversal order, chosen so the ground truth itself is
        /// reproducible independent of which implementation (or which
        /// insertion-order tie rule) produced a candidate result being
        /// scored against it. Returns up to `k` `(index, squared_distance)`
        /// pairs (fewer iff `data` holds fewer than `k` points), sorted
        /// ascending primarily by distance, secondarily by index.
        ///
        /// Panics if `dim == 0`, `query.len() != dim`, or `data.len()` is
        /// not a multiple of `dim`.
        pub fn $name(data: &[$t], dim: usize, query: &[$t], k: usize) -> Vec<(u32, $t)> {
            assert!(dim > 0, "{}: dim must be > 0", stringify!($name));
            assert_eq!(query.len(), dim, "{}: query.len() must equal dim", stringify!($name));
            assert_eq!(data.len() % dim, 0, "{}: data.len() must be a multiple of dim", stringify!($name));
            let n = data.len() / dim;
            let mut all: Vec<(u32, $t)> = (0..n)
                .map(|i| {
                    let pt = &data[i * dim..(i + 1) * dim];
                    let d: $t = pt.iter().zip(query.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    (i as u32, d)
                })
                .collect();
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
pub fn with_ctx<F: FnOnce() + std::panic::UnwindSafe>(ctx: impl std::fmt::Display, f: F) {
    if let Err(e) = std::panic::catch_unwind(f) {
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
}
