//! Cross-validation: radius search / box search parity between the Rust
//! `KdTree` and the C++ nanoflann 1.12.1 oracle. Bit-exact positional
//! (`ties = false`) throughout, except radius `sorted=true` on
//! duplicate/all-identical data, which uses `ties = true` (exact-tie-group
//! mode -- see `xval::assert_knn_equal_f64`'s doc comment) to absorb C++'s
//! own `std::sort`-tie-order latitude WITHOUT relaxing distance bit-equality
//! at all. See the task-11 brief for the exact config matrix.

use nanoflann_ref::{RefIndexF32, RefIndexF64};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use xval::{
    all_identical, build_rust_f32, build_rust_f64, cfg_seed, queries, to_f32, uniform, with_ctx,
    with_duplicates, BuildThreads, XMetric,
};

// ---------------------------------------------------------------------
// Shared config plumbing
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Uniform,
    WithDuplicates,
}

impl DatasetKind {
    fn generate(self, seed: u64, n: usize, dim: usize) -> Vec<f64> {
        match self {
            DatasetKind::Uniform => uniform(seed, n, dim),
            DatasetKind::WithDuplicates => with_duplicates(seed, n, dim, 0.3),
        }
    }

    fn tag(self) -> usize {
        match self {
            DatasetKind::Uniform => 0,
            DatasetKind::WithDuplicates => 1,
        }
    }
}

/// Approximate SQUARED-L2 radii tuned (via the n-ball volume formula, n=300
/// points uniform over `[-10,10)^dim`) to cover roughly 10 ("selective") or
/// roughly 100 ("broad") points on the `Uniform` dataset. Not asserted
/// exactly -- these are guidance for exercising both a small and a large
/// result set, not a correctness requirement (the bit-exact cross-validation
/// holds regardless of how many points actually fall inside).
fn radii_for_dim(dim: usize) -> (f64, f64) {
    match dim {
        2 => (4.5, 45.0),
        3 => (16.0, 75.0),
        8 => (125.0, 220.0),
        _ => unreachable!("radii_for_dim: unconfigured dim {dim}"),
    }
}

// ---------------------------------------------------------------------
// Radius sorted=true
// ---------------------------------------------------------------------

macro_rules! radius_sorted_true_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_radius:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let dims = [2usize, 3, 8];
            let datasets = [DatasetKind::Uniform, DatasetKind::WithDuplicates];
            let leaves = [1usize, 10];
            let n = 300usize;
            let n_queries = 60usize;

            for &dim in &dims {
                let (selective, broad) = radii_for_dim(dim);
                for &dsk in &datasets {
                    for &leaf in &leaves {
                        let seed = cfg_seed("radius_sorted_data", &[dim, dsk.tag(), leaf]);
                        let qseed = cfg_seed("radius_sorted_q", &[dim, dsk.tag(), leaf]);
                        let data64 = dsk.generate(seed, n, dim);
                        let q64 = queries(qseed, &data64, dim, n_queries);
                        let data: Vec<$t> = $cast(&data64);
                        let q: Vec<$t> = $cast(&q64);

                        let rust_idx = $build_rust(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
                        let cpp_idx = $RefIndex::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

                        for &radius in &[selective as $t, broad as $t] {
                            for qi in 0..n_queries {
                                let query = &q[qi * dim..(qi + 1) * dim];
                                let r = rust_idx.radius(query, radius, true, 0.0);
                                let c = cpp_idx.radius(query, radius, true, 0.0);
                                let ctx = format!(
                                    "radius_sorted_true config: dim={dim} dataset={:?} leaf={leaf} radius={radius:?} qi={qi} data_seed={seed} query_seed={qseed}",
                                    dsk
                                );
                                // ties=true (NOT false) here, deliberately: C++'s
                                // RadiusResultSet::sort() (nanoflann.hpp:463)
                                // is a plain `std::sort` over (index,dist)
                                // pairs -- UNSTABLE, so exact-distance ties get
                                // an implementation-defined order, unlike KNN's
                                // insertion-order-sorted incremental insert.
                                // nanoflann-rs's own RadiusResultSet::sort
                                // (result_set.rs) documents this exact latitude
                                // and picks a stable sort as "one deterministic
                                // choice within it" -- so positional comparison
                                // of TIED entries here would fail on a
                                // documented, intentional, non-bug divergence.
                                // ties=true activates assert_radius_equal's
                                // exact-tie-group mode: index order is relaxed
                                // ONLY within a maximal run of BIT-IDENTICAL
                                // distances (ulp_diff==0), and even then every
                                // rank's distance is still required bit-equal
                                // cross-side -- this is the mechanism the
                                // comparator was designed for, not a tolerance
                                // weakening (verified: on `WithDuplicates`/
                                // `AllIdentical` data the only observed
                                // mismatches were index-order swaps between
                                // BIT-IDENTICAL distances, never a genuine
                                // distance divergence).
                                with_ctx(ctx, || {
                                    xval::$assert_radius(&r, &c, true, true);
                                });
                            }
                        }
                    }
                }
            }
        }
    };
}

radius_sorted_true_test!(
    radius_sorted_true_f64,
    f64,
    build_rust_f64,
    RefIndexF64,
    assert_radius_equal_f64,
    |v: &Vec<f64>| v.clone()
);
radius_sorted_true_test!(
    radius_sorted_true_f32,
    f32,
    build_rust_f32,
    RefIndexF32,
    assert_radius_equal_f32,
    |v: &Vec<f64>| to_f32(v)
);

// ---------------------------------------------------------------------
// Radius sorted=false: multiset AND exact-sequence (traversal order).
// ---------------------------------------------------------------------

macro_rules! radius_sorted_false_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_radius:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let dims = [2usize, 3, 8];
            let datasets = [DatasetKind::Uniform, DatasetKind::WithDuplicates];
            let leaves = [1usize, 10];
            let n = 300usize;
            let n_queries = 60usize;

            for &dim in &dims {
                let (selective, broad) = radii_for_dim(dim);
                for &dsk in &datasets {
                    for &leaf in &leaves {
                        let seed = cfg_seed("radius_unsorted_data", &[dim, dsk.tag(), leaf]);
                        let qseed = cfg_seed("radius_unsorted_q", &[dim, dsk.tag(), leaf]);
                        let data64 = dsk.generate(seed, n, dim);
                        let q64 = queries(qseed, &data64, dim, n_queries);
                        let data: Vec<$t> = $cast(&data64);
                        let q: Vec<$t> = $cast(&q64);

                        let rust_idx = $build_rust(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
                        let cpp_idx = $RefIndex::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

                        for &radius in &[selective as $t, broad as $t] {
                            for qi in 0..n_queries {
                                let query = &q[qi * dim..(qi + 1) * dim];
                                let r = rust_idx.radius(query, radius, false, 0.0);
                                let c = cpp_idx.radius(query, radius, false, 0.0);
                                let ctx = format!(
                                    "radius_sorted_false config: dim={dim} dataset={:?} leaf={leaf} radius={radius:?} qi={qi} data_seed={seed} query_seed={qseed}",
                                    dsk
                                );
                                with_ctx(ctx.clone(), || {
                                    xval::$assert_radius(&r, &c, false, false);
                                });
                                // Multiset equality alone doesn't test traversal
                                // ORDER -- assert the raw (unsorted) index
                                // sequences are IDENTICAL too. Per the brief: if
                                // this ever fails while the multiset check
                                // passes, that's a concern to report, not a
                                // tolerance to weaken.
                                with_ctx(ctx, || {
                                    let r_seq: Vec<u32> = r.iter().map(|x| x.0).collect();
                                    let c_seq: Vec<u32> = c.iter().map(|x| x.0).collect();
                                    assert_eq!(
                                        r_seq, c_seq,
                                        "radius sorted=false traversal-order sequence mismatch \
                                         (multiset check passed above -- this is a STRONGER \
                                         check on raw traversal order)"
                                    );
                                });
                            }
                        }
                    }
                }
            }
        }
    };
}

radius_sorted_false_test!(
    radius_sorted_false_f64,
    f64,
    build_rust_f64,
    RefIndexF64,
    assert_radius_equal_f64,
    |v: &Vec<f64>| v.clone()
);
radius_sorted_false_test!(
    radius_sorted_false_f32,
    f32,
    build_rust_f32,
    RefIndexF32,
    assert_radius_equal_f32,
    |v: &Vec<f64>| to_f32(v)
);

// ---------------------------------------------------------------------
// Exact-boundary radius: the 5th-NN distance from the C++ side, passed as
// radius to BOTH sides -- the boundary point must be excluded on BOTH.
// ---------------------------------------------------------------------

#[test]
fn radius_exact_boundary_excludes_on_both_sides_f64() {
    let n = 300usize;
    let dim = 3usize;
    let leaf = 10usize;
    let n_queries = 20usize;
    let seed = cfg_seed("radius_boundary", &[0]);
    let qseed = cfg_seed("radius_boundary_q", &[0]);
    let data = uniform(seed, n, dim);
    let q = queries(qseed, &data, dim, n_queries);

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    for qi in 0..n_queries {
        let query = &q[qi * dim..(qi + 1) * dim];
        let (c5_idx, c5_dist) = cpp_idx.knn(query, 5, 0.0);
        assert_eq!(c5_idx.len(), 5, "expected 5 neighbors to exist (n=300 >> 5)");
        let boundary_idx = c5_idx[4];
        let boundary_radius = c5_dist[4];

        let r = rust_idx.radius(query, boundary_radius, true, 0.0);
        let c = cpp_idx.radius(query, boundary_radius, true, 0.0);

        let ctx = format!(
            "radius_exact_boundary_excludes_on_both_sides_f64: qi={qi} boundary_idx={boundary_idx} \
             boundary_radius={boundary_radius} data_seed={seed} query_seed={qseed}"
        );
        with_ctx(ctx.clone(), || {
            assert!(
                !r.iter().any(|(i, _)| *i == boundary_idx),
                "rust radius_search must EXCLUDE the exact-boundary point (strict <)"
            );
            assert!(
                !c.iter().any(|(i, _)| *i == boundary_idx),
                "cpp radius_search must EXCLUDE the exact-boundary point (strict <)"
            );
        });
        with_ctx(ctx, || {
            xval::assert_radius_equal_f64(&r, &c, true, false);
        });
    }
}

#[test]
fn radius_exact_boundary_excludes_on_both_sides_f32() {
    let n = 300usize;
    let dim = 3usize;
    let leaf = 10usize;
    let n_queries = 20usize;
    let seed = cfg_seed("radius_boundary", &[1]);
    let qseed = cfg_seed("radius_boundary_q", &[1]);
    let data64 = uniform(seed, n, dim);
    let q64 = queries(qseed, &data64, dim, n_queries);
    let data = to_f32(&data64);
    let q = to_f32(&q64);

    let rust_idx = build_rust_f32(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF32::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    for qi in 0..n_queries {
        let query = &q[qi * dim..(qi + 1) * dim];
        let (c5_idx, c5_dist) = cpp_idx.knn(query, 5, 0.0);
        assert_eq!(c5_idx.len(), 5, "expected 5 neighbors to exist (n=300 >> 5)");
        let boundary_idx = c5_idx[4];
        let boundary_radius = c5_dist[4];

        let r = rust_idx.radius(query, boundary_radius, true, 0.0);
        let c = cpp_idx.radius(query, boundary_radius, true, 0.0);

        let ctx = format!(
            "radius_exact_boundary_excludes_on_both_sides_f32: qi={qi} boundary_idx={boundary_idx} \
             boundary_radius={boundary_radius} data_seed={seed} query_seed={qseed}"
        );
        with_ctx(ctx.clone(), || {
            assert!(
                !r.iter().any(|(i, _)| *i == boundary_idx),
                "rust radius_search must EXCLUDE the exact-boundary point (strict <)"
            );
            assert!(
                !c.iter().any(|(i, _)| *i == boundary_idx),
                "cpp radius_search must EXCLUDE the exact-boundary point (strict <)"
            );
        });
        with_ctx(ctx, || {
            xval::assert_radius_equal_f32(&r, &c, true, false);
        });
    }
}

// ---------------------------------------------------------------------
// Box search
// ---------------------------------------------------------------------

/// 5x5x5 integer grid, dim 3, row-major over (x, y, z) each in `0..5`. Index
/// of point `(x,y,z)` is `x*25 + y*5 + z`.
fn integer_grid_5x5x5() -> Vec<f64> {
    let mut out = Vec::with_capacity(125 * 3);
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                out.push(x as f64);
                out.push(y as f64);
                out.push(z as f64);
            }
        }
    }
    out
}

#[test]
fn box_integer_grid_exact_sequences_match_f64() {
    let data = integer_grid_5x5x5();
    let leaf = 4usize;
    let rust_idx = build_rust_f64(&data, 3, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, 3, XMetric::L2.to_ref(), leaf, 1);

    // Box faces exactly on point coordinates: the 3x3x3 subgrid [1,3]^3.
    let lo = [1.0, 1.0, 1.0];
    let hi = [3.0, 3.0, 3.0];
    let r = rust_idx.find_within_box(&lo, &hi);
    let c = cpp_idx.find_within_box(&lo, &hi);
    assert_eq!(r, c, "box EXACT SEQUENCE (traversal order) must match on the integer grid");
    assert_eq!(r.len(), 27, "expected the full 3x3x3=27-point subgrid");

    // Degenerate box: lo == hi exactly on a grid point -> that point present
    // on both sides.
    let point_idx = 2 * 25 + 2 * 5 + 2; // (2,2,2)
    let lo2 = [2.0, 2.0, 2.0];
    let hi2 = [2.0, 2.0, 2.0];
    let r2 = rust_idx.find_within_box(&lo2, &hi2);
    let c2 = cpp_idx.find_within_box(&lo2, &hi2);
    assert_eq!(r2, c2, "degenerate box exact sequence must match");
    assert_eq!(r2, vec![point_idx as u32], "degenerate box must return exactly the one point");
}

#[test]
fn box_uniform_exact_sequences_match_f64() {
    let n = 300usize;
    let dim = 3usize;
    let leaf = 10usize;
    let n_queries = 30usize;
    let seed = cfg_seed("box_uniform", &[0]);
    let data = uniform(seed, n, dim);

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    // Random axis-aligned boxes, centered on random dataset points (so at
    // least the center point is always inside), half-width chosen to cover
    // a modest fraction of the domain.
    let mut rng = ChaCha8Rng::seed_from_u64(cfg_seed("box_uniform_boxes", &[0]));
    for bi in 0..n_queries {
        let center_idx = rng.gen_range(0..n);
        let half = rng.gen_range(1.0f64..4.0);
        let mut lo = [0.0f64; 3];
        let mut hi = [0.0f64; 3];
        for d in 0..3 {
            let c = data[center_idx * dim + d];
            lo[d] = c - half;
            hi[d] = c + half;
        }
        let r = rust_idx.find_within_box(&lo, &hi);
        let c = cpp_idx.find_within_box(&lo, &hi);
        let ctx = format!(
            "box_uniform_exact_sequences_match_f64: bi={bi} center_idx={center_idx} half={half} data_seed={seed}"
        );
        with_ctx(ctx, || {
            assert_eq!(r, c, "box exact sequence (traversal order) must match");
            assert!(
                r.contains(&(center_idx as u32)),
                "the box's own center point must be inside its box"
            );
        });
    }
}

// ---------------------------------------------------------------------
// Empty tree: box / radius -> 0 on both sides.
// ---------------------------------------------------------------------

#[test]
fn empty_tree_radius_and_box_return_zero_both_sides_f64() {
    let empty: [f64; 0] = [];
    let rust_idx = build_rust_f64(&empty, 3, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&empty, 3, XMetric::L2.to_ref(), 10, 1);

    let r = rust_idx.radius(&[0.0, 0.0, 0.0], 100.0, true, 0.0);
    let c = cpp_idx.radius(&[0.0, 0.0, 0.0], 100.0, true, 0.0);
    assert_eq!(r.len(), 0);
    assert_eq!(c.len(), 0);

    let rb = rust_idx.find_within_box(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]);
    let cb = cpp_idx.find_within_box(&[-1.0, -1.0, -1.0], &[1.0, 1.0, 1.0]);
    assert_eq!(rb.len(), 0);
    assert_eq!(cb.len(), 0);
}

// ---------------------------------------------------------------------
// Metric breadth: L1, SO2-at-dim-3, all_identical.
// ---------------------------------------------------------------------

#[test]
fn radius_metric_breadth_l1_f64() {
    let n = 300usize;
    let dim = 3usize;
    let leaf = 10usize;
    let n_queries = 40usize;
    let seed = cfg_seed("radius_l1", &[0]);
    let qseed = cfg_seed("radius_l1_q", &[0]);
    let data = uniform(seed, n, dim);
    let q = queries(qseed, &data, dim, n_queries);

    // L1 is unsquared sum-of-abs-diffs, so its natural scale differs from
    // L2's squared scale -- pick a radius on that scale (max possible L1
    // dist in-domain is 20*3=60).
    let radius = 8.0f64;

    let rust_idx = build_rust_f64(&data, dim, XMetric::L1, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L1.to_ref(), leaf, 1);

    for qi in 0..n_queries {
        let query = &q[qi * dim..(qi + 1) * dim];
        let r = rust_idx.radius(query, radius, true, 0.0);
        let c = cpp_idx.radius(query, radius, true, 0.0);
        let ctx = format!("radius_metric_breadth_l1_f64: qi={qi} data_seed={seed} query_seed={qseed}");
        with_ctx(ctx, || {
            xval::assert_radius_equal_f64(&r, &c, true, false);
        });
    }
}

/// on_circle_so2 data extended with a junk MIDDLE axis: `[junk0, junk_mid,
/// angle]`. SO2's `eval` reads only the LAST axis (`angle`), but
/// `accum_dist` -- used for every axis's per-node bound math -- wraps EVERY
/// axis, including the junk ones. This is exactly where a Rust/C++
/// divergence in the per-axis bound computation would hide (an eval-only
/// test would never touch it).
fn on_circle_so2_dim3(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        let junk0 = rng.gen_range(-10.0f64..10.0);
        let junk_mid = rng.gen_range(-10.0f64..10.0);
        let angle = rng.gen_range(-core::f64::consts::PI..core::f64::consts::PI);
        out.push(junk0);
        out.push(junk_mid);
        out.push(angle);
    }
    out
}

#[test]
fn radius_metric_breadth_so2_dim3_f64() {
    let n = 300usize;
    let dim = 3usize;
    let leaf = 10usize;
    let n_queries = 40usize;
    let seed = cfg_seed("radius_so2_dim3", &[0]);
    let qseed = cfg_seed("radius_so2_dim3_q", &[0]);
    let data = on_circle_so2_dim3(seed, n);
    let q = on_circle_so2_dim3(qseed, n_queries);

    // SO2's eval is unsquared angular distance in [0, pi]; pick a mid-range
    // radius.
    let radius = 1.5f64;

    let rust_idx = build_rust_f64(&data, dim, XMetric::SO2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::SO2.to_ref(), leaf, 1);

    for qi in 0..n_queries {
        let query = &q[qi * dim..(qi + 1) * dim];
        let r = rust_idx.radius(query, radius, true, 0.0);
        let c = cpp_idx.radius(query, radius, true, 0.0);
        let ctx = format!(
            "radius_metric_breadth_so2_dim3_f64: qi={qi} data_seed={seed} query_seed={qseed}"
        );
        with_ctx(ctx, || {
            xval::assert_radius_equal_f64(&r, &c, true, false);
        });
    }
}

#[test]
fn radius_metric_breadth_all_identical_f64() {
    let n = 200usize;
    let dim = 3usize;
    let leaf = 10usize;
    let data = all_identical(n, dim);
    // Every point is at squared distance 0 from every other -- any positive
    // radius includes everything; a zero radius includes nothing (strict <).
    let queries = [
        ([1.25, 1.25, 1.25], 1.0f64),
        ([1.25, 1.25, 1.25], 0.0f64),
    ];

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    for (qi, (query, radius)) in queries.iter().enumerate() {
        let r = rust_idx.radius(query, *radius, true, 0.0);
        let c = cpp_idx.radius(query, *radius, true, 0.0);
        let ctx = format!("radius_metric_breadth_all_identical_f64: qi={qi} radius={radius}");
        // ties=true, not false -- same reasoning as radius_sorted_true_test!:
        // C++'s RadiusResultSet::sort() is unstable std::sort, so for
        // all-identical data (EVERY distance bit-identical at 0.0, i.e. one
        // giant exact-tie group) tie ORDER is unspecified on the C++ side;
        // exact-tie-group comparison relaxes ONLY that index order, while
        // still requiring every rank's distance bit-equal cross-side.
        with_ctx(ctx, || {
            xval::assert_radius_equal_f64(&r, &c, true, true);
        });
    }
    // Sanity: the positive-radius query really does hit everything, so this
    // test isn't vacuously comparing two empty lists.
    let (all_r, _) = &(rust_idx.radius(&queries[0].0, queries[0].1, true, 0.0), ());
    assert_eq!(all_r.len(), n, "expected the positive-radius query to include every point");
}

// ---------------------------------------------------------------------
// Suite self-test (mutation sanity), grouped-mode canary: the exact-tie-group
// comparator's `ties=true` path must still catch a genuine cross-side
// DISTANCE divergence, not just an index-order mismatch (this is what
// `mutation_canary_tie_rule` in xval_knn.rs does NOT cover -- that canary
// exercises a real tie-ORDER divergence via SmallestIndexWins, not a
// distance divergence). Perturbs one C++ result's distance by exactly 1 ULP
// before comparison and asserts `assert_radius_equal_f64(..., ties=true)`
// FAILS. Proves the fix from code review round 1 (grouped mode used to
// perform NO cross-side distance comparison at all) actually holds.
// ---------------------------------------------------------------------

#[test]
#[ignore]
fn mutation_canary_distance_perturbation() {
    let n = 200usize;
    let dim = 3usize;
    let leaf = 10usize;
    let data = all_identical(n, dim);
    let query = [1.25f64, 1.25, 1.25];
    let radius = 1.0f64; // positive radius over all-identical data -> every point is a bit-identical (0.0) exact tie.

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    let r = rust_idx.radius(&query, radius, true, 0.0);
    let mut c = cpp_idx.radius(&query, radius, true, 0.0);
    assert_eq!(r.len(), n, "expected every point to be included (sanity, not the canary itself)");
    assert_eq!(c.len(), n, "expected every point to be included (sanity, not the canary itself)");

    // Sanity: BEFORE perturbation, this must PASS (it's real cross-validated
    // data) -- otherwise the canary below would be meaningless (it could
    // "fail" for an unrelated reason).
    xval::assert_radius_equal_f64(&r, &c, true, true);

    // Perturb exactly one C++ distance by 1 ULP -- still well within any
    // "these are basically the same tie" intuition, but no longer BIT-EQUAL,
    // which is what the fixed comparator requires unconditionally at every
    // rank even inside an exact-tie group.
    let bits = c[0].1.to_bits();
    c[0].1 = f64::from_bits(bits + 1);

    let result = std::panic::catch_unwind(|| {
        xval::assert_radius_equal_f64(&r, &c, true, true);
    });

    assert!(
        result.is_err(),
        "mutation_canary_distance_perturbation: expected assert_radius_equal_f64(..., ties=true) \
         to FAIL after perturbing one cpp distance by 1 ULP -- if this never fails, the \
         exact-tie-group comparator is not actually comparing distances cross-side (the exact \
         review-round-1 bug)"
    );
}
