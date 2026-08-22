//! Cross-validation: knn / rknn parity between the Rust `KdTree` and the C++
//! nanoflann 1.12.1 oracle. Every assertion is bit-exact (`max_ulps = 0`)
//! unless a section says otherwise. See the task-11 brief for the exact
//! config matrix this file implements.

use nanoflann_rs::ConstDim;
use nanoflann_ref::{RefIndexF32, RefIndexF64};
use xval::{
    all_identical, build_rust_f32, build_rust_f64, cfg_seed, clustered, on_circle_so2, queries,
    to_f32, uniform, with_ctx, with_duplicates, BuildThreads, XMetric,
};

// ---------------------------------------------------------------------
// Shared config plumbing
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Uniform,
    Clustered,
    WithDuplicates,
    AllIdentical,
}

const DATASET_KINDS: [DatasetKind; 4] = [
    DatasetKind::Uniform,
    DatasetKind::Clustered,
    DatasetKind::WithDuplicates,
    DatasetKind::AllIdentical,
];

impl DatasetKind {
    fn generate(self, seed: u64, n: usize, dim: usize) -> Vec<f64> {
        match self {
            DatasetKind::Uniform => uniform(seed, n, dim),
            DatasetKind::Clustered => clustered(seed, n, dim, 5),
            DatasetKind::WithDuplicates => with_duplicates(seed, n, dim, 0.3),
            DatasetKind::AllIdentical => all_identical(n, dim),
        }
    }
}

const MAIN_METRICS: [XMetric; 4] = [XMetric::L1, XMetric::L2, XMetric::L2Simple, XMetric::SO3];

fn metric_tag(m: XMetric) -> usize {
    match m {
        XMetric::L1 => 0,
        XMetric::L2 => 1,
        XMetric::L2Simple => 2,
        XMetric::SO2 => 3,
        XMetric::SO3 => 4,
    }
}

fn dataset_tag(d: DatasetKind) -> usize {
    match d {
        DatasetKind::Uniform => 0,
        DatasetKind::Clustered => 1,
        DatasetKind::WithDuplicates => 2,
        DatasetKind::AllIdentical => 3,
    }
}

// ---------------------------------------------------------------------
// Main matrix: dims x metrics x datasets x leaf, n=300, 60 queries, k in {1,10}
// ---------------------------------------------------------------------

macro_rules! knn_matrix_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_knn:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let dims = [2usize, 3, 8, 16, 32];
            let leaves = [1usize, 10, 64];
            let ks = [1usize, 10];
            let n = 300usize;
            let n_queries = 60usize;

            for &dim in &dims {
                for &metric in &MAIN_METRICS {
                    for &dsk in &DATASET_KINDS {
                        for &leaf in &leaves {
                            let seed =
                                cfg_seed("knn_matrix_data", &[dim, metric_tag(metric), dataset_tag(dsk), leaf]);
                            let qseed =
                                cfg_seed("knn_matrix_q", &[dim, metric_tag(metric), dataset_tag(dsk), leaf]);
                            let data64 = dsk.generate(seed, n, dim);
                            let q64 = queries(qseed, &data64, dim, n_queries);
                            let data: Vec<$t> = $cast(&data64);
                            let q: Vec<$t> = $cast(&q64);

                            let rust_idx =
                                $build_rust(&data, dim, metric, leaf, BuildThreads::Sequential);
                            let cpp_idx = $RefIndex::build(&data, dim, metric.to_ref(), leaf, 1);

                            for &k in &ks {
                                for qi in 0..n_queries {
                                    let query = &q[qi * dim..(qi + 1) * dim];
                                    let (r_idx, r_dist) = rust_idx.knn(query, k, 0.0);
                                    let (c_idx, c_dist) = cpp_idx.knn(query, k, 0.0);
                                    let ctx = format!(
                                        "knn_matrix config: dim={dim} metric={:?} dataset={:?} leaf={leaf} k={k} qi={qi} data_seed={seed} query_seed={qseed}",
                                        metric, dsk
                                    );
                                    with_ctx(ctx, || {
                                        xval::$assert_knn((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

knn_matrix_test!(knn_matrix_f64, f64, build_rust_f64, RefIndexF64, assert_knn_equal_f64, |v: &Vec<f64>| v.clone());
knn_matrix_test!(knn_matrix_f32, f32, build_rust_f32, RefIndexF32, assert_knn_equal_f32, |v: &Vec<f64>| to_f32(v));

// ---------------------------------------------------------------------
// k > n
// ---------------------------------------------------------------------

#[test]
fn knn_k_greater_than_n_f64() {
    let n = 50usize;
    let k = 101usize;
    let dim = 3usize;
    let seed = cfg_seed("knn_k_gt_n", &[0]);
    let data = uniform(seed, n, dim);
    let q = queries(seed + 1, &data, dim, 20);

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), 10, 1);

    for qi in 0..20 {
        let query = &q[qi * dim..(qi + 1) * dim];
        let (r_idx, r_dist) = rust_idx.knn(query, k, 0.0);
        let (c_idx, c_dist) = cpp_idx.knn(query, k, 0.0);
        let ctx = format!("knn_k_greater_than_n_f64: qi={qi} seed={seed}");
        with_ctx(ctx.clone(), || {
            assert_eq!(r_idx.len(), c_idx.len(), "found-count mismatch (k>n case)");
            assert_eq!(r_idx.len(), n, "expected exactly n={n} results when k>n");
        });
        with_ctx(ctx, || {
            xval::assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
        });
    }
}

#[test]
fn knn_k_greater_than_n_f32() {
    let n = 50usize;
    let k = 101usize;
    let dim = 3usize;
    let seed = cfg_seed("knn_k_gt_n", &[1]);
    let data64 = uniform(seed, n, dim);
    let q64 = queries(seed + 1, &data64, dim, 20);
    let data = to_f32(&data64);
    let q = to_f32(&q64);

    let rust_idx = build_rust_f32(&data, dim, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_idx = RefIndexF32::build(&data, dim, XMetric::L2.to_ref(), 10, 1);

    for qi in 0..20 {
        let query = &q[qi * dim..(qi + 1) * dim];
        let (r_idx, r_dist) = rust_idx.knn(query, k, 0.0);
        let (c_idx, c_dist) = cpp_idx.knn(query, k, 0.0);
        let ctx = format!("knn_k_greater_than_n_f32: qi={qi} seed={seed}");
        with_ctx(ctx.clone(), || {
            assert_eq!(r_idx.len(), c_idx.len(), "found-count mismatch (k>n case)");
            assert_eq!(r_idx.len(), n, "expected exactly n={n} results when k>n");
        });
        with_ctx(ctx, || {
            xval::assert_knn_equal_f32((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
        });
    }
}

// ---------------------------------------------------------------------
// eps: both sides get the same eps; approximate search is deterministic
// given identical trees, so results must still match exactly.
// ---------------------------------------------------------------------

macro_rules! knn_eps_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_knn:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let dims = [3usize, 16];
            let epss = [0.0f32, 0.1, 1.0];
            let n = 300usize;
            let leaf = 10usize;
            let k = 10usize;
            let n_queries = 40usize;

            for &dim in &dims {
                let seed = cfg_seed("knn_eps_data", &[dim]);
                let qseed = cfg_seed("knn_eps_q", &[dim]);
                let data64 = uniform(seed, n, dim);
                let q64 = queries(qseed, &data64, dim, n_queries);
                let data: Vec<$t> = $cast(&data64);
                let q: Vec<$t> = $cast(&q64);

                let rust_idx = $build_rust(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
                let cpp_idx = $RefIndex::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

                for &eps in &epss {
                    for qi in 0..n_queries {
                        let query = &q[qi * dim..(qi + 1) * dim];
                        let (r_idx, r_dist) = rust_idx.knn(query, k, eps);
                        let (c_idx, c_dist) = cpp_idx.knn(query, k, eps);
                        let ctx = format!(
                            "knn_eps config: dim={dim} eps={eps} qi={qi} data_seed={seed} query_seed={qseed}"
                        );
                        with_ctx(ctx, || {
                            xval::$assert_knn((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
                        });
                    }
                }
            }
        }
    };
}

knn_eps_test!(knn_eps_f64, f64, build_rust_f64, RefIndexF64, assert_knn_equal_f64, |v: &Vec<f64>| v.clone());
knn_eps_test!(knn_eps_f32, f32, build_rust_f32, RefIndexF32, assert_knn_equal_f32, |v: &Vec<f64>| to_f32(v));

// ---------------------------------------------------------------------
// SO2: wrap-around parity lock.
// ---------------------------------------------------------------------

macro_rules! knn_so2_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_knn:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let leaves = [1usize, 10];
            let ks = [1usize, 10];
            let n = 300usize;
            let n_queries = 60usize;
            let dim = 2usize;

            let seed = cfg_seed("knn_so2_data", &[0]);
            let qseed = cfg_seed("knn_so2_q", &[0]);
            let data64 = on_circle_so2(seed, n);
            let q64 = on_circle_so2(qseed, n_queries);
            let data: Vec<$t> = $cast(&data64);
            let q: Vec<$t> = $cast(&q64);

            for &leaf in &leaves {
                let rust_idx = $build_rust(&data, dim, XMetric::SO2, leaf, BuildThreads::Sequential);
                let cpp_idx = $RefIndex::build(&data, dim, XMetric::SO2.to_ref(), leaf, 1);

                for &k in &ks {
                    for qi in 0..n_queries {
                        let query = &q[qi * dim..(qi + 1) * dim];
                        let (r_idx, r_dist) = rust_idx.knn(query, k, 0.0);
                        let (c_idx, c_dist) = cpp_idx.knn(query, k, 0.0);
                        let ctx = format!(
                            "knn_so2 config: leaf={leaf} k={k} qi={qi} data_seed={seed} query_seed={qseed}"
                        );
                        with_ctx(ctx, || {
                            xval::$assert_knn((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
                        });
                    }
                }
            }
        }
    };
}

knn_so2_test!(knn_so2_f64, f64, build_rust_f64, RefIndexF64, assert_knn_equal_f64, |v: &Vec<f64>| v.clone());
knn_so2_test!(knn_so2_f32, f32, build_rust_f32, RefIndexF32, assert_knn_equal_f32, |v: &Vec<f64>| to_f32(v));

// ---------------------------------------------------------------------
// rknn: tiny radius (covers < k) and huge radius (covers all), per query.
// ---------------------------------------------------------------------

macro_rules! rknn_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $assert_knn:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let n = 300usize;
            let dim = 3usize;
            let leaf = 10usize;
            let k = 10usize;
            let n_queries = 30usize;
            // Squared-L2 radii: tiny (raw dist 0.1) covers far fewer than k
            // points on data spread over [-10,10)^3; huge covers all of it
            // (max possible squared dist in-domain is 20^2*3=1200).
            let tiny_radius: $t = 0.01 as $t;
            let huge_radius: $t = 100_000.0 as $t;

            let seed = cfg_seed("rknn_data", &[0]);
            let qseed = cfg_seed("rknn_q", &[0]);
            let data64 = uniform(seed, n, dim);
            let q64 = queries(qseed, &data64, dim, n_queries);
            let data: Vec<$t> = $cast(&data64);
            let q: Vec<$t> = $cast(&q64);

            let rust_idx = $build_rust(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
            let cpp_idx = $RefIndex::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

            for &radius in &[tiny_radius, huge_radius] {
                for qi in 0..n_queries {
                    let query = &q[qi * dim..(qi + 1) * dim];
                    let (r_idx, r_dist) = rust_idx.rknn(query, k, radius, 0.0);
                    let (c_idx, c_dist) = cpp_idx.rknn(query, k, radius, 0.0);
                    let ctx = format!(
                        "rknn config: radius={radius:?} qi={qi} data_seed={seed} query_seed={qseed}"
                    );
                    with_ctx(ctx.clone(), || {
                        assert_eq!(r_idx.len(), c_idx.len(), "rknn found-count mismatch");
                    });
                    with_ctx(ctx, || {
                        xval::$assert_knn((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
                    });
                }
            }
        }
    };
}

rknn_test!(rknn_f64, f64, build_rust_f64, RefIndexF64, assert_knn_equal_f64, |v: &Vec<f64>| v.clone());
rknn_test!(rknn_f32, f32, build_rust_f32, RefIndexF32, assert_knn_equal_f32, |v: &Vec<f64>| to_f32(v));

// ---------------------------------------------------------------------
// ConstDim spot check: direct KdTreeBuilder (not the DynDim helper) vs
// C++ dim-3 runtime index.
// ---------------------------------------------------------------------

#[test]
fn const_dim_3_spot_check_f64() {
    let n = 300usize;
    let n_queries = 60usize;
    let k = 10usize;
    let seed = cfg_seed("const_dim3", &[0]);
    let qseed = cfg_seed("const_dim3_q", &[0]);
    let data = uniform(seed, n, 3);
    let q = queries(qseed, &data, 3, n_queries);

    let pts: Vec<[f64; 3]> = data.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let rust_tree = nanoflann_rs::KdTreeBuilder::new(ConstDim::<3>, pts.as_slice()).build();
    let cpp_idx = RefIndexF64::build(&data, 3, XMetric::L2.to_ref(), 10, 1);

    for qi in 0..n_queries {
        let query: [f64; 3] = [q[qi * 3], q[qi * 3 + 1], q[qi * 3 + 2]];
        let mut r_idx = vec![0u32; k];
        let mut r_dist = vec![0.0f64; k];
        let found = rust_tree.knn_search(&query, &mut r_idx, &mut r_dist);
        r_idx.truncate(found);
        r_dist.truncate(found);
        let (c_idx, c_dist) = cpp_idx.knn(&query, k, 0.0);
        let ctx = format!("const_dim_3_spot_check_f64: qi={qi} data_seed={seed} query_seed={qseed}");
        with_ctx(ctx, || {
            xval::assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
        });
    }
}

// ---------------------------------------------------------------------
// Empty tree
// ---------------------------------------------------------------------

#[test]
fn empty_tree_knn_returns_zero_both_sides_f64() {
    let empty: [f64; 0] = [];
    let rust_idx = build_rust_f64(&empty, 3, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&empty, 3, XMetric::L2.to_ref(), 10, 1);
    assert_eq!(rust_idx.size(), 0);
    assert_eq!(cpp_idx.size(), 0);
    let (r_idx, r_dist) = rust_idx.knn(&[0.0, 0.0, 0.0], 5, 0.0);
    let (c_idx, c_dist) = cpp_idx.knn(&[0.0, 0.0, 0.0], 5, 0.0);
    assert_eq!(r_idx.len(), 0);
    assert_eq!(c_idx.len(), 0);
    xval::assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
}

#[test]
fn empty_tree_knn_returns_zero_both_sides_f32() {
    let empty: [f32; 0] = [];
    let rust_idx = build_rust_f32(&empty, 3, XMetric::L2, 10, BuildThreads::Sequential);
    let cpp_idx = RefIndexF32::build(&empty, 3, XMetric::L2.to_ref(), 10, 1);
    assert_eq!(rust_idx.size(), 0);
    assert_eq!(cpp_idx.size(), 0);
    let (r_idx, r_dist) = rust_idx.knn(&[0.0, 0.0, 0.0], 5, 0.0);
    let (c_idx, c_dist) = cpp_idx.knn(&[0.0, 0.0, 0.0], 5, 0.0);
    assert_eq!(r_idx.len(), 0);
    assert_eq!(c_idx.len(), 0);
    xval::assert_knn_equal_f32((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
}

// ---------------------------------------------------------------------
// Suite self-test (mutation sanity): the Rust side deliberately uses
// SmallestIndexWins (the NON-default tie rule) on duplicate-heavy data,
// against the C++ oracle's default (KeepInsertionOrder-equivalent) tie
// behavior -- assert_knn_equal must FAIL. Proves the suite actually
// constrains tie behavior, rather than never exercising it.
// ---------------------------------------------------------------------

#[test]
#[ignore]
fn mutation_canary_tie_rule() {
    use nanoflann_rs::data_source::FlatSlice;
    use nanoflann_rs::dim::DynDim;
    use nanoflann_rs::result_set::SmallestIndexWins;

    let dim = 3usize;
    let n = 300usize;
    let leaf = 4usize;
    let k = 10usize;
    let seed = cfg_seed("mutation_canary", &[0]);
    let qseed = cfg_seed("mutation_canary_q", &[0]);
    // Heavily duplicated data maximizes the chance of ties landing within
    // the k-nearest set, which is exactly where tie-break order matters.
    let data = with_duplicates(seed, n, dim, 0.7);
    let q = queries(qseed, &data, dim, 30);

    let ds = FlatSlice::new(&data, dim);
    let rust_tree = nanoflann_rs::KdTreeBuilder::new(DynDim(dim), ds)
        .leaf_max_size(leaf)
        .tie_break::<SmallestIndexWins>()
        .build();
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

    // At least one query must produce a mismatch (SmallestIndexWins
    // diverging from the C++ oracle's KeepInsertionOrder-equivalent
    // default) for this canary to prove anything.
    let mut saw_mismatch = false;
    for qi in 0..30 {
        let query = &q[qi * dim..(qi + 1) * dim];
        let mut r_idx = vec![0u32; k];
        let mut r_dist = vec![0.0f64; k];
        let found = rust_tree.knn_search(query, &mut r_idx, &mut r_dist);
        r_idx.truncate(found);
        r_dist.truncate(found);
        let (c_idx, c_dist) = cpp_idx.knn(query, k, 0.0);

        let result = std::panic::catch_unwind(|| {
            xval::assert_knn_equal_f64((&r_idx, &r_dist), (&c_idx, &c_dist), 0);
        });
        if result.is_err() {
            saw_mismatch = true;
            break;
        }
    }

    assert!(
        saw_mismatch,
        "mutation_canary_tie_rule: expected assert_knn_equal_f64 to FAIL for at least one \
         query when the Rust side uses SmallestIndexWins against the C++ oracle's default tie \
         rule -- if this never fails, the suite is not actually constraining tie order \
         (seed={seed} qseed={qseed})"
    );
}
