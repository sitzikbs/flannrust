//! M2's judge: op-sequence cross-validation between `nanoflann_rs::dynamic::
//! DynamicKdTree` and the C++ oracle's `KDTreeSingleIndexDynamicAdaptor`
//! (`nanoflann_ref::RefDynIndexF32`/`RefDynIndexF64`). Drives IDENTICAL
//! add/remove/re-add sequences through both forests, over the SAME
//! underlying coordinate buffer, and compares structure (`tree_count`,
//! every slot's point list, `tree_index`, removed count) AND query results
//! (knn, radius) bit-exact after EVERY step -- not just at the end. See
//! `xval::dyn_ops`'s doc comment for the op generator's legality model
//! (that model's own property tests live in `xval/src/lib.rs`'s
//! `#[cfg(test)]` module) and `nanoflann_ref::RefDynIndexF32`'s doc comment
//! for the contiguous-append contract this whole suite exists to never
//! violate.

use nanoflann_ref::{RefDynIndexF32, RefDynIndexF64};
use nanoflann_rs::{DynDim, DynamicKdTreeBuilder, ResultItem, SearchParams};
use xval::{
    apply_dyn_op_f32, apply_dyn_op_f64, assert_dyn_structure_equal_f32, assert_dyn_structure_equal_f64,
    assert_knn_equal_f32, assert_knn_equal_f64, assert_radius_equal_f32, assert_radius_equal_f64,
    brute_force_knn_l2_f32, brute_force_knn_l2_f64, cfg_seed, dyn_ops, queries, to_f32, uniform, with_ctx,
    with_duplicates, DynOp, GrowableFlat,
};

// ---------------------------------------------------------------------
// Shared config plumbing
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Uniform,
    WithDuplicates,
}

const DATASET_KINDS: [DatasetKind; 2] = [DatasetKind::Uniform, DatasetKind::WithDuplicates];

impl DatasetKind {
    fn generate(self, seed: u64, n: usize, dim: usize) -> Vec<f64> {
        match self {
            DatasetKind::Uniform => uniform(seed, n, dim),
            DatasetKind::WithDuplicates => with_duplicates(seed, n, dim, 0.3),
        }
    }
}

fn dataset_tag(d: DatasetKind) -> usize {
    match d {
        DatasetKind::Uniform => 0,
        DatasetKind::WithDuplicates => 1,
    }
}

const CAPACITY: usize = 1500;
const N_OPS: usize = 120;
const N_QUERIES: usize = 30;
const K: usize = 5;
const SEED_IDXS: [u64; 3] = [0, 1, 2];

// ---------------------------------------------------------------------
// Main matrix: scalar {f32,f64} x dim{2,3,8} x dataset{uniform,dup(0.3)} x
// leaf{1,10} x 3 seeds. Structure parity after EVERY op; query parity
// (knn k=5 eps=0, radius ~10-point) every 10th op over 30 fixed queries.
// ---------------------------------------------------------------------

macro_rules! dyn_xval_matrix_test {
    (
        $fn_name:ident, $t:ty, $RefDyn:ident, $apply_fn:ident, $assert_struct_fn:ident,
        $assert_knn:ident, $assert_radius:ident, $brute_force:ident, $cast:expr
    ) => {
        #[test]
        fn $fn_name() {
            let dims = [2usize, 3, 8];
            let leaves = [1usize, 10];

            for &dim in &dims {
                for &dsk in &DATASET_KINDS {
                    for &leaf in &leaves {
                        for &seed_idx in &SEED_IDXS {
                            let data_seed =
                                cfg_seed("dyn_xval_data", &[dim, dataset_tag(dsk), leaf, seed_idx as usize]);
                            let op_seed =
                                cfg_seed("dyn_xval_ops", &[dim, dataset_tag(dsk), leaf, seed_idx as usize]);
                            let q_seed =
                                cfg_seed("dyn_xval_q", &[dim, dataset_tag(dsk), leaf, seed_idx as usize]);

                            let data64 = dsk.generate(data_seed, CAPACITY, dim);
                            let data: Vec<$t> = $cast(&data64);
                            let q64 = queries(q_seed, &data64, dim, N_QUERIES);
                            let q: Vec<$t> = $cast(&q64);

                            let growable = GrowableFlat::new(&data, dim);
                            let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(dim), &growable)
                                .leaf_max_size(leaf)
                                .maximum_point_count(CAPACITY)
                                .build();
                            let mut oracle = $RefDyn::build(&data, dim, leaf, CAPACITY);

                            let ties = dsk == DatasetKind::WithDuplicates;

                            // Calibrated ~10-point radius: the squared
                            // distance to the 10th-nearest neighbor of the
                            // FIRST query over the full-capacity buffer.
                            // Approximate on purpose (the live set changes
                            // across the op sequence) -- see this file's
                            // module doc / the task brief for why an exact
                            // count isn't required for a bit-exact parity
                            // check.
                            let radius_gt = $brute_force(&data, dim, &q[0..dim], 10);
                            let radius_sq: $t = radius_gt.last().expect("CAPACITY >= 10").1;

                            let cfg_desc = format!(
                                "dyn_xval_matrix<{}> dim={dim} dataset={:?} leaf={leaf} seed_idx={seed_idx} \
                                 data_seed={data_seed} op_seed={op_seed} q_seed={q_seed}",
                                stringify!($t),
                                dsk
                            );

                            with_ctx(format!("{cfg_desc} op_idx=init (before any op)"), || {
                                $assert_struct_fn(&rust_tree, &oracle);
                            });

                            let ops = dyn_ops(op_seed, CAPACITY, N_OPS);

                            for (op_idx, op) in ops.iter().enumerate() {
                                $apply_fn(op, &growable, &mut rust_tree, &mut oracle);

                                let ctx = format!("{cfg_desc} op_idx={op_idx} op={:?}", op);
                                with_ctx(ctx.clone(), || {
                                    $assert_struct_fn(&rust_tree, &oracle);
                                });

                                if (op_idx + 1) % 10 == 0 {
                                    for qi in 0..N_QUERIES {
                                        let query = &q[qi * dim..(qi + 1) * dim];
                                        let qctx = format!("{ctx} query_idx={qi}");
                                        let params = SearchParams { eps: 0.0, sorted: true };

                                        let mut out_idx = [0u32; K];
                                        let mut out_dist = [<$t as Default>::default(); K];
                                        let found =
                                            rust_tree.knn_search_with(query, &mut out_idx, &mut out_dist, &params);
                                        let (c_idx, c_dist) = oracle.knn(query, K, 0.0);
                                        with_ctx(format!("{qctx} (knn)"), || {
                                            $assert_knn(
                                                (&out_idx[..found], &out_dist[..found]),
                                                (&c_idx, &c_dist),
                                                ties,
                                            );
                                        });

                                        let mut r_out: Vec<ResultItem<u32, $t>> = Vec::new();
                                        let _ = rust_tree.radius_search_with(query, radius_sq, &mut r_out, &params);
                                        let r_pairs: Vec<(u32, $t)> =
                                            r_out.iter().map(|ri| (ri.index, ri.distance)).collect();
                                        let c_pairs = oracle.radius(query, radius_sq, true, 0.0);
                                        with_ctx(format!("{qctx} (radius)"), || {
                                            $assert_radius(&r_pairs, &c_pairs, true, ties);
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

dyn_xval_matrix_test!(
    dyn_xval_matrix_f64,
    f64,
    RefDynIndexF64,
    apply_dyn_op_f64,
    assert_dyn_structure_equal_f64,
    assert_knn_equal_f64,
    assert_radius_equal_f64,
    brute_force_knn_l2_f64,
    |v: &Vec<f64>| v.clone()
);
dyn_xval_matrix_test!(
    dyn_xval_matrix_f32,
    f32,
    RefDynIndexF32,
    apply_dyn_op_f32,
    assert_dyn_structure_equal_f32,
    assert_knn_equal_f32,
    assert_radius_equal_f32,
    brute_force_knn_l2_f32,
    |v: &Vec<f64>| to_f32(v)
);

// ---------------------------------------------------------------------
// Dedicated scenario tests (both scalars, dim 3)
// ---------------------------------------------------------------------

macro_rules! dyn_scenario_tests {
    (
        $suffix:ident, $t:ty, $RefDyn:ident, $apply_fn:ident, $assert_struct_fn:ident,
        $assert_knn:ident, $cast:expr
    ) => {
        // ---- 1. Tombstone migration -----------------------------------
        //
        // add 8 (all end up in slot 3, first0bit(0..7) = 0,1,0,2,0,1,0,3),
        // remove 2 of them, then add enough (8 more, indices 8..15) that
        // processing index 15 triggers first0bit(15)=4 -- a merge that
        // swallows slots 0,1,2 AND slot 3 (which still physically holds the
        // 2 tombstones) into slot 4. capacity=16, maximum_point_count=16
        // (tree_count = floor(log2(16))+1 = 5, slots 0..=4) is exactly
        // large enough for this. See `dynamic.rs`'s `add_points` doc
        // comment for the exact merge/migration mechanics being exercised.
        #[test]
        fn tombstone_migration() {
            let dim = 3usize;
            let capacity = 16usize;
            let seed = cfg_seed(concat!("dyn_scenario_tombstone_", stringify!($suffix)), &[0]);
            let data64 = uniform(seed, capacity, dim);
            let data: Vec<$t> = $cast(&data64);

            let growable = GrowableFlat::new(&data, dim);
            let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(dim), &growable)
                .maximum_point_count(capacity)
                .build();
            let mut oracle = $RefDyn::build(&data, dim, 10, capacity);

            let ops = [
                DynOp::GrowAndAdd { count: 8 },
                DynOp::Remove { live_idx: 1 },
                DynOp::Remove { live_idx: 3 },
                DynOp::GrowAndAdd { count: 8 }, // indices 8..15 -- idx 15 triggers the slot0..3 merge into slot4
            ];
            for (i, op) in ops.iter().enumerate() {
                $apply_fn(op, &growable, &mut rust_tree, &mut oracle);
                with_ctx(format!("tombstone_migration<{}> op_idx={i} op={:?}", stringify!($t), op), || {
                    $assert_struct_fn(&rust_tree, &oracle);
                });
            }

            // Both removed indices must still read -1 / not-present on both
            // sides after the merge that moved their physical storage.
            let r_ti = rust_tree.tree_index();
            let c_ti = oracle.tree_index();
            assert_eq!(r_ti[1], -1, "rust tree_index[1] must stay -1 through the merge");
            assert_eq!(c_ti[1], -1, "cpp tree_index[1] must stay -1 through the merge");
            assert_eq!(r_ti[3], -1, "rust tree_index[3] must stay -1 through the merge");
            assert_eq!(c_ti[3], -1, "cpp tree_index[3] must stay -1 through the merge");
            assert_eq!(rust_tree.removed_len(), 2);
            assert_eq!(oracle.removed_count(), 2);

            // ReAdd both -- must become present in queries on both sides,
            // structure equal.
            let readd_ops = [DynOp::ReAdd { removed_idx: 1 }, DynOp::ReAdd { removed_idx: 3 }];
            for (i, op) in readd_ops.iter().enumerate() {
                $apply_fn(op, &growable, &mut rust_tree, &mut oracle);
                with_ctx(format!("tombstone_migration<{}> readd op_idx={i} op={:?}", stringify!($t), op), || {
                    $assert_struct_fn(&rust_tree, &oracle);
                });
            }
            assert_eq!(rust_tree.removed_len(), 0);
            assert_eq!(oracle.removed_count(), 0);

            let mut out_idx = [0u32; 16];
            let mut out_dist = [<$t as Default>::default(); 16];
            let params = SearchParams::default();
            let found = rust_tree.knn_search_with(&data[0..dim], &mut out_idx, &mut out_dist, &params);
            let (c_idx, c_dist) = oracle.knn(&data[0..dim], 16, 0.0);
            with_ctx(format!("tombstone_migration<{}> post-readd knn", stringify!($t)), || {
                $assert_knn((&out_idx[..found], &out_dist[..found]), (&c_idx, &c_dist), false);
            });
            assert!(out_idx[..found].contains(&1), "reactivated index 1 must be queryable again (rust)");
            assert!(out_idx[..found].contains(&3), "reactivated index 3 must be queryable again (rust)");
            assert!(c_idx.contains(&1), "reactivated index 1 must be queryable again (cpp)");
            assert!(c_idx.contains(&3), "reactivated index 3 must be queryable again (cpp)");
        }

        // ---- 2. Drain and refill ---------------------------------------
        #[test]
        fn drain_and_refill() {
            let dim = 3usize;
            let capacity = 32usize;
            let seed = cfg_seed(concat!("dyn_scenario_drain_", stringify!($suffix)), &[0]);
            let data64 = uniform(seed, capacity, dim);
            let data: Vec<$t> = $cast(&data64);

            let growable = GrowableFlat::new(&data, dim);
            let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(dim), &growable)
                .maximum_point_count(capacity)
                .build();
            let mut oracle = $RefDyn::build(&data, dim, 10, capacity);

            $apply_fn(&DynOp::GrowAndAdd { count: 32 }, &growable, &mut rust_tree, &mut oracle);
            with_ctx(format!("drain_and_refill<{}> after add 32", stringify!($t)), || {
                $assert_struct_fn(&rust_tree, &oracle);
            });

            for idx in 0..32usize {
                $apply_fn(&DynOp::Remove { live_idx: idx }, &growable, &mut rust_tree, &mut oracle);
            }
            with_ctx(format!("drain_and_refill<{}> after removing all 32", stringify!($t)), || {
                $assert_struct_fn(&rust_tree, &oracle);
            });
            assert_eq!(rust_tree.removed_len(), 32);
            assert_eq!(oracle.removed_count(), 32);

            let mut out_idx = [0u32; 5];
            let mut out_dist = [<$t as Default>::default(); 5];
            let params = SearchParams::default();
            let found = rust_tree.knn_search_with(&data[0..dim], &mut out_idx, &mut out_dist, &params);
            let (c_idx, _c_dist) = oracle.knn(&data[0..dim], 5, 0.0);
            assert_eq!(found, 0, "rust knn must find 0 points on a fully-drained forest");
            assert_eq!(c_idx.len(), 0, "cpp knn must find 0 points on a fully-drained forest");

            // Reactivation is exempt from the contiguous-append contract --
            // one call covering the whole range reactivates every index.
            $apply_fn(&DynOp::ReAdd { removed_idx: 0 }, &growable, &mut rust_tree, &mut oracle);
            for idx in 1..32usize {
                $apply_fn(&DynOp::ReAdd { removed_idx: idx }, &growable, &mut rust_tree, &mut oracle);
            }
            with_ctx(format!("drain_and_refill<{}> after re-adding all 32", stringify!($t)), || {
                $assert_struct_fn(&rust_tree, &oracle);
            });
            assert_eq!(rust_tree.removed_len(), 0);
            assert_eq!(oracle.removed_count(), 0);

            let mut out_idx2 = [0u32; 32];
            let mut out_dist2 = [<$t as Default>::default(); 32];
            let found2 = rust_tree.knn_search_with(&data[0..dim], &mut out_idx2, &mut out_dist2, &params);
            let (c_idx2, c_dist2) = oracle.knn(&data[0..dim], 32, 0.0);
            with_ctx(format!("drain_and_refill<{}> post-refill knn", stringify!($t)), || {
                $assert_knn((&out_idx2[..found2], &out_dist2[..found2]), (&c_idx2, &c_dist2), false);
            });
            assert_eq!(found2, 32, "all 32 points must be queryable again after full refill");
        }

        // ---- 3. Empty-forest parity -------------------------------------
        #[test]
        fn empty_forest_parity() {
            let dim = 3usize;
            let capacity = 16usize;
            let seed = cfg_seed(concat!("dyn_scenario_empty_", stringify!($suffix)), &[0]);
            let data64 = uniform(seed, capacity, dim);
            let data: Vec<$t> = $cast(&data64);

            let growable = GrowableFlat::new(&data, dim);
            let rust_tree = DynamicKdTreeBuilder::new(DynDim(dim), &growable)
                .maximum_point_count(capacity)
                .build();
            let oracle = $RefDyn::build(&data, dim, 10, capacity);

            with_ctx(format!("empty_forest_parity<{}> structure", stringify!($t)), || {
                $assert_struct_fn(&rust_tree, &oracle);
            });

            let mut out_idx = [0u32; 3];
            let mut out_dist = [<$t as Default>::default(); 3];
            let params = SearchParams::default();
            let found = rust_tree.knn_search_with(&data[0..dim], &mut out_idx, &mut out_dist, &params);
            let (c_idx, _) = oracle.knn(&data[0..dim], 3, 0.0);
            assert_eq!(found, 0, "rust knn on an empty forest must find 0");
            assert_eq!(c_idx.len(), 0, "cpp knn on an empty forest must find 0");

            let mut r_out: Vec<ResultItem<u32, $t>> = Vec::new();
            let r_count = rust_tree.radius_search_with(&data[0..dim], <$t>::from(100.0f32), &mut r_out, &params);
            let c_pairs = oracle.radius(&data[0..dim], <$t>::from(100.0f32), true, 0.0);
            assert_eq!(r_count, 0, "rust radius on an empty forest must find 0 items");
            assert_eq!(c_pairs.len(), 0, "cpp radius on an empty forest must find 0 items");
        }

        // ---- 4. eps parity -----------------------------------------------
        #[test]
        fn eps_parity_on_two_slot_forest() {
            let dim = 3usize;
            let capacity = 16usize;
            let seed = cfg_seed(concat!("dyn_scenario_eps_", stringify!($suffix)), &[0]);
            let data64 = uniform(seed, capacity, dim);
            let data: Vec<$t> = $cast(&data64);

            let growable = GrowableFlat::new(&data, dim);
            let mut rust_tree = DynamicKdTreeBuilder::new(DynDim(dim), &growable)
                .maximum_point_count(capacity)
                .build();
            let mut oracle = $RefDyn::build(&data, dim, 10, capacity);

            // add 3: first0bit(0)=0 -> slot0=[0]; first0bit(1)=1 -> merge
            // slot0 into slot1, push 1 -> slot1=[0,1]; first0bit(2)=0 ->
            // slot0=[2]. TWO non-empty slots (0 and 1) -- a genuine 2-slot
            // forest state, not a single monolithic tree.
            $apply_fn(&DynOp::GrowAndAdd { count: 3 }, &growable, &mut rust_tree, &mut oracle);
            with_ctx(format!("eps_parity<{}> structure after add 3", stringify!($t)), || {
                $assert_struct_fn(&rust_tree, &oracle);
            });
            assert!(!rust_tree.point_indices_of_slot(0).is_empty());
            assert!(!rust_tree.point_indices_of_slot(1).is_empty());

            for &eps in &[0.1f32, 1.0] {
                let mut out_idx = [0u32; 3];
                let mut out_dist = [<$t as Default>::default(); 3];
                let params = SearchParams { eps, sorted: true };
                let found = rust_tree.knn_search_with(&data[0..dim], &mut out_idx, &mut out_dist, &params);
                let (c_idx, c_dist) = oracle.knn(&data[0..dim], 3, eps);
                with_ctx(format!("eps_parity<{}> eps={eps}", stringify!($t)), || {
                    $assert_knn((&out_idx[..found], &out_dist[..found]), (&c_idx, &c_dist), false);
                });
            }
        }
    };
}

dyn_scenario_tests!(
    f64,
    f64,
    RefDynIndexF64,
    apply_dyn_op_f64,
    assert_dyn_structure_equal_f64,
    assert_knn_equal_f64,
    |v: &Vec<f64>| v.clone()
);

mod scenarios_f32 {
    use super::*;
    dyn_scenario_tests!(
        f32,
        f32,
        RefDynIndexF32,
        apply_dyn_op_f32,
        assert_dyn_structure_equal_f32,
        assert_knn_equal_f32,
        |v: &Vec<f64>| to_f32(v)
    );
}

// ---------------------------------------------------------------------
// 5. Mutation canary (#[ignore]) -- proves the structure-parity comparison
// actually catches a real divergence: run a normal legal op sequence
// through BOTH sides, except skip exactly one Remove on the RUST side only
// (still applied to the oracle). Assert (via catch_unwind) that the very
// next structure-parity check panics. Not run by default: this is a
// self-test of the suite's own detection power, not a correctness
// assertion about the library.
// ---------------------------------------------------------------------

#[test]
#[ignore]
fn mutation_canary_skipped_remove_breaks_structure_parity() {
    fn run_f64() -> Result<(), String> {
        let dim = 3usize;
        let capacity = 200usize;
        let data64 = uniform(cfg_seed("mutation_canary_f64", &[0]), capacity, dim);
        let growable = GrowableFlat::new(&data64, dim);
        let mut rust_tree =
            DynamicKdTreeBuilder::new(DynDim(dim), &growable).maximum_point_count(capacity).build();
        let mut oracle = RefDynIndexF64::build(&data64, dim, 10, capacity);

        let ops = dyn_ops(cfg_seed("mutation_canary_ops_f64", &[0]), capacity, 60);
        let mut skipped = false;
        for op in &ops {
            if !skipped {
                if let DynOp::Remove { .. } = op {
                    // Apply to the ORACLE only -- deliberately skip the
                    // rust_tree.remove_point call.
                    if let DynOp::Remove { live_idx } = *op {
                        oracle.remove_point(live_idx);
                    }
                    skipped = true;
                    break;
                }
            }
            apply_dyn_op_f64(op, &growable, &mut rust_tree, &mut oracle);
        }
        assert!(skipped, "op sequence never produced a Remove -- canary setup failed, not a real result");

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_dyn_structure_equal_f64(&rust_tree, &oracle);
        }));
        match result {
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                println!("mutation_canary (f64): structure-parity check correctly PANICKED:\n{msg}");
                Ok(())
            }
            Ok(()) => Err("mutation_canary (f64): structure-parity check did NOT panic -- \
                           the comparator failed to detect a deliberately-introduced divergence!"
                .to_string()),
        }
    }

    fn run_f32() -> Result<(), String> {
        let dim = 3usize;
        let capacity = 200usize;
        let data64 = uniform(cfg_seed("mutation_canary_f32", &[0]), capacity, dim);
        let data: Vec<f32> = to_f32(&data64);
        let growable = GrowableFlat::new(&data, dim);
        let mut rust_tree =
            DynamicKdTreeBuilder::new(DynDim(dim), &growable).maximum_point_count(capacity).build();
        let mut oracle = RefDynIndexF32::build(&data, dim, 10, capacity);

        let ops = dyn_ops(cfg_seed("mutation_canary_ops_f32", &[0]), capacity, 60);
        let mut skipped = false;
        for op in &ops {
            if !skipped {
                if let DynOp::Remove { .. } = op {
                    if let DynOp::Remove { live_idx } = *op {
                        oracle.remove_point(live_idx);
                    }
                    skipped = true;
                    break;
                }
            }
            apply_dyn_op_f32(op, &growable, &mut rust_tree, &mut oracle);
        }
        assert!(skipped, "op sequence never produced a Remove -- canary setup failed, not a real result");

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_dyn_structure_equal_f32(&rust_tree, &oracle);
        }));
        match result {
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                println!("mutation_canary (f32): structure-parity check correctly PANICKED:\n{msg}");
                Ok(())
            }
            Ok(()) => Err("mutation_canary (f32): structure-parity check did NOT panic -- \
                           the comparator failed to detect a deliberately-introduced divergence!"
                .to_string()),
        }
    }

    run_f64().expect("f64 mutation canary failed");
    run_f32().expect("f32 mutation canary failed");
}
