//! Deterministic parallel tree builder (feature "parallel"): a rayon port of
//! `SubtreeBuilder::build` (`build.rs`) that produces a tree BIT-IDENTICAL to
//! the sequential build — same `vind`, same arena contents, same node order.
//!
//! This is stronger than C++ `divideTreeConcurrent` (nanoflann.hpp
//! ~1422-1479): the C++ also partitions the whole range BEFORE spawning (so
//! its `vAcc_` matches the sequential build), but its node ALLOCATION order
//! differs from `divideTree`'s because both branches push into one shared
//! pool concurrently. Here, each recursive call builds a task-local `Vec`
//! arena (no shared mutable state, hence no need for a mutex or atomic
//! counter — `vind.split_at_mut` alone proves the two halves never alias),
//! and the merge step splices `[parent, left-subtree-preorder,
//! right-subtree-preorder]` back together — EXACTLY the layout the
//! sequential iterative builder produces, so the two arenas end up
//! byte-for-byte, node-for-node identical.
//!
//! # Determinism argument
//!
//! `middle_split` (the partition step) runs entirely BEFORE `rayon::join`
//! spawns anything — the split index, cut feature and cut value are all
//! fixed, single-threaded, deterministic values before any parallelism
//! happens at all. The two recursive calls then operate on DISJOINT
//! sub-slices of `vind` (`split_at_mut`) and build into their OWN local
//! arenas — there is no shared mutable state during the parallel region, so
//! there is nothing for a scheduler to race on. The only place indices are
//! rewritten is the merge, which is a pure, sequential, single-threaded
//! function of the two (now-finished) child arenas' lengths — not of
//! scheduling order, thread count, or which child happened to finish first
//! (`rayon::join`'s return is a tuple keyed by ARGUMENT position, not
//! completion order).

use crate::bbox::Interval;
use crate::build::{cpp_max, cpp_min, middle_split, SubtreeBuilder};
use crate::data_source::DataSource;
use crate::node::Node;
use crate::scalar::{IndexType, Scalar};

/// Below this many points, a subtree is built sequentially rather than
/// spawning further rayon tasks — task/allocation overhead stops paying for
/// itself long before this, and it bounds how many tiny `Vec` arenas get
/// allocated and re-merged.
const PARALLEL_CUTOFF: usize = 4096;

/// `ceil(log2(x))` for `x >= 1`, computed without floating point (this value
/// only ever gates parallel-vs-sequential dispatch — a control decision, not
/// data — so it has no bearing on the determinism claim; integer arithmetic
/// is simply the cheapest way to compute it).
fn ceil_log2(x: usize) -> u32 {
    if x <= 1 {
        0
    } else {
        (usize::BITS) - (x - 1).leading_zeros()
    }
}

/// `(2 * ceil(log2(max(n / PARALLEL_CUTOFF, 2))) + 8).min(64)` — enough
/// recursion depth to reach the cutoff on balanced data with margin.
/// Degenerate (highly unbalanced) splits burn through this budget faster
/// than balanced ones and fall back to the sequential base case once it hits
/// zero: safe (the sequential builder is iterative/stack-safe, see
/// `build.rs`'s module doc), it just loses parallelism on adversarial data
/// — that tradeoff is intentional, not a bug.
pub(crate) fn depth_budget_for(n: usize) -> usize {
    let ratio = (n / PARALLEL_CUTOFF).max(2);
    let budget = 2 * ceil_log2(ratio) as usize + 8;
    budget.min(64)
}

/// Parallel subtree build. Same contract as `SubtreeBuilder::build`, but
/// returns a task-local arena (root at its index 0-relative position)
/// instead of appending to a shared one. `bbox` is mutated to the tight box,
/// exactly like `SubtreeBuilder::build`.
pub(crate) fn build_subtree_parallel<T, DS, Idx>(
    ds: &DS,
    dim: usize,
    leaf_max_size: usize,
    base: u32,
    vind: &mut [Idx],
    bbox: &mut [Interval<T>],
    depth_budget: usize,
) -> (u32, Vec<Node<T>>)
where
    T: Scalar,
    DS: DataSource<T> + ?Sized + Sync,
    Idx: IndexType,
{
    let n = vind.len();

    // Base case: bound native stack depth on degenerate trees by handing off
    // to the ITERATIVE sequential builder (see build.rs's module doc) rather
    // than ever recursing further here — this function's own recursion is
    // capped by `depth_budget`, but that cap alone doesn't bound the
    // SEQUENTIAL builder's recursion if it were naively recursive; routing
    // to `SubtreeBuilder` sidesteps the question entirely.
    if n <= PARALLEL_CUTOFF || depth_budget == 0 || n <= leaf_max_size {
        let mut arena = Vec::new();
        let root = {
            let mut builder = SubtreeBuilder {
                ds,
                dim,
                leaf_max_size,
                base,
                vind,
                arena: &mut arena,
            };
            builder.build(bbox)
        };
        return (root, arena);
    }

    // Partition BEFORE spawning — determinism hinges on this: split_index,
    // cutfeat and cutval are fixed, single-threaded values before any
    // parallel work begins.
    let (split_index, cutfeat, cutval) = middle_split(ds, dim, vind, bbox);
    let cutfeat = cutfeat as usize;

    let mut left_bbox = bbox.to_vec();
    left_bbox[cutfeat].high = cutval;
    let mut right_bbox = bbox.to_vec();
    right_bbox[cutfeat].low = cutval;

    // Disjoint sub-slices — `split_at_mut` is itself the proof of
    // data-race-freedom the module doc refers to.
    let (lv, rv) = vind.split_at_mut(split_index);
    let right_base = base + split_index as u32;

    let ((left_root, mut left_arena), (right_root, mut right_arena)) = rayon::join(
        || build_subtree_parallel(ds, dim, leaf_max_size, base, lv, &mut left_bbox, depth_budget - 1),
        || build_subtree_parallel(ds, dim, leaf_max_size, right_base, rv, &mut right_bbox, depth_budget - 1),
    );
    // `left_bbox`/`right_bbox` are now each child's TIGHT bbox (mutated by
    // the recursive call, exactly like `SubtreeBuilder::build`'s contract).

    // Merge (pre-order preserving): [parent, left-subtree, right-subtree] —
    // EXACTLY the sequential arena's layout.
    let l_len = left_arena.len() as u32;
    let mut arena = Vec::with_capacity(1 + left_arena.len() + right_arena.len());
    // Push the split node FIRST; children/div-bounds patched below once its
    // final index (0) and the child arenas' final offsets are known.
    arena.push(Node::split(cutfeat as u32, T::default(), T::default()));

    // Left subtree lands at arena[1..1+l_len) — every INTERIOR left node's
    // children shift by 1 (the parent's own slot). Leaf ranges are GLOBAL
    // (base-relative) already and must NOT be offset — see
    // `Node::offset_children`'s doc comment.
    for node in left_arena.iter_mut() {
        if !node.is_leaf() {
            node.offset_children(1);
        }
    }
    arena.append(&mut left_arena);

    // Right subtree lands at arena[1+l_len..) — every INTERIOR right node's
    // children shift by `1 + l_len`.
    for node in right_arena.iter_mut() {
        if !node.is_leaf() {
            node.offset_children(1 + l_len);
        }
    }
    arena.append(&mut right_arena);

    arena[0].set_children(1 + left_root, 1 + l_len + right_root);
    arena[0].set_div_bounds(left_bbox[cutfeat].high, right_bbox[cutfeat].low);

    // Union the two tight child bboxes into `bbox` — same std::min/std::max
    // argument order (left first, right second) as `SubtreeBuilder::build`'s
    // `Work::Finalize` arm.
    for d in 0..dim {
        bbox[d] = Interval {
            low: cpp_min(left_bbox[d].low, right_bbox[d].low),
            high: cpp_max(left_bbox[d].high, right_bbox[d].high),
        };
    }

    (0, arena)
}

/// Top-level entry used by `tree.rs`: builds the whole tree's arena over
/// `vind` (identity permutation on entry, permuted on return, exactly like
/// `SubtreeBuilder::build`'s top-level caller), computes the depth budget
/// from `vind.len()`, and returns the arena (root always at index 0).
pub(crate) fn build_tree_parallel<T, DS, Idx>(
    ds: &DS,
    dim: usize,
    leaf_max_size: usize,
    vind: &mut [Idx],
    bbox: &mut [Interval<T>],
) -> Vec<Node<T>>
where
    T: Scalar,
    DS: DataSource<T> + ?Sized + Sync,
    Idx: IndexType,
{
    let depth_budget = depth_budget_for(vind.len());
    let (root, arena) = build_subtree_parallel(ds, dim, leaf_max_size, 0, vind, bbox, depth_budget);
    debug_assert_eq!(root, 0, "root is expected to always be arena index 0");
    arena
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::build::init_vind;
    use crate::dim::ConstDim;
    use crate::metric::L2;
    use crate::result_set::{KnnResultSet, ResultSet};
    use crate::search::{find_neighbors, SearchCtx};

    // ---------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn seeded_points<const N: usize>(seed: u64, n: usize, scale: f64) -> Vec<[f64; N]> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let mut p = [0.0f64; N];
                for d in 0..N {
                    p[d] = rng.next_f64() * scale;
                }
                p
            })
            .collect()
    }

    /// Builds SEQUENTIALLY via `SubtreeBuilder`. Returns (arena, vind, root, bbox).
    fn build_sequential<const N: usize>(
        points: &[[f64; N]],
        leaf_max_size: usize,
    ) -> (Vec<Node<f64>>, Vec<u32>, u32, Vec<Interval<f64>>) {
        let dim = N;
        let n = points.len();
        let mut vind: Vec<u32> = init_vind(n);
        let mut bbox = vec![Interval { low: 0.0, high: 0.0 }; dim];
        crate::bbox::compute_bounding_box(&points, dim, &mut bbox);
        let mut arena = Vec::new();
        let root = {
            let mut builder = SubtreeBuilder {
                ds: &points,
                dim,
                leaf_max_size,
                base: 0,
                vind: &mut vind,
                arena: &mut arena,
            };
            builder.build(&mut bbox)
        };
        (arena, vind, root, bbox)
    }

    /// Builds via the PARALLEL entry point. Returns (arena, vind, root=0, bbox).
    fn build_parallel<const N: usize>(
        points: &[[f64; N]],
        leaf_max_size: usize,
    ) -> (Vec<Node<f64>>, Vec<u32>, u32, Vec<Interval<f64>>) {
        let dim = N;
        let n = points.len();
        let mut vind: Vec<u32> = init_vind(n);
        let mut bbox = vec![Interval { low: 0.0, high: 0.0 }; dim];
        crate::bbox::compute_bounding_box(&points, dim, &mut bbox);
        let arena = build_tree_parallel(&points, dim, leaf_max_size, &mut vind, &mut bbox);
        (arena, vind, 0, bbox)
    }

    /// Runs `build_parallel` inside an n-thread scoped rayon pool (models
    /// `BuildThreads::Threads(n)`).
    fn build_parallel_with_pool<const N: usize>(
        points: &[[f64; N]],
        leaf_max_size: usize,
        n_threads: usize,
    ) -> (Vec<Node<f64>>, Vec<u32>, u32, Vec<Interval<f64>>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("rayon pool");
        pool.install(|| build_parallel(points, leaf_max_size))
    }

    fn assert_builds_bit_identical<const N: usize>(points: &[[f64; N]], leaf_max_size: usize) {
        let (seq_arena, seq_vind, seq_root, seq_bbox) = build_sequential(points, leaf_max_size);

        let (auto_arena, auto_vind, auto_root, auto_bbox) = build_parallel(points, leaf_max_size);
        assert_eq!(seq_vind, auto_vind, "vind mismatch (Auto)");
        assert_eq!(seq_root, auto_root, "root mismatch (Auto)");
        assert_eq!(seq_arena, auto_arena, "arena mismatch (Auto)");
        assert_eq!(seq_bbox, auto_bbox, "root_bbox mismatch (Auto)");

        let (t2_arena, t2_vind, t2_root, t2_bbox) = build_parallel_with_pool(points, leaf_max_size, 2);
        assert_eq!(seq_vind, t2_vind, "vind mismatch (Threads(2))");
        assert_eq!(seq_root, t2_root, "root mismatch (Threads(2))");
        assert_eq!(seq_arena, t2_arena, "arena mismatch (Threads(2))");
        assert_eq!(seq_bbox, t2_bbox, "root_bbox mismatch (Threads(2))");
    }

    // ---------------------------------------------------------------
    // 1. Bit-identical to sequential: 5000 seeded uniform points, dim 3.
    // ---------------------------------------------------------------

    #[test]
    fn bit_identical_to_sequential_5000_uniform_dim3() {
        let points = seeded_points::<3>(0xC0FFEE_u64, 5000, 1000.0);
        assert_builds_bit_identical(&points, 10);
    }

    // ---------------------------------------------------------------
    // 2. Query equivalence sanity: 200 knn queries, k=10, identical results.
    // ---------------------------------------------------------------

    #[test]
    fn query_equivalence_200_knn_queries_k10() {
        let points = seeded_points::<3>(0xC0FFEE_u64, 5000, 1000.0);
        let (seq_arena, seq_vind, _root, seq_bbox) = build_sequential(&points, 10);
        let (auto_arena, auto_vind, _root2, auto_bbox) = build_parallel(&points, 10);

        let seq_ctx = SearchCtx {
            ds: &points.as_slice(),
            metric: &L2,
            dim: ConstDim::<3>,
            nodes: &seq_arena,
            vind: &seq_vind,
            root_bbox: &seq_bbox,
        };
        let auto_ctx = SearchCtx {
            ds: &points.as_slice(),
            metric: &L2,
            dim: ConstDim::<3>,
            nodes: &auto_arena,
            vind: &auto_vind,
            root_bbox: &auto_bbox,
        };

        let mut rng = Lcg(0xA5A5_u64);
        let params = crate::params::SearchParams::default();
        for _ in 0..200 {
            let query = [rng.next_f64() * 1000.0, rng.next_f64() * 1000.0, rng.next_f64() * 1000.0];
            let k = 10;

            let mut seq_idx = vec![0u32; k];
            let mut seq_dist = vec![0.0f64; k];
            let mut seq_rs = KnnResultSet::<f64, u32>::new(&mut seq_idx, &mut seq_dist);
            let mut scratch = vec![0.0f64; 3];
            find_neighbors(&seq_ctx, &mut seq_rs, &query, &params, &crate::filter::AcceptAll, &mut scratch);
            let seq_found = seq_rs.size();

            let mut auto_idx = vec![0u32; k];
            let mut auto_dist = vec![0.0f64; k];
            let mut auto_rs = KnnResultSet::<f64, u32>::new(&mut auto_idx, &mut auto_dist);
            let mut scratch2 = vec![0.0f64; 3];
            find_neighbors(&auto_ctx, &mut auto_rs, &query, &params, &crate::filter::AcceptAll, &mut scratch2);
            let auto_found = auto_rs.size();

            assert_eq!(seq_found, auto_found);
            assert_eq!(seq_idx, auto_idx);
            assert_eq!(seq_dist, auto_dist);
        }
    }

    // ---------------------------------------------------------------
    // 3. Duplicate-heavy determinism: 5000 points, 50% duplicates.
    // ---------------------------------------------------------------

    #[test]
    fn bit_identical_duplicate_heavy_5000_points_50_percent() {
        let mut rng = Lcg(0xDEADBEEF_u64);
        let unique_n = 2500usize;
        let mut points: Vec<[f64; 3]> = (0..unique_n)
            .map(|_| [rng.next_f64() * 500.0, rng.next_f64() * 500.0, rng.next_f64() * 500.0])
            .collect();
        for i in 0..(5000 - unique_n) {
            points.push(points[i % unique_n]);
        }
        assert_eq!(points.len(), 5000);
        assert_builds_bit_identical(&points, 10);
    }

    // ---------------------------------------------------------------
    // 4. Small n: n < cutoff -> trivially exercises the base case.
    // ---------------------------------------------------------------

    #[test]
    fn small_n_below_cutoff_matches_sequential_trivially() {
        let points = seeded_points::<3>(0x1234_u64, 200, 50.0);
        assert!(points.len() <= PARALLEL_CUTOFF);
        assert_builds_bit_identical(&points, 10);
    }

    // ---------------------------------------------------------------
    // 5. Heavy degenerate (release, ignored): 1M exponential-spacing points.
    // ---------------------------------------------------------------

    /// Iterative max-depth (root = depth 1) — same helper shape as build.rs's.
    fn max_depth(arena: &[Node<f64>], root: u32) -> usize {
        let mut stack = vec![(root, 1usize)];
        let mut max_d = 0usize;
        while let Some((idx, d)) = stack.pop() {
            if d > max_d {
                max_d = d;
            }
            let node = &arena[idx as usize];
            if !node.is_leaf() {
                let (c1, c2) = node.children();
                stack.push((c1, d + 1));
                stack.push((c2, d + 1));
            }
        }
        max_d
    }

    /// Structural invariant checker (iterative), mirroring build.rs's
    /// `check_tree`: every point appears in exactly one leaf, node count <=
    /// 2n, and every interior node's div bounds exactly match its children's
    /// tight bboxes on the split axis.
    fn check_tree(points: &[[f64; 1]], arena: &[Node<f64>], vind: &[u32], root: u32, leaf_max_size: usize) {
        let dim = 1usize;
        let n = points.len();
        assert!(arena.len() <= 2 * n, "node count {} exceeds 2n ({})", arena.len(), 2 * n);

        enum St {
            Visit(u32),
            Process(u32),
        }

        let mut seen = vec![false; n];
        let mut stack = vec![St::Visit(root)];
        let mut results: Vec<Vec<Interval<f64>>> = Vec::new();

        while let Some(item) = stack.pop() {
            match item {
                St::Visit(idx) => {
                    let node = &arena[idx as usize];
                    if node.is_leaf() {
                        let (l, r) = node.leaf_range();
                        assert!(r - l <= leaf_max_size, "leaf [{l},{r}) has {} points > leaf_max_size {leaf_max_size}", r - l);
                        let mut leaf_bbox = vec![Interval { low: 0.0, high: 0.0 }; dim];
                        for (i, k) in (l..r).enumerate() {
                            let pt = vind[k] as usize;
                            assert!(!seen[pt], "point {pt} appears in more than one leaf");
                            seen[pt] = true;
                            for d in 0..dim {
                                let v = points[pt][d];
                                if i == 0 {
                                    leaf_bbox[d] = Interval { low: v, high: v };
                                } else {
                                    if v < leaf_bbox[d].low {
                                        leaf_bbox[d].low = v;
                                    }
                                    if v > leaf_bbox[d].high {
                                        leaf_bbox[d].high = v;
                                    }
                                }
                            }
                        }
                        results.push(leaf_bbox);
                    } else {
                        let (c1, c2) = node.children();
                        stack.push(St::Process(idx));
                        stack.push(St::Visit(c2));
                        stack.push(St::Visit(c1));
                    }
                }
                St::Process(idx) => {
                    let node = &arena[idx as usize];
                    let cutfeat = node.split_dim();
                    let right_bbox = results.pop().expect("missing right subtree bbox");
                    let left_bbox = results.pop().expect("missing left subtree bbox");

                    assert_eq!(left_bbox[cutfeat].high, node.div_low(), "node {idx}: div_low mismatch");
                    assert_eq!(right_bbox[cutfeat].low, node.div_high(), "node {idx}: div_high mismatch");

                    let mut combined = vec![Interval { low: 0.0, high: 0.0 }; dim];
                    for d in 0..dim {
                        combined[d] = Interval {
                            low: cpp_min(left_bbox[d].low, right_bbox[d].low),
                            high: cpp_max(left_bbox[d].high, right_bbox[d].high),
                        };
                    }
                    results.push(combined);
                }
            }
        }

        assert!(seen.iter().all(|&b| b), "not every point index appears in a leaf");
        assert_eq!(results.len(), 1);
    }

    #[test]
    #[ignore]
    fn heavy_exponential_parallel_build_1m() {
        let n = 1_000_000usize;
        // Same spine construction as build.rs's heavy_exponential_build_1m
        // (see that test's doc comment for the derivation of why this
        // reaches a maximally degenerate ~2115-deep tree).
        let mut spine: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(1023);
        while v > 0.0 && spine.len() < n - 1 {
            spine.push(v);
            v /= 2.0;
        }
        let mut values = spine;
        values.resize(n, 0.0);
        let points: Vec<[f64; 1]> = values.into_iter().map(|v| [v]).collect();

        let (arena, vind, root, _bbox) = build_parallel(&points, 10);

        let depth = max_depth(&arena, root);
        assert!(depth > 2_000, "expected a deeply degenerate tree, got depth {depth}");

        check_tree(&points, &arena, &vind, root, 10);
    }
}
