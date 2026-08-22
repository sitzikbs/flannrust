//! Query core — the exact port of nanoflann's `searchLevel` +
//! `computeInitialDistances` + the static `findNeighbors` driver
//! (nanoflann.hpp: `searchLevel` ~1228-1287, `computeInitialDistances`
//! ~1595-1605, `findNeighbors` ~1990-2011).

use crate::bbox::Interval;
use crate::build::IndexAccess;
use crate::data_source::DataSource;
use crate::filter::PointFilter;
use crate::metric::Distance;
use crate::node::Node;
use crate::params::SearchParams;
use crate::result_set::ResultSet;
use crate::scalar::{DistanceValue, Scalar};

/// Everything a query needs, borrowed: dataset, metric, dimensionality, the
/// built arena + permuted index vector, and the tree's root bounding box.
/// The root node is always arena index 0 (task 6's `SubtreeBuilder::build`
/// always allocates the top-level node first).
pub(crate) struct SearchCtx<'a, T: Scalar, DS: DataSource<T> + ?Sized, M: Distance<T>, Idx> {
    pub ds: &'a DS,
    pub metric: &'a M,
    pub dim: usize,
    pub nodes: &'a [Node<T>],
    pub vind: &'a [Idx],
    pub root_bbox: &'a [Interval<T>],
}

/// = static `findNeighbors` (nanoflann.hpp ~1990-2011) minus the unbuilt-tree
/// throw (unrepresentable here: `SearchCtx` can only be constructed from an
/// already-built tree in later tasks). Empty tree (`ctx.nodes` empty) →
/// `false`, matching C++'s `if (this->size(*this) == 0) return false;`.
///
/// `dists_scratch` is caller-provided, length `ctx.dim`, zeroed here (callers
/// with `ConstDim` pass a stack array — no per-query heap allocation).
#[allow(dead_code)] // TODO(task-10): called by the tree façade's query methods (knn_search / radius_search / rknn_search)
pub(crate) fn find_neighbors<T, DS, M, Idx, R, F>(
    ctx: &SearchCtx<T, DS, M, Idx>,
    result: &mut R,
    query: &[T],
    params: &SearchParams,
    filter: &F,
    dists_scratch: &mut [M::DistanceType],
) -> bool
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexAccess,
    R: ResultSet<M::DistanceType, Idx>,
    F: PointFilter<Idx>,
{
    if ctx.nodes.is_empty() {
        return false;
    }
    debug_assert!(query.len() >= ctx.dim, "query vector shorter than dim");
    debug_assert_eq!(dists_scratch.len(), ctx.dim, "dists_scratch.len() != dim");

    // C++ (nanoflann.hpp ~2006): `DistanceType epsError = 1 + static_cast<DistanceType>(searchParams.eps);`
    // — literally casts `eps` to `DistanceType` FIRST, then adds `1` in
    // `DistanceType` arithmetic. Per the M1 ruling (task-8 brief context),
    // we instead compute `1.0f32 + eps` in `f32` FIRST and only THEN widen
    // via `DistanceValue::from_f32`, to get f64 bit-parity with the pinned
    // set of test vectors. Deliberate discrepancy from the literal C++ cast
    // order — see task-8 report.
    let eps_error = M::DistanceType::from_f32(1.0f32 + params.eps);

    for d in dists_scratch.iter_mut() {
        *d = M::DistanceType::ZERO;
    }

    let mindist = compute_initial_distances(ctx, query, dists_scratch);

    // C++ discards `searchLevel`'s own return value here — it's only used to
    // propagate an abort UP THROUGH the recursion, not by the top-level
    // driver.
    let _ = search_level(ctx, result, query, 0, mindist, dists_scratch, eps_error, filter);

    if params.sorted {
        result.sort();
    }

    result.full()
}

/// Port of `computeInitialDistances` (nanoflann.hpp ~1595-1605). For each
/// dim `i`: if `query[i] < root_bbox[i].low`, accumulate the per-axis
/// component distance to the low edge; ELSE IF `query[i] > root_bbox[i].high`
/// (an `else if`, NOT an independent second `if` — the brief's prose says
/// "two independent ifs", but the vendored C++ source at nanoflann.hpp:1608
/// uses `else if`; functionally equivalent for a well-formed bbox where
/// `low <= high`, but we mirror the source exactly per task instructions).
/// Returns the summed lower-bound distance.
fn compute_initial_distances<T, DS, M, Idx>(
    ctx: &SearchCtx<T, DS, M, Idx>,
    query: &[T],
    dists: &mut [M::DistanceType],
) -> M::DistanceType
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
{
    let mut dist = M::DistanceType::ZERO;
    for i in 0..ctx.dim {
        if query[i] < ctx.root_bbox[i].low {
            dists[i] = ctx.metric.accum_dist(query[i], ctx.root_bbox[i].low, i);
            dist = dist + dists[i];
        } else if query[i] > ctx.root_bbox[i].high {
            dists[i] = ctx.metric.accum_dist(query[i], ctx.root_bbox[i].high, i);
            dist = dist + dists[i];
        }
    }
    dist
}

/// Port of `searchLevel` (nanoflann.hpp ~1228-1287). Native recursion —
/// depth = tree depth (bounded because task 6's builder survives degenerate
/// data without a matching depth blowup on the SEARCH side, since a search
/// path only ever follows one branch down to a leaf, plus O(depth) pruned
/// side-branches — this is NOT a stack-safety concern the way tree
/// construction was).
///
/// Returns `false` to abort (propagated from `ResultSet::add_point`
/// returning `false`, nanoflann.hpp ~1245-1250), `true` to continue.
#[allow(clippy::too_many_arguments)]
fn search_level<T, DS, M, Idx, R, F>(
    ctx: &SearchCtx<T, DS, M, Idx>,
    result: &mut R,
    query: &[T],
    node_idx: u32,
    mut mindist: M::DistanceType,
    dists: &mut [M::DistanceType],
    eps_error: M::DistanceType,
    filter: &F,
) -> bool
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    M: Distance<T>,
    Idx: IndexAccess,
    R: ResultSet<M::DistanceType, Idx>,
    F: PointFilter<Idx>,
{
    let node = &ctx.nodes[node_idx as usize];

    if node.is_leaf() {
        let (left, right) = node.leaf_range();
        for i in left..right {
            let accessor = ctx.vind[i];
            if !filter.is_active(accessor) {
                continue;
            }
            let dist = ctx.metric.eval(query, ctx.ds, accessor.to_usize(), ctx.dim);
            // C++ (nanoflann.hpp ~1259): `result_set.worstDist()` is called
            // LIVE, inline in the loop condition, on every iteration — NOT
            // hoisted into a local cached before the loop (the brief's
            // pseudocode suggests hoisting; the vendored source does not).
            // Functionally equivalent (worst_dist only changes via
            // add_point), but we mirror the exact call site per task
            // instructions.
            if dist < result.worst_dist() {
                if !result.add_point(dist, accessor) {
                    return false;
                }
            }
        }
        return true;
    }

    // Which child branch should be taken first?
    let idx = node.split_dim();
    let val = query[idx];
    // Kept in element type `T` (not `M::DistanceType`): C++'s `diff1`/`diff2`
    // are declared `DistanceType` but assigned from an `ElementType`
    // subtraction, and only ever used for a sign comparison — this crate
    // doesn't assume `T == M::DistanceType`, so the comparison is done
    // entirely in `T`.
    let diff1 = val - node.div_low();
    let diff2 = val - node.div_high();

    let (c1, c2) = node.children();
    let (best_child, other_child, cut_dist) = if (diff1 + diff2) < T::default() {
        (c1, c2, ctx.metric.accum_dist(val, node.div_high(), idx))
    } else {
        (c2, c1, ctx.metric.accum_dist(val, node.div_low(), idx))
    };

    // Call recursively to search next level down.
    if !search_level(ctx, result, query, best_child, mindist, dists, eps_error, filter) {
        return false;
    }

    let dst = dists[idx];
    mindist = mindist + cut_dist - dst;
    dists[idx] = cut_dist;
    if mindist * eps_error <= result.worst_dist() {
        if !search_level(ctx, result, query, other_child, mindist, dists, eps_error, filter) {
            return false;
        }
    }
    dists[idx] = dst;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::compute_bounding_box;
    use crate::build::{init_vind, SubtreeBuilder};
    use crate::filter::AcceptAll;
    use crate::metric::L2;
    use crate::result_set::KnnResultSet;

    /// Build a full tree over `pts` (T = f64) via `SubtreeBuilder`, base 0,
    /// identity `vind`, tight root bbox from `compute_bounding_box`. The
    /// root is always arena index 0 (see `SearchCtx` doc comment).
    fn build_tree<const N: usize>(
        pts: &[[f64; N]],
        leaf_max_size: usize,
    ) -> (Vec<u32>, Vec<Node<f64>>, Vec<Interval<f64>>) {
        let dim = N;
        let n = pts.len();
        let mut vind: Vec<u32> = init_vind(n);
        let mut bbox = vec![Interval { low: 0.0, high: 0.0 }; dim];
        compute_bounding_box(&pts, dim, &mut bbox);
        let mut arena = Vec::new();
        {
            let mut builder = SubtreeBuilder {
                ds: &pts,
                dim,
                leaf_max_size,
                base: 0,
                vind: &mut vind,
                arena: &mut arena,
            };
            let root = builder.build(&mut bbox);
            assert_eq!(root, 0, "root is expected to always be arena index 0");
        }
        (vind, arena, bbox)
    }

    // ---------------------------------------------------------------
    // Test 1: brute-force property (RED evidence for now — stub always
    // returns `false` / touches nothing, so this fails against ANY non-empty
    // case).
    // ---------------------------------------------------------------

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn brute_force_knn<const N: usize>(pts: &[[f64; N]], query: &[f64; N], k: usize) -> (Vec<u32>, Vec<f64>) {
        let dim = N;
        let metric = L2;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            for i in 0..pts.len() {
                let d = metric.eval(query.as_slice(), &pts, i, dim);
                rs.add_point(d, i as u32);
            }
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists)
    }

    fn tree_knn<const N: usize>(
        vind: &[u32],
        arena: &[Node<f64>],
        bbox: &[Interval<f64>],
        pts: &[[f64; N]],
        query: &[f64; N],
        k: usize,
        eps: f32,
    ) -> (Vec<u32>, Vec<f64>, bool) {
        let dim = N;
        let metric = L2;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim, nodes: arena, vind, root_bbox: bbox };
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        let full;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = vec![0.0f64; dim];
            let params = SearchParams { eps, sorted: true };
            full = find_neighbors(&ctx, &mut rs, query.as_slice(), &params, &AcceptAll, &mut scratch);
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);
        (indices, dists, full)
    }

    fn run_brute_force_case<const N: usize>(seed: u64, n: usize, k: usize, leaf_max_size: usize) {
        let mut rng = Lcg(seed);
        let pts: Vec<[f64; N]> = (0..n)
            .map(|_| {
                let mut p = [0.0f64; N];
                for d in 0..N {
                    p[d] = rng.next_f64() * 200.0 - 100.0;
                }
                p
            })
            .collect();
        let mut query = [0.0f64; N];
        for d in 0..N {
            query[d] = rng.next_f64() * 200.0 - 100.0;
        }

        let (vind, arena, bbox) = build_tree(&pts, leaf_max_size);
        let (tree_idx, tree_dists, tree_full) = tree_knn(&vind, &arena, &bbox, &pts, &query, k, 0.0);
        let (bf_idx, bf_dists) = brute_force_knn(&pts, &query, k);

        assert_eq!(tree_idx, bf_idx, "seed={seed} n={n} dim={N} k={k}: index mismatch");
        assert_eq!(tree_dists, bf_dists, "seed={seed} n={n} dim={N} k={k}: distance mismatch");
        assert_eq!(tree_full, bf_idx.len() == k, "seed={seed} n={n} dim={N} k={k}: full() mismatch");
    }

    #[test]
    fn brute_force_property_200_seeded_cases() {
        let mut seed = 0xF00Du64;
        for case in 0..200u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let dim_choice = case % 3;
            let mut rng = Lcg(seed ^ 0xABCD);
            let n = 1 + (rng.next_f64() * 200.0) as usize;
            let n = n.min(200).max(1);
            let k_max = n.min(20);
            let k = 1 + (rng.next_f64() * (k_max as f64)) as usize;
            let k = k.min(k_max).max(1);

            match dim_choice {
                0 => run_brute_force_case::<2>(seed, n, k, 10),
                1 => run_brute_force_case::<3>(seed, n, k, 10),
                _ => run_brute_force_case::<8>(seed, n, k, 10),
            }
        }
    }

    // ---------------------------------------------------------------
    // Test 2: k > n
    // ---------------------------------------------------------------

    #[test]
    fn k_greater_than_n_returns_exactly_n_results_and_not_full() {
        let pts: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let (idx, _dists, full) = tree_knn(&vind, &arena, &bbox, &pts, &[0.5, 0.5], 10, 0.0);
        assert_eq!(idx.len(), 3);
        assert!(!full);
    }

    // ---------------------------------------------------------------
    // Test 3: query far outside root bbox
    // ---------------------------------------------------------------

    #[test]
    fn query_far_outside_bbox_50_seeded_cases() {
        let mut seed = 0xBEEFu64;
        for case in 0..50u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut rng = Lcg(seed);
            let n = 1 + (rng.next_f64() * 50.0) as usize;
            let pts: Vec<[f64; 3]> = (0..n)
                .map(|_| [rng.next_f64() * 20.0, rng.next_f64() * 20.0, rng.next_f64() * 20.0])
                .collect();
            let sign = if case % 2 == 0 { 1.0 } else { -1.0 };
            let query = [
                sign * 1000.0 + rng.next_f64(),
                sign * 1000.0 + rng.next_f64(),
                sign * 1000.0 + rng.next_f64(),
            ];
            let k = 1 + (rng.next_f64() * (n.min(5) as f64)) as usize;
            let k = k.min(n).max(1);

            let (vind, arena, bbox) = build_tree(&pts, 10);
            let (tree_idx, tree_dists, _) = tree_knn(&vind, &arena, &bbox, &pts, &query, k, 0.0);
            let (bf_idx, bf_dists) = brute_force_knn(&pts, &query, k);
            assert_eq!(tree_idx, bf_idx, "case={case}");
            assert_eq!(tree_dists, bf_dists, "case={case}");
        }
    }

    // ---------------------------------------------------------------
    // Test 9: empty tree
    // ---------------------------------------------------------------

    #[test]
    fn empty_tree_returns_false_and_leaves_result_untouched() {
        let pts: &[[f64; 2]] = &[];
        let arena: Vec<Node<f64>> = Vec::new();
        let vind: Vec<u32> = Vec::new();
        let bbox: Vec<Interval<f64>> = vec![Interval { low: 0.0, high: 0.0 }; 2];
        let metric = L2;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 2, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut indices = [99u32; 3];
        let mut dists = [77.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
        let mut scratch = [0.0f64; 2];
        let params = SearchParams::default();

        let full = find_neighbors(&ctx, &mut rs, &[0.0, 0.0], &params, &AcceptAll, &mut scratch);
        assert!(!full);
        assert_eq!(indices, [99, 99, 99]);
        assert_eq!(dists, [77.0, 77.0, 77.0]);
    }

    // ---------------------------------------------------------------
    // Test 10: leaf_max_size 1 and n
    // ---------------------------------------------------------------

    #[test]
    fn leaf_max_size_one_brute_force_equality_20_seeded_cases() {
        for case in 0..20u64 {
            let seed = 0x1EAF_0001u64.wrapping_add(case.wrapping_mul(747796405));
            run_brute_force_case::<3>(seed, 1 + (case as usize) * 3, 1 + (case as usize % 5), 1);
        }
    }

    #[test]
    fn leaf_max_size_n_brute_force_equality_20_seeded_cases() {
        for case in 0..20u64 {
            let seed = 0x1EAF_000Fu64.wrapping_add(case.wrapping_mul(747796405));
            let n = 1 + (case as usize) * 3;
            run_brute_force_case::<3>(seed, n, 1 + (case as usize % n.max(1)), n.max(1));
        }
    }

    // ---------------------------------------------------------------
    // Test 6: RKNN
    // ---------------------------------------------------------------

    #[test]
    fn rknn_partial_coverage_returns_false() {
        // 5 points at squared distances 1,4,9,16,25 from the query along one
        // axis; radius 10.0 covers only the first 3 (dist < 10.0: 1,4,9).
        let pts: Vec<[f64; 1]> = vec![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut indices = [0u32; 5];
        let mut dists = [0.0f64; 5];
        let full;
        {
            let mut rs = crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 10.0);
            let mut scratch = [0.0f64; 1];
            full = find_neighbors(&ctx, &mut rs, &[0.0], &SearchParams::default(), &AcceptAll, &mut scratch);
            assert_eq!(rs.size(), 3);
        }
        assert!(!full);
        assert_eq!(&indices[..3], &[0, 1, 2]);
    }

    #[test]
    fn rknn_full_coverage_returns_closest_k_and_true() {
        let pts: Vec<[f64; 1]> = vec![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]];
        let (vind, arena, bbox) = build_tree(&pts, 3);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut indices = [0u32; 5];
        let mut dists = [0.0f64; 5];
        let full;
        {
            // radius 100.0 (squared) covers all 10 points.
            let mut rs = crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 100.0);
            let mut scratch = [0.0f64; 1];
            full = find_neighbors(&ctx, &mut rs, &[0.0], &SearchParams::default(), &AcceptAll, &mut scratch);
            assert_eq!(rs.size(), 5);
        }
        assert!(full);
        // Closest 5 to 0.0 are points at 1,2,3,4,5 -> indices 0..5.
        assert_eq!(indices, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn rknn_exact_radius_boundary_excludes_point() {
        // Single point at exact distance^2 == radius from the query.
        let pts: Vec<[f64; 1]> = vec![[2.0]]; // dist^2 from 0.0 = 4.0
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut indices = [0u32; 1];
        let mut dists = [0.0f64; 1];
        {
            let mut rs = crate::result_set::RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 4.0);
            let mut scratch = [0.0f64; 1];
            find_neighbors(&ctx, &mut rs, &[0.0], &SearchParams::default(), &AcceptAll, &mut scratch);
            assert_eq!(rs.size(), 0, "point at exact squared radius must be excluded (dist < worst_dist gate)");
        }
    }

    // ---------------------------------------------------------------
    // Test 5: radius strict boundary + sort order
    // ---------------------------------------------------------------

    #[test]
    fn radius_strict_boundary_excludes_exact_and_includes_next_up() {
        // Grid points; one point at exact squared distance 4.0 from query (0,0): (2,0).
        let pts: Vec<[f64; 2]> = vec![[2.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 2, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(4.0f64, &mut items);
            let mut scratch = [0.0f64; 2];
            find_neighbors(&ctx, &mut rs, &[0.0, 0.0], &SearchParams::default(), &AcceptAll, &mut scratch);
        }
        assert!(
            !items.iter().any(|it| it.index == 0),
            "point at exact squared radius 4.0 must be absent"
        );

        let radius_next_up = f64::from_bits(4.0f64.to_bits() + 1);
        let mut items2 = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(radius_next_up, &mut items2);
            let mut scratch = [0.0f64; 2];
            find_neighbors(&ctx, &mut rs, &[0.0, 0.0], &SearchParams::default(), &AcceptAll, &mut scratch);
        }
        assert!(
            items2.iter().any(|it| it.index == 0),
            "point at radius next_up(4.0) must be present"
        );
    }

    #[test]
    fn radius_sorted_true_gives_ascending_distances() {
        let pts: Vec<[f64; 2]> = vec![[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
        let (vind, arena, bbox) = build_tree(&pts, 10);
        let metric = L2;
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 2, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(100.0f64, &mut items);
            let mut scratch = [0.0f64; 2];
            let params = SearchParams { eps: 0.0, sorted: true };
            find_neighbors(&ctx, &mut rs, &[0.0, 0.0], &params, &AcceptAll, &mut scratch);
        }
        let ds: Vec<f64> = items.iter().map(|it| it.distance).collect();
        let mut sorted = ds.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ds, sorted, "sorted=true must yield ascending distances");
    }

    #[test]
    fn radius_sorted_false_gives_tree_traversal_order() {
        // Small fixed 2-leaf tree, hand-derived traversal order.
        //
        // Points (dim-1, original index -> value): 0->0.0, 1->1.0, 2->10.0,
        // 3->11.0. Initial vind=[0,1,2,3], bbox=[0.0,11.0].
        //
        // middle_split: only candidate dim is 0 (trivially, dim=1); cutval =
        // clamp((0+11)/2, min=0, max=11) = 5.5. plane_split(ind=[0,1,2,3],
        // cutfeat=0, cutval=5.5), tracing nanoflann's 3-cursor partition
        // (build.rs::plane_split) exactly:
        //   left=0 mid=0 right=3
        //   mid=0: v[ind[0]]=v[0]=0.0  < 5.5 -> swap(0,0); left=1; mid=1   ind unchanged
        //   mid=1: v[ind[1]]=v[1]=1.0  < 5.5 -> swap(1,1); left=2; mid=2   ind unchanged
        //   mid=2: v[ind[2]]=v[2]=10.0 > 5.5 -> swap(2,3); right=2        ind=[0,1,3,2]
        //   mid=2: v[ind[2]]=v[3]=11.0 > 5.5 -> swap(2,2); right=1        ind unchanged
        //   mid=2 > right=1 -> loop ends. lim1=left=2, lim2=mid=2.
        // half = 4/2 = 2. lim1(2)>half(2)? no. lim2(2)<half(2)? no. -> index=half=2.
        //
        // So the split places vind[0..2]=[0,1] in the LEFT leaf (values
        // 0.0,1.0, in that order) and vind[2..4]=[3,2] in the RIGHT leaf —
        // note the swap left the right partition as [3,2], NOT input order
        // [2,3] (values 11.0,10.0 in that order): `plane_split` is not a
        // stable partition, per build.rs's own documented swap semantics.
        //
        // finalizeSplitNode: divlow = max(left subtree, dim0) = max(0,1) =
        // 1.0; divhigh = min(right subtree, dim0) = min(11,10) = 10.0
        // (order-independent, unaffected by the swap above).
        //
        // Query = 0.4 (near leaf A). Root: diff1=val-divlow=0.4-1.0=-0.6;
        // diff2=val-divhigh=0.4-10.0=-9.6; sum=-10.2 < 0 -> best_child = left
        // (A), visited FIRST, leaf loop scans its range in vind order [0,1]
        // (unsorted by distance). cut_dist=accum_dist(0.4,1.0)=(0.4-1.0)^2=0.36.
        // Both A's points are well within radius 200 -> added in order 0,
        // then 1. mindist = 0 + 0.36 - 0 = 0.36; eps=0 -> 0.36*1 <= 200 ->
        // visit other (B), leaf loop scans ITS vind range [3,2] (per the
        // swap above) -> added 3, then 2.
        //
        // Expected unsorted (traversal) order: indices [0, 1, 3, 2].
        let pts: Vec<[f64; 1]> = vec![[0.0], [1.0], [10.0], [11.0]];
        let (vind, arena, bbox) = build_tree(&pts, 2);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut items = Vec::new();
        {
            let mut rs = crate::result_set::RadiusResultSet::new(200.0f64, &mut items);
            let mut scratch = [0.0f64; 1];
            let params = SearchParams { eps: 0.0, sorted: false };
            find_neighbors(&ctx, &mut rs, &[0.4], &params, &AcceptAll, &mut scratch);
        }
        let order: Vec<u32> = items.iter().map(|it| it.index).collect();
        assert_eq!(order, vec![0, 1, 3, 2]);
    }

    // ---------------------------------------------------------------
    // Test 7: abort plumbing
    // ---------------------------------------------------------------

    struct AbortAfterN {
        limit: usize,
        adds: usize,
    }

    impl ResultSet<f64, u32> for AbortAfterN {
        fn worst_dist(&self) -> f64 {
            f64::MAX
        }
        fn add_point(&mut self, _dist: f64, _index: u32) -> bool {
            self.adds += 1;
            self.adds < self.limit
        }
        fn full(&self) -> bool {
            self.adds >= self.limit
        }
        fn size(&self) -> usize {
            self.adds
        }
    }

    #[test]
    fn abort_plumbing_stops_search_and_propagates_full() {
        // leaf_max_size=1 so every point lives in its own leaf; with a
        // ResultSet that always accepts (worst_dist == MAX) but returns
        // `false` from add_point after the 3rd add, the search must stop
        // dead after exactly 3 adds, regardless of how many points remain.
        let pts: Vec<[f64; 1]> = (0..20).map(|i| [i as f64]).collect();
        let (vind, arena, bbox) = build_tree(&pts, 1);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut rs = AbortAfterN { limit: 3, adds: 0 };
        let mut scratch = [0.0f64; 1];
        let full = find_neighbors(&ctx, &mut rs, &[0.0], &SearchParams::default(), &AcceptAll, &mut scratch);

        assert_eq!(rs.adds, 3, "exactly 3 adds should have happened before the 4th add_point aborts");
        assert_eq!(full, rs.full());
    }

    // ---------------------------------------------------------------
    // Test 8: filter plumbing
    // ---------------------------------------------------------------

    struct RejectEven;
    impl PointFilter<u32> for RejectEven {
        fn is_active(&self, idx: u32) -> bool {
            idx % 2 == 1
        }
    }

    #[test]
    fn filter_plumbing_rejects_even_indices() {
        let mut rng = Lcg(0x0DD_F17E7u64);
        let n = 37;
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| [rng.next_f64() * 50.0, rng.next_f64() * 50.0, rng.next_f64() * 50.0])
            .collect();
        let query = [rng.next_f64() * 50.0, rng.next_f64() * 50.0, rng.next_f64() * 50.0];

        let (vind, arena, bbox) = build_tree(&pts, 4);
        let metric = L2;
        let pts: &[[f64; 3]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 3, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let k = 10;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let count;
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 3];
            find_neighbors(&ctx, &mut rs, &query, &SearchParams::default(), &RejectEven, &mut scratch);
            count = rs.size();
        }
        indices.truncate(count);
        dists.truncate(count);

        assert!(indices.iter().all(|&i| i % 2 == 1), "even indices leaked through the filter");

        // Brute force restricted to odd indices only. Uses `metric.eval`
        // (not a hand-rolled sum) so the summation order matches the tree
        // search bit-for-bit — L2's 4-wide-unrolled remainder loop does NOT
        // sum components in index order for dim=3 (see metric.rs), so a
        // naive `(0..3).sum()` can differ by an ULP.
        let mut scored: Vec<(f64, u32)> = (0..pts.len())
            .filter(|&i| i % 2 == 1)
            .map(|i| (metric.eval(&query, &pts, i, 3), i as u32))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        scored.truncate(k);
        let want_indices: Vec<u32> = scored.iter().map(|(_, i)| *i).collect();
        let want_dists: Vec<f64> = scored.iter().map(|(d, _)| *d).collect();

        assert_eq!(indices, want_indices);
        assert_eq!(dists, want_dists);
    }

    // ---------------------------------------------------------------
    // Test 11: SO2 end-to-end
    // ---------------------------------------------------------------

    #[test]
    fn so2_end_to_end_wrapped_angle_wins_over_raw_difference() {
        use crate::metric::SO2;
        let pi = core::f64::consts::PI;
        // dim-2: first component varies wildly (irrelevant to SO2, which
        // only looks at the LAST dim), second component is the angle.
        // Point A: angle = -pi + 0.05 (just past -pi, wraps close to +pi).
        // Point B: angle = pi - 0.5 (raw-closer to query's pi - 0.1 by plain
        // subtraction, but farther by wrapped angular distance).
        let query = [999.0, pi - 0.1];
        let pts: Vec<[f64; 2]> = vec![
            [12345.0, -pi + 0.05], // wrapped dist to query: pi-0.1 -> -pi+0.05 wraps to ~0.15
            [-6789.0, pi - 0.5],   // raw dist: 0.4 (no wrap needed)
        ];

        // Sanity: point 0's wrapped distance (~0.15) < point 1's distance (0.4).
        let metric = SO2;
        let d0 = metric.accum_dist(query[1], pts[0][1], 1);
        let d1 = metric.accum_dist(query[1], pts[1][1], 1);
        assert!(d0 < d1, "test setup: expected point 0 to be the wrapped-nearer angle, d0={d0} d1={d1}");

        let (vind, arena, bbox) = build_tree(&pts, 10);
        let pts: &[[f64; 2]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 2, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let mut indices = [0u32; 1];
        let mut dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 2];
            find_neighbors(&ctx, &mut rs, &query, &SearchParams::default(), &AcceptAll, &mut scratch);
        }
        assert_eq!(indices, [0], "wrapped-nearer angle should win, first component ignored");
    }

    // ---------------------------------------------------------------
    // Test 4: eps
    // ---------------------------------------------------------------

    #[test]
    fn eps_large_visits_only_one_leaf() {
        use core::cell::Cell;
        struct CountingFilter(Cell<usize>);
        impl PointFilter<u32> for CountingFilter {
            fn is_active(&self, _idx: u32) -> bool {
                self.0.set(self.0.get() + 1);
                true
            }
        }

        // 8 well-separated points, leaf_max_size = 1 so a "leaf" == one point:
        // with eps=10 (eps_error=11), the huge inter-point gaps make
        // `mindist * 11 <= worst_dist` false at every interior node except
        // along the single best-child descent path, so exactly ONE leaf (one
        // point) gets visited.
        let pts: Vec<[f64; 1]> = vec![[-300.0], [-200.0], [-100.0], [-0.1], [0.1], [100.0], [200.0], [300.0]];
        let (vind, arena, bbox) = build_tree(&pts, 1);
        let metric = L2;
        let pts: &[[f64; 1]] = &pts;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 1, nodes: &arena, vind: &vind, root_bbox: &bbox };

        let filter = CountingFilter(Cell::new(0));
        let mut indices = [0u32; 1];
        let mut dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);
            let mut scratch = [0.0f64; 1];
            let params = SearchParams { eps: 10.0, sorted: true };
            find_neighbors(&ctx, &mut rs, &[0.05], &params, &filter, &mut scratch);
        }
        assert_eq!(filter.0.get(), 1, "eps=10 should visit exactly one leaf (one point)");
        assert_eq!(indices, [4], "should still find the nearest point in the visited leaf");
    }

    // BRIEF DISCREPANCY, resolved by proof (see task-8-report.md for the
    // full write-up): the brief asks for "a crafted dim-1 dataset where the
    // pruned branch holds the true nearest" to exercise the eps-approximate
    // bound for k=1. This is UNCONSTRUCTIBLE in dim-1.
    //
    // Proof sketch: at any interior node, `div_low` is the ACTUAL max
    // coordinate of a real point in the left subtree (build.rs's
    // `finalizeSplitNode` sets it from the left subtree's own tight bbox,
    // not merely a bound), and `div_high` is likewise the real min of the
    // right subtree. The heuristic condition `(diff1+diff2) < 0` is
    // algebraically identical to `|val - div_low| < |val - div_high|`
    // (expand both cases: `val` inside the [div_low,div_high] gap, or
    // outside it on either side — the two forms agree in every case). Since
    // `div_low`/`div_high` are themselves points reachable inside their
    // respective subtrees, `TRUE_MIN(subtree) <= dist(val, its own
    // boundary)` always (the boundary point is a candidate, so it can only
    // be beaten or tied, never exceeded, by the subtree's true minimum).
    // Chaining: `TRUE_MIN(chosen_best) <= |val-div_low or div_high| <=
    // |val - other's boundary| = TRUE_MIN(other)`. So whichever child the
    // heuristic labels "best" PROVABLY has a true minimum <= the other
    // child's true minimum, at EVERY node, all the way to a leaf — meaning
    // pure greedy best-child descent (no "other" branch ever needed) already
    // finds the exact global 1-NN in 1D, for ANY eps (the "other" branch
    // visit is exact-search-only insurance for ties, never a source of
    // improvement). Confirmed empirically: 20,000 random dim-1 trials
    // (leaf_max_size in {1,2}, n up to 48, uniform query range) never
    // produced a single case where eps=0.5's k=1 answer differed from
    // eps=0's.
    //
    // The mechanism DOES exist in dim>=2, because `mindist` there is a
    // SINGLE-AXIS bound (the split axis's `accum_dist` only) while
    // `worst_dist` is the FULL multi-axis distance — so `mindist` is a
    // valid but genuinely non-tight lower bound, and a point in the "other"
    // branch can have a small full-distance despite the pruned axis alone
    // suggesting otherwise. This test hand-derives such a dim-2 case (bypasses
    // the builder for exact control over `div_low`/`div_high`, same
    // technique as `radius_sorted_false_gives_tree_traversal_order`).
    //
    // Construction (all values exact in f64, verified by hand below and by
    // running the resulting assertions): points P0=(0.35, 0.3) ["found"/best
    // child, visited unconditionally first], P1=(-0.3, 0.05) ["other"/pruned
    // child — the TRUE nearest]. Split axis 0, div_low=-0.3 (=P1's own x,
    // tight), div_high=0.35 (=P0's own x, tight). Query=(0.06, 0.0).
    //
    // Heuristic: diff1=val-div_low=0.06-(-0.3)=0.36; diff2=val-div_high=
    // 0.06-0.35=-0.29; sum=0.07 >= 0 -> best=P0 (child2), matching the
    // labeling above.
    //
    // Exact (eps=0): visit P0 first, worst_dist=(0.06-0.35)^2+(0-0.3)^2=
    // 0.0841+0.09=0.1741. cut_dist=accum_dist(0.06,-0.3,axis0)=(0.36)^2=
    // 0.1296=mindist. eps=0 -> 0.1296*1<=0.1741 -> visit P1: true dist=
    // (0.06+0.3)^2+(0-0.05)^2=0.1296+0.0025=0.1321 < 0.1741 -> UPDATES,
    // exact result = P1, dist 0.1321 (the true global nearest).
    //
    // eps=0.5 (eps_error=1.5): after P0, worst_dist=0.1741, mindist=0.1296.
    // 0.1296*1.5=0.1944 > 0.1741 -> PRUNED, P1 never visited. Approx result
    // = P0, dist 0.1741.
    //
    // Bound check: 0.1741 <= 1.5 * 0.1321 = 0.19815 -- holds, with margin.
    #[test]
    fn eps_moderate_bounds_approximate_error_dim2_hand_derived() {
        let pts: Vec<[f64; 2]> = vec![[0.35, 0.3], [-0.3, 0.05]]; // index0=P0, index1=P1
        let pts: &[[f64; 2]] = &pts;

        // vind = [1, 0]: leaf-left (arena idx 1) = vind[0..1] = [1] (P1);
        // leaf-right (arena idx 2) = vind[1..2] = [0] (P0).
        let vind: Vec<u32> = vec![1, 0];
        let mut arena: Vec<Node<f64>> = Vec::new();
        arena.push(Node::split(0, -0.3, 0.35)); // root, will patch children below
        arena.push(Node::leaf(0, 1)); // left leaf: P1
        arena.push(Node::leaf(1, 2)); // right leaf: P0
        {
            // Node has no public setter accessible from outside build.rs's
            // module boundary other than what's already pub(crate); reuse
            // the same crate-private API the builder itself uses.
            let root = &mut arena[0];
            root.set_children(1, 2);
        }

        let metric = L2;
        let ctx = SearchCtx { ds: &pts, metric: &metric, dim: 2, nodes: &arena, vind: &vind, root_bbox: &[] };
        let query = [0.06f64, 0.0];

        // Exact (eps_error = 1.0): bypass find_neighbors/compute_initial_distances
        // entirely (its bbox-based initial mindist isn't part of this
        // hand-derivation, which assumes mindist starts at 0 exactly as if
        // query were inside the root bbox on every axis) — call
        // `search_level` directly with mindist=0, dists=[0,0].
        let mut exact_indices = [0u32; 1];
        let mut exact_dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut exact_indices, &mut exact_dists);
            let mut dists = [0.0f64, 0.0];
            search_level(&ctx, &mut rs, &query, 0, 0.0f64, &mut dists, 1.0f64, &AcceptAll);
        }
        assert_eq!(exact_indices, [1], "exact search should find P1 (the true nearest)");
        assert!((exact_dists[0] - 0.1321).abs() < 1e-9, "exact_dists[0]={}", exact_dists[0]);

        // Approximate (eps=0.5 -> eps_error = 1.5).
        let mut approx_indices = [0u32; 1];
        let mut approx_dists = [0.0f64; 1];
        {
            let mut rs = KnnResultSet::<f64, u32>::new(&mut approx_indices, &mut approx_dists);
            let mut dists = [0.0f64, 0.0];
            search_level(&ctx, &mut rs, &query, 0, 0.0f64, &mut dists, 1.5f64, &AcceptAll);
        }
        assert_eq!(approx_indices, [0], "eps=0.5 should have pruned P1 and kept the (suboptimal) P0");
        assert!((approx_dists[0] - 0.1741).abs() < 1e-9, "approx_dists[0]={}", approx_dists[0]);

        // The eps-error contract: returned distance <= (1+eps) * true nearest.
        let true_best = exact_dists[0];
        let got = approx_dists[0];
        assert!(got > true_best, "eps=0.5 result should be strictly worse than exact here (that's the point)");
        assert!(
            got <= 1.5 * true_best + 1e-9,
            "approx dist {got} exceeds (1+eps)*true_best {}",
            1.5 * true_best
        );
    }
}
