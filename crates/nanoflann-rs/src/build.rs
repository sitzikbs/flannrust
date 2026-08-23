//! Sequential, explicit-stack tree builder — a faithful port of nanoflann's
//! `divideTree` / `middleSplit_` / `planeSplit` (nanoflann.hpp, see the
//! per-function doc comments below for the exact source lines consulted).
//!
//! The C++ builds the tree with native recursion. Adversarial inputs (e.g.
//! near-sorted or geometrically-spaced coordinates) can drive `middleSplit_`
//! to peel off a handful of points per level, making the recursion depth
//! O(n) — enough to blow a 2 MiB worker stack under a future parallel build.
//! We therefore use two explicit `Vec`-backed stacks (a work stack and a
//! result stack) instead: heap-allocated, so depth is bounded only by
//! available memory, not by the OS thread stack. `heavy_exponential_build_1m`
//! exercises this directly.
//!
//! Every control-flow choice below (comparison operators, swap order,
//! std::min/std::max argument order) is called out against the C++ source it
//! mirrors, because nanoflann-rs's contract is bit-identical trees, not just
//! "a valid kd-tree".

use crate::bbox::Interval;
use crate::data_source::DataSource;
use crate::node::Node;
use crate::scalar::{IndexType, Scalar};

/// Identity permutation `0..n` (C++ `init_vind`, nanoflann.hpp:2174-2180:
/// `for (IndexType i = 0; i < size_; i++) vAcc_[i] = i;`). Panics if
/// `n > u32::MAX as usize` (leaf offsets are stored as `u32` in `Node`).
pub(crate) fn init_vind<Idx: IndexType>(n: usize) -> Vec<Idx> {
    assert!(
        n <= u32::MAX as usize,
        "init_vind: point count {n} exceeds u32::MAX (leaf offsets are u32)"
    );
    (0..n).map(Idx::from_usize).collect()
}

/// `std::min(a, b)` semantics exactly (nanoflann.hpp:1380 uses `std::min`,
/// not a raw `<` compare): `(b < a) ? b : a` — returns `a` on a tie.
/// `pub(crate)`: also used by `build_parallel.rs`'s bbox-union merge step,
/// which must match this sequential union bit-for-bit.
#[inline]
pub(crate) fn cpp_min<T: Scalar>(a: T, b: T) -> T {
    if b < a {
        b
    } else {
        a
    }
}

/// `std::max(a, b)` semantics exactly (nanoflann.hpp:1381): `(a < b) ? b : a`
/// — returns `a` on a tie. `pub(crate)`: see `cpp_min`.
#[inline]
pub(crate) fn cpp_max<T: Scalar>(a: T, b: T) -> T {
    if a < b {
        b
    } else {
        a
    }
}

/// Plain min/max scan over `ind` on dimension `dim` (nanoflann's
/// `computeMinMax`, nanoflann.hpp:1193-1205). The C++ has a 4-way-unrolled
/// variant inline inside `middleSplit_` (nanoflann.hpp:1510-1530) for
/// performance, but a plain sequential loop computes the identical min/max
/// (unrolling only reorders independent `<`/`>` comparisons — associative
/// and commutative for min/max, so the result is bit-identical regardless of
/// grouping; no NaNs are in play here).
fn compute_min_max<T, DS, Idx>(ds: &DS, ind: &[Idx], dim: usize) -> (T, T)
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    let mut min_elem = ds.point_component(ind[0].to_usize(), dim);
    let mut max_elem = min_elem;
    for &i in &ind[1..] {
        let v = ds.point_component(i.to_usize(), dim);
        if v < min_elem {
            min_elem = v;
        }
        if v > max_elem {
            max_elem = v;
        }
    }
    (min_elem, max_elem)
}

/// Three-way partition of `ind` around `cutval` on `cutfeat`:
/// on return `ind[..lim1]` < cutval <= `ind[lim1..lim2]` == cutval < `ind[lim2..]`.
/// Returns `(lim1, lim2)`.
///
/// Port of nanoflann's `planeSplit` (nanoflann.hpp:1565-1597). NOTE: the C++
/// here is a single-pass Dutch National Flag partition with three cursors
/// (`left`, `mid`, `right`), NOT the two-Hoare-sweep algorithm this task's
/// brief describes — the brief's prose is stale relative to
/// `nanoflann-ref/cpp/nanoflann.hpp` (vendored 1.12.1); per the task
/// instructions we port the algorithm actually in the source. Exact C++:
/// ```cpp
/// Offset left = 0, mid = 0, right = count - 1;
/// while (mid <= right) {
///     val = dataset_get(vAcc_[ind + mid], cutfeat);
///     if (val < cutval)      { swap(vAcc_[ind+left], vAcc_[ind+mid]); left++; mid++; }
///     else if (val > cutval) { swap(vAcc_[ind+mid], vAcc_[ind+right]); right--; }
///     else                   { mid++; }
/// }
/// lim1 = left; lim2 = mid;
/// ```
/// `right` is C++ `Offset` (unsigned) and can in principle wrap past 0 if the
/// very last surviving element is `> cutval` (`right-- ` from 0). We translate
/// `right` as `isize` so the loop condition `mid <= right` is checked with
/// ordinary signed comparison instead of relying on unsigned wraparound —
/// same swap sequence for every input where `right` never needs to go
/// negative in the C++ (which is every input `middleSplit_` ever produces,
/// since `cutval` is clamped to `[min_elem, max_elem]` and therefore always
/// has at least one element `<= cutval`), and no UB for the pathological
/// inputs (e.g. `count == 1` with a caller-chosen `cutval` less than the sole
/// element) that a raw unsigned `right` would mishandle.
pub(crate) fn plane_split<T, DS, Idx>(
    ds: &DS,
    ind: &mut [Idx],
    cutfeat: usize,
    cutval: T,
) -> (usize, usize)
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    let count = ind.len();
    let mut left: usize = 0;
    let mut mid: usize = 0;
    let mut right: isize = count as isize - 1;

    while (mid as isize) <= right {
        let val = ds.point_component(ind[mid].to_usize(), cutfeat);
        if val < cutval {
            ind.swap(left, mid);
            left += 1;
            mid += 1;
        } else if val > cutval {
            ind.swap(mid, right as usize);
            right -= 1;
        } else {
            mid += 1;
        }
    }

    (left, mid)
}

/// Returns `(index, cutfeat, cutval)`; permutes `ind` (via `plane_split`).
///
/// Port of nanoflann's `middleSplit_` (nanoflann.hpp:1481-1554).
// The `max_span`/candidate-dims loops below are explicit `for d in
// 0..dim`/`1..dim` ranges (not `bbox.iter()`) to mirror the C++ source's
// exact per-axis loop structure (nanoflann.hpp:1489-1494, 1503-1540) --
// `d` is also threaded into `compute_min_max(ds, ind, d)`, not just used to
// index `bbox`, so an iterator-only rewrite wouldn't stay a clean 1:1 port.
#[allow(clippy::needless_range_loop)]
pub(crate) fn middle_split<T, DS, Idx>(
    ds: &DS,
    dim: usize,
    ind: &mut [Idx],
    bbox: &[Interval<T>],
) -> (usize, u32, T)
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    let count = ind.len();
    // nanoflann.hpp:1486: `const auto EPS = static_cast<DistanceType>(0.00001);`
    // — the C++ names this constant (and `split_val` below) `DistanceType`,
    // but computes it from `ElementType` inputs (`bbox`'s bounds, `dim`'s
    // min/max) and only ever compares it against other `ElementType`
    // values, so it's really element-type arithmetic wearing a
    // `DistanceType` label. This port computes `eps`/`split_val` in `T`
    // (the element/coordinate type) throughout, which is bit-identical to
    // the C++ for every metric this crate ships (`L1`/`L2`/`L2Simple`/
    // `SO2`/`SO3` all set `Distance::DistanceType = T`) — this function has
    // no way to observe a metric's `DistanceType` at all (it isn't generic
    // over `Distance`), so if a future custom metric ever set
    // `DistanceType != T`, that distinction still wouldn't matter here: the
    // TREE STRUCTURE this function builds is metric-independent.
    let eps = T::from_f64(0.00001);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);

    // nanoflann.hpp:1489-1494: pre-compute max_span once over all dims.
    let mut max_span = bbox[0].high - bbox[0].low;
    for d in 1..dim {
        let span = bbox[d].high - bbox[d].low;
        if span > max_span {
            max_span = span;
        }
    }

    // nanoflann.hpp:1498-1501.
    let mut cutfeat: usize = 0;
    let mut max_spread = T::from_f64(-1.0);
    let mut min_elem = T::default();
    let mut max_elem = T::default();
    let threshold = (one - eps) * max_span;

    // nanoflann.hpp:1503-1540: candidate dims are those NOT skipped, i.e.
    // `bbox[dim].high - bbox[dim].low >= threshold` (source uses
    // `if (span < threshold) continue;`, so the candidate condition is `>=`,
    // not the brief's suggested strict `>`).
    for d in 0..dim {
        if bbox[d].high - bbox[d].low < threshold {
            continue;
        }
        let (local_min, local_max) = compute_min_max(ds, ind, d);
        let spread = local_max - local_min;
        // nanoflann.hpp:1533: strictly greater — first candidate dim wins ties.
        if spread > max_spread {
            cutfeat = d;
            max_spread = spread;
            min_elem = local_min;
            max_elem = local_max;
        }
    }

    // nanoflann.hpp:1543-1547: median-of-bbox, clamped into the actual data range.
    let mut split_val = (bbox[cutfeat].low + bbox[cutfeat].high) / two;
    if split_val < min_elem {
        split_val = min_elem;
    }
    if split_val > max_elem {
        split_val = max_elem;
    }
    let cutval = split_val;

    let (lim1, lim2) = plane_split(ds, ind, cutfeat, cutval);

    // nanoflann.hpp:1553.
    let half = count / 2;
    let index = if lim1 > half {
        lim1
    } else if lim2 < half {
        lim2
    } else {
        half
    };

    (index, cutfeat as u32, cutval)
}

/// Everything the builder needs to know, borrowed. `vind` is the sub-range of
/// the permuted index vector this call owns (the whole vector for a full
/// build; disjoint sub-slices in the parallel build), `base` its offset in
/// the global vector: leaf nodes store global offsets `base + local`.
pub(crate) struct SubtreeBuilder<'a, T: Scalar, DS: DataSource<T> + ?Sized, Idx: Copy> {
    pub ds: &'a DS,
    pub dim: usize,
    pub leaf_max_size: usize,
    pub base: u32,
    pub vind: &'a mut [Idx],
    pub arena: &'a mut Vec<Node<T>>,
}

/// Explicit work-stack item, replacing the C++ `divideTree` call frame.
enum Work<T> {
    /// Build the subtree over `vind[left..right)` whose (loose, on entry)
    /// bbox is `bbox`.
    Build {
        left: usize,
        right: usize,
        bbox: Vec<Interval<T>>,
    },
    /// Both children's results are on the result stack (right on top).
    Finalize { node: u32, cutfeat: usize },
}

impl<'a, T, DS, Idx> SubtreeBuilder<'a, T, DS, Idx>
where
    T: Scalar,
    DS: DataSource<T> + ?Sized,
    Idx: IndexType,
{
    /// Build the subtree over all of `self.vind`, whose bounding box is
    /// `bbox` (mutated to the TIGHT box of the actual points, exactly like
    /// C++ `divideTree(..., BoundingBox& bbox)`). Returns the arena index of
    /// the subtree root. `vind` must be non-empty.
    ///
    /// Port of nanoflann's `divideTree` + `makeNode` + `finalizeSplitNode`
    /// (nanoflann.hpp:1328-1408), restructured from recursion into two
    /// explicit `Vec` stacks (see module doc). `divideTreeConcurrent`
    /// (nanoflann.hpp:1422-1479) was read too, to confirm it partitions
    /// before spawning and calls the same `makeNode`/`finalizeSplitNode`
    /// helpers — i.e. it produces an identical tree to `divideTree`, just
    /// built with threads; nothing in it changes how a *sequential* subtree
    /// must be built.
    // The leaf tight-bbox loops below index `sub_bbox`/`self.vind` by an
    // explicit `d`/`k` range (not an iterator over `sub_bbox` alone) to
    // mirror nanoflann.hpp:1341-1355's exact per-axis structure.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn build(&mut self, bbox: &mut [Interval<T>]) -> u32 {
        assert!(
            !self.vind.is_empty(),
            "SubtreeBuilder::build called with empty vind"
        );
        debug_assert_eq!(bbox.len(), self.dim);

        // Reused pool of `Vec<Interval<T>>` scratch buffers (audit SF-2):
        // every bbox buffer this builder ever touches is exactly `self.dim`
        // long, so once one is no longer needed its ALLOCATION can be
        // handed to whichever site needs a fresh buffer next instead of
        // being freed and a new one malloc'd. The two allocation sites this
        // replaces (one `Vec::clone()` per interior split, for `left_bbox`;
        // one `Vec::with_capacity` per interior finalize, for
        // `parent_bbox`) are exactly balanced by the two release sites
        // below (`right_bbox` returned to the pool at Finalize;
        // `left_bbox`'s allocation reused IN PLACE as `parent_bbox`) — pool
        // depth stays bounded by tree depth, never grows unbounded. Values
        // are identical either way; only which heap allocation backs a
        // given bbox buffer changes.
        let mut bbox_pool: Vec<Vec<Interval<T>>> = Vec::new();

        let mut work: Vec<Work<T>> = vec![Work::Build {
            left: 0,
            right: self.vind.len(),
            bbox: bbox.to_vec(),
        }];
        // Each result is the TIGHT bbox of the subtree just finished.
        let mut results: Vec<(u32, Vec<Interval<T>>)> = Vec::new();

        while let Some(item) = work.pop() {
            match item {
                Work::Build {
                    left,
                    right,
                    bbox: mut sub_bbox,
                } => {
                    let count = right - left;
                    // nanoflann.hpp:1335: leaf when `(right - left) <=
                    // leaf_max_size_` — count EQUAL to leaf_max_size is a leaf.
                    if count <= self.leaf_max_size {
                        debug_assert!(left <= u32::MAX as usize && right <= u32::MAX as usize);
                        let node_idx = self.arena.len() as u32;
                        self.arena.push(Node::leaf(
                            self.base + left as u32,
                            self.base + right as u32,
                        ));

                        // nanoflann.hpp:1341-1355: tight bbox of the leaf's
                        // actual points, overwriting whatever loose bbox we
                        // were handed.
                        for d in 0..self.dim {
                            let v = self.ds.point_component(self.vind[left].to_usize(), d);
                            sub_bbox[d] = Interval { low: v, high: v };
                        }
                        for k in (left + 1)..right {
                            for d in 0..self.dim {
                                let v = self.ds.point_component(self.vind[k].to_usize(), d);
                                if sub_bbox[d].low > v {
                                    sub_bbox[d].low = v;
                                }
                                if sub_bbox[d].high < v {
                                    sub_bbox[d].high = v;
                                }
                            }
                        }
                        results.push((node_idx, sub_bbox));
                    } else {
                        let (split_index, cutfeat, cutval) =
                            middle_split(self.ds, self.dim, &mut self.vind[left..right], &sub_bbox);
                        let cutfeat = cutfeat as usize;

                        let node_idx = self.arena.len() as u32;
                        // Bounds are patched by the matching Finalize once
                        // both children are known.
                        self.arena.push(Node::split(cutfeat as u32, T::default(), T::default()));

                        // Take a pooled buffer for `left_bbox` when one is
                        // available (overwriting its stale contents with
                        // `sub_bbox`'s CURRENT values — identical to what
                        // `sub_bbox.clone()` would produce), else fall back
                        // to allocating; `right_bbox` reuses `sub_bbox`'s
                        // own allocation directly (moved, never cloned,
                        // exactly as before).
                        let mut left_bbox = match bbox_pool.pop() {
                            Some(mut buf) => {
                                buf.clear();
                                buf.extend_from_slice(&sub_bbox);
                                buf
                            }
                            None => sub_bbox.clone(),
                        };
                        left_bbox[cutfeat].high = cutval;
                        let mut right_bbox = sub_bbox;
                        right_bbox[cutfeat].low = cutval;

                        // Push order: Finalize first (bottom), then the
                        // right subtree, then the left subtree on top so the
                        // left subtree is popped (and thus built) first —
                        // exactly C++'s recursion order (left recursion
                        // happens before right, nanoflann.hpp:1398-1403), so
                        // the arena fills in the same pre-order as C++'s pool
                        // allocator.
                        work.push(Work::Finalize {
                            node: node_idx,
                            cutfeat,
                        });
                        work.push(Work::Build {
                            left: left + split_index,
                            right,
                            bbox: right_bbox,
                        });
                        work.push(Work::Build {
                            left,
                            right: left + split_index,
                            bbox: left_bbox,
                        });
                    }
                }
                Work::Finalize { node, cutfeat } => {
                    let (right_node, right_bbox) = results.pop().expect("right result missing");
                    let (left_node, mut left_bbox) = results.pop().expect("left result missing");

                    self.arena[node as usize].set_children(left_node, right_node);
                    // nanoflann.hpp:1374-1375.
                    self.arena[node as usize]
                        .set_div_bounds(left_bbox[cutfeat].high, right_bbox[cutfeat].low);

                    // nanoflann.hpp:1377-1382: union via std::min/std::max
                    // (argument order: left first, right second). Written
                    // IN PLACE into `left_bbox`'s own allocation (values
                    // identical to the `parent_bbox: Vec::with_capacity`
                    // this replaces) — `left_bbox[d]` on the right-hand
                    // side is read before that same slot is overwritten, so
                    // this is a safe elementwise transform, not a
                    // read-after-write hazard. `right_bbox`'s now-unneeded
                    // allocation returns to the pool for reuse.
                    for d in 0..self.dim {
                        left_bbox[d] = Interval {
                            low: cpp_min(left_bbox[d].low, right_bbox[d].low),
                            high: cpp_max(left_bbox[d].high, right_bbox[d].high),
                        };
                    }
                    bbox_pool.push(right_bbox);
                    results.push((node, left_bbox));
                }
            }
        }

        let (root_node, root_bbox) = results.pop().expect("build produced no result");
        debug_assert!(results.is_empty());
        bbox.copy_from_slice(&root_bbox);
        root_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::compute_bounding_box;

    // ---------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------

    /// Build a full tree over `points` (T = f64) via `SubtreeBuilder`, base 0,
    /// identity `vind`, root bbox from `compute_bounding_box`.
    fn build_full_tree<const N: usize>(
        points: &[[f64; N]],
        leaf_max_size: usize,
    ) -> (Vec<Node<f64>>, Vec<u32>, u32, Vec<Interval<f64>>) {
        let dim = N;
        let n = points.len();
        let mut vind: Vec<u32> = init_vind(n);
        let mut bbox = vec![Interval { low: 0.0, high: 0.0 }; dim];
        compute_bounding_box(&points, dim, &mut bbox);
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

    /// Structural invariant checker (iterative — safe for very deep trees).
    /// Walks the arena from `root`:
    /// - every point index in `0..points.len()` appears in exactly one leaf
    ///   (`vind` is a permutation restricted to the tree's points);
    /// - for every interior node, `div_low` EXACTLY equals the max of the
    ///   left subtree's `cutfeat` coordinate, and `div_high` EXACTLY equals
    ///   the min of the right subtree's `cutfeat` coordinate (this is what
    ///   `finalizeSplitNode` computes; a weaker `<=`/`>=` check would miss a
    ///   builder that unions bboxes incorrectly);
    /// - node count <= 2n.
    ///
    /// Computes each subtree's per-dimension bbox bottom-up (post-order,
    /// two-stack technique) instead of rescanning from scratch at every
    /// ancestor, so the whole check is O(n) total — required to stay fast
    /// enough for the 1M-point heavy test.
    fn check_tree<const N: usize>(
        points: &[[f64; N]],
        arena: &[Node<f64>],
        vind: &[u32],
        root: u32,
        leaf_max_size: usize,
    ) {
        let dim = N;
        let n = points.len();
        assert!(
            arena.len() <= 2 * n,
            "node count {} exceeds 2n ({})",
            arena.len(),
            2 * n
        );

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
                        assert!(
                            r - l <= leaf_max_size,
                            "leaf [{l},{r}) has {} points > leaf_max_size {leaf_max_size}",
                            r - l
                        );
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

                    assert_eq!(
                        left_bbox[cutfeat].high,
                        node.div_low(),
                        "node {idx}: div_low != max of left subtree's coord[{cutfeat}]"
                    );
                    assert_eq!(
                        right_bbox[cutfeat].low,
                        node.div_high(),
                        "node {idx}: div_high != min of right subtree's coord[{cutfeat}]"
                    );

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

        assert!(
            seen.iter().all(|&b| b),
            "not every point index appears in a leaf"
        );
        assert_eq!(results.len(), 1);
    }

    /// Iterative max-depth (root = depth 1).
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

    // ---------------------------------------------------------------
    // 1. plane_split postcondition property
    // ---------------------------------------------------------------

    fn assert_plane_split_postcondition(values: &[f64], cutval: f64) {
        let points: Vec<[f64; 1]> = values.iter().map(|&v| [v]).collect();
        let mut ind: Vec<u32> = (0..values.len() as u32).collect();
        let original: Vec<u32> = ind.clone();

        let (lim1, lim2) = plane_split(&points.as_slice(), &mut ind, 0, cutval);

        for &i in &ind[..lim1] {
            assert!(points[i as usize][0] < cutval, "ind[..lim1] violated");
        }
        for &i in &ind[lim1..lim2] {
            assert_eq!(points[i as usize][0], cutval, "ind[lim1..lim2] violated");
        }
        for &i in &ind[lim2..] {
            assert!(points[i as usize][0] > cutval, "ind[lim2..] violated");
        }

        // permutation check
        let mut sorted_ind = ind.clone();
        sorted_ind.sort_unstable();
        let mut sorted_orig = original.clone();
        sorted_orig.sort_unstable();
        assert_eq!(sorted_ind, sorted_orig, "ind is not a permutation of input");
    }

    #[test]
    fn plane_split_postcondition_with_duplicates() {
        assert_plane_split_postcondition(&[5.0, 3.0, 5.0, 1.0, 5.0, 8.0, 1.0, 5.0], 5.0);
    }

    #[test]
    fn plane_split_postcondition_all_less() {
        assert_plane_split_postcondition(&[1.0, 2.0, 3.0, 4.0], 10.0);
    }

    #[test]
    fn plane_split_postcondition_all_greater() {
        assert_plane_split_postcondition(&[11.0, 12.0, 13.0, 14.0], 10.0);
    }

    #[test]
    fn plane_split_postcondition_all_equal() {
        assert_plane_split_postcondition(&[7.0, 7.0, 7.0, 7.0, 7.0], 7.0);
    }

    #[test]
    fn plane_split_postcondition_single_element() {
        assert_plane_split_postcondition(&[42.0], 42.0);
    }

    // ---------------------------------------------------------------
    // 2. plane_split exact permutation — hand-simulated swap-for-swap parity
    // ---------------------------------------------------------------

    /// Hand derivation of C++ `planeSplit` (nanoflann.hpp:1565-1597) on
    /// values `v = [5, 3, 5, 1, 5, 8, 1, 5]` (indexed by point index 0..8),
    /// `cutfeat = 0`, `cutval = 5`. Initial `ind = [0,1,2,3,4,5,6,7]`.
    ///
    /// left=0 mid=0 right=7
    /// mid=0: v[ind[0]]=v[0]=5 == cutval        -> mid=1                          ind unchanged
    /// mid=1: v[ind[1]]=v[1]=3  < cutval        -> swap(0,1); left=1; mid=2       ind=[1,0,2,3,4,5,6,7]
    /// mid=2: v[ind[2]]=v[2]=5 == cutval        -> mid=3                          ind unchanged
    /// mid=3: v[ind[3]]=v[3]=1  < cutval        -> swap(1,3); left=2; mid=4       ind=[1,3,2,0,4,5,6,7]
    /// mid=4: v[ind[4]]=v[4]=5 == cutval        -> mid=5                          ind unchanged
    /// mid=5: v[ind[5]]=v[5]=8  > cutval        -> swap(5,7); right=6             ind=[1,3,2,0,4,7,6,5]
    /// mid=5: v[ind[5]]=v[7]=5 == cutval        -> mid=6                          ind unchanged
    /// mid=6: v[ind[6]]=v[6]=1  < cutval        -> swap(2,6); left=3; mid=7       ind=[1,3,6,0,4,7,2,5]
    /// mid=7 > right=6                          -> loop ends
    ///
    /// lim1 = left = 3, lim2 = mid = 7.
    /// Final ind = [1, 3, 6, 0, 4, 7, 2, 5].
    #[test]
    fn plane_split_exact_permutation_hand_simulated() {
        let values = [5.0f64, 3.0, 5.0, 1.0, 5.0, 8.0, 1.0, 5.0];
        let points: Vec<[f64; 1]> = values.iter().map(|&v| [v]).collect();
        let mut ind: Vec<u32> = (0..8).collect();

        let (lim1, lim2) = plane_split(&points.as_slice(), &mut ind, 0, 5.0);

        assert_eq!(ind, vec![1, 3, 6, 0, 4, 7, 2, 5]);
        assert_eq!((lim1, lim2), (3, 7));
    }

    // ---------------------------------------------------------------
    // 3. middle_split rules
    // ---------------------------------------------------------------

    #[test]
    fn middle_split_picks_second_dim_when_spans_tie_but_spread_larger() {
        // bbox: both dims [0, 10], span 10 == max_span -> both candidates.
        // Actual points: dim0 spread = 6 (2..8), dim1 spread = 8 (1..9).
        let points: &[[f64; 2]] = &[[2.0, 1.0], [8.0, 1.0], [2.0, 9.0], [8.0, 9.0]];
        let mut ind: Vec<u32> = (0..4).collect();
        let bbox = [
            Interval { low: 0.0, high: 10.0 },
            Interval { low: 0.0, high: 10.0 },
        ];

        let (index, cutfeat, cutval) = middle_split(&points, 2, &mut ind, &bbox);

        assert_eq!(cutfeat, 1, "should pick dim 1 (larger actual spread)");
        assert_eq!(cutval, 5.0);
        assert_eq!(index, 2);
    }

    #[test]
    fn middle_split_clamps_split_val_to_min_elem() {
        // bbox [0, 100] but points clustered high: 80, 81, 82.
        // split_val = 50 < min_elem(80) -> clamped to 80.
        let points: &[[f64; 1]] = &[[80.0], [81.0], [82.0]];
        let mut ind: Vec<u32> = (0..3).collect();
        let bbox = [Interval { low: 0.0, high: 100.0 }];

        let (index, cutfeat, cutval) = middle_split(&points, 1, &mut ind, &bbox);

        assert_eq!(cutfeat, 0);
        assert_eq!(cutval, 80.0);
        assert_eq!(index, 1);
    }

    #[test]
    fn middle_split_lim1_greater_than_half_selects_lim1() {
        // values [1,1,1,1,5,9]; bbox [1,9]; split_val = 5.
        // plane_split -> lim1=4 (four 1's), lim2=5 (one 5). half = 3.
        // lim1(4) > half(3) -> index = lim1 = 4.
        let points: &[[f64; 1]] = &[[1.0], [1.0], [1.0], [1.0], [5.0], [9.0]];
        let mut ind: Vec<u32> = (0..6).collect();
        let bbox = [Interval { low: 1.0, high: 9.0 }];

        let (index, cutfeat, cutval) = middle_split(&points, 1, &mut ind, &bbox);

        assert_eq!(cutfeat, 0);
        assert_eq!(cutval, 5.0);
        assert_eq!(index, 4);
    }

    #[test]
    fn middle_split_lim2_less_than_half_selects_lim2() {
        // values [1,9,9,9,9,9]; bbox [1,9]; split_val = 5.
        // plane_split -> lim1=1 (one 1), lim2=1 (no elements == 5). half = 3.
        // lim1(1) > half(3)? no. lim2(1) < half(3)? yes -> index = lim2 = 1.
        let points: &[[f64; 1]] = &[[1.0], [9.0], [9.0], [9.0], [9.0], [9.0]];
        let mut ind: Vec<u32> = (0..6).collect();
        let bbox = [Interval { low: 1.0, high: 9.0 }];

        let (index, cutfeat, cutval) = middle_split(&points, 1, &mut ind, &bbox);

        assert_eq!(cutfeat, 0);
        assert_eq!(cutval, 5.0);
        assert_eq!(index, 1);
    }

    // ---------------------------------------------------------------
    // 4. Leaf-equal boundary
    // ---------------------------------------------------------------

    #[test]
    fn leaf_equal_boundary_is_a_single_leaf() {
        let points: Vec<[f64; 2]> = (0..10).map(|i| [i as f64, (i * 2) as f64]).collect();
        let (arena, vind, root, _bbox) = build_full_tree(&points, 10);

        assert_eq!(arena.len(), 1);
        assert!(arena[root as usize].is_leaf());
        assert_eq!(arena[root as usize].leaf_range(), (0, 10));
        assert_eq!(vind.len(), 10);
    }

    // ---------------------------------------------------------------
    // 5. Structural invariant checker on 100 uniform random points
    // ---------------------------------------------------------------

    /// Tiny hand-rolled LCG (no new deps), matching common textbook
    /// constants (Numerical Recipes). Deterministic across runs.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Take the top 53 bits for a uniform-ish [0, 1) double.
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn invariant_holds_on_100_uniform_random_points_dim3() {
        let mut rng = Lcg(0xC0FFEE_u64);
        let points: Vec<[f64; 3]> = (0..100)
            .map(|_| [rng.next_f64() * 100.0, rng.next_f64() * 100.0, rng.next_f64() * 100.0])
            .collect();

        let (arena, vind, root, _bbox) = build_full_tree(&points, 10);
        check_tree(&points, &arena, &vind, root, 10);
    }

    // ---------------------------------------------------------------
    // 6. All-identical points
    // ---------------------------------------------------------------

    #[test]
    fn all_identical_points_build_balanced_tree() {
        let points: Vec<[f64; 3]> = vec![[3.5, -2.0, 7.25]; 1000];
        let (arena, vind, root, _bbox) = build_full_tree(&points, 10);

        check_tree(&points, &arena, &vind, root, 10);

        for node in &arena {
            if node.is_leaf() {
                let (l, r) = node.leaf_range();
                assert!(r - l <= 10);
            } else {
                // All points identical -> gap collapses to the coordinate.
                assert_eq!(node.div_low(), node.div_high());
            }
        }

        let depth = max_depth(&arena, root);
        // log2(1000/10) ~= 6.6; allow generous slack.
        assert!(depth <= 12, "tree too deep for balanced splits: {depth}");
        assert_eq!(vind.len(), 1000);
    }

    // ---------------------------------------------------------------
    // 7. leaf_max_size = 1 and leaf_max_size >= n
    // ---------------------------------------------------------------

    #[test]
    fn leaf_max_size_one_gives_singleton_leaves() {
        let points: Vec<[f64; 2]> = (0..64).map(|i| [i as f64, (i * i) as f64 % 37.0]).collect();
        let (arena, vind, root, _bbox) = build_full_tree(&points, 1);

        check_tree(&points, &arena, &vind, root, 1);
        for node in &arena {
            if node.is_leaf() {
                let (l, r) = node.leaf_range();
                assert_eq!(r - l, 1);
            }
        }
    }

    #[test]
    fn leaf_max_size_at_least_n_gives_single_node() {
        let points: Vec<[f64; 2]> = (0..64).map(|i| [i as f64, (i * i) as f64 % 37.0]).collect();
        let (arena, _vind, root, _bbox) = build_full_tree(&points, 1000);

        assert_eq!(arena.len(), 1);
        assert!(arena[root as usize].is_leaf());
    }

    // ---------------------------------------------------------------
    // 8. n = 1
    // ---------------------------------------------------------------

    #[test]
    fn single_point_tree() {
        let points: Vec<[f64; 3]> = vec![[1.5, -2.5, 3.5]];
        let (arena, vind, root, bbox) = build_full_tree(&points, 10);

        assert_eq!(arena.len(), 1);
        assert!(arena[root as usize].is_leaf());
        assert_eq!(vind, vec![0]);
        assert_eq!(bbox[0], Interval { low: 1.5, high: 1.5 });
        assert_eq!(bbox[1], Interval { low: -2.5, high: -2.5 });
        assert_eq!(bbox[2], Interval { low: 3.5, high: 3.5 });
    }

    // ---------------------------------------------------------------
    // 9. Tight-bbox propagation from a deliberately loose input bbox
    // ---------------------------------------------------------------

    #[test]
    fn build_tightens_a_loose_input_bbox() {
        let points: Vec<[f64; 2]> = vec![[1.0, 5.0], [2.0, 6.0], [3.0, 4.0], [1.5, 5.5]];
        let mut vind: Vec<u32> = init_vind(points.len());
        // Deliberately inflated / loose bbox, unlike the tight bbox
        // `compute_bounding_box` would produce.
        let mut bbox = vec![
            Interval { low: -100.0, high: 100.0 },
            Interval { low: -100.0, high: 100.0 },
        ];
        let mut arena = Vec::new();
        let root = {
            let mut builder = SubtreeBuilder {
                ds: &points.as_slice(),
                dim: 2,
                leaf_max_size: 10,
                base: 0,
                vind: &mut vind,
                arena: &mut arena,
            };
            builder.build(&mut bbox)
        };

        assert!(arena[root as usize].is_leaf());
        assert_eq!(bbox[0], Interval { low: 1.0, high: 3.0 });
        assert_eq!(bbox[1], Interval { low: 4.0, high: 6.0 });
    }

    // ---------------------------------------------------------------
    // 10. Exponential spacing, stack safety (heavy, release-only)
    // ---------------------------------------------------------------

    #[test]
    #[ignore]
    fn heavy_exponential_build_1m() {
        let n = 1_000_000usize;
        // Resolved by direct measurement: neither of two plausible
        // exponential spacings (`0.9999^i`, depth 156; `2.0^(-i/50.0)`,
        // depth 1092) nor the "obvious" harmonic worst-case `1/(n-i)` (depth
        // 38 — it self-corrects because the minority side's size DOUBLES
        // each level once the bbox's stale low bound stops dominating)
        // reach a depth > 10_000 for ONE-DIMENSIONAL data. Worked out why:
        // every one-sided "peel a
        // small chunk, recurse into the big remainder" degenerate chain that
        // `middle_split` can produce on a SINGLE axis is, structurally, a
        // geometric halving of the bbox toward a value anchored at 0 or at
        // a stale inherited bound — and IEEE-754 `f64` has a hard, finite
        // dynamic range on any one axis (~2^1023 down to the smallest
        // denormal ~2^-1074, about 2098 halvings total). That is the
        // ceiling on how many such levels a SINGLE-AXIS, magnitude-based
        // degenerate construction can sustain over `[f64; 1]` data,
        // regardless of formula — confirmed by building the most extreme
        // possible one-axis chain below (full-range halving from ~2^1023
        // down to the smallest positive denormal) and measuring its actual
        // depth, which tops out around ~2100-2200.
        //
        // This is NOT a claim that depth > 10_000 is unreachable in
        // general: `middle_split` always splits the currently-widest axis,
        // so with D roughly-equal-width axes the same per-axis dynamic-range
        // budget can be spent once per axis, round-robin, instead of once
        // total — see `heavy_exponential_build_1m_dim8` below, which
        // reaches depth > 10_000 exactly this way. This test's job is
        // narrower: it is the deepest tree `f64` permits over ONE-dimensional
        // data, and it already proves the point this task cares about — the
        // explicit stack handles a tree ~120x deeper than a balanced tree
        // over 1M points would ever be (~17 levels), with no native
        // recursion involved at all.
        //
        // This spine (each step exactly half the previous, so no rounding
        // noise) peels ~2 points per level once the exact-tie behavior
        // kicks in (successive powers of two land exactly on the bbox
        // midpoint, see the report), giving ~1049 degenerate levels, plus a
        // final balanced remainder over the padding once the spine hits
        // exactly 0.0. The assertion threshold (2_000) sits safely below
        // the measured depth (2115) with margin.
        let mut spine: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(1023);
        while v > 0.0 && spine.len() < n - 1 {
            spine.push(v);
            v /= 2.0;
        }
        let mut values = spine;
        values.resize(n, 0.0);
        let points: Vec<[f64; 1]> = values.into_iter().map(|v| [v]).collect();

        let (arena, vind, root, _bbox) = build_full_tree(&points, 10);

        let depth = max_depth(&arena, root);
        // Measured empirically at 2115 (see comment above); threshold kept
        // well below that with margin.
        assert!(
            depth > 2_000,
            "expected a deeply degenerate tree to exercise stack safety, got depth {depth}"
        );

        check_tree(&points, &arena, &vind, root, 10);
    }

    // ---------------------------------------------------------------
    // 10b. Same idea, 8 dimensions: round-robin the per-axis dynamic-range
    //      budget across axes to exceed the single-axis ceiling above.
    // ---------------------------------------------------------------

    /// `middle_split` always splits the axis with (a) bbox span within
    /// `(1-EPS)` of the widest bbox span, and, among those candidates, (b)
    /// the strictly largest ACTUAL spread (first candidate wins ties). If
    /// all 8 axes start with equal-width bboxes, every axis is always a
    /// candidate by (a); whichever axis currently has the largest actual
    /// spread wins (b). This construction gives axis `d` (`d` = 0..7) its
    /// OWN full-range halving ladder `2^1023, 2^1022, ..., ~2^-1074`
    /// (exactly the ladder `heavy_exponential_build_1m` uses alone), placed
    /// on a round-robin schedule: spine point `k` sets ONLY axis `k % 8` to
    /// `ladder[k / 8]` (all other axes `0.0` for that point). So axis `d`'s
    /// per-round spread is identical in shape to the single-axis case, just
    /// interleaved with the other 7 axes' identical ladders — verified in a
    /// small-scale Python prototype (200 rounds, dim 8, 1800 points) to
    /// pick axes in the exact rotating order 0,1,2,...,7,0,1,2,...,7,...
    /// and reach depth 1598 (~= the 1600-point spine length), i.e. the
    /// per-axis ~2098-halving budget is spent ONCE PER AXIS instead of once
    /// total. With 8 axes that gives a projected depth around `8 * 2098 ~=
    /// 16_784` — well past `10_000`, confirmed by measurement below.
    #[test]
    #[ignore]
    fn heavy_exponential_build_1m_dim8() {
        const DIM: usize = 8;
        let n = 1_000_000usize;

        let mut ladder: Vec<f64> = Vec::new();
        let mut v = 2f64.powi(1023);
        while v > 0.0 {
            ladder.push(v);
            v /= 2.0;
        }

        let spine_len = (ladder.len() * DIM).min(n - 1);
        let mut points: Vec<[f64; DIM]> = Vec::with_capacity(n);
        for k in 0..spine_len {
            let d = k % DIM;
            let m = k / DIM;
            let mut coords = [0.0f64; DIM];
            coords[d] = ladder[m];
            points.push(coords);
        }
        points.resize(n, [0.0f64; DIM]);

        let (arena, vind, root, _bbox) = build_full_tree(&points, 10);

        let depth = max_depth(&arena, root);
        // Measured empirically at 16794 (see doc comment above); threshold
        // kept well below that with margin.
        assert!(
            depth > 10_000,
            "expected round-robin degenerate tree to exceed the single-axis ceiling, got depth {depth}"
        );

        check_tree(&points, &arena, &vind, root, 10);
    }
}
