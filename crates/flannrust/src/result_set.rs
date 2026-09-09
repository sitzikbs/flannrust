//! Result-set collectors: the `ResultSet` trait the tree walk feeds
//! candidates into, its built-in implementations (`KnnResultSet`,
//! `RknnResultSet`, `RadiusResultSet`), and the `TieBreak` policy that
//! decides equal-distance ordering.

use core::marker::PhantomData;

use crate::scalar::DistanceValue;

/// One (index, distance) result pair. `#[repr(C)]` with index first — matches
/// the field ORDER of nanoflann's standard-layout `ResultItem { first; second }`
/// (their `first` is the index) so FFI consumers can share buffers.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResultItem<Idx, D> {
    /// Point index into the dataset.
    pub index: Idx,
    /// Distance from the query point, in the metric's native scale.
    pub distance: D,
}

/// Result-set contract used by `find_neighbors` (= nanoflann RESULTSET duck type).
pub trait ResultSet<D: DistanceValue, Idx: Copy> {
    /// The worst (largest) distance this set would currently accept — the
    /// tree walk's per-candidate prune gate (`dist < worst_dist()`).
    fn worst_dist(&self) -> D;
    /// Returns `false` to abort the search (nanoflann.hpp:1245-1250 honors this;
    /// all built-in sets always return `true`).
    fn add_point(&mut self, dist: D, index: Idx) -> bool;
    /// Whether the set has reached its capacity/coverage limit.
    fn full(&self) -> bool;
    /// Number of entries currently held.
    fn size(&self) -> usize;
    /// Called by `find_neighbors` when `SearchParams::sorted` (default true).
    /// KNN/RKNN are sorted by construction → default no-op, same as C++.
    fn sort(&mut self) {}
}

/// Tie-break policy for the shared sorted insert — the compile-time analog of
/// nanoflann's `NANOFLANN_FIRST_MATCH` macro (nanoflann.hpp:257-263).
pub trait TieBreak: Send + Sync + 'static {
    /// Should the existing entry `(prev_d, prev_i)` shift right to make room
    /// for the incoming `(d, i)`?
    fn shift<D: DistanceValue, Idx: Copy + PartialOrd>(
        prev_d: D,
        prev_i: Idx,
        d: D,
        i: Idx,
    ) -> bool;
}

/// C++ default: equal distances keep insertion (traversal) order.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeepInsertionOrder;

impl TieBreak for KeepInsertionOrder {
    #[inline(always)]
    fn shift<D: DistanceValue, Idx: Copy + PartialOrd>(
        prev_d: D,
        _prev_i: Idx,
        d: D,
        _i: Idx,
    ) -> bool {
        prev_d > d
    }
}

/// `NANOFLANN_FIRST_MATCH`: on equal distance the smaller index wins.
#[derive(Debug, Clone, Copy, Default)]
pub struct SmallestIndexWins;

impl TieBreak for SmallestIndexWins {
    #[inline(always)]
    fn shift<D: DistanceValue, Idx: Copy + PartialOrd>(
        prev_d: D,
        prev_i: Idx,
        d: D,
        i: Idx,
    ) -> bool {
        prev_d > d || (d == prev_d && prev_i > i)
    }
}

/// Shared sorted insert (= `detail::addPointToSortedResultSet`, nanoflann.hpp:252-283).
/// `count` is the number of valid entries; capacity is `dists.len()`.
/// Returns the new count.
fn add_point_to_sorted<D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak>(
    indices: &mut [Idx],
    dists: &mut [D],
    count: usize,
    dist: D,
    index: Idx,
) -> usize {
    let capacity = dists.len();

    // ponytail: binary search + copy_within for large k (O(log k) vs O(k)
    // comparisons, single memmove vs element-by-element shift)
    if capacity >= 32 {
        let search_end = count.min(capacity);
        // Binary search: find insertion point using TB::shift monotonicity
        let mut lo = 0usize;
        let mut hi = search_end;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if TB::shift(dists[mid], indices[mid], dist, index) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if lo < capacity {
            let shift_end = search_end.min(capacity - 1);
            if lo < shift_end {
                dists.copy_within(lo..shift_end, lo + 1);
                indices.copy_within(lo..shift_end, lo + 1);
            }
            dists[lo] = dist;
            indices[lo] = index;
        }
    } else {
        let mut i = count;
        while i > 0 {
            if TB::shift(dists[i - 1], indices[i - 1], dist, index) {
                if i < capacity {
                    dists[i] = dists[i - 1];
                    indices[i] = indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < capacity {
            dists[i] = dist;
            indices[i] = index;
        }
    }

    if count < capacity {
        count + 1
    } else {
        count
    }
}

/// K-nearest-neighbors result set (= `KNNResultSet`). Sorted ascending by
/// construction. `k = indices.len() = dists.len()`.
pub struct KnnResultSet<'a, D, Idx, TB = KeepInsertionOrder> {
    indices: &'a mut [Idx],
    dists: &'a mut [D],
    count: usize,
    cached_worst: D,
    _tb: PhantomData<TB>,
}

impl<'a, D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak> KnnResultSet<'a, D, Idx, TB> {
    /// Panics if the two buffers differ in length.
    pub fn new(indices: &'a mut [Idx], dists: &'a mut [D]) -> Self {
        assert_eq!(indices.len(), dists.len(), "indices/dists length mismatch");
        Self {
            indices,
            dists,
            count: 0,
            cached_worst: D::MAX,
            _tb: PhantomData,
        }
    }
}

impl<'a, D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak> ResultSet<D, Idx>
    for KnnResultSet<'a, D, Idx, TB>
{
    #[inline]
    fn worst_dist(&self) -> D {
        self.cached_worst
    }

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

    #[inline]
    fn full(&self) -> bool {
        self.count == self.dists.len()
    }

    #[inline]
    fn size(&self) -> usize {
        self.count
    }
}

/// Radius-capped KNN (= `RKNNResultSet`). `max_radius` uses the metric's scale:
/// SQUARED for L2 metrics (caller squares), plain angle for SO2 — same as C++,
/// where no squaring happens inside.
pub struct RknnResultSet<'a, D, Idx, TB = KeepInsertionOrder> {
    indices: &'a mut [Idx],
    dists: &'a mut [D],
    count: usize,
    max_radius: D,
    cached_worst: D,
    _tb: PhantomData<TB>,
}

impl<'a, D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak> RknnResultSet<'a, D, Idx, TB> {
    /// Panics if the two buffers differ in length.
    pub fn new(indices: &'a mut [Idx], dists: &'a mut [D], max_radius: D) -> Self {
        assert_eq!(indices.len(), dists.len(), "indices/dists length mismatch");
        Self {
            indices,
            dists,
            count: 0,
            max_radius,
            cached_worst: max_radius,
            _tb: PhantomData,
        }
    }
}

impl<'a, D: DistanceValue, Idx: Copy + PartialOrd, TB: TieBreak> ResultSet<D, Idx>
    for RknnResultSet<'a, D, Idx, TB>
{
    #[inline]
    fn worst_dist(&self) -> D {
        self.cached_worst
    }

    #[inline]
    fn add_point(&mut self, dist: D, index: Idx) -> bool {
        self.count =
            add_point_to_sorted::<D, Idx, TB>(self.indices, self.dists, self.count, dist, index);
        self.cached_worst = if self.count < self.dists.len() || self.count == 0 {
            self.max_radius
        } else {
            self.dists[self.count - 1]
        };
        true
    }

    #[inline]
    fn full(&self) -> bool {
        self.count == self.dists.len()
    }

    #[inline]
    fn size(&self) -> usize {
        self.count
    }
}

/// All points within a radius (= `RadiusResultSet`). STRICTLY `dist < radius`:
/// a point at exactly the radius is excluded (nanoflann.hpp:440-444).
pub struct RadiusResultSet<'a, D, Idx> {
    radius: D,
    items: &'a mut Vec<ResultItem<Idx, D>>,
}

impl<'a, D: DistanceValue, Idx: Copy> RadiusResultSet<'a, D, Idx> {
    /// Clears `items` (C++ ctor calls `init()` which clears).
    pub fn new(radius: D, items: &'a mut Vec<ResultItem<Idx, D>>) -> Self {
        items.clear();
        Self { radius, items }
    }
}

impl<'a, D: DistanceValue, Idx: Copy> ResultSet<D, Idx> for RadiusResultSet<'a, D, Idx> {
    #[inline]
    fn worst_dist(&self) -> D {
        self.radius
    }

    #[inline]
    fn add_point(&mut self, dist: D, index: Idx) -> bool {
        if dist < self.radius {
            self.items.push(ResultItem {
                index,
                distance: dist,
            });
        }
        true
    }

    /// Hardwired true (C++ nanoflann.hpp:433) — makes `find_neighbors` return true.
    #[inline]
    fn full(&self) -> bool {
        true
    }

    #[inline]
    fn size(&self) -> usize {
        self.items.len()
    }

    /// Sort by distance (= `IndexDist_Sorter`). C++ uses unstable `std::sort`,
    /// leaving equal-distance order unspecified; we use a stable sort, which is
    /// one deterministic choice within that latitude.
    fn sort(&mut self) {
        self.items.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .expect("NaN distance in radius results")
        });
    }
}

/// Box-search collector (= `BoxResultSet`). Used only by `find_within_box`,
/// which never sorts and returns a plain count — so this stays crate-private
/// and carries no distances. (C++'s manual-only `BoxResultSet::sort()` is
/// deliberately not ported; callers own the output Vec and can sort it.)
pub(crate) struct BoxResultSet<'a, Idx> {
    pub(crate) indices: &'a mut Vec<Idx>,
}

impl<'a, Idx: Copy> BoxResultSet<'a, Idx> {
    pub(crate) fn new(indices: &'a mut Vec<Idx>) -> Self {
        indices.clear();
        Self { indices }
    }

    #[inline]
    pub(crate) fn add(&mut self, index: Idx) {
        self.indices.push(index);
    }

    pub(crate) fn size(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Sorted insert
    #[test]
    fn test_sorted_insert() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);

        rs.add_point(5.0, 0);
        rs.add_point(1.0, 1);
        rs.add_point(3.0, 2);
        rs.add_point(2.0, 3);

        assert_eq!(rs.dists, [1.0, 2.0, 3.0]);
        assert_eq!(rs.indices, [1, 3, 2]);
        assert_eq!(rs.size(), 3);
        assert!(rs.full());
    }

    // Test 2: Capacity eviction
    #[test]
    fn test_capacity_eviction() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);

        rs.add_point(5.0, 0);
        rs.add_point(1.0, 1);
        rs.add_point(3.0, 2);
        rs.add_point(2.0, 3);
        rs.add_point(0.5, 4);

        assert_eq!(rs.dists, [0.5, 1.0, 2.0]);
        assert_eq!(rs.indices, [4, 1, 3]);
    }

    // Test 3: Ties with KeepInsertionOrder
    #[test]
    fn test_ties_keep_insertion_order() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32, KeepInsertionOrder>::new(&mut indices, &mut dists);

        rs.add_point(1.0, 7);
        rs.add_point(1.0, 3);
        rs.add_point(1.0, 9);

        assert_eq!(rs.indices, [7, 3, 9]);
    }

    // Test 4: Ties with SmallestIndexWins
    #[test]
    fn test_ties_smallest_index_wins() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = KnnResultSet::<f64, u32, SmallestIndexWins>::new(&mut indices, &mut dists);

        rs.add_point(1.0, 7);
        rs.add_point(1.0, 3);
        rs.add_point(1.0, 9);

        assert_eq!(rs.indices, [3, 7, 9]);
    }

    // Test 5: Tie at capacity boundary
    #[test]
    fn test_tie_at_capacity_boundary_keep_insertion() {
        let mut indices = [0u32; 2];
        let mut dists = [0.0f64; 2];
        let mut rs = KnnResultSet::<f64, u32, KeepInsertionOrder>::new(&mut indices, &mut dists);

        rs.add_point(1.0, 7);
        rs.add_point(2.0, 5);
        rs.add_point(2.0, 1);

        assert_eq!(rs.indices, [7, 5]);
    }

    #[test]
    fn test_tie_at_capacity_boundary_smallest_index() {
        let mut indices = [0u32; 2];
        let mut dists = [0.0f64; 2];
        let mut rs = KnnResultSet::<f64, u32, SmallestIndexWins>::new(&mut indices, &mut dists);

        rs.add_point(1.0, 7);
        rs.add_point(2.0, 5);
        rs.add_point(2.0, 1);

        assert_eq!(rs.indices, [7, 1]);
    }

    // Test 6: Knn worst_dist transition
    #[test]
    fn test_knn_worst_dist_transition() {
        let mut indices = [0u32; 2];
        let mut dists = [0.0f64; 2];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);

        // Empty
        assert_eq!(rs.worst_dist(), f64::MAX);

        // One point
        rs.add_point(1.0, 0);
        assert_eq!(rs.worst_dist(), f64::MAX);

        // Two points (full)
        rs.add_point(4.0, 1);
        assert_eq!(rs.worst_dist(), 4.0);
    }

    // Test 7: Rknn worst_dist transition
    #[test]
    fn test_rknn_worst_dist_transition() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 10.0);

        // Empty
        assert_eq!(rs.worst_dist(), 10.0);

        // Two points
        rs.add_point(1.0, 0);
        rs.add_point(2.0, 1);
        assert_eq!(rs.worst_dist(), 10.0);

        // Three points (full)
        rs.add_point(3.0, 2);
        assert_eq!(rs.worst_dist(), 3.0);

        // Push 2.5
        rs.add_point(2.5, 3);
        assert_eq!(rs.dists[..rs.size()], [1.0, 2.0, 2.5][..]);
        assert_eq!(rs.worst_dist(), 2.5);
    }

    // Test 8: Rknn radius gating is caller-side
    #[test]
    fn test_rknn_radius_gating_caller_side() {
        let mut indices = [0u32; 3];
        let mut dists = [0.0f64; 3];
        let mut rs = RknnResultSet::<f64, u32>::new(&mut indices, &mut dists, 10.0);

        rs.add_point(1.0, 0);
        // Directly add point beyond max_radius (11.0 > 10.0)
        rs.add_point(11.0, 9);

        // The set itself doesn't check — it just inserts
        assert!(rs.size() >= 2);
    }

    // Test 9: Radius strict <
    #[test]
    fn test_radius_strict_less_than() {
        let mut items = Vec::new();
        let mut rs = RadiusResultSet::new(4.0, &mut items);

        rs.add_point(4.0, 1); // Should NOT add (equal)
        assert_eq!(rs.size(), 0);

        rs.add_point(3.9999999, 2); // Should add
        assert_eq!(rs.size(), 1);

        assert!(rs.full()); // Always true
        assert_eq!(rs.worst_dist(), 4.0);
    }

    // Test 10: Radius sort
    #[test]
    fn test_radius_sort() {
        let mut items = Vec::new();
        let mut rs = RadiusResultSet::new(10.0, &mut items);

        rs.add_point(3.0, 0);
        rs.add_point(1.0, 1);
        rs.add_point(2.0, 2);
        rs.add_point(1.0, 3);

        rs.sort();

        let sorted_dists: Vec<_> = rs.items.iter().map(|r| r.distance).collect();
        assert_eq!(sorted_dists, [1.0, 1.0, 2.0, 3.0]);

        // Check stable sort preserves insertion order for equal distances
        let indices_for_1: Vec<_> = rs
            .items
            .iter()
            .filter(|r| r.distance == 1.0)
            .map(|r| r.index)
            .collect();
        assert_eq!(indices_for_1, [1, 3]); // Insertion order preserved
    }

    // Test 11: RadiusResultSet::new clears
    #[test]
    fn test_radius_new_clears() {
        let mut items = vec![ResultItem {
            index: 1u32,
            distance: 5.0f64,
        }];
        let rs = RadiusResultSet::new(10.0, &mut items);

        assert_eq!(rs.size(), 0);
    }

    // Test 12: ResultItem layout
    #[test]
    fn test_result_item_layout() {
        use core::mem::{align_of, size_of};
        use core::ptr::addr_of;

        // Size checks
        assert_eq!(size_of::<ResultItem<u32, f32>>(), 8);
        assert_eq!(align_of::<ResultItem<u32, f32>>(), 4);

        assert_eq!(size_of::<ResultItem<u32, f64>>(), 16);

        // Field offset
        let item = ResultItem {
            index: 42u32,
            distance: 3.5f64,
        };
        let item_ptr = &item as *const ResultItem<u32, f64> as usize;
        let index_ptr = addr_of!(item.index) as usize;

        assert_eq!(index_ptr - item_ptr, 0);
    }

    // Test 13: Binary search path (k=64) matches linear scan behavior
    #[test]
    fn test_large_k_binary_search_keep_insertion() {
        let k = 64;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let mut rs = KnnResultSet::<f64, u32, KeepInsertionOrder>::new(&mut indices, &mut dists);

        // Insert 2*k points in reverse distance order (worst case for linear scan)
        for i in (0..2 * k as u32).rev() {
            rs.add_point(i as f64, i);
        }

        // First k points should be 0..k, sorted ascending
        assert_eq!(rs.size(), k);
        for i in 0..k {
            assert_eq!(rs.dists[i], i as f64);
            assert_eq!(rs.indices[i], i as u32);
        }
        assert_eq!(rs.worst_dist(), (k - 1) as f64);
    }

    #[test]
    fn test_large_k_binary_search_smallest_index() {
        let k = 64;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let mut rs = KnnResultSet::<f64, u32, SmallestIndexWins>::new(&mut indices, &mut dists);

        // Insert points with tied distances
        for i in 0..2 * k as u32 {
            rs.add_point((i % 4) as f64, 200 - i);
        }

        // Verify sorted by (distance ASC, index ASC)
        for w in rs.dists.windows(2) {
            assert!(w[0] <= w[1], "distances not sorted");
        }
        for i in 0..k - 1 {
            if rs.dists[i] == rs.dists[i + 1] {
                assert!(
                    rs.indices[i] < rs.indices[i + 1],
                    "tied distances not sorted by index: {}({}) vs {}({})",
                    rs.indices[i], rs.dists[i],
                    rs.indices[i + 1], rs.dists[i + 1]
                );
            }
        }
    }

    #[test]
    fn test_large_k_binary_vs_linear_equivalence() {
        // Build same result set with k=50 (binary) and k=10 (linear),
        // verify matching subset behavior
        let data: Vec<(f64, u32)> = (0..200)
            .map(|i| ((i * 37 % 100) as f64, i as u32))
            .collect();

        let k = 50;
        let mut indices = vec![0u32; k];
        let mut dists = vec![0.0f64; k];
        let mut rs = KnnResultSet::<f64, u32, KeepInsertionOrder>::new(&mut indices, &mut dists);
        for &(d, idx) in &data {
            rs.add_point(d, idx);
        }

        // Brute force: sort data by distance, take first k
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sorted.truncate(k);

        // Distances must match (indices may differ on ties with same distance)
        let rs_dists: Vec<f64> = rs.dists.to_vec();
        let expected_dists: Vec<f64> = sorted.iter().map(|&(d, _)| d).collect();
        assert_eq!(rs_dists, expected_dists);
    }

    // Test 14: k=0 KnnResultSet
    #[test]
    fn test_k_zero_knn_result_set() {
        let mut indices = [];
        let mut dists = [];
        let mut rs = KnnResultSet::<f64, u32>::new(&mut indices, &mut dists);

        assert!(rs.add_point(5.0, 0)); // No-op
        assert!(rs.full()); // 0 == 0
        assert_eq!(rs.worst_dist(), f64::MAX); // count==0 branch
    }
}
