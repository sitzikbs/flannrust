use crate::scalar::Scalar;

/// Arena kd-tree node. Interior nodes store the split "gap" (`divlow` =
/// left child's bbox high on the split axis, `divhigh` = right child's low)
/// exactly like nanoflann; leaves store a `[left, right)` range into the
/// permuted point-index vector.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Node<T> {
    /// Leaf: range start into `vind`. Interior: arena index of child 1.
    a: u32,
    /// Leaf: range end (exclusive). Interior: arena index of child 2.
    b: u32,
    /// Interior: split dimension. Leaf: `LEAF` sentinel.
    divfeat: u32,
    /// Interior split bounds; zero on leaves. Stored as the ELEMENT type `T`,
    /// not the distance type: they are coordinates (the children's tightened
    /// bbox edges on the split axis), and the search feeds them to
    /// `accum_dist(_, _: T, axis)`. (C++ stores them as DistanceType only as
    /// a template artifact.)
    divlow: T,
    divhigh: T,
}

pub(crate) const LEAF: u32 = u32::MAX;

impl<T: Scalar> Node<T> {
    pub(crate) fn leaf(left: u32, right: u32) -> Self {
        Self {
            a: left,
            b: right,
            divfeat: LEAF,
            divlow: T::default(),
            divhigh: T::default(),
        }
    }

    /// A split node whose children are not yet known (arena build allocates
    /// the parent slot first, then patches). `set_children` completes it.
    pub(crate) fn split(divfeat: u32, divlow: T, divhigh: T) -> Self {
        Self {
            a: 0,
            b: 0,
            divfeat,
            divlow,
            divhigh,
        }
    }

    pub(crate) fn set_children(&mut self, child1: u32, child2: u32) {
        self.a = child1;
        self.b = child2;
    }

    /// Interior only: update the gap bounds (set by `finalize_split` after
    /// both children's bboxes are known).
    pub(crate) fn set_div_bounds(&mut self, divlow: T, divhigh: T) {
        self.divlow = divlow;
        self.divhigh = divhigh;
    }

    /// Interior only (`debug_assert!(!self.is_leaf())`): shift both child
    /// arena indices by `off`, used when splicing a subtree's local arena
    /// into a larger one at a nonzero offset (the parallel builder's merge
    /// step). Leaf ranges (`a`/`b` on a leaf) are `[left, right)` offsets
    /// into `vind` that are already GLOBAL (base-relative, see
    /// `SubtreeBuilder`/`build_subtree_parallel`'s `base` parameter) and
    /// must NEVER be offset here — only interior child ARENA indices shift
    /// when an arena is relocated.
    ///
    /// Only used by `build_parallel.rs`, which is compiled solely under the
    /// "parallel" feature — `allow(dead_code)` avoids a spurious warning on
    /// `--no-default-features` builds.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(crate) fn offset_children(&mut self, off: u32) {
        debug_assert!(!self.is_leaf());
        self.a += off;
        self.b += off;
    }
}

/// Readers used to WALK an already-built tree (as opposed to the writers
/// above, used to BUILD one). Consumed by `search.rs`'s tree walk
/// (`search_level`).
impl<T: Scalar> Node<T> {
    #[inline(always)]
    pub(crate) fn is_leaf(&self) -> bool {
        self.divfeat == LEAF
    }

    /// Leaf only: the `[left, right)` range into `vind`.
    #[inline(always)]
    pub(crate) fn leaf_range(&self) -> (usize, usize) {
        debug_assert!(self.is_leaf());
        (self.a as usize, self.b as usize)
    }

    /// Interior only: (child1, child2) arena indices.
    #[inline(always)]
    pub(crate) fn children(&self) -> (u32, u32) {
        debug_assert!(!self.is_leaf());
        (self.a, self.b)
    }

    /// Interior only.
    #[inline(always)]
    pub(crate) fn split_dim(&self) -> usize {
        debug_assert!(!self.is_leaf());
        self.divfeat as usize
    }

    #[inline(always)]
    pub(crate) fn div_low(&self) -> T {
        self.divlow
    }

    #[inline(always)]
    pub(crate) fn div_high(&self) -> T {
        self.divhigh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn test_node_f32_size_and_alignment() {
        // Intentional layout lock — vs C++'s ~48-byte alignas(16) node;
        // a future task may A/B test align(16).
        assert_eq!(size_of::<Node<f32>>(), 20);
        assert_eq!(align_of::<Node<f32>>(), 4);
    }

    #[test]
    fn test_node_f64_size_and_alignment() {
        // Intentional layout lock — vs C++'s ~48-byte alignas(16) node;
        // a future task may A/B test align(16).
        assert_eq!(size_of::<Node<f64>>(), 32);
        assert_eq!(align_of::<Node<f64>>(), 8);
    }

    #[test]
    fn test_leaf_round_trip() {
        let node = Node::<f32>::leaf(3, 9);
        assert!(node.is_leaf());
        assert_eq!(node.leaf_range(), (3, 9));
    }

    #[test]
    fn test_split_round_trip() {
        let mut node = Node::<f64>::split(2, 1.5, 2.5);
        node.set_children(4, 7);

        assert!(!node.is_leaf());
        assert_eq!(node.children(), (4, 7));
        assert_eq!(node.split_dim(), 2);
        assert_eq!(node.div_low(), 1.5);
        assert_eq!(node.div_high(), 2.5);

        node.set_div_bounds(1.0, 3.0);
        assert_eq!(node.div_low(), 1.0);
        assert_eq!(node.div_high(), 3.0);
    }

    #[test]
    fn test_leaf_sentinel_headroom() {
        let node = Node::<f32>::split(u32::MAX - 1, 0.0, 1.0);
        assert!(!node.is_leaf());
    }

    #[test]
    fn test_offset_children_shifts_interior_only() {
        let mut node = Node::<f64>::split(2, 1.5, 2.5);
        node.set_children(4, 7);
        node.offset_children(10);
        assert_eq!(node.children(), (14, 17));
    }

    #[test]
    fn test_node_equality() {
        let mut a = Node::<f64>::split(1, 1.0, 2.0);
        a.set_children(3, 5);
        let mut b = Node::<f64>::split(1, 1.0, 2.0);
        b.set_children(3, 5);
        assert_eq!(a, b);

        let leaf_a = Node::<f64>::leaf(0, 3);
        let leaf_b = Node::<f64>::leaf(0, 3);
        assert_eq!(leaf_a, leaf_b);
        assert_ne!(leaf_a, Node::<f64>::leaf(0, 4));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_offset_children_panics_on_leaf_in_debug() {
        let mut node = Node::<f64>::leaf(0, 3);
        node.offset_children(1);
    }
}
