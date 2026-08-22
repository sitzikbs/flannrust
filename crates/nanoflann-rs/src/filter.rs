//! Per-point search filter — the M2 seam for nanoflann's CRTP `isActive()`
//! hook (nanoflann.hpp ~1240: `if (!obj.isActive(accessor)) continue;`). The
//! static adaptor's `isActive` always returns `true`; the (future, M2)
//! dynamic adaptor overrides it to skip tombstoned (removed) points.

/// Per-point search filter.
pub trait PointFilter<Idx: Copy> {
    fn is_active(&self, idx: Idx) -> bool;
}

/// The static tree's filter: always true, provably erased after inlining.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceptAll;

impl<Idx: Copy> PointFilter<Idx> for AcceptAll {
    #[inline(always)]
    fn is_active(&self, _idx: Idx) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_all_always_true() {
        let f = AcceptAll;
        assert!(f.is_active(0u32));
        assert!(f.is_active(u32::MAX));
    }
}
