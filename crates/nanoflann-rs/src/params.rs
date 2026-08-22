//! Search-time parameters (nanoflann's `SearchParameters`, nanoflann.hpp:874-881).

/// = nanoflann `SearchParameters` (its legacy `checks` field is dead upstream
/// and not ported). `eps` is multiplicative slack on the node-bound prune:
/// a node is visited iff `mindist * (1 + eps) <= worst_dist`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    pub eps: f32,
    pub sorted: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self { eps: 0.0, sorted: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_cpp_ctor_defaults() {
        let p = SearchParams::default();
        assert_eq!(p.eps, 0.0);
        assert!(p.sorted);
    }
}
