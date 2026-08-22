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

/// Build-thread policy. C++ mapping: Sequential = n_thread_build 1 (the
/// default), Auto = 0 (hardware concurrency via rayon's global pool),
/// Threads(n) = a scoped rayon pool of n threads. DEVIATION (documented):
/// Threads(n) guarantees "at most n rayon workers", not C++'s exact
/// async-gating behavior.
///
/// The type exists regardless of features; only the non-`Sequential`
/// BEHAVIOR is feature-gated: without the "parallel" feature,
/// `KdTreeBuilder::build()` panics on `Auto`/`Threads(_)` with a message
/// mirroring C++ `NANOFLANN_NO_THREADS`'s throw ("Multithreading is
/// disabled").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildThreads {
    Sequential,
    Auto,
    Threads(core::num::NonZeroU32),
}

impl Default for BuildThreads {
    fn default() -> Self {
        Self::Sequential
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

    #[test]
    fn build_threads_default_is_sequential() {
        assert_eq!(BuildThreads::default(), BuildThreads::Sequential);
    }
}
