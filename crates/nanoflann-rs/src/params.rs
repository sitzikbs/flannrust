//! Search-time parameters (nanoflann's `SearchParameters`, nanoflann.hpp:874-881).

/// = nanoflann `SearchParameters` (its legacy `checks` field is dead upstream
/// and not ported). `eps` is multiplicative slack on the node-bound prune:
/// a node is visited iff `mindist * (1 + eps) <= worst_dist`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchParams {
    /// Multiplicative slack on the node-bound prune (see the struct doc).
    pub eps: f32,
    /// Whether results are sorted ascending by distance before returning.
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
///
/// A second, narrower gate exists even WITH the feature on: `build()`'s
/// signature additionally requires `DS: Sync` whenever "parallel" is
/// enabled (the parallel path shares `&DS` across real rayon worker
/// threads, which requires `Sync`) — so a non-`Sync` `DataSource` (e.g.
/// `Rc`-backed interior mutability) cannot call `build()` at all under the
/// default feature set, even to request a plain `Sequential` build. This is
/// a BOUND on `build()`, not a behavior gate on this enum, and it is what
/// `KdTreeBuilder::build_sequential()` exists to route around: that method
/// carries no `Sync` bound, always compiles, and always builds
/// (`Sequential` only — it panics on `Auto`/`Threads(_)`, since those
/// genuinely cannot run without `Sync`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildThreads {
    /// Build on the calling thread only (the default).
    #[default]
    Sequential,
    /// Build on rayon's ambient/global thread pool (requires the "parallel" feature).
    Auto,
    /// Build inside a scoped rayon pool of exactly this many threads
    /// (requires the "parallel" feature).
    Threads(core::num::NonZeroU32),
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
