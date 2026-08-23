//! FFI oracle: C++ nanoflann 1.12.1 (vendored verbatim in `cpp/nanoflann.hpp`),
//! compiled by `build.rs` and exposed through the extern-C surface in
//! `cpp/wrapper.cpp`. This crate adds no algorithmic logic of its own -- it
//! is plumbing: raw `extern "C"` declarations below, then safe RAII wrappers
//! that own a C++ index handle and forward queries to it. It exists to be
//! the correctness oracle for cross-validation against the Rust port, and
//! the benchmark baseline; `publish = false` (test/bench infrastructure
//! only).
//!
//! Two build.rs flags are load-bearing for that role (see `build.rs` for
//! the full rationale): `-O3` unconditionally (the oracle must always be the
//! fast baseline, even under `cargo test`'s debug profile) and
//! `-ffp-contract=off` (matches rustc's no-FMA-contraction default, so
//! distance computations stay bit-comparable across the FFI boundary).

use std::marker::PhantomData;
use std::os::raw::{c_int, c_uint};

/// Distance metric selector, mirrored 1:1 with `nfr_metric` in
/// `cpp/wrapper.cpp` (and, transitively, with nanoflann's `metric_L1` /
/// `metric_L2` / `metric_L2_Simple` / `metric_SO2` / `metric_SO3` traits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Metric {
    L1 = 0,
    L2 = 1,
    L2Simple = 2,
    SO2 = 3,
    SO3 = 4,
}

/// Raw `unsafe extern "C"` declarations for the symbols defined in
/// `cpp/wrapper.cpp`. Not part of the public API -- callers should use the
/// safe RAII wrappers below (`RefIndexF32`, `RefIndexF64`, `RefIndex3F32`,
/// `RefIndex3F64`).
mod raw {
    use std::os::raw::{c_int, c_uint};

    // Opaque handle types: never constructed or read from Rust, only passed
    // around as pointers returned by / passed to the C++ side.
    #[repr(C)]
    pub struct nfr_index_f {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct nfr_index_d {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct nfr3_index_f {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct nfr3_index_d {
        _private: [u8; 0],
    }

    extern "C" {
        // ---- f32 ----
        pub fn nfr_build_f(
            pts: *const f32,
            n: usize,
            dim: c_int,
            metric: c_int,
            leaf_max_size: usize,
            n_thread_build: c_uint,
        ) -> *mut nfr_index_f;
        pub fn nfr_free_f(h: *mut nfr_index_f);
        pub fn nfr_size_f(h: *const nfr_index_f) -> usize;
        pub fn nfr_used_memory_f(h: *const nfr_index_f) -> usize;
        pub fn nfr_vind_f(h: *const nfr_index_f, out: *mut u32, cap: usize) -> usize;
        pub fn nfr_knn_f(
            h: *const nfr_index_f,
            q: *const f32,
            k: usize,
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f32,
        ) -> usize;
        pub fn nfr_rknn_f(
            h: *const nfr_index_f,
            q: *const f32,
            k: usize,
            radius: f32,
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f32,
        ) -> usize;
        pub fn nfr_radius_count_f(
            h: *mut nfr_index_f,
            q: *const f32,
            radius: f32,
            sorted: c_int,
            eps: f32,
        ) -> usize;
        pub fn nfr_radius_fetch_f(
            h: *const nfr_index_f,
            out_idx: *mut u32,
            out_dist: *mut f32,
            cap: usize,
        ) -> usize;
        pub fn nfr_box_count_f(h: *mut nfr_index_f, lo: *const f32, hi: *const f32) -> usize;
        pub fn nfr_box_fetch_f(h: *const nfr_index_f, out_idx: *mut u32, cap: usize) -> usize;

        pub fn nfr3_build_f(
            pts: *const f32,
            n: usize,
            leaf_max_size: usize,
            n_thread_build: c_uint,
        ) -> *mut nfr3_index_f;
        pub fn nfr3_free_f(h: *mut nfr3_index_f);
        pub fn nfr3_knn_f(
            h: *const nfr3_index_f,
            q: *const f32,
            k: usize,
            out_idx: *mut u32,
            out_dist: *mut f32,
        ) -> usize;

        // ---- f64 ----
        pub fn nfr_build_d(
            pts: *const f64,
            n: usize,
            dim: c_int,
            metric: c_int,
            leaf_max_size: usize,
            n_thread_build: c_uint,
        ) -> *mut nfr_index_d;
        pub fn nfr_free_d(h: *mut nfr_index_d);
        pub fn nfr_size_d(h: *const nfr_index_d) -> usize;
        pub fn nfr_used_memory_d(h: *const nfr_index_d) -> usize;
        pub fn nfr_vind_d(h: *const nfr_index_d, out: *mut u32, cap: usize) -> usize;
        pub fn nfr_knn_d(
            h: *const nfr_index_d,
            q: *const f64,
            k: usize,
            // eps is `float` on the C++ side (`SearchParameters::eps` is
            // hard-typed `float` regardless of the tree's scalar type) --
            // declared `f32` here, NOT `f64`, so there is no narrowing
            // conversion anywhere in the FFI boundary.
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f64,
        ) -> usize;
        pub fn nfr_rknn_d(
            h: *const nfr_index_d,
            q: *const f64,
            k: usize,
            radius: f64,
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f64,
        ) -> usize;
        pub fn nfr_radius_count_d(
            h: *mut nfr_index_d,
            q: *const f64,
            radius: f64,
            sorted: c_int,
            eps: f32,
        ) -> usize;
        pub fn nfr_radius_fetch_d(
            h: *const nfr_index_d,
            out_idx: *mut u32,
            out_dist: *mut f64,
            cap: usize,
        ) -> usize;
        pub fn nfr_box_count_d(h: *mut nfr_index_d, lo: *const f64, hi: *const f64) -> usize;
        pub fn nfr_box_fetch_d(h: *const nfr_index_d, out_idx: *mut u32, cap: usize) -> usize;

        pub fn nfr3_build_d(
            pts: *const f64,
            n: usize,
            leaf_max_size: usize,
            n_thread_build: c_uint,
        ) -> *mut nfr3_index_d;
        pub fn nfr3_free_d(h: *mut nfr3_index_d);
        pub fn nfr3_knn_d(
            h: *const nfr3_index_d,
            q: *const f64,
            k: usize,
            out_idx: *mut u32,
            out_dist: *mut f64,
        ) -> usize;
    }
}

/// Defines the runtime-dim safe wrapper `$Name<'a>` around the raw
/// `nfr_*_$suf` entry points for scalar type `$t`.
///
/// Safety invariants upheld by every method below:
/// - The C++ index borrows `pts` (via nanoflann's `RowMajorAdaptor`) for its
///   entire life; it is never copied. `PhantomData<&'a [$t]>` ties this
///   wrapper's lifetime to `pts`'s, so the borrow can't outlive the data.
/// - Every out-buffer passed to the C++ side is sized to the query's `k` (or
///   to a count obtained from a paired `_count` call) *before* the call, so
///   the C++ side never writes past the end of a Rust-owned `Vec`.
/// - `radius`/`find_within_box` use a two-call (`_count` then `_fetch`)
///   protocol against per-handle C++ scratch state; both calls are made
///   back-to-back inside one safe method here and the intermediate raw
///   `_count`/`_fetch` functions are never exposed outside this module, so
///   the pairing contract can't be violated by a caller.
/// - Every `&self` method below (`knn`, `radius`, `find_within_box`, ...)
///   mutates C++-side scratch state through `self.handle`'s raw pointer
///   despite taking `&self`, not `&mut self` (the two-call radius/box
///   protocol writes into `nfr_index<T>`'s `radius_scratch`/`box_scratch`
///   fields via a `*mut`/`*const` cast on the C++ side). This is sound only
///   because `$Name<'a>` is `!Sync` (its only field beside `PhantomData` is
///   a raw pointer, which is neither `Send` nor `Sync` by default, and nothing
///   here opts back in) — never add `unsafe impl Sync for` any of these
///   types; doing so would let two threads call `&self` methods
///   concurrently and race on that shared C++ scratch state.
macro_rules! define_ref_index {
    (
        $Name:ident, $t:ty, $Handle:ty,
        $build:path, $free:path, $size:path, $used_memory:path, $vind:path,
        $knn:path, $rknn:path, $radius_count:path, $radius_fetch:path,
        $box_count:path, $box_fetch:path
    ) => {
        /// Owned handle to a C++ nanoflann index borrowing `pts` (row-major,
        /// `n * dim` elements) for its whole life.
        pub struct $Name<'a> {
            handle: *mut $Handle,
            dim: usize,
            _borrow: PhantomData<&'a [$t]>,
        }

        impl<'a> $Name<'a> {
            /// Builds the index over `pts`. `pts.len()` must be a multiple
            /// of `dim`. Mirrors nanoflann's real behavior for `n == 0`
            /// (empty dataset) unmodified: the wrapper does not special-case
            /// it (see `cpp/wrapper.cpp`'s `nfr_build_impl` notes).
            pub fn build(
                pts: &'a [$t],
                dim: usize,
                metric: Metric,
                leaf_max_size: usize,
                n_thread_build: u32,
            ) -> Self {
                assert!(dim > 0, "dim must be > 0");
                assert_eq!(
                    pts.len() % dim,
                    0,
                    "pts.len() ({}) must be a multiple of dim ({})",
                    pts.len(),
                    dim
                );
                let n = pts.len() / dim;
                // SAFETY: `pts` is borrowed for lifetime 'a and outlives
                // `handle` by construction (PhantomData<&'a [$t]> ties this
                // struct's lifetime to it, and Drop frees the C++ index
                // before 'a can end). `pts.as_ptr()` is valid even for an
                // empty slice (dangling-but-aligned, never dereferenced
                // since n == 0 in that case).
                let handle = unsafe {
                    $build(
                        pts.as_ptr(),
                        n,
                        dim as c_int,
                        metric as c_int,
                        leaf_max_size,
                        n_thread_build as c_uint,
                    )
                };
                assert!(
                    !handle.is_null(),
                    "nanoflann-ref: C++ index construction failed (exception caught in wrapper)"
                );
                Self { handle, dim, _borrow: PhantomData }
            }

            /// Number of points in the tree (`Base::size_`).
            pub fn size(&self) -> usize {
                // SAFETY: `self.handle` was built by `$build` and is valid
                // for the lifetime of `self`.
                unsafe { $size(self.handle) }
            }

            /// C++ `usedMemory(*index)`.
            pub fn used_memory(&self) -> usize {
                // SAFETY: see `size`.
                unsafe { $used_memory(self.handle) }
            }

            /// Copy of the tree's permutation array (`KDTreeBaseClass::vAcc_`).
            pub fn vind(&self) -> Vec<u32> {
                let n = self.size();
                let mut out = vec![0u32; n];
                // SAFETY: `out` is sized to `n` (the tree's point count,
                // just queried above) before the call; the C++ side copies
                // min(cap, n) == n entries into it and returns n.
                let returned = unsafe { $vind(self.handle, out.as_mut_ptr(), out.len()) };
                debug_assert_eq!(returned, n);
                out
            }

            /// `k`-nearest-neighbor search (findNeighbors + KNNResultSet).
            /// Returns `(indices, squared-or-metric distances)`, truncated
            /// to the number actually found (less than `k` only if the tree
            /// holds fewer than `k` points). `eps` is `f32` regardless of
            /// `$t` -- matches the C++ `SearchParameters::eps`, which is
            /// hard-typed `float` no matter the tree's scalar type, so
            /// there is no narrowing conversion anywhere on this path.
            pub fn knn(&self, q: &[$t], k: usize, eps: f32) -> (Vec<u32>, Vec<$t>) {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                // SAFETY: `idx`/`dist` are sized to `k` before the call; the
                // C++ side (KNNResultSet capacity == k) writes at most `k`
                // entries and returns the number actually written, which we
                // use to truncate immediately below.
                let found = unsafe {
                    $knn(self.handle, q.as_ptr(), k, eps, idx.as_mut_ptr(), dist.as_mut_ptr())
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// Zero-allocation `knn`: writes into caller-owned `out_idx`/
            /// `out_dist` (both MUST have length exactly `k` -- asserted)
            /// instead of allocating fresh `Vec`s, so a caller can reuse the
            /// same buffers across many calls (benchmark/perf-gate query
            /// loops). Returns the found count (`<= k`, valid entries are
            /// `out_idx[..found]`/`out_dist[..found]`; anything beyond
            /// `found` is leftover from the buffer's previous contents, NOT
            /// meaningful). Same semantics as `knn` otherwise.
            pub fn knn_into(&self, q: &[$t], k: usize, eps: f32, out_idx: &mut [u32], out_dist: &mut [$t]) -> usize {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                // SAFETY: `out_idx`/`out_dist` are sized to `k` (asserted
                // above); the C++ side (KNNResultSet capacity == k) writes
                // at most `k` entries and returns the number actually
                // written.
                unsafe { $knn(self.handle, q.as_ptr(), k, eps, out_idx.as_mut_ptr(), out_dist.as_mut_ptr()) }
            }

            /// `k`-nearest-neighbor search bounded by a maximum radius
            /// (RKNNResultSet). `radius` is passed through raw -- for L2
            /// metrics the caller must pre-square it, matching nanoflann's
            /// own squared-distance convention. `eps` is `f32` -- see `knn`.
            pub fn rknn(&self, q: &[$t], k: usize, radius: $t, eps: f32) -> (Vec<u32>, Vec<$t>) {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                // SAFETY: see `knn` -- same capacity/return-count contract.
                let found = unsafe {
                    $rknn(
                        self.handle,
                        q.as_ptr(),
                        k,
                        radius,
                        eps,
                        idx.as_mut_ptr(),
                        dist.as_mut_ptr(),
                    )
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// All points within `radius` (strict `<`, matching nanoflann's
            /// `RadiusResultSet::addPoint`), as `(index, distance)` pairs.
            /// Ascending order iff `sorted`. `eps` is `f32` -- see `knn`.
            pub fn radius(&self, q: &[$t], radius: $t, sorted: bool, eps: f32) -> Vec<(u32, $t)> {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                // SAFETY: `_count` populates the handle's internal C++
                // scratch vector; `_fetch` is called immediately afterward,
                // on the same handle, only from within this method -- the
                // two-call contract is never exposed as separate public API
                // so callers cannot violate the pairing.
                let count = unsafe {
                    $radius_count(self.handle, q.as_ptr(), radius, sorted as c_int, eps)
                };
                let mut idx = vec![0u32; count];
                let mut dist = vec![<$t as Default>::default(); count];
                // SAFETY: `idx`/`dist` sized to `count`, obtained from the
                // immediately-preceding `_count` call on this same handle.
                let returned = unsafe {
                    $radius_fetch(self.handle, idx.as_mut_ptr(), dist.as_mut_ptr(), count)
                };
                debug_assert_eq!(returned, count);
                idx.into_iter().zip(dist).collect()
            }

            /// Zero-(re)allocation `radius`: writes into caller-owned
            /// `out_idx`/`out_dist` `Vec`s instead of allocating fresh ones
            /// per call. Contract: both `Vec`s are `resize`d to the exact
            /// result count internally (grows only if the caller's current
            /// capacity is insufficient -- Rust's `Vec::resize` never
            /// reallocates when shrinking or when capacity already
            /// suffices, so a caller reusing the same `Vec`s across many
            /// calls with similar result sizes pays for at most the first
            /// few calls' worth of growth). Returns the result count (==
            /// `out_idx.len() == out_dist.len()` after the call). Same
            /// ordering/strict-`<` semantics as `radius` otherwise.
            pub fn radius_into(
                &self,
                q: &[$t],
                radius: $t,
                sorted: bool,
                eps: f32,
                out_idx: &mut Vec<u32>,
                out_dist: &mut Vec<$t>,
            ) -> usize {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                // SAFETY: see `radius` above -- same two-call pairing
                // discipline, encapsulated entirely within this method.
                let count = unsafe {
                    $radius_count(self.handle, q.as_ptr(), radius, sorted as c_int, eps)
                };
                out_idx.resize(count, 0);
                out_dist.resize(count, <$t as Default>::default());
                // SAFETY: `out_idx`/`out_dist` sized to `count`, obtained
                // from the immediately-preceding `_count` call on this same
                // handle.
                let returned = unsafe {
                    $radius_fetch(self.handle, out_idx.as_mut_ptr(), out_dist.as_mut_ptr(), count)
                };
                debug_assert_eq!(returned, count);
                count
            }

            /// Indices of all points inside the axis-aligned box `[lo, hi]`
            /// (inclusive on every face, matching nanoflann's
            /// `findWithinBox`). No distances -- box results are
            /// indices-only.
            pub fn find_within_box(&self, lo: &[$t], hi: &[$t]) -> Vec<u32> {
                assert_eq!(lo.len(), self.dim, "lo dim mismatch");
                assert_eq!(hi.len(), self.dim, "hi dim mismatch");
                // SAFETY: same two-call pairing discipline as `radius`
                // above, encapsulated entirely within this method.
                let count = unsafe { $box_count(self.handle, lo.as_ptr(), hi.as_ptr()) };
                let mut idx = vec![0u32; count];
                // SAFETY: `idx` is sized to `count`, obtained from the
                // immediately-preceding `_count` call on this same handle.
                let returned = unsafe { $box_fetch(self.handle, idx.as_mut_ptr(), count) };
                debug_assert_eq!(returned, count);
                idx
            }
        }

        impl<'a> Drop for $Name<'a> {
            fn drop(&mut self) {
                // SAFETY: `self.handle` was created by `$build` in
                // `Self::build` and is freed exactly once, here.
                unsafe { $free(self.handle) }
            }
        }
    };
}

define_ref_index!(
    RefIndexF32,
    f32,
    raw::nfr_index_f,
    raw::nfr_build_f,
    raw::nfr_free_f,
    raw::nfr_size_f,
    raw::nfr_used_memory_f,
    raw::nfr_vind_f,
    raw::nfr_knn_f,
    raw::nfr_rknn_f,
    raw::nfr_radius_count_f,
    raw::nfr_radius_fetch_f,
    raw::nfr_box_count_f,
    raw::nfr_box_fetch_f
);

define_ref_index!(
    RefIndexF64,
    f64,
    raw::nfr_index_d,
    raw::nfr_build_d,
    raw::nfr_free_d,
    raw::nfr_size_d,
    raw::nfr_used_memory_d,
    raw::nfr_vind_d,
    raw::nfr_knn_d,
    raw::nfr_rknn_d,
    raw::nfr_radius_count_d,
    raw::nfr_radius_fetch_d,
    raw::nfr_box_count_d,
    raw::nfr_box_fetch_d
);

/// Defines the fixed-DIM=3, L2-only, benchmark-only fast-path wrapper
/// `$Name<'a>` (build + knn only, per the C++ side's `nfr3_*` entry points).
macro_rules! define_ref_index3 {
    ($Name:ident, $t:ty, $Handle:ty, $build:path, $free:path, $knn:path) => {
        /// Fixed-DIM=3 L2 fast-path index (benchmark baseline only -- no
        /// eps, no radius/box queries). Same borrowing/lifetime contract as
        /// the runtime-dim wrappers above.
        pub struct $Name<'a> {
            handle: *mut $Handle,
            _borrow: PhantomData<&'a [$t]>,
        }

        impl<'a> $Name<'a> {
            /// `pts.len()` must be a multiple of 3.
            pub fn build(pts: &'a [$t], leaf_max_size: usize, n_thread_build: u32) -> Self {
                assert_eq!(pts.len() % 3, 0, "pts.len() must be a multiple of 3");
                let n = pts.len() / 3;
                // SAFETY: same argument as `RefIndex*::build` above.
                let handle = unsafe {
                    $build(pts.as_ptr(), n, leaf_max_size, n_thread_build as c_uint)
                };
                assert!(
                    !handle.is_null(),
                    "nanoflann-ref: C++ index construction failed (exception caught in wrapper)"
                );
                Self { handle, _borrow: PhantomData }
            }

            /// `k`-nearest-neighbor search via nanoflann's plain
            /// `knnSearch` convenience method (no eps parameter -- default
            /// `SearchParameters` per their own best-known benchmark
            /// config).
            pub fn knn(&self, q: &[$t], k: usize) -> (Vec<u32>, Vec<$t>) {
                assert_eq!(q.len(), 3, "query dim mismatch (fixed DIM=3)");
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                // SAFETY: see `RefIndex*::knn` above -- identical
                // capacity/return-count contract.
                let found =
                    unsafe { $knn(self.handle, q.as_ptr(), k, idx.as_mut_ptr(), dist.as_mut_ptr()) };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// Zero-allocation `knn`: writes into caller-owned `out_idx`/
            /// `out_dist` (both MUST have length exactly `k` -- asserted)
            /// instead of allocating fresh `Vec`s. Returns the found count
            /// (`<= k`); see `RefIndexF32::knn_into`'s doc comment for the
            /// exact valid-entries contract.
            pub fn knn_into(&self, q: &[$t], k: usize, out_idx: &mut [u32], out_dist: &mut [$t]) -> usize {
                assert_eq!(q.len(), 3, "query dim mismatch (fixed DIM=3)");
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                // SAFETY: see `RefIndex*::knn_into` above -- identical
                // capacity/return-count contract.
                unsafe { $knn(self.handle, q.as_ptr(), k, out_idx.as_mut_ptr(), out_dist.as_mut_ptr()) }
            }
        }

        impl<'a> Drop for $Name<'a> {
            fn drop(&mut self) {
                // SAFETY: see `RefIndex*::drop` above.
                unsafe { $free(self.handle) }
            }
        }
    };
}

define_ref_index3!(RefIndex3F32, f32, raw::nfr3_index_f, raw::nfr3_build_f, raw::nfr3_free_f, raw::nfr3_knn_f);
define_ref_index3!(RefIndex3F64, f64, raw::nfr3_index_d, raw::nfr3_build_d, raw::nfr3_free_d, raw::nfr3_knn_d);
