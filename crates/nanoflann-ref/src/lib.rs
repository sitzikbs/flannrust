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
    #[repr(C)]
    pub struct nfrd_index_f {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct nfrd_index_d {
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

        // ---- dynamic forest (KDTreeSingleIndexDynamicAdaptor), f32 -------
        pub fn nfrd_build_f(
            pts: *const f32,
            n_capacity: usize,
            dim: c_int,
            leaf_max_size: usize,
            maximum_point_count: usize,
        ) -> *mut nfrd_index_f;
        pub fn nfrd_free_f(h: *mut nfrd_index_f);
        pub fn nfrd_set_current_n_f(h: *mut nfrd_index_f, n: usize);
        pub fn nfrd_add_points_f(h: *mut nfrd_index_f, start: u32, end: u32);
        pub fn nfrd_remove_point_f(h: *mut nfrd_index_f, idx: usize);
        pub fn nfrd_knn_f(
            h: *const nfrd_index_f,
            q: *const f32,
            k: usize,
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f32,
        ) -> usize;
        pub fn nfrd_radius_count_f(
            h: *mut nfrd_index_f,
            q: *const f32,
            radius_sq: f32,
            sorted: c_int,
            eps: f32,
        ) -> usize;
        pub fn nfrd_radius_fetch_f(
            h: *const nfrd_index_f,
            out_idx: *mut u32,
            out_dist: *mut f32,
            cap: usize,
        ) -> usize;
        pub fn nfrd_tree_count_f(h: *const nfrd_index_f) -> usize;
        pub fn nfrd_slot_vacc_f(
            h: *const nfrd_index_f,
            slot: usize,
            out: *mut u32,
            cap: usize,
        ) -> usize;
        pub fn nfrd_tree_index_f(h: *const nfrd_index_f, out: *mut i32, cap: usize) -> usize;
        pub fn nfrd_removed_count_f(h: *const nfrd_index_f) -> usize;

        // ---- dynamic forest (KDTreeSingleIndexDynamicAdaptor), f64 -------
        pub fn nfrd_build_d(
            pts: *const f64,
            n_capacity: usize,
            dim: c_int,
            leaf_max_size: usize,
            maximum_point_count: usize,
        ) -> *mut nfrd_index_d;
        pub fn nfrd_free_d(h: *mut nfrd_index_d);
        pub fn nfrd_set_current_n_d(h: *mut nfrd_index_d, n: usize);
        pub fn nfrd_add_points_d(h: *mut nfrd_index_d, start: u32, end: u32);
        pub fn nfrd_remove_point_d(h: *mut nfrd_index_d, idx: usize);
        pub fn nfrd_knn_d(
            h: *const nfrd_index_d,
            q: *const f64,
            k: usize,
            // eps is `float` on the C++ side -- see nfr_knn_d above for the
            // same rationale (SearchParameters::eps is hard-typed float).
            eps: f32,
            out_idx: *mut u32,
            out_dist: *mut f64,
        ) -> usize;
        pub fn nfrd_radius_count_d(
            h: *mut nfrd_index_d,
            q: *const f64,
            radius_sq: f64,
            sorted: c_int,
            eps: f32,
        ) -> usize;
        pub fn nfrd_radius_fetch_d(
            h: *const nfrd_index_d,
            out_idx: *mut u32,
            out_dist: *mut f64,
            cap: usize,
        ) -> usize;
        pub fn nfrd_tree_count_d(h: *const nfrd_index_d) -> usize;
        pub fn nfrd_slot_vacc_d(
            h: *const nfrd_index_d,
            slot: usize,
            out: *mut u32,
            cap: usize,
        ) -> usize;
        pub fn nfrd_tree_index_d(h: *const nfrd_index_d, out: *mut i32, cap: usize) -> usize;
        pub fn nfrd_removed_count_d(h: *const nfrd_index_d) -> usize;
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
                Self {
                    handle,
                    dim,
                    _borrow: PhantomData,
                }
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
                    $knn(
                        self.handle,
                        q.as_ptr(),
                        k,
                        eps,
                        idx.as_mut_ptr(),
                        dist.as_mut_ptr(),
                    )
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
            pub fn knn_into(
                &self,
                q: &[$t],
                k: usize,
                eps: f32,
                out_idx: &mut [u32],
                out_dist: &mut [$t],
            ) -> usize {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                // SAFETY: `out_idx`/`out_dist` are sized to `k` (asserted
                // above); the C++ side (KNNResultSet capacity == k) writes
                // at most `k` entries and returns the number actually
                // written.
                unsafe {
                    $knn(
                        self.handle,
                        q.as_ptr(),
                        k,
                        eps,
                        out_idx.as_mut_ptr(),
                        out_dist.as_mut_ptr(),
                    )
                }
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
                let count =
                    unsafe { $radius_count(self.handle, q.as_ptr(), radius, sorted as c_int, eps) };
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
                let count =
                    unsafe { $radius_count(self.handle, q.as_ptr(), radius, sorted as c_int, eps) };
                out_idx.resize(count, 0);
                out_dist.resize(count, <$t as Default>::default());
                // SAFETY: `out_idx`/`out_dist` sized to `count`, obtained
                // from the immediately-preceding `_count` call on this same
                // handle.
                let returned = unsafe {
                    $radius_fetch(
                        self.handle,
                        out_idx.as_mut_ptr(),
                        out_dist.as_mut_ptr(),
                        count,
                    )
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
                let handle =
                    unsafe { $build(pts.as_ptr(), n, leaf_max_size, n_thread_build as c_uint) };
                assert!(
                    !handle.is_null(),
                    "nanoflann-ref: C++ index construction failed (exception caught in wrapper)"
                );
                Self {
                    handle,
                    _borrow: PhantomData,
                }
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
                let found = unsafe {
                    $knn(
                        self.handle,
                        q.as_ptr(),
                        k,
                        idx.as_mut_ptr(),
                        dist.as_mut_ptr(),
                    )
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// Zero-allocation `knn`: writes into caller-owned `out_idx`/
            /// `out_dist` (both MUST have length exactly `k` -- asserted)
            /// instead of allocating fresh `Vec`s. Returns the found count
            /// (`<= k`); see `RefIndexF32::knn_into`'s doc comment for the
            /// exact valid-entries contract.
            pub fn knn_into(
                &self,
                q: &[$t],
                k: usize,
                out_idx: &mut [u32],
                out_dist: &mut [$t],
            ) -> usize {
                assert_eq!(q.len(), 3, "query dim mismatch (fixed DIM=3)");
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                // SAFETY: see `RefIndex*::knn_into` above -- identical
                // capacity/return-count contract.
                unsafe {
                    $knn(
                        self.handle,
                        q.as_ptr(),
                        k,
                        out_idx.as_mut_ptr(),
                        out_dist.as_mut_ptr(),
                    )
                }
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

define_ref_index3!(
    RefIndex3F32,
    f32,
    raw::nfr3_index_f,
    raw::nfr3_build_f,
    raw::nfr3_free_f,
    raw::nfr3_knn_f
);
define_ref_index3!(
    RefIndex3F64,
    f64,
    raw::nfr3_index_d,
    raw::nfr3_build_d,
    raw::nfr3_free_d,
    raw::nfr3_knn_d
);

/// Defines the dynamic-forest (`KDTreeSingleIndexDynamicAdaptor`) safe
/// wrapper `$Name<'a>` around the raw `nfrd_*_$suf` entry points for scalar
/// type `$t`. L2 metric only (M2's dynamic cross-validation scope).
///
/// Design note (mutable `current_n`): C++'s ctor auto-adds every point
/// currently reported by `kdtree_get_point_count()` the instant the index is
/// constructed, with no way to opt out. So that a caller (the cross-
/// validation harness) can drive deterministic add/remove sequences instead
/// of getting every point auto-inserted at construction time, the dataset
/// adaptor on the C++ side (`RowMajorDynAdaptor` in `cpp/wrapper.cpp`)
/// starts its logical size (`current_n`) at 0 and only grows it when
/// `set_current_n` is called. `pts` itself is never resized or copied --
/// only the logical size nanoflann sees changes.
///
/// Mutation methods (`set_current_n`, `add_points`, `remove_point`) take
/// `&mut self`: unlike `knn`/`radius`'s internal two-call scratch-buffer
/// protocol (mutating C++-side state transparently under `&self`, per the
/// static wrappers' convention), these calls change the tree's actual point
/// membership, which is a real, externally-visible mutation of `self` and
/// so gets Rust's ordinary `&mut` treatment.
///
/// Safety invariants mirror `define_ref_index!` above: `pts` is borrowed for
/// `'a` and outlives `handle` (enforced by `PhantomData<&'a [$t]>` + `Drop`);
/// every out-buffer passed across the FFI boundary is sized before the call
/// so the C++ side never writes past a Rust-owned `Vec`'s end; the
/// radius two-call (`_count` then `_fetch`) protocol is encapsulated
/// entirely within one safe method so its pairing can't be violated by a
/// caller. `current_n` is tracked redundantly on the Rust side (the C++
/// surface exposes no getter for it) purely so `set_current_n`/`add_points`
/// can assert against the buffer bounds before ever reaching the FFI call.
macro_rules! define_ref_dyn_index {
    (
        $Name:ident, $t:ty, $Handle:ty,
        $build:path, $free:path, $set_current_n:path, $add_points:path, $remove_point:path,
        $knn:path, $radius_count:path, $radius_fetch:path,
        $tree_count:path, $slot_vacc:path, $tree_index:path, $removed_count:path
    ) => {
        /// Owned handle to a C++ `KDTreeSingleIndexDynamicAdaptor` borrowing
        /// `pts` (row-major, `n_capacity * dim` elements) for its whole
        /// life. `pts`'s logical size starts at 0 (see the macro's doc
        /// comment) and only grows via `set_current_n`.
        pub struct $Name<'a> {
            handle: *mut $Handle,
            dim: usize,
            n_capacity: usize,
            current_n: usize,
            _borrow: PhantomData<&'a [$t]>,
        }

        impl<'a> $Name<'a> {
            /// Builds an initially-EMPTY dynamic forest over the buffer
            /// `pts` (capacity `pts.len() / dim` points; `pts.len()` must be
            /// a multiple of `dim`). No points are queryable until
            /// `set_current_n` + `add_points` are called -- see the macro's
            /// doc comment for why (C++'s ctor auto-add path is a no-op
            /// here by construction, not by a wrapper-side special case).
            pub fn build(pts: &'a [$t], dim: usize, leaf_max_size: usize, maximum_point_count: usize) -> Self {
                assert!(dim > 0, "dim must be > 0");
                assert_eq!(
                    pts.len() % dim,
                    0,
                    "pts.len() ({}) must be a multiple of dim ({})",
                    pts.len(),
                    dim
                );
                let n_capacity = pts.len() / dim;
                // SAFETY: `pts` is borrowed for lifetime 'a and outlives
                // `handle` by construction (PhantomData<&'a [$t]> ties this
                // struct's lifetime to it, and Drop frees the C++ index
                // before 'a can end). `pts.as_ptr()` is valid even for an
                // empty slice (dangling-but-aligned, never dereferenced
                // since current_n starts at 0 and the C++ side reads at
                // most `current_n` rows at any point in time).
                let handle = unsafe {
                    $build(pts.as_ptr(), n_capacity, dim as c_int, leaf_max_size, maximum_point_count)
                };
                assert!(
                    !handle.is_null(),
                    "nanoflann-ref: C++ dynamic index construction failed (exception caught in wrapper)"
                );
                Self { handle, dim, n_capacity, current_n: 0, _borrow: PhantomData }
            }

            /// Grows (or shrinks) the logical dataset size that the C++
            /// side's `kdtree_get_point_count()` reports. Does NOT insert
            /// any points into the tree by itself -- points are only
            /// queryable after a subsequent `add_points` call covering
            /// their index range. `n * dim` must not exceed the buffer
            /// passed to `build`.
            ///
            /// Shrinking has NO effect on points already added: the C++
            /// side only ever consults `kdtree_get_point_count()` once, at
            /// construction time (to decide whether to auto-add an initial
            /// batch), and never again afterward -- `set_current_n` exists
            /// purely so this wrapper's own buffer-bound asserts (here and
            /// in `add_points`) have something to check against. Shrinking
            /// `n` below a point's index does not remove or hide that point
            /// from the tree; the caller owns keeping `add_points`/
            /// `remove_point` calls consistent with whatever `n` it sets.
            pub fn set_current_n(&mut self, n: usize) {
                assert!(
                    n <= self.n_capacity,
                    "set_current_n: n ({}) exceeds buffer capacity ({} points)",
                    n,
                    self.n_capacity
                );
                // SAFETY: `self.handle` was built by `$build` and is valid
                // for the lifetime of `self`; the C++ side additionally
                // bounds-checks `n` against the capacity it was given at
                // build time.
                unsafe { $set_current_n(self.handle, n) };
                self.current_n = n;
            }

            /// C++ `addPoints(start, end)` -- END-INCLUSIVE (adds every
            /// index in `start..=end`), unmodified. Points previously
            /// removed via `remove_point` are reactivated in place rather
            /// than duplicated (nanoflann's own tombstone-reuse behavior).
            ///
            /// CONTIGUOUS-APPEND CONTRACT (nanoflann's own usage contract,
            /// not a wrapper-added restriction -- nanoflann.hpp ~2647-2668):
            /// for every NEW (never-before-added) index `idx` in the call's
            /// range, the C++ side writes `treeIndex_[pointCount_] = pos`,
            /// keyed by its own running `pointCount_` counter, NOT by `idx`.
            /// This only produces a correct `treeIndex_[idx]` mapping when
            /// `idx == pointCount_` at the moment it is processed -- i.e.
            /// `start` must equal the number of points ever added so far
            /// (every prior `add_points` call's new, non-reactivated
            /// indices, summed). In practice this means: always grow
            /// `current_n` and call `add_points` in contiguous blocks
            /// starting from 0, never with gaps or out-of-order `start`
            /// values (re-adding a previously-removed index via
            /// `add_points(idx, idx)` is fine -- that path is a
            /// reactivation, not a new-index insert, and does not advance
            /// `pointCount_`). A misaligned `start` does NOT panic or abort
            /// on either side of the FFI boundary -- it SILENTLY corrupts
            /// `treeIndex_` (see `add_points_misaligned_start_documents_silent_corruption`
            /// in `tests/oracle_dynamic.rs` for the observed symptom).
            pub fn add_points(&mut self, start: u32, end_inclusive: u32) {
                assert!(
                    (end_inclusive as usize) < self.current_n,
                    "add_points: end ({}) must be < current_n ({}); call set_current_n first",
                    end_inclusive,
                    self.current_n
                );
                // SAFETY: `self.handle` is valid for `self`'s lifetime;
                // `start..=end_inclusive` was just asserted to lie within
                // `current_n`, which is itself bounded by the buffer
                // capacity (enforced in `set_current_n`).
                unsafe { $add_points(self.handle, start, end_inclusive) };
            }

            /// C++ `removePoint(idx)` (lazy deletion: marks `idx` inactive
            /// without touching the underlying sub-tree storage). A no-op
            /// if `idx` is out of range or already removed (matches C++'s
            /// own early-return behavior).
            pub fn remove_point(&mut self, idx: usize) {
                // SAFETY: `self.handle` is valid for `self`'s lifetime; the
                // C++ side itself bounds-checks `idx` against `pointCount_`
                // and no-ops if out of range or already removed, so there is
                // no precondition to assert here beyond handle validity.
                unsafe { $remove_point(self.handle, idx) };
            }

            /// `k`-nearest-neighbor search over the live (non-removed)
            /// points across every sub-tree in the forest. Same
            /// `(indices, distances)` / truncation / `eps` contract as the
            /// static wrappers' `knn`.
            pub fn knn(&self, q: &[$t], k: usize, eps: f32) -> (Vec<u32>, Vec<$t>) {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                let mut idx = vec![0u32; k];
                let mut dist = vec![<$t as Default>::default(); k];
                // SAFETY: `idx`/`dist` are sized to `k` before the call; the
                // C++ side (KNNResultSet capacity == k) writes at most `k`
                // entries and returns the number actually written.
                let found = unsafe {
                    $knn(self.handle, q.as_ptr(), k, eps, idx.as_mut_ptr(), dist.as_mut_ptr())
                };
                idx.truncate(found);
                dist.truncate(found);
                (idx, dist)
            }

            /// Zero-allocation `knn` -- see the static wrappers'
            /// `knn_into` for the exact buffer-length/valid-entries
            /// contract (identical here).
            pub fn knn_into(&self, q: &[$t], k: usize, eps: f32, out_idx: &mut [u32], out_dist: &mut [$t]) -> usize {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                assert_eq!(out_idx.len(), k, "knn_into: out_idx.len() must equal k");
                assert_eq!(out_dist.len(), k, "knn_into: out_dist.len() must equal k");
                // SAFETY: see `knn` above -- identical capacity/return-count contract.
                unsafe { $knn(self.handle, q.as_ptr(), k, eps, out_idx.as_mut_ptr(), out_dist.as_mut_ptr()) }
            }

            /// All live points within `radius_sq` (strict `<`, matching
            /// nanoflann's `RadiusResultSet::addPoint`) as `(index,
            /// distance)` pairs. Ascending order iff `sorted`. `radius_sq`
            /// is the squared radius, matching the L2 convention used
            /// throughout this crate.
            pub fn radius(&self, q: &[$t], radius_sq: $t, sorted: bool, eps: f32) -> Vec<(u32, $t)> {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                // SAFETY: `_count` populates the handle's internal C++
                // scratch vector; `_fetch` is called immediately afterward
                // on the same handle, only from within this method -- the
                // two-call contract is never exposed as separate public API.
                let count = unsafe {
                    $radius_count(self.handle, q.as_ptr(), radius_sq, sorted as c_int, eps)
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

            /// Zero-(re)allocation `radius` -- see the static wrappers'
            /// `radius_into` for the exact resize/reuse contract (identical
            /// here).
            pub fn radius_into(
                &self,
                q: &[$t],
                radius_sq: $t,
                sorted: bool,
                eps: f32,
                out_idx: &mut Vec<u32>,
                out_dist: &mut Vec<$t>,
            ) -> usize {
                assert_eq!(q.len(), self.dim, "query dim mismatch");
                // SAFETY: see `radius` above -- same two-call pairing
                // discipline, encapsulated entirely within this method.
                let count = unsafe {
                    $radius_count(self.handle, q.as_ptr(), radius_sq, sorted as c_int, eps)
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

            /// Number of sub-trees currently in the Bentley-Saxe forest
            /// (`getAllIndices().size()`, i.e. C++'s `treeCount_`). Fixed
            /// at construction time (`floor(log2(maximum_point_count)) +
            /// 1`), so this never changes across the index's lifetime.
            pub fn tree_count(&self) -> usize {
                // SAFETY: `self.handle` is valid for the lifetime of `self`.
                unsafe { $tree_count(self.handle) }
            }

            /// Copy of sub-tree `slot`'s permutation array (`vAcc_`) --
            /// the point indices physically stored in that sub-tree
            /// (including any lazily-removed ones, which stay in `vAcc_`
            /// but are excluded from search results via `tree_index()`
            /// being `-1` for them). Panics (via a C++-side abort) if
            /// `slot >= tree_count()`.
            pub fn slot_vacc(&self, slot: usize) -> Vec<u32> {
                // SAFETY: `self.handle` is valid; we query the slot's
                // length via a zero-capacity probe call first so the
                // output `Vec` is sized correctly before the real call
                // writes into it (mirrors the static wrappers' two-call
                // count/fetch pattern, collapsed into one method here since
                // `nfrd_slot_vacc_*` itself both returns the length and
                // copies up to `cap` entries in a single call).
                let len = unsafe { $slot_vacc(self.handle, slot, std::ptr::null_mut(), 0) };
                let mut out = vec![0u32; len];
                if len > 0 {
                    let returned = unsafe { $slot_vacc(self.handle, slot, out.as_mut_ptr(), out.len()) };
                    debug_assert_eq!(returned, len);
                }
                out
            }

            /// Copy of `treeIndex_` (C++'s per-point sub-tree assignment;
            /// `-1` means removed). Length equals the number of points
            /// ever added (matching nanoflann's `pointCount_`), NOT
            /// `n_capacity`.
            pub fn tree_index(&self) -> Vec<i32> {
                // SAFETY: same zero-capacity-probe-then-fetch pattern as
                // `slot_vacc` above.
                let len = unsafe { $tree_index(self.handle, std::ptr::null_mut(), 0) };
                let mut out = vec![0i32; len];
                if len > 0 {
                    let returned = unsafe { $tree_index(self.handle, out.as_mut_ptr(), out.len()) };
                    debug_assert_eq!(returned, len);
                }
                out
            }

            /// `removedPoints_.size()` -- the number of currently-removed
            /// (not yet reactivated) points.
            pub fn removed_count(&self) -> usize {
                // SAFETY: `self.handle` is valid for the lifetime of `self`.
                unsafe { $removed_count(self.handle) }
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

define_ref_dyn_index!(
    RefDynIndexF32,
    f32,
    raw::nfrd_index_f,
    raw::nfrd_build_f,
    raw::nfrd_free_f,
    raw::nfrd_set_current_n_f,
    raw::nfrd_add_points_f,
    raw::nfrd_remove_point_f,
    raw::nfrd_knn_f,
    raw::nfrd_radius_count_f,
    raw::nfrd_radius_fetch_f,
    raw::nfrd_tree_count_f,
    raw::nfrd_slot_vacc_f,
    raw::nfrd_tree_index_f,
    raw::nfrd_removed_count_f
);

define_ref_dyn_index!(
    RefDynIndexF64,
    f64,
    raw::nfrd_index_d,
    raw::nfrd_build_d,
    raw::nfrd_free_d,
    raw::nfrd_set_current_n_d,
    raw::nfrd_add_points_d,
    raw::nfrd_remove_point_d,
    raw::nfrd_knn_d,
    raw::nfrd_radius_count_d,
    raw::nfrd_radius_fetch_d,
    raw::nfrd_tree_count_d,
    raw::nfrd_slot_vacc_d,
    raw::nfrd_tree_index_d,
    raw::nfrd_removed_count_d
);
