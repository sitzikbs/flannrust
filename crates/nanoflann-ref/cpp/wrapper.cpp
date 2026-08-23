// extern "C" surface over the vendored, UNMODIFIED nanoflann.hpp (1.12.1).
// This file adds no algorithmic logic of its own -- it only builds the
// correct nanoflann template instantiations, forwards calls, and copies
// results into caller-owned buffers so the Rust side can be plain, safe FFI.
//
// Compiled with NANOFLANN_FIRST_MATCH left UNDEFINED (default nanoflann tie
// behavior: ties broken by insertion order into the sorted result buffer,
// not by index).
//
// All indices exposed across the C boundary are uint32_t: both the tree's
// IndexType (4th template parameter, `index_t`) and every ResultSet's
// IndexType are explicitly instantiated as uint32_t, so there is no
// size_t/uint32_t mismatch anywhere in this file.

#include "nanoflann.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>  // std::abort
#include <cstring>
#include <utility>
#include <vector>

using nanoflann::BoxResultSet;
using nanoflann::KDTreeSingleIndexAdaptor;
using nanoflann::KDTreeSingleIndexAdaptorParams;
using nanoflann::KDTreeSingleIndexDynamicAdaptor;
using nanoflann::KNNResultSet;
using nanoflann::L1_Adaptor;
using nanoflann::L2_Adaptor;
using nanoflann::L2_Simple_Adaptor;
using nanoflann::RadiusResultSet;
using nanoflann::RKNNResultSet;
using nanoflann::ResultItem;
using nanoflann::SearchParameters;
using nanoflann::SO2_Adaptor;
using nanoflann::SO3_Adaptor;

namespace
{

// Row-major point-cloud adaptor: pts[idx * dim + d]. Borrows `pts`, never
// copies -- the caller (Rust side) owns the buffer for the tree's lifetime.
template <typename T>
struct RowMajorAdaptor
{
    const T* pts;
    size_t   n;
    size_t   dim;

    inline size_t kdtree_get_point_count() const { return n; }
    inline T      kdtree_get_pt(const size_t idx, const size_t d) const { return pts[idx * dim + d]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

// Metric tag, mirrored on the Rust side as `Metric`.
enum nfr_metric
{
    NFR_L1        = 0,
    NFR_L2        = 1,
    NFR_L2_SIMPLE = 2,
    NFR_SO2       = 3,
    NFR_SO3       = 4,
};

// Runtime-dim (DIM = -1) tree, one alias per metric. index_t = uint32_t
// throughout (both the tree's IndexType and every ResultSet below use
// uint32_t explicitly).
template <typename T, typename Distance>
using Tree = KDTreeSingleIndexAdaptor<Distance, RowMajorAdaptor<T>, -1, uint32_t>;

template <typename T>
using L1Tree = Tree<T, L1_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>>;
template <typename T>
using L2Tree = Tree<T, L2_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>>;
template <typename T>
using L2SimpleTree = Tree<T, L2_Simple_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>>;
template <typename T>
using SO2Tree = Tree<T, SO2_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>>;
template <typename T>
using SO3Tree = Tree<T, SO3_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>>;

// Fixed DIM=3 fast-path tree, L2 metric only (benchmark-only entry points).
template <typename T>
using Tree3 = KDTreeSingleIndexAdaptor<
    L2_Adaptor<T, RowMajorAdaptor<T>, T, uint32_t>, RowMajorAdaptor<T>, 3, uint32_t>;

// Owning handle for the runtime-dim entry points: the adaptor (borrows the
// caller's points), a type-erased pointer to one of the 5 concrete tree
// instantiations above (selected by `metric` at build time), and scratch
// buffers for the two-call fetch pattern (radius/box search sizes are not
// known up front).
template <typename T>
struct nfr_index
{
    nfr_metric          metric;
    RowMajorAdaptor<T>  adaptor;
    void*               tree = nullptr;
    std::vector<ResultItem<uint32_t, T>> radius_scratch;
    std::vector<uint32_t>                box_scratch;
};

template <typename T>
struct nfr3_index
{
    RowMajorAdaptor<T> adaptor;
    Tree3<T>*          tree = nullptr;
};

// Dispatches to the concrete tree type selected by `h->metric` and invokes
// `f` on it. This is a single switch per call site (inlined/monomorphized
// via the template, one instantiation of `f` per metric) -- NOT a virtual
// call. nanoflann's DatasetAdaptor/Distance are compile-time template
// parameters, so there is no common base class to dispatch through
// virtually; a switch on a small enum is the only option, and keeping it to
// exactly one switch per entry point (via this shared helper) keeps the
// hot query paths free of vtable indirection.
template <typename T, typename F>
auto dispatch(nfr_index<T>* h, F&& f) -> decltype(f(std::declval<L2Tree<T>&>()))
{
    switch (h->metric)
    {
        case NFR_L1: return f(*static_cast<L1Tree<T>*>(h->tree));
        case NFR_L2: return f(*static_cast<L2Tree<T>*>(h->tree));
        case NFR_L2_SIMPLE: return f(*static_cast<L2SimpleTree<T>*>(h->tree));
        case NFR_SO2: return f(*static_cast<SO2Tree<T>*>(h->tree));
        case NFR_SO3: return f(*static_cast<SO3Tree<T>*>(h->tree));
    }
    // Unreachable: nfr_metric only ever holds one of the 5 values above,
    // assigned from the `metric` argument of the corresponding nfr*_build_*
    // call, which the Rust side constrains to the `Metric` enum's range.
    // Abort loudly instead of silently dispatching to the wrong tree type
    // (a `reinterpret_cast`-style fallback here would be a memory-safety
    // bug hiding as a correctness one) -- this indicates memory corruption
    // or a caller that bypassed the Rust `Metric` enum entirely.
    std::abort();
}

template <typename T, typename F>
auto dispatch(const nfr_index<T>* h, F&& f) -> decltype(f(std::declval<const L2Tree<T>&>()))
{
    switch (h->metric)
    {
        case NFR_L1: return f(*static_cast<const L1Tree<T>*>(h->tree));
        case NFR_L2: return f(*static_cast<const L2Tree<T>*>(h->tree));
        case NFR_L2_SIMPLE: return f(*static_cast<const L2SimpleTree<T>*>(h->tree));
        case NFR_SO2: return f(*static_cast<const SO2Tree<T>*>(h->tree));
        case NFR_SO3: return f(*static_cast<const SO3Tree<T>*>(h->tree));
    }
    // Unreachable -- see the non-const overload above for the rationale.
    std::abort();
}

// ---- Generic (templated on T) entry-point implementations -----------------

template <typename T>
nfr_index<T>* nfr_build_impl(
    const T* pts, size_t n, int dim, int metric, size_t leaf_max_size, unsigned n_thread_build)
{
    auto* h    = new nfr_index<T>();
    h->metric  = static_cast<nfr_metric>(metric);
    h->adaptor = RowMajorAdaptor<T>{pts, n, static_cast<size_t>(dim)};

    const KDTreeSingleIndexAdaptorParams params(
        leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_thread_build);

    switch (h->metric)
    {
        case NFR_L1:
            h->tree = new L1Tree<T>(dim, h->adaptor, params);
            break;
        case NFR_L2:
            h->tree = new L2Tree<T>(dim, h->adaptor, params);
            break;
        case NFR_L2_SIMPLE:
            h->tree = new L2SimpleTree<T>(dim, h->adaptor, params);
            break;
        case NFR_SO2:
            h->tree = new SO2Tree<T>(dim, h->adaptor, params);
            break;
        case NFR_SO3:
            h->tree = new SO3Tree<T>(dim, h->adaptor, params);
            break;
    }
    return h;
}

template <typename T>
void nfr_free_impl(nfr_index<T>* h)
{
    if (!h) return;
    switch (h->metric)
    {
        case NFR_L1: delete static_cast<L1Tree<T>*>(h->tree); break;
        case NFR_L2: delete static_cast<L2Tree<T>*>(h->tree); break;
        case NFR_L2_SIMPLE: delete static_cast<L2SimpleTree<T>*>(h->tree); break;
        case NFR_SO2: delete static_cast<SO2Tree<T>*>(h->tree); break;
        case NFR_SO3: delete static_cast<SO3Tree<T>*>(h->tree); break;
    }
    delete h;
}

template <typename T>
size_t nfr_size_impl(const nfr_index<T>* h)
{
    return dispatch(h, [](const auto& tree) { return tree.size(tree); });
}

template <typename T>
size_t nfr_used_memory_impl(const nfr_index<T>* h)
{
    return dispatch(h, [](const auto& tree) { return tree.usedMemory(tree); });
}

template <typename T>
size_t nfr_vind_impl(const nfr_index<T>* h, uint32_t* out, size_t cap)
{
    return dispatch(h, [&](const auto& tree) {
        const size_t n = tree.vAcc_.size();
        const size_t copy_n = cap < n ? cap : n;
        if (copy_n > 0) std::memcpy(out, tree.vAcc_.data(), copy_n * sizeof(uint32_t));
        return n;
    });
}

// `eps` is `float` here (NOT `T`) for every T, matching nanoflann's own
// `SearchParameters::eps` (nanoflann.hpp:874-881, hard-typed `float`
// regardless of the tree's scalar type). Previously this template took `T
// eps` for T=double entry points, which the Rust FFI declared as `f64` --
// that `f64` then silently narrowed to `float` HERE on construction of
// `SearchParameters`, so a value that round-tripped f32->f64->float on the
// Rust side was fine, but nothing enforced that the Rust caller was in fact
// widening from f32. Taking `float` directly makes both sides pass the exact
// same bits with no implicit narrowing anywhere -- eps parity is structural.
template <typename T>
size_t nfr_knn_impl(
    const nfr_index<T>* h, const T* q, size_t k, float eps, uint32_t* out_idx, T* out_dist)
{
    return dispatch(h, [&](const auto& tree) {
        KNNResultSet<T, uint32_t> result_set(k);
        result_set.init(out_idx, out_dist);
        tree.findNeighbors(result_set, q, SearchParameters(eps, /*sorted=*/true));
        return result_set.size();
    });
}

template <typename T>
size_t nfr_rknn_impl(
    const nfr_index<T>* h, const T* q, size_t k, T radius, float eps, uint32_t* out_idx, T* out_dist)
{
    return dispatch(h, [&](const auto& tree) {
        RKNNResultSet<T, uint32_t> result_set(k, radius);
        result_set.init(out_idx, out_dist);
        tree.findNeighbors(result_set, q, SearchParameters(eps, /*sorted=*/true));
        return result_set.size();
    });
}

template <typename T>
size_t nfr_radius_count_impl(nfr_index<T>* h, const T* q, T radius, int sorted, float eps)
{
    return dispatch(h, [&](const auto& tree) {
        const SearchParameters params(eps, sorted != 0);
        return tree.radiusSearch(q, radius, h->radius_scratch, params);
    });
}

template <typename T>
size_t nfr_radius_fetch_impl(const nfr_index<T>* h, uint32_t* out_idx, T* out_dist, size_t cap)
{
    const size_t n      = h->radius_scratch.size();
    const size_t copy_n = cap < n ? cap : n;
    for (size_t i = 0; i < copy_n; ++i)
    {
        out_idx[i]  = h->radius_scratch[i].first;
        out_dist[i] = h->radius_scratch[i].second;
    }
    return n;
}

template <typename T>
size_t nfr_box_count_impl(nfr_index<T>* h, const T* lo, const T* hi)
{
    return dispatch(h, [&](auto& tree) {
        using TreeT               = std::remove_reference_t<decltype(tree)>;
        typename TreeT::BoundingBox bbox;
        bbox.resize(h->adaptor.dim);
        for (size_t i = 0; i < h->adaptor.dim; ++i)
        {
            bbox[i].low  = lo[i];
            bbox[i].high = hi[i];
        }
        BoxResultSet<uint32_t> result_set(h->box_scratch);
        return tree.findWithinBox(result_set, bbox);
    });
}

template <typename T>
size_t nfr_box_fetch_impl(const nfr_index<T>* h, uint32_t* out_idx, size_t cap)
{
    const size_t n      = h->box_scratch.size();
    const size_t copy_n = cap < n ? cap : n;
    if (copy_n > 0) std::memcpy(out_idx, h->box_scratch.data(), copy_n * sizeof(uint32_t));
    return n;
}

// ---- Fixed DIM=3 fast-path implementations (benchmark-only) --------------

template <typename T>
nfr3_index<T>* nfr3_build_impl(const T* pts, size_t n, size_t leaf_max_size, unsigned n_thread_build)
{
    auto* h    = new nfr3_index<T>();
    h->adaptor = RowMajorAdaptor<T>{pts, n, 3};
    const KDTreeSingleIndexAdaptorParams params(
        leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, n_thread_build);
    h->tree = new Tree3<T>(3, h->adaptor, params);
    return h;
}

template <typename T>
void nfr3_free_impl(nfr3_index<T>* h)
{
    if (!h) return;
    delete h->tree;
    delete h;
}

template <typename T>
size_t nfr3_knn_impl(const nfr3_index<T>* h, const T* q, size_t k, uint32_t* out_idx, T* out_dist)
{
    // Their best-known config: the plain `knnSearch` convenience method
    // (default SearchParameters: eps=0, sorted=true), no eps parameter.
    return h->tree->knnSearch(q, k, out_idx, out_dist);
}

// ---- Dynamic-forest (KDTreeSingleIndexDynamicAdaptor) implementations ----
//
// L2 metric only (M2's dynamic cross-validation scope -- unlike the static
// `nfr_*` surface above, there is no per-metric switch/dispatch here).

// Row-major point-cloud adaptor for the dynamic forest: unlike the static
// `RowMajorAdaptor`, `kdtree_get_point_count()` returns a MUTABLE
// `current_n` rather than a fixed `n`. This is load-bearing: nanoflann's
// `KDTreeSingleIndexDynamicAdaptor` ctor auto-adds every existing point
// (`addPoints(0, kdtree_get_point_count()-1)`) the instant it is
// constructed, with no way to opt out. Starting `current_n` at 0 makes that
// auto-add a no-op, so `nfrd_build_*` always begins with an empty forest and
// the caller (the xval harness, via `nfrd_set_current_n_*` +
// `nfrd_add_points_*`) drives every subsequent insertion explicitly and
// deterministically instead.
template <typename T>
struct RowMajorDynAdaptor
{
    const T* pts        = nullptr;
    size_t   dim        = 0;
    size_t   n_capacity = 0;  // rows physically available in `pts` (n_capacity * dim elements)
    size_t   current_n  = 0;  // logical size reported to nanoflann; grown by nfrd_set_current_n_*

    inline size_t kdtree_get_point_count() const { return current_n; }
    inline T      kdtree_get_pt(const size_t idx, const size_t d) const { return pts[idx * dim + d]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
        return false;
    }
};

template <typename T>
using DynTreeBase = KDTreeSingleIndexDynamicAdaptor<
    L2_Adaptor<T, RowMajorDynAdaptor<T>, T, uint32_t>, RowMajorDynAdaptor<T>, -1, uint32_t>;

// `treeIndex_`, `removedPoints_`, and `treeCount_` are `protected` on
// `KDTreeSingleIndexDynamicAdaptor` (public only to derived classes, not to
// arbitrary external code) -- nanoflann.hpp:2532-2557. A protected member is
// exactly as accessible from a subclass as nanoflann's own design intends,
// so this thin subclass (pure `using`-declarations, no logic) is the
// faithful way to expose them across the FFI boundary, rather than
// recomputing equivalent values from public accessors.
template <typename T>
struct DynTree : public DynTreeBase<T>
{
    using Base = DynTreeBase<T>;
    using Base::Base;
    using Base::pointCount_;
    using Base::removedPoints_;
    using Base::treeCount_;
    using Base::treeIndex_;
};

template <typename T>
struct nfrd_index
{
    RowMajorDynAdaptor<T>                adaptor;
    DynTree<T>*                          tree = nullptr;
    std::vector<ResultItem<uint32_t, T>> radius_scratch;
};

template <typename T>
nfrd_index<T>* nfrd_build_impl(
    const T* pts, size_t n_capacity, int dim, size_t leaf_max_size, size_t maximum_point_count)
{
    auto* h            = new nfrd_index<T>();
    h->adaptor.pts     = pts;
    h->adaptor.dim     = static_cast<size_t>(dim);
    h->adaptor.n_capacity = n_capacity;
    h->adaptor.current_n  = 0;  // ctor's auto-add-existing-points path is therefore a no-op.

    const KDTreeSingleIndexAdaptorParams params(
        leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None, /*n_thread_build=*/1u);
    h->tree = new DynTree<T>(dim, h->adaptor, params, maximum_point_count);
    return h;
}

template <typename T>
void nfrd_free_impl(nfrd_index<T>* h)
{
    if (!h) return;
    delete h->tree;
    delete h;
}

template <typename T>
void nfrd_set_current_n_impl(nfrd_index<T>* h, size_t n)
{
    if (n > h->adaptor.n_capacity)
        throw std::out_of_range("nfrd_set_current_n: n exceeds the buffer capacity passed to nfrd_build");
    h->adaptor.current_n = n;
}

template <typename T>
void nfrd_add_points_impl(nfrd_index<T>* h, uint32_t start, uint32_t end)
{
    // C++'s own addPoints(start, end) -- END-INCLUSIVE (`idx <= end` in the
    // vendored header), unmodified.
    //
    // CONTIGUOUS-APPEND CONTRACT (nanoflann's own usage contract, not
    // something this wrapper imposes -- see nanoflann.hpp:2647-2668): inside
    // addPoints' loop, for every NEW (never-before-added) index `idx` the
    // header does `treeIndex_[pointCount_] = pos`, keyed by the class's own
    // running `pointCount_` counter, NOT by `idx`. That only yields a
    // correct `treeIndex_[idx]` mapping when `idx == pointCount_` at the
    // moment it is processed, i.e. `start` must equal the total count of
    // points ever added so far and every call must be a contiguous,
    // in-order block (reactivating a previously-removed index via
    // addPoints(idx, idx) is exempt -- that path restores from the
    // `removedPoints_` tombstone instead of touching `pointCount_`). This
    // wrapper does not (and per the "no logic beyond plumbing" rule, must
    // not) validate that contract -- a misaligned `start` silently corrupts
    // `treeIndex_` on the C++ side, with no exception and no abort. See
    // `add_points_misaligned_start_documents_silent_corruption` in
    // `tests/oracle_dynamic.rs` for the observed symptom, and the
    // `add_points` doc comment on the Rust wrappers in src/lib.rs for the
    // caller-facing version of this note.
    h->tree->addPoints(start, end);
}

template <typename T>
void nfrd_remove_point_impl(nfrd_index<T>* h, size_t idx)
{
    h->tree->removePoint(idx);
}

template <typename T>
size_t nfrd_knn_impl(
    const nfrd_index<T>* h, const T* q, size_t k, float eps, uint32_t* out_idx, T* out_dist)
{
    KNNResultSet<T, uint32_t> result_set(k);
    result_set.init(out_idx, out_dist);
    h->tree->findNeighbors(result_set, q, SearchParameters(eps, /*sorted=*/true));
    return result_set.size();
}

template <typename T>
size_t nfrd_radius_count_impl(nfrd_index<T>* h, const T* q, T radius_sq, int sorted, float eps)
{
    // Unlike the static tree, KDTreeSingleIndexDynamicAdaptor has no
    // radiusSearch()/radiusSearchCustomCallback() convenience wrapper --
    // only the sub-tree class (KDTreeSingleIndexDynamicAdaptor_) does. So
    // this builds the RadiusResultSet directly and drives it through the
    // outer class's findNeighbors(), exactly mirroring what the sub-tree
    // class's own radiusSearch() does internally
    // (nanoflann.hpp:2463-2471).
    const SearchParameters                   params(eps, sorted != 0);
    RadiusResultSet<T, uint32_t>             result_set(radius_sq, h->radius_scratch);
    h->tree->findNeighbors(result_set, q, params);
    return result_set.size();
}

template <typename T>
size_t nfrd_radius_fetch_impl(const nfrd_index<T>* h, uint32_t* out_idx, T* out_dist, size_t cap)
{
    const size_t n      = h->radius_scratch.size();
    const size_t copy_n = cap < n ? cap : n;
    for (size_t i = 0; i < copy_n; ++i)
    {
        out_idx[i]  = h->radius_scratch[i].first;
        out_dist[i] = h->radius_scratch[i].second;
    }
    return n;
}

template <typename T>
size_t nfrd_tree_count_impl(const nfrd_index<T>* h)
{
    return h->tree->getAllIndices().size();
}

template <typename T>
size_t nfrd_slot_vacc_impl(const nfrd_index<T>* h, size_t slot, uint32_t* out, size_t cap)
{
    // `.at(slot)` bounds-checks and throws `std::out_of_range` on an invalid
    // slot, which the extern-C entry point below catches and turns into an
    // abort (same "no sane per-call recovery" convention as every other
    // non-build entry point in this file).
    const auto&  vacc   = h->tree->getAllIndices().at(slot).vAcc_;
    const size_t n      = vacc.size();
    const size_t copy_n = cap < n ? cap : n;
    if (copy_n > 0) std::memcpy(out, vacc.data(), copy_n * sizeof(uint32_t));
    return n;
}

template <typename T>
size_t nfrd_tree_index_impl(const nfrd_index<T>* h, int32_t* out, size_t cap)
{
    const auto&  ti     = h->tree->treeIndex_;
    const size_t n      = ti.size();
    const size_t copy_n = cap < n ? cap : n;
    for (size_t i = 0; i < copy_n; ++i) out[i] = static_cast<int32_t>(ti[i]);
    return n;
}

template <typename T>
size_t nfrd_removed_count_impl(const nfrd_index<T>* h)
{
    return h->tree->removedPoints_.size();
}

}  // namespace

// ---- extern "C" surface ----------------------------------------------------
//
// One pair of entry points per scalar (f32 -> `_f` suffix, f64 -> `_d`
// suffix), generated by macro to avoid duplicating the (already-generic)
// implementations above.

extern "C"
{
    struct nfr_index_f;
    struct nfr_index_d;
    struct nfr3_index_f;
    struct nfr3_index_d;
    struct nfrd_index_f;
    struct nfrd_index_d;
}

// A C++ exception unwinding across an `extern "C"` boundary into Rust is
// undefined behavior -- Rust has no C++ personality routine to continue the
// unwind through. Every entry point below is therefore wrapped in
// `try { ... } catch (...) { ... }`:
// - `nfr*_build_*`: catch returns a null handle. This is a REPORTABLE
//   failure (allocation failure, an nanoflann-internal throw on
//   pathological input) with an obvious, already-checked signal on the
//   Rust side: `RefIndexF32::build`/`RefIndexF64::build`/`RefIndex3*::build`
//   already `assert!(!handle.is_null(), ...)` on every construction path.
// - every other entry point: there is no sane per-call recovery once an
//   index is already live and mid-query state (radius/box scratch buffers)
//   may be in an inconsistent state, so catch aborts loudly instead of
//   returning a bogus count/silently corrupting caller state.
#define NFR_DEFINE_ENTRY_POINTS(T, SUF)                                                          \
    extern "C" nfr_index_##SUF* nfr_build_##SUF(                                                 \
        const T* pts, size_t n, int dim, int metric, size_t leaf_max_size,                        \
        unsigned n_thread_build)                                                                  \
    {                                                                                             \
        try {                                                                                     \
            return reinterpret_cast<nfr_index_##SUF*>(                                            \
                nfr_build_impl<T>(pts, n, dim, metric, leaf_max_size, n_thread_build));            \
        } catch (...) {                                                                           \
            return nullptr;                                                                       \
        }                                                                                          \
    }                                                                                              \
    extern "C" void nfr_free_##SUF(nfr_index_##SUF* h)                                            \
    {                                                                                              \
        try {                                                                                      \
            nfr_free_impl<T>(reinterpret_cast<nfr_index<T>*>(h));                                  \
        } catch (...) {                                                                            \
            std::abort();                                                                          \
        }                                                                                          \
    }                                                                                              \
    extern "C" size_t nfr_size_##SUF(const nfr_index_##SUF* h)                                    \
    {                                                                                              \
        try {                                                                                      \
            return nfr_size_impl<T>(reinterpret_cast<const nfr_index<T>*>(h));                     \
        } catch (...) {                                                                            \
            std::abort();                                                                          \
        }                                                                                          \
    }                                                                                              \
    extern "C" size_t nfr_used_memory_##SUF(const nfr_index_##SUF* h)                              \
    {                                                                                              \
        try {                                                                                      \
            return nfr_used_memory_impl<T>(reinterpret_cast<const nfr_index<T>*>(h));              \
        } catch (...) {                                                                            \
            std::abort();                                                                          \
        }                                                                                          \
    }                                                                                              \
    extern "C" size_t nfr_vind_##SUF(const nfr_index_##SUF* h, uint32_t* out, size_t cap)          \
    {                                                                                              \
        try {                                                                                      \
            return nfr_vind_impl<T>(reinterpret_cast<const nfr_index<T>*>(h), out, cap);           \
        } catch (...) {                                                                            \
            std::abort();                                                                          \
        }                                                                                          \
    }                                                                                              \
    extern "C" size_t nfr_knn_##SUF(                                                              \
        const nfr_index_##SUF* h, const T* q, size_t k, float eps, uint32_t* out_idx, T* out_dist) \
    {                                                                                              \
        try {                                                                                       \
            return nfr_knn_impl<T>(reinterpret_cast<const nfr_index<T>*>(h), q, k, eps, out_idx, out_dist); \
        } catch (...) {                                                                             \
            std::abort();                                                                           \
        }                                                                                            \
    }                                                                                               \
    extern "C" size_t nfr_rknn_##SUF(                                                              \
        const nfr_index_##SUF* h, const T* q, size_t k, T radius, float eps, uint32_t* out_idx,    \
        T* out_dist)                                                                                \
    {                                                                                               \
        try {                                                                                        \
            return nfr_rknn_impl<T>(                                                                 \
                reinterpret_cast<const nfr_index<T>*>(h), q, k, radius, eps, out_idx, out_dist);     \
        } catch (...) {                                                                              \
            std::abort();                                                                            \
        }                                                                                             \
    }                                                                                                \
    extern "C" size_t nfr_radius_count_##SUF(                                                       \
        nfr_index_##SUF* h, const T* q, T radius, int sorted, float eps)                            \
    {                                                                                                \
        try {                                                                                         \
            return nfr_radius_count_impl<T>(reinterpret_cast<nfr_index<T>*>(h), q, radius, sorted, eps); \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }                                                                                                \
    extern "C" size_t nfr_radius_fetch_##SUF(                                                       \
        const nfr_index_##SUF* h, uint32_t* out_idx, T* out_dist, size_t cap)                       \
    {                                                                                                \
        try {                                                                                         \
            return nfr_radius_fetch_impl<T>(                                                          \
                reinterpret_cast<const nfr_index<T>*>(h), out_idx, out_dist, cap);                    \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }                                                                                                \
    extern "C" size_t nfr_box_count_##SUF(nfr_index_##SUF* h, const T* lo, const T* hi)             \
    {                                                                                                \
        try {                                                                                         \
            return nfr_box_count_impl<T>(reinterpret_cast<nfr_index<T>*>(h), lo, hi);                 \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }                                                                                                \
    extern "C" size_t nfr_box_fetch_##SUF(const nfr_index_##SUF* h, uint32_t* out_idx, size_t cap)  \
    {                                                                                                \
        try {                                                                                         \
            return nfr_box_fetch_impl<T>(reinterpret_cast<const nfr_index<T>*>(h), out_idx, cap);     \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }                                                                                                \
    extern "C" nfr3_index_##SUF* nfr3_build_##SUF(                                                  \
        const T* pts, size_t n, size_t leaf_max_size, unsigned n_thread_build)                       \
    {                                                                                                \
        try {                                                                                         \
            return reinterpret_cast<nfr3_index_##SUF*>(                                               \
                nfr3_build_impl<T>(pts, n, leaf_max_size, n_thread_build));                           \
        } catch (...) {                                                                               \
            return nullptr;                                                                           \
        }                                                                                              \
    }                                                                                                \
    extern "C" void nfr3_free_##SUF(nfr3_index_##SUF* h)                                             \
    {                                                                                                \
        try {                                                                                         \
            nfr3_free_impl<T>(reinterpret_cast<nfr3_index<T>*>(h));                                   \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }                                                                                                \
    extern "C" size_t nfr3_knn_##SUF(                                                                \
        const nfr3_index_##SUF* h, const T* q, size_t k, uint32_t* out_idx, T* out_dist)             \
    {                                                                                                \
        try {                                                                                         \
            return nfr3_knn_impl<T>(reinterpret_cast<const nfr3_index<T>*>(h), q, k, out_idx, out_dist); \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                              \
    }

NFR_DEFINE_ENTRY_POINTS(float, f)
NFR_DEFINE_ENTRY_POINTS(double, d)

#undef NFR_DEFINE_ENTRY_POINTS

// ---- Dynamic-forest extern "C" surface (nfrd_*) ---------------------------
//
// Same exception-safety convention as NFR_DEFINE_ENTRY_POINTS above:
// `nfrd_build_*` catches and returns null (checked by the Rust
// `RefDynIndexF32::build`/`RefDynIndexF64::build` asserts); every other
// entry point catches and aborts, since there is no sane per-call recovery
// once an index is live.
#define NFRD_DEFINE_ENTRY_POINTS(T, SUF)                                                         \
    extern "C" nfrd_index_##SUF* nfrd_build_##SUF(                                               \
        const T* pts, size_t n_capacity, int dim, size_t leaf_max_size,                          \
        size_t maximum_point_count)                                                              \
    {                                                                                             \
        try {                                                                                     \
            return reinterpret_cast<nfrd_index_##SUF*>(                                           \
                nfrd_build_impl<T>(pts, n_capacity, dim, leaf_max_size, maximum_point_count));     \
        } catch (...) {                                                                           \
            return nullptr;                                                                       \
        }                                                                                         \
    }                                                                                              \
    extern "C" void nfrd_free_##SUF(nfrd_index_##SUF* h)                                          \
    {                                                                                              \
        try {                                                                                      \
            nfrd_free_impl<T>(reinterpret_cast<nfrd_index<T>*>(h));                                \
        } catch (...) {                                                                            \
            std::abort();                                                                          \
        }                                                                                          \
    }                                                                                               \
    extern "C" void nfrd_set_current_n_##SUF(nfrd_index_##SUF* h, size_t n)                        \
    {                                                                                               \
        try {                                                                                       \
            nfrd_set_current_n_impl<T>(reinterpret_cast<nfrd_index<T>*>(h), n);                     \
        } catch (...) {                                                                             \
            std::abort();                                                                           \
        }                                                                                           \
    }                                                                                                \
    extern "C" void nfrd_add_points_##SUF(nfrd_index_##SUF* h, uint32_t start, uint32_t end)         \
    {                                                                                                \
        try {                                                                                        \
            nfrd_add_points_impl<T>(reinterpret_cast<nfrd_index<T>*>(h), start, end);                \
        } catch (...) {                                                                              \
            std::abort();                                                                            \
        }                                                                                            \
    }                                                                                                 \
    extern "C" void nfrd_remove_point_##SUF(nfrd_index_##SUF* h, size_t idx)                          \
    {                                                                                                 \
        try {                                                                                         \
            nfrd_remove_point_impl<T>(reinterpret_cast<nfrd_index<T>*>(h), idx);                      \
        } catch (...) {                                                                               \
            std::abort();                                                                             \
        }                                                                                             \
    }                                                                                                 \
    extern "C" size_t nfrd_knn_##SUF(                                                                \
        const nfrd_index_##SUF* h, const T* q, size_t k, float eps, uint32_t* out_idx, T* out_dist)  \
    {                                                                                                 \
        try {                                                                                          \
            return nfrd_knn_impl<T>(reinterpret_cast<const nfrd_index<T>*>(h), q, k, eps, out_idx, out_dist); \
        } catch (...) {                                                                                \
            std::abort();                                                                              \
        }                                                                                               \
    }                                                                                                    \
    extern "C" size_t nfrd_radius_count_##SUF(                                                          \
        nfrd_index_##SUF* h, const T* q, T radius_sq, int sorted, float eps)                            \
    {                                                                                                    \
        try {                                                                                             \
            return nfrd_radius_count_impl<T>(reinterpret_cast<nfrd_index<T>*>(h), q, radius_sq, sorted, eps); \
        } catch (...) {                                                                                    \
            std::abort();                                                                                  \
        }                                                                                                   \
    }                                                                                                        \
    extern "C" size_t nfrd_radius_fetch_##SUF(                                                              \
        const nfrd_index_##SUF* h, uint32_t* out_idx, T* out_dist, size_t cap)                               \
    {                                                                                                        \
        try {                                                                                                  \
            return nfrd_radius_fetch_impl<T>(                                                                  \
                reinterpret_cast<const nfrd_index<T>*>(h), out_idx, out_dist, cap);                            \
        } catch (...) {                                                                                        \
            std::abort();                                                                                      \
        }                                                                                                       \
    }                                                                                                            \
    extern "C" size_t nfrd_tree_count_##SUF(const nfrd_index_##SUF* h)                                           \
    {                                                                                                            \
        try {                                                                                                     \
            return nfrd_tree_count_impl<T>(reinterpret_cast<const nfrd_index<T>*>(h));                            \
        } catch (...) {                                                                                            \
            std::abort();                                                                                          \
        }                                                                                                           \
    }                                                                                                                \
    extern "C" size_t nfrd_slot_vacc_##SUF(                                                                          \
        const nfrd_index_##SUF* h, size_t slot, uint32_t* out, size_t cap)                                           \
    {                                                                                                                \
        try {                                                                                                         \
            return nfrd_slot_vacc_impl<T>(reinterpret_cast<const nfrd_index<T>*>(h), slot, out, cap);                 \
        } catch (...) {                                                                                                \
            std::abort();                                                                                              \
        }                                                                                                               \
    }                                                                                                                    \
    extern "C" size_t nfrd_tree_index_##SUF(const nfrd_index_##SUF* h, int32_t* out, size_t cap)                         \
    {                                                                                                                     \
        try {                                                                                                              \
            return nfrd_tree_index_impl<T>(reinterpret_cast<const nfrd_index<T>*>(h), out, cap);                          \
        } catch (...) {                                                                                                    \
            std::abort();                                                                                                  \
        }                                                                                                                   \
    }                                                                                                                        \
    extern "C" size_t nfrd_removed_count_##SUF(const nfrd_index_##SUF* h)                                                    \
    {                                                                                                                        \
        try {                                                                                                                 \
            return nfrd_removed_count_impl<T>(reinterpret_cast<const nfrd_index<T>*>(h));                                     \
        } catch (...) {                                                                                                       \
            std::abort();                                                                                                     \
        }                                                                                                                      \
    }

NFRD_DEFINE_ENTRY_POINTS(float, f)
NFRD_DEFINE_ENTRY_POINTS(double, d)

#undef NFRD_DEFINE_ENTRY_POINTS
