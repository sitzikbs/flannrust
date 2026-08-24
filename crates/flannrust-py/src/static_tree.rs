//! The Python `KDTree` class: a static kd-tree over a copied-in
//! `OwnedRows` dataset, dispatched at runtime over a small monomorphization
//! matrix (dtype x dimensionality x metric) via the [`StaticTree`] enum and
//! the [`for_each_variant`] macro.

use std::num::NonZeroU32;
use std::ops::Deref;

use flannrust::{BuildThreads, ConstDim, Distance, DynDim, KdTree, KdTreeBuilder, L1, L2, L2Simple, OwnedRows, Scalar, SearchParams};
use numpy::{Element, IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::convert::{as_rows_2d, ndim, to_ndarray};

/// The 10-way monomorphization matrix: `{f32, f64} x {ConstDim<2>,
/// ConstDim<3>, DynDim}` for `L2`, plus `{f32, f64} x DynDim` for `L1`/
/// `L2Simple` (the const-dim fast path is only wired up for `L2` -- see the
/// design spec's dispatch table).
enum StaticTree {
    F32D2(KdTree<f32, ConstDim<2>, OwnedRows<f32>, L2>),
    F32D3(KdTree<f32, ConstDim<3>, OwnedRows<f32>, L2>),
    F64D2(KdTree<f64, ConstDim<2>, OwnedRows<f64>, L2>),
    F64D3(KdTree<f64, ConstDim<3>, OwnedRows<f64>, L2>),
    F32Dyn(KdTree<f32, DynDim, OwnedRows<f32>, L2>),
    F64Dyn(KdTree<f64, DynDim, OwnedRows<f64>, L2>),
    F32DynL1(KdTree<f32, DynDim, OwnedRows<f32>, L1>),
    F64DynL1(KdTree<f64, DynDim, OwnedRows<f64>, L1>),
    F32DynL2s(KdTree<f32, DynDim, OwnedRows<f32>, L2Simple>),
    F64DynL2s(KdTree<f64, DynDim, OwnedRows<f64>, L2Simple>),
}

/// Dispatches `$body` (an expression using `$t: &KdTree<T, D, OwnedRows<T>, M>`,
/// borrowed from `$self`) over every [`StaticTree`] variant. `T`/`D`/`M` are
/// concrete (monomorphized) inside each arm, so `$body` may call any
/// `KdTree`/`DataSource` method directly; it must still type-check
/// identically (typically via a `T`-generic helper function) across all ten
/// arms since a `match` requires one common result type.
macro_rules! for_each_variant {
    ($self:expr, $t:ident => $body:expr) => {
        match $self {
            StaticTree::F32D2($t) => $body,
            StaticTree::F32D3($t) => $body,
            StaticTree::F64D2($t) => $body,
            StaticTree::F64D3($t) => $body,
            StaticTree::F32Dyn($t) => $body,
            StaticTree::F64Dyn($t) => $body,
            StaticTree::F32DynL1($t) => $body,
            StaticTree::F64DynL1($t) => $body,
            StaticTree::F32DynL2s($t) => $body,
            StaticTree::F64DynL2s($t) => $body,
        }
    };
}

/// `flannrust.KDTree(points, leaf_size=10, metric="l2", threads=None)`.
///
/// Distances (and the `query(..., r=...)` radius argument) are SQUARED for
/// `l2`/`l2_simple` -- NOT euclidean like `scipy.spatial.cKDTree`. `l1` is
/// the summed absolute value (already unsquared).
#[pyclass(module = "flannrust")]
pub struct KDTree {
    tree: StaticTree,
    n: usize,
    dim: usize,
    leaf_size: usize,
    metric: String,
    dtype: &'static str,
}

fn parse_metric(metric: &str) -> PyResult<&'static str> {
    match metric.to_lowercase().as_str() {
        "l2" => Ok("l2"),
        "l1" => Ok("l1"),
        "l2_simple" => Ok("l2_simple"),
        other => Err(PyValueError::new_err(format!(
            "metric must be one of \"l2\", \"l1\", \"l2_simple\" (got {other:?})"
        ))),
    }
}

fn parse_threads(threads: Option<i64>) -> PyResult<BuildThreads> {
    match threads {
        None => Ok(BuildThreads::Auto),
        Some(1) => Ok(BuildThreads::Sequential),
        Some(n) if n > 1 => Ok(BuildThreads::Threads(NonZeroU32::new(n as u32).unwrap())),
        Some(n) => Err(PyValueError::new_err(format!(
            "threads must be None or a positive integer (got {n})"
        ))),
    }
}

/// Builds one monomorphized `KdTree` variant's worth of state. Shared by
/// every dtype x dim x metric combination in [`KDTree::new`].
fn build_tree<T, D, M>(dim: D, data: Vec<T>, row_dim: usize, leaf_size: usize, metric: M, threads: BuildThreads) -> KdTree<T, D, OwnedRows<T>, M>
where
    T: Scalar,
    D: flannrust::Dim,
    M: Distance<T>,
    L2: Distance<T>,
{
    let dataset = OwnedRows::new(data, row_dim);
    KdTreeBuilder::new(dim, dataset)
        .with_metric(metric)
        .leaf_max_size(leaf_size)
        .threads(threads)
        .build()
}

#[pymethods]
impl KDTree {
    #[new]
    #[pyo3(signature = (points, leaf_size=10, metric="l2", threads=None))]
    fn new(py: Python<'_>, points: &Bound<'_, PyAny>, leaf_size: usize, metric: &str, threads: Option<i64>) -> PyResult<Self> {
        if leaf_size == 0 {
            return Err(PyValueError::new_err("leaf_size must be >= 1"));
        }
        let metric_norm = parse_metric(metric)?;
        let build_threads = parse_threads(threads)?;

        let arr = to_ndarray(py, points)?;
        let rank = ndim(&arr)?;
        if rank != 2 {
            return Err(PyValueError::new_err(format!("points must be 2-D (n, d), got {rank}-D")));
        }

        // Try f32 then f64; dtype must match one of them exactly (checked
        // by `as_rows_2d`'s underlying `PyReadonlyArray` extraction).
        if let Ok((data, n, d)) = as_rows_2d::<f32>(&arr) {
            if d == 0 {
                return Err(PyValueError::new_err("points must have dim >= 1"));
            }
            let tree = build_f32(metric_norm, d, data, leaf_size, build_threads);
            return Ok(KDTree { tree, n, dim: d, leaf_size, metric: metric_norm.to_string(), dtype: "float32" });
        }
        if let Ok((data, n, d)) = as_rows_2d::<f64>(&arr) {
            if d == 0 {
                return Err(PyValueError::new_err("points must have dim >= 1"));
            }
            let tree = build_f64(metric_norm, d, data, leaf_size, build_threads);
            return Ok(KDTree { tree, n, dim: d, leaf_size, metric: metric_norm.to_string(), dtype: "float64" });
        }
        Err(PyTypeError::new_err("points dtype must be float32 or float64"))
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn dtype(&self) -> &str {
        self.dtype
    }

    #[getter]
    fn leaf_size(&self) -> usize {
        self.leaf_size
    }

    #[getter]
    fn metric(&self) -> &str {
        &self.metric
    }

    /// A read-only numpy `(n, dim)` copy of the owned dataset.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        for_each_variant!(&self.tree, t => data_array(py, t))
    }

    #[pyo3(signature = (x, k=1, r=None, eps=0.0, workers=1))]
    fn query<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>, k: usize, r: Option<f64>, eps: f32, workers: i64) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        if !(workers == -1 || workers >= 1) {
            return Err(PyValueError::new_err(format!("workers must be -1 or >= 1 (got {workers})")));
        }

        let arr = to_ndarray(py, x)?;
        let is_1d = ndim(&arr)? == 1;

        for_each_variant!(&self.tree, t => do_query(py, t, &arr, k, r, eps, workers, is_1d))
    }
}

fn build_f32(metric: &str, d: usize, data: Vec<f32>, leaf_size: usize, threads: BuildThreads) -> StaticTree {
    match (metric, d) {
        ("l2", 2) => StaticTree::F32D2(build_tree(ConstDim::<2>, data, d, leaf_size, L2, threads)),
        ("l2", 3) => StaticTree::F32D3(build_tree(ConstDim::<3>, data, d, leaf_size, L2, threads)),
        ("l2", _) => StaticTree::F32Dyn(build_tree(DynDim(d), data, d, leaf_size, L2, threads)),
        ("l1", _) => StaticTree::F32DynL1(build_tree(DynDim(d), data, d, leaf_size, L1, threads)),
        ("l2_simple", _) => StaticTree::F32DynL2s(build_tree(DynDim(d), data, d, leaf_size, L2Simple, threads)),
        _ => unreachable!("parse_metric only returns l2/l1/l2_simple"),
    }
}

fn build_f64(metric: &str, d: usize, data: Vec<f64>, leaf_size: usize, threads: BuildThreads) -> StaticTree {
    match (metric, d) {
        ("l2", 2) => StaticTree::F64D2(build_tree(ConstDim::<2>, data, d, leaf_size, L2, threads)),
        ("l2", 3) => StaticTree::F64D3(build_tree(ConstDim::<3>, data, d, leaf_size, L2, threads)),
        ("l2", _) => StaticTree::F64Dyn(build_tree(DynDim(d), data, d, leaf_size, L2, threads)),
        ("l1", _) => StaticTree::F64DynL1(build_tree(DynDim(d), data, d, leaf_size, L1, threads)),
        ("l2_simple", _) => StaticTree::F64DynL2s(build_tree(DynDim(d), data, d, leaf_size, L2Simple, threads)),
        _ => unreachable!("parse_metric only returns l2/l1/l2_simple"),
    }
}

/// A read-only numpy `(n, dim)` copy of `tree`'s dataset.
fn data_array<'py, T, D, M>(py: Python<'py>, tree: &KdTree<T, D, OwnedRows<T>, M>) -> PyResult<Py<PyAny>>
where
    T: Scalar + Element,
    D: flannrust::Dim,
    M: Distance<T>,
{
    let ds = tree.dataset();
    let flat: Vec<T> = ds.as_slice().to_vec();
    let n = ds.len();
    let dim = ds.dim();
    let flat_arr = PyArray1::from_vec(py, flat);
    let arr2d = flat_arr.reshape((n, dim))?;
    let readonly = arr2d.into_readwrite().make_nonwriteable();
    Ok(readonly.deref().clone().into_any().unbind())
}

/// Runs `tree.query(x, k, r, eps, workers)`, returning `(dists, idxs)` as
/// numpy arrays shaped `(k,)` (`is_1d`) or `(m, k)`.
#[allow(clippy::too_many_arguments)]
fn do_query<'py, T, D, M>(
    py: Python<'py>,
    tree: &KdTree<T, D, OwnedRows<T>, M>,
    x: &Bound<'py, PyAny>,
    k: usize,
    r: Option<f64>,
    eps: f32,
    workers: i64,
    is_1d: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: Scalar + Element,
    D: flannrust::Dim,
    M: Distance<T, DistanceType = T> + Sync,
{
    let (query_data, m, d) = as_rows_2d::<T>(x)?;
    let expected_dim = tree.dim();
    if d != expected_dim {
        return Err(PyValueError::new_err(format!("x has dim {d}, tree has dim {expected_dim}")));
    }

    let n = tree.size();
    let pad_idx = u32::try_from(n).unwrap_or(u32::MAX);
    let pad_dist = T::from_f64(f64::INFINITY);

    let mut out_idx = vec![pad_idx; m * k];
    let mut out_dist = vec![pad_dist; m * k];
    let params = SearchParams { eps, sorted: true };

    let search_one = |q: &[T], oi: &mut [u32], od: &mut [T]| {
        if let Some(radius) = r {
            tree.rknn_search_with(q, T::from_f64(radius), oi, od, &params);
        } else {
            tree.knn_search_with(q, oi, od, &params);
        }
    };

    py.detach(|| {
        if workers == 1 {
            for ((oi, od), q) in out_idx.chunks_mut(k).zip(out_dist.chunks_mut(k)).zip(query_data.chunks(d)) {
                search_one(q, oi, od);
            }
        } else {
            let mut run = || {
                out_idx
                    .par_chunks_mut(k)
                    .zip(out_dist.par_chunks_mut(k))
                    .zip(query_data.par_chunks(d))
                    .for_each(|((oi, od), q)| search_one(q, oi, od));
            };
            if workers > 1 {
                // Cap at exactly `workers` rayon workers via a scoped pool.
                let pool = rayon::ThreadPoolBuilder::new().num_threads(workers as usize).build().expect("thread pool build");
                pool.install(run);
            } else {
                // workers == -1: the ambient/global rayon pool (all cores).
                run();
            }
        }
    });

    let idx_arr = out_idx.into_pyarray(py);
    let dist_arr = out_dist.into_pyarray(py);
    if is_1d {
        Ok((dist_arr.into_any().unbind(), idx_arr.into_any().unbind()))
    } else {
        let idx2d = idx_arr.reshape((m, k))?;
        let dist2d = dist_arr.reshape((m, k))?;
        Ok((dist2d.into_any().unbind(), idx2d.into_any().unbind()))
    }
}
