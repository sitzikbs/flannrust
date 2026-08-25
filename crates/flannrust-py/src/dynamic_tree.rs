//! The Python `DynamicKDTree` class: a growable/shrinkable (Bentley-Saxe
//! forest) kd-tree over an owned `OwnedRows` dataset, dispatched at runtime
//! over the same dtype x dimensionality x metric matrix as the static
//! [`crate::static_tree::KDTree`] via the [`DynTree`] enum and the
//! [`for_each_dyn_variant`] macro.

use flannrust::{ConstDim, Dim, Distance, DynDim, DynamicKdTree, DynamicKdTreeBuilder, L1, L2, L2Simple, OwnedRows, ResultItem, Scalar, SearchParams};
use numpy::{Element, IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::convert::{as_rows_2d, capped_workers, ndim, to_ndarray};

/// The same 10-way monomorphization matrix as [`crate::static_tree::StaticTree`],
/// wrapping [`DynamicKdTree`] instead of the static `KdTree`.
enum DynTree {
    F32D2(DynamicKdTree<f32, ConstDim<2>, OwnedRows<f32>, L2>),
    F32D3(DynamicKdTree<f32, ConstDim<3>, OwnedRows<f32>, L2>),
    F64D2(DynamicKdTree<f64, ConstDim<2>, OwnedRows<f64>, L2>),
    F64D3(DynamicKdTree<f64, ConstDim<3>, OwnedRows<f64>, L2>),
    F32Dyn(DynamicKdTree<f32, DynDim, OwnedRows<f32>, L2>),
    F64Dyn(DynamicKdTree<f64, DynDim, OwnedRows<f64>, L2>),
    F32DynL1(DynamicKdTree<f32, DynDim, OwnedRows<f32>, L1>),
    F64DynL1(DynamicKdTree<f64, DynDim, OwnedRows<f64>, L1>),
    F32DynL2s(DynamicKdTree<f32, DynDim, OwnedRows<f32>, L2Simple>),
    F64DynL2s(DynamicKdTree<f64, DynDim, OwnedRows<f64>, L2Simple>),
}

/// Same dispatch shape as `static_tree`'s `for_each_variant!`, over
/// [`DynTree`] instead. Works for both `&self.tree` and `&mut self.tree`
/// call sites -- the match arms bind whatever mutability the input
/// reference has.
macro_rules! for_each_dyn_variant {
    ($self:expr, $t:ident => $body:expr) => {
        match $self {
            DynTree::F32D2($t) => $body,
            DynTree::F32D3($t) => $body,
            DynTree::F64D2($t) => $body,
            DynTree::F64D3($t) => $body,
            DynTree::F32Dyn($t) => $body,
            DynTree::F64Dyn($t) => $body,
            DynTree::F32DynL1($t) => $body,
            DynTree::F64DynL1($t) => $body,
            DynTree::F32DynL2s($t) => $body,
            DynTree::F64DynL2s($t) => $body,
        }
    };
}

/// `flannrust.DynamicKDTree(dim, dtype="float32", leaf_size=10, metric="l2", capacity=None)`.
///
/// A growable/shrinkable kd-tree: points are appended via `add_points` and
/// lazily tombstoned via `remove_point`. Distances (and `query(..., r=...)`)
/// are SQUARED for `l2`/`l2_simple`, same convention as the static `KDTree`.
#[pyclass(module = "flannrust")]
pub struct DynamicKDTree {
    tree: DynTree,
    dim: usize,
    leaf_size: usize,
    metric: String,
    dtype: &'static str,
    capacity: Option<usize>,
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

fn parse_dtype(dtype: &str) -> PyResult<&'static str> {
    match dtype {
        "float32" => Ok("float32"),
        "float64" => Ok("float64"),
        other => Err(PyValueError::new_err(format!("dtype must be \"float32\" or \"float64\" (got {other:?})"))),
    }
}

/// Builds one monomorphized, initially-empty `DynamicKdTree` variant's
/// worth of state -- shared by every dtype x dim x metric combination in
/// [`build_f32`]/[`build_f64`].
fn build_dyn<T, D, M>(dim: D, dataset: OwnedRows<T>, leaf_size: usize, metric: M, capacity: Option<usize>) -> DynamicKdTree<T, D, OwnedRows<T>, M>
where
    T: Scalar,
    D: Dim,
    M: Distance<T>,
    L2: Distance<T>,
{
    let mut builder = DynamicKdTreeBuilder::new(dim, dataset).with_metric(metric).leaf_max_size(leaf_size);
    if let Some(cap) = capacity {
        builder = builder.maximum_point_count(cap);
    }
    builder.build()
}

fn build_f32(metric: &str, d: usize, leaf_size: usize, capacity: Option<usize>) -> DynTree {
    let ds = OwnedRows::<f32>::with_capacity(d, capacity.unwrap_or(0));
    match (metric, d) {
        ("l2", 2) => DynTree::F32D2(build_dyn(ConstDim::<2>, ds, leaf_size, L2, capacity)),
        ("l2", 3) => DynTree::F32D3(build_dyn(ConstDim::<3>, ds, leaf_size, L2, capacity)),
        ("l2", _) => DynTree::F32Dyn(build_dyn(DynDim(d), ds, leaf_size, L2, capacity)),
        ("l1", _) => DynTree::F32DynL1(build_dyn(DynDim(d), ds, leaf_size, L1, capacity)),
        ("l2_simple", _) => DynTree::F32DynL2s(build_dyn(DynDim(d), ds, leaf_size, L2Simple, capacity)),
        _ => unreachable!("parse_metric only returns l2/l1/l2_simple"),
    }
}

fn build_f64(metric: &str, d: usize, leaf_size: usize, capacity: Option<usize>) -> DynTree {
    let ds = OwnedRows::<f64>::with_capacity(d, capacity.unwrap_or(0));
    match (metric, d) {
        ("l2", 2) => DynTree::F64D2(build_dyn(ConstDim::<2>, ds, leaf_size, L2, capacity)),
        ("l2", 3) => DynTree::F64D3(build_dyn(ConstDim::<3>, ds, leaf_size, L2, capacity)),
        ("l2", _) => DynTree::F64Dyn(build_dyn(DynDim(d), ds, leaf_size, L2, capacity)),
        ("l1", _) => DynTree::F64DynL1(build_dyn(DynDim(d), ds, leaf_size, L1, capacity)),
        ("l2_simple", _) => DynTree::F64DynL2s(build_dyn(DynDim(d), ds, leaf_size, L2Simple, capacity)),
        _ => unreachable!("parse_metric only returns l2/l1/l2_simple"),
    }
}

#[pymethods]
impl DynamicKDTree {
    #[new]
    #[pyo3(signature = (dim, dtype="float32", leaf_size=10, metric="l2", capacity=None))]
    fn new(dim: usize, dtype: &str, leaf_size: usize, metric: &str, capacity: Option<usize>) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be >= 1"));
        }
        if leaf_size == 0 {
            return Err(PyValueError::new_err("leaf_size must be >= 1"));
        }
        if capacity == Some(0) {
            return Err(PyValueError::new_err("capacity must be >= 1"));
        }
        let metric_norm = parse_metric(metric)?;
        let dtype_norm = parse_dtype(dtype)?;

        let tree = match dtype_norm {
            "float32" => build_f32(metric_norm, dim, leaf_size, capacity),
            "float64" => build_f64(metric_norm, dim, leaf_size, capacity),
            _ => unreachable!("parse_dtype only returns float32/float64"),
        };

        Ok(DynamicKDTree { tree, dim, leaf_size, metric: metric_norm.to_string(), dtype: dtype_norm, capacity })
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

    #[getter]
    fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Number of currently-live (non-removed) points.
    #[getter]
    fn n_active(&self) -> usize {
        for_each_dyn_variant!(&self.tree, t => t.active_count())
    }

    /// Number of dataset rows ever added (including removed/tombstoned
    /// ones) -- the dataset's own length.
    #[getter]
    fn n_total(&self) -> usize {
        for_each_dyn_variant!(&self.tree, t => t.dataset().len())
    }

    /// Number of currently-removed (tombstoned) points.
    #[getter]
    fn removed_len(&self) -> usize {
        for_each_dyn_variant!(&self.tree, t => t.removed_len())
    }

    /// Appends `points` (a `(d,)` single point or `(m, d)` batch) to the
    /// dataset and the forest, returning the newly-added HALF-OPEN index
    /// range `(start, end)` (`points[i]` landed at dataset index
    /// `start + i`). An empty `(0, d)` batch is a no-op returning
    /// `(n_total, n_total)`.
    fn add_points<'py>(&mut self, py: Python<'py>, points: &Bound<'py, PyAny>) -> PyResult<(usize, usize)> {
        let arr = to_ndarray(py, points)?;
        let expected_dim = self.dim;
        let capacity = self.capacity;
        for_each_dyn_variant!(&mut self.tree, t => do_add_points(t, &arr, expected_dim, capacity))
    }

    /// Lazily removes the point at dataset index `idx`. Returns `False`
    /// (no-op) if `idx` was never added or is already removed.
    fn remove_point(&mut self, idx: usize) -> bool {
        for_each_dyn_variant!(&mut self.tree, t => t.remove_point(idx))
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

        for_each_dyn_variant!(&self.tree, t => do_query(py, t, &arr, k, r, eps, workers, is_1d))
    }

    /// `r` is the SQUARED radius for `l2`/`l2_simple` (unsquared for `l1`),
    /// strict `<` -- same convention as `query`'s `r=`. Returns two lists
    /// (length `m`, or `1` for a `(d,)` query) of ragged 1-D arrays:
    /// `(idxs, dists)`. `sorted=False` returns traversal order.
    #[pyo3(signature = (x, r, sorted=true, eps=0.0, workers=1))]
    fn query_radius<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>, r: f64, sorted: bool, eps: f32, workers: i64) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        if !(workers == -1 || workers >= 1) {
            return Err(PyValueError::new_err(format!("workers must be -1 or >= 1 (got {workers})")));
        }

        let arr = to_ndarray(py, x)?;
        for_each_dyn_variant!(&self.tree, t => do_query_radius(py, t, &arr, r, sorted, eps, workers))
    }
}

/// Pushes `points` onto `tree`'s dataset and calls the core's (END-
/// INCLUSIVE) `add_points`, converting to/from the Python-facing HALF-OPEN
/// convention. An empty batch is short-circuited BEFORE touching the core
/// (`push_rows`/`add_points` are never called). A `capacity` overflow is
/// also rejected before touching the core.
fn do_add_points<T, D, M>(tree: &mut DynamicKdTree<T, D, OwnedRows<T>, M>, arr: &Bound<'_, PyAny>, expected_dim: usize, capacity: Option<usize>) -> PyResult<(usize, usize)>
where
    T: Scalar + Element,
    D: Dim,
    M: Distance<T>,
{
    let (data, m, d) = as_rows_2d::<T>(arr)?;
    if d != expected_dim {
        return Err(PyValueError::new_err(format!("points has dim {d}, tree has dim {expected_dim}")));
    }

    let start = tree.dataset().len();
    if m == 0 {
        return Ok((start, start));
    }

    if let Some(cap) = capacity {
        if start + m > cap {
            return Err(PyValueError::new_err(format!(
                "add_points: adding {m} point(s) to {start} would exceed capacity {cap}"
            )));
        }
    }

    tree.dataset_mut().push_rows(&data);
    let end_inclusive = start + m - 1;
    tree.add_points(start, end_inclusive);
    Ok((start, start + m))
}

/// Runs `tree.knn_search_with`/`rknn_search_with`, returning `(dists, idxs)`
/// as numpy arrays shaped `(k,)` (`is_1d`) or `(m, k)`. Same padding
/// convention as the static tree's `do_query` (`pad_idx = n_total`,
/// `pad_dist = inf`).
#[allow(clippy::too_many_arguments)]
fn do_query<'py, T, D, M>(
    py: Python<'py>,
    tree: &DynamicKdTree<T, D, OwnedRows<T>, M>,
    x: &Bound<'py, PyAny>,
    k: usize,
    r: Option<f64>,
    eps: f32,
    workers: i64,
    is_1d: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: Scalar + Element,
    D: Dim,
    M: Distance<T, DistanceType = T> + Sync,
{
    let (query_data, m, d) = as_rows_2d::<T>(x)?;
    let expected_dim = tree.dataset().dim();
    if d != expected_dim {
        return Err(PyValueError::new_err(format!("x has dim {d}, tree has dim {expected_dim}")));
    }

    let n = tree.dataset().len();
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
                let pool = rayon::ThreadPoolBuilder::new().num_threads(capped_workers(workers)).build().expect("thread pool build");
                pool.install(run);
            } else {
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

/// Runs `tree.radius_search_with`, returning `(idxs, dists)` as two Python
/// lists of length `m` (ragged 1-D numpy arrays per query row).
fn do_query_radius<'py, T, D, M>(
    py: Python<'py>,
    tree: &DynamicKdTree<T, D, OwnedRows<T>, M>,
    x: &Bound<'py, PyAny>,
    r: f64,
    sorted: bool,
    eps: f32,
    workers: i64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: Scalar + Element,
    D: Dim,
    M: Distance<T, DistanceType = T> + Sync,
{
    let (query_data, m, d) = as_rows_2d::<T>(x)?;
    let expected_dim = tree.dataset().dim();
    if d != expected_dim {
        return Err(PyValueError::new_err(format!("x has dim {d}, tree has dim {expected_dim}")));
    }

    let radius = T::from_f64(r);
    let params = SearchParams { eps, sorted };
    let mut results: Vec<Vec<ResultItem<u32, T>>> = vec![Vec::new(); m];

    py.detach(|| {
        let search_one = |q: &[T], out: &mut Vec<ResultItem<u32, T>>| {
            tree.radius_search_with(q, radius, out, &params);
        };
        if workers == 1 {
            for (out, q) in results.iter_mut().zip(query_data.chunks(d)) {
                search_one(q, out);
            }
        } else {
            let mut run = || {
                results.par_iter_mut().zip(query_data.par_chunks(d)).for_each(|(out, q)| search_one(q, out));
            };
            if workers > 1 {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(capped_workers(workers)).build().expect("thread pool build");
                pool.install(run);
            } else {
                run();
            }
        }
    });

    let idx_list = PyList::empty(py);
    let dist_list = PyList::empty(py);
    for res in results {
        let mut idxs = Vec::with_capacity(res.len());
        let mut dists = Vec::with_capacity(res.len());
        for item in res {
            idxs.push(item.index);
            dists.push(item.distance);
        }
        idx_list.append(idxs.into_pyarray(py))?;
        dist_list.append(dists.into_pyarray(py))?;
    }

    Ok((idx_list.into_any().unbind(), dist_list.into_any().unbind()))
}
