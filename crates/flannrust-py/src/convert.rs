//! numpy <-> `flannrust` marshalling helpers. No `unsafe`: every array is
//! read via `PyReadonlyArray` (dtype/bounds-checked by the `numpy` crate).
//!
//! Optimizations over the naive per-element copy:
//! - `to_ndarray` caches the numpy module in a `GILOnceCell` (avoids a
//!   `PyModule::import("numpy")` call per query).
//! - `as_rows_2d` uses `as_slice().to_vec()` (single memcpy) for
//!   C-contiguous arrays; falls back to per-element copy for strided views.

use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyAny;

static NUMPY_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

fn numpy<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyModule>> {
    if let Some(m) = NUMPY_MODULE.get(py) {
        return Ok(m.bind(py));
    }
    let m = PyModule::import(py, "numpy")?;
    Ok(NUMPY_MODULE.get_or_init(py, || m.unbind()).bind(py))
}

/// Normalizes `x` to a numpy array via `numpy.asarray` -- accepts an
/// existing ndarray (no-op, no-copy) as well as nested lists/tuples
/// (`np.ascontiguousarray`-equivalent semantics per the design spec).
pub fn to_ndarray<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    numpy(py)?.call_method1("asarray", (x,))
}

/// `x` must already be a numpy array (see [`to_ndarray`]), 1-D `(d,)` or
/// 2-D `(m, d)`. Returns the row-major point data, point count `m` (`1`
/// for a 1-D input) and per-point dimension `d`. `T`'s dtype must match
/// `x`'s exactly (no silent widening/narrowing) -- a mismatch is a
/// `TypeError`; a wrong rank is a `ValueError`.
pub fn as_rows_2d<'py, T: Element + Copy>(
    x: &Bound<'py, PyAny>,
) -> PyResult<(Vec<T>, usize, usize)> {
    let untyped = x
        .cast::<PyUntypedArray>()
        .map_err(|_| PyValueError::new_err("expected a numpy array"))?;
    match untyped.ndim() {
        1 => {
            let a: PyReadonlyArray1<T> = x.extract().map_err(|_| dtype_mismatch::<T>())?;
            let view = a.as_array();
            let d = view.len();
            let v = match view.as_slice() {
                Some(s) => s.to_vec(),
                None => view.iter().copied().collect(),
            };
            Ok((v, 1, d))
        }
        2 => {
            let a: PyReadonlyArray2<T> = x.extract().map_err(|_| dtype_mismatch::<T>())?;
            let view = a.as_array();
            let m = view.shape()[0];
            let d = view.shape()[1];
            let v = match view.as_slice() {
                Some(s) => s.to_vec(),
                None => {
                    let mut v = Vec::with_capacity(m * d);
                    for row in view.outer_iter() {
                        v.extend(row.iter().copied());
                    }
                    v
                }
            };
            Ok((v, m, d))
        }
        n => Err(PyValueError::new_err(format!(
            "expected a 1-D or 2-D array, got {n}-D"
        ))),
    }
}

/// `x` must already be a numpy array. Rank only (no dtype extraction) --
/// used by callers that need to decide `(d,)`-vs-`(m,d)` output shaping
/// before/without pulling the data out via [`as_rows_2d`].
pub fn ndim(x: &Bound<'_, PyAny>) -> PyResult<usize> {
    let untyped = x
        .cast::<PyUntypedArray>()
        .map_err(|_| PyValueError::new_err("expected a numpy array"))?;
    Ok(untyped.ndim())
}

fn dtype_mismatch<T>() -> PyErr {
    PyTypeError::new_err(format!(
        "array dtype does not match the tree's dtype ({})",
        std::any::type_name::<T>()
    ))
}

/// Caps a validated `workers > 1` count at the number of available CPUs
/// before it's handed to `rayon::ThreadPoolBuilder`. Without this, a caller
/// passing an absurd value (e.g. `workers=2**40`) makes rayon spend minutes
/// spawning threads and then panic building the pool -- since a query can
/// never usefully run on more threads than there are cores, silently
/// clamping is correct behavior, not a validation error.
pub fn capped_workers(workers: i64) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    (workers as usize).min(cpus)
}
