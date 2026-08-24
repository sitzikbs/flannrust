//! Python bindings for `flannrust`, imported as `import flannrust`.
//!
//! # Distances are SQUARED
//!
//! `L2`/`L2Simple` results (and the `r=` radius argument) are the squared
//! Euclidean distance, matching the Rust core and nanoflann -- NOT the
//! euclidean distance `scipy.spatial.cKDTree` returns. `L1` is the summed
//! absolute value (already unsquared).

use pyo3::prelude::*;

mod convert;
mod dynamic_tree;
mod static_tree;

use dynamic_tree::DynamicKDTree;
use static_tree::KDTree;

#[pymodule]
fn flannrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KDTree>()?;
    m.add_class::<DynamicKDTree>()?;
    Ok(())
}
