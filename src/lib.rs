//! # Damavand
//!
//! `damavand` is an HPC quantum circuit simulator.
//! It allows to simulate a quantum circuit

pub mod qubit_backend;
pub mod utils;

#[cfg(feature = "profiling")]
pub mod profiler;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qubit_backend::circuit::Circuit;

#[pyfunction]
pub fn initialize_mpi() {
    mpi::initialize();
}

#[pymodule]
fn damavand(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Circuit>()?;
    m.add_wrapped(wrap_pyfunction!(initialize_mpi))?;
    Ok(())
}
