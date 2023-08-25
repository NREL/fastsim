//! Crate that wraps `fastsim-core` and enables the `pyo3` feature to
//! expose most structs, methods, and functions to Python.

use fastsim_core::prelude::*;
pub use pyo3::exceptions::{
    PyAttributeError, PyFileNotFoundError, PyIndexError, PyNotImplementedError, PyRuntimeError,
};
pub use pyo3::prelude::*;
pub use pyo3::types::PyType;
pub use pyo3_polars::PyDataFrame;

#[pymodule]
fn fastsim_core_py(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<FuelConverter>()?;
    m.add_class::<FuelConverterState>()?;
    m.add_class::<FuelConverterStateHistoryVec>()?;
    m.add_class::<ReversibleEnergyStorage>()?;
    m.add_class::<ReversibleEnergyStorageState>()?;
    m.add_class::<ReversibleEnergyStorageStateHistoryVec>()?;
    m.add_class::<ElectricMachine>()?;
    m.add_class::<ElectricMachineState>()?;
    m.add_class::<ElectricMachineStateHistoryVec>()?;

    m.add_class::<Pyo3VecWrapper>()?;
    m.add_class::<Pyo3Vec2Wrapper>()?;
    m.add_class::<Pyo3Vec3Wrapper>()?;
    m.add_class::<Pyo3VecBoolWrapper>()?;

    Ok(())
}
