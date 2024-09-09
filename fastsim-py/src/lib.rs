//! Crate that wraps `fastsim-core` and enables the `pyo3` feature to
//! expose most structs, methods, and functions to Python.

use fastsim_core::air_properties::get_density_air_py;
use fastsim_core::prelude::*;
pub use pyo3::exceptions::{
    PyAttributeError, PyFileNotFoundError, PyIndexError, PyNotImplementedError, PyRuntimeError,
};
pub use pyo3::prelude::*;
pub use pyo3::types::PyType;

#[pymodule]
fn fastsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FuelConverter>()?;
    m.add_class::<FuelConverterState>()?;
    m.add_class::<FuelConverterStateHistoryVec>()?;
    m.add_class::<ReversibleEnergyStorage>()?;
    m.add_class::<ReversibleEnergyStorageState>()?;
    m.add_class::<ReversibleEnergyStorageStateHistoryVec>()?;
    m.add_class::<ElectricMachine>()?;
    m.add_class::<ElectricMachineState>()?;
    m.add_class::<ElectricMachineStateHistoryVec>()?;
    m.add_class::<Cycle>()?;
    m.add_class::<CycleElement>()?;
    m.add_class::<Vehicle>()?;
    m.add_class::<SimDrive>()?;
    m.add_class::<fastsim_2::simdrive::RustSimDrive>()?;
    m.add_class::<Pyo3VecWrapper>()?;
    m.add_class::<Pyo3Vec2Wrapper>()?;
    m.add_class::<Pyo3Vec3Wrapper>()?;
    m.add_class::<Pyo3VecBoolWrapper>()?;
    m.add_function(wrap_pyfunction!(get_density_air_py, m)?)?;

    // List enabled features
    m.add_function(wrap_pyfunction!(fastsim_core::enabled_features, m)?)?;

    // initialize logging
    #[cfg(feature = "logging")]
    m.add_function(wrap_pyfunction!(pyo3_log_init, m)?)?;

    Ok(())
}

#[cfg_attr(feature = "logging", pyfunction)]
fn pyo3_log_init() {
    #[cfg(feature = "logging")]
    pyo3_log::init();
}
