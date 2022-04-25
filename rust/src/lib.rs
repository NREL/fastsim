extern crate ndarray;
use pyo3::prelude::*;

pub mod params;
use params::RustPhysicalProperties;
pub mod utils;
pub mod cycle;
use cycle::RustCycle;
pub mod vehicle;
use vehicle::RustVehicle;
pub mod simdrive;
use simdrive::{RustSimDrive, RustSimDriveParams};
pub mod simdrive_impl;

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustCycle>()?;
    m.add_class::<RustVehicle>()?;
    m.add_class::<RustPhysicalProperties>()?;
    m.add_class::<RustSimDriveParams>()?;
    m.add_class::<RustSimDrive>()?;
    m.add_class::<utils::Pyo3ArrayBool>()?;
    m.add_class::<utils::Pyo3ArrayF64>()?;
    m.add_class::<utils::Pyo3ArrayU32>()?;
    cycle::register(py, m)?;
    Ok(())
}
