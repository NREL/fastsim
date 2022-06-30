extern crate ndarray;
use pyo3::prelude::*;

extern crate proc_macros;

pub mod params;
use params::RustPhysicalProperties;
pub mod utils;
use utils::{Pyo3ArrayU32,Pyo3ArrayF64, Pyo3ArrayBool, Pyo3VecF64};
pub mod cycle;
use cycle::RustCycle;
pub mod vehicle;
use vehicle::RustVehicle;
pub mod simdrive;
use simdrive::{RustSimDrive, RustSimDriveParams};
pub mod air;
pub mod simdrive_impl;
pub mod thermal;

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustCycle>()?;
    m.add_class::<RustVehicle>()?;
    m.add_class::<RustPhysicalProperties>()?;
    m.add_class::<Pyo3ArrayU32>()?;
    m.add_class::<Pyo3ArrayF64>()?;
    m.add_class::<Pyo3ArrayBool>()?;
    m.add_class::<Pyo3VecF64>()?;
    m.add_class::<RustSimDriveParams>()?;
    m.add_class::<RustSimDrive>()?;
    cycle::register(py, m)?;
    Ok(())
}
