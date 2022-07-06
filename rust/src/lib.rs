// This needs to be a square logo to avoid stretching, and can have transparency
#![doc(html_logo_url = "https://www.nrel.gov/transportation/assets/images/icon-fastsim.jpg")]
//! Documentation for the Rust backend of the Future Automotive Systems Technology Simulator (FASTSim).

//! # Overview
//! FASTSim provides a simple way to compare powertrains and estimate the impact of technology
//! improvements on light-, medium-, and heavy-duty vehicle efficiency, performance, cost, and battery life.  
//! More information here: <https://www.nrel.gov/transportation/fastsim.html>

//! # Installation
//! Currently, the Rust backend is only available through a Python API.  
//! For installation instructions, see: <https://github.nrel.gov/MBAP/fastsim/blob/rust-port/fastsim/docs/README.md>

//! # Python Examples
//! ```python
//! import fastsim
//!
//! ## Load drive cycle by name
//! cyc_py = fastsim.cycle.Cycle.from_file("udds")
//! cyc_rust = cyc_py.to_rust()
//!
//! ## Load vehicle using database vehicle ID number
//! vnum = 1  
//! veh_py = fastsim.vehicle.Vehicle.from_vehdb(vnum)
//! veh_rust = veh_py.to_rust()
//!
//! ## Simulate
//! sd = fastsim.RustSimDrive(cyc_rust, veh_rust)
//! sd.sim_drive()
//! ```

extern crate ndarray;
use pyo3::prelude::*;

extern crate proc_macros;

pub mod params;
use params::RustPhysicalProperties;
pub mod utils;
use utils::{Pyo3ArrayBool, Pyo3ArrayF64, Pyo3ArrayU32, Pyo3VecF64};
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
    m.add_class::<Pyo3ArrayU32>()?;
    m.add_class::<Pyo3ArrayF64>()?;
    m.add_class::<Pyo3ArrayBool>()?;
    m.add_class::<Pyo3VecF64>()?;
    m.add_class::<RustSimDriveParams>()?;
    m.add_class::<RustSimDrive>()?;
    cycle::register(py, m)?;
    Ok(())
}
