//! # Crate features
//! * **full** - When enabled (which is default), include additional capabilities that
//!   require additional dependencies
//! * **resources** - When enabled (which is triggered by enabling full (thus default)
//!   or enabling this feature directly), compiles commonly used resources (e.g.
//!   standard drive cycles) for faster access.

use fastsim_core::*;
use pyo3imports::*;

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "full")]
    pyo3_log::init();
    m.add_class::<cycle::RustCycle>()?;
    m.add_class::<vehicle::RustVehicle>()?;
    m.add_class::<params::RustPhysicalProperties>()?;
    m.add_class::<params::AdjCoef>()?;
    m.add_class::<params::RustLongParams>()?;
    m.add_class::<utils::Pyo3ArrayU32>()?;
    m.add_class::<utils::Pyo3ArrayF64>()?;
    m.add_class::<utils::Pyo3ArrayBool>()?;
    m.add_class::<utils::Pyo3VecF64>()?;
    m.add_class::<simdrive::RustSimDriveParams>()?;
    m.add_class::<simdrive::RustSimDrive>()?;
    m.add_class::<thermal::SimDriveHot>()?;
    m.add_class::<vehicle_thermal::VehicleThermal>()?;
    m.add_class::<thermal::ThermalState>()?;
    m.add_class::<vehicle_thermal::HVACModel>()?;
    m.add_class::<vehicle_import::OtherVehicleInputs>()?;
    m.add_class::<simdrivelabel::LabelFe>()?;
    m.add_class::<simdrivelabel::LabelFePHEV>()?;
    m.add_class::<simdrivelabel::PHEVCycleCalc>()?;
    m.add_class::<simdrive::simdrive_iter::SimDriveVec>()?;
    fastsim_core::pyfunctions::add_pyfunctions(m)?;
    cycle::register(py, m)?;
    Ok(())
}
