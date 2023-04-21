use fastsim_core::*;

use pyo3imports::*;

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<cycle::Cycle>()?;
    m.add_class::<vehicle::RustVehicle>()?;
    m.add_class::<params::PhysicalProperties>()?;
    m.add_class::<utils::Pyo3ArrayU32>()?;
    m.add_class::<utils::Pyo3ArrayF64>()?;
    m.add_class::<utils::Pyo3ArrayBool>()?;
    m.add_class::<utils::Pyo3VecF64>()?;
    m.add_class::<simdrive::SimDriveParams>()?;
    m.add_class::<simdrive::SimDrive>()?;
    m.add_class::<thermal::SimDriveHot>()?;
    m.add_class::<vehicle::vehicle_thermal::VehicleThermal>()?;
    m.add_class::<thermal::ThermalState>()?;
    m.add_class::<vehicle::vehicle_thermal::HVACModel>()?;
    cycle::register(py, m)?;
    utils::register(py, m)?;
    m.add_function(wrap_pyfunction!(
        vehicle::vehicle_utils::abc_to_drag_coeffs,
        m
    )?)?;
    Ok(())
}
