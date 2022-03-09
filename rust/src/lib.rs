extern crate ndarray;
use pyo3::prelude::*;

pub mod params;
use params::*;
pub mod utils;
pub mod cycle;
use cycle::*;
pub mod vehicle;
use vehicle::*;



/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustCycle>()?;
    m.add_class::<RustVehicle>()?;
    m.add_class::<RustPhysicalProperties>()?;
    // m.add_class::<SimDrive>()?;
    Ok(())
}
