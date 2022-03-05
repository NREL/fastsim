extern crate ndarray;
use pyo3::prelude::*;

mod cycle;
use cycle::*;



/// Function for adding Rust structs as Python Classes
#[pymodule]
fn rusty_train(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Cycle>()?;
    // m.add_class::<Vehicle>()?;
    // m.add_class::<SimDrive>()?;
    Ok(())
}
