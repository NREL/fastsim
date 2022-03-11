extern crate ndarray;
// use ndarray::{Array, Array1}; 
extern crate pyo3;
use pyo3::prelude::*;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustSimDriveParams{
    missed_trace_correction:bool, // if True, missed trace correction is active, default = False
}

#[pymethods]
impl RustSimDriveParams{
    #[new]
    pub fn __new__(
    ) -> Self{
        let missed_trace_correction = false;
        RustSimDriveParams{
            missed_trace_correction
        }
    }
}
