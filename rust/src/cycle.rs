use pyo3::prelude::*;
extern crate ndarray;
use ndarray::{Array, Array1}; 

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data
pub struct Cycle{
    /// array of time [s]
    cycSecs: Array1<f64>,
    /// array of speed [m/s]
    cycMps: Array1<f64>,    
    /// array of grade
    cycGrade: Array1<f64>
}


