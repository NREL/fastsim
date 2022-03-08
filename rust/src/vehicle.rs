extern crate ndarray;
use ndarray::{Array, Array1}; 
extern crate pyo3;
use pyo3::prelude::*;
// use numpy::pyo3::Python;
// use numpy::ndarray::array;
// use numpy::{ToPyArray, PyArray};

// local 
use super::params::*;
use super::utils::*;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct Vehcile{
}