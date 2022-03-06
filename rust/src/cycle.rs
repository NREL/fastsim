extern crate ndarray;
use ndarray::{Array, Array1, concatenate}; 
extern crate numpy;
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
pub struct Cycle{
    /// array of time [s]
    time_s: Array1<f64>,
    /// array of speed [m/s]
    mps: Array1<f64>,    
    /// array of grade [rise/run]
    grade: Array1<f64>,
    /// array of max possible charge rate from roadway
    // road_type: Array1<bool>, // TODO: implement this
    #[pyo3(get, set)]
    name: String    
}


#[pymethods]
/// Cycle class for containing: 
/// -- time_s, 
/// -- mps (speed [m/s])
/// -- grade [rise/run]
/// -- road_type (this is legacy and will likely change to road charging capacity [kW])
impl Cycle{
    #[new]
    pub fn __new__(time_s: Vec<f64>, mps: Vec<f64>, grade: Vec<f64>, road_type: Vec<bool>, name:String) -> Self{
        let time_s = Array::from_vec(time_s);
        let mps = Array::from_vec(mps);
        let grade = Array::from_vec(grade);
        let road_type = Array::from_vec(road_type);
        Cycle {time_s, mps, grade, road_type, name}
    }
    #[getter]
    pub fn get_mph(&self) -> PyResult<Vec<f64>>{
        Ok((self.mps * MPH_PER_MPS).to_vec())
    }    
    #[setter]
    pub fn set_mph(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.mps = Array::from_vec(new_value / params.mphPerMps);
        OK(())
    }
    #[getter]
    /// array of sim time stamps
    pub fn get_time_s(&self) -> PyResult<Vec<f64>>{
        Ok(self.time_s.to_vec())
    }
    #[setter]
    pub fn set_time_s(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.time = Array::from_vec(new_value);
        Ok(())
    }
    #[getter]
    /// array of time steps
    pub fn get_dt_s(&self) -> PyResult<Vec<f64>>{
    }
    /// rust-internal time steps
    pub fn dt_s(&self) -> Array<f64> {
        self.get_dt_s()
    }
    /// distance covered in each time step
    pub fn dist_m(self) -> Array<f64>{
        self.mps * diff(self.time_s)
    }

    // pub fn delta_elev_m(self):
    //     """
    //      elevation change w.r.t. to initial
    //     """
    //     return (self.dist_m * self.cycGrade).cumsum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist() {
        let time_s = Array1::<f64>::range(0.0, 10.0, 1.0);
        let speed_mps = Array1::<f64>::range(0.0, 10.0, 1.0);
        let grade = Array::zeros(10);
        let name = String::from("test");
        let cyc = Cycle::__new__(time_s, speed_mps, grade, name);
        assert_eq!(diff());
    }
}