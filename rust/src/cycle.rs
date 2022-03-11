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
pub struct RustCycle{
    /// array of time [s]
    pub time_s: Array1<f64>,
    /// array of speed [m/s]
    pub mps: Array1<f64>,    
    /// array of grade [rise/run]
    pub grade: Array1<f64>,
    /// array of max possible charge rate from roadway
    pub road_type: Array1<f64>, 
    #[pyo3(get, set)]
    name: String    
}


/// RustCycle class for containing: 
/// -- time_s, 
/// -- mps (speed [m/s])
/// -- grade [rise/run]
/// -- road_type (this is legacy and will likely change to road charging capacity [kW])
#[pymethods]
impl RustCycle{
    #[new]
    pub fn __new__(time_s: Vec<f64>, mps: Vec<f64>, grade: Vec<f64>, road_type:Vec<f64>, name:String) -> Self{
        let time_s = Array::from_vec(time_s);
        let mps = Array::from_vec(mps);
        let grade = Array::from_vec(grade);
        let road_type = Array::from_vec(road_type);
        RustCycle {time_s, mps, grade, road_type, name}
    }
    #[getter]
    pub fn get_mps(&self) -> PyResult<Vec<f64>>{
        Ok((&self.mps).to_vec())
    }    
    #[setter]
    pub fn set_mps(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.mps = Array::from_vec(new_value);
        Ok(())
    }
    #[getter]
    pub fn get_grade(&self) -> PyResult<Vec<f64>>{
        Ok((&self.grade).to_vec())
    }    
    #[setter]
    pub fn set_grade(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.grade = Array::from_vec(new_value);
        Ok(())
    }
    #[getter]
    pub fn get_road_type(&self) -> PyResult<Vec<f64>>{
        Ok((&self.road_type).to_vec())
    }    
    #[setter]
    pub fn set_road_type(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.road_type = Array::from_vec(new_value);
        Ok(())
    }
    #[getter]
    pub fn get_mph(&self) -> PyResult<Vec<f64>>{
        Ok((&self.mps * MPH_PER_MPS).to_vec())
    }    
    #[setter]
    pub fn set_mph(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.mps = Array::from_vec(new_value) / MPH_PER_MPS;
        Ok(())
    }
    #[getter]
    /// array of sim time stamps
    pub fn get_time_s(&self) -> PyResult<Vec<f64>>{
        Ok(self.time_s.to_vec())
    }
    #[setter]
    pub fn set_time_s(&mut self, new_value:Vec<f64>) -> PyResult<()>{
        self.time_s = Array::from_vec(new_value);
        Ok(())
    }
    #[getter]
    /// array of time steps
    pub fn get_dt_s(&self) -> PyResult<Vec<f64>>{
        Ok(self.dt_s().to_vec())
    }
    #[getter]
    /// cycle length
    pub fn get_len(&self) -> PyResult<usize>{
        Ok(self.time_s.len())
    }
}

/// pure Rust methods that need to be separate due to pymethods incompatibility
impl RustCycle{
    /// rust-internal time steps
    pub fn dt_s(&self) -> Array1<f64> {
        diff(&self.time_s)
    }
    /// distance covered in each time step
    pub fn dist_m(&self) -> Array1<f64>{
        &self.mps * self.dt_s()
    }

    // pub fn delta_elev_m(self):
    //     """
    //      elevation change w.r.t. to initial
    //     """
    //     return (self.dist_m * self.cycGrade).cumsum() // TODO: find a good way to implement cumsum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist() {
        let time_s = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let speed_mps = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let grade = Array::zeros(10).to_vec();
        let name = String::from("test");
        let cyc = RustCycle::__new__(time_s, speed_mps, grade, name);
        assert_eq!(cyc.dist_m().sum(), 45.0);
    }
}