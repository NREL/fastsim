extern crate ndarray;

use std::fs::File;
use std::path::PathBuf;

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
    // #[new]
    // pub fn __new__(time_s: Vec<f64>, mps: Vec<f64>, grade: Vec<f64>, road_type:Vec<f64>, name:String) -> Self{
    //     let time_s = Array::from_vec(time_s);
    //     let mps = Array::from_vec(mps);
    //     let grade = Array::from_vec(grade);
    //     let road_type = Array::from_vec(road_type);
    //     RustCycle {time_s, mps, grade, road_type, name}
    // }

    #[new]
    pub fn __new__(pathstr: String) -> Self {
        Self::from_file(pathstr).unwrap()
    }
    
    fn len(&self) -> usize{
        self.time_s.len()
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
    pub fn new(time_s: Vec<f64>, mps: Vec<f64>, grade: Vec<f64>, road_type:Vec<f64>, name:String) -> Self{
        let time_s = Array::from_vec(time_s);
        let mps = Array::from_vec(mps);
        let grade = Array::from_vec(grade);
        let road_type = Array::from_vec(road_type);
        RustCycle {time_s, mps, grade, road_type, name}
    }

    /// rust-internal time steps
    pub fn dt_s(&self) -> Array1<f64> {
        diff(&self.time_s)
    }
    /// distance covered in each time step
    pub fn dist_m(&self) -> Array1<f64>{
        &self.mps * self.dt_s()
    }

    /// get mph from self.mps
    pub fn mph(&self) -> Array1<f64> {
        &self.mps * MPH_PER_MPS
    }

    /// Load cycle from csv file
    pub fn from_file(pathstr: String) -> Result<RustCycle, String> {
        // TODO: figure out how to make this a pymethod 
        // needs trait implementation for String to work as argument
        let pathbuf = PathBuf::from(pathstr);
        if pathbuf.exists() {
            let mut time_s = Vec::<f64>::new();
            let mut speed_mps = Vec::<f64>::new();
            let mut grade = Vec::<f64>::new();
            let mut road_type = Vec::<f64>::new();
            let name = String::from(pathbuf.file_stem().unwrap().to_str().unwrap());
            let file = File::open(pathbuf).expect("an existing file");
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(file);
            for result in rdr.records() {
                let record = result.expect("a proper csv row");
                time_s.push(record[0].parse::<f64>().unwrap());
                speed_mps.push(record[1].parse::<f64>().unwrap());
                grade.push(record[2].parse::<f64>().unwrap());
                road_type.push(record[3].parse::<f64>().unwrap());
            }
            Ok(RustCycle::new(time_s, speed_mps, grade, road_type, name))    
        } else {
            Err(String::from("path doesn't exist"))
        }
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
        let road_type = Array::zeros(10).to_vec();        
        let name = String::from("test");
        let cyc = RustCycle::new(time_s, speed_mps, grade, road_type, name);
        assert_eq!(cyc.dist_m().sum(), 45.0);
    }

    #[test]
    fn test_loading_a_cycle_from_the_filesystem() {
        let pathstr = String::from("../fastsim/resources/cycles/udds.csv");
        let expected_udds_length: usize = 1370;
        match RustCycle::from_file(pathstr) {
            Ok(cyc) => {
                assert_eq!(cyc.name, String::from("udds"));
                let num_entries = cyc.time_s.len();
                assert!(num_entries > 0);
                assert_eq!(num_entries, cyc.time_s.len());
                assert_eq!(num_entries, cyc.mps.len());
                assert_eq!(num_entries, cyc.grade.len());
                assert_eq!(num_entries, cyc.road_type.len());
                assert_eq!(num_entries, expected_udds_length);
            },
            Err(s) => panic!("{}", s),
        }
    }
}