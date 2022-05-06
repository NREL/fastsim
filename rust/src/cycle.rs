extern crate ndarray;

use std::fs::File;
use std::path::PathBuf;
use std::collections::HashMap;

use ndarray::{Axis, Array, Array1, s, concatenate}; 
extern crate pyo3;
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyType;
// use numpy::pyo3::Python;
// use numpy::ndarray::array;
// use numpy::{ToPyArray, PyArray};

// local 
use super::params::*;
use super::utils::*;

#[pyfunction]
/// Num Num Num Num Num Int -> (Dict 'jerk_m__s3' Num 'accel_m__s2' Num)
/// INPUTS:
/// - n: Int, number of time-steps away from rendezvous
/// - D0: Num, distance of simulated vehicle (m/s)
/// - v0: Num, speed of simulated vehicle (m/s)
/// - Dr: Num, distance of rendezvous point (m)
/// - vr: Num, speed of rendezvous point (m/s)
/// - dt: Num, step duration (s)
/// RETURNS: (Tuple 'jerk_m__s3': Num, 'accel_m__s2': Num)
/// Returns the constant jerk and acceleration for initial time step.
pub fn calc_constant_jerk_trajectory(n: usize, d0:f64, v0:f64, dr:f64, vr:f64, dt:f64)->(f64, f64){
    assert!(n > 1);
    assert!(dr > d0);
    let n = n as f64;
    let ddr = dr - d0;
    let dvr = vr - v0;
    let k = (dvr - (2.0 * ddr / (n * dt)) + 2.0 * v0) / (
        0.5 * n * (n - 1.0) * dt
        - (1.0 / 3.0) * (n - 1.0) * (n - 2.0) * dt
        - 0.5 * (n - 1.0) * dt * dt
    );
    let a0 = (
        (ddr / dt)
        - n * v0
        - ((1.0 / 6.0) * n * (n - 1.0) * (n - 2.0) * dt + 0.25 * n * (n - 1.0) * dt * dt) * k
    ) / (0.5 * n * n * dt);
    (k, a0)
}

#[pyfunction]
/// Calculate distance (m) after n timesteps
/// INPUTS:
/// - n: Int, numer of timesteps away to calculate
/// - d0: Num, initial distance (m)
/// - v0: Num, initial speed (m/s)
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk
/// - dt: Num, duration of a timestep (s)
/// NOTE:
/// - this is the distance traveled from start (i.e., n=0) measured at sample point n
/// RETURN: Num, the distance at n timesteps away (m)
pub fn dist_for_constant_jerk(n:usize, d0:f64, v0:f64, a0:f64, k:f64, dt:f64) -> f64 {
    let n = n as f64;
    let term1 = dt * (
        (n * v0)
        + (0.5 * n * (n - 1.0) * a0 * dt)
        + ((1.0 / 6.0) * k * dt * (n - 2.0) * (n - 1.0) * n)
    );
    let term2 = 0.5 * dt * dt * ((n * a0) + (0.5 * n * (n - 1.0) * k * dt));
    d0 + term1 + term2
}

#[pyfunction]
/// Calculate speed (m/s) n timesteps away via a constant-jerk acceleration
/// INPUTS:
/// - n: Int, numer of timesteps away to calculate
/// - v0: Num, initial speed (m/s)
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk
/// - dt: Num, duration of a timestep (s)
/// NOTE:
/// - this is the speed at sample n
/// - if n == 0, speed is v0
/// - if n == 1, speed is v0 + a0*dt, etc.
/// RETURN: Num, the speed n timesteps away (m/s)
pub fn speed_for_constant_jerk(n:usize, v0:f64, a0:f64, k:f64, dt:f64)->f64 {
    let n = n as f64;
    v0 + (n * a0 * dt) + (0.5 * n * (n - 1.0) * k * dt)
}

#[pyfunction]
/// Calculate the acceleration n timesteps away
/// INPUTS:
/// - n: Int, number of times steps away to calculate
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk (m/s3)
/// - dt: Num, time-step duration in seconds
/// NOTE:
/// - this is the constant acceleration over the time-step from sample n to sample n+1
/// RETURN: Num, the acceleration n timesteps away (m/s2)
pub fn accel_for_constant_jerk(n:usize, a0:f64, k:f64, dt:f64) -> f64 {
    let n = n as f64;
    a0 + (n * k * dt)
}

pub fn accel_array_for_constant_jerk(nmax:usize, a0:f64, k:f64, dt:f64) -> Array1::<f64> {
    let mut accels: Vec<f64> = Vec::new();
    for n in 0..nmax {
        accels.push(accel_for_constant_jerk(n, a0, k, dt));
    }
    Array1::from_vec(accels)
}

pub(crate) fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_constant_jerk_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(accel_for_constant_jerk, m)?)?;
    m.add_function(wrap_pyfunction!(speed_for_constant_jerk, m)?)?;
    m.add_function(wrap_pyfunction!(dist_for_constant_jerk, m)?)?;
    Ok(())
}

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

    #[classmethod]
    pub fn from_file_py(_cls: &PyType, pathstr: String) -> PyResult<RustCycle> {
        match Self::from_file(&pathstr) {
            Ok(cyc) => Ok(cyc),
            Err(msg) => Err(PyFileNotFoundError::new_err(msg))
        }
    }

    /// Return a HashMap representing the cycle
    pub fn get_cyc_dict(&self) -> PyResult<HashMap<String, Vec<f64>>>{
        let dict: HashMap<String, Vec<f64>> = HashMap::from([
            ("time_s".to_string(), self.time_s.to_vec()),
            ("mps".to_string(), self.mps.to_vec()),
            ("grade".to_string(), self.grade.to_vec()),
            ("road_type".to_string(), self.road_type.to_vec()),
        ]);
        Ok(dict)
    }
    
    pub fn len(&self) -> usize{
        self.time_s.len()
    }

    pub fn copy(&self) -> PyResult<RustCycle>{
        let time_s = self.time_s.clone();
        let mps = self.mps.clone();
        let grade = self.grade.clone();
        let road_type = self.road_type.clone();
        let name = self.name.clone();
        Ok(RustCycle {time_s, mps, grade, road_type, name})
    }

    pub fn modify_by_const_jerk_trajectory(&mut self, idx:usize, n:usize, jerk_m_per_s3:f64, accel0_m_per_s2:f64)->PyResult<f64>{
        Ok(self.modify_by_const_jerk_trajectory_rust(idx, n, jerk_m_per_s3, accel0_m_per_s2))
    }

    pub fn modify_with_braking_trajectory(&mut self, brake_accel_m_per_s2:f64, idx:usize)->PyResult<f64>{
        Ok(self.modify_with_braking_trajectory_rust(brake_accel_m_per_s2, idx))
    }

    pub fn calc_distance_to_next_stop_from(&self, distance_m: f64) -> PyResult<f64> {
        Ok(self.calc_distance_to_next_stop_from_rust(distance_m))
    }

    pub fn grade_at_distance(&self, distance_m: f64) -> PyResult<f64> {
        Ok(self.grade_at_distance_rust(distance_m))
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
    #[getter]
    /// distance for each time-step in meters
    pub fn get_dist_m(&self) -> PyResult<Vec<f64>>{
        Ok(self.dist_m().to_vec())
    }
    #[getter]
    /// the average speeds over each step in meters per second
    pub fn get_avg_mps(&self) -> PyResult<Vec<f64>>{
        Ok(self.avg_mps().to_vec())
    }
    #[getter]
    /// distance for each time-step in meters based on step-average speed
    pub fn get_dist_v2_m(&self) -> PyResult<Vec<f64>>{
        Ok(self.dist_v2_m().to_vec())
    }
    #[getter]
    pub fn get_delta_elev_m(&self) -> PyResult<Vec<f64>>{
        Ok(self.delta_elev_m().to_vec())
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

    pub fn test_cyc() -> Self {
        let time_s = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let speed_mps = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let grade = Array::zeros(10).to_vec();
        let road_type = Array::zeros(10).to_vec();        
        let name = String::from("test");
        RustCycle::new(time_s, speed_mps, grade, road_type, name)    
    }

    pub fn total_distance_traveled(&self, idx: usize) -> f64 {
        let len = self.time_s.len();
        let mut total_dist_m = 0.0;
        let end_idx = if idx >= len {
            if len == 0 {
                0
            } else {
                len - 1
            }
        } else {
            idx
        };
        for i in 0..end_idx {
            let avg_speed_m_per_s = 0.5 * (self.mps[i+1] + self.mps[i]);
            let dt_s = self.time_s[i+1] - self.time_s[i];
            let dd_m = avg_speed_m_per_s * dt_s;
            total_dist_m += dd_m;
        }
        total_dist_m
    }

    /// Returns the grade at the given distance
    pub fn grade_at_distance_rust(&self, distance_m: f64) -> f64 {
        let delta_dists_m: Array1<f64> = self.dist_v2_m();
        if distance_m <= 0.0 {
            return self.grade[0];
        }
        if distance_m >= delta_dists_m.sum() {
            return self.grade[self.grade.len()-1];
        }
        let mut dist_mark: f64 = 0.0;
        let mut last_grade: f64 = self.grade[0];
        for idx in 0..self.grade.len() {
            let dd = delta_dists_m[idx];
            if (dist_mark <= distance_m) && ((dist_mark + dd) > distance_m) {
                return last_grade;
            }
            dist_mark += dd;
            last_grade = self.grade[idx];
        }
        return last_grade;
    }

    /// Calculate the distance to next stop from `distance_m`
    /// - distance_m: non-negative-number, the current distance from start (m)
    /// RETURN: -1 or non-negative-integer
    /// - if there are no more stops ahead, return -1
    /// - else returns the distance to the next stop from distance_m
    pub fn calc_distance_to_next_stop_from_rust(&self, distance_m: f64) -> f64 {
        let tol: f64 = 1e-6;
        let not_found: f64 = -1.0;
        let mut d: f64 = 0.0;
        for (&dd, &v) in self.dist_v2_m().iter().zip(self.mps.iter()) {
            d += dd;
            if (v < tol) && (d > (distance_m + tol) ){
                return d - distance_m;
            }
        }
        not_found
    }



    /// Modifies the cycle using the given constant-jerk trajectory parameters
    /// - idx: non-negative integer, the point in the cycle to initiate
    ///   modification (note: THIS point is modified since trajectory should be
    ///   calculated from idx-1)
    /// - n: non-negative integer, the number of steps ahead
    /// - jerk_m__s3: number, the "Jerk" associated with the trajectory (m/s3)
    /// - accel0_m__s2: number, the initial acceleration (m/s2)
    /// NOTE:
    /// - modifies cyc in place to hit any critical rendezvous_points by a trajectory adjustment
    /// - CAUTION: NOT ROBUST AGAINST VARIABLE DURATION TIME-STEPS
    /// RETURN: Number, final modified speed (m/s)
    pub fn modify_by_const_jerk_trajectory_rust(&mut self, i:usize, n:usize, jerk_m_per_s3:f64, accel0_m_per_s2:f64)->f64{
        let num_samples = self.time_s.len();
        let v0 = self.mps[i-1];
        let dt = self.dt_s()[i];
        let mut v = v0;
        for ni in 1 .. (n+1) {
            let idx_to_set = (i - 1) + ni;
            if idx_to_set >= num_samples {
                break
            }
            v = speed_for_constant_jerk(ni, v0, accel0_m_per_s2, jerk_m_per_s3, dt);
            self.mps[idx_to_set] = v;
        }
        v
    }

    /// Add a braking trajectory that would cover the same distance as the given constant brake deceleration
    /// - brake_accel_m__s2: negative number, the braking acceleration (m/s2)
    /// - idx: non-negative integer, the index where to initiate the stop trajectory, start of the step (i in FASTSim)
    /// RETURN: non-negative-number, the final speed of the modified trajectory (m/s) 
    /// - modifies the cycle in place for braking
    pub fn modify_with_braking_trajectory_rust(&mut self, brake_accel_m_per_s2:f64, i:usize)->f64{
        assert!(brake_accel_m_per_s2 < 0.0);
        let v0 = self.mps[i-1];
        let dt = self.dt_s()[i];
        // distance-to-stop (m)
        let dts_m = -0.5 * v0 * v0 / brake_accel_m_per_s2;
        // time-to-stop (s)
        let tts_s = -v0 / brake_accel_m_per_s2;
        // number of steps to take
        let n: usize = (tts_s / dt).round() as usize;
        let n: usize = if n < 2 {2} else {n}; // need at least 2 steps
        let (jerk_m_per_s3, accel_m_per_s2) = calc_constant_jerk_trajectory(n, 0.0, v0, dts_m, 0.0, dt);
        self.modify_by_const_jerk_trajectory_rust(i, n, jerk_m_per_s3, accel_m_per_s2)
    }

    /// rust-internal time steps
    pub fn dt_s(&self) -> Array1<f64> {
        diff(&self.time_s)
    }
    /// distance covered in each time step
    pub fn dist_m(&self) -> Array1<f64>{
        &self.mps * self.dt_s()
    }
    pub fn avg_mps(&self) -> Array1<f64>{
        let num_items = self.mps.len();
        if num_items < 1{
            Array::zeros(1)
        } else {
            let vavgs = 0.5 * (
                &self.mps.slice(s![1..num_items])
                + &self.mps.slice(s![0..(num_items-1)])
            );
            concatenate![Axis(0), Array::zeros(1), vavgs]
        }
    }
    /// distance covered in each time step that is based on the average time of each step
    pub fn dist_v2_m(&self) -> Array1<f64>{
        &self.avg_mps() * &self.dt_s()
    }

    /// get mph from self.mps
    pub fn mph(&self) -> Array1<f64> {
        &self.mps * MPH_PER_MPS
    }

    /// Load cycle from csv file
    pub fn from_file(pathstr: &String) -> Result<RustCycle, String> {
        let pathbuf = PathBuf::from(&pathstr);
        if pathbuf.exists() {
            let mut time_s = Vec::<f64>::new();
            let mut speed_mps = Vec::<f64>::new();
            let mut grade = Vec::<f64>::new();
            let mut road_type = Vec::<f64>::new();
            let name = String::from(pathbuf.file_stem().unwrap().to_str().unwrap());
            let file = File::open(pathbuf).expect("Cycle file not found.");
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(file);
            for result in rdr.records() {
                let record = result.expect("Row not able to load.");
                time_s.push(record[0].parse::<f64>().unwrap());
                speed_mps.push(record[1].parse::<f64>().unwrap());
                grade.push(record[2].parse::<f64>().unwrap());
                road_type.push(record[3].parse::<f64>().unwrap());
            }
            Ok(RustCycle::new(time_s, speed_mps, grade, road_type, name))    
        } else {
            Err(format!("path {pathstr} doesn't exist"))
        }
    }

    /// elevation change w.r.t. to initial
    pub fn delta_elev_m(&self) -> Array1<f64> {
        ndarrcumsum(&(self.dist_m() * self.grade.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist() {
        let cyc = RustCycle::test_cyc();
        assert_eq!(cyc.dist_m().sum(), 45.0);
    }

    #[test]
    fn test_average_speeds_and_distances() {
        let time_s = vec![0.0, 10.0, 30.0, 34.0, 40.0];
        let speed_mps = vec![0.0, 10.0, 10.0, 0.0, 0.0];
        let grade = Array::zeros(5).to_vec();
        let road_type = Array::zeros(5).to_vec();        
        let name = String::from("test");
        let cyc = RustCycle::new(time_s, speed_mps, grade, road_type, name);    
        let avg_mps = cyc.avg_mps();
        let expected_avg_mps = Array::from_vec(vec![0.0, 5.0, 10.0, 5.0, 0.0]);
        assert_eq!(expected_avg_mps.len(), avg_mps.len());
        for (expected, actual) in expected_avg_mps.iter().zip(avg_mps.iter()) {
            assert_eq!(expected, actual);
        }
        let dist_m = cyc.dist_v2_m();
        let expected_dist_m = Array::from_vec(vec![0.0, 50.0, 200.0, 20.0, 0.0]);
        assert_eq!(expected_dist_m.len(), dist_m.len());
        for (expected, actual) in expected_dist_m.iter().zip(dist_m.iter()) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_loading_a_cycle_from_the_filesystem() {
        let pathstr = String::from("../fastsim/resources/cycles/udds.csv");
        let expected_udds_length: usize = 1370;
        match RustCycle::from_file(&pathstr) {
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
    #[test]
    fn test_calculating_total_distance_traveled() {
        let time_s = Array::from_vec(vec![0.0, 10.0, 30.0, 40.0]);
        let mps = Array::from_vec(vec![0.0, 10.0, 10.0, 0.0]);
        let grade = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let road_type = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let name = String::from("test");
        let cyc = RustCycle {time_s, mps, grade, road_type, name};
        let expected_dist_0 = 0.0;
        assert_eq!(expected_dist_0, cyc.total_distance_traveled(0));
        let expected_dist_1 = 50.0;
        assert_eq!(expected_dist_1, cyc.total_distance_traveled(1));
        let expected_dist_2 = 250.0;
        assert_eq!(expected_dist_2, cyc.total_distance_traveled(2));
        let expected_dist_3 = 300.0;
        assert_eq!(expected_dist_3, cyc.total_distance_traveled(3));
        let expected_dist_4 = 300.0;
        assert_eq!(expected_dist_4, cyc.total_distance_traveled(4));
    }
}