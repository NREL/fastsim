extern crate ndarray;
// use ndarray::{Array, Array1}; 
extern crate pyo3;
use pyo3::prelude::*;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustSimDriveParams{
    missed_trace_correction:bool, // if True, missed trace correction is active, default = False
    max_time_dilation: f64,
    min_time_dilation: f64,
    time_dilation_tol: f64,
    max_trace_miss_iters: u32,
    trace_miss_speed_mps_tol: f64,
    trace_miss_time_tol: f64,
    trace_miss_dist_tol: f64,
    sim_count_max: u32,
    verbose: bool,
    newton_gain: f64,
    newton_max_iter: u32,
    newton_xtol: f64,
    energy_audit_error_tol: f64,
    max_epa_adj: f64,
}

#[pymethods]
impl RustSimDriveParams{
    #[new]
    pub fn __new__(
    ) -> Self{
        // if True, missed trace correction is active, default = False
        let missed_trace_correction = false;
        // maximum time dilation factor to "catch up" with trace -- e.g. 1.0 means 100% increase in step size
        let max_time_dilation: f64 = 1.0;
        // minimum time dilation margin to let trace "catch up" -- e.g. -0.5 means 50% reduction in step size
        let min_time_dilation: f64 = -0.5;
        let time_dilation_tol: f64 = 5e-4; // convergence criteria for time dilation
        let max_trace_miss_iters: u32 = 5; // number of iterations to achieve time dilation correction
        let trace_miss_speed_mps_tol: f64 = 1.0; // # threshold of error in speed [m/s] that triggers warning
        let trace_miss_time_tol: f64 = 1e-3; // threshold for printing warning when time dilation is active
        let trace_miss_dist_tol: f64 = 1e-3; // threshold of fractional eror in distance that triggers warning
        let sim_count_max: u32 = 30; // max allowable number of HEV SOC iterations
        let verbose = true; // show warning and other messages
        let newton_gain: f64 = 0.9; // newton solver gain
        let newton_max_iter: u32 = 100; // newton solver max iterations
        let newton_xtol: f64 = 1e-9; // newton solver tolerance
        let energy_audit_error_tol: f64 = 0.002; // tolerance for energy audit error warning, i.e. 0.1%
        // EPA fuel economy adjustment parameters
        let max_epa_adj: f64 = 0.3; // maximum EPA adjustment factor
        RustSimDriveParams{
            missed_trace_correction,
            max_time_dilation,
            min_time_dilation,
            time_dilation_tol,
            max_trace_miss_iters,
            trace_miss_speed_mps_tol,
            trace_miss_time_tol,
            trace_miss_dist_tol,
            sim_count_max,
            verbose,
            newton_gain,
            newton_max_iter,
            newton_xtol,
            energy_audit_error_tol,
            max_epa_adj,
        }
    }

    #[getter]
    pub fn get_energy_audit_error_tol(&self) -> PyResult<f64>{
      Ok(self.energy_audit_error_tol)
    }
    #[setter]
    pub fn set_energy_audit_error_tol(&mut self, new_value:f64) -> PyResult<()>{
      self.energy_audit_error_tol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_max_epa_adj(&self) -> PyResult<f64>{
      Ok(self.max_epa_adj)
    }
    #[setter]
    pub fn set_max_epa_adj(&mut self, new_value:f64) -> PyResult<()>{
      self.max_epa_adj = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_max_time_dilation(&self) -> PyResult<f64>{
      Ok(self.max_time_dilation)
    }
    #[setter]
    pub fn set_max_time_dilation(&mut self, new_value:f64) -> PyResult<()>{
      self.max_time_dilation = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_max_trace_miss_iters(&self) -> PyResult<u32>{
      Ok(self.max_trace_miss_iters)
    }
    #[setter]
    pub fn set_max_trace_miss_iters(&mut self, new_value:u32) -> PyResult<()>{
      self.max_trace_miss_iters = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_min_time_dilation(&self) -> PyResult<f64>{
      Ok(self.min_time_dilation)
    }
    #[setter]
    pub fn set_min_time_dilation(&mut self, new_value:f64) -> PyResult<()>{
      self.min_time_dilation = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_missed_trace_correction(&self) -> PyResult<bool>{
      Ok(self.missed_trace_correction)
    }
    #[setter]
    pub fn set_missed_trace_correction(&mut self, new_value:bool) -> PyResult<()>{
      self.missed_trace_correction = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_newton_gain(&self) -> PyResult<f64>{
      Ok(self.newton_gain)
    }
    #[setter]
    pub fn set_newton_gain(&mut self, new_value:f64) -> PyResult<()>{
      self.newton_gain = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_newton_max_iter(&self) -> PyResult<u32>{
      Ok(self.newton_max_iter)
    }
    #[setter]
    pub fn set_newton_max_iter(&mut self, new_value:u32) -> PyResult<()>{
      self.newton_max_iter = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_newton_xtol(&self) -> PyResult<f64>{
      Ok(self.newton_xtol)
    }
    #[setter]
    pub fn set_newton_xtol(&mut self, new_value:f64) -> PyResult<()>{
      self.newton_xtol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_sim_count_max(&self) -> PyResult<u32>{
      Ok(self.sim_count_max)
    }
    #[setter]
    pub fn set_sim_count_max(&mut self, new_value:u32) -> PyResult<()>{
      self.sim_count_max = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_time_dilation_tol(&self) -> PyResult<f64>{
      Ok(self.time_dilation_tol)
    }
    #[setter]
    pub fn set_time_dilation_tol(&mut self, new_value:f64) -> PyResult<()>{
      self.time_dilation_tol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_trace_miss_dist_tol(&self) -> PyResult<f64>{
      Ok(self.trace_miss_dist_tol)
    }
    #[setter]
    pub fn set_trace_miss_dist_tol(&mut self, new_value:f64) -> PyResult<()>{
      self.trace_miss_dist_tol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_trace_miss_speed_mps_tol(&self) -> PyResult<f64>{
      Ok(self.trace_miss_speed_mps_tol)
    }
    #[setter]
    pub fn set_trace_miss_speed_mps_tol(&mut self, new_value:f64) -> PyResult<()>{
      self.trace_miss_speed_mps_tol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_trace_miss_time_tol(&self) -> PyResult<f64>{
      Ok(self.trace_miss_time_tol)
    }
    #[setter]
    pub fn set_trace_miss_time_tol(&mut self, new_value:f64) -> PyResult<()>{
      self.trace_miss_time_tol = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_verbose(&self) -> PyResult<bool>{
      Ok(self.verbose)
    }
    #[setter]
    pub fn set_verbose(&mut self, new_value:bool) -> PyResult<()>{
      self.verbose = new_value;
      Ok(())
    }
}
