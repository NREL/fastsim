extern crate ndarray;
use ndarray::{Array, Array1}; 
extern crate pyo3;
use pyo3::prelude::*;
use super::params::RustPhysicalProperties;
use super::vehicle::RustVehicle;
use super::cycle::RustCycle;


#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustSimDriveParams{
    missed_trace_correction: bool, // if True, missed trace correction is active, default = False
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
    pub fn __new__() -> Self{
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

#[pyclass] 
#[derive(Debug, Clone)]
pub struct RustSimDrive{
    hev_sim_count: usize,
    veh: RustVehicle,
    cyc: RustCycle,
    cyc0: RustCycle,
    sim_params: RustSimDriveParams,
    props: RustPhysicalProperties,
    i: usize, // 1 # initialize step counter for possible use outside sim_drive_walk()
    cur_max_fs_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_trans_lim_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_fs_lim_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_max_kw_in: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_fc_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_cap_lim_dischg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_ess_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_avail_elec_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_cap_lim_chg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_ess_chg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_elec_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mc_elec_in_lim_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mc_transi_lim_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_mc_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_lim_mc_regen_perc_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_lim_mc_regen_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_mech_mc_kw_in: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_trans_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_drag_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_accel_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_ascent_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_trac_kw_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_trac_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    spare_trac_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_rr_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_whl_rad_per_sec: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    cyc_tire_inertia_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_whl_kw_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    regen_contrl_lim_kw_perc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_regen_brake_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_fric_brake_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_trans_kw_out_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cyc_met: Array1<bool>, // np.array([False] * self.cyc.len, dtype=np.bool_)
    trans_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    trans_kw_in_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_soc_target: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    min_mc_kw_2help_fc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mc_mech_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mc_elec_kw_in_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    aux_in_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    roadway_chg_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    min_ess_kw_2help_fc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_kw_out_ach_pct: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_kw_in_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fs_kw_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fs_cumu_mj_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fs_kwh_out_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_cur_kwh: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    soc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    regen_buff_soc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    ess_regen_buff_dischg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    max_ess_regen_buff_chg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    ess_accel_buff_chg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    accel_buff_soc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    max_ess_accell_buff_dischg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    ess_accel_regen_dischg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mc_elec_in_kw_for_max_fc_eff: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    elec_kw_req_4ae: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    can_pwr_all_elec: Array1<f64>, // np.array(  // oddball
    desired_ess_kw_out_for_ae: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_ae_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    er_ae_kw_out: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_desired_kw_4fc_eff: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_kw_if_fc_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    cur_max_mc_elec_kw_in: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_kw_gap_fr_eff: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    er_kw_if_fc_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    mc_elec_kw_in_if_fc_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    mc_kw_if_fc_req: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    fc_forced_on: Array1<bool>, // np.array([False] * self.cyc.len, dtype=np.bool_)
    fc_forced_state: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.int32)
    mc_mech_kw_4forced_fc: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    fc_time_on: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    prev_fc_time_on: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mps_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    mph_ach: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    dist_m: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddbal
    dist_mi: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    high_acc_fc_on_tag: Array1<bool>, // np.array([False] * self.cyc.len, dtype=np.bool_)
    reached_buff: Array1<bool>, // np.array([False] * self.cyc.len, dtype=np.bool_)
    max_trac_mps: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    add_kwh: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    dod_cycs: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_perc_dead: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
    drag_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ess_loss_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    accel_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    ascent_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    rr_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    cur_max_roadway_chg_kw: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    trace_miss_iters: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
    newton_iters: Array1<f64>, // np.zeros(self.cyc.len, dtype=np.float64)
}

#[pymethods]
impl RustSimDrive{
    #[new]
    pub fn __new__(cyc: RustCycle, veh: RustVehicle) -> Self{
        let hev_sim_count: usize = 0;
        let cyc0: RustCycle = cyc.clone();
        let sim_params = RustSimDriveParams::__new__();
        let props = RustPhysicalProperties::__new__();
        let i: usize = 1; // 1 # initialize step counter for possible use outside sim_drive_walk()
        let cyc_len = cyc.time_s.len(); //get_len() as usize;
        let cur_max_fs_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_trans_lim_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_fs_lim_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_max_kw_in = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_fc_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_cap_lim_dischg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_ess_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_avail_elec_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_cap_lim_chg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_ess_chg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_elec_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mc_elec_in_lim_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mc_transi_lim_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_mc_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_lim_mc_regen_perc_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_lim_mc_regen_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_mech_mc_kw_in = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_trans_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_drag_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_accel_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_ascent_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_trac_kw_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_trac_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let spare_trac_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_rr_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_whl_rad_per_sec = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let cyc_tire_inertia_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_whl_kw_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let regen_contrl_lim_kw_perc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_regen_brake_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_fric_brake_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_trans_kw_out_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cyc_met = Array::from_vec(vec![false; cyc_len]); // np.array([False] * self.cyc_len, dtype=np.bool_)
        let trans_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let trans_kw_in_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_soc_target = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let min_mc_kw_2help_fc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mc_mech_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mc_elec_kw_in_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let aux_in_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let roadway_chg_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let min_ess_kw_2help_fc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_kw_out_ach_pct = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_kw_in_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fs_kw_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fs_cumu_mj_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fs_kwh_out_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_cur_kwh = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let soc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let regen_buff_soc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let ess_regen_buff_dischg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let max_ess_regen_buff_chg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let ess_accel_buff_chg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let accel_buff_soc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let max_ess_accell_buff_dischg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let ess_accel_regen_dischg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mc_elec_in_kw_for_max_fc_eff = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let elec_kw_req_4ae = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let can_pwr_all_elec = Array::zeros(cyc_len); // np.array(  // oddball
        let desired_ess_kw_out_for_ae = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_ae_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let er_ae_kw_out = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_desired_kw_4fc_eff = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_kw_if_fc_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let cur_max_mc_elec_kw_in = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_kw_gap_fr_eff = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let er_kw_if_fc_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let mc_elec_kw_in_if_fc_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let mc_kw_if_fc_req = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let fc_forced_on = Array::from_vec(vec![false; cyc_len]); // np.array([False] * self.cyc_len, dtype=np.bool_)
        let fc_forced_state = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.int32)
        let mc_mech_kw_4forced_fc = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let fc_time_on = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let prev_fc_time_on = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mps_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let mph_ach = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let dist_m = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddbal
        let dist_mi = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let high_acc_fc_on_tag = Array::from_vec(vec![false; cyc_len]); // np.array([False] * self.cyc_len, dtype=np.bool_)
        let reached_buff = Array::from_vec(vec![false; cyc_len]); // np.array([False] * self.cyc_len, dtype=np.bool_)
        let max_trac_mps = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let add_kwh = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let dod_cycs = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_perc_dead = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)  // oddball
        let drag_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ess_loss_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let accel_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let ascent_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let rr_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let cur_max_roadway_chg_kw = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let trace_miss_iters = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        let newton_iters = Array::zeros(cyc_len); // np.zeros(self.cyc_len, dtype=np.float64)
        RustSimDrive{
            hev_sim_count,
            veh,
            cyc,
            cyc0,
            sim_params,
            props,
            i, // 1 # initialize step counter for possible use outside sim_drive_walk()
            cur_max_fs_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_trans_lim_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_fs_lim_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_max_kw_in, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_fc_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_cap_lim_dischg_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_ess_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_avail_elec_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_cap_lim_chg_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_ess_chg_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_elec_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            mc_elec_in_lim_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            mc_transi_lim_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_mc_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_lim_mc_regen_perc_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_lim_mc_regen_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_mech_mc_kw_in, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_trans_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_drag_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_accel_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_ascent_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_trac_kw_req, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_trac_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            spare_trac_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_rr_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_whl_rad_per_sec, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            cyc_tire_inertia_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_whl_kw_req, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            regen_contrl_lim_kw_perc, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_regen_brake_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_fric_brake_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_trans_kw_out_req, // np.zeros(self.cyc.len, dtype=np.float64)
            cyc_met, // np.array([False] * self.cyc.len, dtype=np.bool_)
            trans_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            trans_kw_in_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_soc_target, // np.zeros(self.cyc.len, dtype=np.float64)
            min_mc_kw_2help_fc, // np.zeros(self.cyc.len, dtype=np.float64)
            mc_mech_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            mc_elec_kw_in_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            aux_in_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            roadway_chg_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            min_ess_kw_2help_fc, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_kw_out_ach_pct, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_kw_in_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            fs_kw_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            fs_cumu_mj_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            fs_kwh_out_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_cur_kwh, // np.zeros(self.cyc.len, dtype=np.float64)
            soc, // np.zeros(self.cyc.len, dtype=np.float64)
            regen_buff_soc, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            ess_regen_buff_dischg_kw, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            max_ess_regen_buff_chg_kw, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            ess_accel_buff_chg_kw, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            accel_buff_soc, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            max_ess_accell_buff_dischg_kw, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            ess_accel_regen_dischg_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            mc_elec_in_kw_for_max_fc_eff, // np.zeros(self.cyc.len, dtype=np.float64)
            elec_kw_req_4ae, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            can_pwr_all_elec, // np.array(  // oddball
            desired_ess_kw_out_for_ae, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_ae_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            er_ae_kw_out, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_desired_kw_4fc_eff, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_kw_if_fc_req, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            cur_max_mc_elec_kw_in, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_kw_gap_fr_eff, // np.zeros(self.cyc.len, dtype=np.float64)
            er_kw_if_fc_req, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            mc_elec_kw_in_if_fc_req, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            mc_kw_if_fc_req, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            fc_forced_on, // np.array([False] * self.cyc.len, dtype=np.bool_)
            fc_forced_state, // np.zeros(self.cyc.len, dtype=np.int32)
            mc_mech_kw_4forced_fc, // np.zeros(self.cyc.len, dtype=np.float64)
            fc_time_on, // np.zeros(self.cyc.len, dtype=np.float64)
            prev_fc_time_on, // np.zeros(self.cyc.len, dtype=np.float64)
            mps_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            mph_ach, // np.zeros(self.cyc.len, dtype=np.float64)
            dist_m, // np.zeros(self.cyc.len, dtype=np.float64)  // oddbal
            dist_mi, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            high_acc_fc_on_tag, // np.array([False] * self.cyc.len, dtype=np.bool_)
            reached_buff, // np.array([False] * self.cyc.len, dtype=np.bool_)
            max_trac_mps, // np.zeros(self.cyc.len, dtype=np.float64)
            add_kwh, // np.zeros(self.cyc.len, dtype=np.float64)
            dod_cycs, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_perc_dead, // np.zeros(self.cyc.len, dtype=np.float64)  // oddball
            drag_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            ess_loss_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            accel_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            ascent_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            rr_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            cur_max_roadway_chg_kw, // np.zeros(self.cyc.len, dtype=np.float64)
            trace_miss_iters, // np.zeros(self.cyc.len, dtype=np.float64)
            newton_iters, // np.zeros(self.cyc.len, dtype=np.float64)
        }
    }

    #[getter]
    pub fn get_veh(&self) -> PyResult<RustVehicle>{
        Ok(self.veh.clone())
    }
    #[setter]
    pub fn set_veh(&mut self, new_value:RustVehicle) -> PyResult<()>{
        self.veh = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_cyc(&self) -> PyResult<RustCycle>{
        Ok(self.cyc.clone())
    }
    #[setter]
    pub fn set_cyc(&mut self, new_value:RustCycle) -> PyResult<()>{
        self.cyc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_cyc0(&self) -> PyResult<RustCycle>{
        Ok(self.cyc0.clone())
    }
    #[setter]
    pub fn set_cyc0(&mut self, new_value:RustCycle) -> PyResult<()>{
        self.cyc0 = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_sim_params(&self) -> PyResult<RustSimDriveParams>{
        Ok(self.sim_params.clone())
    }
    #[setter]
    pub fn set_sim_params(&mut self, new_value:RustSimDriveParams) -> PyResult<()>{
        self.sim_params = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_props(&self) -> PyResult<RustPhysicalProperties>{
        Ok(self.props.clone())
    }
    #[setter]
    pub fn set_props(&mut self, new_value:RustPhysicalProperties) -> PyResult<()>{
        self.props = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_accel_buff_soc(&self) -> PyResult<Vec<f64>>{
      Ok(self.accel_buff_soc.to_vec())
    }
    #[setter]
    pub fn set_accel_buff_soc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.accel_buff_soc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_accel_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.accel_kw.to_vec())
    }
    #[setter]
    pub fn set_accel_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.accel_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_add_kwh(&self) -> PyResult<Vec<f64>>{
      Ok(self.add_kwh.to_vec())
    }
    #[setter]
    pub fn set_add_kwh(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.add_kwh = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ascent_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ascent_kw.to_vec())
    }
    #[setter]
    pub fn set_ascent_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ascent_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_aux_in_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.aux_in_kw.to_vec())
    }
    #[setter]
    pub fn set_aux_in_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.aux_in_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_can_pwr_all_elec(&self) -> PyResult<Vec<f64>>{
      Ok(self.can_pwr_all_elec.to_vec())
    }
    #[setter]
    pub fn set_can_pwr_all_elec(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.can_pwr_all_elec = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_avail_elec_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_avail_elec_kw.to_vec())
    }
    #[setter]
    pub fn set_cur_max_avail_elec_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_avail_elec_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_elec_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_elec_kw.to_vec())
    }
    #[setter]
    pub fn set_cur_max_elec_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_elec_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_ess_chg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_ess_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_cur_max_ess_chg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_ess_chg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_ess_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_ess_kw_out.to_vec())
    }
    #[setter]
    pub fn set_cur_max_ess_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_ess_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_fc_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_fc_kw_out.to_vec())
    }
    #[setter]
    pub fn set_cur_max_fc_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_fc_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_fs_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_fs_kw_out.to_vec())
    }
    #[setter]
    pub fn set_cur_max_fs_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_fs_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_mc_elec_kw_in(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_mc_elec_kw_in.to_vec())
    }
    #[setter]
    pub fn set_cur_max_mc_elec_kw_in(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_mc_elec_kw_in = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_mc_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_mc_kw_out.to_vec())
    }
    #[setter]
    pub fn set_cur_max_mc_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_mc_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_mech_mc_kw_in(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_mech_mc_kw_in.to_vec())
    }
    #[setter]
    pub fn set_cur_max_mech_mc_kw_in(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_mech_mc_kw_in = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_roadway_chg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_roadway_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_cur_max_roadway_chg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_roadway_chg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_trac_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_trac_kw.to_vec())
    }
    #[setter]
    pub fn set_cur_max_trac_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_trac_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_max_trans_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_max_trans_kw_out.to_vec())
    }
    #[setter]
    pub fn set_cur_max_trans_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_max_trans_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cur_soc_target(&self) -> PyResult<Vec<f64>>{
      Ok(self.cur_soc_target.to_vec())
    }
    #[setter]
    pub fn set_cur_soc_target(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cur_soc_target = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_accel_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_accel_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_accel_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_accel_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_ascent_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_ascent_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_ascent_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_ascent_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_drag_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_drag_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_drag_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_drag_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_fric_brake_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_fric_brake_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_fric_brake_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_fric_brake_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_met(&self) -> PyResult<Vec<bool>>{
      Ok(self.cyc_met.to_vec())
    }
    #[setter]
    pub fn set_cyc_met(&mut self, new_value:Vec<bool>) -> PyResult<()>{
      self.cyc_met = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_regen_brake_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_regen_brake_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_regen_brake_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_regen_brake_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_rr_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_rr_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_rr_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_rr_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_tire_inertia_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_tire_inertia_kw.to_vec())
    }
    #[setter]
    pub fn set_cyc_tire_inertia_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_tire_inertia_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_trac_kw_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_trac_kw_req.to_vec())
    }
    #[setter]
    pub fn set_cyc_trac_kw_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_trac_kw_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_trans_kw_out_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_trans_kw_out_req.to_vec())
    }
    #[setter]
    pub fn set_cyc_trans_kw_out_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_trans_kw_out_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_whl_kw_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_whl_kw_req.to_vec())
    }
    #[setter]
    pub fn set_cyc_whl_kw_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_whl_kw_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_cyc_whl_rad_per_sec(&self) -> PyResult<Vec<f64>>{
      Ok(self.cyc_whl_rad_per_sec.to_vec())
    }
    #[setter]
    pub fn set_cyc_whl_rad_per_sec(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.cyc_whl_rad_per_sec = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_desired_ess_kw_out_for_ae(&self) -> PyResult<Vec<f64>>{
      Ok(self.desired_ess_kw_out_for_ae.to_vec())
    }
    #[setter]
    pub fn set_desired_ess_kw_out_for_ae(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.desired_ess_kw_out_for_ae = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_dist_m(&self) -> PyResult<Vec<f64>>{
      Ok(self.dist_m.to_vec())
    }
    #[setter]
    pub fn set_dist_m(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.dist_m = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_dist_mi(&self) -> PyResult<Vec<f64>>{
      Ok(self.dist_mi.to_vec())
    }
    #[setter]
    pub fn set_dist_mi(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.dist_mi = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_dod_cycs(&self) -> PyResult<Vec<f64>>{
      Ok(self.dod_cycs.to_vec())
    }
    #[setter]
    pub fn set_dod_cycs(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.dod_cycs = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_drag_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.drag_kw.to_vec())
    }
    #[setter]
    pub fn set_drag_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.drag_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_elec_kw_req_4ae(&self) -> PyResult<Vec<f64>>{
      Ok(self.elec_kw_req_4ae.to_vec())
    }
    #[setter]
    pub fn set_elec_kw_req_4ae(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.elec_kw_req_4ae = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_er_ae_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.er_ae_kw_out.to_vec())
    }
    #[setter]
    pub fn set_er_ae_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.er_ae_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_er_kw_if_fc_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.er_kw_if_fc_req.to_vec())
    }
    #[setter]
    pub fn set_er_kw_if_fc_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.er_kw_if_fc_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_accel_buff_chg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_accel_buff_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_accel_buff_chg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_accel_buff_chg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_accel_regen_dischg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_accel_regen_dischg_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_accel_regen_dischg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_accel_regen_dischg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_ae_kw_out(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_ae_kw_out.to_vec())
    }
    #[setter]
    pub fn set_ess_ae_kw_out(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_ae_kw_out = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_cap_lim_chg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_cap_lim_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_cap_lim_chg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_cap_lim_chg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_cap_lim_dischg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_cap_lim_dischg_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_cap_lim_dischg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_cap_lim_dischg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_cur_kwh(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_cur_kwh.to_vec())
    }
    #[setter]
    pub fn set_ess_cur_kwh(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_cur_kwh = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_desired_kw_4fc_eff(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_desired_kw_4fc_eff.to_vec())
    }
    #[setter]
    pub fn set_ess_desired_kw_4fc_eff(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_desired_kw_4fc_eff = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_kw_if_fc_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_kw_if_fc_req.to_vec())
    }
    #[setter]
    pub fn set_ess_kw_if_fc_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_kw_if_fc_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_ess_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_lim_mc_regen_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_lim_mc_regen_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_lim_mc_regen_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_lim_mc_regen_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_lim_mc_regen_perc_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_lim_mc_regen_perc_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_lim_mc_regen_perc_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_lim_mc_regen_perc_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_loss_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_loss_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_loss_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_loss_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_perc_dead(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_perc_dead.to_vec())
    }
    #[setter]
    pub fn set_ess_perc_dead(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_perc_dead = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_ess_regen_buff_dischg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.ess_regen_buff_dischg_kw.to_vec())
    }
    #[setter]
    pub fn set_ess_regen_buff_dischg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.ess_regen_buff_dischg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_forced_on(&self) -> PyResult<Vec<bool>>{
      Ok(self.fc_forced_on.to_vec())
    }
    #[setter]
    pub fn set_fc_forced_on(&mut self, new_value:Vec<bool>) -> PyResult<()>{
      self.fc_forced_on = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_forced_state(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_forced_state.to_vec())
    }
    #[setter]
    pub fn set_fc_forced_state(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_forced_state = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_fs_lim_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_fs_lim_kw.to_vec())
    }
    #[setter]
    pub fn set_fc_fs_lim_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_fs_lim_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_kw_gap_fr_eff(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_kw_gap_fr_eff.to_vec())
    }
    #[setter]
    pub fn set_fc_kw_gap_fr_eff(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_kw_gap_fr_eff = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_kw_in_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_kw_in_ach.to_vec())
    }
    #[setter]
    pub fn set_fc_kw_in_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_kw_in_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_fc_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_kw_out_ach_pct(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_kw_out_ach_pct.to_vec())
    }
    #[setter]
    pub fn set_fc_kw_out_ach_pct(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_kw_out_ach_pct = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_max_kw_in(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_max_kw_in.to_vec())
    }
    #[setter]
    pub fn set_fc_max_kw_in(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_max_kw_in = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_time_on(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_time_on.to_vec())
    }
    #[setter]
    pub fn set_fc_time_on(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_time_on = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fc_trans_lim_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.fc_trans_lim_kw.to_vec())
    }
    #[setter]
    pub fn set_fc_trans_lim_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fc_trans_lim_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fs_cumu_mj_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.fs_cumu_mj_out_ach.to_vec())
    }
    #[setter]
    pub fn set_fs_cumu_mj_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fs_cumu_mj_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fs_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.fs_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_fs_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fs_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_fs_kwh_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.fs_kwh_out_ach.to_vec())
    }
    #[setter]
    pub fn set_fs_kwh_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.fs_kwh_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_hev_sim_count(&self) -> PyResult<usize>{
      Ok(self.hev_sim_count)
    }
    #[setter]
    pub fn set_hev_sim_count(&mut self, new_value:usize) -> PyResult<()>{
      self.hev_sim_count = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_high_acc_fc_on_tag(&self) -> PyResult<Vec<bool>>{
      Ok(self.high_acc_fc_on_tag.to_vec())
    }
    #[setter]
    pub fn set_high_acc_fc_on_tag(&mut self, new_value:Vec<bool>) -> PyResult<()>{
      self.high_acc_fc_on_tag = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_i(&self) -> PyResult<usize>{
      Ok(self.i)
    }
    #[setter]
    pub fn set_i(&mut self, new_value:usize) -> PyResult<()>{
      self.i = new_value;
      Ok(())
    }

    #[getter]
    pub fn get_max_ess_accell_buff_dischg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.max_ess_accell_buff_dischg_kw.to_vec())
    }
    #[setter]
    pub fn set_max_ess_accell_buff_dischg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.max_ess_accell_buff_dischg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_max_ess_regen_buff_chg_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.max_ess_regen_buff_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_max_ess_regen_buff_chg_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.max_ess_regen_buff_chg_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_max_trac_mps(&self) -> PyResult<Vec<f64>>{
      Ok(self.max_trac_mps.to_vec())
    }
    #[setter]
    pub fn set_max_trac_mps(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.max_trac_mps = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_elec_in_kw_for_max_fc_eff(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_elec_in_kw_for_max_fc_eff.to_vec())
    }
    #[setter]
    pub fn set_mc_elec_in_kw_for_max_fc_eff(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_elec_in_kw_for_max_fc_eff = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_elec_in_lim_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_elec_in_lim_kw.to_vec())
    }
    #[setter]
    pub fn set_mc_elec_in_lim_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_elec_in_lim_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_elec_kw_in_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_elec_kw_in_ach.to_vec())
    }
    #[setter]
    pub fn set_mc_elec_kw_in_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_elec_kw_in_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_elec_kw_in_if_fc_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_elec_kw_in_if_fc_req.to_vec())
    }
    #[setter]
    pub fn set_mc_elec_kw_in_if_fc_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_elec_kw_in_if_fc_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_kw_if_fc_req(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_kw_if_fc_req.to_vec())
    }
    #[setter]
    pub fn set_mc_kw_if_fc_req(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_kw_if_fc_req = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_mech_kw_4forced_fc(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_mech_kw_4forced_fc.to_vec())
    }
    #[setter]
    pub fn set_mc_mech_kw_4forced_fc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_mech_kw_4forced_fc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_mech_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_mech_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_mc_mech_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_mech_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mc_transi_lim_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.mc_transi_lim_kw.to_vec())
    }
    #[setter]
    pub fn set_mc_transi_lim_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mc_transi_lim_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_min_ess_kw_2help_fc(&self) -> PyResult<Vec<f64>>{
      Ok(self.min_ess_kw_2help_fc.to_vec())
    }
    #[setter]
    pub fn set_min_ess_kw_2help_fc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.min_ess_kw_2help_fc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_min_mc_kw_2help_fc(&self) -> PyResult<Vec<f64>>{
      Ok(self.min_mc_kw_2help_fc.to_vec())
    }
    #[setter]
    pub fn set_min_mc_kw_2help_fc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.min_mc_kw_2help_fc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mph_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.mph_ach.to_vec())
    }
    #[setter]
    pub fn set_mph_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mph_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_mps_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.mps_ach.to_vec())
    }
    #[setter]
    pub fn set_mps_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.mps_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_newton_iters(&self) -> PyResult<Vec<f64>>{
      Ok(self.newton_iters.to_vec())
    }
    #[setter]
    pub fn set_newton_iters(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.newton_iters = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_prev_fc_time_on(&self) -> PyResult<Vec<f64>>{
      Ok(self.prev_fc_time_on.to_vec())
    }
    #[setter]
    pub fn set_prev_fc_time_on(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.prev_fc_time_on = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_reached_buff(&self) -> PyResult<Vec<bool>>{
      Ok(self.reached_buff.to_vec())
    }
    #[setter]
    pub fn set_reached_buff(&mut self, new_value:Vec<bool>) -> PyResult<()>{
      self.reached_buff = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_regen_buff_soc(&self) -> PyResult<Vec<f64>>{
      Ok(self.regen_buff_soc.to_vec())
    }
    #[setter]
    pub fn set_regen_buff_soc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.regen_buff_soc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_regen_contrl_lim_kw_perc(&self) -> PyResult<Vec<f64>>{
      Ok(self.regen_contrl_lim_kw_perc.to_vec())
    }
    #[setter]
    pub fn set_regen_contrl_lim_kw_perc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.regen_contrl_lim_kw_perc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_roadway_chg_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.roadway_chg_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_roadway_chg_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.roadway_chg_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_rr_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.rr_kw.to_vec())
    }
    #[setter]
    pub fn set_rr_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.rr_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_soc(&self) -> PyResult<Vec<f64>>{
      Ok(self.soc.to_vec())
    }
    #[setter]
    pub fn set_soc(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.soc = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_spare_trac_kw(&self) -> PyResult<Vec<f64>>{
      Ok(self.spare_trac_kw.to_vec())
    }
    #[setter]
    pub fn set_spare_trac_kw(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.spare_trac_kw = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_trace_miss_iters(&self) -> PyResult<Vec<f64>>{
      Ok(self.trace_miss_iters.to_vec())
    }
    #[setter]
    pub fn set_trace_miss_iters(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.trace_miss_iters = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_trans_kw_in_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.trans_kw_in_ach.to_vec())
    }
    #[setter]
    pub fn set_trans_kw_in_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.trans_kw_in_ach = Array::from_vec(new_value);
      Ok(())
    }

    #[getter]
    pub fn get_trans_kw_out_ach(&self) -> PyResult<Vec<f64>>{
      Ok(self.trans_kw_out_ach.to_vec())
    }
    #[setter]
    pub fn set_trans_kw_out_ach(&mut self, new_value:Vec<f64>) -> PyResult<()>{
      self.trans_kw_out_ach = Array::from_vec(new_value);
      Ok(())
    }

}