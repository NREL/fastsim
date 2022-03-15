extern crate ndarray;
use ndarray::{Array, Array1, array, s}; 
extern crate pyo3;
use pyo3::prelude::*;
use std::cmp;

use super::utils::*;
use super::params::RustPhysicalProperties;
use super::params;
use super::vehicle::*;
use super::cycle::RustCycle;


#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustSimDriveParams{
    pub missed_trace_correction: bool, // if true, missed trace correction is active, default = false
    pub max_time_dilation: f64,
    pub min_time_dilation: f64,
    pub time_dilation_tol: f64,
    pub max_trace_miss_iters: u32,
    pub trace_miss_speed_mps_tol: f64,
    pub trace_miss_time_tol: f64,
    pub trace_miss_dist_tol: f64,
    pub sim_count_max: u32,
    pub verbose: bool,
    pub newton_gain: f64,
    pub newton_max_iter: u32,
    pub newton_xtol: f64,
    pub energy_audit_error_tol: f64,
    pub max_epa_adj: f64,
}

#[pymethods]
impl RustSimDriveParams{
    #[new]
    pub fn __new__() -> Self{
        // if true, missed trace correction is active, default = false
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
    #[pyo3(get, set)]
    pub hev_sim_count: usize,
    pub veh: RustVehicle,
    pub cyc: RustCycle,
    pub cyc0: RustCycle,
    pub sim_params: RustSimDriveParams,
    pub props: RustPhysicalProperties,
    #[pyo3(get, set)]
    pub i: usize, // 1 # initialize step counter for possible use outside sim_drive_walk()
    pub cur_max_fs_kw_out: Array1<f64>, 
    pub fc_trans_lim_kw: Array1<f64>, 
    pub fc_fs_lim_kw: Array1<f64>, 
    pub fc_max_kw_in: Array1<f64>, 
    pub cur_max_fc_kw_out: Array1<f64>, 
    pub ess_cap_lim_dischg_kw: Array1<f64>, 
    pub cur_max_ess_kw_out: Array1<f64>, 
    pub cur_max_avail_elec_kw: Array1<f64>, 
    pub ess_cap_lim_chg_kw: Array1<f64>, 
    pub cur_max_ess_chg_kw: Array1<f64>, 
    pub cur_max_elec_kw: Array1<f64>, 
    pub mc_elec_in_lim_kw: Array1<f64>, 
    pub mc_transi_lim_kw: Array1<f64>, 
    pub cur_max_mc_kw_out: Array1<f64>, 
    pub ess_lim_mc_regen_perc_kw: Array1<f64>, 
    pub ess_lim_mc_regen_kw: Array1<f64>, 
    pub cur_max_mech_mc_kw_in: Array1<f64>, 
    pub cur_max_trans_kw_out: Array1<f64>, 
    pub cyc_drag_kw: Array1<f64>, 
    pub cyc_accel_kw: Array1<f64>, 
    pub cyc_ascent_kw: Array1<f64>, 
    pub cyc_trac_kw_req: Array1<f64>, 
    pub cur_max_trac_kw: Array1<f64>, 
    pub spare_trac_kw: Array1<f64>, 
    pub cyc_rr_kw: Array1<f64>, 
    pub cyc_whl_rad_per_sec: Array1<f64>,   
    pub cyc_tire_inertia_kw: Array1<f64>, 
    pub cyc_whl_kw_req: Array1<f64>,   
    pub regen_contrl_lim_kw_perc: Array1<f64>, 
    pub cyc_regen_brake_kw: Array1<f64>, 
    pub cyc_fric_brake_kw: Array1<f64>, 
    pub cyc_trans_kw_out_req: Array1<f64>, 
    pub cyc_met: Array1<bool>, 
    pub trans_kw_out_ach: Array1<f64>, 
    pub trans_kw_in_ach: Array1<f64>, 
    pub cur_soc_target: Array1<f64>, 
    pub min_mc_kw_2help_fc: Array1<f64>, 
    pub mc_mech_kw_out_ach: Array1<f64>, 
    pub mc_elec_kw_in_ach: Array1<f64>, 
    pub aux_in_kw: Array1<f64>, 
    pub roadway_chg_kw_out_ach: Array1<f64>, 
    pub min_ess_kw_2help_fc: Array1<f64>, 
    pub ess_kw_out_ach: Array1<f64>, 
    pub fc_kw_out_ach: Array1<f64>, 
    pub fc_kw_out_ach_pct: Array1<f64>, 
    pub fc_kw_in_ach: Array1<f64>, 
    pub fs_kw_out_ach: Array1<f64>, 
    pub fs_cumu_mj_out_ach: Array1<f64>, 
    pub fs_kwh_out_ach: Array1<f64>, 
    pub ess_cur_kwh: Array1<f64>, 
    pub soc: Array1<f64>, 
    pub regen_buff_soc: Array1<f64>,   
    pub ess_regen_buff_dischg_kw: Array1<f64>,   
    pub max_ess_regen_buff_chg_kw: Array1<f64>,   
    pub ess_accel_buff_chg_kw: Array1<f64>,   
    pub accel_buff_soc: Array1<f64>,   
    pub max_ess_accell_buff_dischg_kw: Array1<f64>,   
    pub ess_accel_regen_dischg_kw: Array1<f64>, 
    pub mc_elec_in_kw_for_max_fc_eff: Array1<f64>, 
    pub elec_kw_req_4ae: Array1<f64>,   
    pub can_pwr_all_elec: Array1<f64>, 
    pub desired_ess_kw_out_for_ae: Array1<f64>, 
    pub ess_ae_kw_out: Array1<f64>, 
    pub er_ae_kw_out: Array1<f64>, 
    pub ess_desired_kw_4fc_eff: Array1<f64>, 
    pub ess_kw_if_fc_req: Array1<f64>,   
    pub cur_max_mc_elec_kw_in: Array1<f64>, 
    pub fc_kw_gap_fr_eff: Array1<f64>, 
    pub er_kw_if_fc_req: Array1<f64>,   
    pub mc_elec_kw_in_if_fc_req: Array1<f64>,   
    pub mc_kw_if_fc_req: Array1<f64>,   
    pub fc_forced_on: Array1<bool>, 
    pub fc_forced_state: Array1<f64>, 
    pub mc_mech_kw_4forced_fc: Array1<f64>, 
    pub fc_time_on: Array1<f64>, 
    pub prev_fc_time_on: Array1<f64>, 
    pub mps_ach: Array1<f64>, 
    pub mph_ach: Array1<f64>, 
    pub dist_m: Array1<f64>,   
    pub dist_mi: Array1<f64>,   
    pub high_acc_fc_on_tag: Array1<bool>, 
    pub reached_buff: Array1<bool>, 
    pub max_trac_mps: Array1<f64>, 
    pub add_kwh: Array1<f64>, 
    pub dod_cycs: Array1<f64>, 
    pub ess_perc_dead: Array1<f64>,   
    pub drag_kw: Array1<f64>, 
    pub ess_loss_kw: Array1<f64>, 
    pub accel_kw: Array1<f64>, 
    pub ascent_kw: Array1<f64>, 
    pub rr_kw: Array1<f64>, 
    pub cur_max_roadway_chg_kw: Array1<f64>, 
    pub trace_miss_iters: Array1<f64>, 
    pub newton_iters: Array1<u32>, 
}

#[pymethods]
impl RustSimDrive{
    /// method for instantiating SimDriveRust
    #[new]
    pub fn __new__(cyc: RustCycle, veh: RustVehicle) -> Self{
        let hev_sim_count: usize = 0;
        let cyc0: RustCycle = cyc.clone();
        let sim_params = RustSimDriveParams::__new__();
        let props = RustPhysicalProperties::__new__();
        let i: usize = 1; // 1 # initialize step counter for possible use outside sim_drive_walk()
        let cyc_len = cyc.time_s.len(); //get_len() as usize;
        let cur_max_fs_kw_out = Array::zeros(cyc_len); 
        let fc_trans_lim_kw = Array::zeros(cyc_len); 
        let fc_fs_lim_kw = Array::zeros(cyc_len); 
        let fc_max_kw_in = Array::zeros(cyc_len); 
        let cur_max_fc_kw_out = Array::zeros(cyc_len); 
        let ess_cap_lim_dischg_kw = Array::zeros(cyc_len); 
        let cur_max_ess_kw_out = Array::zeros(cyc_len); 
        let cur_max_avail_elec_kw = Array::zeros(cyc_len); 
        let ess_cap_lim_chg_kw = Array::zeros(cyc_len); 
        let cur_max_ess_chg_kw = Array::zeros(cyc_len); 
        let cur_max_elec_kw = Array::zeros(cyc_len); 
        let mc_elec_in_lim_kw = Array::zeros(cyc_len); 
        let mc_transi_lim_kw = Array::zeros(cyc_len); 
        let cur_max_mc_kw_out = Array::zeros(cyc_len); 
        let ess_lim_mc_regen_perc_kw = Array::zeros(cyc_len); 
        let ess_lim_mc_regen_kw = Array::zeros(cyc_len); 
        let cur_max_mech_mc_kw_in = Array::zeros(cyc_len); 
        let cur_max_trans_kw_out = Array::zeros(cyc_len); 
        let cyc_drag_kw = Array::zeros(cyc_len); 
        let cyc_accel_kw = Array::zeros(cyc_len); 
        let cyc_ascent_kw = Array::zeros(cyc_len); 
        let cyc_trac_kw_req = Array::zeros(cyc_len); 
        let cur_max_trac_kw = Array::zeros(cyc_len); 
        let spare_trac_kw = Array::zeros(cyc_len); 
        let cyc_rr_kw = Array::zeros(cyc_len); 
        let cyc_whl_rad_per_sec = Array::zeros(cyc_len);   
        let cyc_tire_inertia_kw = Array::zeros(cyc_len); 
        let cyc_whl_kw_req = Array::zeros(cyc_len);   
        let regen_contrl_lim_kw_perc = Array::zeros(cyc_len); 
        let cyc_regen_brake_kw = Array::zeros(cyc_len); 
        let cyc_fric_brake_kw = Array::zeros(cyc_len); 
        let cyc_trans_kw_out_req = Array::zeros(cyc_len); 
        let cyc_met = Array::from_vec(vec![false; cyc_len]); 
        let trans_kw_out_ach = Array::zeros(cyc_len); 
        let trans_kw_in_ach = Array::zeros(cyc_len); 
        let cur_soc_target = Array::zeros(cyc_len); 
        let min_mc_kw_2help_fc = Array::zeros(cyc_len); 
        let mc_mech_kw_out_ach = Array::zeros(cyc_len); 
        let mc_elec_kw_in_ach = Array::zeros(cyc_len); 
        let aux_in_kw = Array::zeros(cyc_len); 
        let roadway_chg_kw_out_ach = Array::zeros(cyc_len); 
        let min_ess_kw_2help_fc = Array::zeros(cyc_len); 
        let ess_kw_out_ach = Array::zeros(cyc_len); 
        let fc_kw_out_ach = Array::zeros(cyc_len); 
        let fc_kw_out_ach_pct = Array::zeros(cyc_len); 
        let fc_kw_in_ach = Array::zeros(cyc_len); 
        let fs_kw_out_ach = Array::zeros(cyc_len); 
        let fs_cumu_mj_out_ach = Array::zeros(cyc_len); 
        let fs_kwh_out_ach = Array::zeros(cyc_len); 
        let ess_cur_kwh = Array::zeros(cyc_len); 
        let soc = Array::zeros(cyc_len); 
        let regen_buff_soc = Array::zeros(cyc_len);   
        let ess_regen_buff_dischg_kw = Array::zeros(cyc_len);   
        let max_ess_regen_buff_chg_kw = Array::zeros(cyc_len);   
        let ess_accel_buff_chg_kw = Array::zeros(cyc_len);   
        let accel_buff_soc = Array::zeros(cyc_len);   
        let max_ess_accell_buff_dischg_kw = Array::zeros(cyc_len);   
        let ess_accel_regen_dischg_kw = Array::zeros(cyc_len); 
        let mc_elec_in_kw_for_max_fc_eff = Array::zeros(cyc_len); 
        let elec_kw_req_4ae = Array::zeros(cyc_len);   
        let can_pwr_all_elec = Array::zeros(cyc_len); 
        let desired_ess_kw_out_for_ae = Array::zeros(cyc_len); 
        let ess_ae_kw_out = Array::zeros(cyc_len); 
        let er_ae_kw_out = Array::zeros(cyc_len); 
        let ess_desired_kw_4fc_eff = Array::zeros(cyc_len); 
        let ess_kw_if_fc_req = Array::zeros(cyc_len);   
        let cur_max_mc_elec_kw_in = Array::zeros(cyc_len); 
        let fc_kw_gap_fr_eff = Array::zeros(cyc_len); 
        let er_kw_if_fc_req = Array::zeros(cyc_len);   
        let mc_elec_kw_in_if_fc_req = Array::zeros(cyc_len);   
        let mc_kw_if_fc_req = Array::zeros(cyc_len);   
        let fc_forced_on = Array::from_vec(vec![false; cyc_len]); 
        let fc_forced_state = Array::zeros(cyc_len); 
        let mc_mech_kw_4forced_fc = Array::zeros(cyc_len); 
        let fc_time_on = Array::zeros(cyc_len); 
        let prev_fc_time_on = Array::zeros(cyc_len); 
        let mps_ach = Array::zeros(cyc_len); 
        let mph_ach = Array::zeros(cyc_len); 
        let dist_m = Array::zeros(cyc_len);  
        let dist_mi = Array::zeros(cyc_len);   
        let high_acc_fc_on_tag = Array::from_vec(vec![false; cyc_len]); 
        let reached_buff = Array::from_vec(vec![false; cyc_len]); 
        let max_trac_mps = Array::zeros(cyc_len); 
        let add_kwh = Array::zeros(cyc_len); 
        let dod_cycs = Array::zeros(cyc_len); 
        let ess_perc_dead = Array::zeros(cyc_len);   
        let drag_kw = Array::zeros(cyc_len); 
        let ess_loss_kw = Array::zeros(cyc_len); 
        let accel_kw = Array::zeros(cyc_len); 
        let ascent_kw = Array::zeros(cyc_len); 
        let rr_kw = Array::zeros(cyc_len); 
        let cur_max_roadway_chg_kw = Array::zeros(cyc_len); 
        let trace_miss_iters = Array::zeros(cyc_len); 
        let newton_iters = Array::zeros(cyc_len); 
        RustSimDrive{
            hev_sim_count,
            veh,
            cyc,
            cyc0,
            sim_params,
            props,
            i, // 1 # initialize step counter for possible use outside sim_drive_walk()
            cur_max_fs_kw_out, 
            fc_trans_lim_kw, 
            fc_fs_lim_kw, 
            fc_max_kw_in, 
            cur_max_fc_kw_out, 
            ess_cap_lim_dischg_kw, 
            cur_max_ess_kw_out, 
            cur_max_avail_elec_kw, 
            ess_cap_lim_chg_kw, 
            cur_max_ess_chg_kw, 
            cur_max_elec_kw, 
            mc_elec_in_lim_kw, 
            mc_transi_lim_kw, 
            cur_max_mc_kw_out, 
            ess_lim_mc_regen_perc_kw, 
            ess_lim_mc_regen_kw, 
            cur_max_mech_mc_kw_in, 
            cur_max_trans_kw_out, 
            cyc_drag_kw, 
            cyc_accel_kw, 
            cyc_ascent_kw, 
            cyc_trac_kw_req, 
            cur_max_trac_kw, 
            spare_trac_kw, 
            cyc_rr_kw, 
            cyc_whl_rad_per_sec,   
            cyc_tire_inertia_kw, 
            cyc_whl_kw_req,   
            regen_contrl_lim_kw_perc, 
            cyc_regen_brake_kw, 
            cyc_fric_brake_kw, 
            cyc_trans_kw_out_req, 
            cyc_met, 
            trans_kw_out_ach, 
            trans_kw_in_ach, 
            cur_soc_target, 
            min_mc_kw_2help_fc, 
            mc_mech_kw_out_ach, 
            mc_elec_kw_in_ach, 
            aux_in_kw, 
            roadway_chg_kw_out_ach, 
            min_ess_kw_2help_fc, 
            ess_kw_out_ach, 
            fc_kw_out_ach, 
            fc_kw_out_ach_pct, 
            fc_kw_in_ach, 
            fs_kw_out_ach, 
            fs_cumu_mj_out_ach, 
            fs_kwh_out_ach, 
            ess_cur_kwh, 
            soc, 
            regen_buff_soc,   
            ess_regen_buff_dischg_kw,   
            max_ess_regen_buff_chg_kw,   
            ess_accel_buff_chg_kw,   
            accel_buff_soc,   
            max_ess_accell_buff_dischg_kw,   
            ess_accel_regen_dischg_kw, 
            mc_elec_in_kw_for_max_fc_eff, 
            elec_kw_req_4ae,   
            can_pwr_all_elec, 
            desired_ess_kw_out_for_ae, 
            ess_ae_kw_out, 
            er_ae_kw_out, 
            ess_desired_kw_4fc_eff, 
            ess_kw_if_fc_req,   
            cur_max_mc_elec_kw_in, 
            fc_kw_gap_fr_eff, 
            er_kw_if_fc_req,   
            mc_elec_kw_in_if_fc_req,   
            mc_kw_if_fc_req,   
            fc_forced_on,
            fc_forced_state,
            mc_mech_kw_4forced_fc, 
            fc_time_on, 
            prev_fc_time_on, 
            mps_ach, 
            mph_ach, 
            dist_m,   
            dist_mi,   
            high_acc_fc_on_tag, 
            reached_buff,
            max_trac_mps, 
            add_kwh, 
            dod_cycs, 
            ess_perc_dead,   
            drag_kw, 
            ess_loss_kw, 
            accel_kw, 
            ascent_kw, 
            rr_kw, 
            cur_max_roadway_chg_kw, 
            trace_miss_iters, 
            newton_iters, 
        }
    }

    /// Receives second-by-second cycle information, vehicle properties, 
    /// and an initial state of charge and runs sim_drive_step to perform a 
    /// backward facing powertrain simulation. Method 'sim_drive' runs this
    /// iteratively to achieve correct SOC initial and final conditions, as 
    /// needed.
    /// 
    /// Arguments
    /// ------------
    /// initSoc (optional): initial battery state-of-charge (SOC) for electrified vehicles
    /// auxInKw: auxInKw override.  Array of same length as cyc.time_s.  
    ///         Default of np.zeros(1) causes veh.aux_kw to be used. If zero is actually
    ///         desired as an override, either set veh.aux_kw = 0 before instantiaton of
    ///         SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
    ///         the final value to non-zero prevents override mechanism.  
    pub fn sim_drive_walk(&mut self, init_soc:f64) {
        // TODO: implement method for `init_arrays`
        // TODO: implement method for aux_in_kw_override
        self.cyc_met[0] = true;
        self.cur_soc_target[0] = self.veh.max_soc;
        self.ess_cur_kwh[0] = init_soc * self.veh.max_ess_kwh;
        self.soc[0] = init_soc;
        self.mps_ach[0] = self.cyc0.mps[0];
        self.mph_ach[0] = self.cyc0.mph()[0];

        if self.sim_params.missed_trace_correction {
            self.cyc = self.cyc0.clone();  // reset the cycle in case it has been manipulated
        }
        self.i = 1; // time step counter
        while self.i < self.cyc.time_s.len() {
            self.sim_drive_step()
        }

    // TODO: uncomment and implement
    //    if (self.cyc.dt_s > 5).any() and self.sim_params.verbose:
    //         if self.sim_params.missed_trace_correction:
    //             print('Max time dilation factor =', (round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)))
    //         print("Warning: large time steps affect accuracy significantly.") 
    //         print("To suppress this message, view the doc string for simdrive.SimDriveParams.")
    //         print('Max time step =', (round(self.cyc.dt_s.max(), 3)))
    }
    
    /// Step through 1 time step.
    fn sim_drive_step(&mut self) {
        self.solve_step(self.i);
        
        // TODO: implement and uncomment
        // if self.sim_params.missed_trace_correction && (self.cyc0.dist_m.slice(s![0..self.i]).sum() > 0){
        //     self.set_time_dilation(self.i)
        // }

        // TODO: implement something for coasting here
        // if self.impose_coast[i] == true
        //     self.set_coast_speeed(i)

        self.i += 1  // increment time step counter
    }
    
    /// Perform all the calculations to solve 1 time step.
    fn solve_step(&mut self, i:usize) {
        self.set_misc_calcs(i);
        self.set_comp_lims(i);
        self.set_power_calcs(i);
        self.set_ach_speed(i);
        // TODO: uncomment these!
        // self.set_hybrid_cont_calcs(i)
        // self.set_fc_forced_state(i) # can probably be *mostly* done with list comprehension in post processing
        // self.set_hybrid_cont_decisions(i)
        // self.set_fc_power(i)
    }

    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    fn set_misc_calcs(&mut self, i:usize) {
        // if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        if self.aux_in_kw.slice(s![i..self.len()-1]).iter().all(|&x| x == 0.0) {
            // if all elements after i-1 are zero, trigger default behavior; otherwise, use override value 
            if self.veh.no_elec_aux{
                self.aux_in_kw[i] = self.veh.aux_kw / self.veh.alt_eff;
            } else {
                self.aux_in_kw[i] = self.veh.aux_kw;
            }
        }
        // Is SOC below min threshold?
        if self.soc[i-1] < (self.veh.min_soc + self.veh.perc_high_acc_buf) {
            self.reached_buff[i] = false;
        } else {
            self.reached_buff[i] = true;
        }

        // Does the engine need to be on for low SOC or high acceleration
        if self.soc[i-1] < self.veh.min_soc || (self.high_acc_fc_on_tag[i-1] && !(self.reached_buff[i])){
            self.high_acc_fc_on_tag[i] = true } else{
            self.high_acc_fc_on_tag[i] = false
        }
        self.max_trac_mps[i] = self.mps_ach[i-1] + (self.veh.max_trac_mps2 * self.cyc.dt_s()[i])

    }

    /// Sets component limits for time step 'i'
    /// Arguments
    /// ------------
    /// i: index of time step
    /// initSoc: initial SOC for electrified vehicles
    fn set_comp_lims(&mut self, i:usize) {
        // max fuel storage power output
        self.cur_max_fs_kw_out[i] = min(
            self.veh.max_fuel_stor_kw,
            self.fs_kw_out_ach[i-1] + 
                self.veh.max_fuel_stor_kw / self.veh.fuel_stor_secs_to_peak_pwr * self.cyc.dt_s()[i]);
        // maximum fuel storage power output rate of change
        self.fc_trans_lim_kw[i] = self.fc_kw_out_ach[i-1] + (
            self.veh.max_fuel_conv_kw / self.veh.fuel_conv_secs_to_peak_pwr * self.cyc.dt_s()[i]
        );

        self.fc_max_kw_in[i] = min(self.cur_max_fs_kw_out[i], self.veh.max_fuel_stor_kw);
        self.fc_fs_lim_kw[i] = arrmax(&self.veh.input_kw_out_array);
        self.cur_max_fc_kw_out[i] = min(
            self.veh.max_fuel_conv_kw, 
            min(self.fc_fs_lim_kw[i], self.fc_trans_lim_kw[i]));

        if self.veh.max_ess_kwh == 0.0 || self.soc[i-1] < self.veh.min_soc {
            self.ess_cap_lim_dischg_kw[i] = 0.0;
        } else {
            self.ess_cap_lim_dischg_kw[i] = self.veh.max_ess_kwh * self.veh.ess_round_trip_eff.sqrt() * 3.6e3 * (
                self.soc[i-1] - self.veh.min_soc) / self.cyc.dt_s()[i];
        }
        self.cur_max_ess_kw_out[i] = min(
            self.veh.max_ess_kw, self.ess_cap_lim_dischg_kw[i]);

        if self.veh.max_ess_kwh == 0.0 || self.veh.max_ess_kw == 0.0 {
            self.ess_cap_lim_chg_kw[i] = 0.0;
        } else {
            self.ess_cap_lim_chg_kw[i] = max(
                (self.veh.max_soc - self.soc[i-1]) * self.veh.max_ess_kwh / self.veh.ess_round_trip_eff.sqrt() / 
                (self.cyc.dt_s()[i] / 3.6e3), 
                0.0);
        }

        self.cur_max_ess_chg_kw[i] = min(self.ess_cap_lim_chg_kw[i], self.veh.max_ess_kw);

        // Current maximum electrical power that can go toward propulsion, not including motor limitations
        if self.veh.fc_eff_type == H2FC {
            self.cur_max_elec_kw[i] = self.cur_max_fc_kw_out[i] + self.cur_max_roadway_chg_kw[i] + self.cur_max_ess_kw_out[i] - self.aux_in_kw[i];
        } else {
            self.cur_max_elec_kw[i] = self.cur_max_roadway_chg_kw[i] + self.cur_max_ess_kw_out[i] - self.aux_in_kw[i];
        }

        // Current maximum electrical power that can go toward propulsion, including motor limitations
        self.cur_max_avail_elec_kw[i] = min(self.cur_max_elec_kw[i], self.veh.mc_max_elec_in_kw);

        if self.cur_max_elec_kw[i] > 0.0 {
            // limit power going into e-machine controller to
            if self.cur_max_avail_elec_kw[i] == arrmax(&self.veh.mc_kw_in_array) {
                let end = self.veh.mc_kw_out_array.len() - 1;
                self.mc_elec_in_lim_kw[i] = min(self.veh.mc_kw_out_array[end], self.veh.max_motor_kw);
            }
            else {
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[np_argmax(
                        &self.veh.mc_kw_in_array.map(|x| *x > min(
                            arrmax(&self.veh.mc_kw_in_array) - 0.01, 
                            self.cur_max_avail_elec_kw[i])
                        )
                    ) - 1 as usize],
                    self.veh.max_motor_kw)}
        }
        else {
            self.mc_elec_in_lim_kw[i] = 0.0;
        }

        // Motor transient power limit
        self.mc_transi_lim_kw[i] = self.mc_mech_kw_out_ach[i-1].abs() + self.veh.max_motor_kw / self.veh.motor_secs_to_peak_pwr * self.cyc.dt_s()[i];

        self.cur_max_mc_kw_out[i] = max(
            min(min(
                self.mc_elec_in_lim_kw[i], 
                self.mc_transi_lim_kw[i]),
                if self.veh.stop_start {0.0} else {1.0} * self.veh.max_motor_kw),
            -self.veh.max_motor_kw
        );

        if self.cur_max_mc_kw_out[i] == 0.0 {
            self.cur_max_mc_elec_kw_in[i] = 0.0;
        } else {
            if self.cur_max_mc_kw_out[i] == self.veh.max_motor_kw {
                let end = self.veh.mc_full_eff_array.len() - 1;
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array[end];
            } else {
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array[cmp::max(
                    1, 
                    np_argmax(
                        &self.veh.mc_kw_out_array.map(|x| *x > min(
                            self.veh.max_motor_kw - 0.01, self.cur_max_mc_kw_out[i]))) - 1 
                        )
                    ]
            };
        }

        if self.veh.max_motor_kw == 0.0 {
            self.ess_lim_mc_regen_perc_kw[i] = 0.0;
        }
        else {
            self.ess_lim_mc_regen_perc_kw[i] = min(
                (self.cur_max_ess_chg_kw[i] + self.aux_in_kw[i]) / self.veh.max_motor_kw, 1.0);
        }
        if self.cur_max_ess_chg_kw[i] == 0.0 {
            self.ess_lim_mc_regen_kw[i] = 0.0;
        } else {
            if self.veh.max_motor_kw == self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i] {
                let end = self.veh.mc_full_eff_array.len() - 1;
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw, self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[end]);
            }
            else {
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw, 
                    self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[
                        cmp::max(1, 
                            np_argmax(
                                &self.veh.mc_kw_out_array.map(|x| *x > min(
                                    self.veh.max_motor_kw - 0.01, 
                                    self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]
                                ))
                            ) - 1
                        )
                    ]
                );
            }
        }
        self.cur_max_mech_mc_kw_in[i] = min(
            self.ess_lim_mc_regen_kw[i], self.veh.max_motor_kw);
        
        self.cur_max_trac_kw[i] = self.veh.wheel_coef_of_fric * self.veh.drive_axle_weight_frac * self.veh.veh_kg * self.props.a_grav_mps2
            / (1.0 + self.veh.veh_cg_m * self.veh.wheel_coef_of_fric / self.veh.wheel_base_m) / 1e3 * self.max_trac_mps[i];

        if self.veh.fc_eff_type == H2FC {
            if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] - self.aux_in_kw[i]) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                );
            } else 
                {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] - min(self.cur_max_elec_kw[i], 0.0)) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                );
            }
        }
        else {
            if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] - self.aux_in_kw[i]) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                );
            }
            else {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] - min(self.cur_max_elec_kw[i], 0.0)) * self.veh.trans_eff, 
                    self.cur_max_trac_kw[i] / self.veh.trans_eff
                );
            }
        }
    }

    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.  
    /// Arguments
    /// ------------
    /// i: index of time step
    fn set_power_calcs(&mut self, i:usize) {
        let mps_ach = if &self.newton_iters[i] > &0u32 {
            self.mps_ach[i]
        } else {
            self.cyc.mps[i]
        };

        self.cyc_drag_kw[i] = 0.5 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2 * (
            (self.mps_ach[i-1] + mps_ach) / 2.0).powf(3.0) / 1e3;
        self.cyc_accel_kw[i] = self.veh.veh_kg / (2.0 * self.cyc.dt_s()[i]) * (mps_ach.powf(2.0) - self.mps_ach[i-1].powf(2.0)) / 1e3;
        self.cyc_ascent_kw[i] = self.props.a_grav_mps2 * self.cyc.grade[i].atan().sin() * 
            self.veh.veh_kg * (self.mps_ach[i-1] + mps_ach) / 2.0 / 1e3;
        self.cyc_trac_kw_req[i] = self.cyc_drag_kw[i] + self.cyc_accel_kw[i] + self.cyc_ascent_kw[i];
        self.spare_trac_kw[i] = self.cur_max_trac_kw[i] - self.cyc_trac_kw_req[i];
        self.cyc_rr_kw[i] = self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef * 
            self.cyc.grade[i].atan().cos() * (self.mps_ach[i-1] + mps_ach) / 2.0 / 1e3;
        self.cyc_whl_rad_per_sec[i] = mps_ach / self.veh.wheel_radius_m;
        self.cyc_tire_inertia_kw[i] = (
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * self.cyc_whl_rad_per_sec[i].powf(2.0) / self.cyc.dt_s()[i] -
            0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * (self.mps_ach[i-1] / self.veh.wheel_radius_m).powf(2.0) / 
            self.cyc.dt_s()[i]) / 1e3;

        self.cyc_whl_kw_req[i] = self.cyc_trac_kw_req[i] + self.cyc_rr_kw[i] + self.cyc_tire_inertia_kw[i];
        self.regen_contrl_lim_kw_perc[i] = self.veh.max_regen / (1.0 + self.veh.regen_a * (-self.veh.regen_b * (
            (self.cyc.mph()[i] + self.mps_ach[i-1] * params::MPH_PER_MPS) / 2.0 + 1.0)).exp());
        self.cyc_regen_brake_kw[i] = max(min(
                self.cur_max_mech_mc_kw_in[i] * self.veh.trans_eff, 
                self.regen_contrl_lim_kw_perc[i] * -self.cyc_whl_kw_req[i]), 
            0.0
        );
        self.cyc_fric_brake_kw[i] = -min(self.cyc_regen_brake_kw[i] + self.cyc_whl_kw_req[i], 0.0);
        self.cyc_trans_kw_out_req[i] = self.cyc_whl_kw_req[i] + self.cyc_fric_brake_kw[i];

        if self.cyc_trans_kw_out_req[i] <= self.cur_max_trans_kw_out[i] {
            self.cyc_met[i] = true;
            self.trans_kw_out_ach[i] = self.cyc_trans_kw_out_req[i];
        }

        else {
            self.cyc_met[i] = false;
            self.trans_kw_out_ach[i] = self.cur_max_trans_kw_out[i];
        }

        if self.trans_kw_out_ach[i] > 0.0 {
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] / self.veh.trans_eff;
        }
        else {
            self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] * self.veh.trans_eff;
        }

        if self.cyc_met[i]{
            if self.veh.fc_eff_type == H2FC {
                self.min_mc_kw_2help_fc[i] = max(
                    self.trans_kw_in_ach[i], -self.cur_max_mech_mc_kw_in[i]);
            } else {
                self.min_mc_kw_2help_fc[i] = max(
                    self.trans_kw_in_ach[i] - self.cur_max_fc_kw_out[i], -self.cur_max_mech_mc_kw_in[i]);
            }
        } else {
            self.min_mc_kw_2help_fc[i] = max(
                self.cur_max_mc_kw_out[i], -self.cur_max_mech_mc_kw_in[i]);
        }
    }

    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    fn set_ach_speed(&mut self, i:usize) {
        // Cycle is met
        if self.cyc_met[i] {
            self.mps_ach[i] = self.cyc.mps[i];
        }

        //Cycle is not met
        else {
            let drag3 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * 
                self.veh.drag_coef * self.veh.frontal_area_m2;
            let accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s()[i];
            let drag2 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * 
                self.veh.drag_coef * self.veh.frontal_area_m2 * self.mps_ach[i-1];
            let wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * 
                self.veh.num_wheels / (self.cyc.dt_s()[i] * self.veh.wheel_radius_m.powf(2.0));
            let drag1 = 3.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * 
                self.veh.frontal_area_m2 * self.mps_ach[i-1].powf(2.0);
            let roll1 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * self.veh.wheel_rr_coef 
                * self.cyc.grade[i].atan().cos();
            let ascent1 = 0.5 * self.props.a_grav_mps2 * self.cyc.grade[i].atan().sin() * self.veh.veh_kg;
            let accel0 = -0.5 * self.veh.veh_kg * self.mps_ach[i-1].powf(2.0) / self.cyc.dt_s()[i];
            let drag0 = 1.0 / 16.0 * self.props.air_density_kg_per_m3 * self.veh.drag_coef * 
                self.veh.frontal_area_m2 * self.mps_ach[i-1].powf(3.0);
            let roll0 = 0.5 * self.veh.veh_kg * self.props.a_grav_mps2 * 
                self.veh.wheel_rr_coef * self.cyc.grade[i].atan().cos() * self.mps_ach[i-1];
            let ascent0 = 0.5 * self.props.a_grav_mps2 * self.cyc.grade[i].atan().sin()
                * self.veh.veh_kg * self.mps_ach[i-1];
            let wheel0 = -0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels * 
                self.mps_ach[i-1].powf(2.0) / (self.cyc.dt_s()[i] * self.veh.wheel_radius_m.powf(2.0));

            let total3 = drag3 / 1e3;
            let total2 = (accel2 + drag2 + wheel2) / 1e3;
            let total1 = (drag1 + roll1 + ascent1) / 1e3;
            let total0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / 1e3 - self.cur_max_trans_kw_out[i];

            let totals = array![total3, total2, total1, total0];

            let t3 = totals[0];
            let t2 = totals[1];
            let t1 = totals[2];
            let t0 = totals[3];
            // initial guess
            let xi = max(1.0, self.mps_ach[i-1]);
            // stop criteria
            let max_iter = self.sim_params.newton_max_iter;
            let xtol = self.sim_params.newton_xtol;
            // solver gain
            let g = self.sim_params.newton_gain;
            let yi = t3 * xi.powf(3.0) + t2 * xi.powf(2.0) + t1 * xi + t0;
            let mi = 3.0 * t3 * xi.powf(2.0) + 2.0 * t2 * xi + t1;
            let bi = yi - xi * mi;
            let mut xs = vec![xi];
            let mut ys = vec![yi];
            let mut ms = vec![mi];
            let mut bs = vec![bi];
            let mut iterate = 1;
            let mut converged = false;
            while iterate < max_iter && !converged {
                let end = xs.len() - 1;
                let xi = xs[end] * (1.0 - g) - g * bs[end] / ms[end];
                let yi = t3 * xi.powf(3.0) + t2 * xi.powf(2.0) + t1 * xi + t0;
                let mi = 3.0 * t3 * xi.powf(2.0) + 2.0 * t2 * xi + t1;
                let bi = yi - xi * mi;
                xs.push(xi);
                ys.push(yi);
                ms.push(mi);
                bs.push(bi);
                let end = xs.len() - 1;
                converged = ((xs[end] - xs[end-1]) / xs[end-1]).abs() < xtol;
                iterate += 1;
            }
            
            self.newton_iters[i] = iterate;

            let _ys = Array::from_vec(ys).map(|x| x.abs());
            self.mps_ach[i] = xs[_ys.iter().position(|&x| x == arrmin(&_ys)).unwrap()];
        }

        self.set_power_calcs(i);

        self.mph_ach[i] = self.mps_ach[i] * params::MPH_PER_MPS;
        self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s()[i];
        self.dist_mi[i] = self.dist_m[i] * 1.0 / params::M_PER_MI;
    }  

    // Methods for getting and setting arrays and other complex fields
    // note that python cannot specify a specific index to set but must reset the entire array 

    fn len(&self) -> usize {
        self.cyc.time_s.len()
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
    pub fn get_newton_iters(&self) -> PyResult<Vec<u32>>{
      Ok(self.newton_iters.to_vec())
    }
    #[setter]
    pub fn set_newton_iters(&mut self, new_value:Vec<u32>) -> PyResult<()>{
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_drive_walk() {
        // CYCLE
        let time_s = vec![0.0, 1.0, 2.0];
        let speed_mps = vec![0.0, 1.0, 0.0];
        let grade = vec![0.0, 0.0, 0.0];
        let road_type = vec![0.0, 0.0, 0.0];        
        let name = String::from("test");
        let cycle_length: usize = time_s.len();
        let cyc = RustCycle::__new__(time_s, speed_mps, grade, road_type, name);

        // VEHICLE
        let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
        let selection: u32 = 5;
        let veh_year: u32 = 2016;
        let veh_pt_type = String::from("Conv");
        let drag_coef: f64 = 0.355;
        let frontal_area_m2: f64 = 3.066;
        let glider_kg: f64 = 1359.166;
        let veh_cg_m: f64 = 0.53;
        let drive_axle_weight_frac: f64 = 0.59;
        let wheel_base_m: f64 = 2.6;
        let cargo_kg: f64 = 136.0;
        let veh_override_kg: f64 = f64::NAN;
        let comp_mass_multiplier: f64 = 1.4;
        let max_fuel_stor_kw: f64 = 2000.0;
        let fuel_stor_secs_to_peak_pwr: f64 = 1.0;
        let fuel_stor_kwh: f64 = 504.0;
        let fuel_stor_kwh_per_kg: f64 = 9.89;
        let max_fuel_conv_kw: f64 = 125.0;
        let fc_pwr_out_perc: Vec<f64> = vec![0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0];
        let fc_eff_map: Vec<f64> = vec![0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3];
        let fc_eff_type: String = String::from("SI");
        let fuel_conv_secs_to_peak_pwr: f64 = 6.0;
        let fuel_conv_base_kg: f64 = 61.0;
        let fuel_conv_kw_per_kg: f64 = 2.13;
        let min_fc_time_on: f64 = 30.0;
        let idle_fc_kw: f64 = 2.5;
        let max_motor_kw: f64 = 0.0;
        let mc_pwr_out_perc: Vec<f64> = vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mc_eff_map: Vec<f64> = vec![0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92];
        let motor_secs_to_peak_pwr: f64 = 4.0;
        let mc_pe_kg_per_kw: f64 = 0.833;
        let mc_pe_base_kg: f64 = 21.6;
        let max_ess_kw: f64 = 0.0;
        let max_ess_kwh: f64 = 0.0;
        let ess_kg_per_kwh: f64 = 8.0;
        let ess_base_kg: f64 = 75.0;
        let ess_round_trip_eff: f64 = 0.97;
        let ess_life_coef_a: f64 = 110.0;
        let ess_life_coef_b: f64 = -0.6811;
        let min_soc: f64 = 0.4;
        let max_soc: f64 = 0.8;
        let ess_dischg_to_fc_max_eff_perc: f64 = 0.0;
        let ess_chg_to_fc_max_eff_perc: f64 = 0.0;
        let wheel_inertia_kg_m2: f64 = 0.815;
        let num_wheels: f64 = 4.0;
        let wheel_rr_coef: f64 = 0.006;
        let wheel_radius_m: f64 = 0.336;
        let wheel_coef_of_fric: f64 = 0.7;
        let max_accel_buffer_mph: f64 = 60.0;
        let max_accel_buffer_perc_of_useable_soc: f64 = 0.2;
        let perc_high_acc_buf: f64 = 0.0;
        let mph_fc_on: f64 = 30.0;
        let kw_demand_fc_on: f64 = 100.0;
        let max_regen: f64 = 0.98;
        let stop_start: bool = false;
        let force_aux_on_fc: f64 = 0.0;
        let alt_eff: f64 = 1.0;
        let chg_eff: f64 = 0.86;
        let aux_kw: f64 = 0.7;
        let trans_kg: f64 = 114.0;
        let trans_eff: f64 = 0.92;
        let ess_to_fuel_ok_error: f64 = 0.005;
        let val_udds_mpgge: f64 = 23.0;
        let val_hwy_mpgge: f64 = 32.0;
        let val_comb_mpgge: f64 = 26.0;
        let val_udds_kwh_per_mile: f64 = f64::NAN;
        let val_hwy_kwh_per_mile: f64 = f64::NAN;
        let val_comb_kwh_per_mile: f64 = f64::NAN;
        let val_cd_range_mi: f64 = f64::NAN;
        let val_const65_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const60_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const55_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const45_mph_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_udds_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_hwy_kwh_per_mile: f64 = f64::NAN;
        let val0_to60_mph: f64 = 9.9;
        let val_ess_life_miles: f64 = f64::NAN;
        let val_range_miles: f64 = f64::NAN;
        let val_veh_base_cost: f64 = f64::NAN;
        let val_msrp: f64 = f64::NAN;
        let props = RustPhysicalProperties::__new__();  
        let large_baseline_eff: Vec<f64> = vec![0.83, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.94, 0.94, 0.93, 0.92];
        let small_baseline_eff: Vec<f64> = vec![0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92];
        let small_motor_power_kw: f64 = 7.5;
        let large_motor_power_kw: f64 = 75.0;
        let fc_perc_out_array: Vec<f64> = vec![
          0.0  , 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01 , 0.011,
          0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023,
          0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.035, 0.04 , 0.045, 0.05 , 0.055,
          0.06 , 0.065, 0.07 , 0.08 , 0.09 , 0.1  , 0.11 , 0.12 , 0.13 , 0.14 , 0.15 , 0.16 ,
          0.17 , 0.18 , 0.19 , 0.2  , 0.21 , 0.22 , 0.23 , 0.24 , 0.25 , 0.26 , 0.27 , 0.28 ,
          0.29 , 0.3  , 0.31 , 0.32 , 0.33 , 0.34 , 0.35 , 0.36 , 0.37 , 0.38 , 0.39 , 0.4  ,
          0.41 , 0.42 , 0.43 , 0.44 , 0.45 , 0.46 , 0.47 , 0.48 , 0.49 , 0.5  , 0.51 , 0.52 ,
          0.53 , 0.54 , 0.55 , 0.56 , 0.57 , 0.58 , 0.59 , 0.6  , 0.65 , 0.7  , 0.75 , 0.8  ,
          0.85 , 0.9  , 0.95 , 1.0
        ];
        let max_roadway_chg_kw: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let charging_on: bool = false;
        let no_elec_sys: bool = true;
        let no_elec_aux: bool = true;
        let veh = RustVehicle::__new__(
          scenario_name,
          selection,
          veh_year,
          veh_pt_type,
          drag_coef,
          frontal_area_m2,
          glider_kg,
          veh_cg_m,
          drive_axle_weight_frac,
          wheel_base_m,
          cargo_kg,
          veh_override_kg,
          comp_mass_multiplier,
          max_fuel_stor_kw,
          fuel_stor_secs_to_peak_pwr,
          fuel_stor_kwh,
          fuel_stor_kwh_per_kg,
          max_fuel_conv_kw,
          fc_pwr_out_perc,
          fc_eff_map,
          fc_eff_type,
          fuel_conv_secs_to_peak_pwr,
          fuel_conv_base_kg,
          fuel_conv_kw_per_kg,
          min_fc_time_on,
          idle_fc_kw,
          max_motor_kw,
          mc_pwr_out_perc,
          mc_eff_map,
          motor_secs_to_peak_pwr,
          mc_pe_kg_per_kw,
          mc_pe_base_kg,
          max_ess_kw,
          max_ess_kwh,
          ess_kg_per_kwh,
          ess_base_kg,
          ess_round_trip_eff,
          ess_life_coef_a,
          ess_life_coef_b,
          min_soc,
          max_soc,
          ess_dischg_to_fc_max_eff_perc,
          ess_chg_to_fc_max_eff_perc,
          wheel_inertia_kg_m2,
          num_wheels,
          wheel_rr_coef,
          wheel_radius_m,
          wheel_coef_of_fric,
          max_accel_buffer_mph,
          max_accel_buffer_perc_of_useable_soc,
          perc_high_acc_buf,
          mph_fc_on,
          kw_demand_fc_on,
          max_regen,
          stop_start,
          force_aux_on_fc,
          alt_eff,
          chg_eff,
          aux_kw,
          trans_kg,
          trans_eff,
          ess_to_fuel_ok_error,
          val_udds_mpgge,
          val_hwy_mpgge,
          val_comb_mpgge,
          val_udds_kwh_per_mile,
          val_hwy_kwh_per_mile,
          val_comb_kwh_per_mile,
          val_cd_range_mi,
          val_const65_mph_kwh_per_mile,
          val_const60_mph_kwh_per_mile,
          val_const55_mph_kwh_per_mile,
          val_const45_mph_kwh_per_mile,
          val_unadj_udds_kwh_per_mile,
          val_unadj_hwy_kwh_per_mile,
          val0_to60_mph,
          val_ess_life_miles,
          val_range_miles,
          val_veh_base_cost,
          val_msrp,
          props,  
          large_baseline_eff,
          small_baseline_eff,
          small_motor_power_kw,
          large_motor_power_kw,
          fc_perc_out_array,
          charging_on,
          no_elec_sys,
          no_elec_aux,
          max_roadway_chg_kw,
        );

        // SIM DRIVE
        let mut sd = RustSimDrive::__new__(cyc, veh);
        let init_soc: f64 = 0.5;
        sd.sim_drive_walk(init_soc);

        let expected_final_i: usize = cycle_length;
        assert_eq!(sd.i, expected_final_i);
    }
}