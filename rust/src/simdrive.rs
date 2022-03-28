extern crate ndarray;
use ndarray::{Array, Array1};
extern crate pyo3;
use pyo3::prelude::*;
use pyo3::exceptions;

use super::params::RustPhysicalProperties;
use super::vehicle::*;
use super::cycle::RustCycle;


fn handle_sd_res(res:Result<(), String>) -> PyResult<()> {
    match res {
        Ok(()) => Ok(()),
        Err(msg) => Err(exceptions::PyRuntimeError::new_err(msg))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
/// Struct containing time trace data
pub struct RustSimDriveParams{
    #[pyo3(get, set)]  // enables get/set access from python for simple data types
    pub missed_trace_correction: bool, // if true, missed trace correction is active, default = false
    #[pyo3(get, set)]
    pub max_time_dilation: f64,
    #[pyo3(get, set)]
    pub min_time_dilation: f64,
    #[pyo3(get, set)]
    pub time_dilation_tol: f64,
    #[pyo3(get, set)]
    pub max_trace_miss_iters: u32,
    #[pyo3(get, set)]
    pub trace_miss_speed_mps_tol: f64,
    #[pyo3(get, set)]
    pub trace_miss_time_tol: f64,
    #[pyo3(get, set)]
    pub trace_miss_dist_tol: f64,
    #[pyo3(get, set)]
    pub sim_count_max: usize,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub newton_gain: f64,
    #[pyo3(get, set)]
    pub newton_max_iter: u32,
    #[pyo3(get, set)]
    pub newton_xtol: f64,
    #[pyo3(get, set)]
    pub energy_audit_error_tol: f64,
    #[pyo3(get, set)]
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
        let sim_count_max: usize = 30; // max allowable number of HEV SOC iterations
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
    pub can_pwr_all_elec: Array1<bool>,
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
    pub fc_forced_state: Array1<u32>,
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
    pub fuel_kj: f64,
    pub ess_dischg_kj: f64,
    #[pyo3(get)]
    pub energy_audit_error: f64,
    #[pyo3(get)]
    pub mpgge: f64,
    #[pyo3(get)]
    pub roadway_chg_kj: f64,
    #[pyo3(get)]
    pub battery_kwh_per_mi: f64,
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
        let can_pwr_all_elec = Array::from_vec(vec![false; cyc_len]);
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
        let fuel_kj: f64 = 0.0;
        let ess_dischg_kj: f64 = 0.0;
        let energy_audit_error: f64 = 0.0;
        let mpgge: f64 = 0.0;
        let roadway_chg_kj: f64 = 0.0;
        let battery_kwh_per_mi: f64 = 0.0;
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
            fuel_kj,
            ess_dischg_kj,
            energy_audit_error,
            mpgge,
            roadway_chg_kj,
            battery_kwh_per_mi,
        }
    }

    // wrappers for core methods
    // TODO, put doc strings on these and all structs
    // comments preceding a struct, method, or function definition with `///` instead of `\\`
    // get interpreted as doc strings in python

    /// Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
    /// Arguments
    /// ------------
    /// init_soc: initial SOC for electrified vehicles.  
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
    ///     Default of None causes veh.aux_kw to be used. 
    pub fn sim_drive(&mut self, init_soc:Option<f64>, aux_in_kw_override:Option<Vec<f64>>) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(|x| Array1::from(x));
        handle_sd_res(self.sim_drive_rust(init_soc, aux_in_kw_override))
    }

    /// Receives second-by-second cycle information, vehicle properties,
    /// and an initial state of charge and runs sim_drive_step to perform a
    /// backward facing powertrain simulation. Method 'sim_drive' runs this
    /// iteratively to achieve correct SOC initial and final conditions, as
    /// needed.
    ///
    /// Arguments
    /// ------------
    /// init_soc (optional): initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///         None causes veh.aux_kw to be used. 
    pub fn sim_drive_walk(&mut self, init_soc: f64, aux_in_kw_override:Option<Vec<f64>>) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(|x| Array1::from(x));
        handle_sd_res(self.walk(init_soc, aux_in_kw_override))
    }

    /// Step through 1 time step.
    pub fn sim_drive_step(&mut self) -> PyResult<()> {
        handle_sd_res(self.step())
    }

    /// Perform all the calculations to solve 1 time step.
    pub fn solve_step(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.solve_step_rust(i))
    }

    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    pub fn set_misc_calcs(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_misc_calcs_rust(i))
    }

    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_comp_lims(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_comp_lims_rust(i))
    }

    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_power_calcs_rust(i))
    }

    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_ach_speed_rust(i))
    }

    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_hybrid_cont_calcs_rust(i))
    }

    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    /// `_py` extension is needed to avoid name collision with getter/setter methods
    pub fn set_fc_forced_state_py(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_fc_forced_state_rust(i))
    }

    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_hybrid_cont_decisions_rust(i))
    }

    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_ess_power(&mut self, i:usize) -> PyResult<()> {
        handle_sd_res(self.set_fc_ess_power_rust(i))
    }

    /// Sets scalar variables that can be calculated after a cycle is run. 
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars(& mut self) -> PyResult<()> {
        handle_sd_res(self.set_post_scalars_rust())
    }

    /// Return length of time arrays
    pub fn len(&self) -> usize {
        self.cyc.time_s.len()
    }

    // Methods for getting and setting arrays and other complex fields
    // note that python cannot specify a specific index to set but must reset the entire array
    // doc strings not needed for getters or setters

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
    pub fn get_can_pwr_all_elec(&self) -> PyResult<Vec<bool>>{
      Ok(self.can_pwr_all_elec.to_vec())
    }
    #[setter]
    pub fn set_can_pwr_all_elec(&mut self, new_value:Vec<bool>) -> PyResult<()>{
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
    pub fn get_fc_forced_state(&self) -> PyResult<Vec<u32>>{
      Ok(self.fc_forced_state.to_vec())
    }
    #[setter]
    pub fn set_fc_forced_state(&mut self, new_value:Vec<u32>) -> PyResult<()>{
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

    #[getter]
    pub fn get_fuel_kj(&self) -> PyResult<f64>{
      Ok(self.fuel_kj)
    }

    #[getter]
    pub fn get_ess_dischg_kj(&self) -> PyResult<f64>{
      Ok(self.ess_dischg_kj)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_walk() {
//         // CYCLE
//         let cyc = RustCycle::test_cyc();
//         let cycle_length = cyc.len();

//         // VEHICLE

//         let veh = RustVehicle::test_veh();

//         // SIM DRIVE
//         let mut sd = RustSimDrive::__new__(cyc, veh);
//         let init_soc: f64 = 0.5;
//         sd.walk(init_soc);

//         let expected_final_i: usize = cycle_length;
//         assert_eq!(sd.i, expected_final_i);
//     }
// }