//! Module containing vehicle struct and related functions.
// crate local
use crate::cycle::{RustCycle, RustCycleCache};
use crate::imports::*;
use crate::params::RustPhysicalProperties;
use crate::proc_macros::add_pyo3_api;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::vehicle::*;
pub mod cyc_mods;
pub mod simdrive_impl;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[add_pyo3_api(
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn __new__(
        favor_grade_accuracy: bool,
        missed_trace_correction: bool, // if true, missed trace correction is active, default = false
        max_time_dilation: f64,
        min_time_dilation: f64,
        time_dilation_tol: f64,
        max_trace_miss_iters: u32,
        trace_miss_speed_mps_tol: f64,
        trace_miss_time_tol: f64,
        trace_miss_dist_tol: f64,
        sim_count_max: usize,
        newton_gain: f64,
        newton_max_iter: u32,
        newton_xtol: f64,
        energy_audit_error_tol: f64,
        coast_allow: bool,
        coast_allow_passing: bool,
        coast_max_speed_m_per_s: f64,
        coast_brake_accel_m_per_s2: f64,
        coast_brake_start_speed_m_per_s: f64,
        coast_start_speed_m_per_s: f64,
        coast_time_horizon_for_adjustment_s: f64,
        idm_allow: bool,
        // IDM - Intelligent Driver Model, Adaptive Cruise Control version
        idm_v_desired_m_per_s: f64,
        idm_dt_headway_s: f64,
        idm_minimum_gap_m: f64,
        idm_delta: f64,
        idm_accel_m_per_s2: f64,
        idm_decel_m_per_s2: f64,
        idm_v_desired_in_m_per_s_by_distance_m: Option<Vec<(f64, f64)>>,
        // Other, Misc.
        max_epa_adj: f64,
    ) -> Self {
        Self {
            favor_grade_accuracy,
            missed_trace_correction, // if true, missed trace correction is active, default = false
            max_time_dilation,
            min_time_dilation,
            time_dilation_tol,
            max_trace_miss_iters,
            trace_miss_speed_mps_tol,
            trace_miss_time_tol,
            trace_miss_dist_tol,
            sim_count_max,
            newton_gain,
            newton_max_iter,
            newton_xtol,
            energy_audit_error_tol,
            coast_allow,
            coast_allow_passing,
            coast_max_speed_m_per_s,
            coast_brake_accel_m_per_s2,
            coast_brake_start_speed_m_per_s,
            coast_start_speed_m_per_s,
            coast_time_horizon_for_adjustment_s,
            idm_allow,
            // IDM - Intelligent Driver Model, Adaptive Cruise Control version
            idm_v_desired_m_per_s,
            idm_dt_headway_s,
            idm_minimum_gap_m,
            idm_delta,
            idm_accel_m_per_s2,
            idm_decel_m_per_s2,
            idm_v_desired_in_m_per_s_by_distance_m,
            // Other, Misc.
            max_epa_adj,
            orphaned: false
        }
    }
    pub fn __getnewargs__(&self) {
        todo!();
    }
)]
/// Struct containing time trace data
pub struct RustSimDriveParams {
    pub favor_grade_accuracy: bool,
    pub missed_trace_correction: bool, // if true, missed trace correction is active, default = false
    pub max_time_dilation: f64,
    pub min_time_dilation: f64,
    pub time_dilation_tol: f64,
    pub max_trace_miss_iters: u32,
    pub trace_miss_speed_mps_tol: f64,
    pub trace_miss_time_tol: f64,
    pub trace_miss_dist_tol: f64,
    pub sim_count_max: usize,
    pub newton_gain: f64,
    pub newton_max_iter: u32,
    pub newton_xtol: f64,
    pub energy_audit_error_tol: f64,
    pub coast_allow: bool,
    pub coast_allow_passing: bool,
    pub coast_max_speed_m_per_s: f64,
    pub coast_brake_accel_m_per_s2: f64,
    pub coast_brake_start_speed_m_per_s: f64,
    pub coast_start_speed_m_per_s: f64,
    pub coast_time_horizon_for_adjustment_s: f64,
    pub idm_allow: bool,
    // IDM - Intelligent Driver Model, Adaptive Cruise Control version
    pub idm_v_desired_m_per_s: f64,
    pub idm_dt_headway_s: f64,
    pub idm_minimum_gap_m: f64,
    pub idm_delta: f64,
    pub idm_accel_m_per_s2: f64,
    pub idm_decel_m_per_s2: f64,
    pub idm_v_desired_in_m_per_s_by_distance_m: Option<Vec<(f64, f64)>>,
    // Other, Misc.
    pub max_epa_adj: f64,
    #[serde(skip)]
    pub orphaned: bool,
}

impl Default for RustSimDriveParams {
    fn default() -> Self {
        // if True, accuracy will be favored over performance for grade per step estimates
        // Specifically, for performance, grade for a step will be assumed to be the grade
        // looked up at step start distance. For accuracy, the actual elevations will be
        // used. This distinciton only makes a difference for CAV maneuvers.
        let favor_grade_accuracy = true;
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
        let newton_gain: f64 = 0.9; // newton solver gain
        let newton_max_iter: u32 = 100; // newton solver max iterations
        let newton_xtol: f64 = 1e-9; // newton solver tolerance
        let energy_audit_error_tol: f64 = 0.002; // tolerance for energy audit error warning, i.e. 0.1%
                                                 // Coasting
        let coast_allow = false;
        let coast_allow_passing = false;
        let coast_max_speed_m_per_s = 40.0;
        let coast_brake_accel_m_per_s2 = -2.5;
        let coast_brake_start_speed_m_per_s = 7.5;
        let coast_start_speed_m_per_s = 0.0; // m/s, if > 0, initiates coast when vehicle hits this speed; mostly for testing
        let coast_time_horizon_for_adjustment_s = 20.0;
        // Following
        let idm_allow = false;
        // IDM - Intelligent Driver Model, Adaptive Cruise Control version
        let idm_v_desired_m_per_s = 33.33;
        let idm_dt_headway_s = 1.0;
        let idm_minimum_gap_m = 2.0;
        let idm_delta = 4.0;
        let idm_accel_m_per_s2 = 1.0;
        let idm_decel_m_per_s2 = 1.5;
        let idm_v_desired_in_m_per_s_by_distance_m = None;
        // EPA fuel economy adjustment parameters
        let max_epa_adj: f64 = 0.3; // maximum EPA adjustment factor
        Self {
            favor_grade_accuracy,
            missed_trace_correction,
            max_time_dilation,
            min_time_dilation,
            time_dilation_tol,
            max_trace_miss_iters,
            trace_miss_speed_mps_tol,
            trace_miss_time_tol,
            trace_miss_dist_tol,
            sim_count_max,
            newton_gain,
            newton_max_iter,
            newton_xtol,
            energy_audit_error_tol,
            coast_allow,
            coast_allow_passing,
            coast_max_speed_m_per_s,
            coast_brake_accel_m_per_s2,
            coast_brake_start_speed_m_per_s,
            coast_start_speed_m_per_s,
            coast_time_horizon_for_adjustment_s,
            idm_allow,
            idm_v_desired_m_per_s,
            idm_dt_headway_s,
            idm_minimum_gap_m,
            idm_delta,
            idm_accel_m_per_s2,
            idm_decel_m_per_s2,
            idm_v_desired_in_m_per_s_by_distance_m,
            max_epa_adj,
            orphaned: false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[add_pyo3_api(
    /// method for instantiating SimDriveRust
    #[new]
    pub fn __new__(cyc: RustCycle, veh: RustVehicle) -> Self {
        Self::new(cyc, veh)
    }

    pub fn __getnewargs__(&self) {
        todo!();
    }

    // wrappers for core methods

    #[pyo3(name = "gap_to_lead_vehicle_m")]
    /// Provides the gap-with lead vehicle from start to finish
    pub fn gap_to_lead_vehicle_m_py(&self) -> PyResult<Vec<f64>> {
        Ok(self.gap_to_lead_vehicle_m().to_vec())
    }

    #[pyo3(name = "sim_drive")]
    /// Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
    /// Arguments
    /// ------------
    /// init_soc: initial SOC for electrified vehicles.
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///     Default of None causes veh.aux_kw to be used.
    pub fn sim_drive_py(
        &mut self,
        init_soc: Option<f64>,
        aux_in_kw_override: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        Ok(self.sim_drive(init_soc, aux_in_kw_override)?)
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
    pub fn sim_drive_walk(
        &mut self,
        init_soc: f64,
        aux_in_kw_override: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        Ok(self.walk(init_soc, aux_in_kw_override)?)
    }

    /// Sets the intelligent driver model parameters for an eco-cruise driving trajectory.
    /// This is a convenience method instead of setting the sim_params.idm* parameters yourself.
    /// - by_microtrip: bool, if True, target speed is set by microtrip, else by cycle
    /// - extend_fraction: float, the fraction of time to extend the cycle to allow for catch-up
    ///     of the following vehicle
    /// - blend_factor: float, a value between 0 and 1; only used of by_microtrip is True, blends
    ///     between microtrip average speed and microtrip average speed when moving. Must be
    ///     between 0 and 1 inclusive
    pub fn activate_eco_cruise(
        &mut self,
        by_microtrip: Option<bool>,
        extend_fraction: Option<f64>,
        blend_factor: Option<f64>,
        min_target_speed_m_per_s: Option<f64>,
    ) -> PyResult<()> {
        let by_microtrip: bool = by_microtrip.unwrap_or(false);
        let extend_fraction: f64 = extend_fraction.unwrap_or(0.1);
        let blend_factor: f64 = blend_factor.unwrap_or(0.0);
        let min_target_speed_m_per_s = min_target_speed_m_per_s.unwrap_or(8.0);
            Ok(self.activate_eco_cruise_rust(
                by_microtrip, extend_fraction, blend_factor, min_target_speed_m_per_s)?)
    }

    #[pyo3(name = "init_for_step")]
    /// This is a specialty method which should be called prior to using
    /// sim_drive_step in a loop.
    /// Arguments
    /// ------------
    /// init_soc: initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///         Default of None causes veh.aux_kw to be used.
    pub fn init_for_step_py(
        &mut self,
        init_soc:f64,
        aux_in_kw_override: Option<Vec<f64>>
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        Ok(self.init_for_step(init_soc, aux_in_kw_override)?)
    }

    /// Step through 1 time step.
    pub fn sim_drive_step(&mut self) -> PyResult<()> {
        Ok(self.step()?)
    }

    #[pyo3(name = "solve_step")]
    /// Perform all the calculations to solve 1 time step.
    pub fn solve_step_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.solve_step(i)?)
    }

    #[pyo3(name = "set_misc_calcs")]
    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    pub fn set_misc_calcs_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_misc_calcs(i)?)
    }

    #[pyo3(name = "set_comp_lims")]
    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_comp_lims_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_comp_lims(i)?)
    }

    #[pyo3(name = "set_power_calcs")]
    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_power_calcs(i)?)
    }

    #[pyo3(name = "set_ach_speed")]
    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_ach_speed(i)?)
    }

    #[pyo3(name = "set_hybrid_cont_calcs")]
    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_hybrid_cont_calcs(i)?)
    }

    #[pyo3(name = "set_fc_forced_state")]
    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    /// `_py` extension is needed to avoid name collision with getter/setter methods
    pub fn set_fc_forced_state_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_fc_forced_state_rust(i)?)
    }

    #[pyo3(name = "set_hybrid_cont_decisions")]
    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_hybrid_cont_decisions(i)?)
    }

    #[pyo3(name = "set_fc_power")]
    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_power_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_fc_power(i)?)
    }

    #[pyo3(name = "set_time_dilation")]
    /// Sets the time dilation for the current step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_time_dilation_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_time_dilation(i)?)
    }

    #[pyo3(name = "set_post_scalars")]
    /// Sets scalar variables that can be calculated after a cycle is run.
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars_py(&mut self) -> PyResult<()> {
        Ok(self.set_post_scalars()?)
    }

    #[pyo3(name = "len")]
    pub fn len_py(&self) -> usize {
        self.len()
    }

    /// added to make clippy happy
    /// not sure whether there is any benefit to this or not for our purposes
    /// Return self.cyc.time_is.is_empty()
    pub fn is_empty(&self) -> bool {
        self.cyc.time_s.is_empty()
    }

    #[getter]
    pub fn get_fs_cumu_mj_out_ach(&self) -> PyResult<Pyo3ArrayF64> {
        Ok(
            Pyo3ArrayF64::new(ndarrcumsum(&(self.fs_kw_out_ach.clone() * self.cyc.dt_s() * 1e-3)))
        )
    }
)]
pub struct RustSimDrive {
    pub hev_sim_count: usize,
    #[api(has_orphaned)]
    pub veh: RustVehicle,
    #[api(has_orphaned)]
    pub cyc: RustCycle,
    #[api(has_orphaned)]
    pub cyc0: RustCycle,
    #[api(has_orphaned)]
    pub sim_params: RustSimDriveParams,
    #[serde(skip)]
    #[api(has_orphaned)]
    pub props: RustPhysicalProperties,
    pub i: usize, // 1 # initialize step counter for possible use outside sim_drive_walk()
    pub cur_max_fs_kw_out: Array1<f64>,
    pub fc_trans_lim_kw: Array1<f64>,
    pub fc_fs_lim_kw: Array1<f64>,
    pub fc_max_kw_in: Array1<f64>,
    pub cur_max_fc_kw_out: Array1<f64>,
    pub ess_cap_lim_dischg_kw: Array1<f64>,
    pub cur_ess_max_kw_out: Array1<f64>,
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
    pub cyc_trac_kw_req: Array1<f64>,
    pub cur_max_trac_kw: Array1<f64>,
    pub spare_trac_kw: Array1<f64>,
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
    pub impose_coast: Array1<bool>,
    pub roadway_chg_kw_out_ach: Array1<f64>,
    pub min_ess_kw_2help_fc: Array1<f64>,
    pub ess_kw_out_ach: Array1<f64>,
    pub fc_kw_out_ach: Array1<f64>,
    pub fc_kw_out_ach_pct: Array1<f64>,
    pub fc_kw_in_ach: Array1<f64>,
    pub fs_kw_out_ach: Array1<f64>,
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
    pub trace_miss_iters: Array1<u32>,
    pub newton_iters: Array1<u32>,
    pub fuel_kj: f64,
    pub ess_dischg_kj: f64,
    pub energy_audit_error: f64,
    pub mpgge: f64,
    pub roadway_chg_kj: f64,
    pub battery_kwh_per_mi: f64,
    pub electric_kwh_per_mi: f64,
    pub ess2fuel_kwh: f64,
    pub drag_kj: f64,
    pub ascent_kj: f64,
    pub rr_kj: f64,
    pub brake_kj: f64,
    pub trans_kj: f64,
    pub mc_kj: f64,
    pub ess_eff_kj: f64,
    pub aux_kj: f64,
    pub fc_kj: f64,
    pub net_kj: f64,
    pub ke_kj: f64,
    pub trace_miss: bool,
    pub trace_miss_dist_frac: f64,
    pub trace_miss_time_frac: f64,
    pub trace_miss_speed_mps: f64,
    #[serde(skip)]
    pub orphaned: bool,
    pub coast_delay_index: Array1<i32>,
    pub idm_target_speed_m_per_s: Array1<f64>,
    #[serde(skip)]
    pub cyc0_cache: RustCycleCache,
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
