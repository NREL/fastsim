//! Module containing implementations for [simdrive](super::simdrive).

use super::cycle::{
    accel_array_for_constant_jerk, accel_for_constant_jerk, calc_constant_jerk_trajectory,
    detect_passing, trapz_distance_for_step, trapz_step_distances, trapz_step_start_distance,
    PassingInfo, RustCycle,
};
use super::params;
use super::utils::{
    add_from, arrmax, first_grtr, max, min, ndarrcumsum, ndarrmax, ndarrmin, ndarrunique,
};
use super::vehicle::*;
use ndarray::{array, s, Array, Array1};
use std::cmp;

use super::simdrive::{RustSimDrive, RustSimDriveParams};

use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

pub const SIMDRIVE_DEFAULT_FOLDER: &str = "fastsim/resources";

struct RendezvousTrajectory {
    pub found_trajectory: bool,
    pub idx: usize,
    pub n: usize,
    pub full_brake_steps: usize,
    pub jerk_m_per_s3: f64,
    pub accel0_m_per_s2: f64,
    pub accel_spread: f64,
}

struct CoastTrajectory {
    pub found_trajectory: bool,
    pub distance_to_stop_via_coast_m: f64,
    pub start_idx: usize,
    pub speeds_m_per_s: Option<Vec<f64>>,
    pub distance_to_brake_m: Option<f64>,
}

impl RustSimDrive {
    pub fn new(cyc: RustCycle, veh: RustVehicle) -> Self {
        let hev_sim_count: usize = 0;
        let cyc0: RustCycle = cyc.clone();
        let sim_params = RustSimDriveParams::default();
        let props = params::RustPhysicalProperties::default();
        let i: usize = 1; // 1 # initialize step counter for possible use outside sim_drive_walk()
        let cyc_len = cyc.time_s.len(); //get_len() as usize;
        let cur_max_fs_kw_out = Array::zeros(cyc_len);
        let fc_trans_lim_kw = Array::zeros(cyc_len);
        let fc_fs_lim_kw = Array::zeros(cyc_len);
        let fc_max_kw_in = Array::zeros(cyc_len);
        let cur_max_fc_kw_out = Array::zeros(cyc_len);
        let ess_cap_lim_dischg_kw = Array::zeros(cyc_len);
        let cur_ess_max_kw_out = Array::zeros(cyc_len);
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
        let cyc_trac_kw_req = Array::zeros(cyc_len);
        let cur_max_trac_kw = Array::zeros(cyc_len);
        let spare_trac_kw = Array::zeros(cyc_len);
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
        let impose_coast = Array::from_vec(vec![false; cyc_len]);
        let roadway_chg_kw_out_ach = Array::zeros(cyc_len);
        let min_ess_kw_2help_fc = Array::zeros(cyc_len);
        let ess_kw_out_ach = Array::zeros(cyc_len);
        let fc_kw_out_ach = Array::zeros(cyc_len);
        let fc_kw_out_ach_pct = Array::zeros(cyc_len);
        let fc_kw_in_ach = Array::zeros(cyc_len);
        let fs_kw_out_ach = Array::zeros(cyc_len);
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
        let electric_kwh_per_mi: f64 = 0.0;
        let ess2fuel_kwh: f64 = 0.0;
        let drag_kj: f64 = 0.0;
        let ascent_kj: f64 = 0.0;
        let rr_kj: f64 = 0.0;
        let brake_kj: f64 = 0.0;
        let trans_kj: f64 = 0.0;
        let mc_kj: f64 = 0.0;
        let ess_eff_kj: f64 = 0.0;
        let aux_kj: f64 = 0.0;
        let fc_kj: f64 = 0.0;
        let net_kj: f64 = 0.0;
        let ke_kj: f64 = 0.0;
        let trace_miss = false;
        let trace_miss_dist_frac: f64 = 0.0;
        let trace_miss_time_frac: f64 = 0.0;
        let trace_miss_speed_mps: f64 = 0.0;
        let coast_delay_index = Array::zeros(cyc_len);
        RustSimDrive {
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
            cur_ess_max_kw_out,
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
            cyc_trac_kw_req,
            cur_max_trac_kw,
            spare_trac_kw,
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
            impose_coast,
            roadway_chg_kw_out_ach,
            min_ess_kw_2help_fc,
            ess_kw_out_ach,
            fc_kw_out_ach,
            fc_kw_out_ach_pct,
            fc_kw_in_ach,
            fs_kw_out_ach,
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
            electric_kwh_per_mi,
            ess2fuel_kwh,
            drag_kj,
            ascent_kj,
            rr_kj,
            brake_kj,
            trans_kj,
            mc_kj,
            ess_eff_kj,
            aux_kj,
            fc_kj,
            net_kj,
            ke_kj,
            trace_miss,
            trace_miss_dist_frac,
            trace_miss_time_frac,
            trace_miss_speed_mps,
            orphaned: false,
            coast_delay_index,
        }
    }

    impl_serde!(RustSimDrive, SIMDRIVE_DEFAULT_FOLDER);
    impl_from_file!();

    /// Return length of time arrays
    pub fn len(&self) -> usize {
        self.cyc.time_s.len()
    }

    // TODO: probably shouldn't be public...?
    pub fn init_arrays(&mut self) {
        self.i = 1; // initialize step counter for possible use outside sim_drive_walk()
        let cyc_len = self.cyc0.time_s.len(); //get_len() as usize;

        // Component Limits -- calculated dynamically
        self.cur_max_fs_kw_out = Array::zeros(cyc_len);
        self.fc_trans_lim_kw = Array::zeros(cyc_len);
        self.fc_fs_lim_kw = Array::zeros(cyc_len);
        self.fc_max_kw_in = Array::zeros(cyc_len);
        self.cur_max_fc_kw_out = Array::zeros(cyc_len);
        self.ess_cap_lim_dischg_kw = Array::zeros(cyc_len);
        self.cur_ess_max_kw_out = Array::zeros(cyc_len);
        self.cur_max_avail_elec_kw = Array::zeros(cyc_len);
        self.ess_cap_lim_chg_kw = Array::zeros(cyc_len);
        self.cur_max_ess_chg_kw = Array::zeros(cyc_len);
        self.cur_max_elec_kw = Array::zeros(cyc_len);
        self.mc_elec_in_lim_kw = Array::zeros(cyc_len);
        self.mc_transi_lim_kw = Array::zeros(cyc_len);
        self.cur_max_mc_kw_out = Array::zeros(cyc_len);
        self.ess_lim_mc_regen_perc_kw = Array::zeros(cyc_len);
        self.ess_lim_mc_regen_kw = Array::zeros(cyc_len);
        self.cur_max_mech_mc_kw_in = Array::zeros(cyc_len);
        self.cur_max_trans_kw_out = Array::zeros(cyc_len);

        // Drive Train
        self.cyc_trac_kw_req = Array::zeros(cyc_len);
        self.cur_max_trac_kw = Array::zeros(cyc_len);
        self.spare_trac_kw = Array::zeros(cyc_len);
        self.cyc_whl_rad_per_sec = Array::zeros(cyc_len);
        self.cyc_tire_inertia_kw = Array::zeros(cyc_len);
        self.cyc_whl_kw_req = Array::zeros(cyc_len);
        self.regen_contrl_lim_kw_perc = Array::zeros(cyc_len);
        self.cyc_regen_brake_kw = Array::zeros(cyc_len);
        self.cyc_fric_brake_kw = Array::zeros(cyc_len);
        self.cyc_trans_kw_out_req = Array::zeros(cyc_len);
        self.cyc_met = Array::from_vec(vec![false; cyc_len]);
        self.trans_kw_out_ach = Array::zeros(cyc_len);
        self.trans_kw_in_ach = Array::zeros(cyc_len);
        self.cur_soc_target = Array::zeros(cyc_len);
        self.min_mc_kw_2help_fc = Array::zeros(cyc_len);
        self.mc_mech_kw_out_ach = Array::zeros(cyc_len);
        self.mc_elec_kw_in_ach = Array::zeros(cyc_len);
        self.aux_in_kw = Array::zeros(cyc_len);
        self.roadway_chg_kw_out_ach = Array::zeros(cyc_len);
        self.min_ess_kw_2help_fc = Array::zeros(cyc_len);
        self.ess_kw_out_ach = Array::zeros(cyc_len);
        self.fc_kw_out_ach = Array::zeros(cyc_len);
        self.fc_kw_out_ach_pct = Array::zeros(cyc_len);
        self.fc_kw_in_ach = Array::zeros(cyc_len);
        self.fs_kw_out_ach = Array::zeros(cyc_len);
        self.fs_kwh_out_ach = Array::zeros(cyc_len);
        self.ess_cur_kwh = Array::zeros(cyc_len);
        self.soc = Array::zeros(cyc_len);

        // Vehicle Attributes, Control Variables
        self.regen_buff_soc = Array::zeros(cyc_len);
        self.ess_regen_buff_dischg_kw = Array::zeros(cyc_len);
        self.max_ess_regen_buff_chg_kw = Array::zeros(cyc_len);
        self.ess_accel_buff_chg_kw = Array::zeros(cyc_len);
        self.accel_buff_soc = Array::zeros(cyc_len);
        self.max_ess_accell_buff_dischg_kw = Array::zeros(cyc_len);
        self.ess_accel_regen_dischg_kw = Array::zeros(cyc_len);
        self.mc_elec_in_kw_for_max_fc_eff = Array::zeros(cyc_len);
        self.elec_kw_req_4ae = Array::zeros(cyc_len);
        self.can_pwr_all_elec = Array::from_vec(vec![false; cyc_len]);
        self.desired_ess_kw_out_for_ae = Array::zeros(cyc_len);
        self.ess_ae_kw_out = Array::zeros(cyc_len);
        self.er_ae_kw_out = Array::zeros(cyc_len);
        self.ess_desired_kw_4fc_eff = Array::zeros(cyc_len);
        self.ess_kw_if_fc_req = Array::zeros(cyc_len);
        self.cur_max_mc_elec_kw_in = Array::zeros(cyc_len);
        self.fc_kw_gap_fr_eff = Array::zeros(cyc_len);
        self.er_kw_if_fc_req = Array::zeros(cyc_len);
        self.mc_elec_kw_in_if_fc_req = Array::zeros(cyc_len);
        self.mc_kw_if_fc_req = Array::zeros(cyc_len);
        self.fc_forced_on = Array::from_vec(vec![false; cyc_len]);
        self.fc_forced_state = Array::zeros(cyc_len);
        self.mc_mech_kw_4forced_fc = Array::zeros(cyc_len);
        self.fc_time_on = Array::zeros(cyc_len);
        self.prev_fc_time_on = Array::zeros(cyc_len);

        // Additional Variables
        self.mps_ach = Array::zeros(cyc_len);
        self.mph_ach = Array::zeros(cyc_len);
        self.dist_m = Array::zeros(cyc_len);
        self.dist_mi = Array::zeros(cyc_len);
        self.high_acc_fc_on_tag = Array::from_vec(vec![false; cyc_len]);
        self.reached_buff = Array::from_vec(vec![false; cyc_len]);
        self.max_trac_mps = Array::zeros(cyc_len);
        self.add_kwh = Array::zeros(cyc_len);
        self.dod_cycs = Array::zeros(cyc_len);
        self.ess_perc_dead = Array::zeros(cyc_len);
        self.drag_kw = Array::zeros(cyc_len);
        self.ess_loss_kw = Array::zeros(cyc_len);
        self.accel_kw = Array::zeros(cyc_len);
        self.ascent_kw = Array::zeros(cyc_len);
        self.rr_kw = Array::zeros(cyc_len);
        self.cur_max_roadway_chg_kw = Array::zeros(cyc_len);
        self.trace_miss_iters = Array::zeros(cyc_len);
        self.newton_iters = Array::zeros(cyc_len);
        self.coast_delay_index = Array::zeros(cyc_len);
        self.impose_coast = Array::from_vec(vec![false; cyc_len]);
    }

    /// Provides the gap-with lead vehicle from start to finish
    pub fn gap_to_lead_vehicle_m(&self) -> Array1<f64> {
        // TODO: consider basing on dist_m?
        let mut gaps_m = ndarrcumsum(&trapz_step_distances(&self.cyc0))
            - ndarrcumsum(&trapz_step_distances(&self.cyc));
        if self.sim_params.follow_allow {
            gaps_m += self.sim_params.idm_minimum_gap_m;
        }
        gaps_m
    }

    /// Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
    /// Arguments
    /// ------------
    /// init_soc: initial SOC for electrified vehicles.  
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
    ///     Default of None causes veh.aux_kw to be used.
    pub fn sim_drive(
        &mut self,
        init_soc: Option<f64>,
        aux_in_kw_override: Option<Array1<f64>>,
    ) -> Result<(), String> {
        self.hev_sim_count = 0;

        let init_soc = match init_soc {
            Some(x) => x,
            None => {
                if self.veh.veh_pt_type == CONV {
                    // If no EV / Hybrid components, no SOC considerations.
                    (self.veh.max_soc + self.veh.min_soc) / 2.0
                } else if self.veh.veh_pt_type == HEV {
                    // ####################################
                    // ### Charge Balancing Vehicle SOC ###
                    // ####################################
                    // Charge balancing SOC for HEV vehicle types. Iterating init_soc and comparing to final SOC.
                    // Iterating until tolerance met or 30 attempts made.
                    let mut init_soc = (self.veh.max_soc + self.veh.min_soc) / 2.0;
                    let mut ess_2fuel_kwh = 1.0;
                    while ess_2fuel_kwh > self.veh.ess_to_fuel_ok_error
                        && self.hev_sim_count < self.sim_params.sim_count_max
                    {
                        self.hev_sim_count += 1;
                        self.walk(init_soc, aux_in_kw_override.clone())?;
                        let fuel_kj = (&self.fs_kw_out_ach * self.cyc.dt_s()).sum();
                        let roadway_chg_kj = (&self.roadway_chg_kw_out_ach * self.cyc.dt_s()).sum();
                        if (fuel_kj + roadway_chg_kj) > 0.0 {
                            ess_2fuel_kwh = ((self.soc[0] - self.soc[self.len() - 1])
                                * self.veh.ess_max_kwh
                                * 3.6e3
                                / (fuel_kj + roadway_chg_kj))
                                .abs();
                        } else {
                            ess_2fuel_kwh = 0.0;
                        }
                        init_soc = min(1.0, max(0.0, self.soc[self.len() - 1]));
                    }
                    init_soc
                } else if self.veh.veh_pt_type == PHEV || self.veh.veh_pt_type == BEV {
                    // If EV, initializing initial SOC to maximum SOC.
                    self.veh.max_soc
                } else {
                    panic!("Failed to properly initialize SOC.");
                }
            }
        };

        self.walk(init_soc, aux_in_kw_override)?;

        self.set_post_scalars()?;
        Ok(())
    }

    /// Receives second-by-second cycle information, vehicle properties,
    /// and an initial state of charge and runs sim_drive_step to perform a
    /// backward facing powertrain simulation. Method `sim_drive` runs this
    /// iteratively to achieve correct SOC initial and final conditions, as
    /// needed.
    ///
    /// Arguments
    /// ------------
    /// init_soc: initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: (Optional) aux_in_kw override.  Array of same length as cyc.time_s.
    ///         None causes veh.aux_kw to be used.
    pub fn walk(
        &mut self,
        init_soc: f64,
        aux_in_kw_override: Option<Array1<f64>>,
    ) -> Result<(), String> {
        self.init_for_step(init_soc, aux_in_kw_override)?;
        while self.i < self.cyc.time_s.len() {
            self.step()?;
        }

        // TODO: uncomment and implement
        //    if (self.cyc.dt_s > 5).any() and self.sim_params.verbose:
        //         if self.sim_params.missed_trace_correction:
        //             print('Max time dilation factor =', (round((self.cyc.dt_s / self.cyc0.dt_s).max(), 3)))
        //         print("Warning: large time steps affect accuracy significantly.")
        //         print("To suppress this message, view the doc string for simdrive.SimDriveParams.")
        //         print('Max time step =', (round(self.cyc.dt_s.max(), 3)))
        Ok(())
    }

    /// This is a specialty method which should be called prior to using
    /// sim_drive_step in a loop.
    /// Arguments
    /// ------------
    /// init_soc: initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
    ///         Default of None causes veh.aux_kw to be used.
    pub fn init_for_step(
        &mut self,
        init_soc: f64,
        aux_in_kw_override: Option<Array1<f64>>,
    ) -> Result<(), String> {
        let init_soc = if !(self.veh.min_soc..=self.veh.max_soc).contains(&init_soc) {
            println!("WARNING! Provided init_soc is outside range [min_soc, max_soc]: [{}, {}]. Setting init_soc to max_soc.",
                self.veh.min_soc, self.veh.max_soc);
            self.veh.max_soc
        } else {
            init_soc
        };
        self.init_arrays();

        if let Some(arr) = aux_in_kw_override {
            self.aux_in_kw = arr;
        }

        self.cyc_met[0] = true;
        self.cur_soc_target[0] = self.veh.max_soc;
        self.ess_cur_kwh[0] = init_soc * self.veh.ess_max_kwh;
        self.soc[0] = init_soc;
        self.mps_ach[0] = self.cyc0.mps[0];
        self.mph_ach[0] = self.cyc0.mph_at_i(0);

        if self.sim_params.missed_trace_correction
            || self.sim_params.follow_allow
            || self.sim_params.coast_allow
        {
            self.cyc = self.cyc0.clone(); // reset the cycle in case it has been manipulated
        }
        self.i = 1; // time step counter
        Ok(())
    }

    /// Calculate the next speed by the Intelligent Driver Model
    /// - i: int, the index
    /// - a_m_per_s2: number, max acceleration (m/s2)
    /// - b_m_per_s2: number, max deceleration (m/s2)
    /// - dt_headway_s: number, the headway between us and the lead vehicle in seconds
    /// - s0_m: number, the initial gap between us and the lead vehicle in meters
    /// - v_desired_m_per_s: number, the desired speed in (m/s)
    /// - delta: number, a shape parameter; typical value is 4.0
    /// RETURN: number, the next speed (m/s)
    /// REFERENCE:
    /// Treiber, Martin and Kesting, Arne. 2013. "Chapter 11: Car-Following Models Based on Driving Strategies".
    ///     Traffic Flow Dynamics: Data, Models and Simulation. Springer-Verlag. Springer, Berlin, Heidelberg.
    ///     DOI: https://doi.org/10.1007/978-3-642-32460-4.
    #[allow(clippy::too_many_arguments)]
    pub fn next_speed_by_idm(
        &mut self,
        i: usize,
        a_m_per_s2: f64,
        b_m_per_s2: f64,
        dt_headway_s: f64,
        s0_m: f64,
        v_desired_m_per_s: f64,
        delta: f64,
    ) -> f64 {
        if v_desired_m_per_s <= 0.0 {
            return 0.0;
        }
        let a_m_per_s2 = a_m_per_s2.abs();
        let b_m_per_s2 = b_m_per_s2.abs();
        let dt_headway_s = max(dt_headway_s, 0.0);
        // we assume the vehicles start out a "minimum gap" apart
        let s0_m = max(0.0, s0_m);
        // DERIVED VALUES
        let sqrt_ab = (a_m_per_s2 * b_m_per_s2).powf(0.5);
        let v0_m_per_s = self.mps_ach[i - 1];
        let v0_lead_m_per_s = self.cyc0.mps[i - 1];
        let dv0_m_per_s = v0_m_per_s - v0_lead_m_per_s;
        let d0_lead_m: f64 = trapz_step_start_distance(&self.cyc0, i) + s0_m;
        let d0_m = trapz_step_start_distance(&self.cyc, i);
        let s_m = max(d0_lead_m - d0_m, 0.01);
        // IDM EQUATIONS
        let s_target_m = s0_m
            + max(
                0.0,
                (v0_m_per_s * dt_headway_s) + ((v0_m_per_s * dv0_m_per_s) / (2.0 * sqrt_ab)),
            );
        let accel_target_m_per_s2 = a_m_per_s2
            * (1.0 - ((v0_m_per_s / v_desired_m_per_s).powf(delta)) - ((s_target_m / s_m).powi(2)));
        max(
            v0_m_per_s + (accel_target_m_per_s2 * self.cyc.dt_s_at_i(i)),
            0.0,
        )
    }

    /// Set gap
    /// - i: non-negative integer, the step index
    /// RETURN: None
    /// EFFECTS:
    /// - sets the next speed (m/s)
    /// EQUATION:
    /// parameters:
    ///     - v_desired: the desired speed (m/s)
    ///     - delta: number, typical value is 4.0
    ///     - a: max acceleration, (m/s2)
    ///     - b: max deceleration, (m/s2)
    /// s = d_lead - d
    /// dv/dt = a * (1 - (v/v_desired)**delta - (s_desired(v,v-v_lead)/s)**2)
    /// s_desired(v, dv) = s0 + max(0, v*dt_headway + (v * dv)/(2.0 * sqrt(a*b)))
    /// REFERENCE:
    /// Treiber, Martin and Kesting, Arne. 2013. "Chapter 11: Car-Following Models Based on Driving Strategies".
    ///     Traffic Flow Dynamics: Data, Models and Simulation. Springer-Verlag. Springer, Berlin, Heidelberg.
    ///     DOI: <https://doi.org/10.1007/978-3-642-32460-4>
    pub fn set_speed_for_target_gap_using_idm(&mut self, i: usize) {
        // PARAMETERS
        let v_desired_m_per_s = if self.sim_params.idm_v_desired_m_per_s > 0.0 {
            self.sim_params.idm_v_desired_m_per_s
        } else {
            ndarrmax(&self.cyc0.mps)
        };
        // DERIVED VALUES
        self.cyc.mps[i] = self.next_speed_by_idm(
            i,
            self.sim_params.idm_accel_m_per_s2,
            self.sim_params.idm_decel_m_per_s2,
            self.sim_params.idm_dt_headway_s,
            self.sim_params.idm_minimum_gap_m,
            v_desired_m_per_s,
            self.sim_params.idm_delta,
        );
    }

    /// - i: non-negative integer, the step index
    /// RETURN: None
    /// EFFECTS:
    /// - sets the next speed (m/s)
    pub fn set_speed_for_target_gap(&mut self, i: usize) {
        self.set_speed_for_target_gap_using_idm(i);
    }

    /// Provides a quick estimate for grade based only on the distance traveled
    /// at the start of the current step. If the grade is constant over the
    /// step, this is both quick and accurate.
    /// NOTE:
    ///     If not allowing coasting (i.e., sim_params.coast_allow == False)
    ///     and not allowing IDM/following (i.e., self.sim_params.follow_allow == False)
    ///     then returns self.cyc.grade[i]
    pub fn estimate_grade_for_step(&self, i: usize) -> f64 {
        if !self.sim_params.coast_allow && !self.sim_params.follow_allow {
            return self.cyc.grade[i];
        }
        self.cyc0
            .average_grade_over_range(trapz_step_start_distance(&self.cyc, i), 0.0)
    }

    /// For situations where cyc can deviate from cyc0, this method
    /// looks up and accurately interpolates what the average grade over
    /// the step should be.
    /// If mps_ach is not None, the mps_ach value is used to predict the
    /// distance traveled over the step.
    /// NOTE:
    ///     If not allowing coasting (i.e., sim_params.coast_allow == False)
    ///     and not allowing IDM/following (i.e., self.sim_params.follow_allow == False)
    ///     then returns self.cyc.grade[i]
    pub fn lookup_grade_for_step(&self, i: usize, mps_ach: Option<f64>) -> f64 {
        if !self.sim_params.coast_allow && !self.sim_params.follow_allow {
            return self.cyc.grade[i];
        }
        match mps_ach {
            Some(mps_ach) => self.cyc0.average_grade_over_range(
                trapz_step_start_distance(&self.cyc, i),
                0.5 * (mps_ach + self.mps_ach[i - 1]) * self.cyc.dt_s_at_i(i),
            ),
            None => self.cyc0.average_grade_over_range(
                trapz_step_start_distance(&self.cyc, i),
                trapz_distance_for_step(&self.cyc, i),
            ),
        }
    }

    /// Step through 1 time step.
    pub fn step(&mut self) -> Result<(), String> {
        if self.sim_params.follow_allow {
            self.set_speed_for_target_gap(self.i);
        }
        if self.sim_params.coast_allow {
            self.set_coast_speed(self.i)?;
        }
        self.solve_step(self.i)?;

        if self.sim_params.missed_trace_correction
            && (self.cyc0.dist_m().slice(s![0..self.i]).sum() > 0.0)
        {
            self.set_time_dilation(self.i)?;
        }
        // TODO: shouldn't the below code always set cyc? Whether coasting or not?
        if self.sim_params.coast_allow || self.sim_params.follow_allow {
            self.cyc.mps[self.i] = self.mps_ach[self.i];
            self.cyc.grade[self.i] = self.lookup_grade_for_step(self.i, None);
        }

        self.i += 1; // increment time step counter
        Ok(())
    }

    /// Perform all the calculations to solve 1 time step.
    pub fn solve_step(&mut self, i: usize) -> Result<(), String> {
        self.set_misc_calcs(i)?;
        self.set_comp_lims(i)?;
        self.set_power_calcs(i)?;
        self.set_ach_speed(i)?;
        self.set_hybrid_cont_calcs(i)?;
        self.set_fc_forced_state_rust(i)?;
        self.set_hybrid_cont_decisions(i)?;
        self.set_fc_power(i)?;
        Ok(())
    }

    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    pub fn set_misc_calcs(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            // if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
            // TODO: this is probably computationally expensive and was probably a workaround for numba
            // figure out a way to not need this
            if self.aux_in_kw.slice(s![i..]).iter().all(|&x| x == 0.0) {
                // if all elements after i-1 are zero, trigger default behavior; otherwise, use override value
                if self.veh.no_elec_aux {
                    self.aux_in_kw[i] = self.veh.aux_kw / self.veh.alt_eff;
                } else {
                    self.aux_in_kw[i] = self.veh.aux_kw;
                }
            }
            // Is SOC below min threshold?
            if self.soc[i - 1] < (self.veh.min_soc + self.veh.perc_high_acc_buf) {
                self.reached_buff[i] = false;
            } else {
                self.reached_buff[i] = true;
            }

            // Does the engine need to be on for low SOC or high acceleration
            if self.soc[i - 1] < self.veh.min_soc
                || (self.high_acc_fc_on_tag[i - 1] && !(self.reached_buff[i]))
            {
                self.high_acc_fc_on_tag[i] = true
            } else {
                self.high_acc_fc_on_tag[i] = false
            }
            self.max_trac_mps[i] =
                self.mps_ach[i - 1] + (self.veh.max_trac_mps2 * self.cyc.dt_s_at_i(i));
            Ok(())
        };

        if let Err(()) = res() {
            Err(format!("`set_misc_calcs_rust` failed at time step {}", i))
        } else {
            Ok(())
        }
    }

    /// Sets component limits for time step 'i'
    /// Arguments
    /// ------------
    /// i: index of time step
    /// initSoc: initial SOC for electrified vehicles
    pub fn set_comp_lims(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            // max fuel storage power output
            self.cur_max_fs_kw_out[i] = min(
                self.veh.fs_max_kw,
                self.fs_kw_out_ach[i - 1]
                    + self.veh.fs_max_kw / self.veh.fs_secs_to_peak_pwr * self.cyc.dt_s_at_i(i),
            );
            // maximum fuel storage power output rate of change
            self.fc_trans_lim_kw[i] = self.fc_kw_out_ach[i - 1]
                + (self.veh.fc_max_kw / self.veh.fc_sec_to_peak_pwr * self.cyc.dt_s_at_i(i));

            self.fc_max_kw_in[i] = min(self.cur_max_fs_kw_out[i], self.veh.fs_max_kw);
            self.fc_fs_lim_kw[i] = self.veh.fc_max_kw;
            self.cur_max_fc_kw_out[i] = min(
                self.veh.fc_max_kw,
                min(self.fc_fs_lim_kw[i], self.fc_trans_lim_kw[i]),
            );

            if self.veh.ess_max_kwh == 0.0 || self.soc[i - 1] < self.veh.min_soc {
                self.ess_cap_lim_dischg_kw[i] = 0.0;
            } else {
                self.ess_cap_lim_dischg_kw[i] = self.veh.ess_max_kwh
                    * self.veh.ess_round_trip_eff.sqrt()
                    * 3.6e3
                    * (self.soc[i - 1] - self.veh.min_soc)
                    / self.cyc.dt_s_at_i(i);
            }
            self.cur_ess_max_kw_out[i] = min(self.veh.ess_max_kw, self.ess_cap_lim_dischg_kw[i]);

            if self.veh.ess_max_kwh == 0.0 || self.veh.ess_max_kw == 0.0 {
                self.ess_cap_lim_chg_kw[i] = 0.0;
            } else {
                self.ess_cap_lim_chg_kw[i] = max(
                    (self.veh.max_soc - self.soc[i - 1]) * self.veh.ess_max_kwh
                        / self.veh.ess_round_trip_eff.sqrt()
                        / (self.cyc.dt_s_at_i(i) / 3.6e3),
                    0.0,
                );
            }

            self.cur_max_ess_chg_kw[i] = min(self.ess_cap_lim_chg_kw[i], self.veh.ess_max_kw);

            // Current maximum electrical power that can go toward propulsion, not including motor limitations
            if self.veh.fc_eff_type == H2FC {
                self.cur_max_elec_kw[i] = self.cur_max_fc_kw_out[i]
                    + self.cur_max_roadway_chg_kw[i]
                    + self.cur_ess_max_kw_out[i]
                    - self.aux_in_kw[i];
            } else {
                self.cur_max_elec_kw[i] =
                    self.cur_max_roadway_chg_kw[i] + self.cur_ess_max_kw_out[i] - self.aux_in_kw[i];
            }

            // Current maximum electrical power that can go toward propulsion, including motor limitations
            self.cur_max_avail_elec_kw[i] =
                min(self.cur_max_elec_kw[i], self.veh.mc_max_elec_in_kw);

            if self.cur_max_elec_kw[i] > 0.0 {
                // limit power going into e-machine controller to
                if self.cur_max_avail_elec_kw[i] == arrmax(&self.veh.mc_kw_in_array) {
                    self.mc_elec_in_lim_kw[i] = min(
                        self.veh.mc_kw_out_array[self.veh.mc_kw_out_array.len() - 1],
                        self.veh.mc_max_kw,
                    );
                } else {
                    self.mc_elec_in_lim_kw[i] = min(
                        self.veh.mc_kw_out_array[first_grtr(
                            &self.veh.mc_kw_in_array,
                            min(
                                arrmax(&self.veh.mc_kw_in_array) - 0.01,
                                self.cur_max_avail_elec_kw[i],
                            ),
                        )
                        .unwrap_or(0)
                            - 1_usize],
                        self.veh.mc_max_kw,
                    )
                }
            } else {
                self.mc_elec_in_lim_kw[i] = 0.0;
            }

            // Motor transient power limit
            self.mc_transi_lim_kw[i] = self.mc_mech_kw_out_ach[i - 1].abs()
                + self.veh.mc_max_kw / self.veh.mc_sec_to_peak_pwr * self.cyc.dt_s_at_i(i);

            self.cur_max_mc_kw_out[i] = max(
                min(
                    min(self.mc_elec_in_lim_kw[i], self.mc_transi_lim_kw[i]),
                    if self.veh.stop_start { 0.0 } else { 1.0 } * self.veh.mc_max_kw,
                ),
                -self.veh.mc_max_kw,
            );

            if self.cur_max_mc_kw_out[i] == 0.0 {
                self.cur_max_mc_elec_kw_in[i] = 0.0;
            } else if self.cur_max_mc_kw_out[i] == self.veh.mc_max_kw {
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i]
                    / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1];
            } else {
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i]
                    / self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_out_array,
                            min(self.veh.mc_max_kw - 0.01, self.cur_max_mc_kw_out[i]),
                        )
                        .unwrap_or(0)
                            - 1,
                    )]
            }

            if self.veh.mc_max_kw == 0.0 {
                self.ess_lim_mc_regen_perc_kw[i] = 0.0;
            } else {
                self.ess_lim_mc_regen_perc_kw[i] = min(
                    (self.cur_max_ess_chg_kw[i] + self.aux_in_kw[i]) / self.veh.mc_max_kw,
                    1.0,
                );
            }
            if self.cur_max_ess_chg_kw[i] == 0.0 {
                self.ess_lim_mc_regen_kw[i] = 0.0;
            } else if self.veh.mc_max_kw
                == self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]
            {
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.mc_max_kw,
                    self.cur_max_ess_chg_kw[i]
                        / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1],
                );
            } else {
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.mc_max_kw,
                    self.cur_max_ess_chg_kw[i]
                        / self.veh.mc_full_eff_array[cmp::max(
                            1,
                            first_grtr(
                                &self.veh.mc_kw_out_array,
                                min(
                                    self.veh.mc_max_kw - 0.01,
                                    self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i],
                                ),
                            )
                            .unwrap_or(0)
                                - 1,
                        )],
                );
            }
            self.cur_max_mech_mc_kw_in[i] = min(self.ess_lim_mc_regen_kw[i], self.veh.mc_max_kw);

            self.cur_max_trac_kw[i] = self.veh.wheel_coef_of_fric
                * self.veh.drive_axle_weight_frac
                * self.veh.veh_kg
                * self.props.a_grav_mps2
                / (1.0 + self.veh.veh_cg_m * self.veh.wheel_coef_of_fric / self.veh.wheel_base_m)
                / 1e3
                * self.max_trac_mps[i];

            if self.veh.fc_eff_type == H2FC {
                if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
                    self.cur_max_trans_kw_out[i] = min(
                        (self.cur_max_mc_kw_out[i] - self.aux_in_kw[i]) * self.veh.trans_eff,
                        self.cur_max_trac_kw[i] / self.veh.trans_eff,
                    );
                } else {
                    self.cur_max_trans_kw_out[i] = min(
                        (self.cur_max_mc_kw_out[i] - min(self.cur_max_elec_kw[i], 0.0))
                            * self.veh.trans_eff,
                        self.cur_max_trac_kw[i] / self.veh.trans_eff,
                    );
                }
            } else if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i] - self.aux_in_kw[i])
                        * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff,
                );
            } else {
                self.cur_max_trans_kw_out[i] = min(
                    (self.cur_max_mc_kw_out[i] + self.cur_max_fc_kw_out[i]
                        - min(self.cur_max_elec_kw[i], 0.0))
                        * self.veh.trans_eff,
                    self.cur_max_trac_kw[i] / self.veh.trans_eff,
                );
            }
            if self.impose_coast[i] {
                self.cur_max_trans_kw_out[i] = 0.0;
            }
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_comp_lims_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            let mps_ach = if self.newton_iters[i] > 0u32 {
                self.mps_ach[i]
            } else {
                self.cyc.mps[i]
            };

            let grade = self.lookup_grade_for_step(i, Some(mps_ach));

            self.drag_kw[i] = 0.5
                * self.props.air_density_kg_per_m3
                * self.veh.drag_coef
                * self.veh.frontal_area_m2
                * ((self.mps_ach[i - 1] + mps_ach) / 2.0).powf(3.0)
                / 1e3;
            self.accel_kw[i] = self.veh.veh_kg / (2.0 * self.cyc.dt_s_at_i(i))
                * (mps_ach.powf(2.0) - self.mps_ach[i - 1].powf(2.0))
                / 1e3;
            self.ascent_kw[i] = self.props.a_grav_mps2
                * grade.atan().sin()
                * self.veh.veh_kg
                * (self.mps_ach[i - 1] + mps_ach)
                / 2.0
                / 1e3;
            self.cyc_trac_kw_req[i] = self.drag_kw[i] + self.accel_kw[i] + self.ascent_kw[i];
            self.spare_trac_kw[i] = self.cur_max_trac_kw[i] - self.cyc_trac_kw_req[i];
            self.rr_kw[i] = self.veh.veh_kg
                * self.props.a_grav_mps2
                * self.veh.wheel_rr_coef
                * grade.atan().cos()
                * (self.mps_ach[i - 1] + mps_ach)
                / 2.0
                / 1e3;
            self.cyc_whl_rad_per_sec[i] = mps_ach / self.veh.wheel_radius_m;
            self.cyc_tire_inertia_kw[i] = (0.5
                * self.veh.wheel_inertia_kg_m2
                * self.veh.num_wheels
                * self.cyc_whl_rad_per_sec[i].powf(2.0)
                / self.cyc.dt_s_at_i(i)
                - 0.5
                    * self.veh.wheel_inertia_kg_m2
                    * self.veh.num_wheels
                    * (self.mps_ach[i - 1] / self.veh.wheel_radius_m).powf(2.0)
                    / self.cyc.dt_s_at_i(i))
                / 1e3;

            self.cyc_whl_kw_req[i] =
                self.cyc_trac_kw_req[i] + self.rr_kw[i] + self.cyc_tire_inertia_kw[i];
            self.regen_contrl_lim_kw_perc[i] = self.veh.max_regen
                / (1.0
                    + self.veh.regen_a
                        * (-self.veh.regen_b
                            * ((self.cyc.mph_at_i(i)
                                + self.mps_ach[i - 1] * params::MPH_PER_MPS)
                                / 2.0
                                + 1.0))
                            .exp());
            self.cyc_regen_brake_kw[i] = max(
                min(
                    self.cur_max_mech_mc_kw_in[i] * self.veh.trans_eff,
                    self.regen_contrl_lim_kw_perc[i] * -self.cyc_whl_kw_req[i],
                ),
                0.0,
            );
            self.cyc_fric_brake_kw[i] =
                -min(self.cyc_regen_brake_kw[i] + self.cyc_whl_kw_req[i], 0.0);
            self.cyc_trans_kw_out_req[i] = self.cyc_whl_kw_req[i] + self.cyc_fric_brake_kw[i];

            if self.cyc_trans_kw_out_req[i] <= self.cur_max_trans_kw_out[i] {
                self.cyc_met[i] = true;
                self.trans_kw_out_ach[i] = self.cyc_trans_kw_out_req[i];
            } else {
                self.cyc_met[i] = false;
                self.trans_kw_out_ach[i] = self.cur_max_trans_kw_out[i];
            }

            if self.trans_kw_out_ach[i] > 0.0 {
                self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] / self.veh.trans_eff;
            } else {
                self.trans_kw_in_ach[i] = self.trans_kw_out_ach[i] * self.veh.trans_eff;
            }

            if self.cyc_met[i] {
                if self.veh.fc_eff_type == H2FC {
                    self.min_mc_kw_2help_fc[i] =
                        max(self.trans_kw_in_ach[i], -self.cur_max_mech_mc_kw_in[i]);
                } else {
                    self.min_mc_kw_2help_fc[i] = max(
                        self.trans_kw_in_ach[i] - self.cur_max_fc_kw_out[i],
                        -self.cur_max_mech_mc_kw_in[i],
                    );
                }
            } else {
                self.min_mc_kw_2help_fc[i] =
                    max(self.cur_max_mc_kw_out[i], -self.cur_max_mech_mc_kw_in[i]);
            }
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_power_calcs_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), String> {
            // Cycle is met
            if self.cyc_met[i] {
                self.mps_ach[i] = self.cyc.mps[i];
            }
            //Cycle is not met
            else {
                let mut grade_estimate = self.estimate_grade_for_step(i);
                let mut grade: f64;
                let grade_tol = 1e-4;
                let mut grade_diff = grade_tol + 1.0;
                let max_grade_iter = 3;
                let mut grade_iter = 0;
                while grade_diff > grade_tol && grade_iter < max_grade_iter {
                    grade_iter += 1;
                    grade = grade_estimate;

                    let drag3 = 1.0 / 16.0
                        * self.props.air_density_kg_per_m3
                        * self.veh.drag_coef
                        * self.veh.frontal_area_m2;
                    let accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s_at_i(i);
                    let drag2 = 3.0 / 16.0
                        * self.props.air_density_kg_per_m3
                        * self.veh.drag_coef
                        * self.veh.frontal_area_m2
                        * self.mps_ach[i - 1];
                    let wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels
                        / (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m.powf(2.0));
                    let drag1 = 3.0 / 16.0
                        * self.props.air_density_kg_per_m3
                        * self.veh.drag_coef
                        * self.veh.frontal_area_m2
                        * self.mps_ach[i - 1].powf(2.0);
                    let roll1 = 0.5
                        * self.veh.veh_kg
                        * self.props.a_grav_mps2
                        * self.veh.wheel_rr_coef
                        * grade.atan().cos();
                    let ascent1 =
                        0.5 * self.props.a_grav_mps2 * grade.atan().sin() * self.veh.veh_kg;
                    let accel0 = -0.5 * self.veh.veh_kg * self.mps_ach[i - 1].powf(2.0)
                        / self.cyc.dt_s_at_i(i);
                    let drag0 = 1.0 / 16.0
                        * self.props.air_density_kg_per_m3
                        * self.veh.drag_coef
                        * self.veh.frontal_area_m2
                        * self.mps_ach[i - 1].powf(3.0);
                    let roll0 = 0.5
                        * self.veh.veh_kg
                        * self.props.a_grav_mps2
                        * self.veh.wheel_rr_coef
                        * grade.atan().cos()
                        * self.mps_ach[i - 1];
                    let ascent0 = 0.5
                        * self.props.a_grav_mps2
                        * grade.atan().sin()
                        * self.veh.veh_kg
                        * self.mps_ach[i - 1];
                    let wheel0 = -0.5
                        * self.veh.wheel_inertia_kg_m2
                        * self.veh.num_wheels
                        * self.mps_ach[i - 1].powf(2.0)
                        / (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m.powf(2.0));

                    let total3 = drag3 / 1e3;
                    let total2 = (accel2 + drag2 + wheel2) / 1e3;
                    let total1 = (drag1 + roll1 + ascent1) / 1e3;
                    let total0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / 1e3
                        - self.cur_max_trans_kw_out[i];

                    let totals = array![total3, total2, total1, total0];

                    let t3 = totals[0];
                    let t2 = totals[1];
                    let t1 = totals[2];
                    let t0 = totals[3];
                    // initial guess
                    let xi = max(1.0, self.mps_ach[i - 1]);
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
                        // let end = ;
                        let xi =
                            xs[xs.len() - 1] * (1.0 - g) - g * bs[xs.len() - 1] / ms[xs.len() - 1];
                        let yi = t3 * xi.powf(3.0) + t2 * xi.powf(2.0) + t1 * xi + t0;
                        let mi = 3.0 * t3 * xi.powf(2.0) + 2.0 * t2 * xi + t1;
                        let bi = yi - xi * mi;
                        xs.push(xi);
                        ys.push(yi);
                        ms.push(mi);
                        bs.push(bi);
                        converged =
                            ((xs[xs.len() - 1] - xs[xs.len() - 2]) / xs[xs.len() - 2]).abs() < xtol;
                        iterate += 1;
                    }

                    self.newton_iters[i] = iterate;

                    let _ys = Array::from_vec(ys).map(|x| x.abs());
                    self.mps_ach[i] = max(
                        xs[_ys.iter().position(|&x| x == ndarrmin(&_ys)).unwrap()],
                        0.0,
                    );
                    grade_estimate = self.lookup_grade_for_step(i, Some(self.mps_ach[i]));
                    grade_diff = (grade - grade_estimate).abs();
                }
            }

            if let Err(message) = self.set_power_calcs(i) {
                Err(
                    "call to `set_power_calcs_rust` failed within `set_ach_speed_rust`: "
                        .to_string()
                        + &message,
                )
            } else {
                self.mph_ach[i] = self.mps_ach[i] * params::MPH_PER_MPS;
                self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s_at_i(i);
                self.dist_mi[i] = self.dist_m[i] * 1.0 / params::M_PER_MI;
                Ok(())
            }
        };

        if let Err(message) = res() {
            Err("`set_ach_speed_rust` failed: ".to_string() + &message)
        } else {
            Ok(())
        }
    }

    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            if self.veh.no_elec_sys {
                self.regen_buff_soc[i] = 0.0;
            } else if self.veh.charging_on {
                self.regen_buff_soc[i] = max(
                    self.veh.max_soc - (self.veh.max_regen_kwh() / self.veh.ess_max_kwh),
                    (self.veh.max_soc + self.veh.min_soc) / 2.0,
                );
            } else {
                self.regen_buff_soc[i] = max(
                    (self.veh.ess_max_kwh * self.veh.max_soc
                        - 0.5
                            * self.veh.veh_kg
                            * (self.cyc.mps[i].powf(2.0))
                            * (1.0 / 1_000.0)
                            * (1.0 / 3_600.0)
                            * self.veh.mc_peak_eff()
                            * self.veh.max_regen)
                        / self.veh.ess_max_kwh,
                    self.veh.min_soc,
                );

                self.ess_regen_buff_dischg_kw[i] = min(
                    self.cur_ess_max_kw_out[i],
                    max(
                        0.0,
                        (self.soc[i - 1] - self.regen_buff_soc[i]) * self.veh.ess_max_kwh * 3_600.0
                            / self.cyc.dt_s_at_i(i),
                    ),
                );

                self.max_ess_regen_buff_chg_kw[i] = min(
                    max(
                        0.0,
                        (self.regen_buff_soc[i] - self.soc[i - 1]) * self.veh.ess_max_kwh * 3.6e3
                            / self.cyc.dt_s_at_i(i),
                    ),
                    self.cur_max_ess_chg_kw[i],
                );
            }
            if self.veh.no_elec_sys {
                self.accel_buff_soc[i] = 0.0;
            } else {
                self.accel_buff_soc[i] = min(
                    max(
                        ((self.veh.max_accel_buffer_mph / params::MPH_PER_MPS).powf(2.0)
                            - self.cyc.mps[i].powf(2.0))
                            / (self.veh.max_accel_buffer_mph / params::MPH_PER_MPS).powf(2.0)
                            * min(
                                self.veh.max_accel_buffer_perc_of_useable_soc
                                    * (self.veh.max_soc - self.veh.min_soc),
                                self.veh.max_regen_kwh() / self.veh.ess_max_kwh,
                            )
                            * self.veh.ess_max_kwh
                            / self.veh.ess_max_kwh
                            + self.veh.min_soc,
                        self.veh.min_soc,
                    ),
                    self.veh.max_soc,
                );

                self.ess_accel_buff_chg_kw[i] = max(
                    0.0,
                    (self.accel_buff_soc[i] - self.soc[i - 1]) * self.veh.ess_max_kwh * 3.6e3
                        / self.cyc.dt_s_at_i(i),
                );
                self.max_ess_accell_buff_dischg_kw[i] = min(
                    max(
                        0.0,
                        (self.soc[i - 1] - self.accel_buff_soc[i]) * self.veh.ess_max_kwh * 3.6e3
                            / self.cyc.dt_s_at_i(i),
                    ),
                    self.cur_ess_max_kw_out[i],
                );
            }
            if self.regen_buff_soc[i] < self.accel_buff_soc[i] {
                self.ess_accel_regen_dischg_kw[i] = max(
                    min(
                        (self.soc[i - 1] - (self.regen_buff_soc[i] + self.accel_buff_soc[i]) / 2.0)
                            * self.veh.ess_max_kwh
                            * 3.6e3
                            / self.cyc.dt_s_at_i(i),
                        self.cur_ess_max_kw_out[i],
                    ),
                    -self.cur_max_ess_chg_kw[i],
                );
            } else if self.soc[i - 1] > self.regen_buff_soc[i] {
                self.ess_accel_regen_dischg_kw[i] = max(
                    min(self.ess_regen_buff_dischg_kw[i], self.cur_ess_max_kw_out[i]),
                    -self.cur_max_ess_chg_kw[i],
                );
            } else if self.soc[i - 1] < self.accel_buff_soc[i] {
                self.ess_accel_regen_dischg_kw[i] = max(
                    min(
                        -1.0 * self.ess_accel_buff_chg_kw[i],
                        self.cur_ess_max_kw_out[i],
                    ),
                    -self.cur_max_ess_chg_kw[i],
                );
            } else {
                self.ess_accel_regen_dischg_kw[i] = max(
                    min(0.0, self.cur_ess_max_kw_out[i]),
                    -self.cur_max_ess_chg_kw[i],
                );
            }
            self.fc_kw_gap_fr_eff[i] = (self.trans_kw_out_ach[i] - self.veh.max_fc_eff_kw()).abs();

            if self.veh.no_elec_sys {
                self.mc_elec_in_kw_for_max_fc_eff[i] = 0.0;
            } else if self.trans_kw_out_ach[i] < self.veh.max_fc_eff_kw() {
                if self.fc_kw_gap_fr_eff[i] == self.veh.mc_max_kw {
                    self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i]
                        / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1];
                } else {
                    self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i]
                        / self.veh.mc_full_eff_array[cmp::max(
                            1,
                            first_grtr(
                                &self.veh.mc_kw_out_array,
                                min(self.veh.mc_max_kw - 0.01, self.fc_kw_gap_fr_eff[i]),
                            )
                            .unwrap_or(0)
                                - 1,
                        )];
                }
            } else if self.fc_kw_gap_fr_eff[i] == self.veh.mc_max_kw {
                self.mc_elec_in_kw_for_max_fc_eff[i] =
                    self.veh.mc_kw_in_array[self.veh.mc_kw_in_array.len() - 1];
            } else {
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[first_grtr(
                    &self.veh.mc_kw_out_array,
                    min(self.veh.mc_max_kw - 0.01, self.fc_kw_gap_fr_eff[i]),
                )
                .unwrap_or(0)
                    - 1];
            }
            if self.veh.no_elec_sys {
                self.elec_kw_req_4ae[i] = 0.0;
            } else if self.trans_kw_in_ach[i] > 0.0 {
                if self.trans_kw_in_ach[i] == self.veh.mc_max_kw {
                    self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i]
                        / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1]
                        + self.aux_in_kw[i];
                } else {
                    self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i]
                        / self.veh.mc_full_eff_array[cmp::max(
                            1,
                            first_grtr(
                                &self.veh.mc_kw_out_array,
                                min(self.veh.mc_max_kw - 0.01, self.trans_kw_in_ach[i]),
                            )
                            .unwrap_or(0)
                                - 1,
                        )]
                        + self.aux_in_kw[i];
                }
            } else {
                self.elec_kw_req_4ae[i] = 0.0;
            }

            self.prev_fc_time_on[i] = self.fc_time_on[i - 1];

            // some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.
            if self.veh.fc_max_kw == 0.0 {
                self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i - 1]
                    && (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i]
                    && (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i]
                        || self.veh.fc_max_kw == 0.0);
            } else {
                self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i - 1]
                    && (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i]
                    && (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i]
                        || self.veh.fc_max_kw == 0.0)
                    && ((self.cyc.mph_at_i(i) - 1e-6) <= self.veh.mph_fc_on
                        || self.veh.charging_on)
                    && self.elec_kw_req_4ae[i] <= self.veh.kw_demand_fc_on;
            }
            if self.can_pwr_all_elec[i] {
                if self.trans_kw_in_ach[i] < self.aux_in_kw[i] {
                    self.desired_ess_kw_out_for_ae[i] = self.aux_in_kw[i] + self.trans_kw_in_ach[i];
                } else if self.regen_buff_soc[i] < self.accel_buff_soc[i] {
                    self.desired_ess_kw_out_for_ae[i] = self.ess_accel_regen_dischg_kw[i];
                } else if self.soc[i - 1] > self.regen_buff_soc[i] {
                    self.desired_ess_kw_out_for_ae[i] = self.ess_regen_buff_dischg_kw[i];
                } else if self.soc[i - 1] < self.accel_buff_soc[i] {
                    self.desired_ess_kw_out_for_ae[i] = -self.ess_accel_buff_chg_kw[i];
                } else {
                    self.desired_ess_kw_out_for_ae[i] = self.trans_kw_in_ach[i] + self.aux_in_kw[i]
                        - self.cur_max_roadway_chg_kw[i];
                }
            } else {
                self.desired_ess_kw_out_for_ae[i] = 0.0;
            }

            if self.can_pwr_all_elec[i] {
                self.ess_ae_kw_out[i] = max(
                    -self.cur_max_ess_chg_kw[i],
                    max(
                        -self.max_ess_regen_buff_chg_kw[i],
                        max(
                            min(
                                0.0,
                                self.cur_max_roadway_chg_kw[i] - self.trans_kw_in_ach[i]
                                    + self.aux_in_kw[i],
                            ),
                            min(
                                self.cur_ess_max_kw_out[i],
                                self.desired_ess_kw_out_for_ae[i],
                            ),
                        ),
                    ),
                );
            } else {
                self.ess_ae_kw_out[i] = 0.0;
            }

            self.er_ae_kw_out[i] = min(
                max(
                    0.0,
                    self.trans_kw_in_ach[i] + self.aux_in_kw[i] - self.ess_ae_kw_out[i],
                ),
                self.cur_max_roadway_chg_kw[i],
            );
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_hybrid_cont_calcs_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_forced_state_rust(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            // force fuel converter on if it was on in the previous time step, but only if fc
            // has not been on longer than minFcTimeOn
            if self.prev_fc_time_on[i] > 0.0
                && self.prev_fc_time_on[i] < self.veh.min_fc_time_on - self.cyc.dt_s_at_i(i)
            {
                self.fc_forced_on[i] = true;
            } else {
                self.fc_forced_on[i] = false
            }

            if !self.fc_forced_on[i] || !self.can_pwr_all_elec[i] {
                self.fc_forced_state[i] = 1;
                self.mc_mech_kw_4forced_fc[i] = 0.0;
            } else if self.trans_kw_in_ach[i] < 0.0 {
                self.fc_forced_state[i] = 2;
                self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i];
            } else if self.veh.max_fc_eff_kw() == self.trans_kw_in_ach[i] {
                self.fc_forced_state[i] = 3;
                self.mc_mech_kw_4forced_fc[i] = 0.0;
            } else if self.veh.idle_fc_kw > self.trans_kw_in_ach[i] && self.accel_kw[i] >= 0.0 {
                self.fc_forced_state[i] = 4;
                self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.idle_fc_kw;
            } else if self.veh.max_fc_eff_kw() > self.trans_kw_in_ach[i] {
                self.fc_forced_state[i] = 5;
                self.mc_mech_kw_4forced_fc[i] = 0.0;
            } else {
                self.fc_forced_state[i] = 6;
                self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.max_fc_eff_kw();
            }
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_fc_forced_state_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            if (-self.mc_elec_in_kw_for_max_fc_eff[i] - self.cur_max_roadway_chg_kw[i]) > 0.0 {
                self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i]
                    - self.cur_max_roadway_chg_kw[i])
                    * self.veh.ess_dischg_to_fc_max_eff_perc;
            } else {
                self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i]
                    - self.cur_max_roadway_chg_kw[i])
                    * self.veh.ess_chg_to_fc_max_eff_perc;
            }

            if self.accel_buff_soc[i] > self.regen_buff_soc[i] {
                self.ess_kw_if_fc_req[i] = min(
                    self.cur_ess_max_kw_out[i],
                    min(
                        self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                        min(
                            self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                            max(
                                -self.cur_max_ess_chg_kw[i],
                                self.ess_accel_regen_dischg_kw[i],
                            ),
                        ),
                    ),
                );
            } else if self.ess_regen_buff_dischg_kw[i] > 0.0 {
                self.ess_kw_if_fc_req[i] = min(
                    self.cur_ess_max_kw_out[i],
                    min(
                        self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                        min(
                            self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                            max(
                                -self.cur_max_ess_chg_kw[i],
                                min(
                                    self.ess_accel_regen_dischg_kw[i],
                                    min(
                                        self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i],
                                        max(
                                            self.ess_regen_buff_dischg_kw[i],
                                            self.ess_desired_kw_4fc_eff[i],
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                );
            } else if self.ess_accel_buff_chg_kw[i] > 0.0 {
                self.ess_kw_if_fc_req[i] = min(
                    self.cur_ess_max_kw_out[i],
                    min(
                        self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                        min(
                            self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                            max(
                                -self.cur_max_ess_chg_kw[i],
                                max(
                                    -self.max_ess_regen_buff_chg_kw[i],
                                    min(
                                        -self.ess_accel_buff_chg_kw[i],
                                        self.ess_desired_kw_4fc_eff[i],
                                    ),
                                ),
                            ),
                        ),
                    ),
                );
            } else if self.ess_desired_kw_4fc_eff[i] > 0.0 {
                self.ess_kw_if_fc_req[i] = min(
                    self.cur_ess_max_kw_out[i],
                    min(
                        self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                        min(
                            self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                            max(
                                -self.cur_max_ess_chg_kw[i],
                                min(
                                    self.ess_desired_kw_4fc_eff[i],
                                    self.max_ess_accell_buff_dischg_kw[i],
                                ),
                            ),
                        ),
                    ),
                );
            } else {
                self.ess_kw_if_fc_req[i] = min(
                    self.cur_ess_max_kw_out[i],
                    min(
                        self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                        min(
                            self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                            max(
                                -self.cur_max_ess_chg_kw[i],
                                max(
                                    self.ess_desired_kw_4fc_eff[i],
                                    -self.max_ess_regen_buff_chg_kw[i],
                                ),
                            ),
                        ),
                    ),
                );
            }

            self.er_kw_if_fc_req[i] = max(
                0.0,
                min(
                    self.cur_max_roadway_chg_kw[i],
                    min(
                        self.cur_max_mech_mc_kw_in[i],
                        self.ess_kw_if_fc_req[i] - self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i],
                    ),
                ),
            );

            self.mc_elec_kw_in_if_fc_req[i] =
                self.ess_kw_if_fc_req[i] + self.er_kw_if_fc_req[i] - self.aux_in_kw[i];

            if self.veh.no_elec_sys || self.mc_elec_kw_in_if_fc_req[i] == 0.0 {
                self.mc_kw_if_fc_req[i] = 0.0;
            } else if self.mc_elec_kw_in_if_fc_req[i] > 0.0 {
                if self.mc_elec_kw_in_if_fc_req[i] == arrmax(&self.veh.mc_kw_in_array) {
                    self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                        * self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1];
                } else {
                    self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                        * self.veh.mc_full_eff_array[cmp::max(
                            1,
                            first_grtr(
                                &self.veh.mc_kw_in_array,
                                min(
                                    arrmax(&self.veh.mc_kw_in_array) - 0.01,
                                    self.mc_elec_kw_in_if_fc_req[i],
                                ),
                            )
                            .unwrap_or(0)
                                - 1,
                        )]
                }
            } else if -self.mc_elec_kw_in_if_fc_req[i] == arrmax(&self.veh.mc_kw_in_array) {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                    / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1];
            } else {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                    / self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_in_array,
                            min(
                                arrmax(&self.veh.mc_kw_in_array) - 0.01,
                                -self.mc_elec_kw_in_if_fc_req[i],
                            ),
                        )
                        .unwrap_or(0)
                            - 1,
                    )];
            }

            if self.veh.mc_max_kw == 0.0 {
                self.mc_mech_kw_out_ach[i] = 0.0;
            } else if self.fc_forced_on[i]
                && self.can_pwr_all_elec[i]
                && (self.veh.veh_pt_type == HEV || self.veh.veh_pt_type == PHEV)
                && (self.veh.fc_eff_type != H2FC)
            {
                self.mc_mech_kw_out_ach[i] = self.mc_mech_kw_4forced_fc[i];
            } else if self.trans_kw_in_ach[i] <= 0.0 {
                if self.veh.fc_eff_type != H2FC && self.veh.fc_max_kw > 0.0 {
                    if self.can_pwr_all_elec[i] {
                        self.mc_mech_kw_out_ach[i] =
                            -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]);
                    } else {
                        self.mc_mech_kw_out_ach[i] = min(
                            -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                            max(-self.cur_max_fc_kw_out[i], self.mc_kw_if_fc_req[i]),
                        );
                    }
                } else {
                    self.mc_mech_kw_out_ach[i] = min(
                        -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                        -self.trans_kw_in_ach[i],
                    );
                }
            } else if self.can_pwr_all_elec[i] {
                self.mc_mech_kw_out_ach[i] = self.trans_kw_in_ach[i]
            } else {
                self.mc_mech_kw_out_ach[i] =
                    max(self.min_mc_kw_2help_fc[i], self.mc_kw_if_fc_req[i])
            }

            if self.mc_mech_kw_out_ach[i] == 0.0 {
                self.mc_elec_kw_in_ach[i] = 0.0;
            } else if self.mc_mech_kw_out_ach[i] < 0.0 {
                if -self.mc_mech_kw_out_ach[i] == arrmax(&self.veh.mc_kw_in_array) {
                    self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                        * self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1]
                } else {
                    self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                        * self.veh.mc_full_eff_array[cmp::max(
                            1,
                            first_grtr(
                                &self.veh.mc_kw_in_array,
                                min(
                                    arrmax(&self.veh.mc_kw_in_array) - 0.01,
                                    -self.mc_mech_kw_out_ach[i],
                                ),
                            )
                            .unwrap_or(0)
                                - 1,
                        )];
                }
            } else if self.veh.mc_max_kw == self.mc_mech_kw_out_ach[i] {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                    / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1]
            } else {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                    / self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_out_array,
                            min(self.veh.mc_max_kw - 0.01, self.mc_mech_kw_out_ach[i]),
                        )
                        .unwrap_or(0)
                            - 1,
                    )];
            }

            if self.cur_max_roadway_chg_kw[i] == 0.0 {
                self.roadway_chg_kw_out_ach[i] = 0.0
            } else if self.veh.fc_eff_type == H2FC {
                self.roadway_chg_kw_out_ach[i] = max(
                    0.0,
                    max(
                        self.mc_elec_kw_in_ach[i],
                        max(
                            self.max_ess_regen_buff_chg_kw[i],
                            max(
                                self.ess_regen_buff_dischg_kw[i],
                                self.cur_max_roadway_chg_kw[i],
                            ),
                        ),
                    ),
                );
            } else if self.can_pwr_all_elec[i] {
                self.roadway_chg_kw_out_ach[i] = self.er_ae_kw_out[i];
            } else {
                self.roadway_chg_kw_out_ach[i] = self.er_kw_if_fc_req[i];
            }

            self.min_ess_kw_2help_fc[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i]
                - self.cur_max_fc_kw_out[i]
                - self.roadway_chg_kw_out_ach[i];

            if self.veh.ess_max_kw == 0.0 || self.veh.ess_max_kwh == 0.0 {
                self.ess_kw_out_ach[i] = 0.0;
            } else if self.veh.fc_eff_type == H2FC {
                if self.trans_kw_out_ach[i] >= 0.0 {
                    self.ess_kw_out_ach[i] = min(
                        self.cur_ess_max_kw_out[i],
                        min(
                            self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i]
                                - self.roadway_chg_kw_out_ach[i],
                            max(
                                self.min_ess_kw_2help_fc[i],
                                max(
                                    self.ess_desired_kw_4fc_eff[i],
                                    self.ess_accel_regen_dischg_kw[i],
                                ),
                            ),
                        ),
                    );
                } else {
                    self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i]
                        - self.roadway_chg_kw_out_ach[i];
                }
            } else if self.high_acc_fc_on_tag[i] || self.veh.no_elec_aux {
                self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] - self.roadway_chg_kw_out_ach[i];
            } else {
                self.ess_kw_out_ach[i] =
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i];
            }

            if self.veh.no_elec_sys {
                self.ess_cur_kwh[i] = 0.0
            } else if self.ess_kw_out_ach[i] < 0.0 {
                self.ess_cur_kwh[i] = self.ess_cur_kwh[i - 1]
                    - self.ess_kw_out_ach[i] * self.cyc.dt_s_at_i(i) / 3.6e3
                        * self.veh.ess_round_trip_eff.sqrt();
            } else {
                self.ess_cur_kwh[i] = self.ess_cur_kwh[i - 1]
                    - self.ess_kw_out_ach[i] * self.cyc.dt_s_at_i(i) / 3.6e3
                        * (1.0 / self.veh.ess_round_trip_eff.sqrt());
            }

            if self.veh.ess_max_kwh == 0.0 {
                self.soc[i] = 0.0;
            } else {
                self.soc[i] = self.ess_cur_kwh[i] / self.veh.ess_max_kwh;
            }

            if self.can_pwr_all_elec[i] && !self.fc_forced_on[i] && self.fc_kw_out_ach[i] == 0.0 {
                self.fc_time_on[i] = 0.0
            } else {
                self.fc_time_on[i] = self.fc_time_on[i - 1] + self.cyc.dt_s_at_i(i);
            }
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_hybrid_cont_decisions_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_power(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            if self.veh.fc_max_kw == 0.0 {
                self.fc_kw_out_ach[i] = 0.0;
            } else if self.veh.fc_eff_type == H2FC {
                self.fc_kw_out_ach[i] = min(
                    self.cur_max_fc_kw_out[i],
                    max(
                        0.0,
                        self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i]
                            - self.ess_kw_out_ach[i]
                            - self.roadway_chg_kw_out_ach[i],
                    ),
                );
            } else if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
                self.fc_kw_out_ach[i] = min(
                    self.cur_max_fc_kw_out[i],
                    max(
                        0.0,
                        self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i] + self.aux_in_kw[i],
                    ),
                );
            } else {
                self.fc_kw_out_ach[i] = min(
                    self.cur_max_fc_kw_out[i],
                    max(0.0, self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i]),
                );
            }

            if self.veh.fc_max_kw == 0.0 {
                self.fc_kw_out_ach_pct[i] = 0.0;
            } else {
                self.fc_kw_out_ach_pct[i] = self.fc_kw_out_ach[i] / self.veh.fc_max_kw;
            }

            if self.fc_kw_out_ach[i] == 0.0 {
                self.fc_kw_in_ach[i] = 0.0;
                self.fc_kw_out_ach_pct[i] = 0.0;
            } else if self.veh.fc_eff_array[first_grtr(
                &self.veh.fc_kw_out_array,
                min(self.fc_kw_out_ach[i], self.veh.fc_max_kw),
            )
            .unwrap_or(0)
                - 1]
                != 0.0
            {
                self.fc_kw_in_ach[i] = self.fc_kw_out_ach[i]
                    / (self.veh.fc_eff_array[first_grtr(
                        &self.veh.fc_kw_out_array,
                        min(self.fc_kw_out_ach[i], self.veh.fc_max_kw),
                    )
                    .unwrap_or(0)
                        - 1]);
            } else {
                self.fc_kw_in_ach[i] = 0.0
            }

            self.fs_kw_out_ach[i] = self.fc_kw_in_ach[i];

            self.fs_kwh_out_ach[i] = self.fs_kw_out_ach[i] * self.cyc.dt_s_at_i(i) / 3.6e3;
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_fc_power_rust` failed".to_string())
        } else {
            Ok(())
        }
    }

    ///
    pub fn set_time_dilation(&mut self, i: usize) -> Result<(), String> {
        let mut res = || -> Result<(), String> {
            // if prescribed speed is zero, trace is met to avoid div-by-zero errors and other possible wackiness
            let mut trace_met = (self.cyc.dist_m().slice(s![0..(i + 1)]).sum()
                - self.dist_m.slice(s![0..(i + 1)]).sum())
            .abs()
                / self.cyc0.dist_m().slice(s![0..(i + 1)]).sum()
                < self.sim_params.time_dilation_tol
                || self.cyc.mps[i] == 0.0;

            let mut d_short: Vec<f64> = vec![];
            let mut t_dilation: Vec<f64> = vec![0.0]; // no time dilation initially
            if !trace_met {
                self.trace_miss_iters[i] += 1;

                d_short.push(
                    self.cyc0.dist_m().slice(s![0..i + 1]).sum()
                        - self.dist_m.slice(s![0..i + 1]).sum(),
                ); // positive if behind trace
                t_dilation.push(min(
                    max(
                        d_short[d_short.len() - 1] / self.cyc0.dt_s_at_i(i) / self.mps_ach[i], // initial guess, speed that needed to be achived per speed that was achieved
                        self.sim_params.min_time_dilation,
                    ),
                    self.sim_params.max_time_dilation,
                ));

                // add time dilation factor * step size to current and subsequent times
                self.cyc.time_s = add_from(
                    &self.cyc.time_s,
                    i,
                    self.cyc.dt_s_at_i(i) * t_dilation[t_dilation.len() - 1],
                );
                if let Err(message) = self.solve_step(i) {
                    return Err(message);
                }

                trace_met =
                    // convergence criteria
                    (self.cyc0.dist_m().slice(s![0..i+1]).sum() - self.dist_m.slice(s![0..i+1]).sum()).abs() / self.cyc0.dist_m().slice(s![0..i+1]).sum()
                    < self.sim_params.time_dilation_tol
                    // exceeding max time dilation
                    || t_dilation[t_dilation.len()-1] >= self.sim_params.max_time_dilation
                    // lower than min time dilation
                    || t_dilation[t_dilation.len()-1] <= self.sim_params.min_time_dilation;
            }
            while !trace_met {
                // iterate newton's method until time dilation has converged or other exit criteria trigger trace_met == True
                // distance shortfall [m]
                // correct time steps
                d_short.push(
                    self.cyc0.dist_m().slice(s![0..i + 1]).sum()
                        - self.dist_m.slice(s![0..i + 1]).sum(),
                );
                t_dilation.push(min(
                    max(
                        t_dilation[t_dilation.len() - 1]
                            - (t_dilation[t_dilation.len() - 1] - t_dilation[t_dilation.len() - 2])
                                / (d_short[d_short.len() - 1] - d_short[d_short.len() - 2])
                                * d_short[d_short.len() - 1],
                        self.sim_params.min_time_dilation,
                    ),
                    self.sim_params.max_time_dilation,
                ));
                self.cyc.time_s = add_from(
                    &self.cyc.time_s,
                    i,
                    self.cyc.dt_s_at_i(i) * t_dilation[t_dilation.len() - 1],
                );

                self.solve_step(i)?;

                if let Err(message) = self.solve_step(i) {
                    return Err(message);
                }
                self.trace_miss_iters[i] += 1;

                trace_met =
                    // convergence criteria
                    (self.cyc0.dist_m().slice(s![0..i+1]).sum() - self.dist_m.slice(s![0..i+1]).sum()).abs() / self.cyc0.dist_m().slice(s![0..i+1]).sum()
                    < self.sim_params.time_dilation_tol
                    // max iterations
                    || self.trace_miss_iters[i] >= self.sim_params.max_trace_miss_iters
                    // exceeding max time dilation
                    || t_dilation[t_dilation.len()-1] >= self.sim_params.max_time_dilation
                    // lower than min time dilation
                    || t_dilation[t_dilation.len()-1] <= self.sim_params.min_time_dilation;
            }
            Ok(())
        };

        if let Err(message) = res() {
            Err("`set_time_dilation_rust` failed: ".to_string() + &message)
        } else {
            Ok(())
        }
    }

    // Calculates the derivative dv/dd (change in speed by change in distance)
    // - v: number, the speed at which to evaluate dv/dd (m/s)
    // - grade: number, the road grade as a decimal fraction
    // RETURN: number, the dv/dd for these conditions
    fn calc_dvdd(&self, v: f64, grade: f64) -> f64 {
        if v <= 0.0 {
            0.0
        } else {
            let atan_grade = grade.atan(); // float(np.arctan(grade))
            let g = self.props.a_grav_mps2;
            let m = self.veh.veh_kg;
            let rho_cdfa =
                self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2;
            let rrc = self.veh.wheel_rr_coef;
            -1.0 * ((g / v) * (atan_grade.sin() + rrc * atan_grade.cos())
                + (0.5 * rho_cdfa * (1.0 / m) * v))
        }
    }

    fn apply_coast_trajectory(&mut self, coast_traj: CoastTrajectory) {
        if coast_traj.found_trajectory {
            let num_speeds = match coast_traj.speeds_m_per_s {
                Some(speeds_m_per_s) => {
                    for (di, &new_speed) in speeds_m_per_s.iter().enumerate() {
                        let idx = coast_traj.start_idx + di;
                        if idx >= self.mps_ach.len() {
                            break;
                        }
                        self.cyc.mps[idx] = new_speed;
                    }
                    speeds_m_per_s.len()
                }
                None => 0,
            };
            let (_, n) = self.cyc.modify_with_braking_trajectory(
                self.sim_params.coast_brake_accel_m_per_s2,
                coast_traj.start_idx + num_speeds,
                coast_traj.distance_to_brake_m,
            );
            for di in 0..(self.cyc0.mps.len() - coast_traj.start_idx) {
                let idx = coast_traj.start_idx + di;
                self.impose_coast[idx] = di < num_speeds + n;
            }
        }
    }

    /// Generate a coast trajectory without actually modifying the cycle.
    /// This can be used to calculate the distance to stop via coast using
    /// actual time-stepping and dynamically changing grade.
    fn generate_coast_trajectory(&self, i: usize) -> CoastTrajectory {
        let v0 = self.mps_ach[i - 1];
        let v_brake = self.sim_params.coast_brake_start_speed_m_per_s;
        let a_brake = self.sim_params.coast_brake_accel_m_per_s2;
        assert![a_brake <= 0.0];
        let ds = ndarrcumsum(&trapz_step_distances(&self.cyc0));
        let gs = self.cyc0.grade.clone();
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let mut distances_m: Vec<f64> = vec![];
        let mut grade_by_distance: Vec<f64> = vec![];
        for idx in 0..ds.len() {
            if ds[idx] >= d0 {
                distances_m.push(ds[idx] - d0);
                grade_by_distance.push(gs[idx])
            }
        }
        if distances_m.is_empty() {
            return CoastTrajectory {
                found_trajectory: false,
                distance_to_stop_via_coast_m: 0.0,
                start_idx: 0,
                speeds_m_per_s: None,
                distance_to_brake_m: None,
            };
        }
        let distances_m = Array::from_vec(distances_m);
        let grade_by_distance = Array::from_vec(grade_by_distance);
        // distance traveled while stopping via friction-braking (i.e., distance to brake)
        if v0 <= v_brake {
            return CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: -0.5 * v0 * v0 / a_brake,
                start_idx: i,
                speeds_m_per_s: None,
                distance_to_brake_m: None,
            };
        }
        let dtb = -0.5 * v_brake * v_brake / a_brake;
        let mut d = 0.0;
        let d_max = distances_m[distances_m.len() - 1] - dtb;
        let unique_grades = ndarrunique(&grade_by_distance);
        let unique_grade: Option<f64> = if unique_grades.len() == 1 {
            Some(unique_grades[0])
        } else {
            None
        };
        let has_unique_grade: bool = unique_grade.is_some();
        let max_iter = 2000;
        let iters_per_step = 2;
        let mut new_speeds_m_per_s: Vec<f64> = vec![];
        let mut v = v0;
        let mut iter = 0;
        let mut idx = i;
        let dts0 = self.cyc0.calc_distance_to_next_stop_from(d0);
        while v > v_brake && v >= 0.0 && d <= d_max && iter < max_iter && idx < self.mps_ach.len() {
            let dt_s = self.cyc0.dt_s_at_i(idx);
            let mut gr = match unique_grade {
                Some(g) => g,
                None => self.cyc0.average_grade_over_range(d + d0, 0.0),
            };
            let mut k = self.calc_dvdd(v, gr);
            let mut v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s);
            let mut vavg = 0.5 * (v + v_next);
            let mut dd: f64;
            for _ in 0..iters_per_step {
                k = self.calc_dvdd(vavg, gr);
                v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s);
                vavg = 0.5 * (v + v_next);
                dd = vavg * dt_s;
                gr = match unique_grade {
                    Some(g) => g,
                    None => self.cyc0.average_grade_over_range(d + d0, dd),
                };
            }
            if k >= 0.0 && has_unique_grade {
                // there is no solution for coastdown -- speed will never decrease
                return CoastTrajectory {
                    found_trajectory: false,
                    distance_to_stop_via_coast_m: 0.0,
                    start_idx: 0,
                    speeds_m_per_s: None,
                    distance_to_brake_m: None,
                };
            }
            if v_next <= v_brake {
                break;
            }
            vavg = 0.5 * (v + v_next);
            dd = vavg * dt_s;
            let dtb = -0.5 * v_next * v_next / a_brake;
            d += dd;
            new_speeds_m_per_s.push(v_next);
            v = v_next;
            if d + dtb > dts0 {
                break;
            }
            iter += 1;
            idx += 1;
        }
        if iter < max_iter && idx < self.mps_ach.len() {
            let dtb = -0.5 * v * v / a_brake;
            let dtb_target = min(max(dts0 - d, 0.5 * dtb), 2.0 * dtb);
            let dtsc = d + dtb_target;
            return CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: dtsc,
                start_idx: i,
                speeds_m_per_s: Some(new_speeds_m_per_s),
                distance_to_brake_m: Some(dtb_target),
            };
        }
        CoastTrajectory {
            found_trajectory: false,
            distance_to_stop_via_coast_m: 0.0,
            start_idx: 0,
            speeds_m_per_s: None,
            distance_to_brake_m: None,
        }
    }

    /// Calculate the distance to stop via coasting in meters.
    /// - i: non-negative-integer, the current index
    /// RETURN: non-negative-number or -1.0
    /// - if -1.0, it means there is no solution to a coast-down distance.
    ///     This can happen due to being too close to the given
    ///     stop or perhaps due to coasting downhill
    /// - if a non-negative-number, the distance in meters that the vehicle
    ///     would freely coast if unobstructed. Accounts for grade between
    ///     the current point and end-point
    fn calc_distance_to_stop_coast_v2(&self, i: usize) -> f64 {
        let not_found = -1.0;
        let v0 = self.cyc.mps[i - 1];
        let v_brake = self.sim_params.coast_brake_start_speed_m_per_s;
        let a_brake = self.sim_params.coast_brake_accel_m_per_s2;
        let ds = ndarrcumsum(&trapz_step_distances(&self.cyc0));
        let gs = self.cyc0.grade.clone();
        assert!(
            ds.len() == gs.len(),
            "Assumed length of ds and gs the same but actually ds.len():{} and gs.len():{}",
            ds.len(),
            gs.len()
        );
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let mut grade_by_distance: Vec<f64> = vec![];
        for idx in 0..ds.len() {
            if ds[idx] >= d0 {
                grade_by_distance.push(gs[idx]);
            }
        }
        let grade_by_distance = Array::from_vec(grade_by_distance);
        let veh_mass_kg = self.veh.veh_kg;
        let air_density_kg_per_m3 = self.props.air_density_kg_per_m3;
        let cdfa_m2 = self.veh.drag_coef * self.veh.frontal_area_m2;
        let rrc = self.veh.wheel_rr_coef;
        let gravity_m_per_s2 = self.props.a_grav_mps2;
        // distance traveled while stopping via friction-braking (i.e., distance to brake)
        let dtb = -0.5 * v_brake * v_brake / a_brake;
        if v0 <= v_brake {
            return -0.5 * v0 * v0 / a_brake;
        }
        let unique_grades = ndarrunique(&grade_by_distance);
        if unique_grades.len() == 1 {
            // if there is only one grade, there may be a closed-form solution
            let unique_grade = unique_grades[0];
            let theta = unique_grade.atan();
            let c1 = gravity_m_per_s2 * (theta.sin() + rrc * theta.cos());
            let c2 = (air_density_kg_per_m3 * cdfa_m2) / (2.0 * veh_mass_kg);
            let v02 = v0 * v0;
            let vb2 = v_brake * v_brake;
            let mut d = not_found;
            let a1 = c1 + c2 * v02;
            let b1 = c1 + c2 * vb2;
            if c2 == 0.0 {
                if c1 > 0.0 {
                    d = (1.0 / (2.0 * c1)) * (v02 - vb2);
                }
            } else if a1 > 0.0 && b1 > 0.0 {
                d = (1.0 / (2.0 * c2)) * (a1.ln() - b1.ln());
            }
            if d != not_found {
                return d + dtb;
            }
        }
        let ct = self.generate_coast_trajectory(i);
        if ct.found_trajectory {
            ct.distance_to_stop_via_coast_m
        } else {
            not_found
        }
    }

    /// - i: non-negative integer, the current position in cyc
    /// - verbose: Bool, if True, prints out debug information
    /// RETURN: Bool if vehicle should initiate coasting
    /// Coast logic is that the vehicle should coast if it is within coasting distance of a stop:
    /// - if distance to coast from start of step is <= distance to next stop
    /// - AND distance to coast from end of step (using prescribed speed) is > distance to next stop
    /// - ALSO, vehicle must have been at or above the coast brake start speed at beginning of step
    /// - AND, must be at least 4 x distances-to-break away
    fn should_impose_coast(&self, i: usize) -> bool {
        if self.sim_params.coast_start_speed_m_per_s > 0.0 {
            return self.cyc.mps[i] >= self.sim_params.coast_start_speed_m_per_s;
        }
        let v0 = self.mps_ach[i - 1];
        if v0 < self.sim_params.coast_brake_start_speed_m_per_s {
            return false;
        }
        // distance to stop by coasting from start of step (i-1)
        let dtsc0 = self.calc_distance_to_stop_coast_v2(i);
        if dtsc0 < 0.0 {
            return false;
        }
        // distance to next stop (m)
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let dts0 = self.cyc0.calc_distance_to_next_stop_from(d0);
        let dtb = -0.5 * v0 * v0 / self.sim_params.coast_brake_accel_m_per_s2;
        dtsc0 >= dts0 && dts0 >= (4.0 * dtb)
    }

    /// Calculate next rendezvous trajectory for eco-coasting
    /// - i: non-negative integer, the index into cyc for the end of start-of-step
    ///     (i.e., the step that may be modified; should be i)
    /// - min_accel_m__s2: number, the minimum acceleration permitted (m/s2)
    /// - max_accel_m__s2: number, the maximum acceleration permitted (m/s2)
    /// RETURN: (Tuple
    ///     found_rendezvous: Bool, if True the remainder of the data is valid; if False, no rendezvous found
    ///     n: positive integer, the number of steps ahead to rendezvous at
    ///     jerk_m__s3: number, the Jerk or first-derivative of acceleration (m/s3)
    ///     accel_m__s2: number, the initial acceleration of the trajectory (m/s2)
    /// )
    /// If no rendezvous exists within the scope, the returned tuple has False for the first item.
    /// Otherwise, returns the next closest rendezvous in time/space
    fn calc_next_rendezvous_trajectory(
        &self,
        i: usize,
        min_accel_m_per_s2: f64,
        max_accel_m_per_s2: f64,
    ) -> (bool, usize, f64, f64) {
        let tol = 1e-6;
        // v0 is where n=0, i.e., idx-1
        let v0 = self.cyc.mps[i - 1];
        let brake_start_speed_m_per_s = self.sim_params.coast_brake_start_speed_m_per_s;
        let brake_accel_m_per_s2 = self.sim_params.coast_brake_accel_m_per_s2;
        let time_horizon_s = max(self.sim_params.coast_time_horizon_for_adjustment_s, 1.0);
        // distance_horizon_m = 1000.0
        let not_found_n: usize = 0;
        let not_found_jerk_m_per_s3: f64 = 0.0;
        let not_found_accel_m_per_s2: f64 = 0.0;
        let not_found: (bool, usize, f64, f64) = (
            false,
            not_found_n,
            not_found_jerk_m_per_s3,
            not_found_accel_m_per_s2,
        );
        if v0 < (brake_start_speed_m_per_s + tol) {
            // don't process braking
            return not_found;
        }
        let (min_accel_m_per_s2, max_accel_m_per_s2) = if min_accel_m_per_s2 > max_accel_m_per_s2 {
            (max_accel_m_per_s2, min_accel_m_per_s2)
        } else {
            (min_accel_m_per_s2, max_accel_m_per_s2)
        };
        let num_samples = self.cyc.mps.len();
        let d0 = trapz_step_start_distance(&self.cyc, i);
        // a_proposed = (v1 - v0) / dt
        // distance to stop from start of time-step
        let dts0 = self.cyc0.calc_distance_to_next_stop_from(d0);
        if dts0 < 0.0 {
            // no stop to coast towards or we're there...
            return not_found;
        }
        let dt = self.cyc.dt_s_at_i(i);
        // distance to brake from the brake start speed (m/s)
        let dtb =
            -0.5 * brake_start_speed_m_per_s * brake_start_speed_m_per_s / brake_accel_m_per_s2;
        // distance to brake initiation from start of time-step (m)
        let dtbi0 = dts0 - dtb;
        if dtbi0 < 0.0 {
            return not_found;
        }
        // Now, check rendezvous trajectories
        let mut step_idx = i;
        let mut dt_plan = 0.0;
        let mut r_best_found = false;
        let mut r_best_n = 0;
        let mut r_best_jerk_m_per_s3 = 0.0;
        let mut r_best_accel_m_per_s2 = 0.0;
        let mut r_best_accel_spread_m_per_s2 = 0.0;
        while dt_plan <= time_horizon_s && step_idx < num_samples {
            dt_plan += self.cyc0.dt_s_at_i(step_idx);
            let step_ahead = step_idx - (i - 1);
            if step_ahead == 1 {
                // for brake init rendezvous
                let accel = (brake_start_speed_m_per_s - v0) / dt;
                let v1 = max(0.0, v0 + accel * dt);
                let dd_proposed = ((v0 + v1) / 2.0) * dt;
                if (v1 - brake_start_speed_m_per_s).abs() < tol && (dtbi0 - dd_proposed).abs() < tol
                {
                    r_best_found = true;
                    r_best_n = 1;
                    r_best_jerk_m_per_s3 = 0.0;
                    r_best_accel_m_per_s2 = accel;
                    break;
                }
            } else {
                // rendezvous trajectory for brake-start -- assumes fixed time-steps
                if dtbi0 > 0.0 {
                    let (r_bi_jerk_m_per_s3, r_bi_accel_m_per_s2) = calc_constant_jerk_trajectory(
                        step_ahead,
                        0.0,
                        v0,
                        dtbi0,
                        brake_start_speed_m_per_s,
                        dt,
                    );
                    if r_bi_accel_m_per_s2 < max_accel_m_per_s2
                        && r_bi_accel_m_per_s2 > min_accel_m_per_s2
                        && r_bi_jerk_m_per_s3 >= 0.0
                    {
                        let as_bi = accel_array_for_constant_jerk(
                            step_ahead,
                            r_bi_accel_m_per_s2,
                            r_bi_jerk_m_per_s3,
                            dt,
                        );
                        let as_bi_min: f64 =
                            as_bi.to_vec().into_iter().reduce(f64::min).unwrap_or(0.0);
                        let as_bi_max: f64 =
                            as_bi.to_vec().into_iter().reduce(f64::max).unwrap_or(0.0);
                        let accel_spread = (as_bi_max - as_bi_min).abs();
                        let flag = (as_bi_max < (max_accel_m_per_s2 + 1e-6)
                            && as_bi_min > (min_accel_m_per_s2 - 1e-6))
                            && (!r_best_found || (accel_spread < r_best_accel_spread_m_per_s2));
                        if flag {
                            r_best_found = true;
                            r_best_n = step_ahead;
                            r_best_accel_m_per_s2 = r_bi_accel_m_per_s2;
                            r_best_jerk_m_per_s3 = r_bi_jerk_m_per_s3;
                            r_best_accel_spread_m_per_s2 = accel_spread;
                        }
                    }
                }
            }
            step_idx += 1;
        }
        if r_best_found {
            return (
                r_best_found,
                r_best_n,
                r_best_jerk_m_per_s3,
                r_best_accel_m_per_s2,
            );
        }
        not_found
    }

    /// Coast Delay allows us to represent coasting to a stop when the lead
    /// vehicle has already moved on from that stop.  In this case, the coasting
    /// vehicle need not dwell at this or any stop while it is lagging behind
    /// the lead vehicle in distance. Instead, the vehicle comes to a stop and
    /// resumes mimicing the lead-vehicle trace at the first time-step the
    /// lead-vehicle moves past the stop-distance. This index is the "coast delay index".
    ///
    /// Arguments
    /// ---------
    /// - i: integer, the step index
    /// NOTE: Resets the coast_delay_index to 0 and calculates and sets the next
    /// appropriate coast_delay_index if appropriate
    fn set_coast_delay(&mut self, i: usize) {
        let speed_tol = 0.01; // m/s
        let dist_tol = 0.1; // meters
        for idx in i..self.cyc.time_s.len() {
            self.coast_delay_index[idx] = 0; // clear all future coast-delays
        }
        let mut coast_delay: Option<i32> = None;
        if !self.sim_params.follow_allow && self.cyc.mps[i] < speed_tol {
            let d0 = trapz_step_start_distance(&self.cyc, i);
            let d0_lv = trapz_step_start_distance(&self.cyc0, i);
            let dtlv0 = d0_lv - d0;
            if dtlv0.abs() > dist_tol {
                let mut d_lv = 0.0;
                let mut min_dtlv: Option<f64> = None;
                for (idx, (&dd, &v)) in trapz_step_distances(&self.cyc0)
                    .iter()
                    .zip(self.cyc0.mps.iter())
                    .enumerate()
                {
                    d_lv += dd;
                    let dtlv = (d_lv - d0).abs();
                    if v < speed_tol && (min_dtlv.is_none() || dtlv <= min_dtlv.unwrap()) {
                        if min_dtlv.is_none()
                            || dtlv < min_dtlv.unwrap()
                            || (d0 < d0_lv && min_dtlv.unwrap() == dtlv)
                        {
                            let i_i32 = i32::try_from(i).unwrap();
                            let idx_i32 = i32::try_from(idx).unwrap();
                            coast_delay = Some(i_i32 - idx_i32);
                        }
                        min_dtlv = Some(dtlv);
                    }
                    if min_dtlv.is_some() && dtlv > min_dtlv.unwrap() {
                        break;
                    }
                }
            }
        }
        match coast_delay {
            Some(cd) => {
                if cd < 0 {
                    let mut new_cd = cd;
                    for idx in i..self.cyc0.mps.len() {
                        self.coast_delay_index[idx] = new_cd;
                        new_cd += 1;
                        if new_cd == 0 {
                            break;
                        }
                    }
                } else {
                    for idx in i..self.cyc0.mps.len() {
                        self.coast_delay_index[idx] = cd;
                    }
                }
            }
            None => (),
        }
    }

    /// Prevent collision between the vehicle in cyc and the one in cyc0.
    /// If a collision will take place, reworks the cyc such that a rendezvous occurs instead.
    /// Arguments
    /// - i: int, index for consideration
    /// - passing_tol_m: None | float, tolerance for how far we have to go past the lead vehicle to be considered "passing"
    /// RETURN: Bool, True if cyc was modified
    fn prevent_collisions(&mut self, i: usize, passing_tol_m: Option<f64>) -> bool {
        let passing_tol_m = passing_tol_m.unwrap_or(1.0);
        let collision: PassingInfo = detect_passing(&self.cyc, &self.cyc0, i, Some(passing_tol_m));
        if !collision.has_collision {
            return false;
        }
        let mut best: RendezvousTrajectory = RendezvousTrajectory {
            found_trajectory: false,
            idx: 0,
            n: 0,
            full_brake_steps: 0,
            jerk_m_per_s3: 0.0,
            accel0_m_per_s2: 0.0,
            accel_spread: 0.0,
        };
        let a_brake_m_per_s2 = self.sim_params.coast_brake_accel_m_per_s2;
        assert!(
            a_brake_m_per_s2 < 0.0,
            "brake acceleration must be negative; got {} m/s2",
            a_brake_m_per_s2
        );
        for full_brake_steps in 0..4 {
            for di in 0..(self.mps_ach.len() - i) {
                let idx = i + di;
                if !self.impose_coast[idx] {
                    if idx == i {
                        break;
                    } else {
                        continue;
                    }
                }
                let n = collision.idx - idx + 1 - full_brake_steps;
                if n < 2 {
                    break;
                }
                if (idx - 1 + full_brake_steps) >= self.cyc.time_s.len() {
                    break;
                }
                let dt = collision.time_step_duration_s;
                let v_start_m_per_s = self.cyc.mps[idx - 1];
                let dt_full_brake =
                    self.cyc.time_s[idx - 1 + full_brake_steps] - self.cyc.time_s[idx - 1];
                let dv_full_brake = dt_full_brake * a_brake_m_per_s2;
                let v_start_jerk_m_per_s = max(v_start_m_per_s + dv_full_brake, 0.0);
                let dd_full_brake = 0.5 * (v_start_m_per_s + v_start_jerk_m_per_s) * dt_full_brake;
                let d_start_m = trapz_step_start_distance(&self.cyc, idx) + dd_full_brake;
                if collision.distance_m <= d_start_m {
                    continue;
                }
                let (jerk_m_per_s3, accel0_m_per_s2) = calc_constant_jerk_trajectory(
                    n,
                    d_start_m,
                    v_start_jerk_m_per_s,
                    collision.distance_m,
                    collision.speed_m_per_s,
                    dt,
                );
                let mut accels_m_per_s2: Vec<f64> = vec![];
                let mut trace_accels_m_per_s2: Vec<f64> = vec![];
                for ni in 0..n {
                    if (ni + idx + full_brake_steps) >= self.cyc.time_s.len() {
                        break;
                    }
                    accels_m_per_s2.push(accel_for_constant_jerk(
                        ni,
                        accel0_m_per_s2,
                        jerk_m_per_s3,
                        dt,
                    ));
                    trace_accels_m_per_s2.push(
                        (self.cyc.mps[ni + idx + full_brake_steps]
                            - self.cyc.mps[ni + idx - 1 + full_brake_steps])
                            / self.cyc.dt_s()[ni + idx + full_brake_steps],
                    );
                }
                let all_sub_coast: bool = trace_accels_m_per_s2
                    .clone()
                    .into_iter()
                    .zip(accels_m_per_s2.clone().into_iter())
                    .fold(
                        true,
                        |all_sc_flag: bool, (trace_accel, accel): (f64, f64)| {
                            if !all_sc_flag {
                                return all_sc_flag;
                            }
                            trace_accel >= accel
                        },
                    );
                let accels_ndarr = Array1::from(accels_m_per_s2.clone());
                let min_accel_m_per_s2 = ndarrmin(&accels_ndarr);
                let max_accel_m_per_s2 = ndarrmax(&accels_ndarr);
                let accept = all_sub_coast;
                let accel_spread = (max_accel_m_per_s2 - min_accel_m_per_s2).abs();
                if accept && (!best.found_trajectory || accel_spread < best.accel_spread) {
                    best = RendezvousTrajectory {
                        found_trajectory: true,
                        idx,
                        n,
                        full_brake_steps,
                        jerk_m_per_s3,
                        accel0_m_per_s2,
                        accel_spread,
                    };
                }
            }
            if best.found_trajectory {
                break;
            }
        }
        if !best.found_trajectory {
            let new_passing_tol_m = if passing_tol_m < 10.0 {
                10.0
            } else {
                passing_tol_m + 5.0
            };
            if new_passing_tol_m > 60.0 {
                return false;
            }
            return self.prevent_collisions(i, Some(new_passing_tol_m));
        }
        for fbs in 0..best.full_brake_steps {
            if (best.idx + fbs) >= self.cyc.time_s.len() {
                break;
            }
            let dt = self.cyc.time_s[best.idx + fbs] - self.cyc.time_s[best.idx - 1];
            let dv = a_brake_m_per_s2 * dt;
            let v_start = self.cyc.mps[best.idx - 1];
            self.cyc.mps[best.idx + fbs] = max(v_start + dv, 0.0);
            self.impose_coast[best.idx + fbs] = true;
            self.coast_delay_index[best.idx + fbs] = 0;
        }
        self.cyc.modify_by_const_jerk_trajectory(
            best.idx + best.full_brake_steps,
            best.n,
            best.jerk_m_per_s3,
            best.accel0_m_per_s2,
        );
        for idx in (best.idx + best.n)..self.cyc0.mps.len() {
            self.impose_coast[idx] = false;
            self.coast_delay_index[idx] = 0;
        }
        true
    }

    /// Placeholder for method to impose coasting.
    /// Might be good to include logic for deciding when to coast.
    /// Solve for the next-step speed that will yield a zero roadload
    fn set_coast_speed(&mut self, i: usize) -> Result<(), String> {
        let tol = 1e-6;
        let v0 = self.mps_ach[i - 1];
        if v0 > tol && !self.impose_coast[i] && self.should_impose_coast(i) {
            let ct = self.generate_coast_trajectory(i);
            if ct.found_trajectory {
                let d = ct.distance_to_stop_via_coast_m;
                if d < 0.0 {
                    for idx in i..self.cyc0.mps.len() {
                        self.impose_coast[idx] = false;
                    }
                } else {
                    self.apply_coast_trajectory(ct);
                }
                if !self.sim_params.coast_allow_passing {
                    self.prevent_collisions(i, None);
                }
            }
        }
        if !self.impose_coast[i] {
            if !self.sim_params.follow_allow {
                let i_i32 = i32::try_from(i).ok();
                let target_idx = match i_i32 {
                    Some(v) => Some(v - self.coast_delay_index[i]),
                    None => None,
                };
                let target_idx = match target_idx {
                    Some(ti) => {
                        if ti < 0 {
                            Some(0)
                        } else {
                            usize::try_from(ti).ok()
                        }
                    }
                    None => None,
                };
                match target_idx {
                    Some(ti) => {
                        self.cyc.mps[i] = self.cyc0.mps[cmp::min(ti, self.cyc0.mps.len() - 1)];
                    }
                    None => (),
                }
            }
            return Ok(());
        }
        let v1_traj = self.cyc.mps[i];
        if v0 > self.sim_params.coast_brake_start_speed_m_per_s {
            if self.sim_params.coast_allow_passing {
                // we could be coasting downhill so could in theory go to a higher speed
                // since we can pass, allow vehicle to go up to max coasting speed (m/s)
                // the solver will show us what we can actually achieve
                self.cyc.mps[i] = self.sim_params.coast_max_speed_m_per_s;
            } else {
                self.cyc.mps[i] = min(v1_traj, self.sim_params.coast_max_speed_m_per_s);
            }
        }
        // Solve for the actual coasting speed
        if let Err(message) = self.solve_step(i) {
            return Err(
                "call to `solve_step_rust` failed within `set_coast_speed`: ".to_string()
                    + &message,
            );
        }
        self.newton_iters[i] = 0; // reset newton iters
        self.cyc.mps[i] = self.mps_ach[i];
        let accel_proposed = (self.cyc.mps[i] - self.cyc.mps[i - 1]) / self.cyc.dt_s_at_i(i);
        if self.cyc.mps[i] < tol {
            for idx in i..self.cyc0.mps.len() {
                self.impose_coast[idx] = false;
            }
            self.set_coast_delay(i);
            self.cyc.mps[i] = 0.0;
            return Ok(());
        }
        if (self.cyc.mps[i] - v1_traj).abs() > tol {
            let mut adjusted_current_speed = false;
            let brake_speed_start_tol_m_per_s = 0.1;
            if self.cyc.mps[i]
                < (self.sim_params.coast_brake_start_speed_m_per_s - brake_speed_start_tol_m_per_s)
            {
                let v1_before = self.cyc.mps[i];
                let (_, num_steps) = self.cyc.modify_with_braking_trajectory(
                    self.sim_params.coast_brake_accel_m_per_s2,
                    i,
                    None,
                );
                for idx in i..self.cyc.time_s.len() {
                    self.impose_coast[idx] = idx < (i + num_steps);
                }
                let v1_after = self.cyc.mps[i];
                assert_ne!(v1_before, v1_after);
                adjusted_current_speed = true;
            } else {
                let (traj_found, traj_n, traj_jerk_m_per_s3, traj_accel_m_per_s2) = self
                    .calc_next_rendezvous_trajectory(
                        i,
                        self.sim_params.coast_brake_accel_m_per_s2,
                        min(accel_proposed, 0.0),
                    );
                if traj_found {
                    // adjust cyc to perform the trajectory
                    let final_speed_m_per_s = self.cyc.modify_by_const_jerk_trajectory(
                        i,
                        traj_n,
                        traj_jerk_m_per_s3,
                        traj_accel_m_per_s2,
                    );
                    for idx in i..self.cyc0.mps.len() {
                        self.impose_coast[idx] = idx < (i + traj_n);
                    }
                    adjusted_current_speed = true;
                    let i_for_brake = i + traj_n;
                    if (final_speed_m_per_s - self.sim_params.coast_brake_start_speed_m_per_s).abs()
                        < 0.1
                    {
                        let (_, num_steps) = self.cyc.modify_with_braking_trajectory(
                            self.sim_params.coast_brake_accel_m_per_s2,
                            i_for_brake,
                            None,
                        );
                        for idx in i_for_brake..self.cyc0.mps.len() {
                            self.impose_coast[idx] = idx < i_for_brake + num_steps;
                        }
                        adjusted_current_speed = true;
                    } else {
                        eprintln!("WARNING! final_speed_m_per_s not close to coast_brake_start_speed for i = {}", i);
                        eprintln!("... final_speed_m_per_s = {}", final_speed_m_per_s);
                        eprintln!(
                            "... self.sim_params.coast_brake_start_speed_m_per_s = {}",
                            self.sim_params.coast_brake_start_speed_m_per_s
                        );
                        eprintln!("... i_for_brake = {}", i_for_brake);
                        eprintln!("... traj_n = {}", traj_n);
                    }
                }
            }
            if adjusted_current_speed {
                if !self.sim_params.coast_allow_passing {
                    self.prevent_collisions(i, None);
                }
                if let Err(message) = self.solve_step(i) {
                    return Err(
                        "call to `solve_step_rust` failed within `set_coast_speed`: ".to_string()
                            + &message,
                    );
                }
                self.newton_iters[i] = 0; // reset newton iters
                self.cyc.mps[i] = self.mps_ach[i];
            }
        }
        Ok(())
    }

    /// Sets scalar variables that can be calculated after a cycle is run.
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars(&mut self) -> Result<(), String> {
        let mut res = || -> Result<(), ()> {
            if self.fs_kwh_out_ach.sum() == 0.0 {
                self.mpgge = 0.0;
            } else {
                self.mpgge =
                    self.dist_mi.sum() / (self.fs_kwh_out_ach.sum() / self.props.kwh_per_gge);
            }

            self.roadway_chg_kj = (self.roadway_chg_kw_out_ach.clone() * self.cyc.dt_s()).sum();
            self.ess_dischg_kj =
                -1.0 * (self.soc[self.soc.len() - 1] - self.soc[0]) * self.veh.ess_max_kwh * 3.6e3;
            let dist_mi = self.dist_mi.sum();
            self.battery_kwh_per_mi = if dist_mi > 0.0 {
                (self.ess_dischg_kj / 3.6e3) / dist_mi
            } else {
                0.0
            };
            self.electric_kwh_per_mi = if dist_mi > 0.0 {
                ((self.roadway_chg_kj + self.ess_dischg_kj) / 3.6e3) / dist_mi
            } else {
                0.0
            };
            self.fuel_kj = (self.fs_kw_out_ach.clone() * self.cyc.dt_s()).sum();

            if (self.fuel_kj + self.roadway_chg_kj) == 0.0 {
                self.ess2fuel_kwh = 1.0
            } else {
                self.ess2fuel_kwh = self.ess_dischg_kj / (self.fuel_kj + self.roadway_chg_kj);
            }

            // energy audit calcs
            self.drag_kj = (self.drag_kw.clone() * self.cyc.dt_s()).sum();
            self.ascent_kj = (self.ascent_kw.clone() * self.cyc.dt_s()).sum();
            self.rr_kj = (self.rr_kw.clone() * self.cyc.dt_s()).sum();

            for i in 1..self.cyc.time_s.len() {
                if self.veh.ess_max_kw == 0.0 || self.veh.ess_max_kwh == 0.0 {
                    self.ess_loss_kw[i] = 0.0;
                } else if self.ess_kw_out_ach[i] < 0.0 {
                    self.ess_loss_kw[i] = -self.ess_kw_out_ach[i]
                        - (-self.ess_kw_out_ach[i] * self.veh.ess_round_trip_eff.sqrt());
                } else {
                    self.ess_loss_kw[i] = self.ess_kw_out_ach[i]
                        * (1.0 / self.veh.ess_round_trip_eff.sqrt())
                        - self.ess_kw_out_ach[i];
                }
            }

            self.brake_kj = (self.cyc_fric_brake_kw.clone() * self.cyc.dt_s()).sum();
            self.trans_kj = ((self.trans_kw_in_ach.clone() - self.trans_kw_out_ach.clone())
                * self.cyc.dt_s())
            .sum();
            self.mc_kj = ((self.mc_elec_kw_in_ach.clone() - self.mc_mech_kw_out_ach.clone())
                * self.cyc.dt_s())
            .sum();
            self.ess_eff_kj = (self.ess_loss_kw.clone() * self.cyc.dt_s()).sum();
            self.aux_kj = (self.aux_in_kw.clone() * self.cyc.dt_s()).sum();
            self.fc_kj =
                ((self.fc_kw_in_ach.clone() - self.fc_kw_out_ach.clone()) * self.cyc.dt_s()).sum();

            self.net_kj = self.drag_kj
                + self.ascent_kj
                + self.rr_kj
                + self.brake_kj
                + self.trans_kj
                + self.mc_kj
                + self.ess_eff_kj
                + self.aux_kj
                + self.fc_kj;

            self.ke_kj = 0.5
                * self.veh.veh_kg
                * (self.mps_ach[0].powf(2.0) - self.mps_ach[self.mps_ach.len() - 1].powf(2.0))
                / 1_000.0;

            self.energy_audit_error =
                ((self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj)
                    - self.net_kj)
                    / (self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj);

            if (self.energy_audit_error.abs() > self.sim_params.energy_audit_error_tol)
                && self.sim_params.verbose
            {
                println!("Warning: There is a problem with conservation of energy.");
                println!("Energy Audit Error: {:.5}", self.energy_audit_error);
            }
            for i in 1..self.cyc.dt_s().len() {
                self.accel_kw[i] = self.veh.veh_kg / (2.0 * self.cyc.dt_s_at_i(i))
                    * (self.mps_ach[i].powf(2.0) - self.mps_ach[i - 1].powf(2.0))
                    / 1_000.0;
            }

            self.trace_miss = false;
            let dist_m = self.cyc0.dist_m().sum();
            self.trace_miss_dist_frac = if dist_m > 0.0 {
                (self.dist_m.sum() - self.cyc0.dist_m().sum()).abs() / dist_m
            } else {
                0.0
            };
            self.trace_miss_time_frac = (self.cyc.time_s[self.cyc.time_s.len() - 1]
                - self.cyc0.time_s[self.cyc0.time_s.len() - 1])
                / self.cyc0.time_s[self.cyc0.time_s.len() - 1];

            if !self.sim_params.missed_trace_correction {
                if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol {
                    self.trace_miss = true;
                    if self.sim_params.verbose {
                        println!(
                            "Warning: Trace miss distance fraction: {:.5}",
                            self.trace_miss_dist_frac
                        );
                        println!(
                            "exceeds tolerance of: {:.5}",
                            self.sim_params.trace_miss_dist_tol
                        );
                    }
                }
            } else if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol {
                self.trace_miss = true;
                if self.sim_params.verbose {
                    println!(
                        "Warning: Trace miss time fraction: {:.5}",
                        self.trace_miss_time_frac
                    );
                    println!(
                        "exceeds tolerance of: {:.5}",
                        self.sim_params.trace_miss_time_tol
                    );
                }
            }

            self.trace_miss_speed_mps =
                ndarrmax(&(self.mps_ach.clone() - self.cyc.mps.clone()).map(|x| x.abs()));
            if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol {
                self.trace_miss = true;
                if self.sim_params.verbose {
                    println!(
                        "Warning: Trace miss speed [m/s]: {:.5}",
                        self.trace_miss_speed_mps
                    );
                    println!(
                        "exceeds tolerance of: {:.5}",
                        self.sim_params.trace_miss_speed_mps_tol
                    );
                }
            }
            Ok(())
        };

        if let Err(()) = res() {
            Err("`set_post_scalars_rust` failed".to_string())
        } else {
            Ok(())
        }
    }
}
