//! Module containing implementations for [simdrive](crate::simdrive).

use crate::cycle::{RustCycle, RustCycleCache};
use crate::imports::*;
use crate::params;
use crate::simdrive::{RustSimDrive, RustSimDriveParams};
use crate::utils::{arrmax, first_grtr, max, min, ndarrmax, ndarrmin};
use crate::vehicle::*;

pub struct RendezvousTrajectory {
    pub found_trajectory: bool,
    pub idx: usize,
    pub n: usize,
    pub full_brake_steps: usize,
    pub jerk_m_per_s3: f64,
    pub accel0_m_per_s2: f64,
    pub accel_spread: f64,
}

pub struct CoastTrajectory {
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
        let idm_target_speed_m_per_s = Array::zeros(cyc_len);
        let cyc0_cache = RustCycleCache::new(&cyc0);
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
            idm_target_speed_m_per_s,
            cyc0_cache,
        }
    }

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
        self.idm_target_speed_m_per_s = Array::zeros(cyc_len);
        self.cyc0_cache = self.cyc0.build_cache();
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
    ) -> Result<(), anyhow::Error> {
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
                            ess_2fuel_kwh = ((self.soc[0] - self.soc.last().unwrap())
                                * self.veh.ess_max_kwh
                                * 3.6e3
                                / (fuel_kj + roadway_chg_kj))
                                .abs();
                        } else {
                            ess_2fuel_kwh = 0.0;
                        }
                        init_soc = min(
                            self.veh.max_soc,
                            max(self.veh.min_soc, *self.soc.last().unwrap()),
                        );
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

    pub fn sim_drive_accel(
        &mut self,
        init_soc: Option<f64>,
        aux_in_kw_override: Option<Array1<f64>>,
    ) -> Result<(), anyhow::Error> {
        // Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
        let init_soc_auto: f64 = match self.veh.veh_pt_type.as_str() {
            // If no EV / Hybrid components, no SOC considerations.
            CONV => (self.veh.max_soc + self.veh.min_soc) / 2.0,
            HEV => (self.veh.max_soc + self.veh.min_soc) / 2.0,
            // If EV, initializing initial SOC to maximum SOC.
            _ => self.veh.max_soc,
        };
        let init_soc = init_soc.unwrap_or(init_soc_auto);
        self.walk(init_soc, aux_in_kw_override)?;
        self.set_post_scalars()
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
    ) -> Result<(), anyhow::Error> {
        self.init_for_step(init_soc, aux_in_kw_override)?;
        while self.i < self.cyc.time_s.len() {
            self.step()?;
        }

        // TODO: uncomment and implement logging
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
    ) -> Result<(), anyhow::Error> {
        ensure!(
            self.veh.veh_pt_type == CONV
                || (self.veh.min_soc..=self.veh.max_soc).contains(&init_soc),
            "provided init_soc={} is outside range min_soc={} to max_soc={}",
            init_soc,
            self.veh.min_soc,
            self.veh.max_soc
        );

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
            || self.sim_params.idm_allow
            || self.sim_params.coast_allow
        {
            self.cyc = self.cyc0.clone(); // reset the cycle in case it has been manipulated
        }
        self.i = 1; // time step counter
        Ok(())
    }

    /// Step through 1 time step.
    pub fn step(&mut self) -> Result<(), anyhow::Error> {
        if self.sim_params.idm_allow {
            self.idm_target_speed_m_per_s[self.i] =
                match &self.sim_params.idm_v_desired_in_m_per_s_by_distance_m {
                    Some(vtgt_by_dist) => {
                        let mut found_v_target = vtgt_by_dist[0].1;
                        let current_d = self.cyc.dist_m().slice(s![0..self.i]).sum();
                        for (d, v_target) in vtgt_by_dist {
                            if &current_d >= d {
                                found_v_target = *v_target;
                            } else {
                                break;
                            }
                        }
                        found_v_target
                    }
                    None => self.sim_params.idm_v_desired_m_per_s,
                };
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
        if self.sim_params.coast_allow || self.sim_params.idm_allow {
            self.cyc.mps[self.i] = self.mps_ach[self.i];
            self.cyc.grade[self.i] = self.lookup_grade_for_step(self.i, None);
        }

        self.i += 1; // increment time step counter
        Ok(())
    }

    /// Perform all the calculations to solve 1 time step.
    pub fn solve_step(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
    pub fn set_misc_calcs(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
    }

    /// Sets component limits for time step 'i'
    /// Arguments
    /// ------------
    /// i: index of time step
    /// initSoc: initial SOC for electrified vehicles
    pub fn set_comp_lims(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
        self.cur_max_avail_elec_kw[i] = min(self.cur_max_elec_kw[i], self.veh.mc_max_elec_in_kw);

        if self.cur_max_elec_kw[i] > 0.0 {
            // limit power going into e-machine controller to
            if self.cur_max_avail_elec_kw[i] == arrmax(&self.veh.mc_kw_in_array) {
                self.mc_elec_in_lim_kw[i] = min(
                    *self.veh.mc_kw_out_array.last().unwrap(),
                    self.veh.mc_max_kw,
                );
            } else {
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[first_grtr(
                        &self.veh.mc_kw_in_array,
                        min(
                            arrmax(&self.veh.mc_kw_in_array) * 0.9999,
                            self.cur_max_avail_elec_kw[i],
                        ),
                    )
                    .unwrap()
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
            self.cur_max_mc_elec_kw_in[i] =
                self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array.last().unwrap();
        } else {
            self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i]
                / self.veh.mc_full_eff_array[cmp::max(
                    1,
                    first_grtr(
                        &self.veh.mc_kw_out_array,
                        min(self.veh.mc_max_kw * 0.9999, self.cur_max_mc_kw_out[i]),
                    )
                    .unwrap()
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
        } else if self.veh.mc_max_kw == self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]
        {
            self.ess_lim_mc_regen_kw[i] = min(
                self.veh.mc_max_kw,
                self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array.last().unwrap(),
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
                                self.veh.mc_max_kw * 0.9999,
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
    }

    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
                        * ((self.cyc.mph_at_i(i) + self.mps_ach[i - 1] * params::MPH_PER_MPS)
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
        self.cyc_fric_brake_kw[i] = -min(self.cyc_regen_brake_kw[i] + self.cyc_whl_kw_req[i], 0.0);
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
    }

    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
                let ascent1 = 0.5 * self.props.a_grav_mps2 * grade.atan().sin() * self.veh.veh_kg;
                let accel0 =
                    -0.5 * self.veh.veh_kg * self.mps_ach[i - 1].powf(2.0) / self.cyc.dt_s_at_i(i);
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
                    let xi = xs[xs.len() - 1] * (1.0 - g) - g * bs[xs.len() - 1] / ms[xs.len() - 1];
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
            self.set_power_calcs(i)?;
        }

        self.mph_ach[i] = self.mps_ach[i] * params::MPH_PER_MPS;
        self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s_at_i(i);
        self.dist_mi[i] = self.dist_m[i] * 1.0 / params::M_PER_MI;
        Ok(())
    }

    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
                self.mc_elec_in_kw_for_max_fc_eff[i] =
                    -self.fc_kw_gap_fr_eff[i] / self.veh.mc_full_eff_array.last().unwrap();
            } else {
                self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i]
                    / self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_out_array,
                            min(self.veh.mc_max_kw * 0.9999, self.fc_kw_gap_fr_eff[i]),
                        )
                        .unwrap()
                            - 1,
                    )];
            }
        } else if self.fc_kw_gap_fr_eff[i] == self.veh.mc_max_kw {
            self.mc_elec_in_kw_for_max_fc_eff[i] = *self.veh.mc_kw_in_array.last().unwrap();
        } else {
            self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[first_grtr(
                &self.veh.mc_kw_out_array,
                min(self.veh.mc_max_kw * 0.9999, self.fc_kw_gap_fr_eff[i]),
            )
            .unwrap_or(0)
                - 1];
        }
        if self.veh.no_elec_sys {
            self.elec_kw_req_4ae[i] = 0.0;
        } else if self.trans_kw_in_ach[i] > 0.0 {
            if self.trans_kw_in_ach[i] == self.veh.mc_max_kw {
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i]
                    / self.veh.mc_full_eff_array.last().unwrap()
                    + self.aux_in_kw[i];
            } else {
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i]
                    / self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_out_array,
                            min(self.veh.mc_max_kw * 0.9999, self.trans_kw_in_ach[i]),
                        )
                        .unwrap()
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
                && (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] || self.veh.fc_max_kw == 0.0);
        } else {
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i - 1]
                && (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i]
                && (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] || self.veh.fc_max_kw == 0.0)
                && ((self.cyc.mph_at_i(i) - 1e-6) <= self.veh.mph_fc_on || self.veh.charging_on)
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
                self.desired_ess_kw_out_for_ae[i] =
                    self.trans_kw_in_ach[i] + self.aux_in_kw[i] - self.cur_max_roadway_chg_kw[i];
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
    }

    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_forced_state_rust(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
            // fc forced on because:
            // - it was on in the previous time step and hasn't been on long enough
            // - it can't power everything on it's own
            self.fc_forced_state[i] = 1;
            self.mc_mech_kw_4forced_fc[i] = 0.0;
        } else if self.trans_kw_in_ach[i] < 0.0 {
            // not forced on.  transmission needs negative power (i.e. regen)
            self.fc_forced_state[i] = 2;
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i];
        } else if self.veh.max_fc_eff_kw() == self.trans_kw_in_ach[i] {
            // if fc power at which maximum efficiency is achieved equals the transmission input power
            // fc possibly (???) forced on to be more efficient
            // this seems unlikely to ever happen
            self.fc_forced_state[i] = 3;
            self.mc_mech_kw_4forced_fc[i] = 0.0;
        } else if self.veh.idle_fc_kw > self.trans_kw_in_ach[i] && self.accel_kw[i] >= 0.0 {
            // accelerating but idle power is greater than accel power needed by trans
            self.fc_forced_state[i] = 4;
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.idle_fc_kw;
        } else if self.veh.max_fc_eff_kw() > self.trans_kw_in_ach[i] {
            // if fc power at which maximum efficiency is achieved exceeds the transmission input power
            self.fc_forced_state[i] = 5;
            self.mc_mech_kw_4forced_fc[i] = 0.0;
        } else {
            // fc not forced on in previous time step or
            // transmission is not in braking state or
            // transmission input power is not exactly most efficient point on fc map or
            // acceleration is not >= 0 and/or idle power is not > trans input power or
            // fc peak eff point in map is <= trans input power
            self.fc_forced_state[i] = 6;
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.max_fc_eff_kw();
        }
        Ok(())
    }

    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
                self.mc_kw_if_fc_req[i] =
                    self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array.last().unwrap();
            } else {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                    * self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_in_array,
                            min(
                                arrmax(&self.veh.mc_kw_in_array) * 0.9999,
                                self.mc_elec_kw_in_if_fc_req[i],
                            ),
                        )
                        .unwrap()
                            - 1,
                    )]
            }
        } else if -self.mc_elec_kw_in_if_fc_req[i] == arrmax(&self.veh.mc_kw_in_array) {
            self.mc_kw_if_fc_req[i] =
                self.mc_elec_kw_in_if_fc_req[i] / self.veh.mc_full_eff_array.last().unwrap();
        } else {
            self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i]
                / self.veh.mc_full_eff_array[cmp::max(
                    1,
                    first_grtr(
                        &self.veh.mc_kw_in_array,
                        min(
                            arrmax(&self.veh.mc_kw_in_array) * 0.9999,
                            -self.mc_elec_kw_in_if_fc_req[i],
                        ),
                    )
                    .unwrap()
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
            self.mc_mech_kw_out_ach[i] = max(self.min_mc_kw_2help_fc[i], self.mc_kw_if_fc_req[i])
        }

        if self.mc_mech_kw_out_ach[i] == 0.0 {
            self.mc_elec_kw_in_ach[i] = 0.0;
        } else if self.mc_mech_kw_out_ach[i] < 0.0 {
            if -self.mc_mech_kw_out_ach[i] == arrmax(&self.veh.mc_kw_in_array) {
                self.mc_elec_kw_in_ach[i] =
                    self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array.last().unwrap()
            } else {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                    * self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(
                            &self.veh.mc_kw_in_array,
                            min(
                                arrmax(&self.veh.mc_kw_in_array) * 0.9999,
                                -self.mc_mech_kw_out_ach[i],
                            ),
                        )
                        .unwrap()
                            - 1,
                    )];
            }
        } else if self.veh.mc_max_kw == self.mc_mech_kw_out_ach[i] {
            self.mc_elec_kw_in_ach[i] =
                self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array.last().unwrap()
        } else {
            self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i]
                / self.veh.mc_full_eff_array[cmp::max(
                    1,
                    first_grtr(
                        &self.veh.mc_kw_out_array,
                        min(self.veh.mc_max_kw * 0.9999, self.mc_mech_kw_out_ach[i]),
                    )
                    .unwrap()
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
                self.ess_kw_out_ach[i] =
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i];
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
    }

    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_power(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
        .unwrap()
            - 1]
            != 0.0
        {
            self.fc_kw_in_ach[i] = self.fc_kw_out_ach[i]
                / (self.veh.fc_eff_array[first_grtr(
                    &self.veh.fc_kw_out_array,
                    min(self.fc_kw_out_ach[i], self.veh.fc_max_kw),
                )
                .unwrap()
                    - 1]);
        } else {
            self.fc_kw_in_ach[i] = 0.0
        }

        self.fs_kw_out_ach[i] = self.fc_kw_in_ach[i];

        self.fs_kwh_out_ach[i] = self.fs_kw_out_ach[i] * self.cyc.dt_s_at_i(i) / 3.6e3;
        Ok(())
    }

    /// Sets scalar variables that can be calculated after a cycle is run.
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars(&mut self) -> Result<(), anyhow::Error> {
        if self.fs_kwh_out_ach.sum() == 0.0 {
            self.mpgge = 0.0;
        } else {
            self.mpgge = self.dist_mi.sum() / (self.fs_kwh_out_ach.sum() / self.props.kwh_per_gge);
        }

        self.roadway_chg_kj = (self.roadway_chg_kw_out_ach.clone() * self.cyc.dt_s()).sum();
        self.ess_dischg_kj =
            -1.0 * (self.soc.last().unwrap() - self.soc[0]) * self.veh.ess_max_kwh * 3.6e3;
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
            * (self.mps_ach.first().unwrap().powf(2.0) - self.mps_ach.last().unwrap().powf(2.0))
            / 1_000.0;

        self.energy_audit_error =
            ((self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj) - self.net_kj)
                / (self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj);

        if self.energy_audit_error.abs() > self.sim_params.energy_audit_error_tol {
            log::warn!(
                "problem detected with conservation of energy; \
                    energy audit error: {:.5}",
                self.energy_audit_error
            );
        }
        for i in 1..self.cyc.len() {
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
        self.trace_miss_time_frac = (self.cyc.time_s.last().unwrap()
            - self.cyc0.time_s.last().unwrap())
            / self.cyc0.time_s.last().unwrap();

        if !self.sim_params.missed_trace_correction {
            if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol {
                self.trace_miss = true;
                log::warn!(
                    "trace miss distance fraction {:.5} exceeds tolerance of {:.5}",
                    self.trace_miss_dist_frac,
                    self.sim_params.trace_miss_dist_tol
                );
            }
        } else if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol {
            self.trace_miss = true;
            log::warn!(
                "trace miss time fraction {:.5} exceeds tolerance of {:.5}",
                self.trace_miss_time_frac,
                self.sim_params.trace_miss_time_tol
            );
        }

        self.trace_miss_speed_mps =
            ndarrmax(&(self.mps_ach.clone() - self.cyc.mps.clone()).map(|x| x.abs()));
        if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol {
            self.trace_miss = true;
            log::warn!(
                "trace miss speed {:.5} m/s exceeds tolerance of {:.5} m/s",
                self.trace_miss_speed_mps,
                self.sim_params.trace_miss_speed_mps_tol
            );
        }
        Ok(())
    }
}
