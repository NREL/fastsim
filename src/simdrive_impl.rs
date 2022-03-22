use ndarray::{Array, array, s};
use std::cmp;
use super::utils::{arrmax, first_grtr, min, max, ndarrmin};
use super::vehicle::*;
use super::params;

use super::simdrive::RustSimDrive;


impl RustSimDrive {

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
    ///         Default of np.zeros(1) causes veh.aux_kw to be used. If zero is actually
    ///         desired as an override, either set veh.aux_kw = 0 before instantiaton of
    ///         SimDrive*, or use `np.finfo(np.float64).tiny` for auxInKw[-1]. Setting
    ///         the final value to non-zero prevents override mechanism.
    pub fn walk(&mut self, init_soc:f64) {
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
            self.step()
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
    pub fn step(&mut self) {
        self.solve_step_rust(self.i);

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
    pub fn solve_step_rust(&mut self, i:usize) {
        self.set_misc_calcs_rust(i);
        self.set_comp_lims_rust(i);
        self.set_power_calcs_rust(i);
        self.set_ach_speed_rust(i);
        self.set_hybrid_cont_calcs_rust(i);
        self.set_fc_forced_state_rust(i);
        self.set_hybrid_cont_decisions_rust(i);
        self.set_fc_ess_power_rust(i);
    }


    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    pub fn set_misc_calcs_rust(&mut self, i:usize) {
        // if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        if self.aux_in_kw.slice(s![i..]).iter().all(|&x| x == 0.0) {
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
    pub fn set_comp_lims_rust(&mut self, i:usize) {
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
        self.fc_fs_lim_kw[i] = self.veh.max_fuel_conv_kw;
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
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[self.veh.mc_kw_out_array.len() - 1],
                    self.veh.max_motor_kw);
            }
            else {
                self.mc_elec_in_lim_kw[i] = min(
                    self.veh.mc_kw_out_array[first_grtr(
                        &self.veh.mc_kw_in_array, min(
                            arrmax(&self.veh.mc_kw_in_array) - 0.01,
                            self.cur_max_avail_elec_kw[i])
                    ).unwrap_or(0) - 1 as usize],
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
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] /
                    self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1];
            } else {
                self.cur_max_mc_elec_kw_in[i] = self.cur_max_mc_kw_out[i] / self.veh.mc_full_eff_array[cmp::max(
                    1,
                    first_grtr(
                        &self.veh.mc_kw_out_array, min(
                            self.veh.max_motor_kw - 0.01, self.cur_max_mc_kw_out[i])).unwrap_or(0) - 1
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
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw, self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len() - 1]);
            }
            else {
                self.ess_lim_mc_regen_kw[i] = min(
                    self.veh.max_motor_kw,
                    self.cur_max_ess_chg_kw[i] / self.veh.mc_full_eff_array[
                        cmp::max(1,
                            first_grtr(
                                &self.veh.mc_kw_out_array, min(
                                    self.veh.max_motor_kw - 0.01,
                                    self.cur_max_ess_chg_kw[i] - self.cur_max_roadway_chg_kw[i]
                                )
                            ).unwrap_or(0) - 1
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
    pub fn set_power_calcs_rust(&mut self, i:usize) {
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
    pub fn set_ach_speed_rust(&mut self, i:usize) {
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
                // let end = ;
                let xi = xs[xs.len() - 1] * (1.0 - g) - g * bs[xs.len() - 1] / ms[xs.len() - 1];
                let yi = t3 * xi.powf(3.0) + t2 * xi.powf(2.0) + t1 * xi + t0;
                let mi = 3.0 * t3 * xi.powf(2.0) + 2.0 * t2 * xi + t1;
                let bi = yi - xi * mi;
                xs.push(xi);
                ys.push(yi);
                ms.push(mi);
                bs.push(bi);
                converged = ((xs[xs.len()-1] - xs[xs.len()-2]) / xs[xs.len()-2]).abs() < xtol;
                iterate += 1;
            }

            self.newton_iters[i] = iterate;

            let _ys = Array::from_vec(ys).map(|x| x.abs());
            self.mps_ach[i] = xs[_ys.iter().position(|&x| x == ndarrmin(&_ys)).unwrap()];
        }

        self.set_power_calcs(i);

        self.mph_ach[i] = self.mps_ach[i] * params::MPH_PER_MPS;
        self.dist_m[i] = self.mps_ach[i] * self.cyc.dt_s()[i];
        self.dist_mi[i] = self.dist_m[i] * 1.0 / params::M_PER_MI;
    }

    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs_rust(&mut self, i:usize) {
        if self.veh.no_elec_sys {
            self.regen_buff_soc[i] = 0.0;
        }
        else if self.veh.charging_on {
            self.regen_buff_soc[i] = max(
                self.veh.max_soc - (self.veh.max_regen_kwh() / self.veh.max_ess_kwh),
                (self.veh.max_soc + self.veh.min_soc) / 2.0);
        }
        else {
            self.regen_buff_soc[i] = max(
                (self.veh.max_ess_kwh * self.veh.max_soc -
                    0.5 * self.veh.veh_kg * (self.cyc.mps[i].powf(2.0)) * (1.0 / 1_000.0) * (1.0 / 3_600.0) *
                    self.veh.mc_peak_eff() * self.veh.max_regen) / self.veh.max_ess_kwh,
                self.veh.min_soc
            );

            self.ess_regen_buff_dischg_kw[i] = min(self.cur_max_ess_kw_out[i], max(
                0.0, (self.soc[i-1] - self.regen_buff_soc[i]) * self.veh.max_ess_kwh * 3_600.0 / self.cyc.dt_s()[i]));

            self.max_ess_regen_buff_chg_kw[i] = min(max(
                    0.0,
                    (self.regen_buff_soc[i] - self.soc[i-1]) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s()[i]),
                self.cur_max_ess_chg_kw[i]
            );
        }
        if self.veh.no_elec_sys {
            self.accel_buff_soc[i] = 0.0;
        }
        else {
            self.accel_buff_soc[i] = min(
                max(
                    ((self.veh.max_accel_buffer_mph / params::MPH_PER_MPS).powf(2.0) - self.cyc.mps[i].powf(2.0)) /
                    (self.veh.max_accel_buffer_mph / params::MPH_PER_MPS).powf(2.0) * min(
                        self.veh.max_accel_buffer_perc_of_useable_soc * (self.veh.max_soc - self.veh.min_soc),
                        self.veh.max_regen_kwh() / self.veh.max_ess_kwh
                    ) * self.veh.max_ess_kwh / self.veh.max_ess_kwh + self.veh.min_soc,
                    self.veh.min_soc
                ),
                self.veh.max_soc
                );

            self.ess_accel_buff_chg_kw[i] = max(
                0.0, (self.accel_buff_soc[i] - self.soc[i-1]) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s()[i]);
            self.max_ess_accell_buff_dischg_kw[i] = min(
                max(
                    0.0,
                    (self.soc[i-1] - self.accel_buff_soc[i]) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s()[i]),
                self.cur_max_ess_kw_out[i]
            );
        }
        if self.regen_buff_soc[i] < self.accel_buff_soc[i] {
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    (self.soc[i-1] - (self.regen_buff_soc[i] + self.accel_buff_soc[i]) / 2.0) * self.veh.max_ess_kwh * 3.6e3 / self.cyc.dt_s()[i],
                    self.cur_max_ess_kw_out[i]
                ),
                -self.cur_max_ess_chg_kw[i]
            );
        } else if self.soc[i-1] > self.regen_buff_soc[i] {
            self.ess_accel_regen_dischg_kw[i] = max(
                min(
                    self.ess_regen_buff_dischg_kw[i],
                    self.cur_max_ess_kw_out[i]),
                -self.cur_max_ess_chg_kw[i]
            );
        } else if self.soc[i-1] < self.accel_buff_soc[i] {
            self.ess_accel_regen_dischg_kw[i] = max(
                min(-1.0 * self.ess_accel_buff_chg_kw[i], self.cur_max_ess_kw_out[i]), -self.cur_max_ess_chg_kw[i]);
        } else {
            self.ess_accel_regen_dischg_kw[i] = max(
                min(0.0, self.cur_max_ess_kw_out[i]), -self.cur_max_ess_chg_kw[i]);
        }
        self.fc_kw_gap_fr_eff[i] = (self.trans_kw_out_ach[i] - self.veh.max_fc_eff_kw()).abs();

        if self.veh.no_elec_sys {
            self.mc_elec_in_kw_for_max_fc_eff[i] = 0.0;
        }
        else if self.trans_kw_out_ach[i] < self.veh.max_fc_eff_kw() {
            if self.fc_kw_gap_fr_eff[i] == self.veh.max_motor_kw {
                self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i] / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len()-1];
            }
            else {
                self.mc_elec_in_kw_for_max_fc_eff[i] = -self.fc_kw_gap_fr_eff[i] /
                    self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(&self.veh.mc_kw_out_array,
                            min(self.veh.max_motor_kw - 0.01, self.fc_kw_gap_fr_eff[i])).unwrap_or(0) - 1)];
            }
        }
        else {
            if self.fc_kw_gap_fr_eff[i] == self.veh.max_motor_kw {
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[
                    self.veh.mc_kw_in_array.len() - 1];
            }
            else {
                self.mc_elec_in_kw_for_max_fc_eff[i] = self.veh.mc_kw_in_array[
                    first_grtr(
                        &self.veh.mc_kw_out_array,
                            min(self.veh.max_motor_kw - 0.01, self.fc_kw_gap_fr_eff[i])).unwrap_or(0) - 1];
            }
        }
        if self.veh.no_elec_sys {
            self.elec_kw_req_4ae[i] = 0.0;
        }
        else if self.trans_kw_in_ach[i] > 0.0 {
            if self.trans_kw_in_ach[i] == self.veh.max_motor_kw {
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i] / self.veh.mc_full_eff_array[
                    self.veh.mc_full_eff_array.len()-1] + self.aux_in_kw[i];
            }
            else {
                self.elec_kw_req_4ae[i] = self.trans_kw_in_ach[i] /
                    self.veh.mc_full_eff_array[cmp::max(
                        1,
                        first_grtr(&self.veh.mc_kw_out_array,
                            min(self.veh.max_motor_kw - 0.01, self.trans_kw_in_ach[i])).unwrap_or(0) - 1)] + self.aux_in_kw[i]
                ;
            }
        }
        else {
            self.elec_kw_req_4ae[i] = 0.0;
        }

        self.prev_fc_time_on[i] = self.fc_time_on[i-1];

        // some conditions in the following if statement have a buffer of 1e-6 to prevent false positives/negatives because these have been encountered in practice.
        if self.veh.max_fuel_conv_kw == 0.0 {
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] &&
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] &&
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] || self.veh.max_fuel_conv_kw == 0.0);
        }
        else {
            self.can_pwr_all_elec[i] = self.accel_buff_soc[i] < self.soc[i-1] &&
                (self.trans_kw_in_ach[i] - 1e-6) <= self.cur_max_mc_kw_out[i] &&
                (self.elec_kw_req_4ae[i] < self.cur_max_elec_kw[i] || self.veh.max_fuel_conv_kw == 0.0)
                && ((self.cyc.mph()[i] - 1e-6) <= self.veh.mph_fc_on || self.veh.charging_on) &&
                self.elec_kw_req_4ae[i] <= self.veh.kw_demand_fc_on;
        }
        if self.can_pwr_all_elec[i] {

            if self.trans_kw_in_ach[i] < self.aux_in_kw[i] {
                self.desired_ess_kw_out_for_ae[i] = self.aux_in_kw[i] + self.trans_kw_in_ach[i];
            }
            else if self.regen_buff_soc[i] < self.accel_buff_soc[i] {
                self.desired_ess_kw_out_for_ae[i] = self.ess_accel_regen_dischg_kw[i];
            }
            else if self.soc[i-1] > self.regen_buff_soc[i] {
                self.desired_ess_kw_out_for_ae[i] = self.ess_regen_buff_dischg_kw[i];
            }
            else if self.soc[i-1] < self.accel_buff_soc[i] {
                self.desired_ess_kw_out_for_ae[i] = -self.ess_accel_buff_chg_kw[i];
            }
            else {
                self.desired_ess_kw_out_for_ae[i] = self.trans_kw_in_ach[i] + self.aux_in_kw[i] - self.cur_max_roadway_chg_kw[i];
            }
        }
        else {
            self.desired_ess_kw_out_for_ae[i] = 0.0;
        }

        if self.can_pwr_all_elec[i] {
            self.ess_ae_kw_out[i] = max(
                -self.cur_max_ess_chg_kw[i],
                max(-self.max_ess_regen_buff_chg_kw[i],
                    max(min(0.0, self.cur_max_roadway_chg_kw[i] - self.trans_kw_in_ach[i] + self.aux_in_kw[i]),
                        min(self.cur_max_ess_kw_out[i], self.desired_ess_kw_out_for_ae[i])))
            );
        }
        else {
            self.ess_ae_kw_out[i] = 0.0;
        }

        self.er_ae_kw_out[i] = min(
            max(0.0, self.trans_kw_in_ach[i] + self.aux_in_kw[i] - self.ess_ae_kw_out[i]),
            self.cur_max_roadway_chg_kw[i]);
    }

    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_forced_state_rust(&mut self, i:usize) {
        // force fuel converter on if it was on in the previous time step, but only if fc
        // has not been on longer than minFcTimeOn
        if self.prev_fc_time_on[i] > 0.0 && self.prev_fc_time_on[i] < self.veh.min_fc_time_on - self.cyc.dt_s()[i]{
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
            self.mc_mech_kw_4forced_fc[i] = 0.0 } else if self.veh.idle_fc_kw > self.trans_kw_in_ach[i] && self.cyc_accel_kw[i] >= 0.0 {
            self.fc_forced_state[i] = 4;
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.idle_fc_kw;
        } else if self.veh.max_fc_eff_kw() > self.trans_kw_in_ach[i] {
            self.fc_forced_state[i] = 5;
            self.mc_mech_kw_4forced_fc[i] = 0.0;
        } else {
            self.fc_forced_state[i] = 6;
            self.mc_mech_kw_4forced_fc[i] = self.trans_kw_in_ach[i] - self.veh.max_fc_eff_kw();
        }
        // Ok(())
    }


    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions_rust(&mut self, i:usize) {
        if (-self.mc_elec_in_kw_for_max_fc_eff[i] - self.cur_max_roadway_chg_kw[i]) > 0.0 {
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] -
                self.cur_max_roadway_chg_kw[i]) * self.veh.ess_dischg_to_fc_max_eff_perc;
        } else {
            self.ess_desired_kw_4fc_eff[i] = (-self.mc_elec_in_kw_for_max_fc_eff[i] -
                self.cur_max_roadway_chg_kw[i]) * self.veh.ess_chg_to_fc_max_eff_perc;
        }

        if self.accel_buff_soc[i] > self.regen_buff_soc[i] {
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i],
                min(
                    self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], 
                    min(self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                        max(-self.cur_max_ess_chg_kw[i], self.ess_accel_regen_dischg_kw[i])))
                );
        } else if self.ess_regen_buff_dischg_kw[i] > 0.0 {
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i],
                min(self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], 
                    min(self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                        max(-self.cur_max_ess_chg_kw[i],
                            min(self.ess_accel_regen_dischg_kw[i],
                                min(self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i],
                                    max(self.ess_regen_buff_dischg_kw[i], self.ess_desired_kw_4fc_eff[i]))
                            )
                        )
                    )
                )
            );
        } else if self.ess_accel_buff_chg_kw[i] > 0.0 {
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i],
                min(self.veh.mc_max_elec_in_kw + self.aux_in_kw[i], 
                    min(self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                        max(-self.cur_max_ess_chg_kw[i],
                            max(-self.max_ess_regen_buff_chg_kw[i],
                                min(-self.ess_accel_buff_chg_kw[i], self.ess_desired_kw_4fc_eff[i])
                            )
                        )
                    )
                )
            );
        } else if self.ess_desired_kw_4fc_eff[i] > 0.0 {
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i],
                min(self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                    min(self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                        max(-self.cur_max_ess_chg_kw[i],
                            min(self.ess_desired_kw_4fc_eff[i], self.max_ess_accell_buff_dischg_kw[i])
                        )
                    )
                )
            );
        } else {
            self.ess_kw_if_fc_req[i] = min(
                self.cur_max_ess_kw_out[i],
                min(self.veh.mc_max_elec_in_kw + self.aux_in_kw[i],
                    min(self.cur_max_mc_elec_kw_in[i] + self.aux_in_kw[i],
                        max(-self.cur_max_ess_chg_kw[i],
                            max(self.ess_desired_kw_4fc_eff[i], -self.max_ess_regen_buff_chg_kw[i])
                        )
                    )
                )
            );
        }

        self.er_kw_if_fc_req[i] = max(0.0,
            min(
                self.cur_max_roadway_chg_kw[i], 
                min(self.cur_max_mech_mc_kw_in[i],
                    self.ess_kw_if_fc_req[i] - self.mc_elec_in_lim_kw[i] + self.aux_in_kw[i])
            )
        );

        self.mc_elec_kw_in_if_fc_req[i] = self.ess_kw_if_fc_req[i] + self.er_kw_if_fc_req[i] - self.aux_in_kw[i];

        if self.veh.no_elec_sys {
            self.mc_kw_if_fc_req[i] = 0.0;
        } else if self.mc_elec_kw_in_if_fc_req[i] == 0.0 {
            self.mc_kw_if_fc_req[i] = 0.0;
        }

        else if self.mc_elec_kw_in_if_fc_req[i] > 0.0 {
            if self.mc_elec_kw_in_if_fc_req[i] == arrmax(&self.veh.mc_kw_in_array){
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len()-1];
            }
            else {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] * self.veh.mc_full_eff_array[
                    cmp::max(1, first_grtr(
                            &self.veh.mc_kw_in_array, min(arrmax(&self.veh.mc_kw_in_array) - 0.01, self.mc_elec_kw_in_if_fc_req[i])
                        ).unwrap_or(0) - 1
                    )
                ]
            }
        }

        else {
            if -self.mc_elec_kw_in_if_fc_req[i] == arrmax(&self.veh.mc_kw_in_array) {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len()-1];
            }
            else {
                self.mc_kw_if_fc_req[i] = self.mc_elec_kw_in_if_fc_req[i] / self.veh.mc_full_eff_array[
                    cmp::max(1, first_grtr(
                        &self.veh.mc_kw_in_array, min(
                            arrmax(&self.veh.mc_kw_in_array) - 0.01, 
                            -self.mc_elec_kw_in_if_fc_req[i])).unwrap_or(0) - 1
                    )
                ];
            }
        }

        if self.veh.max_motor_kw == 0.0 {
            self.mc_mech_kw_out_ach[i] = 0.0;
        } else if self.fc_forced_on[i] && self.can_pwr_all_elec[i] && (self.veh.veh_pt_type == HEV || self.veh.veh_pt_type == PHEV) && (self.veh.fc_eff_type != H2FC) {
            self.mc_mech_kw_out_ach[i] = self.mc_mech_kw_4forced_fc[i];
        } else if self.trans_kw_in_ach[i] <= 0.0 {
            if self.veh.fc_eff_type !=H2FC && self.veh.max_fuel_conv_kw > 0.0 {
                if self.can_pwr_all_elec[i] {
                    self.mc_mech_kw_out_ach[i] = - min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]);
                } else {
                    self.mc_mech_kw_out_ach[i] = min(
                        -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                        max(-self.cur_max_fc_kw_out[i], self.mc_kw_if_fc_req[i])
                    );
                } 
            } else {
                    self.mc_mech_kw_out_ach[i] = min(
                        -min(self.cur_max_mech_mc_kw_in[i], -self.trans_kw_in_ach[i]),
                        -self.trans_kw_in_ach[i]
                    );
            }
        } else if self.can_pwr_all_elec[i] {
            self.mc_mech_kw_out_ach[i] = self.trans_kw_in_ach[i]
        }

        else {
            self.mc_mech_kw_out_ach[i] = max(self.min_mc_kw_2help_fc[i], self.mc_kw_if_fc_req[i])
        }

        if self.mc_mech_kw_out_ach[i] == 0.0{
            self.mc_elec_kw_in_ach[i] = 0.0;
        }

        else if self.mc_mech_kw_out_ach[i] < 0.0 {
            if -self.mc_mech_kw_out_ach[i] == arrmax(&self.veh.mc_kw_in_array){
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len()-1]
            } else {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] * self.veh.mc_full_eff_array[
                    cmp::max(1, first_grtr(&self.veh.mc_kw_in_array, min(
                        arrmax(&self.veh.mc_kw_in_array) - 0.01,
                        -self.mc_mech_kw_out_ach[i])).unwrap_or(0) - 1
                    )
                ];
            }
        }
        else {
            if self.veh.max_motor_kw == self.mc_mech_kw_out_ach[i] {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array[self.veh.mc_full_eff_array.len()-1]
            } else {
                self.mc_elec_kw_in_ach[i] = self.mc_mech_kw_out_ach[i] / self.veh.mc_full_eff_array[
                    cmp::max(1, first_grtr(&self.veh.mc_kw_out_array, min(
                        self.veh.max_motor_kw - 0.01,
                        self.mc_mech_kw_out_ach[i])).unwrap_or(0) - 1
                    )
                ];
            }
        }

        if self.cur_max_roadway_chg_kw[i] == 0.0 {
            self.roadway_chg_kw_out_ach[i] = 0.0 
        } else if self.veh.fc_eff_type == H2FC {
            self.roadway_chg_kw_out_ach[i] = max(
                0.0,
                max(self.mc_elec_kw_in_ach[i],
                    max(self.max_ess_regen_buff_chg_kw[i],
                        max(self.ess_regen_buff_dischg_kw[i],
                            self.cur_max_roadway_chg_kw[i]
                        )
                    )
                )
            );
        } else if self.can_pwr_all_elec[i] {
            self.roadway_chg_kw_out_ach[i] = self.er_ae_kw_out[i];
        } else {
            self.roadway_chg_kw_out_ach[i] = self.er_kw_if_fc_req[i];
        }

        self.min_ess_kw_2help_fc[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - 
            self.cur_max_fc_kw_out[i] - self.roadway_chg_kw_out_ach[i];

        if self.veh.max_ess_kw == 0.0 || self.veh.max_ess_kwh == 0.0 {
            self.ess_kw_out_ach[i] = 0.0;
        } else if self.veh.fc_eff_type == H2FC {
            if self.trans_kw_out_ach[i] >= 0.0 {
                self.ess_kw_out_ach[i] = min(
                    self.cur_max_ess_kw_out[i],
                    min(self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i],
                        max(
                            self.min_ess_kw_2help_fc[i],
                            max(self.ess_desired_kw_4fc_eff[i],
                                self.ess_accel_regen_dischg_kw[i])))
                );
            } else {
                self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + 
                    self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i];
            }
        } else if self.high_acc_fc_on_tag[i] || self.veh.no_elec_aux{
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] - self.roadway_chg_kw_out_ach[i];
        }

        else {
            self.ess_kw_out_ach[i] = self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.roadway_chg_kw_out_ach[i];
        }

        if self.veh.no_elec_sys{
            self.ess_cur_kwh[i] = 0.0
        } else if self.ess_kw_out_ach[i] < 0.0 {
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s()[i] /
                3.6e3 * self.veh.ess_round_trip_eff.sqrt();
        } else {
            self.ess_cur_kwh[i] = self.ess_cur_kwh[i-1] - self.ess_kw_out_ach[i] * self.cyc.dt_s()[i] / 
                3.6e3 * self.veh.ess_round_trip_eff.sqrt();
        }

        if self.veh.max_ess_kwh == 0.0 {
            self.soc[i] = 0.0;
        } else {
            self.soc[i] = self.ess_cur_kwh[i] / self.veh.max_ess_kwh;
        } 

        if self.can_pwr_all_elec[i] && !self.fc_forced_on[i] && self.fc_kw_out_ach[i] == 0.0{
            self.fc_time_on[i] = 0.0
        } else {
            self.fc_time_on[i] = self.fc_time_on[i-1] + self.cyc.dt_s()[i];
        }
    }

    /// Sets fcKwOutAch and fcKwInAch.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_ess_power_rust(&mut self, i:usize) {
        if self.veh.max_fuel_conv_kw == 0.0 {
            self.fc_kw_out_ach[i] = 0.0;
        } else if self.veh.fc_eff_type == H2FC {
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i], 
                max(0.0, 
                    self.mc_elec_kw_in_ach[i] + self.aux_in_kw[i] - self.ess_kw_out_ach[i] - self.roadway_chg_kw_out_ach[i]
                )
            );
        } else if self.veh.no_elec_sys || self.veh.no_elec_aux || self.high_acc_fc_on_tag[i] {
            self.fc_kw_out_ach[i] = min(
                self.cur_max_fc_kw_out[i], 
                max(
                    0.0, 
                    self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i] + self.aux_in_kw[i]
                )
            );
        } else {
            self.fc_kw_out_ach[i] = min(self.cur_max_fc_kw_out[i], max(
                0.0, self.trans_kw_in_ach[i] - self.mc_mech_kw_out_ach[i]));
            }

        if self.veh.max_fuel_conv_kw == 0.0 {
            self.fc_kw_out_ach_pct[i] = 0.0;
        } else {
            self.fc_kw_out_ach_pct[i] = self.fc_kw_out_ach[i] / self.veh.max_fuel_conv_kw;
        }

        if self.fc_kw_out_ach[i] == 0.0 {
            self.fc_kw_in_ach[i] = 0.0;
            self.fc_kw_out_ach_pct[i] = 0.0;
        } else {
            if self.veh.fc_eff_array[first_grtr(
                &self.veh.fc_kw_out_array, min(self.fc_kw_out_ach[i], self.veh.max_fuel_conv_kw)).unwrap_or(0) - 1] != 0.0 {
                self.fc_kw_in_ach[i] = self.fc_kw_out_ach[i] / (self.veh.fc_eff_array[first_grtr(
                        &self.veh.fc_kw_out_array, min(self.fc_kw_out_ach[i], self.veh.max_fuel_conv_kw)).unwrap_or(0) - 1]);
            } else {
                self.fc_kw_in_ach[i] = 0.0
            }
        }

        self.fs_kw_out_ach[i] = self.fc_kw_in_ach[i];

        self.fs_kwh_out_ach[i] = self.fs_kw_out_ach[i] * self.cyc.dt_s()[i] / 3.6e3;
    }

    /// Sets scalar variables that can be calculated after a cycle is run. 
    /// This includes mpgge, various energy metrics, and others
    /// TODO: finish implementing this
    pub fn set_post_scalars_rust(& mut self) {
       // 
       // self.fs_cumu_mj_out_ach = (self.fs_kw_out_ach * self.cyc.dt_s).cumsum() * 1e-3

       // if self.fs_kwh_out_ach.sum() == 0:
       //     self.mpgge = 0.0

       // else:
       //     self.mpgge = self.dist_mi.sum() / (self.fs_kwh_out_ach.sum() / self.props.kwh_per_gge)

       // self.roadway_chg_kj = (self.roadway_chg_kw_out_ach * self.cyc.dt_s).sum()
       self.ess_dischg_kj = -1.0 * (self.soc[self.soc.len()-1] - self.soc[0]) * self.veh.max_ess_kwh * 3.6e3;
       // self.battery_kwh_per_mi  = (
       //     self.ess_dischg_kj / 3.6e3) / self.dist_mi.sum()
       // self.electric_kwh_per_mi  = (
       //     (self.roadway_chg_kj + self.ess_dischg_kj) / 3.6e3) / self.dist_mi.sum()
       self.fuel_kj = (self.fs_kw_out_ach.clone() * self.cyc.dt_s()).sum();

       // if (self.fuel_kj + self.roadway_chg_kj) == 0:
       //     self.ess2fuel_kwh  = 1.0

       // else:
       //     self.ess2fuel_kwh  = self.ess_dischg_kj / (self.fuel_kj + self.roadway_chg_kj)


       // DO NOT IMPLEMENT THE FOLLOWING: !!!!!!!!!!!!!!!!!!!!!!
       // make sure tests pass without these
       // a downstream project totally abused the `mpgge_elec` so I don't want to provide it anymore  

       // if self.mpgge == 0:
       //     # hardcoded conversion
       //     self.gallons_gas_equivalent_per_mile = self.electric_kwh_per_mi / self.props.kwh_per_gge
       //     grid_gallons_gas_equivalent_per_mile = self.electric_kwh_per_mi / self.props.kwh_per_gge / \
       //         self.veh.chg_eff

       // else:
       //     self.gallons_gas_equivalent_per_mile = 1 / \
       //         self.mpgge + self.electric_kwh_per_mi  / self.props.kwh_per_gge
       //     grid_gallons_gas_equivalent_per_mile = 1 / self.mpgge + \
       //         self.electric_kwh_per_mi / self.props.kwh_per_gge / self.veh.chg_eff

       // self.grid_mpgge_elec = 1 / grid_gallons_gas_equivalent_per_mile
       // self.mpgge_elec = 1 / self.gallons_gas_equivalent_per_mile

        // RESUME IMPLEMENTATION BELOW HERE

        // TODO: where "\w+_kw" and "cyc_\w+_kw" are identical remove the "cyc_\w+_kw" variant
        // for "cyc_\w+_kw" that don't have a corresponding "\w+_kw", remove the cyc_ prefix, as it 
        // provides no value

       // # energy audit calcs
       // self.drag_kw = self.cyc_drag_kw 
       // self.drag_kj = (self.drag_kw * self.cyc.dt_s).sum()
       // self.ascent_kw = self.cyc_ascent_kw
       // self.ascent_kj = (self.ascent_kw * self.cyc.dt_s).sum()
       // self.rr_kw = self.cyc_rr_kw
       // self.rr_kj = (self.rr_kw * self.cyc.dt_s).sum()

       // self.ess_loss_kw[1:] = np.array(
       //     [0 if (self.veh.max_ess_kw == 0 or self.veh.max_ess_kwh == 0)
       //     else -self.ess_kw_out_ach[i] - (-self.ess_kw_out_ach[i] * np.sqrt(self.veh.ess_round_trip_eff))
       //         if self.ess_kw_out_ach[i] < 0
       //     else self.ess_kw_out_ach[i] * (1.0 / np.sqrt(self.veh.ess_round_trip_eff)) - self.ess_kw_out_ach[i]
       //     for i in range(1, len(self.cyc.time_s))]
       // )
       // 
       // self.brake_kj = (self.cyc_fric_brake_kw * self.cyc.dt_s).sum()
       // self.trans_kj = ((self.trans_kw_in_ach - self.trans_kw_out_ach) * self.cyc.dt_s).sum()
       // self.mc_kj = ((self.mc_elec_kw_in_ach - self.mc_mech_kw_out_ach) * self.cyc.dt_s).sum()
       // self.ess_eff_kj = (self.ess_loss_kw * self.cyc.dt_s).sum()
       // self.aux_kj = (self.aux_in_kw * self.cyc.dt_s).sum()
       // self.fc_kj = ((self.fc_kw_in_ach - self.fc_kw_out_ach) * self.cyc.dt_s).sum()
       // 
       // self.net_kj = self.drag_kj + self.ascent_kj + self.rr_kj + self.brake_kj + self.trans_kj \
       //     + self.mc_kj + self.ess_eff_kj + self.aux_kj + self.fc_kj

       // self.ke_kj = 0.5 * self.veh.veh_kg * (self.mps_ach[0] ** 2 - self.mps_ach[-1] ** 2) / 1_000
       // 
       // self.energyAuditError = ((self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj) - self.net_kj
       //     ) / (self.roadway_chg_kj + self.ess_dischg_kj + self.fuel_kj + self.ke_kj)

       // if (np.abs(self.energyAuditError) > self.sim_params.energy_audit_error_tol) and \
       //     self.sim_params.verbose:
       //     print('Warning: There is a problem with conservation of energy.')
       //     print('Energy Audit Error:', np.round(self.energyAuditError, 5))

       // self.accel_kw[1:] = (self.veh.veh_kg / (2.0 * (self.cyc.dt_s[1:]))) * (
       //     self.mps_ach[1:] ** 2 - self.mps_ach[:-1] ** 2) / 1_000

       // self.trace_miss = False
       // self.trace_miss_dist_frac = abs(self.dist_m.sum() - self.cyc0.dist_m.sum()) / self.cyc0.dist_m.sum()
       // self.trace_miss_time_frac = abs(self.cyc.time_s[-1] - self.cyc0.time_s[-1]) / self.cyc0.time_s[-1]

       // if not(self.sim_params.missed_trace_correction):
       //     if self.trace_miss_dist_frac > self.sim_params.trace_miss_dist_tol:
       //         self.trace_miss = True
       //         if self.sim_params.verbose:
       //             print('Warning: Trace miss distance fraction:', np.round(self.trace_miss_dist_frac, 5))
       //             print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_dist_tol, 5))
       // else:
       //     if self.trace_miss_time_frac > self.sim_params.trace_miss_time_tol:
       //         self.trace_miss = True
       //         if self.sim_params.verbose:
       //             print('Warning: Trace miss time fraction:', np.round(self.trace_miss_time_frac, 5))
       //             print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_time_tol, 5))

       // self.trace_miss_speed_mps = max([
       //     abs(self.mps_ach[i] - self.cyc.mps[i]) for i in range(len(self.cyc.time_s))
       // ])
       // if self.trace_miss_speed_mps > self.sim_params.trace_miss_speed_mps_tol:
       //     self.trace_miss = True
       //     if self.sim_params.verbose:
       //         print('Warning: Trace miss speed [m/s]:', np.round(self.trace_miss_speed_mps, 5))
       //         print('exceeds tolerance of: ', np.round(self.sim_params.trace_miss_speed_mps_tol, 5))
    }
}