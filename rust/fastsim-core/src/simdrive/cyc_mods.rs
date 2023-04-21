//! SimDrive methods that manipulate cycle on the fly

use super::simdrive_impl::*;
use super::*;
use crate::cycle::{
    accel_array_for_constant_jerk, accel_for_constant_jerk, calc_constant_jerk_trajectory,
    create_dist_and_target_speeds_by_microtrip, detect_passing, extend_cycle,
    trapz_distance_for_step, trapz_step_distances, trapz_step_start_distance, PassingInfo,
};
use crate::simdrive::RustSimDrive;
use crate::utils::{add_from, max, min, ndarrcumsum, ndarrmax, ndarrmin, ndarrunique};

impl RustSimDrive {
    /// Provides the gap-with lead vehicle from start to finish
    pub fn gap_to_lead_vehicle_m(&self) -> Array1<f64> {
        // TODO: consider basing on dist_m?
        let mut gaps_m = ndarrcumsum(&trapz_step_distances(&self.cyc0))
            - ndarrcumsum(&trapz_step_distances(&self.cyc));
        if self.sim_params.idm_allow {
            gaps_m += self.sim_params.idm_minimum_gap_m;
        }
        gaps_m
    }

    /// Sets the intelligent driver model parameters for an eco-cruise driving trajectory.
    /// This is a convenience method instead of setting the sim_params.idm* parameters yourself.
    /// - by_microtrip: bool, if True, target speed is set by microtrip, else by cycle
    /// - extend_fraction: float, the fraction of time to extend the cycle to allow for catch-up
    ///     of the following vehicle
    /// - blend_factor: float, a value between 0 and 1; only used of by_microtrip is True, blends
    ///     between microtrip average speed and microtrip average speed when moving. Must be
    ///     between 0 and 1 inclusive
    /// - min_target_speed_m_per_s: float, the minimum speed allowed by the eco-cruise algorithm
    /// Mutates the current SimDrive object for eco-cruise.
    pub fn activate_eco_cruise_rust(
        &mut self,
        by_microtrip: bool,            // False
        extend_fraction: f64,          // 0.1
        blend_factor: f64,             // 0.0
        min_target_speed_m_per_s: f64, // 8.0
    ) -> Result<(), anyhow::Error> {
        self.sim_params.idm_allow = true;
        if !by_microtrip {
            self.sim_params.idm_v_desired_m_per_s =
                if !self.cyc0.time_s.is_empty() && self.cyc0.time_s.last().unwrap() > &0.0 {
                    self.cyc0
                        .dist_m()
                        .slice(s![0..self.cyc0.time_s.len()])
                        .sum()
                        / self.cyc0.time_s.last().unwrap()
                } else {
                    0.0
                };
        } else {
            if !(0.0..=1.0).contains(&blend_factor) {
                return Err(anyhow!(
                    "blend_factor must be between 0 and 1 but got {}",
                    blend_factor
                ));
            }
            if min_target_speed_m_per_s < 0.0 {
                return Err(anyhow!(
                    "min_target_speed_m_per_s must be >= 0 but got {}",
                    min_target_speed_m_per_s
                ));
            }
            self.sim_params.idm_v_desired_in_m_per_s_by_distance_m =
                Some(create_dist_and_target_speeds_by_microtrip(
                    &self.cyc0,
                    blend_factor,
                    min_target_speed_m_per_s,
                ));
        }
        // Extend the duration of the base cycle
        if extend_fraction < 0.0 {
            return Err(anyhow!(
                "extend_fraction must be >= 0.0 but got {}",
                extend_fraction
            ));
        }
        if extend_fraction > 0.0 {
            self.cyc0 = extend_cycle(&self.cyc0, None, Some(extend_fraction));
            self.cyc = self.cyc0.clone();
        }
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
        let d0_lead_m: f64 = self.cyc0_cache.trapz_distances_m[(i - 1).max(0)] + s0_m;
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
        let v_desired_m_per_s = if self.idm_target_speed_m_per_s[i] > 0.0 {
            self.idm_target_speed_m_per_s[i]
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
    ///     and not allowing IDM/following (i.e., self.sim_params.idm_allow == False)
    ///     then returns self.cyc.grade[i]
    pub fn estimate_grade_for_step(&self, i: usize) -> f64 {
        if self.cyc0_cache.grade_all_zero {
            return 0.0;
        }
        if !self.sim_params.coast_allow && !self.sim_params.idm_allow {
            return self.cyc.grade[i];
        }
        self.cyc0_cache
            .interp_grade(trapz_step_start_distance(&self.cyc, i))
    }

    /// For situations where cyc can deviate from cyc0, this method
    /// looks up and accurately interpolates what the average grade over
    /// the step should be.
    /// If mps_ach is not None, the mps_ach value is used to predict the
    /// distance traveled over the step.
    /// NOTE:
    ///     If not allowing coasting (i.e., sim_params.coast_allow == False)
    ///     and not allowing IDM/following (i.e., self.sim_params.idm_allow == False)
    ///     then returns self.cyc.grade[i]
    pub fn lookup_grade_for_step(&self, i: usize, mps_ach: Option<f64>) -> f64 {
        if self.cyc0_cache.grade_all_zero {
            return 0.0;
        }
        if !self.sim_params.coast_allow && !self.sim_params.idm_allow {
            return self.cyc.grade[i];
        }
        match mps_ach {
            Some(mps_ach) => self.cyc0.average_grade_over_range(
                trapz_step_start_distance(&self.cyc, i),
                0.5 * (mps_ach + self.mps_ach[i - 1]) * self.cyc.dt_s_at_i(i),
                Some(&self.cyc0_cache),
            ),
            None => self.cyc0.average_grade_over_range(
                trapz_step_start_distance(&self.cyc, i),
                trapz_distance_for_step(&self.cyc, i),
                Some(&self.cyc0_cache),
            ),
        }
    }
    pub fn set_time_dilation(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
                    d_short.last().unwrap() / self.cyc0.dt_s_at_i(i) / self.mps_ach[i], // initial guess, speed that needed to be achived per speed that was achieved
                    self.sim_params.min_time_dilation,
                ),
                self.sim_params.max_time_dilation,
            ));

            // add time dilation factor * step size to current and subsequent times
            self.cyc.time_s = add_from(
                &self.cyc.time_s,
                i,
                self.cyc.dt_s_at_i(i) * t_dilation.last().unwrap(),
            );
            self.solve_step(i)?;

            trace_met =
                    // convergence criteria
                    (self.cyc0.dist_m().slice(s![0..i+1]).sum() - self.dist_m.slice(s![0..i+1]).sum()).abs() / self.cyc0.dist_m().slice(s![0..i+1]).sum()
                    < self.sim_params.time_dilation_tol
                    // exceeding max time dilation
                    || t_dilation.last().unwrap() >= &self.sim_params.max_time_dilation
                    // lower than min time dilation
                    || t_dilation.last().unwrap() <= &self.sim_params.min_time_dilation;
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
                    t_dilation.last().unwrap()
                        - (t_dilation.last().unwrap() - t_dilation[t_dilation.len() - 2])
                            / (d_short.last().unwrap() - d_short[d_short.len() - 2])
                            * d_short.last().unwrap(),
                    self.sim_params.min_time_dilation,
                ),
                self.sim_params.max_time_dilation,
            ));
            self.cyc.time_s = add_from(
                &self.cyc.time_s,
                i,
                self.cyc.dt_s_at_i(i) * t_dilation.last().unwrap(),
            );

            self.solve_step(i)?;

            self.trace_miss_iters[i] += 1;

            trace_met =
                    // convergence criteria
                    (self.cyc0.dist_m().slice(s![0..i+1]).sum() - self.dist_m.slice(s![0..i+1]).sum()).abs() / self.cyc0.dist_m().slice(s![0..i+1]).sum()
                    < self.sim_params.time_dilation_tol
                    // max iterations
                    || self.trace_miss_iters[i] >= self.sim_params.max_trace_miss_iters
                    // exceeding max time dilation
                    || t_dilation.last().unwrap() >= &self.sim_params.max_time_dilation
                    // lower than min time dilation
                    || t_dilation.last().unwrap() <= &self.sim_params.min_time_dilation;
        }
        Ok(())
    }

    // Calculates the derivative dv/dd (change in speed by change in distance)
    // - v: number, the speed at which to evaluate dv/dd (m/s)
    // - grade: number, the road grade as a decimal fraction
    // RETURN: number, the dv/dd for these conditions
    fn calc_dvdd(&self, v: f64, grade: f64) -> f64 {
        if v <= 0.0 {
            0.0
        } else {
            let (atan_grade_sin, atan_grade_cos) = if grade == 0.0 {
                (0.0, 1.0)
            } else {
                let atan_g = grade.atan();
                (atan_g.sin(), atan_g.cos())
            };
            let g = self.props.a_grav_mps2;
            let m = self.veh.veh_kg;
            let rho_cdfa =
                self.props.air_density_kg_per_m3 * self.veh.drag_coef * self.veh.frontal_area_m2;
            let rrc = self.veh.wheel_rr_coef;
            -1.0 * ((g / v) * (atan_grade_sin + rrc * atan_grade_cos)
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
        let ds = &self.cyc0_cache.trapz_distances_m;
        let gs = self.cyc0.grade.clone();
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let mut distances_m: Vec<f64> = Vec::with_capacity(ds.len());
        let mut grade_by_distance: Vec<f64> = Vec::with_capacity(ds.len());
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
        let d_max = distances_m.last().unwrap() - dtb;
        let unique_grades = ndarrunique(&grade_by_distance);
        let unique_grade: Option<f64> = if unique_grades.len() == 1 {
            Some(unique_grades[0])
        } else {
            None
        };
        let has_unique_grade: bool = unique_grade.is_some();
        let max_iter = 180;
        let iters_per_step = if self.sim_params.favor_grade_accuracy {
            2
        } else {
            1
        };
        let mut new_speeds_m_per_s: Vec<f64> = Vec::with_capacity(max_iter as usize);
        let mut v = v0;
        let mut iter = 0;
        let mut idx = i;
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0, Some(&self.cyc0_cache));
        while v > v_brake && v >= 0.0 && d <= d_max && iter < max_iter && idx < self.mps_ach.len() {
            let dt_s = self.cyc0.dt_s_at_i(idx);
            let mut gr = match unique_grade {
                Some(g) => g,
                None => self.cyc0_cache.interp_grade(d + d0),
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
                if self.sim_params.favor_grade_accuracy {
                    gr = match unique_grade {
                        Some(g) => g,
                        None => {
                            self.cyc0
                                .average_grade_over_range(d + d0, dd, Some(&self.cyc0_cache))
                        }
                    };
                }
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
        let ds = &self.cyc0_cache.trapz_distances_m;
        let gs = &self.cyc0.grade;
        assert!(
            ds.len() == gs.len(),
            "Assumed length of ds and gs the same but actually ds.len():{} and gs.len():{}",
            ds.len(),
            gs.len()
        );
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let mut grade_by_distance: Vec<f64> = Vec::with_capacity(ds.len());
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
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0, Some(&self.cyc0_cache));
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
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0, Some(&self.cyc0_cache));
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
        if !self.sim_params.idm_allow && self.cyc.mps[i] < speed_tol {
            let d0 = trapz_step_start_distance(&self.cyc, i);
            let d0_lv = self.cyc0_cache.trapz_distances_m[i - 1];
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
        if let Some(cd) = coast_delay {
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
    pub fn set_coast_speed(&mut self, i: usize) -> Result<(), anyhow::Error> {
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
            if !self.sim_params.idm_allow {
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
                if let Some(ti) = target_idx {
                    self.cyc.mps[i] = self.cyc0.mps[cmp::min(ti, self.cyc0.mps.len() - 1)];
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
        self.solve_step(i)?;
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
                let (_, num_steps) = self.cyc.modify_with_braking_trajectory(
                    self.sim_params.coast_brake_accel_m_per_s2,
                    i,
                    None,
                );
                for idx in i..self.cyc.time_s.len() {
                    self.impose_coast[idx] = idx < (i + num_steps);
                }
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
                        log::warn!(
                            "final_speed_m_per_s={} not close to coast_brake_start_speed={} for i={}; i_for_brake={}, traj_n={}",
                            final_speed_m_per_s,
                            self.sim_params.coast_brake_start_speed_m_per_s,
                            i,
                            i_for_brake,
                            traj_n
                        );
                    }
                }
            }
            if adjusted_current_speed {
                if !self.sim_params.coast_allow_passing {
                    self.prevent_collisions(i, None);
                }
                self.solve_step(i)?;
                self.newton_iters[i] = 0; // reset newton iters
                self.cyc.mps[i] = self.mps_ach[i];
            }
        }
        Ok(())
    }
}
