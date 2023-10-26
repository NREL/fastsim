use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::air_properties::*;
use crate::imports::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI, HistoryMethods)]
/// Solver parameters
pub struct SimParams {
    pub ach_speed_max_iter: u32,
    pub ach_speed_tol: f64,
    pub trace_miss_tol: TraceMissTolerance,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            ach_speed_max_iter: 3,
            ach_speed_tol: 1e-9,
            trace_miss_tol: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI, HistoryMethods)]
pub struct SimDrive {
    #[has_state]
    pub veh: Vehicle,
    pub cyc: Cycle,
    pub sim_params: SimParams,
}

impl SimDrive {
    pub fn walk(&mut self) -> anyhow::Result<()> {
        ensure!(self.cyc.len() >= 2, format_dbg!(self.cyc.len() < 2));
        while self.veh.state.i < self.cyc.len() {
            self.solve_step()?;
            self.step();
        }
        Ok(())
    }

    /// Solves current time step
    /// # Arguments
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        let i = self.veh.state.i;
        let dt = self.cyc.dt_at_i(i);
        self.veh.set_cur_pwr_max_out(self.veh.pwr_aux, dt)?;
        self.set_req_pwr(self.cyc.speed[i], dt)?;
        self.set_ach_speed(dt)?;
        self.solve_powertrain(dt)?;
        Ok(())
    }

    /// Sets power required for given prescribed speed
    /// # Arguments
    /// - `speed`: prescribed or achieved speed
    /// - `speed_prev`: achieved speed at previous time step
    /// - `dt`: time step size
    pub fn set_req_pwr(&mut self, speed: si::Velocity, dt: si::Time) -> anyhow::Result<()> {
        // unwrap on `self.mass` is ok because any method of creating the vehicle should
        // automatically called `SerdeAPI::init`, which will ensure mass is some
        let i = self.veh.state.i;
        let vs = &mut self.veh.state;
        let speed_prev = vs.speed_ach_prev;
        let grade = &self
            .cyc
            .grade
            .as_ref()
            .ok_or_else(|| anyhow!("{}\nGrade should have been set already.", format_dbg!()))?[i];
        let mass = self.veh.mass.ok_or_else(|| {
            anyhow!(
                "{}\nVehicle mass should have been set already.",
                format_dbg!()
            )
        })?;
        vs.pwr_accel = mass / (2.0 * dt)
            * (speed.powi(typenum::P2::new()) - speed_prev.powi(typenum::P2::new()));
        vs.pwr_ascent = uc::ACC_GRAV * (*grade) * mass * (speed_prev + speed) / 2.0;
        vs.pwr_drag = 0.5
            * get_rho_air(None, None)
            * self.veh.drag_coef
            * self.veh.frontal_area
            * ((speed + speed_prev) / 2.0).powi(typenum::P3::new());
        vs.pwr_rr =
            mass * uc::ACC_GRAV * self.veh.wheel_rr * grade.atan().cos() * (speed_prev + speed)
                / 2.;
        vs.pwr_whl_inertia = 0.5
            * self.veh.wheel_inertia_kg_m2
            * uc::KG
            * uc::M2
            * self.veh.num_wheels as f64
            * ((speed / self.veh.wheel_radius).powi(typenum::P2::new())
                - (speed_prev / self.veh.wheel_radius).powi(typenum::P2::new()))
            / self.cyc.dt_at_i(i);

        vs.pwr_tractive = vs.pwr_accel + vs.pwr_ascent + vs.pwr_drag;
        vs.pwr_out = vs.pwr_tractive + vs.pwr_rr + vs.pwr_whl_inertia;

        Ok(())
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - `dt`: time step size
    pub fn set_ach_speed(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // let vs = &mut self.veh.state;
        // vs.cyc_met = vs.pwr_out_max >= vs.pwr_out;
        // if vs.cyc_met {
        //     vs.speed_ach = self.cyc.speed[vs.i]
        // } else {
        //     let grade =
        //         let drag3 = 1.0 / 16.0
        //             * self.props.air_density_kg_per_m3
        //             * self.veh.drag_coef
        //             * self.veh.frontal_area_m2;
        //         let accel2 = 0.5 * self.veh.veh_kg / self.cyc.dt_s_at_i(i);
        //         let drag2 = 3.0 / 16.0
        //             * self.props.air_density_kg_per_m3
        //             * self.veh.drag_coef
        //             * self.veh.frontal_area_m2
        //             * self.mps_ach[i - 1];
        //         let wheel2 = 0.5 * self.veh.wheel_inertia_kg_m2 * self.veh.num_wheels
        //             / (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m.powf(2.0));
        //         let drag1 = 3.0 / 16.0
        //             * self.props.air_density_kg_per_m3
        //             * self.veh.drag_coef
        //             * self.veh.frontal_area_m2
        //             * self.mps_ach[i - 1].powf(2.0);
        //         let roll1 = 0.5
        //             * self.veh.veh_kg
        //             * self.props.a_grav_mps2
        //             * self.veh.wheel_rr_coef
        //             * grade.atan().cos();
        //         let ascent1 = 0.5 * self.props.a_grav_mps2 * grade.atan().sin() * self.veh.veh_kg;
        //         let accel0 =
        //             -0.5 * self.veh.veh_kg * self.mps_ach[i - 1].powf(2.0) / self.cyc.dt_s_at_i(i);
        //         let drag0 = 1.0 / 16.0
        //             * self.props.air_density_kg_per_m3
        //             * self.veh.drag_coef
        //             * self.veh.frontal_area_m2
        //             * self.mps_ach[i - 1].powf(3.0);
        //         let roll0 = 0.5
        //             * self.veh.veh_kg
        //             * self.props.a_grav_mps2
        //             * self.veh.wheel_rr_coef
        //             * grade.atan().cos()
        //             * self.mps_ach[i - 1];
        //         let ascent0 = 0.5
        //             * self.props.a_grav_mps2
        //             * grade.atan().sin()
        //             * self.veh.veh_kg
        //             * self.mps_ach[i - 1];
        //         let wheel0 = -0.5
        //             * self.veh.wheel_inertia_kg_m2
        //             * self.veh.num_wheels
        //             * self.mps_ach[i - 1].powf(2.0)
        //             / (self.cyc.dt_s_at_i(i) * self.veh.wheel_radius_m.powf(2.0));

        //         let t3 = drag3 / 1e3;
        //         let t2 = (accel2 + drag2 + wheel2) / 1e3;
        //         let t1 = (drag1 + roll1 + ascent1) / 1e3;
        //         let t0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) / 1e3
        //             - self.cur_max_trans_kw_out[i];

        //         // initial guess
        //         let speed_guess = max(1.0, self.mps_ach[i - 1]);
        //         // stop criteria
        //         let max_iter = self.sim_params.newton_max_iter;
        //         let xtol = self.sim_params.newton_xtol;
        //         // solver gain
        //         let g = self.sim_params.newton_gain;
        //         let pwr_err_fn = |speed_guess: f64| -> f64 {
        //             t3 * speed_guess.powf(3.0) + t2 * speed_guess.powf(2.0) + t1 * speed_guess + t0
        //         };
        //         let pwr_err_per_speed_guess_fn = |speed_guess: f64| -> f64 {
        //             3.0 * t3 * speed_guess.powf(2.0) + 2.0 * t2 * speed_guess + t1
        //         };
        //         let pwr_err = pwr_err_fn(speed_guess);
        //         let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
        //         let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
        //         let mut speed_guesses = vec![speed_guess];
        //         let mut pwr_errs = vec![pwr_err];
        //         let mut d_pwr_err_per_d_speed_guesses = vec![pwr_err_per_speed_guess];
        //         let mut new_speed_guesses = vec![new_speed_guess];
        //         // speed achieved iteration counter
        //         let mut spd_ach_i = 1;
        //         let mut converged = false;
        //         while spd_ach_i < max_iter && !converged {
        //             let speed_guess = speed_guesses
        //                 .iter()
        //                 .last()
        //                 .ok_or(anyhow!("{}", format_dbg!()))?
        //                 * (1.0 - g)
        //                 - g * new_speed_guesses
        //                     .iter()
        //                     .last()
        //                     .ok_or(anyhow!("{}", format_dbg!()))?
        //                     / d_pwr_err_per_d_speed_guesses[speed_guesses.len() - 1];
        //             let pwr_err = pwr_err_fn(speed_guess);
        //             let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
        //             let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
        //             speed_guesses.push(speed_guess);
        //             pwr_errs.push(pwr_err);
        //             d_pwr_err_per_d_speed_guesses.push(pwr_err_per_speed_guess);
        //             new_speed_guesses.push(new_speed_guess);
        //             converged = ((speed_guesses
        //                 .iter()
        //                 .last()
        //                 .ok_or(anyhow!("{}", format_dbg!()))?
        //                 - speed_guesses[speed_guesses.len() - 2])
        //                 / speed_guesses[speed_guesses.len() - 2])
        //                 .abs()
        //                 < xtol;
        //             spd_ach_i += 1;
        //         }

        //         self.newton_iters[i] = spd_ach_i;

        //         let _ys = Array::from_vec(pwr_errs).map(|x| x.abs());
        //         // Question: could we assume `speed_guesses.iter().last()` is the correct solution?
        //         // This would make for faster running.
        //         self.mps_ach[i] = max(
        //             speed_guesses[_ys
        //                 .iter()
        //                 .position(|&x| x == ndarrmin(&_ys))
        //                 .ok_or_else(|| anyhow!(format_dbg!(ndarrmin(&_ys))))?],
        //             0.0,
        //         );
        // }

        Ok(())
    }

    /// Solves for efficiencies and energy dissipation backwards up the whole powertrain
    /// # Arguments
    /// - `dt`: time step size
    pub fn solve_powertrain(&mut self, dt: si::Time) -> anyhow::Result<()> {
        todo!();
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI, HistoryMethods)]

pub struct TraceMissTolerance {
    tol_dist: si::Length,
    tol_dist_frac: si::Ratio,
    tol_speed: si::Velocity,
    tol_speed_frac: si::Ratio,
}

impl Default for TraceMissTolerance {
    fn default() -> Self {
        Self {
            tol_dist: 666. * uc::M,
            tol_dist_frac: 666. * uc::R,
            tol_speed: 666. * uc::MPS,
            tol_speed_frac: 666. * uc::R,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::vehicle_model::tests::mock_f2_conv_veh;
    #[test]
    fn test_sim_drive() {
        let veh = mock_f2_conv_veh();
        let cyc = Cycle::from_file(todo!()).unwrap();
        let sd = SimDrive {
            veh,
            cyc,
            sim_params: Default::default(),
        };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fuel_converter().unwrap().state.energy_fuel > uc::J * 0.);
    }
}
