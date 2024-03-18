use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::air_properties as air;
use crate::imports::*;

#[pyo3_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI, HistoryMethods)]
/// Solver parameters
pub struct SimParams {
    pub ach_speed_max_iter: u32,
    pub ach_speed_tol: si::Ratio,
    pub ach_speed_solver_gain: f64,
    #[api(skip_get, skip_set)] // TODO: manually write out getter and setter
    pub trace_miss_tol: TraceMissTolerance,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            ach_speed_max_iter: 3,
            ach_speed_tol: 1e-9 * uc::R,
            ach_speed_solver_gain: 0.9,
            trace_miss_tol: Default::default(),
        }
    }
}

#[pyo3_api(
    #[new]
    fn __new__(veh: Vehicle, cyc: Cycle, sim_params: Option<SimParams>) -> anyhow::Result<Self> {
        Ok(SimDrive{
            veh,
            cyc, 
            sim_params: sim_params.unwrap_or_default(),
        })
    }

    #[pyo3(name = "walk")]
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }
)]
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
        // to increment `i` to 1 everywhere
        self.step();
        while self.veh.state.i < self.cyc.len() {
            self.solve_step()
                .with_context(|| format!("time step: {}", self.veh.state.i))?;
            self.save_state();
            self.step();
        }
        Ok(())
    }

    /// Solves current time step
    /// # Arguments
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        let i = self.veh.state.i;
        let dt = self.cyc.dt_at_i(i)?;
        self.veh.set_cur_pwr_max_out(dt)?;
        self.set_req_pwr(self.cyc.speed[i], dt)?;
        self.set_ach_speed(dt)?;
        self.veh.solve_powertrain(dt)?;
        // TODO (URGENT, IMPORTANT): make sure `EnergyMethods` macro/trait gets invoked here to increment all the
        // energies before proceeding
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
        let grade = &self.cyc.grade[i];
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
            * air::get_rho_air(None, None)
            * self.veh.drag_coef
            * self.veh.frontal_area
            * ((speed + speed_prev) / 2.0).powi(typenum::P3::new());
        vs.pwr_rr = mass
            * uc::ACC_GRAV
            * self.veh.wheel_rr_coef
            * grade.atan().cos()
            * (speed_prev + speed)
            / 2.;
        vs.pwr_whl_inertia = 0.5
            * self.veh.wheel_inertia_kg_m2
            * uc::KG
            * uc::M2
            * self.veh.num_wheels as f64
            * ((speed / self.veh.wheel_radius.unwrap()).powi(typenum::P2::new())
                - (speed_prev / self.veh.wheel_radius.unwrap()).powi(typenum::P2::new()))
            / self.cyc.dt_at_i(i)?;

        vs.pwr_tractive =
            vs.pwr_rr + vs.pwr_whl_inertia + vs.pwr_accel + vs.pwr_ascent + vs.pwr_drag;

        Ok(())
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - `dt`: time step size
    pub fn set_ach_speed(&mut self, dt: si::Time) -> anyhow::Result<()> {
        let vs = &mut self.veh.state;
        vs.cyc_met = vs.pwr_tractive_max >= vs.pwr_tractive;
        if vs.cyc_met {
            vs.speed_ach = self.cyc.speed[vs.i]
        } else {
            // assignments to allow for brevity
            let rho_air = air::get_rho_air(None, None);
            let mass = self.veh.mass.ok_or_else(|| {
                anyhow!("{}\nMass should have been set before now", format_dbg!())
            })?;
            let speed_prev = vs.speed_ach_prev;
            // Question: should this be grade at end of time step or start?
            // I'm treating it like grade at start is suitable
            let grade = utils::interp1d(
                &vs.dist.get::<si::meter>(),
                &self
                    .cyc
                    .dist
                    .iter()
                    .map(|d| d.get::<si::meter>())
                    .collect::<Vec<f64>>(),
                &self
                    .cyc
                    .grade
                    .iter()
                    .map(|g| g.get::<si::ratio>())
                    .collect::<Vec<f64>>(),
                utils::Extrapolate::Error,
            )?;

            // actual calucations
            let drag3 = 1.0 / 16.0 * rho_air * self.veh.drag_coef * self.veh.frontal_area;
            let accel2 = 0.5 * mass / dt;
            let drag2 =
                3.0 / 16.0 * rho_air * self.veh.drag_coef * self.veh.frontal_area * speed_prev;
            let wheel2 =
                0.5 * self.veh.wheel_inertia_kg_m2 * uc::KG * uc::M2 * self.veh.num_wheels as f64
                    / (dt * self.veh.wheel_radius.unwrap().powi(typenum::P2::new()));
            let drag1 = 3.0 / 16.0
                * rho_air
                * self.veh.drag_coef
                * self.veh.frontal_area
                * speed_prev.powi(typenum::P2::new());
            let roll1 = 0.5 * mass * uc::ACC_GRAV * self.veh.wheel_rr_coef * grade.atan().cos();
            let ascent1 = 0.5 * uc::ACC_GRAV * grade.atan().sin() * mass;
            let accel0 = -0.5 * mass * speed_prev.powi(typenum::P2::new()) / dt;
            let drag0 = 1.0 / 16.0
                * rho_air
                * self.veh.drag_coef
                * self.veh.frontal_area
                * speed_prev.powi(typenum::P3::new());
            let roll0 = 0.5
                * mass
                * uc::ACC_GRAV
                * self.veh.wheel_rr_coef
                * grade.atan().cos()
                * speed_prev;
            let ascent0 = 0.5 * uc::ACC_GRAV * grade.atan().sin() * mass * speed_prev;
            let wheel0 = -0.5
                * self.veh.wheel_inertia_kg_m2
                * uc::KG
                * uc::M2
                * self.veh.num_wheels as f64
                * speed_prev.powi(typenum::P2::new())
                / (dt * self.veh.wheel_radius.unwrap().powi(typenum::P2::new()));

            let t3 = drag3;
            let t2 = accel2 + drag2 + wheel2;
            let t1 = drag1 + roll1 + ascent1;
            // TODO: verify that final term should be `self.veh.state.pwr_out_max`.  Needs to be same as `self.cur_max_trans_kw_out[i]`
            let t0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) - self.veh.state.pwr_tractive_max;

            // initial guess
            let speed_guess = (1. * uc::MPS).max(self.veh.state.speed_ach);
            // stop criteria
            let max_iter = self.sim_params.ach_speed_max_iter;
            let xtol = self.sim_params.ach_speed_tol;
            // solver gain
            let g = self.sim_params.ach_speed_solver_gain;
            let pwr_err_fn = |speed_guess: si::Velocity| -> si::Power {
                t3 * speed_guess.powi(typenum::P3::new())
                    + t2 * speed_guess.powi(typenum::P2::new())
                    + t1 * speed_guess
                    + t0
            };
            let pwr_err_per_speed_guess_fn = |speed_guess: si::Velocity| {
                3.0 * t3 * speed_guess.powi(typenum::P2::new()) + 2.0 * t2 * speed_guess + t1
            };
            let pwr_err = pwr_err_fn(speed_guess);
            let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
            let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
            let mut speed_guesses = vec![speed_guess];
            let mut pwr_errs = vec![pwr_err];
            let mut d_pwr_err_per_d_speed_guesses = vec![pwr_err_per_speed_guess];
            let mut new_speed_guesses = vec![new_speed_guess];
            // speed achieved iteration counter
            let mut spd_ach_iter_counter = 1;
            let mut converged = false;
            while spd_ach_iter_counter < max_iter && !converged {
                let speed_guess = *speed_guesses
                    .iter()
                    .last()
                    .ok_or(anyhow!("{}", format_dbg!()))?
                    * (1.0 - g)
                    - g * *new_speed_guesses
                        .iter()
                        .last()
                        .ok_or(anyhow!("{}", format_dbg!()))?
                        / d_pwr_err_per_d_speed_guesses[speed_guesses.len() - 1];
                let pwr_err = pwr_err_fn(speed_guess);
                let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
                let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
                speed_guesses.push(speed_guess);
                pwr_errs.push(pwr_err);
                d_pwr_err_per_d_speed_guesses.push(pwr_err_per_speed_guess);
                new_speed_guesses.push(new_speed_guess);
                converged = ((*speed_guesses
                    .iter()
                    .last()
                    .ok_or(anyhow!("{}", format_dbg!()))?
                    - speed_guesses[speed_guesses.len() - 2])
                    / speed_guesses[speed_guesses.len() - 2])
                    .abs()
                    < xtol;
                spd_ach_iter_counter += 1;
            }

            // Question: could we assume `speed_guesses.iter().last()` is the correct solution?
            // This would make for faster running.
            self.veh.state.speed_ach = speed_guesses[pwr_errs
                .iter()
                .position(|&x| x == pwr_errs.iter().fold(uc::W * f64::NAN, |acc, &x| acc.min(x)))
                .ok_or_else(|| {
                    anyhow!(format_dbg!(pwr_errs
                        .iter()
                        .fold(uc::W * f64::NAN, |acc, &x| acc.min(x))))
                })?]
            .max(0.0 * uc::MPS);
        }

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
        let _veh = mock_f2_conv_veh();
        let _cyc = Cycle::from_resource("cycles/udds.csv").unwrap();
        let mut sd = SimDrive {
            veh: _veh,
            cyc: _cyc,
            sim_params: Default::default(),
        };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fuel_converter().unwrap().state.energy_fuel > uc::J * 0.);
    }
}
