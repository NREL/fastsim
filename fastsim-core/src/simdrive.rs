use self::utils::almost_eq_uom;

use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::air_properties as air;
use crate::imports::*;
use crate::prelude::*;

#[fastsim_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, HistoryMethods)]
/// Solver parameters
pub struct SimParams {
    pub ach_speed_max_iter: u32,
    pub ach_speed_tol: si::Ratio,
    pub ach_speed_solver_gain: f64,
    #[api(skip_get, skip_set)]
    pub trace_miss_tol: TraceMissTolerance,
    #[api(skip_get, skip_set)]
    pub trace_miss_opts: TraceMissOptions,
}

impl SerdeAPI for SimParams {}
impl Init for SimParams {}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            ach_speed_max_iter: 3,
            ach_speed_tol: 1e-9 * uc::R,
            ach_speed_solver_gain: 0.9,
            trace_miss_tol: Default::default(),
            trace_miss_opts: Default::default(),
        }
    }
}

#[fastsim_api(
    #[new]
    fn __new__(veh: Vehicle, cyc: Cycle, sim_params: Option<SimParams>) -> anyhow::Result<Self> {
        Ok(SimDrive{
            veh,
            cyc,
            sim_params: sim_params.unwrap_or_default(),
        })
    }

    /// Run vehicle simulation once
    #[pyo3(name = "walk_once")]
    fn walk_once_py(&mut self) -> anyhow::Result<()> {
        self.walk_once()
    }

    /// Run vehicle simulation, and, if applicable, apply powertrain-specific
    /// corrections (e.g. iterate `walk` until SOC balance is achieved -- i.e. initial
    /// and final SOC are nearly identical)
    #[pyo3(name = "walk")]
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[pyo3(name = "to_fastsim2")]
    fn to_fastsim2_py(&self) -> anyhow::Result<fastsim_2::simdrive::RustSimDrive> {
        self.to_fastsim2()
    }

)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, HistoryMethods)]
pub struct SimDrive {
    #[has_state]
    pub veh: Vehicle,
    pub cyc: Cycle,
    pub sim_params: SimParams,
}

impl SerdeAPI for SimDrive {}
impl Init for SimDrive {
    fn init(&mut self) -> anyhow::Result<()> {
        self.veh.init().with_context(|| anyhow!(format_dbg!()))?;
        self.cyc.init().with_context(|| anyhow!(format_dbg!()))?;
        self.sim_params
            .init()
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }
}

impl SimDrive {
    pub fn new(veh: Vehicle, cyc: Cycle, sim_params: Option<SimParams>) -> Self {
        Self {
            veh,
            cyc,
            sim_params: sim_params.unwrap_or_default(),
        }
    }

    // # TODO:
    // - [x] remove separate `walk_hev`
    // - [x] come up with a mechanism of enabling or disabling SOC balance iteration
    // - [x] warn after ~2 (make configurable) iterations and error after ~10 iterations
    // - [x] figure out speed trace miss -- resulted from not having enough SOC buffer
    // - [ ] probably done already -- make sure that when doing lefthand interpolation we have same array length as f2
    // ## Features
    // - [x] engine min time on per f2
    // - [ ] accel buffer per f2
    // - [ ] regen buffer per f2
    // - [ ] regen limiting curve during speeds approaching zero per f2 -- less urgent
    // - [ ] controls to make the engine recharge the battery if below buffers
    // - [ ] controls to make engine run efficiency when on. In f2, this just maxes out
    //       the engine.  We should be able to actually get near peak efficiency in a
    //       smarter way in f3 because the performance penalty is less problematic.
    // - [ ] ability to manipulate friction/regen brake split based on required braking
    //       power -- new feature -- move this to enum
    // - [x] make enum `EngineOnCause::{AlreadyOn, TooCold,
    //       PowerDemand}` and save it in a vec or some such for when there are
    //       multiple causes -- new feature

    /// Run vehicle simulation, and, if applicable, apply powertrain-specific
    /// corrections (e.g. iterate `walk` until SOC balance is achieved -- i.e. initial
    /// and final SOC are nearly identical)
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.veh.state.mass = self
            .veh
            .mass()
            .with_context(|| format_dbg!())?
            .with_context(|| format_dbg!("Expected mass to have been set."))?;
        match self.veh.pt_type {
            PowertrainType::HybridElectricVehicle(_) => {
                // Net battery energy used per amount of fuel used
                // clone initial vehicle to preserve starting state (TODO: figure out if this is a huge CPU burden)
                let veh_init = self.veh.clone();
                loop {
                    self.veh.hev_mut().unwrap().state.soc_bal_iters += 1;
                    self.walk_once()?;
                    let soc_final = self
                        .veh
                        .res()
                        // `unwrap` is ok because it's already been checked
                        .unwrap()
                        .state
                        .soc;
                    let res_per_fuel = self.veh.res().unwrap().state.energy_out_chemical
                        / self.veh.fc().unwrap().state.energy_fuel;
                    if self.veh.hev().unwrap().state.soc_bal_iters
                        > self.veh.hev().unwrap().sim_params.soc_balance_iter_err
                    {
                        bail!(
                            "{}",
                            format_dbg!((
                                self.veh.hev().unwrap().state.soc_bal_iters,
                                self.veh.hev().unwrap().sim_params.soc_balance_iter_err
                            ))
                        );
                    }
                    if res_per_fuel.abs() < self.veh.hev().unwrap().sim_params.res_per_fuel_lim
                        || !self.veh.hev().unwrap().sim_params.balance_soc
                    {
                        break;
                    } else {
                        // prep for another iteration
                        if let Some(&mut ref mut hev) = self.veh.hev_mut() {
                            if hev.sim_params.save_soc_bal_iters {
                                hev.soc_bal_iter_history.push(hev.clone());
                            }
                        }
                        let soc_bal_iters = self.veh.hev().unwrap().state.soc_bal_iters;
                        // reset vehicle to initial state
                        self.veh = veh_init.clone();
                        // retain soc_bal_iters
                        self.veh.hev_mut().unwrap().state.soc_bal_iters = soc_bal_iters;
                        // start SOC at previous final value
                        self.veh.res_mut().unwrap().state.soc = soc_final;
                    }
                }
            }
            _ => self.walk_once()?,
        }
        Ok(())
    }

    /// Run vehicle simulation once
    pub fn walk_once(&mut self) -> anyhow::Result<()> {
        ensure!(self.cyc.len() >= 2, format_dbg!(self.cyc.len() < 2));
        self.save_state();
        // to increment `i` to 1 everywhere
        self.step();
        while self.veh.state.i < self.cyc.len() {
            self.solve_step()
                .with_context(|| format!("{}\ntime step: {}", format_dbg!(), self.veh.state.i))?;
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
        let speed_prev = self.veh.state.speed_ach;
        self.veh
            .set_curr_pwr_out_max(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.set_pwr_prop_for_speed(self.cyc.speed[i], speed_prev, dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.veh.state.pwr_tractive_for_cyc = self.veh.state.pwr_tractive;
        self.set_ach_speed(self.cyc.speed[i], speed_prev, dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.veh
            .solve_powertrain(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.veh.set_cumulative(dt);
        Ok(())
    }

    /// Sets power required for given prescribed speed
    /// # Arguments
    /// - `speed`: prescribed or achieved speed
    /// - `speed_prev`: previously achieved speed
    /// - `dt`: time step size
    pub fn set_pwr_prop_for_speed(
        &mut self,
        speed: si::Velocity,
        speed_prev: si::Velocity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let i = self.veh.state.i;
        let vs = &mut self.veh.state;
        // TODO: get @mokeefe to give this a serious look and think about grade alignment issues that may arise
        vs.grade_curr = if vs.cyc_met_overall {
            *self.cyc.grade.get(i).with_context(|| format_dbg!())?
        } else {
            uc::R
                * self
                    .cyc
                    .grade_interp
                    .as_ref()
                    .with_context(|| format_dbg!("You might have somehow bypassed `init()`"))?
                    .interpolate(&[vs.dist.get::<si::meter>()])?
        };

        let mass = self.veh.mass.with_context(|| {
            format!(
                "{}\nVehicle mass should have been set already.",
                format_dbg!()
            )
        })?;
        vs.pwr_accel = mass / (2.0 * dt)
            * (speed.powi(typenum::P2::new()) - speed_prev.powi(typenum::P2::new()));
        vs.pwr_ascent = uc::ACC_GRAV * vs.grade_curr * mass * (speed_prev + speed) / 2.0;
        vs.pwr_drag = 0.5
            // TODO: feed in elevation
            * air::get_density_air(None, None)
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * ((speed + speed_prev) / 2.0).powi(typenum::P3::new());
        vs.pwr_rr = mass
            * uc::ACC_GRAV
            * self.veh.chassis.wheel_rr_coef
            * vs.grade_curr.atan().cos()
            * (speed_prev + speed)
            / 2.;
        vs.pwr_whl_inertia = 0.5
            * self.veh.chassis.wheel_inertia
            * self.veh.chassis.num_wheels as f64
            * ((speed / self.veh.chassis.wheel_radius.unwrap()).powi(typenum::P2::new())
                - (speed_prev / self.veh.chassis.wheel_radius.unwrap()).powi(typenum::P2::new()))
            / self.cyc.dt_at_i(i)?;

        vs.pwr_tractive =
            vs.pwr_rr + vs.pwr_whl_inertia + vs.pwr_accel + vs.pwr_ascent + vs.pwr_drag;
        vs.cyc_met = vs.pwr_tractive <= vs.pwr_prop_fwd_max;
        if !vs.cyc_met {
            // if current power demand is not met, then this becomes false for
            // the rest of the cycle and should not be manipulated anywhere else
            vs.cyc_met_overall = false;
        }
        Ok(())
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - `cyc_speed`: prescribed speed
    /// - `dt`: time step size
    /// - `speed_prev`: previously achieved speed
    pub fn set_ach_speed(
        &mut self,
        cyc_speed: si::Velocity,
        speed_prev: si::Velocity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // borrow state as `vs` for shorthand
        let vs = &mut self.veh.state;
        if vs.cyc_met {
            vs.speed_ach = cyc_speed;
            return Ok(());
        } else {
            match self.sim_params.trace_miss_opts {
                TraceMissOptions::Allow => {
                    // do nothing because `set_ach_speed` should be allowed to proceed to handle this
                }
                TraceMissOptions::Error => bail!(
                    "{}\nFailed to meet speed trace.
prescribed speed: {} mph
achieved speed: {} mph
pwr for prescribed speed: {} kW
pwr for achieved speed: {} kW
pwr available: {} kW,
pwr deficit: {} kW
",
                    format_dbg!(),
                    cyc_speed.get::<si::mile_per_hour>(),
                    vs.speed_ach.get::<si::mile_per_hour>(),
                    vs.pwr_tractive_for_cyc.get::<si::kilowatt>(),
                    vs.pwr_tractive.get::<si::kilowatt>(),
                    vs.pwr_prop_fwd_max.get::<si::kilowatt>(),
                    (vs.pwr_tractive - vs.pwr_prop_fwd_max)
                        .get::<si::kilowatt>()
                        .format_eng(None)
                ),
                TraceMissOptions::Correct => todo!(),
            }
        }
        let mass = self
            .veh
            .mass
            .with_context(|| format!("{}\nMass should have been set before now", format_dbg!()))?;

        let drag3 = 1.0 / 16.0
            * vs.air_density
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area;
        let accel2 = 0.5 * mass / dt;
        let drag2 = 3.0 / 16.0
            * vs.air_density
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * speed_prev;
        let wheel2 = 0.5 * self.veh.chassis.wheel_inertia * self.veh.chassis.num_wheels as f64
            / (dt
                * self
                    .veh
                    .chassis
                    .wheel_radius
                    .unwrap()
                    .powi(typenum::P2::new()));
        let drag1 = 3.0 / 16.0
            * vs.air_density
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * speed_prev.powi(typenum::P2::new());
        let roll1 =
            0.5 * mass * uc::ACC_GRAV * self.veh.chassis.wheel_rr_coef * vs.grade_curr.atan().cos();
        let ascent1 = 0.5 * uc::ACC_GRAV * vs.grade_curr.atan().sin() * mass;
        let accel0 = -0.5 * mass * speed_prev.powi(typenum::P2::new()) / dt;
        let drag0 = 1.0 / 16.0
            * vs.air_density
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * speed_prev.powi(typenum::P3::new());
        let roll0 = 0.5
            * mass
            * uc::ACC_GRAV
            * self.veh.chassis.wheel_rr_coef
            * vs.grade_curr.atan().cos()
            * speed_prev;
        let ascent0 = 0.5 * uc::ACC_GRAV * vs.grade_curr.atan().sin() * mass * speed_prev;
        let wheel0 = -0.5
            * self.veh.chassis.wheel_inertia
            * self.veh.chassis.num_wheels as f64
            * speed_prev.powi(typenum::P2::new())
            / (dt
                * self
                    .veh
                    .chassis
                    .wheel_radius
                    .unwrap()
                    .powi(typenum::P2::new()));

        let t3 = drag3;
        let t2 = accel2 + drag2 + wheel2;
        let t1 = drag1 + roll1 + ascent1;
        let t0 = (accel0 + drag0 + roll0 + ascent0 + wheel0) - vs.pwr_prop_fwd_max;

        // initial guess
        let speed_guess = (1e-3 * uc::MPS).max(cyc_speed);
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
        if almost_eq_uom(&pwr_err, &(0. * uc::W), Some(1e-6)) {
            vs.speed_ach = cyc_speed;
            return Ok(());
        }
        let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
        let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
        let mut speed_guesses = vec![speed_guess];
        let mut pwr_errs = vec![pwr_err];
        let mut d_pwr_err_per_d_speed_guesses = vec![pwr_err_per_speed_guess];
        let mut new_speed_guesses = vec![new_speed_guess];
        // speed achieved iteration counter
        let mut spd_ach_iter_counter = 1;
        let mut converged = pwr_err <= si::Power::ZERO;
        while spd_ach_iter_counter < max_iter && !converged {
            let speed_guess = *speed_guesses.iter().last().with_context(|| format_dbg!())?
                * (1.0 - g)
                - g * *new_speed_guesses
                    .iter()
                    .last()
                    .with_context(|| format_dbg!())?
                    / d_pwr_err_per_d_speed_guesses[speed_guesses.len() - 1];
            let pwr_err = pwr_err_fn(speed_guess);
            let pwr_err_per_speed_guess = pwr_err_per_speed_guess_fn(speed_guess);
            let new_speed_guess = pwr_err - speed_guess * pwr_err_per_speed_guess;
            speed_guesses.push(speed_guess);
            pwr_errs.push(pwr_err);
            d_pwr_err_per_d_speed_guesses.push(pwr_err_per_speed_guess);
            new_speed_guesses.push(new_speed_guess);
            // is the fractional change between previous and current speed guess smaller than `xtol`
            converged = ((*speed_guesses.iter().last().with_context(|| format_dbg!())?
                - speed_guesses[speed_guesses.len() - 2])
                / speed_guesses[speed_guesses.len() - 2])
                .abs()
                < xtol;
            spd_ach_iter_counter += 1;

            // TODO: verify that assuming `speed_guesses.iter().last()` is the correct solution
            vs.speed_ach = speed_guesses
                .last()
                .with_context(|| format_dbg!("should have had at least one element"))?
                .max(0.0 * uc::MPS);
        }

        // Run it again to make sure it has been updated for achieved speed
        self.set_pwr_prop_for_speed(self.veh.state.speed_ach, speed_prev, dt)
            .with_context(|| format_dbg!())?;

        Ok(())
    }

    pub fn to_fastsim2(&self) -> anyhow::Result<fastsim_2::simdrive::RustSimDrive> {
        let veh2 = self
            .veh
            .to_fastsim2()
            .with_context(|| anyhow!(format_dbg!()))?;
        let cyc2 = self
            .cyc
            .to_fastsim2()
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(fastsim_2::simdrive::RustSimDrive::new(cyc2, veh2))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, HistoryMethods)]
pub struct TraceMissTolerance {
    tol_dist: si::Length,
    tol_dist_frac: si::Ratio,
    tol_speed: si::Velocity,
    tol_speed_frac: si::Ratio,
}

impl SerdeAPI for TraceMissTolerance {}
impl Init for TraceMissTolerance {}

impl Default for TraceMissTolerance {
    fn default() -> Self {
        Self {
            // TODO: update these values
            tol_dist: 666. * uc::M,
            tol_dist_frac: 666. * uc::R,
            tol_speed: 666. * uc::MPS,
            tol_speed_frac: 666. * uc::R,
        }
    }
}

#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq)]
pub enum TraceMissOptions {
    /// Allow trace miss without any fanfare
    Allow,
    #[default]
    /// Error out when trace miss happens
    Error,
    /// Correct trace miss with driver model that catches up
    Correct,
}

impl SerdeAPI for TraceMissOptions {}
impl Init for TraceMissOptions {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::vehicle_model::tests::*;

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_conv() {
        let _veh = mock_conv_veh();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = SimDrive {
            veh: _veh,
            cyc: _cyc,
            sim_params: Default::default(),
        };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fc().unwrap().state.energy_fuel > si::Energy::ZERO);
    }

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_hev() {
        let _veh = mock_hev();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = SimDrive {
            veh: _veh,
            cyc: _cyc,
            sim_params: Default::default(),
        };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fc().unwrap().state.energy_fuel > si::Energy::ZERO);
        assert!(sd.veh.res().unwrap().state.energy_out_chemical != si::Energy::ZERO);
    }
}
