use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::imports::*;
use crate::prelude::*;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, StateMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Solver parameters
pub struct SimParams {
    #[serde(default = "SimParams::def_ach_speed_max_iter")]
    /// max number of iterations allowed in setting achieved speed when trace
    /// cannot be achieved
    pub ach_speed_max_iter: u32,
    #[serde(default = "SimParams::def_ach_speed_tol")]
    /// tolerance in change in speed guess in setting achieved speed when trace
    /// cannot be achieved
    pub ach_speed_tol: si::Ratio,
    #[serde(default = "SimParams::def_ach_speed_solver_gain")]
    /// Newton method gain for setting achieved speed
    pub ach_speed_solver_gain: f64,
    // TODO: plumb this up to actually do something
    /// When implemented, this will set the tolerance on how much trace miss
    /// is allowed
    #[serde(default = "SimParams::def_trace_miss_tol")]
    pub trace_miss_tol: TraceMissTolerance,
    #[serde(default = "SimParams::def_trace_miss_opts")]
    pub trace_miss_opts: TraceMissOptions,
    /// whether to use FASTSim-2 style air density
    #[serde(default = "SimParams::def_f2_const_air_density")]
    pub f2_const_air_density: bool,
}

#[named_struct_pyo3_api]
impl SimParams {
    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }
}

impl SimParams {
    fn def_ach_speed_max_iter() -> u32 {
        Self::default().ach_speed_max_iter
    }
    fn def_ach_speed_tol() -> si::Ratio {
        Self::default().ach_speed_tol
    }
    fn def_ach_speed_solver_gain() -> f64 {
        Self::default().ach_speed_solver_gain
    }
    fn def_trace_miss_tol() -> TraceMissTolerance {
        Self::default().trace_miss_tol
    }
    fn def_trace_miss_opts() -> TraceMissOptions {
        Self::default().trace_miss_opts
    }
    fn def_f2_const_air_density() -> bool {
        Self::default().f2_const_air_density
    }
}

impl SerdeAPI for SimParams {}
impl Init for SimParams {}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            ach_speed_max_iter: 3,
            ach_speed_tol: 1.0e-3 * uc::R,
            ach_speed_solver_gain: 0.9,
            trace_miss_tol: Default::default(),
            trace_miss_opts: Default::default(),
            f2_const_air_density: true,
        }
    }
}

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, StateMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct SimDrive {
    #[has_state]
    pub veh: Vehicle,
    pub cyc: Cycle,
    pub sim_params: SimParams,
}

#[named_struct_pyo3_api]
impl SimDrive {
    #[new]
    #[pyo3(signature = (veh, cyc, sim_params=None))]
    fn __new__(veh: Vehicle, cyc: Cycle, sim_params: Option<SimParams>) -> anyhow::Result<Self> {
        Ok(SimDrive::new(veh, cyc, sim_params))
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
}

impl SerdeAPI for SimDrive {}
impl Init for SimDrive {
    fn init(&mut self) -> Result<(), Error> {
        self.veh
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.cyc
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.sim_params
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
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
    // ## Features
    // - [ ] regen limiting curve during speeds approaching zero per f2 -- less urgent
    // - [ ] ability to manipulate friction/regen brake split based on required braking
    //       power -- new feature -- move this to enum
    // - [x] make enum `EngineOnCause::{AlreadyOn, TooCold,
    //       PowerDemand}` and save it in a vec or some such for when there are
    //       multiple causes -- new feature

    /// Run vehicle simulation, and, if applicable, apply powertrain-specific
    /// corrections (e.g. iterate `walk` until SOC balance is achieved -- i.e. initial
    /// and final SOC are nearly identical)
    pub fn walk(&mut self) -> anyhow::Result<()> {
        match self.veh.pt_type {
            PowertrainType::HybridElectricVehicle(_) => {
                // Net battery energy used per amount of fuel used
                // clone initial vehicle to preserve starting state (TODO: figure out if this is a huge CPU burden)
                let veh_init = self.veh.clone();
                loop {
                    self.veh
                        .hev_mut()
                        .with_context(|| format_dbg!())?
                        .soc_bal_iters
                        .mark_stale();
                    self.veh
                        .hev_mut()
                        .with_context(|| format_dbg!())?
                        .soc_bal_iters
                        .increment(1, || format_dbg!())?;
                    self.walk_once().with_context(|| format_dbg!())?;
                    let soc_final = self
                        .veh
                        .res()
                        .with_context(|| format_dbg!())?
                        .state
                        .soc
                        .clone();
                    let res_per_fuel = *self
                        .veh
                        .res()
                        .with_context(|| format_dbg!())?
                        .state
                        .energy_out_chemical
                        .get_fresh(|| format_dbg!())?
                        / *self
                            .veh
                            .fc()
                            .with_context(|| format_dbg!())?
                            .state
                            .energy_fuel
                            .get_fresh(|| format_dbg!())?;
                    if self
                        .veh
                        .hev()
                        .with_context(|| format_dbg!())?
                        .soc_bal_iters
                        .get_fresh(|| format_dbg!())?
                        > &self
                            .veh
                            .hev()
                            .with_context(|| format_dbg!())?
                            .sim_params
                            .soc_balance_iter_err
                    {
                        bail!(
                            "{}",
                            format_dbg!((
                                self.veh
                                    .hev()
                                    .with_context(|| format_dbg!())?
                                    .soc_bal_iters
                                    .clone(),
                                self.veh
                                    .hev()
                                    .with_context(|| format_dbg!())?
                                    .sim_params
                                    .soc_balance_iter_err
                            ))
                        );
                    }
                    if res_per_fuel.abs()
                        < self
                            .veh
                            .hev()
                            .with_context(|| format_dbg!())?
                            .sim_params
                            .res_per_fuel_lim
                        || !self
                            .veh
                            .hev()
                            .with_context(|| format_dbg!())?
                            .sim_params
                            .balance_soc
                    {
                        break;
                    } else {
                        // prep for another iteration
                        if let Some(&mut ref mut hev) = self.veh.hev_mut() {
                            if hev.sim_params.save_soc_bal_iters {
                                hev.soc_bal_iter_history.push(hev.clone());
                                hev.soc_bal_iters.mark_stale();
                            }
                        }
                        // reset vehicle to initial state
                        self.veh = veh_init.clone();
                        // start SOC at previous final value
                        self.veh.res_mut().with_context(|| format_dbg!())?.state.soc = soc_final;
                    }
                }
            }
            _ => self.walk_once()?,
        }
        Ok(())
    }

    /// Run vehicle simulation once
    pub fn walk_once(&mut self) -> anyhow::Result<()> {
        let len = &self.cyc.len_checked().with_context(|| format_dbg!())?;
        ensure!(len >= &2, format_dbg!(len < &2));
        self.save_state(|| format_dbg!())?;

        self.veh.state.mass.mark_stale();
        self.veh.state.mass.update(
            self.veh
                .mass()
                .with_context(|| format_dbg!())?
                .with_context(|| format_dbg!("Expected mass to have been set."))?,
            || format_dbg!(),
        )?;

        loop {
            self.check_and_reset(|| format_dbg!())?;
            self.veh.state.mass.mark_fresh(|| format_dbg!())?;
            if let Some(res) = self.veh.res_mut() {
                res.state.soh.mark_fresh(|| format_dbg!())?;
            }
            self.step(|| format_dbg!())?;
            self.solve_step()
                .with_context(|| format!("{}\ntime step: {:?}", format_dbg!(), self.veh.state.i))?;
            self.save_state(|| format_dbg!())?;
            if *self.veh.state.i.get_fresh(|| format_dbg!())? == len - 1 {
                break;
            }
        }
        Ok(())
    }

    /// Solves current time step
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        let i = *self.veh.state.i.get_fresh(|| format_dbg!())?;
        self.veh
            .state
            .time
            .update(self.cyc.time[i], || format_dbg!())?;
        let dt = self.cyc.dt_at_i(i)?;
        // maybe make controls like:
        // ```
        // pub enum HVACAuxPriority {
        //     /// Prioritize [ReversibleEnergyStorage] thermal management
        //     ReversibleEnergyStorage
        //     /// Prioritize [Cabin] and [ReversibleEnergyStorage] proportionally to their requests
        //     Proportional
        // }
        // ```

        // `solve_thermal` must happen before the other methods because it impacts aux power demand
        self.veh
            .solve_thermal(self.cyc.temp_amb_air[i], dt)
            .with_context(|| format_dbg!())?;
        self.veh
            // TODO: feed in something different for first arg when CAVs stuff is active @MOK
            .set_curr_pwr_out_max(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.set_pwr_prop_for_speed(
            self.cyc.speed[i],
            *self.veh.state.speed_ach.get_stale(|| format_dbg!())?,
            dt,
        )
        .with_context(|| anyhow!(format_dbg!()))?;
        self.veh.state.pwr_tractive_for_cyc.update(
            *self.veh.state.pwr_tractive.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        self.set_ach_speed(self.cyc.speed[i], dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        if self.sim_params.trace_miss_opts.is_allow_checked() {
            self.sim_params.trace_miss_tol.check_trace_miss(
                self.cyc.speed[i],
                *self.veh.state.speed_ach.get_fresh(|| format_dbg!())?,
                self.cyc.dist[i],
                *self.veh.state.dist.get_fresh(|| format_dbg!())?,
            )?;
        }
        self.veh
            .solve_powertrain(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.veh.set_cumulative(dt)?;
        Ok(())
    }

    /// Sets power required for given prescribed speed
    /// # Arguments
    /// - `speed`: prescribed or achieved speed
    /// - `dt`: simulation time step size
    pub fn set_pwr_prop_for_speed(
        &mut self,
        speed: si::Velocity,
        speed_prev: si::Velocity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let i = *self.veh.state.i.get_fresh(|| format_dbg!())?;
        let vs = &mut self.veh.state;
        // TODO: get @mokeefe to give this a serious look and think about grade alignment issues that may arise
        let interp_pt_dist: &[f64] = match self.cyc.grade_interp {
            Some(Interpolator::Interp0D(..)) => &[],
            Some(Interpolator::Interp1D(..)) => {
                &[vs.dist.get_fresh(|| format_dbg!())?.get::<si::meter>()]
            }
            _ => unreachable!(),
        };
        vs.grade_curr.update(
            if *vs.cyc_met_overall.get_stale(|| format_dbg!())? {
                *self
                    .cyc
                    .grade
                    .get(i)
                    .with_context(|| format_dbg!(self.cyc.grade.len()))?
            } else {
                uc::R
                    * self
                        .cyc
                        .grade_interp
                        .as_ref()
                        .with_context(|| format_dbg!("You might have somehow bypassed `init()`"))?
                        .interpolate(interp_pt_dist)
                        .with_context(|| format_dbg!())?
            },
            || format_dbg!(),
        )?;
        vs.elev_curr.update(
            if *vs.cyc_met_overall.get_stale(|| format_dbg!())? {
                *self.cyc.elev.get(i).with_context(|| format_dbg!())?
            } else {
                uc::M
                    * self
                        .cyc
                        .elev_interp
                        .as_ref()
                        .with_context(|| format_dbg!("You might have somehow bypassed `init()`"))?
                        .interpolate(interp_pt_dist)
                        .with_context(|| format_dbg!())?
            },
            || format_dbg!(),
        )?;

        vs.air_density.update(
            if self.sim_params.f2_const_air_density {
                1.2 * uc::KGPM3
            } else {
                let te_amb_air = {
                    let te_amb_air = self
                        .cyc
                        .temp_amb_air
                        .get(i)
                        .with_context(|| format_dbg!())?;
                    if *te_amb_air == *TE_STD_AIR {
                        None
                    } else {
                        Some(te_amb_air)
                    }
                };
                Air::get_density(
                    te_amb_air.copied(),
                    Some(*vs.elev_curr.get_fresh(|| format_dbg!())?),
                )
            },
            || format_dbg!(),
        )?;

        let mass = self.veh.mass.with_context(|| {
            format!(
                "{}\nVehicle mass should have been set already.",
                format_dbg!()
            )
        })?;
        vs.pwr_accel.update(
            mass / (2.0 * dt)
                * (speed.powi(typenum::P2::new()) - speed_prev.powi(typenum::P2::new())),
            || format_dbg!(),
        )?;
        vs.pwr_ascent.update(
            uc::ACC_GRAV
                * *vs.grade_curr.get_fresh(|| format_dbg!())?
                * mass
                * (speed_prev + speed)
                / 2.0,
            || format_dbg!(),
        )?;
        vs.pwr_drag.update(
            0.5
            // TODO: feed in elevation
            * Air::get_density(None, None)
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * ((speed + speed_prev) / 2.0).powi(typenum::P3::new()),
            || format_dbg!(),
        )?;
        vs.pwr_rr.update(
            mass * uc::ACC_GRAV
                * self.veh.chassis.wheel_rr_coef
                * vs.grade_curr.get_fresh(|| format_dbg!())?.atan().cos()
                * (speed_prev + speed)
                / 2.,
            || format_dbg!(),
        )?;
        vs.pwr_whl_inertia.update(
            0.5 * self.veh.chassis.wheel_inertia
                * self.veh.chassis.num_wheels as f64
                * ((speed
                    / self
                        .veh
                        .chassis
                        .wheel_radius
                        .with_context(|| format_dbg!())?)
                .powi(typenum::P2::new())
                    - (speed_prev
                        / self
                            .veh
                            .chassis
                            .wheel_radius
                            .with_context(|| format_dbg!())?)
                    .powi(typenum::P2::new()))
                / self.cyc.dt_at_i(i).with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;

        vs.pwr_tractive.update(
            *vs.pwr_rr.get_fresh(|| format_dbg!())?
                + *vs.pwr_whl_inertia.get_fresh(|| format_dbg!())?
                + *vs.pwr_accel.get_fresh(|| format_dbg!())?
                + *vs.pwr_ascent.get_fresh(|| format_dbg!())?
                + *vs.pwr_drag.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - `cyc_speed`: prescribed speed
    /// - `dt`: simulation time step size
    pub fn set_ach_speed(&mut self, cyc_speed: si::Velocity, dt: si::Time) -> anyhow::Result<()> {
        let vs = &mut self.veh.state;
        vs.cyc_met.update(
            vs.pwr_tractive.get_fresh(|| format_dbg!())?
                <= vs.pwr_prop_fwd_max.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        vs.cyc_met_overall.update(
            if !*vs.cyc_met.get_fresh(|| format_dbg!())? {
                // if current power demand is not met, then this becomes false for
                // the rest of the cycle and should not be manipulated anywhere else
                false
            } else {
                *vs.cyc_met_overall.get_stale(|| format_dbg!())?
            },
            || format_dbg!(),
        )?;
        let veh = &mut self.veh;
        let speed_prev = *veh.state.speed_ach.get_stale(|| format_dbg!())?;
        if *veh.state.cyc_met.get_fresh(|| format_dbg!())? {
            veh.state.speed_ach.update(cyc_speed, || format_dbg!())?;
            return Ok(());
        } else {
            match self.sim_params.trace_miss_opts {
                TraceMissOptions::Allow => {
                    // do nothing because `set_ach_speed` should be allowed to proceed to handle this
                }
                TraceMissOptions::AllowChecked => {
                    // this will be handled later
                }
                TraceMissOptions::Error => bail!(
                    "{}\nFailed to meet speed trace.
prescribed speed: {} mph
prev speed_ach: {} mph
pwr_tractive_for_cyc: {} kW
pwr_tractive: {} kW
pwr_prop_fwd_max: {} kW,
pwr deficit: {} kW
",
                    format_dbg!(),
                    cyc_speed.get::<si::mile_per_hour>(),
                    veh.state
                        .speed_ach
                        .get_stale(|| format_dbg!())?
                        .get::<si::mile_per_hour>(),
                    veh.state
                        .pwr_tractive_for_cyc
                        .get_fresh(|| format_dbg!())?
                        .get::<si::kilowatt>(),
                    veh.state
                        .pwr_tractive
                        .get_fresh(|| format_dbg!())?
                        .get::<si::kilowatt>(),
                    veh.state
                        .pwr_prop_fwd_max
                        .get_fresh(|| format_dbg!())?
                        .get::<si::kilowatt>(),
                    (*veh.state.pwr_tractive.get_fresh(|| format_dbg!())?
                        - *veh.state.pwr_prop_fwd_max.get_fresh(|| format_dbg!())?)
                    .get::<si::kilowatt>()
                    .format_eng(None)
                ),
                TraceMissOptions::Correct => todo!(),
            }
        }
        let vs = &mut self.veh.state;
        let mass = self
            .veh
            .mass
            .with_context(|| format!("{}\nMass should have been set before now", format_dbg!()))?;

        let drag3 = 1.0 / 16.0
            * *vs.air_density.get_fresh(|| format_dbg!())?
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area;
        let accel2 = 0.5 * mass / dt;
        let drag2 = 3.0 / 16.0
            * *vs.air_density.get_fresh(|| format_dbg!())?
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * speed_prev;
        let wheel2 = 0.5 * self.veh.chassis.wheel_inertia * self.veh.chassis.num_wheels as f64
            / (dt
                * self
                    .veh
                    .chassis
                    .wheel_radius
                    .with_context(|| format_dbg!())?
                    .powi(typenum::P2::new()));
        let drag1 = 3.0 / 16.0
            * *vs.air_density.get_fresh(|| format_dbg!())?
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * vs.speed_ach
                .get_fresh(|| format_dbg!())?
                .powi(typenum::P2::new());
        let roll1 = 0.5
            * mass
            * uc::ACC_GRAV
            * self.veh.chassis.wheel_rr_coef
            * vs.grade_curr.get_fresh(|| format_dbg!())?.atan().cos();
        let ascent1 =
            0.5 * uc::ACC_GRAV * vs.grade_curr.get_fresh(|| format_dbg!())?.atan().sin() * mass;
        let accel0 = -0.5 * mass * speed_prev.powi(typenum::P2::new()) / dt;
        let drag0 = 1.0 / 16.0
            * *vs.air_density.get_fresh(|| format_dbg!())?
            * self.veh.chassis.drag_coef
            * self.veh.chassis.frontal_area
            * speed_prev.powi(typenum::P3::new());
        let roll0 = 0.5
            * mass
            * uc::ACC_GRAV
            * self.veh.chassis.wheel_rr_coef
            * vs.grade_curr.get_fresh(|| format_dbg!())?.atan().cos()
            * speed_prev;
        let ascent0 = 0.5
            * uc::ACC_GRAV
            * vs.grade_curr.get_fresh(|| format_dbg!())?.atan().sin()
            * mass
            * speed_prev;
        let wheel0 = -0.5
            * self.veh.chassis.wheel_inertia
            * self.veh.chassis.num_wheels as f64
            * speed_prev.powi(typenum::P2::new())
            / (dt
                * self
                    .veh
                    .chassis
                    .wheel_radius
                    .with_context(|| format_dbg!())?
                    .powi(typenum::P2::new()));

        let t3 = drag3;
        let t2 = accel2 + drag2 + wheel2;
        let t1 = drag1 + roll1 + ascent1;
        let t0 = (accel0 + drag0 + roll0 + ascent0 + wheel0)
            - *vs.pwr_prop_fwd_max.get_fresh(|| format_dbg!())?;

        // initial guess
        let speed_guess = (1e-3 * uc::MPS).max(cyc_speed);
        // stop criteria
        let max_iter = &self.sim_params.ach_speed_max_iter;
        let xtol = &self.sim_params.ach_speed_tol;
        // solver gain
        let g = &self.sim_params.ach_speed_solver_gain;
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
            // TODO: maybe change to `updated_unchecked`
            vs.speed_ach.update(cyc_speed, || format_dbg!())?;
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
        let mut speed_ach: si::Velocity = Default::default();
        while &spd_ach_iter_counter < max_iter && !converged {
            let speed_guess = *speed_guesses.iter().last().with_context(|| format_dbg!())?
                * (1.0 - g)
                - *g * *new_speed_guesses
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
            converged = &((*speed_guesses.iter().last().with_context(|| format_dbg!())?
                - speed_guesses[speed_guesses.len() - 2])
                / speed_guesses[speed_guesses.len() - 2])
                .abs()
                < xtol;
            spd_ach_iter_counter += 1;

            // TODO: verify that assuming `speed_guesses.iter().last()` is the correct solution
            speed_ach = speed_guesses
                .last()
                .with_context(|| format_dbg!("should have had at least one element"))?
                .max(0.0 * uc::MPS);
        }

        vs.speed_ach.update(speed_ach, || format_dbg!())?;

        // Run it again to make sure it has been updated for achieved speed
        self.set_pwr_prop_for_speed(
            *self.veh.state.speed_ach.get_fresh(|| format_dbg!())?,
            speed_prev,
            dt,
        )
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, StateMethods)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
// NOTE: consider embedding this in TraceMissOptions::AllowChecked
pub struct TraceMissTolerance {
    /// if the vehicle falls this far behind trace in terms of absolute
    /// difference and [TraceMissOptions::is_allow_checked], fail
    tol_dist: si::Length,
    /// if the vehicle falls this far behind trace in terms of fractional
    /// difference and [TraceMissOptions::is_allow_checked], fail
    tol_dist_frac: si::Ratio,
    /// if the vehicle falls this far behind instantaneous speed and
    /// [TraceMissOptions::is_allow_checked], fail
    tol_speed: si::Velocity,
    /// if the vehicle falls this far behind instantaneous speed in terms of
    /// fractional difference and [TraceMissOptions::is_allow_checked], fail
    tol_speed_frac: si::Ratio,
}

impl TraceMissTolerance {
    fn check_trace_miss(
        &self,
        cyc_speed: si::Velocity,
        ach_speed: si::Velocity,
        cyc_dist: si::Length,
        ach_dist: si::Length,
    ) -> anyhow::Result<()> {
        ensure!(
            cyc_speed - ach_speed < self.tol_speed,
            "{}\n{}\n{}",
            format_dbg!(cyc_speed),
            format_dbg!(ach_speed),
            format_dbg!(self.tol_speed)
        );
        // if condition to prevent divide-by-zero errors
        if cyc_speed > self.tol_speed {
            ensure!(
                (cyc_speed - ach_speed) / cyc_speed < self.tol_speed_frac,
                "{}\n{}\n{}",
                format_dbg!(cyc_speed),
                format_dbg!(ach_speed),
                format_dbg!(self.tol_speed_frac)
            )
        }
        ensure!(
            (cyc_dist - ach_dist) < self.tol_dist,
            "{}\n{}\n{}",
            format_dbg!(cyc_dist),
            format_dbg!(ach_dist),
            format_dbg!(self.tol_dist)
        );
        // if condition to prevent checking early in cycle
        if cyc_dist > self.tol_dist * 5.0 {
            ensure!(
                (cyc_dist - ach_dist) / cyc_dist < self.tol_dist_frac,
                "{}\n{}\n{}",
                format_dbg!(cyc_dist),
                format_dbg!(ach_dist),
                format_dbg!(self.tol_dist_frac)
            )
        }

        Ok(())
    }
}
impl SerdeAPI for TraceMissTolerance {}
impl Init for TraceMissTolerance {}
impl Default for TraceMissTolerance {
    fn default() -> Self {
        Self {
            tol_dist: 100. * uc::M,
            tol_dist_frac: 0.05 * uc::R,
            tol_speed: 10. * uc::MPS,
            tol_speed_frac: 0.5 * uc::R,
        }
    }
}

#[derive(
    Clone, Default, Debug, Deserialize, Serialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum TraceMissOptions {
    /// Allow trace miss without any fanfare
    Allow,
    /// Allow trace miss within error tolerance
    AllowChecked,
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
        let mut sd = SimDrive::new(_veh, _cyc, Default::default());
        sd.walk().unwrap();
        assert!(
            *sd.veh.state.i.get_fresh(String::new).unwrap() == sd.cyc.len_checked().unwrap() - 1
        );
        assert!(
            *sd.veh
                .fc()
                .unwrap()
                .state
                .energy_fuel
                .get_fresh(String::new)
                .unwrap()
                > si::Energy::ZERO
        );
        assert!(sd.veh.res().is_none());
    }

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_hev() {
        let _veh = mock_hev();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = SimDrive::new(_veh, _cyc, Default::default());
        sd.walk().unwrap();
        assert!(
            *sd.veh.state.i.get_fresh(String::new).unwrap() == sd.cyc.len_checked().unwrap() - 1
        );
        assert!(
            *sd.veh
                .fc()
                .unwrap()
                .state
                .energy_fuel
                .get_fresh(String::new)
                .unwrap()
                > si::Energy::ZERO
        );
        assert!(
            *sd.veh
                .res()
                .unwrap()
                .state
                .energy_out_chemical
                .get_fresh(String::new)
                .unwrap()
                != si::Energy::ZERO
        );
    }

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_bev() {
        let _veh = mock_bev();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = SimDrive {
            veh: _veh,
            cyc: _cyc,
            sim_params: Default::default(),
        };
        sd.walk().unwrap();
        assert!(
            *sd.veh.state.i.get_fresh(String::new).unwrap() == sd.cyc.len_checked().unwrap() - 1
        );
        assert!(sd.veh.fc().is_none());
        assert!(
            *sd.veh
                .res()
                .unwrap()
                .state
                .energy_out_chemical
                .get_fresh(String::new)
                .unwrap()
                != si::Energy::ZERO
        );
    }
}
