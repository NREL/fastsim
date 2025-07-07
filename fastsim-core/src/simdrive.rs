pub mod roadload;

use roadload::StepInfo;

use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::drive_cycle::manipulation_utils::calc_best_rendezvous;
use crate::imports::*;
use crate::prelude::*;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
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
    #[serde(default = "SimParams::def_trace_miss_correct_max_steps")]
    /// the maximum number of steps in which to re-rendezvous with reference
    /// trace after a trace miss. Note: this field only applies when
    /// trace_miss_opts is set to TraceMissOptions::Correct. Note: must
    /// be 2 or greater. Defaults to 6.
    pub trace_miss_correct_max_steps: u32,
    /// whether to use FASTSim-2 style air density
    #[serde(default = "SimParams::def_f2_const_air_density")]
    pub f2_const_air_density: bool,
    /// if true, vehicle is totally inactive except for thermal models
    pub ambient_thermal_soak: bool,
}

#[pyo3_api]
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
    fn def_trace_miss_correct_max_steps() -> u32 {
        Self::default().trace_miss_correct_max_steps
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
            trace_miss_correct_max_steps: 6,
            f2_const_air_density: true,
            ambient_thermal_soak: false,
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

#[pyo3_api]
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
    /// corrections:
    /// - for HEV, set initial SOC to mean of min and max SOC, and then iterate
    ///   `walk` until SOC balance is achieved -- i.e. initial and final SOC are
    ///   nearly identical
    /// - for PHEV, set initial SOC to max SOC, and then simulate once
    /// - for BEV, set initial SOC to max SOC, and then simulate once
    /// - for Conv, simulate once
    ///
    /// # Important Considerations
    /// If you need to run a [ReversibleEnergyStorage]-equipped vehicle for
    /// only one iteration without modifying the initial SOC, then run the
    /// [Self::walk_once] method directly
    pub fn walk(&mut self) -> anyhow::Result<()> {
        match self.veh.pt_type {
            PowertrainType::HybridElectricVehicle(_) => {
                // Net battery energy used per amount of fuel used
                // clone initial vehicle to preserve starting state (TODO: figure out if this is a huge CPU burden)
                let veh_init = self.veh.clone();
                let res_mut = self.veh.res_mut().with_context(|| format_dbg!())?;
                res_mut.state.soc.mark_stale();
                res_mut
                    .state
                    .soc
                    .update(0.5 * (res_mut.min_soc + res_mut.max_soc), || format_dbg!())?;
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
                        || self.sim_params.ambient_thermal_soak
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
            PowertrainType::PlugInHybridElectricVehicle(_) => {
                let res_mut = self.veh.res_mut().with_context(|| format_dbg!())?;
                res_mut.state.soc.mark_stale();
                res_mut
                    .state
                    .soc
                    .update(res_mut.max_soc, || format_dbg!())?;
                self.walk_once()?
            }
            PowertrainType::BatteryElectricVehicle(_) => {
                let res_mut = self.veh.res_mut().with_context(|| format_dbg!())?;
                res_mut.state.soc.mark_stale();
                res_mut
                    .state
                    .soc
                    .update(res_mut.max_soc, || format_dbg!())?;
                self.walk_once()?
            }
            PowertrainType::ConventionalVehicle(_) => self.walk_once()?,
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

    /// Calculates the derivative dv/dd (change in speed by change in distance)
    /// - speed_m_per_s: the speed at which to evaluate dv/dd (m/s)
    /// - grade: the road grade as a decimal fraction
    ///
    /// RETURN: number, the dv/dd for these conditions
    pub fn calc_dvdd(&self, speed_m_per_s: f64, grade: f64) -> anyhow::Result<f64> {
        let v = speed_m_per_s;
        if v <= 0.0 {
            Ok(0.0)
        } else {
            let (atan_grade_sin, atan_grade_cos) = if grade == 0.0 {
                (0.0, 1.0)
            } else {
                let atan_grade = grade.atan();
                (atan_grade.sin(), atan_grade.cos())
            };
            let g = uc::ACC_GRAV.get::<si::meter_per_second_squared>();
            let m = self
                .veh
                .mass
                .with_context(|| {
                    format!(
                        "{}\nVehicle mass should have been set already.",
                        format_dbg!()
                    )
                })?
                .get::<si::kilogram>();
            let rho_cdfa = self
                .veh
                .state
                .air_density
                .get_stale(|| format_dbg!())?
                .get::<si::kilogram_per_cubic_meter>()
                * self.veh.chassis.drag_coef.get::<si::ratio>()
                * self.veh.chassis.frontal_area.get::<si::square_meter>();
            let rrc = self.veh.chassis.wheel_rr_coef.get::<si::ratio>();
            Ok(-1.0
                * ((g / v) * (atan_grade_sin + rrc * atan_grade_cos)
                    + (0.5 * rho_cdfa * (1.0 / m) * v)))
        }
    }

    /// Solves current time step
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        let i = *self.veh.state.i.get_fresh(|| format_dbg!())?;
        let time_prev = *self.veh.state.time.get_stale(|| format_dbg!())?;
        self.veh.state.time.update(
            *self.cyc.time.get(i).with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        let dt = *self.veh.state.time.get_fresh(|| format_dbg!())? - time_prev;
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
        match self.sim_params.ambient_thermal_soak {
            false => {
                self.veh
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
            }
            true => {
                self.veh.mark_non_thermal_fresh()?;
            }
        }
        self.set_cumulative(dt, || format_dbg!())?;
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
        // TODO: memo-ize this
        //     - if we get back on trace or nearly back on trace, revert to just using the index
        //     - we can also shorten the x and y values by removing stuff that's already happened
        let interp_pt_dist: &[f64] = match self.cyc.grade_interp {
            Some(InterpolatorEnum::Interp0D(_)) => &[],
            Some(InterpolatorEnum::Interp1D(_)) => {
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
            mass / (2.0 * dt) * (speed.powi(P2::new()) - speed_prev.powi(P2::new())),
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
            * ((speed + speed_prev) / 2.0).powi(P3::new()),
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
                .powi(P2::new())
                    - (speed_prev
                        / self
                            .veh
                            .chassis
                            .wheel_radius
                            .with_context(|| format_dbg!())?)
                    .powi(P2::new()))
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
                TraceMissOptions::Correct => {
                    // We will correct the deviation from trace by modifying the cycle to re-rendezvous with a later time/distance.
                    // In so doing, we will use a less agressive roadload.
                    // NOTE: actual correction occurs later but we need to calculate
                    // the achieved speed first.
                }
            }
        }
        let vs = &mut self.veh.state;
        let step_info = StepInfo {
            dt,
            speed_prev,
            cyc_speed,
            grade_curr: *vs.grade_curr.get_fresh(|| format_dbg!())?,
            air_density: *vs.air_density.get_fresh(|| format_dbg!())?,
            mass: self.veh.mass.with_context(|| {
                format!("{}\nMass should have been set before now", format_dbg!())
            })?,
            drag_coef: self.veh.chassis.drag_coef,
            frontal_area: self.veh.chassis.frontal_area,
            wheel_inertia: self.veh.chassis.wheel_inertia,
            num_wheels: self.veh.chassis.num_wheels,
            wheel_radius: self
                .veh
                .chassis
                .wheel_radius
                .with_context(|| format_dbg!())?,
            wheel_rr_coef: self.veh.chassis.wheel_rr_coef,
            pwr_prop_fwd_max: *vs.pwr_prop_fwd_max.get_fresh(|| format_dbg!())?,
        };
        let speed_ach = step_info.solve_for_speed(
            self.sim_params.ach_speed_max_iter * 10,
            self.sim_params.ach_speed_tol,
            self.sim_params.ach_speed_solver_gain,
        );
        let speed_ach_floored = {
            // NOTE: what we are doing here is "flooring" the speed to the nearest tenth of a m/s.
            // The purpose is to slightly reduce the target speed below the max power threshold
            // to prevent float precision issues from sending us right back into trace miss.
            let v = ((speed_ach.get::<si::meter_per_second>() * 10.0).floor() / 10.0) * uc::MPS;
            // NOTE: if after "flooring" we happen to exactly be the same as
            // previous, we subtract off a tenth of a m/s but prevent going below 0 m/s.
            if v == speed_ach {
                (v - 0.1 * uc::MPS).max(si::Velocity::ZERO)
            } else {
                v
            }
        };

        vs.speed_ach.update(speed_ach_floored, || format_dbg!())?;
        // NOTE: need to reset tracked state to allow
        // for calling set_pwr_prop_for_speed(.) again this step.
        // set_pwr_prop_for_speed has already been called so the
        // following variables have already been set fresh but need
        // to be re-iterated.
        vs.air_density.mark_stale();
        vs.cyc_met.mark_stale();
        vs.cyc_met_overall.mark_stale();
        vs.elev_curr.mark_stale();
        vs.grade_curr.mark_stale();
        vs.pwr_accel.mark_stale();
        vs.pwr_ascent.mark_stale();
        vs.pwr_drag.mark_stale();
        vs.pwr_rr.mark_stale();
        vs.pwr_tractive.mark_stale();
        vs.pwr_whl_inertia.mark_stale();
        vs.speed_ach.mark_stale();

        // Rerun again to ensure we have updated achieved speed and state
        self.set_pwr_prop_for_speed(speed_ach_floored, speed_prev, dt)
            .with_context(|| format_dbg!())?;
        self.set_ach_speed(speed_ach, dt)
            .with_context(|| anyhow!(format_dbg!()))?;

        if self.sim_params.trace_miss_opts == TraceMissOptions::Correct {
            let i = *self.veh.state.i.get_fresh(|| format_dbg!())?;
            let max_steps = self.sim_params.trace_miss_correct_max_steps.max(2) as usize;
            let correction = calc_best_rendezvous(i, max_steps, &self.cyc, speed_ach_floored);
            if correction.steps >= 2 {
                // NOTE: in theory, grade could be slightly
                // off with this deviation from trace. However, since we
                // rendezvous in a small number of time steps, it should be
                // close. The call again to init() should correct distance
                // and elevation calculations.
                self.cyc.speed[i] = speed_ach_floored;
                self.cyc.modify_by_const_jerk_trajectory(
                    i + 1,
                    correction.steps,
                    correction.jerk_m_per_s3 * uc::MPS3,
                    correction.acceleration_m_per_s2 * uc::MPS2,
                );
                self.cyc.dist.clear();
                self.cyc.elev.clear();
                self.cyc.init().unwrap();
            }
        }

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

impl SetCumulative for SimDrive {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.veh
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
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
    fn test_sim_drive_hev_thrml() {
        let _veh =
            Vehicle::from_resource("2021_Hyundai_Sonata_Hybrid_Blue_thrml.yaml", false).unwrap();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();

        let te_amb: Vec<si::Temperature> = [-6.7, -6.7, 38.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        let te_batt_and_cab_init: Vec<si::Temperature> = [-6.7, 22.0, 45.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        let te_fc_init: Vec<si::Temperature> = [-6.7, 70.0, 90.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        for ((te_amb, te_init), te_fc_init) in
            te_amb.iter().zip(te_batt_and_cab_init).zip(te_fc_init)
        {
            let mut veh = _veh.clone();

            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .update(te_init, || format_dbg!())
                .unwrap();

            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .update(te_init, || format_dbg!())
                .unwrap();
            if let CabinOption::LumpedCabin(lc) = &mut veh.cabin {
                lc.state.temperature.mark_stale();
                lc.state
                    .temperature
                    .update(te_init, || format_dbg!())
                    .unwrap();
                lc.state.temp_prev.mark_stale();
                lc.state
                    .temp_prev
                    .update(te_init, || format_dbg!())
                    .unwrap();
            }

            veh.fc_mut()
                .unwrap()
                .fc_thrml_state_mut()
                .unwrap()
                .temperature
                .mark_stale();
            veh.fc_mut()
                .unwrap()
                .fc_thrml_state_mut()
                .unwrap()
                .temperature
                .update(te_fc_init, || format_dbg!())
                .unwrap();
            let mut cyc = _cyc.clone();
            cyc.temp_amb_air = vec![*te_amb; cyc.len_checked().unwrap()];
            let mut sd = SimDrive::new(veh, cyc, Default::default());
            sd.walk()
                .with_context(|| {
                    format!(
                        "ambient temperature: {}*C\ninit temperature: {}",
                        te_amb.get::<si::degree_celsius>(),
                        te_init.get::<si::degree_celsius>()
                    )
                })
                .unwrap();
            assert!(
                *sd.veh.state.i.get_fresh(String::new).unwrap()
                    == sd.cyc.len_checked().unwrap() - 1
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
    }

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_hev_thrml_soak() {
        let _veh =
            Vehicle::from_resource("2021_Hyundai_Sonata_Hybrid_Blue_thrml.yaml", false).unwrap();
        let mut cyc = Cycle::from_resource("udds.csv", false).unwrap();
        // zero out speed
        cyc.speed.iter_mut().for_each(|v| *v = si::Velocity::ZERO);

        let te_amb: Vec<si::Temperature> = [-6.7, -6.7, 38.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        let te_batt_and_cab_init: Vec<si::Temperature> = [-6.7, 22.0, 45.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        let te_fc_init: Vec<si::Temperature> = [-6.7, 70.0, 90.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        for ((te_amb, te_init), te_fc_init) in
            te_amb.iter().zip(te_batt_and_cab_init).zip(te_fc_init)
        {
            let mut veh = _veh.clone();

            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .update(te_init, || format_dbg!())
                .unwrap();

            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .update(te_init, || format_dbg!())
                .unwrap();
            if let CabinOption::LumpedCabin(lc) = &mut veh.cabin {
                lc.state.temperature.mark_stale();
                lc.state
                    .temperature
                    .update(te_init, || format_dbg!())
                    .unwrap();
                lc.state.temp_prev.mark_stale();
                lc.state
                    .temp_prev
                    .update(te_init, || format_dbg!())
                    .unwrap();
            }

            veh.fc_mut()
                .unwrap()
                .fc_thrml_state_mut()
                .unwrap()
                .temperature
                .mark_stale();
            veh.fc_mut()
                .unwrap()
                .fc_thrml_state_mut()
                .unwrap()
                .temperature
                .update(te_fc_init, || format_dbg!())
                .unwrap();
            let mut cyc = cyc.clone();
            cyc.temp_amb_air = vec![*te_amb; cyc.len_checked().unwrap()];
            let mut sd = SimDrive::new(
                veh,
                cyc,
                Some(SimParams {
                    ambient_thermal_soak: true,
                    ..Default::default()
                }),
            );
            sd.walk()
                .with_context(|| {
                    format!(
                        "ambient temperature: {}*C\ninit temperature: {}",
                        te_amb.get::<si::degree_celsius>(),
                        te_init.get::<si::degree_celsius>()
                    )
                })
                .unwrap();
            assert!(
                *sd.veh.state.i.get_fresh(String::new).unwrap()
                    == sd.cyc.len_checked().unwrap() - 1
            );
        }
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

    #[test]
    #[cfg(feature = "resources")]
    fn test_sim_drive_bev_thrml() {
        let _veh = Vehicle::from_resource("2020 Chevrolet Bolt EV thrml.yaml", false).unwrap();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();

        let te_amb: Vec<si::Temperature> = [-6.7, -6.7, 38.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        let te_batt_and_cab_init: Vec<si::Temperature> = [-6.7, 22.0, 45.0]
            .iter()
            .map(|t| (*t + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)
            .collect();
        for (te_amb, te_init) in te_amb.iter().zip(te_batt_and_cab_init) {
            let mut veh = _veh.clone();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temperature
                .update(te_init, || format_dbg!())
                .unwrap();

            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .mark_stale();
            veh.res_mut()
                .unwrap()
                .res_thrml_state_mut()
                .unwrap()
                .temp_prev
                .update(te_init, || format_dbg!())
                .unwrap();

            if let CabinOption::LumpedCabin(lc) = &mut veh.cabin {
                lc.state.temperature.mark_stale();
                lc.state
                    .temperature
                    .update(te_init, || format_dbg!())
                    .unwrap();

                lc.state.temp_prev.mark_stale();
                lc.state
                    .temp_prev
                    .update(te_init, || format_dbg!())
                    .unwrap();
            } else {
                panic!("cabin should have been configured");
            }
            let mut cyc = _cyc.clone();
            cyc.temp_amb_air = vec![*te_amb; cyc.len_checked().unwrap()];
            let mut sd = SimDrive::new(veh, cyc, Default::default());
            if let CabinOption::LumpedCabin(lc) = sd.veh.cabin.clone() {
                assert_eq!(
                    *lc.state.temperature.get_fresh(|| format_dbg!()).unwrap(),
                    te_init
                );
            } else {
                panic!();
            };
            sd.walk()
                .with_context(|| {
                    format!(
                        "ambient temperature: {}*C\ninit temperature: {}",
                        te_amb.get::<si::degree_celsius>(),
                        te_init.get::<si::degree_celsius>()
                    )
                })
                .unwrap();
            assert!(
                *sd.veh.state.i.get_fresh(String::new).unwrap()
                    == sd.cyc.len_checked().unwrap() - 1
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
            sd.veh.reset_step(|| format_dbg!()).unwrap();
            sd.veh.state.time.mark_stale();
            sd.veh
                .state
                .time
                .update(si::Time::ZERO, || format_dbg!())
                .unwrap();
            assert!(*sd.veh.state.i.get_fresh(|| format_dbg!()).unwrap() == 0);
            sd.walk()
                .with_context(|| {
                    format!(
                        "ambient temperature: {}*C\ninit temperature: {}",
                        te_amb.get::<si::degree_celsius>(),
                        te_init.get::<si::degree_celsius>()
                    )
                })
                .unwrap();
            assert_eq!(*sd.veh.state.i.get_fresh(|| format_dbg!()).unwrap(), 1369);
        }
    }
}
