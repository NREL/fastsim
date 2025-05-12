pub mod roadload;

use std::collections::HashSet;

use roadload::StepInfo;

use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::drive_cycle::manipulation_utils::{
    trapz_step_start_distance, CoastTrajectory, CycleCache,
};
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
    // Coasting Parameters
    /// whether to allow coasting or not.
    #[serde(default = "SimParams::def_coast_allow")]
    pub coast_allow: bool,
    /// for testing: triggers coasting when vehicle passes the given speed
    #[serde(default = "SimParams::def_coast_start_speed")]
    pub coast_start_speed: si::Velocity,
    /// speed at which mechanical braking will initiate during coasting maneuvers
    #[serde(default = "SimParams::def_coast_brake_start_speed")]
    pub coast_brake_start_speed: si::Velocity,
    /// acceleration assumed during braking for coast maneuvers
    /// NOTE: should be negative
    #[serde(default = "SimParams::def_coast_brake_accel")]
    pub coast_brake_accel: si::Acceleration,
    /// if true, accuracy will be favored over performance for grade per step
    /// estimates Specifically, for performance, grade for a step will be
    /// assumed to be the grade looked up at step start distance. For accuracy,
    /// the actual elevations will be used. This distinciton only makes a
    /// difference for CAV maneuvers.
    #[serde(default = "SimParams::def_favor_grade_accuracy")]
    pub favor_grade_accuracy: bool,
    /// if true, coasting vehicle can eclipse the shadow trace (i.e., reference
    /// vehicle in front)
    #[serde(default = "SimParams::def_coast_allow_passing")]
    pub coast_allow_passing: bool,
    /// maximum allowable speed under coast
    #[serde(default = "SimParams::def_coast_max_speed")]
    pub coast_max_speed: si::Velocity,
    // IDM - Intelligent Driver Model, Adaptive Cruise Control version
    /// if true, initiates the IDM - Intelligent Driver Model, Adaptive Cruise
    /// Control version
    #[serde(default = "SimParams::def_idm_allow")]
    pub idm_allow: bool,
}

#[named_struct_pyo3_api]
impl SimParams {
    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Self::default()
    }

    #[pyo3(name = "enable_coasting_with_start_speed")]
    fn enable_coasting_with_start_speed_py(
        &mut self,
        coast_start_speed_m_per_s: f64,
    ) -> anyhow::Result<()> {
        self.coast_allow = true;
        self.coast_start_speed = coast_start_speed_m_per_s * uc::MPS;
        Ok(())
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
    fn def_coast_allow() -> bool {
        Self::default().coast_allow
    }
    fn def_coast_start_speed() -> si::Velocity {
        Self::default().coast_start_speed
    }
    fn def_coast_brake_start_speed() -> si::Velocity {
        Self::default().coast_brake_start_speed
    }
    fn def_coast_brake_accel() -> si::Acceleration {
        Self::default().coast_brake_accel
    }
    fn def_favor_grade_accuracy() -> bool {
        Self::default().favor_grade_accuracy
    }
    fn def_coast_allow_passing() -> bool {
        Self::default().coast_allow_passing
    }
    fn def_coast_max_speed() -> si::Velocity {
        Self::default().coast_max_speed
    }
    fn def_idm_allow() -> bool {
        Self::default().idm_allow
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
            coast_allow: false,
            coast_start_speed: 0.0 * uc::MPS,
            coast_brake_start_speed: 20.0 * uc::MPH,
            coast_brake_accel: -2.5 * uc::MPS2,
            favor_grade_accuracy: true,
            coast_allow_passing: false,
            coast_max_speed: 40.0 * uc::MPS,
            idm_allow: false,
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
    pub cyc0: Cycle,
    pub cyc0_cache: Option<CycleCache>,
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
        // reset the cycle in case it has been manipulated
        self.cyc = self.cyc0.clone();
        self.veh
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.cyc
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.cyc0
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
        let cyc0 = cyc.clone();
        Self {
            veh,
            cyc,
            cyc0,
            cyc0_cache: None,
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

    /// Generate a coast trajectory without actually modifying the cycle.
    /// This can be used to calculate the distance to stop via coast using
    /// actual time-stepping and changing grade.
    pub fn generate_coast_trajectory(&mut self, i: usize) -> anyhow::Result<CoastTrajectory> {
        let v0 = *self.veh.state.speed_ach.get_stale(|| format_dbg!())?;
        let v0 = v0.get::<si::meter_per_second>();
        let v_brake = self
            .sim_params
            .coast_brake_start_speed
            .get::<si::meter_per_second>();
        let a_brake = {
            let result = self
                .sim_params
                .coast_brake_accel
                .get::<si::meter_per_second_squared>();
            if result > 0.0 {
                -result
            } else {
                result
            }
        };
        if self.cyc0_cache.is_none() {
            self.cyc0_cache = Some(CycleCache::new(&self.cyc0));
        }
        let cyc0_cache = self.cyc0_cache.as_ref().unwrap();
        let ds = cyc0_cache.trapz_step_distances_m.clone();
        let d0 = trapz_step_start_distance(&self.cyc, i).get::<si::meter>();
        let mut distances_m = Vec::with_capacity(ds.len());
        let mut grade_by_distance = Vec::with_capacity(ds.len());
        for (idx, d) in ds.iter().enumerate() {
            if *d >= d0 {
                distances_m.push(*d - d0);
                grade_by_distance.push(self.cyc0.grade[idx].get::<si::ratio>());
            }
        }
        if distances_m.is_empty() {
            return Ok(CoastTrajectory {
                found_trajectory: false,
                distance_to_stop_via_coast_m: 0.0,
                start_idx: 0,
                speed_m_per_s: None,
                distance_to_brake_m: None,
            });
        }
        if v0 <= v_brake {
            return Ok(CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: -0.5 * v0 * v0 / a_brake,
                start_idx: i,
                speed_m_per_s: None,
                distance_to_brake_m: None,
            });
        }
        let dtb = -0.5 * v_brake * v_brake / a_brake;
        let mut d = 0.0;
        let d_max = distances_m.last().unwrap() - dtb;
        let mut unique_grades = HashSet::with_capacity(ds.len());
        let grade_mult = 10000.0;
        for g in grade_by_distance.iter() {
            let grade = (g * grade_mult).round() as i32;
            unique_grades.insert(grade);
        }
        let unique_grade = if unique_grades.len() == 1 {
            let ug = unique_grades.iter().nth(0).unwrap();
            let ug = (*ug as f64) / grade_mult;
            Some(ug)
        } else {
            None
        };
        let has_unique_grade = unique_grade.is_some();
        let max_iter = 180;
        let iters_per_step = if self.sim_params.favor_grade_accuracy {
            2
        } else {
            1
        };
        let mut new_speeds_m_per_s = Vec::with_capacity(max_iter as usize);
        let mut v = v0;
        let mut iter = 0;
        let mut idx = i;
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0 * uc::M, Some(cyc0_cache))
            .get::<si::meter>();
        while v > v_brake
            && v >= 0.0
            && d <= d_max
            && iter < max_iter
            && idx < self.cyc0.speed.len()
        {
            let dt_s = self.cyc0.dt_at_i(idx)?.get::<si::second>();
            let mut gr = match unique_grade {
                Some(g) => g,
                None => cyc0_cache.interp_grade(d + d0),
            };
            let mut k = self.calc_dvdd(v, gr)?;
            let mut v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s);
            let mut vavg = 0.5 * (v + v_next);
            let mut dd: f64;
            for _ in 0..iters_per_step {
                k = self.calc_dvdd(vavg, gr)?;
                v_next = v * (1.0 + 0.5 * k * dt_s) / (1.0 - 0.5 * k * dt_s);
                vavg = 0.5 * (v + v_next);
                dd = vavg * dt_s;
                if self.sim_params.favor_grade_accuracy {
                    gr = match unique_grade {
                        Some(g) => g,
                        None => {
                            let dist = (d + d0) * uc::M;
                            let delta_dist = dd * uc::M;
                            self.cyc0
                                .average_grade_over_range(dist, delta_dist, Some(cyc0_cache))
                                .get::<si::ratio>()
                        }
                    };
                }
            }
            if k >= 0.0 && has_unique_grade {
                // there is no solution for coast-down -- speed will never decrease
                return Ok(CoastTrajectory {
                    found_trajectory: false,
                    distance_to_stop_via_coast_m: 0.0,
                    start_idx: 0,
                    speed_m_per_s: None,
                    distance_to_brake_m: None,
                });
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
        if iter < max_iter && idx < self.cyc0.speed.len() {
            let dtb = -0.5 * v * v / a_brake;
            let dtb_target = (dts0 - d).max(0.5 * dtb).min(2.0 * dtb);
            let dtsc = d + dtb_target;
            return Ok(CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: dtsc,
                start_idx: i,
                speed_m_per_s: Some(new_speeds_m_per_s),
                distance_to_brake_m: Some(dtb_target),
            });
        }
        Ok(CoastTrajectory {
            found_trajectory: false,
            distance_to_stop_via_coast_m: 0.0,
            start_idx: 0,
            speed_m_per_s: None,
            distance_to_brake_m: None,
        })
    }

    /// Determine whether the vehicle should go into a 'coasting' state.
    /// Normal coasting logic is that the vehicle will coast if it is
    /// within coasting distance of a stop:
    /// - if distance to coast from start of step <= distance to next stop
    /// - AND distance to coast from end of step (using reference speed) is
    ///   > distance to next step
    /// - AND vehicle was at or above the speed to start braking
    /// - AND at least four time-steps away from where braking would start
    ///
    /// NOTE: for the case when coast-start speed is used, we only worry
    /// about if the vehicle is above the coast-start speed. This is mainly
    /// for testing.
    pub fn should_impose_coast(&self) -> anyhow::Result<bool> {
        if self.sim_params.coast_start_speed > 0.0 * uc::MPS {
            let spd_ach = *self.veh.state.speed_ach.get_stale(|| format_dbg!())?;
            Ok(spd_ach >= self.sim_params.coast_start_speed)
        } else {
            Ok(false)
        }
    }

    fn apply_coast_trajectory(&mut self, coast_traj: &CoastTrajectory) -> anyhow::Result<()> {
        if coast_traj.found_trajectory {
            let num_speeds = match &coast_traj.speed_m_per_s {
                Some(vs) => {
                    for (di, &new_speed) in vs.iter().enumerate() {
                        let idx = coast_traj.start_idx + di;
                        if idx >= self.cyc0.speed.len() {
                            break;
                        }
                        self.cyc.speed[idx] = new_speed * uc::MPS;
                    }
                    vs.len()
                }
                None => 0,
            };
            let (_, _n) = self.cyc.modify_with_braking_trajectory(
                self.sim_params.coast_brake_accel,
                coast_traj.start_idx + num_speeds,
                coast_traj.distance_to_brake_m.map(|d| d * uc::M),
            )?;
            // TODO[mok]: address impose_coast. Do we need it?
            // for di in 0..(self.cyc0.speed.len() - coast_traj.start_idx) {
            //     let idx = coast_traj.start_idx + di;
            //     self.impose_coast[i] = di < num_speeds + n;
            // }
        }
        Ok(())
    }

    /// Determine speed for CAV maneuver: eco-coast or eco-cruise.
    pub fn set_speed(&mut self, i: usize) -> anyhow::Result<()> {
        let tol = 1e-6;
        let v0 = *self.veh.state.speed_ach.get_stale(|| format_dbg!())?;
        let v0 = v0.get::<si::meter_per_second>();
        let mut impose_coast = false;
        if v0 > tol && self.should_impose_coast()? {
            let ct = self.generate_coast_trajectory(i)?;
            if ct.found_trajectory {
                impose_coast = true;
                let d = ct.distance_to_stop_via_coast_m;
                // TODO[mok]: determine how to handle impose_coast
                // ... state/history method?
                // if d < 0.0 {
                //    for idx in i..self.cyc0.speed.len() {
                //       self.impose_coast[idx] = false;
                //    }
                // }
                if d >= 0.0 {
                    self.apply_coast_trajectory(&ct)?;
                }
                // TODO[mok]: add no passing
                // if !self.sim_params.coast_allow_passing {
                //    self.prevent_collisions(i, None)?;
                // }
            }
        }
        // TODO[mok]: add logic for IDM here
        if !impose_coast {
            if !self.sim_params.idm_allow {
                let i_i32 = i32::try_from(i).ok();
                // TODO[mok]: coast_delay_index?
                let target_idx = match i_i32 {
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
                    self.cyc.speed[i] = self.cyc0.speed[ti.min(self.cyc0.speed.len() - 1)];
                }
            }
            return Ok(());
        }
        let v1_traj = self.cyc.speed[i].get::<si::meter_per_second>();
        let v_brake = self
            .sim_params
            .coast_brake_start_speed
            .get::<si::meter_per_second>();
        if v0 > v_brake {
            if self.sim_params.coast_allow_passing {
                // NOTE: We could be coasting downhill so could in theory go
                // to a higher speed. Since we can pass, allow vehicle to go
                // up to max coasting speed (m/s). The solver will show us what
                // we can actually achieve.
                self.cyc.speed[i] = self.sim_params.coast_max_speed;
            } else {
                self.cyc.speed[i] = v1_traj.min(
                    self.sim_params
                        .coast_max_speed
                        .get::<si::meter_per_second>(),
                ) * uc::MPS;
            }
        }
        // Solve for the actual coasting speed
        // ...
        // TODO[mok]: finish this method...
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
        let coasting = if self.sim_params.coast_allow
            && self.sim_params.coast_start_speed > si::Velocity::ZERO
        {
            let v0 = *self.veh.state.speed_ach.get_stale(|| format_dbg!())?;
            let was_coasting = *self.veh.state.coasting.get_stale(|| format_dbg!())?;
            v0 > si::Velocity::ZERO && (was_coasting || v0 > self.sim_params.coast_start_speed)
        } else {
            false
        };
        if coasting {
            let vs = &mut self.veh.state;
            let grade_curr = {
                let interp_pt_dist: &[f64] = match self.cyc0.grade_interp {
                    Some(Interpolator::Interp0D(..)) => &[],
                    Some(Interpolator::Interp1D(..)) => {
                        &[vs.dist.get_fresh(|| format_dbg!())?.get::<si::meter>()]
                    }
                    _ => unreachable!(),
                };
                uc::R
                    * self
                        .cyc0
                        .grade_interp
                        .as_ref()
                        .with_context(|| format_dbg!("You might have somehow bypassed `init()`"))?
                        .interpolate(interp_pt_dist)
                        .with_context(|| format_dbg!())?
            };
            let speed_prev = *vs.speed_ach.get_stale(|| format_dbg!())?;
            let cyc_speed = self.cyc.speed[i];
            let step_info = StepInfo {
                dt,
                speed_prev,
                cyc_speed,
                grade_curr,
                air_density: *vs.air_density.get_stale(|| format_dbg!())?,
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
                pwr_prop_fwd_max: si::Power::ZERO,
            };
            let target_speed = step_info.solve_for_speed(
                self.sim_params.ach_speed_max_iter,
                self.sim_params.ach_speed_tol,
                self.sim_params.ach_speed_solver_gain,
            );
            self.cyc.speed[i] = target_speed;
            // NOTE: force recalculation of dependent fields like distance
            self.cyc.init().unwrap();
        }

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
        self.veh.state.coasting.update(coasting, || format_dbg!())?;
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
        } else if !self.sim_params.coast_allow && !self.sim_params.idm_allow {
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

        /*
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
                .get_stale(|| format_dbg!())?
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
        */
        let speed_ach = step_info.solve_for_speed(
            self.sim_params.ach_speed_max_iter,
            self.sim_params.ach_speed_tol,
            self.sim_params.ach_speed_solver_gain,
        );

        vs.speed_ach.update(speed_ach, || format_dbg!())?;
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
        self.set_pwr_prop_for_speed(
            *self.veh.state.speed_ach.get_stale(|| format_dbg!())?,
            speed_prev,
            dt,
        )
        .with_context(|| format_dbg!())?;
        self.set_ach_speed(speed_ach, dt)
            .with_context(|| anyhow!(format_dbg!()))?;

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
    fn test_sim_drive_hev_thrml() {
        let _veh = Vehicle::from_resource("2021_Hyundai_Sonata_Hybrid_Blue.yaml", false).unwrap();
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
    fn test_sim_drive_bev() {
        let _veh = mock_bev();
        let _cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let _cyc0 = _cyc.clone();
        let mut sd = SimDrive {
            veh: _veh,
            cyc: _cyc,
            cyc0: _cyc0,
            cyc0_cache: None,
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
        let _veh = Vehicle::from_resource("2020 Chevrolet Bolt EV.yaml", false).unwrap();
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
            }
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

    #[test]
    #[cfg(feature = "resources")]
    fn test_coasting() {
        //let mut veh = Vehicle::from_resource("2020 Chevrolet Bolt EV.yaml", false).unwrap();
        let mut veh = Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
        veh.set_save_interval(Some(1))
            .expect("No error expected setting save interval");
        let cyc = Cycle::from_resource("udds.csv", false).unwrap();
        let coast_start_speed = 20.0 * uc::MPS;
        let coast_start_speed_m_per_s = coast_start_speed.get::<si::meter_per_second>();
        let params = SimParams {
            coast_allow: true,
            coast_start_speed,
            ..Default::default()
        };
        let mut sd = SimDrive::new(veh.clone(), cyc.clone(), Some(params));
        sd.walk().unwrap();
        let speed_req_mps: Vec<f64> = cyc
            .speed
            .clone()
            .iter()
            .map(|v| v.get::<si::meter_per_second>())
            .collect();
        let speed_ach_mps: Vec<f64> = sd
            .veh
            .history
            .speed_ach
            .clone()
            .iter()
            .map(|v| {
                v.get_fresh(|| format_dbg!())
                    .unwrap()
                    .get::<si::meter_per_second>()
            })
            .collect();
        assert_eq!(speed_req_mps.len(), speed_ach_mps.len());
        let mut should_not_equal = false;
        for idx in 0..speed_req_mps.len() {
            if should_not_equal || speed_req_mps[idx] > coast_start_speed_m_per_s {
                should_not_equal = true;
            } else {
                assert_eq!(
                    speed_req_mps[idx], speed_ach_mps[idx],
                    "Speeds do not match at {idx}"
                );
            }
        }
    }
}
