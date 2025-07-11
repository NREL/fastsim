use std::collections::HashSet;

use crate::{imports::*, simdrive::roadload::StepInfo, simdrive::SimParams, vehicle::Vehicle};

use super::manipulation_utils::trapz_distance_for_step;
use super::{
    manipulation_utils::{
        accel_array_for_constant_jerk, accel_for_constant_jerk, calc_constant_jerk_trajectory,
        trapz_step_distances, trapz_step_start_distance, CoastTrajectory, CycleCache, PassingInfo,
        RendezvousTrajectory,
    },
    Cycle,
};

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct Maneuver {
    // Cycle Instances
    /// Cycle to apply maneuver to
    #[serde(default)]
    pub cyc: Cycle,
    /// Reference cycle
    #[serde(default)]
    pub cyc0: Cycle,

    // Chassis Data
    /// Constant mass to assume for maneuvers
    #[serde(default)]
    pub mass: si::Mass,
    /// Constant air density to assume for manuvers
    #[serde(default)]
    pub air_density: si::MassDensity,
    /// Constant aerodynamic drag coefficient to assume
    #[serde(default)]
    pub drag_coef: si::Ratio,
    /// Constant Frontal Area of vehicle to assume
    #[serde(default)]
    pub frontal_area: si::Area,
    /// Constant Wheel Rolling Resistance Coefficient to assume
    #[serde(default)]
    pub wheel_rr_coef: si::Ratio,
    /// Wheel inertia per wheel
    #[serde(default)]
    pub wheel_inertia: si::MomentOfInertia,
    /// Number of wheels
    #[serde(default)]
    pub num_wheels: u8,
    /// Wheel radius
    #[serde(default)]
    pub wheel_radius: si::Length,

    // Solver Settings
    /// max number of iterations allowed in setting achieved speed when trace
    /// cannot be achieved
    #[serde(default)]
    pub ach_speed_max_iter: u32,
    /// tolerance in change in speed guess in setting achieved speed when trace
    /// cannot be achieved
    #[serde(default)]
    pub ach_speed_tol: si::Ratio,
    /// Newton method gain for setting achieved speed
    #[serde(default)]
    pub ach_speed_solver_gain: f64,

    // Coasting Parameters
    /// whether to allow coasting or not.
    #[serde(default)]
    pub coast_allow: bool,
    /// for testing: triggers coasting when vehicle passes the given speed
    #[serde(default)]
    pub coast_start_speed: si::Velocity,
    /// speed at which mechanical braking will initiate during coasting maneuvers
    #[serde(default)]
    pub coast_brake_start_speed: si::Velocity,
    /// acceleration assumed during braking for coast maneuvers
    /// NOTE: should be negative
    #[serde(default)]
    pub coast_brake_accel: si::Acceleration,
    /// if true, accuracy will be favored over performance for grade per step
    /// estimates Specifically, for performance, grade for a step will be
    /// assumed to be the grade looked up at step start distance. For accuracy,
    /// the actual elevations will be used. This distinciton only makes a
    /// difference for CAV maneuvers.
    #[serde(default)]
    pub favor_grade_accuracy: bool,
    /// if true, coasting vehicle can eclipse the shadow trace (i.e., reference
    /// vehicle in front)
    #[serde(default)]
    pub coast_allow_passing: bool,
    /// maximum allowable speed under coast
    #[serde(default)]
    pub coast_max_speed: si::Velocity,
    /// "look-ahead" time for speed changes to be considered to feature coasting
    /// to hit a given stopping distance mark
    #[serde(default)]
    pub coast_time_horizon_for_adjustment: si::Time,

    // IDM - Intelligent Driver Model, Adaptive Cruise Control version
    /// if true, initiates the IDM - Intelligent Driver Model, Adaptive Cruise
    /// Control version
    #[serde(default)]
    pub idm_allow: bool,
    /// IDM algorithm: a way to specify desired speed by course distance
    /// traveled. Can simulate changing speed limits over a driving cycle.
    /// optional list of (distance (m), desired speed (m/s)).
    #[serde(default)]
    pub idm_desired_speed_by_distance: Option<Vec<(si::Length, si::Velocity)>>,
    /// IDM algorithm: desired speed (m/s). Only used if
    /// idm_v_desired_in_m_per_s_by_distance_m is NOT set (i.e., is None)
    #[serde(default)]
    pub idm_desired_speed: si::Velocity,
    /// IDM algorithm: headway time desired to vehicle in front (s)
    #[serde(default)]
    pub idm_headway: si::Time,
    /// IDM algorithm: minimum desired gap between vehicle and lead vehicle (m)
    #[serde(default)]
    pub idm_minimum_gap: si::Length,
    /// IDM algorithm: delta parameter
    #[serde(default)]
    pub idm_delta: f64,
    /// IDM algorithm: acceleration parameter
    #[serde(default)]
    pub idm_acceleration: si::Acceleration,
    /// IDM algorithm: deceleration parameter
    #[serde(default)]
    pub idm_deceleration: si::Acceleration,

    // Internal Fields
    pub i: usize,
    pub coast_delay_index: Vec<i32>,
    #[serde(default)]
    pub impose_coast: Vec<bool>,
    #[serde(default)]
    pub idm_target_speed_m_per_s: Vec<f64>,

    pub cyc0_cache: CycleCache,
}

#[pyo3_api]
impl Maneuver {
    #[pyo3(name = "create_from")]
    #[staticmethod]
    /// create a maneuver object based on the given cycle and vehicle.
    fn create_from_py(cyc: &Cycle, veh: &Vehicle) -> PyResult<Self> {
        Ok(Maneuver::from(cyc, veh))
    }

    #[pyo3(name = "apply_maneuvers")]
    /// apply all maneuvers to the cycle and return it.
    fn apply_maneuvers_py(&mut self) -> PyResult<Cycle> {
        self.apply();
        let cyc = self.cyc.clone();
        Ok(cyc)
    }

    #[pyo3(name = "is_coasting")]
    /// return a vector of signals indicating 1.0 for coast, otherwise 0.0
    fn is_coasting_py(&self) -> PyResult<Vec<f64>> {
        let mut result = Vec::with_capacity(self.impose_coast.len());
        for ic in &self.impose_coast {
            if *ic {
                result.push(1.0);
            } else {
                result.push(0.0);
            }
        }
        Ok(result)
    }
}

impl SerdeAPI for Maneuver {
    #[cfg(feature = "resources")]
    const RESOURCES_SUBDIR: &'static str = "maneuvers";
}

impl Init for Maneuver {
    fn init(&mut self) -> Result<(), Error> {
        self.i = 1;
        let n = self.cyc.speed.len();
        self.coast_delay_index = vec![0; n];
        self.impose_coast = vec![false; n];
        self.idm_target_speed_m_per_s = vec![0.0; n];
        self.cyc0_cache = self.cyc0.build_cache();
        Ok(())
    }
}

impl Default for Maneuver {
    fn default() -> Self {
        Self {
            cyc: Cycle::default(),
            cyc0: Cycle::default(),
            mass: si::Mass::default(),
            air_density: si::MassDensity::default(),
            drag_coef: si::Ratio::default(),
            frontal_area: si::Area::default(),
            wheel_rr_coef: si::Ratio::default(),
            wheel_inertia: si::MomentOfInertia::default(),
            num_wheels: u8::default(),
            wheel_radius: si::Length::default(),
            ach_speed_max_iter: 3,
            ach_speed_tol: 1.0e-3 * uc::R,
            ach_speed_solver_gain: 0.9,
            coast_allow: false,
            coast_start_speed: 0.0 * uc::MPS,
            coast_brake_start_speed: 20.0 * uc::MPH,
            coast_brake_accel: -2.5 * uc::MPS2,
            favor_grade_accuracy: true,
            coast_allow_passing: false,
            coast_max_speed: 40.0 * uc::MPS,
            coast_time_horizon_for_adjustment: 20.0 * uc::S,
            idm_allow: false,
            idm_desired_speed_by_distance: None,
            idm_desired_speed: 75.0 * uc::MPH,
            idm_headway: 1.0 * uc::S,
            idm_minimum_gap: 2.0 * uc::M,
            idm_delta: 4.0,
            idm_acceleration: 1.0 * uc::MPS2,
            idm_deceleration: 1.5 * uc::MPS2,
            i: 1,
            coast_delay_index: vec![0, 0],
            impose_coast: vec![false, false],
            idm_target_speed_m_per_s: vec![0.0, 0.0],
            cyc0_cache: CycleCache::default(),
        }
    }
}

impl Maneuver {
    /// Create maneuver object from cycle and vehicle.
    pub fn from(cyc: &Cycle, veh: &Vehicle) -> Self {
        let mut c = cyc.clone();
        c.init().unwrap();
        let mut v = veh.clone();
        v.init().unwrap();
        let cyc0 = c.clone();
        let cyc0_cache = cyc0.build_cache();
        let default_mass = 1200.0 * uc::KG;
        let mass = if let Ok(Some(m)) = veh.mass() {
            m
        } else {
            default_mass
        };
        // TODO[mok]: what is the proper way to get air_density from the
        //            vehicle? The below returns 0.0...
        // *veh.state.air_density.get_fresh(|| format_dbg!()).unwrap();
        let air_density = 1.2 * uc::KGPM3;

        // NOTE: wheel radius should exist as Some(value) after v.init() above.
        let wheel_radius = veh.chassis.wheel_radius.unwrap();
        let params = SimParams::default();
        Self {
            cyc: c,
            cyc0,
            cyc0_cache,
            mass,
            air_density,
            drag_coef: veh.chassis.drag_coef,
            frontal_area: veh.chassis.frontal_area,
            wheel_rr_coef: veh.chassis.wheel_rr_coef,
            wheel_inertia: veh.chassis.wheel_inertia,
            num_wheels: veh.chassis.num_wheels,
            wheel_radius,
            ach_speed_max_iter: params.ach_speed_max_iter,
            ach_speed_tol: params.ach_speed_tol,
            ach_speed_solver_gain: params.ach_speed_solver_gain,
            ..Default::default()
        }
    }

    /// Apply the eco-coast maneuver to the given cycle with
    /// the given reference cycle.
    /// - cyc: cycle to modify
    /// - cyc0: reference cycle
    pub fn apply(&mut self) {
        self.cyc.init().unwrap();
        self.cyc0.init().unwrap();
        self.i = 1;
        let cyc_len = self.cyc.time.len();
        self.coast_delay_index = vec![0; cyc_len];
        self.impose_coast = vec![false; cyc_len];
        self.idm_target_speed_m_per_s = vec![0.0; cyc_len];
        self.cyc0_cache = self.cyc0.build_cache();
        self.walk(cyc_len);
    }

    /// Walk from step to step for the maneuver simulation.
    fn walk(&mut self, cyc_len: usize) {
        while self.i < cyc_len {
            self.step();
        }
        // NOTE: force dist and elev to recalculate
        // TODO: need to investigate re-deriving grade from interpolation of
        // elevation by distance.
        self.cyc.dist = vec![];
        self.cyc.elev = vec![];
        self.cyc.init().unwrap();
    }

    /// Step: compute a single time-step for the maneuver simulation.
    fn step(&mut self) {
        if self.idm_allow {
            self.idm_target_speed_m_per_s[self.i] = match &self.idm_desired_speed_by_distance {
                Some(vtgt_by_dist) => {
                    let mut found_v_target = vtgt_by_dist[0].1;
                    let mut current_d = si::Length::ZERO;
                    for (idx, d) in self.cyc.dist.iter().enumerate() {
                        if idx > self.i {
                            break;
                        }
                        current_d += *d;
                    }
                    for (d, v_target) in vtgt_by_dist {
                        if current_d >= *d {
                            found_v_target = *v_target;
                        } else {
                            break;
                        }
                    }
                    found_v_target.get::<si::meter_per_second>()
                }
                None => self.idm_desired_speed.get::<si::meter_per_second>(),
            };
            self.set_speed_for_target_gap_using_idm(self.i);
        }
        if self.coast_allow {
            self.set_coast_speed(self.i);
            self.cyc.grade[self.i] = self.lookup_grade_for_step(self.i, None);
        }
        self.i += 1;
    }

    /// Set gap
    /// - i: non-negative integer, the step index
    ///
    /// RETURN: None
    ///
    /// EFFECTS:
    /// - sets the next speed (m/s)
    ///
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
            let mut v = self.cyc0.speed[0];
            for vi in &self.cyc0.speed {
                if *vi > v {
                    v = *vi;
                }
            }
            v.get::<si::meter_per_second>()
        };
        // DERIVED VALUES
        self.cyc.speed[i] = self.next_speed_by_idm(
            i,
            self.idm_acceleration.get::<si::meter_per_second_squared>(),
            self.idm_deceleration.get::<si::meter_per_second_squared>(),
            self.idm_headway.get::<si::second>(),
            self.idm_minimum_gap.get::<si::meter>(),
            v_desired_m_per_s,
            self.idm_delta,
        ) * uc::MPS;
    }

    /// Calculate the next speed by the Intelligent Driver Model
    /// - i: int, the index
    /// - a_m_per_s2: number, max acceleration (m/s2)
    /// - b_m_per_s2: number, max deceleration (m/s2)
    /// - dt_headway_s: number, the headway between us and the lead vehicle in seconds
    /// - s0_m: number, the initial gap between us and the lead vehicle in meters
    /// - v_desired_m_per_s: number, the desired speed in (m/s)
    /// - delta: number, a shape parameter; typical value is 4.0
    ///
    /// RETURN: number, the next speed (m/s)
    ///
    /// REFERENCE:
    /// Treiber, Martin and Kesting, Arne. 2013. "Chapter 11: Car-Following Models Based on Driving Strategies".
    ///     Traffic Flow Dynamics: Data, Models and Simulation. Springer-Verlag. Springer, Berlin, Heidelberg.
    ///     DOI: <https://doi.org/10.1007/978-3-642-32460-4>.
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
        let dt_headway_s = dt_headway_s.max(0.0);
        // we assume the vehicles start out a "minimum gap" apart
        let s0_m = s0_m.max(0.0);
        // DERIVED VALUES
        let sqrt_ab = (a_m_per_s2 * b_m_per_s2).powf(0.5);
        let v0_m_per_s = self.cyc.speed[i - 1].get::<si::meter_per_second>();
        let v0_lead_m_per_s = self.cyc0.speed[i - 1].get::<si::meter_per_second>();
        let dv0_m_per_s = v0_m_per_s - v0_lead_m_per_s;
        let d0_lead_m = self.cyc0_cache.trapz_distances_m[(i - 1).max(0)] + s0_m;
        let d0_m = trapz_step_start_distance(&self.cyc, i).get::<si::meter>();
        let s_m = (d0_lead_m - d0_m).max(0.01);
        let dt = (self.cyc0.time[i] - self.cyc0.time[i - 1]).get::<si::second>();
        // IDM EQUATIONS
        let s_target_m = s0_m
            + ((v0_m_per_s * dt_headway_s) + ((v0_m_per_s * dv0_m_per_s) / (2.0 * sqrt_ab)))
                .max(0.0);
        let accel_target_m_per_s2 = a_m_per_s2
            * (1.0 - ((v0_m_per_s / v_desired_m_per_s).powf(delta)) - ((s_target_m / s_m).powi(2)));
        (v0_m_per_s + (accel_target_m_per_s2 * dt)).max(0.0)
    }

    /// For situations where cyc can deviate from cyc0, this method
    /// looks up and accurately interpolates what the average grade over
    /// the step should be. The achieved value is used to predict the
    /// distance traveled over the step.
    ///
    /// NOTE:
    /// If not allowing coasting (i.e., sim_params.coast_allow == False)
    /// and not allowing IDM/following (i.e., self.sim_params.idm_allow
    /// == False) then returns self.cyc.grade\[i\]
    pub fn lookup_grade_for_step(&self, i: usize, speed_ach: Option<si::Velocity>) -> si::Ratio {
        if self.cyc0_cache.grade_all_zero {
            return 0.0 * uc::R;
        }
        if !self.coast_allow && !self.idm_allow {
            return self.cyc.grade[i];
        }
        match speed_ach {
            Some(v1) => {
                let dt = self.cyc.time[i] - self.cyc.time[i - 1];
                self.cyc0.average_grade_over_range(
                    trapz_step_start_distance(&self.cyc, i),
                    0.5 * (v1 + self.cyc.speed[i - 1]) * dt,
                    Some(&self.cyc0_cache),
                )
            }
            None => self.cyc0.average_grade_over_range(
                trapz_step_start_distance(&self.cyc, i),
                trapz_distance_for_step(&self.cyc, i),
                Some(&self.cyc0_cache),
            ),
        }
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
    pub fn should_impose_coast(&mut self, i: usize) -> bool {
        let v0 = self.cyc.speed[i - 1];
        if self.coast_start_speed > si::Velocity::ZERO {
            return v0 >= self.coast_start_speed;
        }
        if v0 < self.coast_brake_start_speed {
            return false;
        }
        // distance to stop by coasting from start of step (i - 1)
        let dtsc0 = self.calc_distance_to_stop_coast_v2(i);
        if dtsc0.is_none() {
            return false;
        }
        let dtsc0 = dtsc0.unwrap();
        // distance to next stop (m)
        let d0 = trapz_step_start_distance(&self.cyc, i);
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0, Some(&self.cyc0_cache));
        let dtb = -0.5 * v0 * v0 / self.coast_brake_accel;
        dtsc0 >= dts0 && dts0 >= (4.0 * dtb)
    }

    /// Calculate the distance to stop via coasting.
    /// - i: the current index
    ///
    /// RETURN: if Some, the distance, else None
    ///
    /// NOTES:
    /// - if None, that means there is no solution to a coast-down distance.
    ///   This can happen due to being too close to the given stop or
    ///   perhaps due to coasting downhill (i.e., will not stop).
    /// - if Some, the distance in meters that the vehicle would freely coast
    ///   is unobstructed. We do account fro grade between the current point
    ///   and the end-point.
    pub fn calc_distance_to_stop_coast_v2(&mut self, i: usize) -> Option<si::Length> {
        let not_found = -1.0;
        let v0 = self.cyc.speed[i - 1].get::<si::meter_per_second>();
        let v_brake = self.coast_brake_start_speed.get::<si::meter_per_second>();
        let a_brake = self.coast_brake_accel.get::<si::meter_per_second_squared>();
        let ds = &self.cyc0_cache.trapz_distances_m;
        let gs: Vec<f64> = self
            .cyc0
            .grade
            .iter()
            .map(|g| g.get::<si::ratio>())
            .collect();
        assert!(
            ds.len() == gs.len(),
            "Assumed lengths of distances and grades must equal. ds.len()={}, gs.len()={}",
            ds.len(),
            gs.len(),
        );
        let d0 = trapz_step_start_distance(&self.cyc, i).get::<si::meter>();
        let mut grade_by_distance = Vec::with_capacity(ds.len());
        for idx in 0..ds.len() {
            if ds[idx] >= d0 {
                grade_by_distance.push(gs[idx]);
            }
        }
        let veh_mass_kg = self.mass.get::<si::kilogram>();
        let air_density_kg_per_m3 = self.air_density.get::<si::kilogram_per_cubic_meter>();
        let cdfa_m2 =
            self.drag_coef.get::<si::ratio>() * self.frontal_area.get::<si::square_meter>();
        let rrc = self.wheel_rr_coef.get::<si::ratio>();
        let gravity_m_per_s2 = uc::ACC_GRAV.get::<si::meter_per_second_squared>();
        // distance traveled while stopping via friction-braking (i.e., distance to brake)
        let dtb = -0.5 * v_brake * v_brake / a_brake;
        if v0 <= v_brake {
            let result = -0.5 * v0 * v0 / a_brake;
            return Some(result * uc::M);
        }
        let grade_mult = 10000;
        let unique_grades = {
            let mut result = HashSet::new();
            for gr in &grade_by_distance {
                let gr_to_store = (*gr * (grade_mult as f64)).round() as i32;
                result.insert(gr_to_store);
            }
            result
        };
        if unique_grades.len() == 1 {
            // if there is only one grade, there may be a closed-form solution
            let unique_grade = (*unique_grades.iter().nth(0).unwrap() as f64) / (grade_mult as f64);
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
                let result = d + dtb;
                return Some(result * uc::M);
            }
        }
        let ct = self.generate_coast_trajectory(i);
        if ct.found_trajectory {
            Some(ct.distance_to_stop_via_coast_m * uc::M)
        } else {
            None
        }
    }

    /// Set the coasting speed.
    pub fn set_coast_speed(&mut self, i: usize) {
        let tol = 1e-6;
        let v0 = self.cyc.speed[i - 1].get::<si::meter_per_second>();
        if v0 > tol && !self.impose_coast[i] && self.should_impose_coast(i) {
            let ct = self.generate_coast_trajectory(i);
            if ct.found_trajectory {
                let d = ct.distance_to_stop_via_coast_m;
                if d < 0.0 {
                    for idx in i..self.cyc0.speed.len() {
                        self.impose_coast[idx] = false;
                    }
                } else {
                    self.apply_coast_trajectory(&ct);
                }
                if !self.coast_allow_passing {
                    self.prevent_collisions(i, None);
                }
            }
        }
        if !self.impose_coast[i] {
            if !self.idm_allow {
                // NOTE: transforming to i32 so we can carry negative indices.
                let i_i32 = i32::try_from(i).ok();
                let target_idx = i_i32.map(|v| v - self.coast_delay_index[i]);
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
                    self.cyc.speed[i] = self.cyc0.speed[ti.min(self.cyc0.speed.len() - 1)];
                }
            }
            return;
        }
        let v1_traj = self.cyc.speed[i].get::<si::meter_per_second>();
        let v_brake = self.coast_brake_start_speed.get::<si::meter_per_second>();
        if v0 > v_brake {
            if self.coast_allow_passing {
                // NOTE: We could be coasting downhill so could in theory go
                // to a higher speed. Since we can pass, allow vehicle to go
                // up to max coasting speed (m/s). The solver will show us what
                // we can actually achieve.
                self.cyc.speed[i] = self.coast_max_speed;
            } else {
                self.cyc.speed[i] =
                    v1_traj.min(self.coast_max_speed.get::<si::meter_per_second>()) * uc::MPS;
            }
        }
        // Solve for the actual coasting speed
        let coast_speed = self.solve_step(i);
        if self.impose_coast[i - 1] && v1_traj <= v_brake {
            // NOTE: if we've been coasting for at least one step already
            // and the current trajectory takes us below v_brake, we should
            // use that as it is the brake trajectory.
            self.cyc.speed[i] = v1_traj * uc::MPS;
        } else {
            self.cyc.speed[i] = coast_speed;
        }
        let v_tol = tol * uc::MPS;
        let dt = self.cyc.time[i] - self.cyc.time[i - 1];
        let accel_proposed = (self.cyc.speed[i] - self.cyc.speed[i - 1]) / dt;
        if self.cyc.speed[i] < v_tol {
            for idx in i..self.cyc0.speed.len() {
                self.impose_coast[idx] = false;
            }
            self.set_coast_delay(i);
            self.cyc.speed[i] = si::Velocity::ZERO;
            return;
        }
        if (self.cyc.speed[i] - v1_traj * uc::MPS).abs() > v_tol {
            let mut adjusted_current_speed = false;
            let brake_speed_start_tol = 0.1 * uc::MPS;
            if self.cyc.speed[i] < (self.coast_brake_start_speed - brake_speed_start_tol) {
                let (_, num_steps) =
                    self.cyc
                        .modify_with_braking_trajectory(self.coast_brake_accel, i, None);
                for idx in i..self.cyc0.speed.len() {
                    self.impose_coast[idx] = idx < (i + num_steps);
                }
                adjusted_current_speed = true;
            } else {
                // NOTE: will the below work when coasting downhill and picking up speed?
                let (traj_found, traj_n, traj_jerk_m_per_s3, traj_accel_m_per_s2) = self
                    .calc_next_rendezvous_trajectory(
                        i,
                        self.coast_brake_accel,
                        accel_proposed.min(0.0 * uc::MPS2),
                    );
                if traj_found {
                    // adjust cyc to perform the trajectory
                    let final_speed = self.cyc.modify_by_const_jerk_trajectory(
                        i,
                        traj_n,
                        traj_jerk_m_per_s3 * uc::MPS3,
                        traj_accel_m_per_s2 * uc::MPS2,
                    );
                    for idx in i..self.cyc0.speed.len() {
                        self.impose_coast[idx] = idx < (i + traj_n);
                    }
                    adjusted_current_speed = true;
                    let i_for_brake = i + traj_n;
                    if (final_speed - self.coast_brake_start_speed).abs() < brake_speed_start_tol {
                        let (_, num_steps) = self.cyc.modify_with_braking_trajectory(
                            self.coast_brake_accel,
                            i_for_brake,
                            None,
                        );
                        for idx in i_for_brake..self.cyc0.speed.len() {
                            self.impose_coast[idx] = idx < i_for_brake + num_steps;
                        }
                        adjusted_current_speed = true;
                    } else {
                        println!("## WARNING ##");
                        println!("final_speed={:?} not close to coast_brake_start_speed={:?} for i={:?}; i_for_brake={:?}, traj_n={:?}",
                            final_speed, self.coast_brake_start_speed, i, i_for_brake, traj_n);
                    }
                }
            }
            if adjusted_current_speed {
                if !self.coast_allow_passing {
                    self.prevent_collisions(i, None);
                }
                self.cyc.speed[i] = self.solve_step(i);
            }
        }
    }

    /// Solve for coast speed and set.
    pub fn solve_step(&mut self, i: usize) -> si::Velocity {
        let dt = self.cyc.time[i] - self.cyc.time[i - 1];
        let step_info = StepInfo {
            dt,
            speed_prev: self.cyc.speed[i - 1],
            cyc_speed: self.cyc.speed[i],
            grade_curr: self.cyc.grade[i],
            air_density: self.air_density,
            mass: self.mass,
            drag_coef: self.drag_coef,
            frontal_area: self.frontal_area,
            wheel_inertia: self.wheel_inertia,
            num_wheels: self.num_wheels,
            wheel_radius: self.wheel_radius,
            wheel_rr_coef: self.wheel_rr_coef,
            pwr_prop_fwd_max: 0.0 * uc::KW,
        };
        let coast_speed = step_info.solve_for_speed(
            self.ach_speed_max_iter,
            self.ach_speed_tol,
            self.ach_speed_solver_gain,
        );
        let max_coast_speed = self.coast_max_speed.min(coast_speed);
        let brake_start_speed = self.coast_brake_start_speed + 0.1 * uc::MPS;
        if coast_speed > brake_start_speed {
            max_coast_speed
        } else {
            // NOTE: We follow trace below the brake start speed as the
            // brake trajectory has been added to the cycle
            self.cyc.speed[i]
                .min(max_coast_speed)
                .max(si::Velocity::ZERO)
        }
    }

    /// Calculate next rendezvous trajectory for eco-coasting
    /// - i: the index into cyc for the end of start-of-step (i.e., the step
    ///   that may be modified; should be i)
    /// - min_accel: the minimum acceleration permitted
    /// - max_accel: the maximum acceleration permitted
    ///
    /// RETURN: (Tuple
    ///     found_rendezvous: Bool, if True the remainder of the data is valid; if False, no rendezvous found
    ///     n: positive integer, the number of steps ahead to rendezvous at
    ///     jerk_m__s3: number, the Jerk or first-derivative of acceleration (m/s3)
    ///     accel_m__s2: number, the initial acceleration of the trajectory (m/s2)
    /// )
    /// If no rendezvous exists within the scope, the returned tuple has False for the first item.
    /// Otherwise, returns the next closest rendezvous in time/space
    pub fn calc_next_rendezvous_trajectory(
        &self,
        i: usize,
        min_accel: si::Acceleration,
        max_accel: si::Acceleration,
    ) -> (bool, usize, f64, f64) {
        let tol = 1e-6;
        let min_accel_m_per_s2 = min_accel.get::<si::meter_per_second_squared>();
        let max_accel_m_per_s2 = max_accel.get::<si::meter_per_second_squared>();
        // v0 is where n=0; i.e., i - 1
        let v0 = self.cyc.speed[i - 1].get::<si::meter_per_second>();
        let brake_start_speed_m_per_s = self.coast_brake_start_speed.get::<si::meter_per_second>();
        let brake_accel_m_per_s2 = self.coast_brake_accel.get::<si::meter_per_second_squared>();
        let time_horizon_s = self.coast_time_horizon_for_adjustment.get::<si::second>();
        // distance_horizon_m = 1_000.0;
        let not_found_n = 0;
        let not_found_jerk_m_per_s3 = 0.0;
        let not_found_accel_m_per_s2 = 0.0;
        let not_found = (
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
        let num_samples = self.cyc.speed.len();
        let d0 = trapz_step_start_distance(&self.cyc, i).get::<si::meter>();
        // a_proposed = (v1 - v0) / dt
        // distance to stop from start of time-step
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0 * uc::M, Some(&self.cyc0_cache))
            .get::<si::meter>();
        if dts0 < 0.0 {
            return not_found;
        }
        let dt = (self.cyc0.time[i] - self.cyc0.time[i - 1]).get::<si::second>();
        // distance to brake from the brake start speed (m/s)
        let dtb =
            -0.5 * brake_start_speed_m_per_s * brake_start_speed_m_per_s / brake_accel_m_per_s2;
        // distance to brake initialization from start of time-step (m)
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
            dt_plan += dt;
            let step_ahead = step_idx - (i - 1);
            if step_ahead == 1 {
                // for brake init rendezvous
                let accel = (brake_start_speed_m_per_s - v0) / dt;
                let v1 = (v0 + accel * dt).max(0.0);
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
                    let r_bi_traj = calc_constant_jerk_trajectory(
                        step_ahead,
                        0.0,
                        v0,
                        dtbi0,
                        brake_start_speed_m_per_s,
                        dt,
                    );
                    let r_bi_jerk_m_per_s3 = r_bi_traj.jerk_m_per_s3;
                    let r_bi_accel_m_per_s2 = r_bi_traj.acceleration_m_per_s2;
                    if r_bi_accel_m_per_s2 < max_accel_m_per_s2
                        && min_accel_m_per_s2 < r_bi_accel_m_per_s2
                        && r_bi_jerk_m_per_s3 >= 0.0
                    {
                        let as_bi = accel_array_for_constant_jerk(
                            step_ahead,
                            r_bi_accel_m_per_s2,
                            r_bi_jerk_m_per_s3,
                            dt,
                        );
                        let as_bi_min = as_bi.iter().cloned().reduce(f64::min).unwrap_or(0.0);
                        let as_bi_max = as_bi.iter().cloned().reduce(f64::max).unwrap_or(0.0);
                        let accel_spread = (as_bi_max - as_bi_min).abs();
                        let flag = as_bi_max < (max_accel_m_per_s2 + 1e-6)
                            && as_bi_min > (min_accel_m_per_s2 - 1e-6)
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
            (
                r_best_found,
                r_best_n,
                r_best_jerk_m_per_s3,
                r_best_accel_m_per_s2,
            )
        } else {
            not_found
        }
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
    /// - i: the step index
    ///
    /// NOTE: Resets the coast_delay_index to 0 and calculates and sets the next
    /// appropriate coast_delay_index if appropriate
    pub fn set_coast_delay(&mut self, i: usize) {
        let speed_tol = 0.01; // m/s
        let dist_tol = 0.1; // m
        for idx in i..self.cyc0.speed.len() {
            // clear all future coast delays
            self.coast_delay_index[idx] = 0;
        }
        let mut coast_delay = None;
        if !self.idm_allow && self.cyc.speed[i].get::<si::meter_per_second>() < speed_tol {
            let d0 = trapz_step_start_distance(&self.cyc, i).get::<si::meter>();
            let d0_lv = self.cyc0_cache.trapz_distances_m[i - 1];
            let dtlv0 = d0_lv - d0;
            if dtlv0.abs() > dist_tol {
                let mut d_lv = 0.0;
                let mut min_dtlv = None;
                for (idx, (&dd, &v)) in trapz_step_distances(&self.cyc0)
                    .iter()
                    .zip(self.cyc0.speed.iter())
                    .enumerate()
                {
                    let dd = dd.get::<si::meter>();
                    let v = v.get::<si::meter_per_second>();
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
                for idx in i..self.cyc0.speed.len() {
                    self.coast_delay_index[idx] = new_cd;
                    new_cd += 1;
                    if new_cd == 0 {
                        break;
                    }
                }
            } else {
                for idx in i..self.cyc0.speed.len() {
                    self.coast_delay_index[idx] = cd;
                }
            }
        }
    }

    /// Generate a coast trajectory without actually modifying the cycle.
    /// This can be used to calculate the distance to stop via coast using
    /// actual time-stepping and changing grade.
    pub fn generate_coast_trajectory(&mut self, i: usize) -> CoastTrajectory {
        let v0 = self.cyc.speed[i - 1].get::<si::meter_per_second>();
        let v_brake = self.coast_brake_start_speed.get::<si::meter_per_second>();
        let a_brake = {
            let result = self.coast_brake_accel.get::<si::meter_per_second_squared>();
            if result > 0.0 {
                -result
            } else {
                result
            }
        };
        let ds = &self.cyc0_cache.trapz_distances_m;
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
            return CoastTrajectory {
                found_trajectory: false,
                distance_to_stop_via_coast_m: 0.0,
                start_idx: 0,
                speed_m_per_s: None,
                distance_to_brake_m: None,
            };
        }
        if v0 <= v_brake {
            return CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: -0.5 * v0 * v0 / a_brake,
                start_idx: i,
                speed_m_per_s: None,
                distance_to_brake_m: None,
            };
        }
        // dtb = distance to brake: distance traveled during the friction
        //       braking part of the maneuver
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
        let iters_per_step = if self.favor_grade_accuracy { 2 } else { 1 };
        let mut new_speeds_m_per_s = Vec::with_capacity(max_iter as usize);
        let mut v = v0;
        let mut iter = 0;
        let mut idx = i;
        // dts0 = distance to next stop from d0
        let dts0 = self
            .cyc0
            .calc_distance_to_next_stop_from(d0 * uc::M, Some(&self.cyc0_cache))
            .get::<si::meter>();
        while v > v_brake
            && v >= 0.0
            && d <= d_max
            && iter < max_iter
            && idx < self.cyc0.speed.len()
        {
            let dt_s = (self.cyc0.time[i] - self.cyc0.time[i - 1]).get::<si::second>();
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
                if self.favor_grade_accuracy {
                    gr = match unique_grade {
                        Some(g) => g,
                        None => {
                            let dist = (d + d0) * uc::M;
                            let delta_dist = dd * uc::M;
                            self.cyc0
                                .average_grade_over_range(dist, delta_dist, Some(&self.cyc0_cache))
                                .get::<si::ratio>()
                        }
                    };
                }
            }
            if k >= 0.0 && has_unique_grade {
                // there is no solution for coast-down -- speed will never decrease
                return CoastTrajectory {
                    found_trajectory: false,
                    distance_to_stop_via_coast_m: 0.0,
                    start_idx: 0,
                    speed_m_per_s: None,
                    distance_to_brake_m: None,
                };
            }
            if v_next <= v_brake {
                break;
            }
            vavg = 0.5 * (v + v_next);
            dd = vavg * dt_s;
            // dtb = distance to break
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
            // dtsc = distance to stop via coasting
            let dtsc = d + dtb_target;
            return CoastTrajectory {
                found_trajectory: true,
                distance_to_stop_via_coast_m: dtsc,
                start_idx: i,
                speed_m_per_s: Some(new_speeds_m_per_s),
                distance_to_brake_m: Some(dtb_target),
            };
        }
        CoastTrajectory {
            found_trajectory: false,
            distance_to_stop_via_coast_m: 0.0,
            start_idx: 0,
            speed_m_per_s: None,
            distance_to_brake_m: None,
        }
    }

    /// Allply the given coasting trajectory to the drive cycle.
    fn apply_coast_trajectory(&mut self, coast_traj: &CoastTrajectory) {
        if !coast_traj.found_trajectory {
            return;
        }
        let num_speeds = match &coast_traj.speed_m_per_s {
            Some(speeds_m_per_s) => {
                for (di, &new_speed) in speeds_m_per_s.iter().enumerate() {
                    let idx = coast_traj.start_idx + di;
                    if idx >= self.cyc0.speed.len() {
                        break;
                    }
                    self.cyc.speed[idx] = new_speed * uc::MPS;
                }
                speeds_m_per_s.len()
            }
            None => 0,
        };
        let (_, n) = self.cyc.modify_with_braking_trajectory(
            self.coast_brake_accel,
            coast_traj.start_idx + num_speeds,
            coast_traj.distance_to_brake_m.map(|d| d * uc::M),
        );
        for di in 0..(self.cyc0.speed.len() - coast_traj.start_idx) {
            let idx = coast_traj.start_idx + di;
            self.impose_coast[idx] = di < num_speeds + n;
        }
    }

    /// Calculates the derivative dv/dd (change in speed by change in distance)
    /// - speed_m_per_s: the speed at which to evaluate dv/dd (m/s)
    /// - grade: the road grade as a decimal fraction
    ///
    /// RETURN: number, the dv/dd for these conditions
    pub fn calc_dvdd(&self, speed_m_per_s: f64, grade: f64) -> f64 {
        let v = speed_m_per_s;
        if v <= 0.0 {
            return 0.0;
        }
        let (atan_grade_sin, atan_grade_cos) = if grade == 0.0 {
            (0.0, 1.0)
        } else {
            let atan_grade = grade.atan();
            (atan_grade.sin(), atan_grade.cos())
        };
        let g = uc::ACC_GRAV.get::<si::meter_per_second_squared>();
        let m = self.mass.get::<si::kilogram>();
        let rho_cdfa = self.air_density.get::<si::kilogram_per_cubic_meter>()
            * self.drag_coef.get::<si::ratio>()
            * self.frontal_area.get::<si::square_meter>();
        let rrc = self.wheel_rr_coef.get::<si::ratio>();
        -((g / v) * (atan_grade_sin + rrc * atan_grade_cos) + (0.5 * rho_cdfa * (1.0 / m) * v))
    }

    /// Prevent collision between the vehicle in cyc and the one in cyc0.
    /// If a collision will take place, reworks the cyc such that a rendezvous occurs instead.
    ///
    /// # Arguments
    /// - i: int, index for consideration
    /// - passing_tol_m: None | float, tolerance for how far we have to go past the lead vehicle to be considered "passing"
    ///
    /// RETURN: Bool, True if cyc was modified
    fn prevent_collisions(&mut self, i: usize, passing_tol: Option<si::Length>) -> bool {
        let passing_tol_m = passing_tol.map(|pt| pt.get::<si::meter>()).unwrap_or(1.0);
        let pass_info = PassingInfo::from(&self.cyc, &self.cyc0, i, passing_tol);
        if !pass_info.passing_detected {
            return false;
        }
        let mut best = RendezvousTrajectory {
            found_trajectory: false,
            idx: 0,
            n: 0,
            full_brake_steps: 0,
            jerk_m_per_s3: 0.0,
            accel0_m_per_s2: 0.0,
            accel_spread: 0.0,
        };
        let a_brake_m_per_s2 = {
            let result = self.coast_brake_accel.get::<si::meter_per_second_squared>();
            if result > 0.0 {
                -result
            } else {
                result
            }
        };
        for full_brake_steps in 0..4 {
            for di in 0..(self.cyc.speed.len() - i) {
                let idx = i + di;
                if !self.impose_coast[idx] {
                    if idx == i {
                        break;
                    } else {
                        continue;
                    }
                }
                let n = pass_info.index - idx + 1 - full_brake_steps;
                if n < 2 {
                    break;
                }
                if (idx - 1 + full_brake_steps) >= self.cyc.speed.len() {
                    break;
                }
                let dt = pass_info.time_step_duration.get::<si::second>();
                let v_start_m_per_s = self.cyc.speed[idx - 1].get::<si::meter_per_second>();
                let dt_full_brake = (self.cyc.time[idx - 1 + full_brake_steps]
                    - self.cyc.time[idx - 1])
                    .get::<si::second>();
                let dv_full_brake = dt_full_brake * a_brake_m_per_s2;
                let v_start_jerk_m_per_s = (v_start_m_per_s + dv_full_brake).max(0.0);
                let dd_full_brake = 0.5 * (v_start_m_per_s + v_start_jerk_m_per_s) * dt_full_brake;
                let d_start_m =
                    trapz_step_start_distance(&self.cyc, idx).get::<si::meter>() + dd_full_brake;
                let pass_distance_m = pass_info.distance.get::<si::meter>();
                if pass_distance_m <= d_start_m {
                    continue;
                }
                let jerk_trajectory = calc_constant_jerk_trajectory(
                    n,
                    d_start_m,
                    v_start_jerk_m_per_s,
                    pass_info.distance.get::<si::meter>(),
                    pass_info.speed.get::<si::meter_per_second>(),
                    dt,
                );
                let mut accels_m_per_s2 = vec![];
                let mut trace_accels_m_per_s2 = vec![];
                for ni in 0..n {
                    if (ni + idx + full_brake_steps) >= self.cyc.time.len() {
                        break;
                    }
                    accels_m_per_s2.push(accel_for_constant_jerk(
                        ni,
                        jerk_trajectory.acceleration_m_per_s2,
                        jerk_trajectory.jerk_m_per_s3,
                        jerk_trajectory.step_duration_s,
                    ));
                    let index1 = ni + idx + full_brake_steps;
                    let index0 = index1 - 1;
                    let dvi = (self.cyc.speed[index1] - self.cyc.speed[index0])
                        .get::<si::meter_per_second>();
                    let dti = (self.cyc.time[index1] - self.cyc.time[index0]).get::<si::second>();
                    trace_accels_m_per_s2.push(dvi / dti);
                }
                let all_sub_coast = trace_accels_m_per_s2
                    .iter()
                    .copied()
                    .zip(accels_m_per_s2.iter().copied())
                    .fold(
                        true,
                        |all_sc_flag: bool, (trace_accel, accel): (f64, f64)| {
                            if !all_sc_flag {
                                return all_sc_flag;
                            }
                            trace_accel >= accel
                        },
                    );
                let (min_accel_m_per_s2, max_accel_m_per_s2) = {
                    if !accels_m_per_s2.is_empty() {
                        let mut a_min = accels_m_per_s2[0];
                        let mut a_max = accels_m_per_s2[0];
                        for a in &accels_m_per_s2 {
                            if *a < a_min {
                                a_min = *a;
                            }
                            if *a > a_max {
                                a_max = *a;
                            }
                        }
                        (a_min, a_max)
                    } else {
                        (0.0, 0.0)
                    }
                };
                let accept = all_sub_coast;
                let accel_spread = (max_accel_m_per_s2 - min_accel_m_per_s2).abs();
                if accept && (!best.found_trajectory || accel_spread < best.accel_spread) {
                    best = RendezvousTrajectory {
                        found_trajectory: true,
                        idx,
                        n,
                        full_brake_steps,
                        jerk_m_per_s3: jerk_trajectory.jerk_m_per_s3,
                        accel0_m_per_s2: jerk_trajectory.acceleration_m_per_s2,
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
            return self.prevent_collisions(i, Some(new_passing_tol_m * uc::M));
        }
        for fbs in 0..best.full_brake_steps {
            if (best.idx + fbs) >= self.cyc.time.len() {
                break;
            }
            let dt =
                (self.cyc.time[best.idx + fbs] - self.cyc.time[best.idx - 1]).get::<si::second>();
            let dv = a_brake_m_per_s2 * dt;
            let v_start = self.cyc.speed[best.idx - 1].get::<si::meter_per_second>();
            self.cyc.speed[best.idx + fbs] = (v_start + dv).max(0.0) * uc::MPS;
            self.impose_coast[best.idx + fbs] = true;
            self.coast_delay_index[best.idx + fbs] = 0;
        }
        self.cyc.modify_by_const_jerk_trajectory(
            best.idx + best.full_brake_steps,
            best.n,
            best.jerk_m_per_s3 * uc::MPS3,
            best.accel0_m_per_s2 * uc::MPS2,
        );
        for idx in (best.idx + best.n)..self.cyc0.speed.len() {
            self.impose_coast[idx] = false;
            self.coast_delay_index[idx] = 0;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::SimDrive;

    use super::*;

    #[test]
    fn test_that_coasting_works() {
        let udds = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let veh = crate::vehicle::Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
        let mut man = Maneuver::from(&udds, &veh);
        man.coast_allow = true;
        man.coast_start_speed = 20.0 * uc::MPS;
        man.coast_allow_passing = true;
        man.apply();
        let udds_mod = man.cyc;
        assert_eq!(udds_mod.time.len(), udds.time.len());
        assert_eq!(udds_mod.speed.len(), udds.speed.len());
        assert_eq!(udds_mod.dist.len(), udds.dist.len());
        let mut speeds_differ = false;
        for idx in 0..udds.time.len() {
            speeds_differ = udds_mod.speed[idx] != udds.speed[idx];
            if speeds_differ {
                break;
            }
        }
        assert!(speeds_differ);
    }

    #[test]
    fn test_advanced_coasting() {
        let udds = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let veh = crate::vehicle::Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
        let mut man = Maneuver::from(&udds, &veh);
        man.coast_allow = true;
        man.coast_start_speed = 0.0 * uc::MPS;
        man.coast_brake_start_speed = 20.0 * uc::MPH;
        man.coast_brake_accel = -2.5 * uc::MPS2;
        man.favor_grade_accuracy = false;
        man.coast_allow_passing = true;
        man.coast_max_speed = 75.0 * uc::MPH;
        man.coast_time_horizon_for_adjustment = 120.0 * uc::S;
        man.apply();
        let udds_mod = man.cyc;
        let mut sd = SimDrive::new(veh, udds_mod, None);
        sd.walk().unwrap();
    }

    #[test]
    fn test_cruise() {
        let udds = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let vavg = udds.average_speed(true);
        let veh = crate::vehicle::Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
        let mut man = Maneuver::from(&udds, &veh);
        man.idm_allow = true;
        man.idm_desired_speed = vavg;
        man.idm_headway = 1.0 * uc::S;
        man.idm_minimum_gap = 1.0 * uc::M;
        man.idm_delta = 4.0;
        man.idm_acceleration = 1.0 * uc::MPS2;
        man.idm_deceleration = 2.5 * uc::MPS2;
        man.coast_allow = false;
        man.apply();
        let udds_mod = man.cyc;
        let mut sd = SimDrive::new(veh, udds_mod, None);
        sd.walk().unwrap();
    }

    #[test]
    fn test_cruise_and_coast() {
        let udds = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let vavg = udds.average_speed(true);
        let veh = crate::vehicle::Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
        let mut man = Maneuver::from(&udds, &veh);
        man.idm_allow = true;
        man.idm_desired_speed = vavg;
        man.idm_headway = 1.0 * uc::S;
        man.idm_minimum_gap = 1.0 * uc::M;
        man.idm_delta = 4.0;
        man.idm_acceleration = 1.0 * uc::MPS2;
        man.idm_deceleration = 2.5 * uc::MPS2;
        man.coast_allow = true;
        man.coast_brake_start_speed = 8.9408 * uc::MPS;
        man.coast_brake_accel = -2.5 * uc::MPS2;
        man.favor_grade_accuracy = true;
        man.coast_allow_passing = true;
        man.coast_time_horizon_for_adjustment = 120.0 * uc::S;
        man.apply();
        let udds_mod = man.cyc;
        let mut found_coast = false;
        for ic in man.impose_coast {
            if ic {
                found_coast = true;
                break;
            }
        }
        assert!(found_coast);
        let mut sd = SimDrive::new(veh, udds_mod, None);
        sd.walk().unwrap();
    }
}
