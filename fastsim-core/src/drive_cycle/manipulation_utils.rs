use crate::drive_cycle::Cycle;
use crate::imports::*;

/// Rendezvous Trajectory that uses a constant-jerk trajectory to rendezvous
/// with another trace in distance/time.
pub struct RendezvousTrajectory {
    pub found_trajectory: bool,
    pub idx: usize,
    pub n: usize,
    pub full_brake_steps: usize,
    pub jerk_m_per_s3: f64,
    pub accel0_m_per_s2: f64,
    pub accel_spread: f64,
}

/// Coasting trajectory that describes the characteristics of a time/speed coast.
pub struct CoastTrajectory {
    pub found_trajectory: bool,
    pub distance_to_stop_via_coast_m: f64,
    pub start_idx: usize,
    pub speed_m_per_s: Option<Vec<f64>>,
    pub distance_to_brake_m: Option<f64>,
}

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Data describing a trajectory with a "constant jerk"
pub struct ConstantJerkTrajectory {
    /// number of steps in the trajectory
    pub steps: usize,
    /// initial elapsed distance at trajectory start (m)
    pub distance_m: f64,
    /// initial speed of the trajectory (m/s)
    pub speed_m_per_s: f64,
    /// initial acceleration of the trajectory (m/s2)
    pub acceleration_m_per_s2: f64,
    /// constant jerk of the trajectory (m/s3)
    pub jerk_m_per_s3: f64,
    /// duration of a single step in seconds
    pub step_duration_s: f64,
}

impl SerdeAPI for ConstantJerkTrajectory {}
impl Init for ConstantJerkTrajectory {}

impl ConstantJerkTrajectory {
    /// Create a constant-jerk trajectory.
    /// - n: the number of steps to use
    /// - d0: the starting distance (m)
    /// - v0: the starting speed (m/s)
    /// - dr: the rendezvous distance (m)
    /// - vr: the rendezvous speed (m/s)
    /// - dt: constant step duration (s)
    ///
    /// RETURN: a ConstantJerkTrajectory
    pub fn from_speed_and_distance_targets(
        n: usize,
        d0: f64,
        v0: f64,
        dr: f64,
        vr: f64,
        dt: f64,
    ) -> ConstantJerkTrajectory {
        assert!(n > 1);
        assert!(dr > d0);
        let n_orig = n;
        let n = n as f64;
        let ddr = dr - d0;
        let dvr = vr - v0;
        let k = (dvr - (2.0 * ddr / (n * dt)) + 2.0 * v0)
            / (0.5 * n * (n - 1.0) * dt
                - (1.0 / 3.0) * (n - 1.0) * (n - 2.0) * dt
                - 0.5 * (n - 1.0) * dt * dt);
        let a0 = ((ddr / dt)
            - n * v0
            - ((1.0 / 6.0) * n * (n - 1.0) * (n - 2.0) * dt + 0.25 * n * (n - 1.0) * dt * dt) * k)
            / (0.5 * n * n * dt);
        ConstantJerkTrajectory {
            steps: n_orig,
            distance_m: d0,
            speed_m_per_s: v0,
            acceleration_m_per_s2: a0,
            jerk_m_per_s3: k,
            step_duration_s: dt,
        }
    }
    /// Calculate the distance traveled by the end of the nth step in m.
    pub fn distance_at_step(&self, n: usize) -> f64 {
        let n = n as f64;
        let d0 = self.distance_m;
        let v0 = self.speed_m_per_s;
        let a0 = self.acceleration_m_per_s2;
        let k = self.jerk_m_per_s3;
        let dt = self.step_duration_s;
        let term1 = dt
            * ((n * v0)
                + (0.5 * n * (n - 1.0) * a0 * dt)
                + ((1.0 / 6.0) * k * dt * (n - 2.0) * (n - 1.0) * n));
        let term2 = 0.5 * dt * dt * ((n * a0) + (0.5 * n * (n - 1.0) * k * dt));
        d0 + term1 + term2
    }
    /// Calculate the ending distance of the trajectory in m.
    pub fn end_distance(&self) -> f64 {
        self.distance_at_step(self.steps)
    }
    /// Calculate the ending speed for the nth step in m/s.
    pub fn speed_at_step(&self, n: usize) -> f64 {
        let n = n as f64;
        let v0 = self.speed_m_per_s;
        let a0 = self.acceleration_m_per_s2;
        let k = self.jerk_m_per_s3;
        let dt = self.step_duration_s;
        v0 + (n * a0 * dt) + (0.5 * n * (n - 1.0) * k * dt)
    }
    /// Calculate the ending speed in m/s.
    pub fn end_speed(&self) -> f64 {
        self.speed_at_step(self.steps)
    }
    /// Calculate the acceleration for step n in m/s2.
    pub fn acceleration_at_step(&self, n: usize) -> f64 {
        let n = n as f64;
        let a0 = self.acceleration_m_per_s2;
        let k = self.jerk_m_per_s3;
        let dt = self.step_duration_s;
        a0 + (n * k * dt)
    }
    /// Calculate the acceleration at end of the trajectory in m/s2.
    pub fn end_acceleration(&self) -> f64 {
        self.acceleration_at_step(self.steps)
    }
    /// Calculate and return a vector of all of the step-wise accelerations for
    /// the trajectory in m/s2.
    pub fn all_accelerations(&self) -> Vec<f64> {
        let mut accels = Vec::with_capacity(self.steps);
        for n_idx in 0..self.steps {
            accels.push(self.acceleration_at_step(n_idx));
        }
        accels
    }
    /// Calculate and return the maximum acceleration over the trajectory in m/s2.
    pub fn maximum_acceleration(&self) -> f64 {
        let accels = self.all_accelerations();
        *accels.max().unwrap_or(&0.0)
    }
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate constant-Jerk trajectory.
/// - n: number of time steps away from rendezvous point
/// - d0: distance from start of simulated vehicle ($m$)
/// - v0: speed of simulated vehicle ($\frac{m}{s}$)
/// - dr: distance from start of rendezvous point ($m$)
/// - vr: speed to hit at rendezvous point ($\frac{m}{s}$)
/// - dt: time-step duration ($s$)
/// RETURN: constant jerk and acceleration for first time step.
pub fn calc_constant_jerk_trajectory(
    n: usize,
    d0: f64,
    v0: f64,
    dr: f64,
    vr: f64,
    dt: f64,
) -> ConstantJerkTrajectory {
    assert!(n > 1);
    assert!(dr > d0);
    assert!(v0 >= 0.0);
    assert!(vr >= 0.0);
    assert!(dt > 0.0);
    ConstantJerkTrajectory::from_speed_and_distance_targets(n, d0, v0, dr, vr, dt)
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate distance (m) after n timesteps
/// - n: number of timesteps away to calculate
/// - d0: initial distance (m)
/// - v0: initial speed (m/s)
/// - a0: initial acceleration (m/s2)
/// - k: constant jerk (m/s3)
/// - dt: duration of a timestep (s)
/// RETURN: distance a n timesteps away (m)
/// NOTE: this is the distance traveled from start (i.e., when n=0)
/// measured at sample point n
pub fn dist_for_constant_jerk(n: usize, d0: f64, v0: f64, a0: f64, k: f64, dt: f64) -> f64 {
    let trajectory = ConstantJerkTrajectory {
        steps: n,
        distance_m: d0,
        speed_m_per_s: v0,
        acceleration_m_per_s2: a0,
        jerk_m_per_s3: k,
        step_duration_s: dt,
    };
    trajectory.end_distance()
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate speed (m/s) n timesteps away via a constant-jerk acceleration
/// - n: number of timesteps away to calculate
/// - v0: initial speed (m/s)
/// - a0: initial acceleration (m/s2)
/// - k: constant jerk (m/s3)
/// - dt: duration of a time step (s)
/// RETURN: the speed n timesteps away (m/s)
/// NOTES:
/// - this is the speed at sample-point n
/// - if n == 0, speed is v0
/// - if n == 1, speed is v* + a0 * dt, etc.
pub fn speed_for_constant_jerk(n: usize, v0: f64, a0: f64, k: f64, dt: f64) -> f64 {
    let trajectory = ConstantJerkTrajectory {
        steps: n,
        distance_m: 0.0,
        speed_m_per_s: v0,
        acceleration_m_per_s2: a0,
        jerk_m_per_s3: k,
        step_duration_s: dt,
    };
    trajectory.end_speed()
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate the acceleration n timesteps away (m/s2)
/// - n: number of time steps away to calculate
/// - a0: initial acceleration (m/s2)
/// - k: constant jerk (m/s3)
/// - dt: time-step duration (s)
/// RETURN: the acceleration n timesteps away (m/s)
/// NOTES:
/// - this is the constant accerlation over the time-step from sample n to n+1
pub fn accel_for_constant_jerk(n: usize, a0: f64, k: f64, dt: f64) -> f64 {
    let trajectory = ConstantJerkTrajectory {
        steps: n,
        distance_m: 0.0,
        speed_m_per_s: 0.0,
        acceleration_m_per_s2: a0,
        jerk_m_per_s3: k,
        step_duration_s: dt,
    };
    trajectory.end_acceleration()
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Apply accel_for_constant_jerk to derive full array of accelerations.
/// - n: number of time steps away to calculate
/// - a0: initial acceleration (m/s2)
/// - k: constant jerk (m/s3)
/// - dt: time-step duration (s)
/// RETURN: the accelerations for each timestep up to n timesteps away (m/s)
pub fn accel_array_for_constant_jerk(n: usize, a0: f64, k: f64, dt: f64) -> Vec<f64> {
    let trajectory = ConstantJerkTrajectory {
        steps: n,
        distance_m: 0.0,
        speed_m_per_s: 0.0,
        acceleration_m_per_s2: a0,
        jerk_m_per_s3: k,
        step_duration_s: dt,
    };
    trajectory.all_accelerations()
}

/// Return the average step speeds of the cycle as vector of velicities.
/// NOTE: the average speed from sample i-1 to i will appear as entry i.
/// - cyc: an instance of the cycle to get average step speeds for.
///
/// RETURN: vector of velocities representing average step speeds.
pub fn average_step_speeds(cyc: &Cycle) -> Vec<si::Velocity> {
    cyc.average_step_speeds()
}

/// Calculate the average step speed at step i
/// (i.e., from sample point i-1 to i)
pub fn average_step_speed_at(cyc: &Cycle, i: usize) -> si::Velocity {
    cyc.average_step_speed_at(i)
}

/// The distances traveled over each step using trapezoidal
/// integration.
pub fn trapz_step_distances(cyc: &Cycle) -> Vec<si::Length> {
    cyc.trapz_step_distances()
}

/// The distance traveled from start to the beginning of step i
/// (i.e., distance traveled up to sample point i-1)
pub fn trapz_step_start_distance(cyc: &Cycle, i: usize) -> si::Length {
    cyc.trapz_step_start_distance(i)
}

/// The distance traveled during the given step
/// (i.e., distance from sample point i-1 to i for step i)
pub fn trapz_distance_for_step(cyc: &Cycle, i: usize) -> si::Length {
    cyc.trapz_distance_for_step(i)
}

/// Calculate the distance from step i_start to the start of step i_end
/// (i.e., distance from sample point i_start - 1 to i_end - 1)
pub fn trapz_distance_over_range(cyc: &Cycle, i_start: usize, i_end: usize) -> si::Length {
    cyc.trapz_distance_over_range(i_start, i_end)
}

/// Calculate the time in a cycle spent moving
/// - stopped_speed_m_per_s: the speed above which we are considered to be moving
///
/// RETURN: the time spent moving in seconds
pub fn time_spent_moving(cyc: &Cycle, stopped_speed: Option<si::Velocity>) -> si::Time {
    cyc.time_spent_moving(stopped_speed)
}

/// Create distance and target speeds by microtrip.
/// Used to set target speeds for each microtrip.
/// - cyc: the cycle to operate on
/// - stop_speed: speed at or below which we consider the vehicle "stopped"
/// - blend_factor: a ratio between 0 and 1. At "0", we use the average speed
///   including stopped time; at "1" we use only the speed while moving to set
///   the average.
/// - min_target_speed: the minimum speed we allow a vehicle to drop down to
///   over a microtrip
///
/// RETURN vector of distance and speed targets. The interpretation is that
/// "at or above" the given distance, the given speed will be in effect as
/// the target speed (until we pass the next entry's distance).
pub fn create_distance_and_target_speeds_by_microtrip(
    cyc: &Cycle,
    stop_speed: Option<si::Velocity>,
    blend_factor: f64,
    min_target_speed: si::Velocity,
) -> Vec<(si::Length, si::Velocity)> {
    cyc.distance_and_target_speeds_by_microtrip(stop_speed, blend_factor, min_target_speed)
}

/// Extend the cycle's time.
/// - absolute_time: an absolute time value
/// - time_fraction: extend by the given fraction of cycle's current time
///
/// RETURN: new cycle with time extended by both the absolute
///         and fraction values
///
/// NOTE: absolute_time and time_faction are optional. Pass None to
/// remove them from the equation.
pub fn extend_cycle_time(
    cyc: &Cycle,
    absolute_time: Option<si::Time>,
    time_fraction: Option<si::Ratio>,
) -> Cycle {
    cyc.extend_time(absolute_time, time_fraction)
}

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Default)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct PassingInfo {
    /// True if first cycle passes the second; NOTE: was `has_collision`
    pub passing_detected: bool,
    /// the index where first cycle passes the second; NOTE: was `idx`
    pub index: usize,
    /// the number of time-steps until index from i
    pub num_steps: usize,
    /// the starting distance of the first cycle at i
    pub start_distance: si::Length,
    /// the distance traveled of the second cycle when the first passes
    pub distance: si::Length,
    /// the starting speed of the first cycle at i
    pub start_speed: si::Velocity,
    /// the speed of the second cycle when first passes
    pub speed: si::Velocity,
    /// the (assumed constant) time step duration throughout the passing investigation
    pub time_step_duration: si::Time,
}

impl PassingInfo {
    /// Create a new PassingInfo struct from a cycle and reference cycle.
    /// - cyc: the cycle to detect passing for
    /// - cyc0: the reference cycle / lead vehicle / shadow cycle to compare cyc with
    /// - i: the time-step index for the start of consideration
    /// - distance_tolerance: the distance away from the lead vehicle at or above which
    ///   we consider ourselves "deviated" or "no longer following" the reference trace
    ///
    /// RETURN: a PassingInfo structure
    pub fn from(
        cyc: &Cycle,
        cyc_ref: &Cycle,
        i: usize,
        distance_tolerance: Option<si::Length>,
    ) -> Self {
        let i = std::cmp::max(i, 1);
        if i >= cyc.time.len() {
            return Self {
                passing_detected: false,
                index: 0,
                num_steps: 0,
                start_distance: 0.0 * uc::M,
                distance: 0.0 * uc::M,
                start_speed: 0.0 * uc::MPS,
                speed: 0.0 * uc::MPS,
                time_step_duration: 1.0 * uc::S,
            };
        }
        let zero_speed_tol = 1e-6 * uc::MPS;
        let distance_tol = distance_tolerance.unwrap_or(0.1 * uc::M);
        let mut v0 = cyc.speed[i - 1];
        let d0 = cyc.trapz_step_start_distance(i);
        let mut v0_lv = cyc_ref.speed[i - 1];
        let d0_lv = cyc_ref.trapz_step_start_distance(i);
        let mut d = d0;
        let mut d_lv = d0_lv;
        let mut rendezvous_index = None;
        let mut rendezvous_num_steps = 0;
        let mut rendezvous_distance = 0.0 * uc::M;
        let mut rendezvous_speed = 0.0 * uc::MPS;
        for di in 0..(cyc.speed.len() - i) {
            let idx = i + di;
            // cycle current speed
            let v = cyc.speed[idx];
            // lead vehicle current speed
            let v_lv = cyc_ref.speed[idx];
            // cycle average speed for step
            let vavg = (v + v0) * 0.5;
            // lead vehicle average speed for step
            let vavg_lv = (v_lv + v0_lv) * 0.5;
            // time step duration
            let dt = cyc.time[idx] - cyc.time[idx - 1];
            // delta distance for step
            let dd = vavg * dt;
            // time step duration for lead vehicle
            let dt_lv = cyc_ref.time[idx] - cyc_ref.time[idx - 1];
            // delta distance for lead vehicle for step
            let dd_lv = vavg_lv * dt_lv;
            // total distance from start
            d += dd;
            // total distance from start for lead vehicle
            d_lv += dd_lv;
            // distance to lead vehicle
            let dtlv = d_lv - d;
            v0 = v;
            v0_lv = v_lv;
            if di > 0 && dtlv < -distance_tol {
                rendezvous_index = Some(idx);
                rendezvous_num_steps = di + 1;
                rendezvous_distance = d_lv;
                rendezvous_speed = v_lv;
                break;
            }
            if v <= zero_speed_tol {
                break;
            }
        }
        Self {
            passing_detected: rendezvous_index.is_some(),
            index: rendezvous_index.unwrap_or(0),
            num_steps: rendezvous_num_steps,
            start_distance: d0,
            distance: rendezvous_distance,
            start_speed: cyc.speed[i - 1],
            speed: rendezvous_speed,
            time_step_duration: cyc.time[i] - cyc.time[i - 1],
        }
    }
}

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct CycleCache {
    /// flag to indicate if cycle has all-zero grade (i.e., flat) or not
    pub grade_all_zero: bool,
    /// distance traveled over each time-step of the cycle
    pub trapz_step_distances_m: Vec<f64>,
    /// distances from start at each sample point
    pub trapz_distances_m: Vec<f64>,
    /// elevations at each sample point
    pub trapz_elevations_m: Vec<f64>,
    /// an array of flags indicating whether stopped (true) or not (false)
    pub stops: Vec<bool>,
    /// interpolation_distances
    interp_ds: Vec<f64>,
    /// interpolation of indices
    interp_is: Vec<f64>,
    /// interpolation of heights (i.e., elevations)
    interp_hs: Vec<f64>,
    /// grades where g[i] applies from distance [i, i+1)
    grades: Vec<f64>,
    /// interpolator for index by distance
    interp_index_by_dist: InterpolatorEnumOwned<f64>,
    /// interpolator for elevation by distance
    interp_elev_by_dist: InterpolatorEnumOwned<f64>,
}

impl Default for CycleCache {
    fn default() -> Self {
        Self {
            grade_all_zero: false,
            trapz_step_distances_m: Default::default(),
            trapz_distances_m: Default::default(),
            trapz_elevations_m: Default::default(),
            stops: Default::default(),
            interp_ds: Default::default(),
            interp_is: Default::default(),
            interp_hs: Default::default(),
            grades: Default::default(),
            interp_index_by_dist: InterpolatorEnum::new_0d(0.0),
            interp_elev_by_dist: InterpolatorEnum::new_0d(0.0),
        }
    }
}

impl Init for CycleCache {}

impl SerdeAPI for CycleCache {}

impl CycleCache {
    /// Create a new cycle cache from a cycle.
    pub fn new(cyc: &Cycle) -> Self {
        let tol = 1e-6;
        let num_items = cyc.time.len();
        let grade_all_zero = cyc.grade.is_empty() || cyc.grade.iter().all(|g| *g == 0.0 * uc::R);
        let trapz_step_distances_m: Vec<f64> = cyc
            .trapz_step_distances()
            .iter()
            .map(|dd| dd.get::<si::meter>())
            .collect();
        debug_assert!(trapz_step_distances_m.len() == num_items);
        let trapz_distances_m: Vec<f64> = {
            let mut ds = Vec::with_capacity(num_items);
            let mut d = 0.0;
            for dd in &trapz_step_distances_m {
                d += *dd;
                ds.push(d);
            }
            ds
        };
        debug_assert!(trapz_distances_m.len() == num_items);
        let trapz_elevations_m = if grade_all_zero {
            let h = cyc.init_elev.unwrap_or(0.0 * uc::M).get::<si::meter>();
            vec![h; num_items]
        } else {
            let dhs: Vec<f64> = cyc
                .grade
                .iter()
                .zip(&trapz_step_distances_m)
                .map(|(g, dd)| {
                    let gr = g.get::<si::ratio>();
                    gr.atan().cos() * dd * gr
                })
                .collect();
            let mut hs = Vec::with_capacity(num_items);
            let mut h = cyc.init_elev.unwrap_or(0.0 * uc::M).get::<si::meter>();
            for dh in &dhs {
                h += *dh;
                hs.push(h);
            }
            hs
        };
        debug_assert!(trapz_elevations_m.len() == num_items);
        let stops = cyc
            .speed
            .iter()
            .map(|v| v.get::<si::meter_per_second>() <= tol)
            .collect();
        let mut interp_ds = Vec::with_capacity(num_items);
        let mut interp_is = Vec::with_capacity(num_items);
        let mut interp_hs = Vec::with_capacity(num_items);
        for idx in 0..num_items {
            let d = trapz_distances_m[idx];
            let h = trapz_elevations_m[idx];
            if interp_ds.is_empty() || d > *interp_ds.last().unwrap() {
                interp_ds.push(d);
                interp_is.push(idx as f64);
                interp_hs.push(h);
            }
        }
        let grades: Vec<f64> = cyc.grade.iter().map(|g| g.get::<si::ratio>()).collect();
        debug_assert!(grades.len() == num_items);
        let interp_index_by_dist = InterpolatorEnum::new_1d(
            interp_ds.clone().into(),
            interp_is.clone().into(),
            strategy::RightNearest,
            Extrapolate::Clamp,
        )
        .unwrap();
        let interp_elev_by_dist = InterpolatorEnum::new_1d(
            interp_ds.clone().into(),
            interp_hs.clone().into(),
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        Self {
            grade_all_zero,
            trapz_step_distances_m,
            trapz_distances_m,
            trapz_elevations_m,
            stops,
            interp_ds,
            interp_is,
            interp_hs,
            grades,
            interp_index_by_dist,
            interp_elev_by_dist,
        }
    }

    /// Interpolate the single-point grade at the given distance.
    /// Assumes that the grade at i applies from sample point (i-1, i]
    pub fn interp_grade(&self, dist_m: f64) -> f64 {
        if self.grade_all_zero {
            0.0
        } else if dist_m <= self.interp_ds[0] {
            self.grades[0]
        } else if dist_m > *self.interp_ds.last().expect("interp_ds.len()>0") {
            *self.grades.last().unwrap()
        } else {
            // NOTE: interp strategy is right nearest; equal to linear + ceil()
            let idx = self.interp_index_by_dist.interpolate(&[dist_m]).unwrap();
            self.grades[idx as usize]
        }
    }

    /// Interpolate the elevation at the given distance
    pub fn interp_elevation(&self, dist_m: f64) -> f64 {
        if self.grade_all_zero {
            0.0
        } else {
            self.interp_elev_by_dist.interpolate(&[dist_m]).unwrap()
        }
    }
}

/// Calculate a rendezvous trajectory for re-rendezvous with reference cycle.
/// - i: the index where speed has changed from reference
/// - max_steps: the maximum number of time-steps ahead that a rendezvous will be considered. Minimum is 2.
/// - cyc: the reference cycle
/// - speed_ach: the current best effort speed that deviates from reference cycle at i.
///
/// RESULT: returns a RendezvousTrajectory which describes a constant-jerk trajectory path
/// that will rendezvous with the reference trace between n=2 and max_steps steps. The trajectory
/// chosen will have the smallest peak acceleration of all options investigated.
pub fn calc_best_rendezvous(
    i: usize,
    max_steps: usize,
    cyc: &Cycle,
    speed_ach: si::Velocity,
) -> ConstantJerkTrajectory {
    let max_steps = (cyc.time.len() - i).min(max_steps);
    let i = i.clamp(1, cyc.time.len() - 1);
    let dt = cyc.time[i] - cyc.time[i - 1];
    let start_distance = 0.5 * (speed_ach + cyc.speed[i - 1]) * dt;
    let mut best = ConstantJerkTrajectory {
        steps: 0,
        distance_m: start_distance.get::<si::meter>(),
        speed_m_per_s: speed_ach.get::<si::meter_per_second>(),
        acceleration_m_per_s2: 0.0,
        jerk_m_per_s3: 0.0,
        step_duration_s: dt.get::<si::second>(),
    };
    if max_steps < 2 {
        return best;
    }
    let mut rendezvous_distance = 0.5 * (cyc.speed[i] + cyc.speed[i - 1]) * dt;
    let mut max_accel_m_per_s2 = 100.0;
    for n in 1..max_steps {
        let j = i + n;
        let dt = cyc.time[j] - cyc.time[j - 1];
        rendezvous_distance += 0.5 * (cyc.speed[j] + cyc.speed[j - 1]) * dt;
        if n >= 2 {
            let candidate = ConstantJerkTrajectory::from_speed_and_distance_targets(
                n,
                start_distance.get::<si::meter>(),
                speed_ach.get::<si::meter_per_second>(),
                rendezvous_distance.get::<si::meter>(),
                cyc.speed[j].get::<si::meter_per_second>(),
                dt.get::<si::second>(),
            );
            let candidate_max_accel_m_per_s2 = candidate.maximum_acceleration();
            if candidate_max_accel_m_per_s2 < max_accel_m_per_s2 {
                max_accel_m_per_s2 = candidate_max_accel_m_per_s2;
                best = candidate;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive_cycle::Cycle;

    fn make_triangle_cycle() -> Cycle {
        Cycle {
            name: String::from("Triangle"),
            init_elev: None,
            time: vec![0.0 * uc::S, 10.0 * uc::S, 20.0 * uc::S, 30.0 * uc::S],
            speed: vec![0.0 * uc::MPS, 4.0 * uc::MPS, 0.0 * uc::MPS, 0.0 * uc::MPS],
            dist: vec![],
            grade: vec![],
            elev: vec![],
            pwr_max_chrg: vec![],
            grade_interp: Default::default(),
            elev_interp: Default::default(),
            temp_amb_air: Default::default(),
            pwr_solar_load: Default::default(),
        }
    }
    fn make_test_trajectory() -> ConstantJerkTrajectory {
        let n = 2;
        let d0_m = 0.0;
        let v0_m_per_s = 0.0;
        let dr_m = 2.0;
        let vr_m_per_s = 2.0;
        let dt_s = 1.0;
        ConstantJerkTrajectory::from_speed_and_distance_targets(
            n, d0_m, v0_m_per_s, dr_m, vr_m_per_s, dt_s,
        )
    }
    #[test]
    fn test_calc_const_jerk_trajectory() {
        let actual = make_test_trajectory();
        let expected = ConstantJerkTrajectory {
            steps: 2,
            distance_m: 0.0,
            speed_m_per_s: 0.0,
            acceleration_m_per_s2: 1.0,
            jerk_m_per_s3: 0.0,
            step_duration_s: 1.0,
        };
        assert_eq!(actual, expected);
    }
    #[test]
    fn test_dist_for_constant_jerk() {
        let trajectory = make_test_trajectory();
        let expected = 2.0; // meters
        let actual = trajectory.end_distance();
        assert_eq!(actual, expected);
    }
    #[test]
    fn test_speed_for_constant_jerk() {
        let n = 2;
        let v0_m_per_s = 0.0;
        let a0_m_per_s2 = 1.0;
        let k_m_per_s3 = 0.0;
        let dt_s = 1.0;
        let expected = 2.0;
        let actual = speed_for_constant_jerk(n, v0_m_per_s, a0_m_per_s2, k_m_per_s3, dt_s);
        assert_eq!(actual, expected);
    }
    #[test]
    fn test_accel_for_constant_jerk() {
        let n = 2;
        let a0_m_per_s2 = 1.0;
        let k_m_per_s3 = 0.0;
        let dt_s = 1.0;
        let expected = 1.0;
        let actual = accel_for_constant_jerk(n, a0_m_per_s2, k_m_per_s3, dt_s);
        assert_eq!(actual, expected);
    }
    #[test]
    fn test_accel_array_for_constant_jerk() {
        let n = 2;
        let a0_m_per_s2 = 1.0;
        let k_m_per_s3 = 0.0;
        let dt_s = 1.0;
        let expected = [1.0, 1.0];
        let actual = accel_array_for_constant_jerk(n, a0_m_per_s2, k_m_per_s3, dt_s);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(actual[i], expected[i]);
        }
    }
    #[test]
    fn test_average_step_speeds() {
        let cyc = make_triangle_cycle();
        let expected = [0.0 * uc::MPS, 2.0 * uc::MPS, 2.0 * uc::MPS, 0.0 * uc::MPS];
        let actual = average_step_speeds(&cyc);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(actual[i], expected[i]);
        }
    }

    #[test]
    fn test_average_step_speed_at() {
        let cyc = make_triangle_cycle();
        let expected = 2.0 * uc::MPS;
        let actual = average_step_speed_at(&cyc, 1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_trapz_step_distances() {
        let cyc = make_triangle_cycle();
        let expected = [0.0 * uc::M, 20.0 * uc::M, 20.0 * uc::M, 0.0 * uc::M];
        let actual = trapz_step_distances(&cyc);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(actual[i], expected[i]);
        }
    }

    #[test]
    fn test_trapz_step_start_distance() {
        let cyc = make_triangle_cycle();
        let expected = 40.0 * uc::M;
        // NOTE: using '30' tests we can overshoot the step index with no problem.
        let actual = trapz_step_start_distance(&cyc, 30);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_trapz_distance_for_step() {
        let cyc = make_triangle_cycle();
        let expected = 20.0 * uc::M;
        let actual = trapz_distance_for_step(&cyc, 1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_trapz_distance_over_range() {
        let cyc = make_triangle_cycle();
        let expected = 40.0 * uc::M;
        // NOTE: the high end step is meant to test out-of-bounds indices.
        let actual = trapz_distance_over_range(&cyc, 0, 1000);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_time_spent_moving() {
        let cyc = make_triangle_cycle();
        let expected = 20.0 * uc::S;
        let actual = time_spent_moving(&cyc, None);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_create_distance_and_target_speeds_by_microtrip() {
        let cyc = make_triangle_cycle();
        let expected = [(0.0 * uc::M, (40.0 / 30.0) * uc::MPS)];
        let v0 = 0.0 * uc::MPS;
        let actual = create_distance_and_target_speeds_by_microtrip(&cyc, None, 0.0, v0);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(actual[i].0, expected[i].0);
            assert_eq!(actual[i].1, expected[i].1);
        }
    }

    #[test]
    fn test_extending_cycle_time() {
        let cyc = make_triangle_cycle();
        let expected = {
            let mut c = Cycle {
                name: cyc.name.clone(),
                init_elev: None,
                time: vec![
                    0.0 * uc::S,
                    10.0 * uc::S,
                    20.0 * uc::S,
                    30.0 * uc::S,
                    31.0 * uc::S,
                    32.0 * uc::S,
                    33.0 * uc::S,
                    34.0 * uc::S,
                    35.0 * uc::S,
                ],
                speed: vec![
                    0.0 * uc::MPS,
                    4.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                ],
                dist: vec![],
                grade: vec![],
                elev: vec![],
                pwr_max_chrg: vec![],
                grade_interp: cyc.grade_interp.clone(),
                elev_interp: cyc.elev_interp.clone(),
                temp_amb_air: Default::default(),
                pwr_solar_load: Default::default(),
            };
            c.init().unwrap();
            c
        };
        let actual = extend_cycle_time(&cyc, Some(2.0 * uc::S), Some(0.10 * uc::R));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_passing_info() {
        let c = {
            // travels 300 m
            let mut cyc = Cycle {
                name: String::from("Main Cycle"),
                time: vec![
                    0.0 * uc::S,
                    10.0 * uc::S,
                    20.0 * uc::S,
                    30.0 * uc::S,
                    40.0 * uc::S,
                ],
                speed: vec![
                    0.0 * uc::MPS,
                    10.0 * uc::MPS,
                    10.0 * uc::MPS,
                    10.0 * uc::MPS,
                    0.0 * uc::MPS,
                ],
                grade: vec![],
                dist: vec![],
                elev: vec![],
                init_elev: Some(0.0 * uc::M),
                pwr_max_chrg: vec![],
                pwr_solar_load: vec![],
                temp_amb_air: vec![],
                grade_interp: Default::default(),
                elev_interp: Default::default(),
            };
            cyc.init().unwrap();
            cyc
        };
        let c_lead = {
            // travels 250 m
            let mut cyc = Cycle {
                name: String::from("Lead Vehicle"),
                time: vec![
                    0.0 * uc::S,
                    10.0 * uc::S,
                    20.0 * uc::S,
                    30.0 * uc::S,
                    40.0 * uc::S,
                ],
                speed: vec![
                    0.0 * uc::MPS,
                    10.0 * uc::MPS,
                    10.0 * uc::MPS,
                    5.0 * uc::MPS,
                    0.0 * uc::MPS,
                ],
                grade: vec![],
                dist: vec![],
                elev: vec![],
                init_elev: Some(0.0 * uc::M),
                pwr_max_chrg: vec![],
                pwr_solar_load: vec![],
                temp_amb_air: vec![],
                grade_interp: Default::default(),
                elev_interp: Default::default(),
            };
            cyc.init().unwrap();
            cyc
        };
        let expected = PassingInfo {
            passing_detected: true,
            index: 3,
            num_steps: 3,
            start_distance: 0.0 * uc::M,
            distance: 225.0 * uc::M,
            start_speed: 0.0 * uc::MPS,
            speed: 5.0 * uc::MPS,
            time_step_duration: 10.0 * uc::S,
        };
        let actual = PassingInfo::from(&c, &c_lead, 1, None);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_making_interp() {
        let interp = InterpolatorEnum::new_1d(
            array![0.0, 2.0, 4.0],
            array![0.0, 4.0, 8.0],
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let value = interp.interpolate(&[1.0]).unwrap();
        let expected = 2.0;
        assert_eq!(value, expected);
    }

    #[test]
    fn test_calc_best_rendezvous() {
        let cyc = {
            let mut c = Cycle {
                name: String::from("Trapezoidal Trace"),
                init_elev: None,
                time: vec![
                    0.0 * uc::S,
                    1.0 * uc::S,
                    2.0 * uc::S,
                    3.0 * uc::S,
                    4.0 * uc::S,
                    5.0 * uc::S,
                    6.0 * uc::S,
                    7.0 * uc::S,
                    8.0 * uc::S,
                ],
                speed: vec![
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                    8.0 * uc::MPS,
                    8.0 * uc::MPS,
                    8.0 * uc::MPS,
                    8.0 * uc::MPS,
                    8.0 * uc::MPS,
                    0.0 * uc::MPS,
                    0.0 * uc::MPS,
                ],
                dist: vec![],
                grade: vec![],
                elev: vec![],
                pwr_max_chrg: vec![],
                temp_amb_air: vec![],
                pwr_solar_load: vec![],
                grade_interp: None,
                elev_interp: None,
            };
            c.init().unwrap();
            c
        };
        let i = 2;
        let max_steps = 4;
        let speed_ach = 4.0 * uc::MPS;
        let result = calc_best_rendezvous(i, max_steps, &cyc, speed_ach);
        assert!(result.steps >= 2);
        let expected_distance_m = 4.0 + 8.0 * (result.steps as f64);
        let actual_distance_m = result.end_distance();
        assert_eq!(actual_distance_m, expected_distance_m);
    }
}
