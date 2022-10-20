//! Module containing drive cycle struct and related functions.

extern crate ndarray;

#[cfg(feature = "pyo3")]
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

// local
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::add_pyo3_api;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::utils::*;

#[cfg_attr(feature = "pyo3", pyfunction)]
/// # Arguments
/// - n: Int, number of time-steps away from rendezvous
/// - d0: Num, distance of simulated vehicle, $\frac{m}{s}$
/// - v0: Num, speed of simulated vehicle, $\frac{m}{s}$
/// - dr: Num, distance of rendezvous point, $m$
/// - vr: Num, speed of rendezvous point, $\frac{m}{s}$
/// - dt: Num, step duration, $s$
///
/// # Returns
/// (Tuple 'jerk_m__s3': Num, 'accel_m__s2': Num)
/// - Constant jerk and acceleration for initial time step.
pub fn calc_constant_jerk_trajectory(
    n: usize,
    d0: f64,
    v0: f64,
    dr: f64,
    vr: f64,
    dt: f64,
) -> (f64, f64) {
    assert!(n > 1);
    assert!(dr > d0);
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
    (k, a0)
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate distance (m) after n timesteps
///
/// INPUTS:
/// - n: Int, numer of timesteps away to calculate
/// - d0: Num, initial distance (m)
/// - v0: Num, initial speed (m/s)
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk
/// - dt: Num, duration of a timestep (s)
///
/// NOTE:
/// - this is the distance traveled from start (i.e., n=0) measured at sample point n
/// RETURN: Num, the distance at n timesteps away (m)
pub fn dist_for_constant_jerk(n: usize, d0: f64, v0: f64, a0: f64, k: f64, dt: f64) -> f64 {
    let n = n as f64;
    let term1 = dt
        * ((n * v0)
            + (0.5 * n * (n - 1.0) * a0 * dt)
            + ((1.0 / 6.0) * k * dt * (n - 2.0) * (n - 1.0) * n));
    let term2 = 0.5 * dt * dt * ((n * a0) + (0.5 * n * (n - 1.0) * k * dt));
    d0 + term1 + term2
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate speed (m/s) n timesteps away via a constant-jerk acceleration
///
/// INPUTS:
/// - n: Int, numer of timesteps away to calculate
/// - v0: Num, initial speed (m/s)
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk
/// - dt: Num, duration of a timestep (s)
///
/// NOTE:
/// - this is the speed at sample n
/// - if n == 0, speed is v0
/// - if n == 1, speed is v0 + a0*dt, etc.
///
/// RETURN: Num, the speed n timesteps away (m/s)
pub fn speed_for_constant_jerk(n: usize, v0: f64, a0: f64, k: f64, dt: f64) -> f64 {
    let n = n as f64;
    v0 + (n * a0 * dt) + (0.5 * n * (n - 1.0) * k * dt)
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Calculate the acceleration n timesteps away
///
/// INPUTS:
/// - n: Int, number of times steps away to calculate
/// - a0: Num, initial acceleration (m/s2)
/// - k: Num, constant jerk (m/s3)
/// - dt: Num, time-step duration in seconds
///
/// NOTE:
/// - this is the constant acceleration over the time-step from sample n to sample n+1
///
/// RETURN: Num, the acceleration n timesteps away (m/s2)
pub fn accel_for_constant_jerk(n: usize, a0: f64, k: f64, dt: f64) -> f64 {
    let n = n as f64;
    a0 + (n * k * dt)
}

/// Apply `accel_for_constant_jerk` to full
pub fn accel_array_for_constant_jerk(nmax: usize, a0: f64, k: f64, dt: f64) -> Array1<f64> {
    let mut accels: Vec<f64> = Vec::new();
    for n in 0..nmax {
        accels.push(accel_for_constant_jerk(n, a0, k, dt));
    }
    Array1::from_vec(accels)
}

/// Calculate the average speed per each step in m/s
pub fn average_step_speeds(cyc: &RustCycle) -> Array1<f64> {
    let mut result: Vec<f64> = vec![0.0];
    for i in 1..cyc.mps.len() {
        result.push(0.5 * (cyc.mps[i] + cyc.mps[i - 1]));
    }
    Array1::from_vec(result)
}

/// Calculate the average step speed at step i in m/s
/// (i.e., from sample point i-1 to i)
pub fn average_step_speed_at(cyc: &RustCycle, i: usize) -> f64 {
    0.5 * (cyc.mps[i] + cyc.mps[i - 1])
}

/// Sum of the distance traveled over each step using
/// trapezoidal integration
pub fn trapz_step_distances(cyc: &RustCycle) -> Array1<f64> {
    average_step_speeds(cyc) * cyc.dt_s()
}

/// The distance traveled from start at the beginning of step i
/// (i.e., distance traveled up to sample point i-1)
/// Distance is in meters.
pub fn trapz_step_start_distance(cyc: &RustCycle, i: usize) -> f64 {
    trapz_step_distances(cyc).slice(s![0..i]).sum()
}

/// The distance traveled during step i in meters
/// (i.e., from sample point i-1 to i)
pub fn trapz_distance_for_step(cyc: &RustCycle, i: usize) -> f64 {
    average_step_speed_at(cyc, i) * cyc.dt_s_at_i(i)
}

/// Calculate the distance from step i_start to the start of step i_end
/// (i.e., distance from sample point i_start-1 to i_end-1)
pub fn trapz_distance_over_range(cyc: &RustCycle, i_start: usize, i_end: usize) -> f64 {
    trapz_step_distances(cyc).slice(s![i_start..i_end]).sum()
}

/// Calculate the time in a cycle spent moving
/// - stopped_speed_m_per_s: the speed above which we are considered to be moving
/// RETURN: the time spent moving in seconds
pub fn time_spent_moving(cyc: &RustCycle, stopped_speed_m_per_s: Option<f64>) -> f64 {
    let mut t_move_s = 0.0;
    let stopped_speed_m_per_s = stopped_speed_m_per_s.unwrap_or(0.0);
    for idx in 1..cyc.time_s.len() {
        let dt = cyc.time_s[idx] - cyc.time_s[idx - 1];
        let vavg = (cyc.mps[idx] + cyc.mps[idx - 1]) / 2.0;
        if vavg > stopped_speed_m_per_s {
            t_move_s += dt;
        }
    }
    t_move_s
}

/// Split a cycle into an array of microtrips with one microtrip being a start
/// to subsequent stop plus any idle (stopped time).
/// Arguments:
/// ----------
/// cycle: drive cycle converted to dictionary by cycle.get_cyc_dict()
/// stop_speed_m__s: speed at which vehicle is considered stopped for trip
///     separation
/// keep_name: (optional) bool, if True and cycle contains "name", adds
///     that name to all microtrips
pub fn to_microtrips(cycle: &RustCycle, stop_speed_m_per_s: Option<f64>) -> Vec<RustCycle> {
    let stop_speed_m_per_s = stop_speed_m_per_s.unwrap_or(1e-6);
    let mut microtrips: Vec<RustCycle> = Vec::new();
    let ts = cycle.time_s.to_vec();
    let vs = cycle.mps.to_vec();
    let gs = cycle.grade.to_vec();
    let rs = cycle.road_type.to_vec();
    let mut mt_ts: Vec<f64> = Vec::new();
    let mut mt_vs: Vec<f64> = Vec::new();
    let mut mt_gs: Vec<f64> = Vec::new();
    let mut mt_rs: Vec<f64> = Vec::new();
    let mut moving = false;
    for idx in 0..ts.len() {
        let t = ts[idx];
        let v = vs[idx];
        let g = gs[idx];
        let r = rs[idx];
        if v > stop_speed_m_per_s && !moving && mt_ts.len() > 1 {
            let last_idx = mt_ts.len() - 1;
            let last_t = mt_ts[last_idx];
            let last_v = mt_vs[last_idx];
            let last_g = mt_gs[last_idx];
            let last_r = mt_rs[last_idx];
            mt_ts = mt_ts.iter().map(|t| -> f64 { t - mt_ts[0] }).collect();
            microtrips.push(RustCycle {
                time_s: Array::from_vec(mt_ts.clone()),
                mps: Array::from_vec(mt_vs.clone()),
                grade: Array::from_vec(mt_gs.clone()),
                road_type: Array::from_vec(mt_rs.clone()),
                name: cycle.name.clone(),
                orphaned: false,
            });
            mt_ts = vec![last_t];
            mt_vs = vec![last_v];
            mt_gs = vec![last_g];
            mt_rs = vec![last_r];
        }
        mt_ts.push(t);
        mt_vs.push(v);
        mt_gs.push(g);
        mt_rs.push(r);
        moving = v > stop_speed_m_per_s;
    }
    if !mt_ts.is_empty() {
        mt_ts = mt_ts.iter().map(|t| -> f64 { t - mt_ts[0] }).collect();
        microtrips.push(RustCycle {
            time_s: Array::from_vec(mt_ts),
            mps: Array::from_vec(mt_vs),
            grade: Array::from_vec(mt_gs),
            road_type: Array::from_vec(mt_rs),
            name: cycle.name.clone(),
            orphaned: false,
        });
    }
    microtrips
}

/// Create distance and target speeds by microtrip
/// This helper function splits a cycle up into microtrips and returns a list of 2-tuples of:
/// (distance from start in meters, target speed in meters/second)
/// - cyc: the cycle to operate on
/// - blend_factor: float, from 0 to 1
///     if 0, use average speed of the microtrip
///     if 1, use average speed while moving (i.e., no stopped time)
///     else something in between
/// - min_target_speed_mps: float, the minimum target speed allowed (m/s)
/// RETURN: list of 2-tuple of (float, float) representing the distance of start of
///     each microtrip and target speed for that microtrip
/// NOTE: target speed per microtrip is not allowed to be below min_target_speed_mps
pub fn create_dist_and_target_speeds_by_microtrip(
    cyc: &RustCycle,
    blend_factor: f64,
    min_target_speed_mps: f64,
) -> Vec<(f64, f64)> {
    let blend_factor = if blend_factor < 0.0 {
        0.0
    } else if blend_factor > 1.0 {
        1.0
    } else {
        blend_factor
    };
    let mut dist_and_tgt_speeds: Vec<(f64, f64)> = Vec::new();
    // Split cycle into microtrips
    let microtrips = to_microtrips(cyc, None);
    let mut dist_at_start_of_microtrip_m = 0.0;
    for mt_cyc in microtrips {
        let mt_dist_m = mt_cyc.dist_m().sum();
        let mt_time_s = mt_cyc.time_s[mt_cyc.time_s.len() - 1] - mt_cyc.time_s[0];
        let mt_moving_time_s = time_spent_moving(&mt_cyc, None);
        let mt_avg_spd_m_per_s = if mt_time_s > 0.0 {
            mt_dist_m / mt_time_s
        } else {
            0.0
        };
        let mt_moving_avg_spd_m_per_s = if mt_moving_time_s > 0.0 {
            mt_dist_m / mt_moving_time_s
        } else {
            0.0
        };
        let mt_target_spd_m_per_s =
            (blend_factor * (mt_moving_avg_spd_m_per_s - mt_avg_spd_m_per_s) + mt_avg_spd_m_per_s)
                .min(mt_moving_avg_spd_m_per_s)
                .max(mt_avg_spd_m_per_s);
        if mt_dist_m > 0.0 {
            dist_and_tgt_speeds.push((
                dist_at_start_of_microtrip_m,
                mt_target_spd_m_per_s.max(min_target_speed_mps),
            ));
            dist_at_start_of_microtrip_m += mt_dist_m;
        }
    }
    dist_and_tgt_speeds
}

/// - cyc: fastsim.cycle.Cycle
/// - absolute_time_s: float, the seconds to extend
/// - time_fraction: float, the fraction of the original cycle time to add on
/// - use_rust: bool, if True, return a RustCycle instance, else a normal Python Cycle
/// RETURNS: fastsim.cycle.Cycle (or fastsimrust.RustCycle), the new cycle with stopped time appended
/// NOTE: additional time is rounded to the nearest second
pub fn extend_cycle(
    cyc: &RustCycle,
    absolute_time_s: Option<f64>, // =0.0,
    time_fraction: Option<f64>,   // =0.0,
) -> RustCycle {
    let absolute_time_s = absolute_time_s.unwrap_or(0.0);
    let time_fraction = time_fraction.unwrap_or(0.0);
    let mut ts = cyc.time_s.to_vec();
    let mut vs = cyc.mps.to_vec();
    let mut gs = cyc.grade.to_vec();
    let mut rs = cyc.road_type.to_vec();
    let extra_time_s = (absolute_time_s + (time_fraction * ts[ts.len() - 1])).round() as i32;
    if extra_time_s == 0 {
        return cyc.clone();
    }
    let dt = 1;
    let t_end = ts[ts.len() - 1];
    let mut idx = 1;
    while dt * idx <= extra_time_s {
        let dt_extra = (dt * idx) as f64;
        ts.push(t_end + dt_extra);
        vs.push(0.0);
        gs.push(0.0);
        rs.push(0.0);
        idx += 1;
    }
    RustCycle {
        time_s: Array::from_vec(ts),
        mps: Array::from_vec(vs),
        grade: Array::from_vec(gs),
        road_type: Array::from_vec(rs),
        name: cyc.name.clone(),
        orphaned: false,
    }
}

#[cfg(feature = "pyo3")]
#[allow(unused)] // not sure what this is doing, may get used in proc macro???
pub fn register(_py: Python<'_>, m: &PyModule) -> Result<(), anyhow::Error> {
    m.add_function(wrap_pyfunction!(calc_constant_jerk_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(accel_for_constant_jerk, m)?)?;
    m.add_function(wrap_pyfunction!(speed_for_constant_jerk, m)?)?;
    m.add_function(wrap_pyfunction!(dist_for_constant_jerk, m)?)?;
    Ok(())
}

#[derive(Default, PartialEq, Clone, Debug, Deserialize, Serialize)]
pub struct RustCycleElement {
    /// time [s]
    #[serde(alias = "cycSecs")]
    pub time_s: f64,
    /// speed [m/s]
    #[serde(alias = "cycMps")]
    pub mps: f64,
    /// grade [rise/run]
    #[serde(alias = "cycGrade")]
    pub grade: f64,
    /// max possible charge rate from roadway
    #[serde(alias = "cycRoadType")]
    pub road_type: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[add_pyo3_api(
    #[new]
    pub fn __new__(
        time_s: Vec<f64>,
        mps: Vec<f64>,
        grade: Vec<f64>,
        road_type: Vec<f64>,
        name: String,
    ) -> Self {
        let time_s = Array::from_vec(time_s);
        let mps = Array::from_vec(mps);
        let grade = Array::from_vec(grade);
        let road_type = Array::from_vec(road_type);
        Self {
            time_s,
            mps,
            grade,
            road_type,
            name,
            orphaned: false,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn __getnewargs__(&self) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, &str)> {
        Ok((self.time_s.to_vec(), self.mps.to_vec(), self.grade.to_vec(), self.road_type.to_vec(), &self.name))
    }

    #[classmethod]
    #[pyo3(name = "from_csv_file")]
    pub fn from_csv_file_py(_cls: &PyType, pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    pub fn to_rust(&self) -> anyhow::Result<Self> {
        Ok(self.clone())
    }

    /// Return a HashMap representing the cycle
    pub fn get_cyc_dict(&self) -> anyhow::Result<HashMap<String, Vec<f64>>> {
        let dict: HashMap<String, Vec<f64>> = HashMap::from([
            ("time_s".to_string(), self.time_s.to_vec()),
            ("mps".to_string(), self.mps.to_vec()),
            ("grade".to_string(), self.grade.to_vec()),
            ("road_type".to_string(), self.road_type.to_vec()),
        ]);
        Ok(dict)
    }

    #[pyo3(name = "modify_by_const_jerk_trajectory")]
    pub fn modify_by_const_jerk_trajectory_py(
        &mut self,
        idx: usize,
        n: usize,
        jerk_m_per_s3: f64,
        accel0_m_per_s2: f64,
    ) -> PyResult<f64> {
        Ok(self.modify_by_const_jerk_trajectory(idx, n, jerk_m_per_s3, accel0_m_per_s2))
    }

    #[pyo3(name = "modify_with_braking_trajectory")]
    pub fn modify_with_braking_trajectory_py(
        &mut self,
        brake_accel_m_per_s2: f64,
        idx: usize,
        dts_m: Option<f64>
    ) -> PyResult<(f64, usize)> {
        Ok(self.modify_with_braking_trajectory(brake_accel_m_per_s2, idx, dts_m))
    }

    #[pyo3(name = "calc_distance_to_next_stop_from")]
    pub fn calc_distance_to_next_stop_from_py(&self, distance_m: f64) -> PyResult<f64> {
        Ok(self.calc_distance_to_next_stop_from(distance_m))
    }

    #[pyo3(name = "average_grade_over_range")]
    pub fn average_grade_over_range_py(
        &self,
        distance_start_m: f64,
        delta_distance_m: f64,
    ) -> PyResult<f64> {
        Ok(self.average_grade_over_range(distance_start_m, delta_distance_m))
    }

    #[pyo3(name = "dt_s_at_i")]
    pub fn dt_s_at_i_py(&self, i: usize) -> PyResult<f64> {
        if i == 0 {
            Ok(0.0)
        } else {
            Ok(self.dt_s_at_i(i))
        }
    }

    #[getter]
    pub fn get_mph(&self) -> PyResult<Vec<f64>> {
        Ok((&self.mps * crate::params::MPH_PER_MPS).to_vec())
    }
    #[setter]
    pub fn set_mph(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mps = Array::from_vec(new_value) / MPH_PER_MPS;
        Ok(())
    }
    #[getter]
    /// array of time steps
    pub fn get_dt_s(&self) -> PyResult<Vec<f64>> {
        Ok(self.dt_s().to_vec())
    }
    #[getter]
    /// cycle length
    pub fn get_len(&self) -> PyResult<usize> {
        Ok(self.len())
    }
    #[getter]
    /// distance for each time step based on final speed
    pub fn get_dist_m(&self) -> PyResult<Vec<f64>> {
        Ok(self.dist_m().to_vec())
    }
    #[getter]
    pub fn get_delta_elev_m(&self) -> PyResult<Vec<f64>> {
        Ok(self.delta_elev_m().to_vec())
    }
)]
/// Struct for containing:
/// * time_s, cycle time, $s$
/// * mps, vehicle speed, $\frac{m}{s}$
/// * grade, road grade/slope, $\frac{rise}{run}$
/// * road_type, $kW$
/// * legacy, will likely change to road charging capacity
///    * Another sublist.
pub struct RustCycle {
    /// array of time [s]
    #[serde(alias = "cycSecs")]
    pub time_s: Array1<f64>,
    /// array of speed [m/s]
    #[serde(alias = "cycMps")]
    pub mps: Array1<f64>,
    /// array of grade [rise/run]
    #[serde(alias = "cycGrade")]
    pub grade: Array1<f64>,
    /// array of max possible charge rate from roadway
    #[serde(alias = "cycRoadType")]
    pub road_type: Array1<f64>,
    pub name: String,
    #[serde(skip)]
    pub orphaned: bool,
}

/// pure Rust methods that need to be separate due to pymethods incompatibility
impl RustCycle {
    pub fn new(
        time_s: Vec<f64>,
        mps: Vec<f64>,
        grade: Vec<f64>,
        road_type: Vec<f64>,
        name: String,
    ) -> Self {
        let time_s = Array::from_vec(time_s);
        let mps = Array::from_vec(mps);
        let grade = Array::from_vec(grade);
        let road_type = Array::from_vec(road_type);
        Self {
            time_s,
            mps,
            grade,
            road_type,
            name,
            orphaned: false,
        }
    }

    pub fn push(&mut self, cyc_elem: RustCycleElement) {
        self.time_s
            .append(Axis(0), array![cyc_elem.time_s].view())
            .unwrap();
        self.mps
            .append(Axis(0), array![cyc_elem.mps].view())
            .unwrap();
        self.grade
            .append(Axis(0), array![cyc_elem.grade].view())
            .unwrap();
        self.road_type
            .append(Axis(0), array![cyc_elem.road_type].view())
            .unwrap();
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.time_s.len()
    }

    pub fn test_cyc() -> Self {
        let time_s = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let speed_mps = Array1::<f64>::range(0.0, 10.0, 1.0).to_vec();
        let grade = Array::zeros(10).to_vec();
        let road_type = Array::zeros(10).to_vec();
        let name = String::from("test");
        Self::new(time_s, speed_mps, grade, road_type, name)
    }

    /// Returns the average grade over the given range of distances
    /// - distance_start_m: non-negative-number, the distance at start of evaluation area (m)
    /// - delta_distance_m: non-negative-number, the distance traveled from distance_start_m (m)
    /// RETURN: number, the average grade (rise over run) over the given distance range
    /// Note: grade is assumed to be constant from just after the previous sample point
    /// until the current sample point. That is, grade[i] applies over the range of
    /// distances, d, from (d[i - 1], d[i]]
    pub fn average_grade_over_range(&self, distance_start_m: f64, delta_distance_m: f64) -> f64 {
        let tol = 1e-6;
        if ndarrallzeros(&self.grade) {
            // short-circuit for no-grade case
            return 0.0;
        }
        let delta_dists = trapz_step_distances(self);
        let distances_m = ndarrcumsum(&delta_dists);
        if delta_distance_m <= tol {
            if distance_start_m <= distances_m[0] {
                return self.grade[0];
            }
            if distance_start_m >= distances_m[distances_m.len() - 1] {
                return self.grade[self.grade.len() - 1];
            }
            let mut gr = self.grade[0];
            for idx in 0..(distances_m.len() - 1) {
                let d = distances_m[idx];
                let d_next = distances_m[idx + 1];
                let g = self.grade[idx + 1];
                if distance_start_m > d && distance_start_m <= d_next {
                    gr = g;
                    break;
                }
            }
            return gr;
        }
        // NOTE: we use the following instead of delta_elev_m in order to use
        // more precise trapezoidal distance and elevation at sample points.
        // This also uses the fully accurate trig functions in case we have large slope angles.
        let elevations_m =
            ndarrcumsum(&(self.grade.mapv(|g| g.atan().cos()) * &delta_dists * self.grade.clone()));
        let e0 = interpolate(&distance_start_m, &distances_m, &elevations_m, false);
        let e1 = interpolate(
            &(distance_start_m + delta_distance_m),
            &distances_m,
            &elevations_m,
            false,
        );
        ((e1 - e0) / delta_distance_m).asin().tan()
    }

    /// Calculate the distance to next stop from `distance_m`
    /// - distance_m: non-negative-number, the current distance from start (m)
    /// RETURN: returns the distance to the next stop from distance_m
    /// NOTE: distance may be negative if we're beyond the last stop
    pub fn calc_distance_to_next_stop_from(&self, distance_m: f64) -> f64 {
        let tol: f64 = 1e-6;
        let mut d: f64 = 0.0;
        for (&dd, &v) in trapz_step_distances(self).iter().zip(self.mps.iter()) {
            d += dd;
            if (v < tol) && (d > (distance_m + tol)) {
                return d - distance_m;
            }
        }
        d - distance_m
    }

    /// Modifies the cycle using the given constant-jerk trajectory parameters
    /// - idx: non-negative integer, the point in the cycle to initiate
    ///   modification (note: THIS point is modified since trajectory should be
    ///   calculated from idx-1)
    /// - n: non-negative integer, the number of steps ahead
    /// - jerk_m__s3: number, the "Jerk" associated with the trajectory (m/s3)
    /// - accel0_m__s2: number, the initial acceleration (m/s2)
    /// NOTE:
    /// - modifies cyc in place to hit any critical rendezvous_points by a trajectory adjustment
    /// - CAUTION: NOT ROBUST AGAINST VARIABLE DURATION TIME-STEPS
    /// RETURN: Number, final modified speed (m/s)
    pub fn modify_by_const_jerk_trajectory(
        &mut self,
        i: usize,
        n: usize,
        jerk_m_per_s3: f64,
        accel0_m_per_s2: f64,
    ) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let num_samples = self.mps.len();
        if i >= num_samples {
            if num_samples > 0 {
                return self.mps[num_samples - 1];
            }
            return 0.0;
        }
        let v0 = self.mps[i - 1];
        let dt = self.dt_s_at_i(i);
        let mut v = v0;
        for ni in 1..(n + 1) {
            let idx_to_set = (i - 1) + ni;
            if idx_to_set >= num_samples {
                break;
            }
            v = speed_for_constant_jerk(ni, v0, accel0_m_per_s2, jerk_m_per_s3, dt);
            self.mps[idx_to_set] = max(v, 0.0);
        }
        v
    }

    /// Add a braking trajectory that would cover the same distance as the given constant brake deceleration
    /// - brake_accel_m__s2: negative number, the braking acceleration (m/s2)
    /// - idx: non-negative integer, the index where to initiate the stop trajectory, start of the step (i in FASTSim)
    /// - dts_m: None | float: if given, this is the desired distance-to-stop in meters. If not given, it is
    ///     calculated based on braking deceleration.
    /// RETURN: (non-negative-number, positive-integer)
    /// - the final speed of the modified trajectory (m/s)
    /// - the number of time-steps required to complete the braking maneuver
    /// NOTE:
    /// - modifies the cycle in place for the braking trajectory
    pub fn modify_with_braking_trajectory(
        &mut self,
        brake_accel_m_per_s2: f64,
        i: usize,
        dts_m: Option<f64>,
    ) -> (f64, usize) {
        assert!(brake_accel_m_per_s2 < 0.0);
        if i >= self.time_s.len() {
            return (self.mps[self.mps.len() - 1], 0);
        }
        let v0 = self.mps[i - 1];
        let dt = self.dt_s_at_i(i);
        // distance-to-stop (m)
        let dts_m = match dts_m {
            Some(value) => {
                if value > 0.0 {
                    value
                } else {
                    -0.5 * v0 * v0 / brake_accel_m_per_s2
                }
            }
            None => -0.5 * v0 * v0 / brake_accel_m_per_s2,
        };
        if dts_m <= 0.0 {
            return (v0, 0);
        }
        // time-to-stop (s)
        let tts_s = -v0 / brake_accel_m_per_s2;
        // number of steps to take
        let n: usize = (tts_s / dt).round() as usize;
        let n: usize = if n < 2 { 2 } else { n }; // need at least 2 steps
        let (jerk_m_per_s3, accel_m_per_s2) =
            calc_constant_jerk_trajectory(n, 0.0, v0, dts_m, 0.0, dt);
        (
            self.modify_by_const_jerk_trajectory(i, n, jerk_m_per_s3, accel_m_per_s2),
            n,
        )
    }

    /// rust-internal time steps
    pub fn dt_s(&self) -> Array1<f64> {
        diff(&self.time_s)
    }

    /// rust-internal time steps at i
    pub fn dt_s_at_i(&self, i: usize) -> f64 {
        self.time_s[i] - self.time_s[i - 1]
    }

    /// distance covered in each time step
    pub fn dist_m(&self) -> Array1<f64> {
        &self.mps * self.dt_s()
    }

    /// get mph from self.mps
    pub fn mph_at_i(&self, i: usize) -> f64 {
        self.mps[i] * MPH_PER_MPS
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty cycle to be populated
        let mut cyc = Self::default();

        // unrwap is ok because if statement checks existence
        let file = File::open(&pathbuf).unwrap();
        let name = String::from(pathbuf.file_stem().unwrap().to_str().unwrap());
        cyc.name = name;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            // TODO: make this more elegant than unwrap
            let cyc_elem: RustCycleElement = result?;
            cyc.push(cyc_elem);
        }

        Ok(cyc)
    }

    /// elevation change w.r.t. to initial
    pub fn delta_elev_m(&self) -> Array1<f64> {
        ndarrcumsum(&(self.dist_m() * self.grade.clone()))
    }

    pub fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            "csv" => Ok(Self::from_csv_file(filename)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

pub struct PassingInfo {
    /// True if first cycle passes the second
    pub has_collision: bool,
    /// the index where first cycle passes the second
    pub idx: usize,
    /// the number of time-steps until idx from i
    pub num_steps: usize,
    /// the starting distance of the first cycle at i
    pub start_distance_m: f64,
    /// the distance (m) traveled of the second cycle when first passes
    pub distance_m: f64,
    /// the starting speed (m/s) of the first cycle at i
    pub start_speed_m_per_s: f64,
    /// the speed (m/s) of the second cycle when first passes
    pub speed_m_per_s: f64,
    /// the time step duration throught the passing investigation
    pub time_step_duration_s: f64,
}

/// Reports back information of the first point where cyc passes cyc0, starting at
/// step i until the next stop of cyc.
/// - cyc: fastsim.Cycle, the proposed cycle of the vehicle under simulation
/// - cyc0: fastsim.Cycle, the reference/lead vehicle/shadow cycle to compare with
/// - i: int, the time-step index to consider
/// - dist_tol_m: float, the distance tolerance away from lead vehicle to be seen as
///     "deviated" from the reference/shadow trace (m)
/// RETURNS: PassingInfo
pub fn detect_passing(
    cyc: &RustCycle,
    cyc0: &RustCycle,
    i: usize,
    dist_tol_m: Option<f64>,
) -> PassingInfo {
    if i >= cyc.time_s.len() {
        return PassingInfo {
            has_collision: false,
            idx: 0,
            num_steps: 0,
            start_distance_m: 0.0,
            distance_m: 0.0,
            start_speed_m_per_s: 0.0,
            speed_m_per_s: 0.0,
            time_step_duration_s: 1.0,
        };
    }
    let zero_speed_tol_m_per_s = 1e-6;
    let dist_tol_m = dist_tol_m.unwrap_or(0.1);
    let mut v0: f64 = cyc.mps[i - 1];
    let d0: f64 = trapz_step_start_distance(cyc, i);
    let mut v0_lv: f64 = cyc0.mps[i - 1];
    let d0_lv: f64 = trapz_step_start_distance(cyc0, i);
    let mut d = d0;
    let mut d_lv = d0_lv;
    let mut rendezvous_idx: Option<usize> = None;
    let mut rendezvous_num_steps: usize = 0;
    let mut rendezvous_distance_m: f64 = 0.0;
    let mut rendezvous_speed_m_per_s: f64 = 0.0;
    for di in 0..(cyc.mps.len() - i) {
        let idx = i + di;
        let v = cyc.mps[idx];
        let v_lv = cyc0.mps[idx];
        let vavg = (v + v0) * 0.5;
        let vavg_lv = (v_lv + v0_lv) * 0.5;
        let dd = vavg * cyc.dt_s_at_i(idx);
        let dd_lv = vavg_lv * cyc0.dt_s_at_i(idx);
        d += dd;
        d_lv += dd_lv;
        let dtlv = d_lv - d;
        v0 = v;
        v0_lv = v_lv;
        if di > 0 && dtlv < -dist_tol_m {
            rendezvous_idx = Some(idx);
            rendezvous_num_steps = di + 1;
            rendezvous_distance_m = d_lv;
            rendezvous_speed_m_per_s = v_lv;
            break;
        }
        if v <= zero_speed_tol_m_per_s {
            break;
        }
    }
    PassingInfo {
        has_collision: rendezvous_idx.is_some(),
        idx: rendezvous_idx.unwrap_or(0),
        num_steps: rendezvous_num_steps,
        start_distance_m: d0,
        distance_m: rendezvous_distance_m,
        start_speed_m_per_s: cyc.mps[i - 1],
        speed_m_per_s: rendezvous_speed_m_per_s,
        time_step_duration_s: cyc.dt_s_at_i(i),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist() {
        let cyc = RustCycle::test_cyc();
        assert_eq!(cyc.dist_m().sum(), 45.0);
    }

    #[test]
    fn test_average_speeds_and_distances() {
        let time_s = vec![0.0, 10.0, 30.0, 34.0, 40.0];
        let speed_mps = vec![0.0, 10.0, 10.0, 0.0, 0.0];
        let grade = Array::zeros(5).to_vec();
        let road_type = Array::zeros(5).to_vec();
        let name = String::from("test");
        let cyc = RustCycle::new(time_s, speed_mps, grade, road_type, name);
        let avg_mps = average_step_speeds(&cyc);
        let expected_avg_mps = Array::from_vec(vec![0.0, 5.0, 10.0, 5.0, 0.0]);
        assert_eq!(expected_avg_mps.len(), avg_mps.len());
        for (expected, actual) in expected_avg_mps.iter().zip(avg_mps.iter()) {
            assert_eq!(expected, actual);
        }
        let dist_m = trapz_step_distances(&cyc);
        let expected_dist_m = Array::from_vec(vec![0.0, 50.0, 200.0, 20.0, 0.0]);
        assert_eq!(expected_dist_m.len(), dist_m.len());
        for (expected, actual) in expected_dist_m.iter().zip(dist_m.iter()) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_loading_a_cycle_from_the_filesystem() {
        let pathstr = String::from("../../fastsim/resources/cycles/udds.csv");
        let expected_udds_length: usize = 1370;
        let cyc = RustCycle::from_csv_file(&pathstr).unwrap();
        assert_eq!(cyc.name, String::from("udds"));
        let num_entries = cyc.time_s.len();
        assert!(num_entries > 0);
        assert_eq!(num_entries, cyc.time_s.len());
        assert_eq!(num_entries, cyc.mps.len());
        assert_eq!(num_entries, cyc.grade.len());
        assert_eq!(num_entries, cyc.road_type.len());
        assert_eq!(num_entries, expected_udds_length);
    }
}
