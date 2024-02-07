//! Module containing drive cycle struct and related functions.

use std::collections::HashMap;

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
) -> anyhow::Result<(f64, f64)> {
    ensure!(n > 1);
    ensure!(dr > d0);
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
    Ok((k, a0))
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
    let mut result: Vec<f64> = Vec::with_capacity(cyc.len());
    result.push(0.0);
    for i in 1..cyc.len() {
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

pub fn trapz_step_distances_primitive(time_s: &Array1<f64>, mps: &Array1<f64>) -> Array1<f64> {
    let mut delta_dists_m: Vec<f64> = Vec::with_capacity(time_s.len());
    delta_dists_m.push(0.0);
    for i in 1..time_s.len() {
        delta_dists_m.push((time_s[i] - time_s[i - 1]) * 0.5 * (mps[i] + mps[i - 1]));
    }
    Array1::from_vec(delta_dists_m)
}

/// The distance traveled from start at the beginning of step i
/// (i.e., distance traveled up to sample point i-1)
/// Distance is in meters.
pub fn trapz_step_start_distance(cyc: &RustCycle, i: usize) -> f64 {
    let mut dist_m: f64 = 0.0;
    for i in 1..i {
        dist_m += (cyc.time_s[i] - cyc.time_s[i - 1]) * 0.5 * (cyc.mps[i] + cyc.mps[i - 1]);
    }
    dist_m
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
    for idx in 1..cyc.len() {
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
/// cycle: drive cycle
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
    let mut mt_ts = Vec::new();
    let mut mt_vs = Vec::new();
    let mut mt_gs = Vec::new();
    let mut mt_rs = Vec::new();
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
                time_s: Array::from_vec(mt_ts),
                mps: Array::from_vec(mt_vs),
                grade: Array::from_vec(mt_gs),
                road_type: Array::from_vec(mt_rs),
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
        let mt_time_s = mt_cyc.time_s.last().unwrap() - mt_cyc.time_s.first().unwrap();
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
    let extra_time_s = (absolute_time_s + (time_fraction * ts.last().unwrap())).round() as i32;
    if extra_time_s == 0 {
        return cyc.clone();
    }
    let dt = 1;
    let t_end = *ts.last().unwrap();
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
#[allow(unused)]
pub fn register(_py: Python<'_>, m: &PyModule) -> anyhow::Result<()> {
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
    pub grade: Option<f64>,
    /// max possible charge rate from roadway
    #[serde(alias = "cycRoadType")]
    pub road_type: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[add_pyo3_api(
    #[new]
    pub fn __new__(
        cyc: &RustCycle,
    ) -> Self {
        Self::new(cyc)
    }
)]
pub struct RustCycleCache {
    pub grade_all_zero: bool,
    pub trapz_step_distances_m: Array1<f64>,
    pub trapz_distances_m: Array1<f64>,
    pub trapz_elevations_m: Array1<f64>,
    pub stops: Array1<bool>,
    interp_ds: Array1<f64>,
    interp_is: Array1<f64>,
    interp_hs: Array1<f64>,
    grades: Array1<f64>,
}

impl SerdeAPI for RustCycleCache {}

impl RustCycleCache {
    pub fn new(cyc: &RustCycle) -> Self {
        let tol = 1e-6;
        let num_items = cyc.len();
        let grade_all_zero = cyc.grade.iter().all(|g| *g == 0.0);
        let trapz_step_distances_m = trapz_step_distances(cyc);
        let trapz_distances_m = ndarrcumsum(&trapz_step_distances_m);
        let trapz_elevations_m = if grade_all_zero {
            Array::zeros(num_items)
        } else {
            let xs = Array::from_iter(
                cyc.grade
                    .iter()
                    .zip(&trapz_step_distances_m)
                    .map(|(g, dd)| g.atan().cos() * dd * g),
            );
            ndarrcumsum(&xs)
        };
        let stops = Array::from_iter(cyc.mps.iter().map(|v| v <= &tol));
        let mut interp_ds: Vec<f64> = Vec::with_capacity(num_items);
        let mut interp_is: Vec<f64> = Vec::with_capacity(num_items);
        let mut interp_hs: Vec<f64> = Vec::with_capacity(num_items);
        for idx in 0..num_items {
            let d = trapz_distances_m[idx];
            if interp_ds.is_empty() || d > *interp_ds.last().unwrap() {
                interp_ds.push(d);
                interp_is.push(idx as f64);
                interp_hs.push(trapz_elevations_m[idx]);
            }
        }
        let interp_ds = Array::from_vec(interp_ds);
        let interp_is = Array::from_vec(interp_is);
        let interp_hs = Array::from_vec(interp_hs);
        Self {
            grade_all_zero,
            trapz_step_distances_m,
            trapz_distances_m,
            trapz_elevations_m,
            stops,
            interp_ds,
            interp_is,
            interp_hs,
            grades: cyc.grade.clone(),
        }
    }

    /// Interpolate the single-point grade at the given distance.
    /// Assumes that the grade at i applies from sample point (i-1, i]
    pub fn interp_grade(&self, dist_m: f64) -> f64 {
        if self.grade_all_zero {
            0.0
        } else if dist_m <= self.interp_ds[0] {
            self.grades[0]
        } else if dist_m > *self.interp_ds.last().unwrap() {
            *self.grades.last().unwrap()
        } else {
            let raw_idx = interpolate(&dist_m, &self.interp_ds, &self.interp_is, false);
            let idx = raw_idx.ceil() as usize;
            self.grades[idx]
        }
    }

    /// Interpolate the elevation at the given distance
    pub fn interp_elevation(&self, dist_m: f64) -> f64 {
        if self.grade_all_zero {
            0.0
        } else {
            interpolate(&dist_m, &self.interp_ds, &self.interp_hs, false)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[add_pyo3_api(
    pub fn __len__(&self) -> usize {
        self.len()
    }

    #[allow(clippy::type_complexity)]
    pub fn __getnewargs__(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, &str) {
        (self.time_s.to_vec(), self.mps.to_vec(), self.grade.to_vec(), self.road_type.to_vec(), &self.name)
    }

    #[staticmethod]
    #[pyo3(name = "from_csv")]
    pub fn from_csv_py(filepath: &PyAny) -> anyhow::Result<Self> {
        Self::from_csv_file(PathBuf::extract(filepath)?)
    }

    pub fn to_rust(&self) -> Self {
        self.clone()
    }

    #[staticmethod]
    pub fn from_dict(dict: &PyDict) -> anyhow::Result<Self> {
        let time_s = Array::from_vec(PyAny::get_item(dict, "time_s")?.extract()?);
        let cyc_len = time_s.len();
        let mut cyc = Self {
            time_s,
            mps: Array::from_vec(PyAny::get_item(dict, "mps")?.extract()?),
            grade: if let Ok(value) = PyAny::get_item(dict, "grade") {
                Array::from_vec(value.extract()?)
            } else {
                Array::default(cyc_len)
            },
            road_type: if let Ok(value) = PyAny::get_item(dict, "road_type") {
                Array::from_vec(value.extract()?)
            } else {
                Array::default(cyc_len)
            },
            name: PyAny::get_item(dict, "name").and_then(String::extract).unwrap_or_default(),
            orphaned: false,
        };
        cyc.init()?;
        Ok(cyc)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> anyhow::Result<&'py PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("time_s", self.time_s.to_vec())?;
        dict.set_item("mps", self.mps.to_vec())?;
        dict.set_item("grade", self.grade.to_vec())?;
        dict.set_item("road_type", self.road_type.to_vec())?;
        dict.set_item("name", self.name.clone())?;
        Ok(dict)
    }

    #[pyo3(name = "to_csv")]
    pub fn to_csv_py(&self) -> PyResult<String> {
        self.to_csv().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
    }

    #[pyo3(name = "modify_by_const_jerk_trajectory")]
    pub fn modify_by_const_jerk_trajectory_py(
        &mut self,
        idx: usize,
        n: usize,
        jerk_m_per_s3: f64,
        accel0_m_per_s2: f64,
    ) -> f64 {
        self.modify_by_const_jerk_trajectory(idx, n, jerk_m_per_s3, accel0_m_per_s2)
    }

    #[pyo3(name = "modify_with_braking_trajectory")]
    pub fn modify_with_braking_trajectory_py(
        &mut self,
        brake_accel_m_per_s2: f64,
        idx: usize,
        dts_m: Option<f64>
    ) -> anyhow::Result<(f64, usize)> {
        self.modify_with_braking_trajectory(brake_accel_m_per_s2, idx, dts_m)
    }

    #[pyo3(name = "calc_distance_to_next_stop_from")]
    pub fn calc_distance_to_next_stop_from_py(&self, distance_m: f64) -> f64 {
        self.calc_distance_to_next_stop_from(distance_m, None)
    }

    #[pyo3(name = "average_grade_over_range")]
    pub fn average_grade_over_range_py(
        &self,
        distance_start_m: f64,
        delta_distance_m: f64,
    ) -> f64 {
        self.average_grade_over_range(distance_start_m, delta_distance_m, None)
    }

    #[pyo3(name = "build_cache")]
    pub fn build_cache_py(&self) -> RustCycleCache {
        self.build_cache()
    }

    #[pyo3(name = "dt_s_at_i")]
    pub fn dt_s_at_i_py(&self, i: usize) -> f64 {
        if i == 0 {
            0.0
        } else {
            self.dt_s_at_i(i)
        }
    }

    #[getter]
    pub fn get_mph(&self) -> Vec<f64> {
        (&self.mps * crate::params::MPH_PER_MPS).to_vec()
    }
    #[setter]
    pub fn set_mph(&mut self, new_value: Vec<f64>) {
        self.mps = Array::from_vec(new_value) / MPH_PER_MPS;
    }
    #[getter]
    /// array of time steps
    pub fn get_dt_s(&self) -> Vec<f64> {
        self.dt_s().to_vec()
    }
    #[getter]
    /// distance for each time step based on final speed
    pub fn get_dist_m(&self) -> Vec<f64> {
        self.dist_m().to_vec()
    }
    #[getter]
    pub fn get_delta_elev_m(&self) -> Vec<f64> {
        self.delta_elev_m().to_vec()
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
    #[serde(default)]
    pub grade: Array1<f64>,
    /// array of max possible charge rate from roadway
    #[serde(alias = "cycRoadType")]
    #[serde(default)]
    pub road_type: Array1<f64>,
    pub name: String,
    #[serde(skip)]
    pub orphaned: bool,
}

impl SerdeAPI for RustCycle {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin", "csv"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json", "csv"];
    const CACHE_FOLDER: &'static str = &"cycles";

    fn init(&mut self) -> anyhow::Result<()> {
        self.init_checks()
    }

    fn to_writer<W: std::io::Write>(&self, wtr: W, format: &str) -> anyhow::Result<()> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)?,
            "json" => serde_json::to_writer(wtr, self)?,
            "bin" => bincode::serialize_into(wtr, self)?,
            "csv" => {
                let mut wtr = csv::Writer::from_writer(wtr);
                for i in 0..self.len() {
                    wtr.serialize(RustCycleElement {
                        time_s: self.time_s[i],
                        mps: self.mps[i],
                        grade: Some(self.grade[i]),
                        road_type: Some(self.road_type[i]),
                    })?;
                }
                wtr.flush()?
            }
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    fn to_str(&self, format: &str) -> anyhow::Result<String> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => self.to_yaml()?,
                "json" => self.to_json()?,
                "csv" => self.to_csv()?,
                _ => {
                    bail!(
                        "Unsupported format {format:?}, must be one of {:?}",
                        Self::ACCEPTED_STR_FORMATS
                    )
                }
            },
        )
    }

    /// Note that using this method to instantiate a RustCycle from CSV, rather
    /// than the `from_csv_str` method, sets the cycle name to an empty string
    fn from_str<S: AsRef<str>>(contents: S, format: &str) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => Self::from_yaml(contents)?,
                "json" => Self::from_json(contents)?,
                "csv" => Self::from_reader(contents.as_ref().as_bytes(), "csv")?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_STR_FORMATS
                ),
            },
        )
    }

    fn from_reader<R: std::io::Read>(rdr: R, format: &str) -> anyhow::Result<Self> {
        let mut deserialized = match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            "json" => serde_json::from_reader(rdr)?,
            "bin" => bincode::deserialize_from(rdr)?,
            "csv" => {
                // Create empty cycle to be populated
                let mut cyc = Self::default();
                let mut rdr = csv::Reader::from_reader(rdr);
                for result in rdr.deserialize() {
                    cyc.push(result?);
                }
                cyc
            }
            _ => {
                bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_BYTE_FORMATS
                )
            }
        };
        deserialized.init()?;
        Ok(deserialized)
    }
}

impl TryFrom<HashMap<String, Vec<f64>>> for RustCycle {
    type Error = anyhow::Error;

    fn try_from(hashmap: HashMap<String, Vec<f64>>) -> anyhow::Result<Self> {
        let time_s = Array::from_vec(
            hashmap
                .get("time_s")
                .with_context(|| format!("`time_s` not in HashMap: {hashmap:?}"))?
                .to_owned(),
        );
        let cyc_len = time_s.len();
        let mut cyc = Self {
            time_s,
            mps: Array::from_vec(
                hashmap
                    .get("mps")
                    .with_context(|| format!("`mps` not in HashMap: {hashmap:?}"))?
                    .to_owned(),
            ),
            grade: Array::from_vec(
                hashmap
                    .get("grade")
                    .unwrap_or(&vec![0.0; cyc_len])
                    .to_owned(),
            ),
            road_type: Array::from_vec(
                hashmap
                    .get("road_type")
                    .unwrap_or(&vec![0.0; cyc_len])
                    .to_owned(),
            ),
            name: String::default(),
            orphaned: false,
        };
        cyc.init()?;
        Ok(cyc)
    }
}

impl From<RustCycle> for HashMap<String, Vec<f64>> {
    fn from(cyc: RustCycle) -> Self {
        HashMap::from([
            ("time_s".into(), cyc.time_s.to_vec()),
            ("mps".into(), cyc.mps.to_vec()),
            ("grade".into(), cyc.grade.to_vec()),
            ("road_type".into(), cyc.road_type.to_vec()),
        ])
    }
}

/// pure Rust methods that need to be separate due to pymethods incompatibility
impl RustCycle {
    fn init_checks(&self) -> anyhow::Result<()> {
        ensure!(!self.is_empty(), "Deserialized cycle is empty");
        ensure!(self.is_sorted(), "Deserialized cycle is not sorted in time");
        ensure!(
            self.are_fields_equal_length(),
            "Deserialized cycle has unequal field lengths\ntime_s: {}\nmps: {}\ngrade: {}\nroad_type: {}",
            self.time_s.len(),
            self.mps.len(),
            self.grade.len(),
            self.road_type.len(),
        );
        Ok(())
    }

    /// Load cycle from CSV file, parsing name from filepath
    pub fn from_csv_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let name = filepath
            .file_stem()
            .and_then(OsStr::to_str)
            .with_context(|| format!("Could not parse cycle name from filepath: {filepath:?}"))?
            .to_string();
        let mut cyc = Self::from_file(filepath)?;
        cyc.name = name;
        Ok(cyc)
    }

    /// Load cycle from CSV string
    pub fn from_csv_str<S: AsRef<str>>(csv_str: S, name: String) -> anyhow::Result<Self> {
        let mut cyc = Self::from_str(csv_str, "csv")?;
        cyc.name = name;
        Ok(cyc)
    }

    /// Write (serialize) cycle to a CSV string
    pub fn to_csv(&self) -> anyhow::Result<String> {
        let mut buf = Vec::with_capacity(self.len());
        self.to_writer(&mut buf, "csv")?;
        Ok(String::from_utf8(buf)?)
    }

    pub fn build_cache(&self) -> RustCycleCache {
        RustCycleCache::new(self)
    }

    pub fn push(&mut self, cyc_elem: RustCycleElement) {
        self.time_s
            .append(Axis(0), array![cyc_elem.time_s].view())
            .unwrap();
        self.mps
            .append(Axis(0), array![cyc_elem.mps].view())
            .unwrap();
        if let Some(grade) = cyc_elem.grade {
            self.grade.append(Axis(0), array![grade].view()).unwrap();
        }
        if let Some(road_type) = cyc_elem.road_type {
            self.road_type
                .append(Axis(0), array![road_type].view())
                .unwrap();
        }
    }

    pub fn len(&self) -> usize {
        self.time_s.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_sorted(&self) -> bool {
        self.time_s
            .as_slice()
            .unwrap()
            .windows(2)
            .all(|window| window[0] < window[1])
    }

    pub fn are_fields_equal_length(&self) -> bool {
        let cyc_len = self.len();
        [self.mps.len(), self.grade.len(), self.road_type.len()]
            .iter()
            .all(|len| len == &cyc_len)
    }

    pub fn test_cyc() -> Self {
        Self {
            time_s: Array::range(0.0, 10.0, 1.0),
            mps: Array::range(0.0, 10.0, 1.0),
            grade: Array::zeros(10),
            road_type: Array::zeros(10),
            name: String::from("test"),
            orphaned: false,
        }
    }

    /// Returns the average grade over the given range of distances
    /// - distance_start_m: non-negative-number, the distance at start of evaluation area (m)
    /// - delta_distance_m: non-negative-number, the distance traveled from distance_start_m (m)
    /// RETURN: number, the average grade (rise over run) over the given distance range
    /// Note: grade is assumed to be constant from just after the previous sample point
    /// until the current sample point. That is, grade\[i\] applies over the range of
    /// distances, d, from (d\[i - 1\], d\[i\])
    pub fn average_grade_over_range(
        &self,
        distance_start_m: f64,
        delta_distance_m: f64,
        cache: Option<&RustCycleCache>,
    ) -> f64 {
        let tol = 1e-6;
        match &cache {
            Some(rcc) => {
                if rcc.grade_all_zero {
                    0.0
                } else if delta_distance_m <= tol {
                    rcc.interp_grade(distance_start_m)
                } else {
                    let e0 = rcc.interp_elevation(distance_start_m);
                    let e1 = rcc.interp_elevation(distance_start_m + delta_distance_m);
                    ((e1 - e0) / delta_distance_m).asin().tan()
                }
            }
            None => {
                let grade_all_zero = {
                    let mut all0 = true;
                    for idx in 0..self.len() {
                        if self.grade[idx] != 0.0 {
                            all0 = false;
                            break;
                        }
                    }
                    all0
                };
                if grade_all_zero {
                    0.0
                } else {
                    let delta_dists = trapz_step_distances(self);
                    let trapz_distances_m = ndarrcumsum(&delta_dists);
                    if delta_distance_m <= tol {
                        if distance_start_m <= trapz_distances_m[0] {
                            return self.grade[0];
                        }
                        let max_idx = self.len() - 1;
                        if distance_start_m > trapz_distances_m[max_idx] {
                            return self.grade[max_idx];
                        }
                        for idx in 1..self.time_s.len() {
                            if distance_start_m > trapz_distances_m[idx - 1]
                                && distance_start_m <= trapz_distances_m[idx]
                            {
                                return self.grade[idx];
                            }
                        }
                        self.grade[max_idx]
                    } else {
                        // NOTE: we use the following instead of delta_elev_m in order to use
                        // more precise trapezoidal distance and elevation at sample points.
                        // This also uses the fully accurate trig functions in case we have large slope angles.
                        let trapz_elevations_m = ndarrcumsum(
                            &(self.grade.mapv(|g| g.atan().cos()) * delta_dists * &self.grade),
                        );
                        let e0 = interpolate(
                            &distance_start_m,
                            &trapz_distances_m,
                            &trapz_elevations_m,
                            false,
                        );
                        let e1 = interpolate(
                            &(distance_start_m + delta_distance_m),
                            &trapz_distances_m,
                            &trapz_elevations_m,
                            false,
                        );
                        ((e1 - e0) / delta_distance_m).asin().tan()
                    }
                }
            }
        }
    }

    /// Calculate the distance to next stop from `distance_m`
    /// - distance_m: non-negative-number, the current distance from start (m)
    /// RETURN: returns the distance to the next stop from distance_m
    /// NOTE: distance may be negative if we're beyond the last stop
    pub fn calc_distance_to_next_stop_from(
        &self,
        distance_m: f64,
        cache: Option<&RustCycleCache>,
    ) -> f64 {
        let tol: f64 = 1e-6;
        match cache {
            Some(rcc) => {
                for (&dist, &v) in rcc.trapz_distances_m.iter().zip(self.mps.iter()) {
                    if (v < tol) && (dist > (distance_m + tol)) {
                        return dist - distance_m;
                    }
                }
                rcc.trapz_distances_m.last().unwrap() - distance_m
            }
            None => {
                let ds = ndarrcumsum(&trapz_step_distances(self));
                for (&dist, &v) in ds.iter().zip(self.mps.iter()) {
                    if (v < tol) && (dist > (distance_m + tol)) {
                        return dist - distance_m;
                    }
                }
                ds.last().unwrap() - distance_m
            }
        }
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
    ) -> anyhow::Result<(f64, usize)> {
        ensure!(brake_accel_m_per_s2 < 0.0);
        if i >= self.time_s.len() {
            return Ok((*self.mps.last().unwrap(), 0));
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
            return Ok((v0, 0));
        }
        // time-to-stop (s)
        let tts_s = -v0 / brake_accel_m_per_s2;
        // number of steps to take
        let n: usize = (tts_s / dt).round() as usize;
        let n: usize = if n < 2 { 2 } else { n }; // need at least 2 steps
        let (jerk_m_per_s3, accel_m_per_s2) =
            calc_constant_jerk_trajectory(n, 0.0, v0, dts_m, 0.0, dt)?;
        Ok((
            self.modify_by_const_jerk_trajectory(i, n, jerk_m_per_s3, accel_m_per_s2),
            n,
        ))
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

    /// elevation change w.r.t. to initial
    pub fn delta_elev_m(&self) -> Array1<f64> {
        ndarrcumsum(&(self.dist_m() * &self.grade))
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
    if i >= cyc.len() {
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
        let cyc = RustCycle {
            time_s: array![0.0, 10.0, 30.0, 34.0, 40.0],
            mps: array![0.0, 10.0, 10.0, 0.0, 0.0],
            grade: Array::zeros(5),
            road_type: Array::zeros(5),
            name: String::from("test"),
            orphaned: false,
        };
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
        let cyc_file_path = resources_path().join("cycles/udds.csv");
        let expected_udds_length: usize = 1370;
        let cyc = RustCycle::from_csv_file(cyc_file_path).unwrap();
        let num_entries = cyc.len();
        assert_eq!(cyc.name, String::from("udds"));
        assert!(num_entries > 0);
        assert_eq!(num_entries, cyc.len());
        assert_eq!(num_entries, cyc.mps.len());
        assert_eq!(num_entries, cyc.grade.len());
        assert_eq!(num_entries, cyc.road_type.len());
        assert_eq!(num_entries, expected_udds_length);
    }

    #[test]
    fn test_str_serde() {
        let cyc = RustCycle::test_cyc();
        for format in RustCycle::ACCEPTED_STR_FORMATS {
            let csv_str = cyc.to_str(format).unwrap();
            RustCycle::from_str(&csv_str, format).unwrap();
        }
    }
}
