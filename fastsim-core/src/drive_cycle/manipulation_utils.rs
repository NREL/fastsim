use crate::drive_cycle::Cycle;
use crate::imports::*;

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
    pub fn from_speed_and_distance_targets(
        n: usize,
        d0: f64,
        v0: f64,
        dr: f64,
        vr: f64,
        dt: f64,
    ) -> ConstantJerkTrajectory {
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
            steps: n as usize,
            distance_m: d0,
            speed_m_per_s: v0,
            acceleration_m_per_s2: a0,
            jerk_m_per_s3: k,
            step_duration_s: dt,
        }
    }
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
    pub fn end_distance(&self) -> f64 {
        self.distance_at_step(self.steps)
    }
    pub fn speed_at_step(&self, n: usize) -> f64 {
        let n = n as f64;
        let v0 = self.speed_m_per_s;
        let a0 = self.acceleration_m_per_s2;
        let k = self.jerk_m_per_s3;
        let dt = self.step_duration_s;
        v0 + (n * a0 * dt) + (0.5 * n * (n - 1.0) * k * dt)
    }
    pub fn end_speed(&self) -> f64 {
        self.speed_at_step(self.steps)
    }
    pub fn acceleration_at_step(&self, n: usize) -> f64 {
        let n = n as f64;
        let a0 = self.acceleration_m_per_s2;
        let k = self.jerk_m_per_s3;
        let dt = self.step_duration_s;
        a0 + (n * k * dt)
    }
    pub fn end_acceleration(&self) -> f64 {
        self.acceleration_at_step(self.steps)
    }
    pub fn all_accelerations(&self) -> Vec<f64> {
        let mut accels = Vec::with_capacity(self.steps);
        for n_idx in 0..self.steps {
            accels.push(self.acceleration_at_step(n_idx));
        }
        accels
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
) -> anyhow::Result<ConstantJerkTrajectory> {
    ensure!(n > 1);
    ensure!(dr > d0);
    ensure!(v0 >= 0.0);
    ensure!(vr >= 0.0);
    ensure!(dt > 0.0);
    Ok(ConstantJerkTrajectory::from_speed_and_distance_targets(
        n, d0, v0, dr, vr, dt,
    ))
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
/// RETURN: new cycle with time extended by both the absolute
///         and fraction values
/// NOTE: absolute_time and time_faction are optional. Pass None to
/// remove them from the equation.
pub fn extend_cycle_time(
    cyc: &Cycle,
    absolute_time: Option<si::Time>,
    time_fraction: Option<si::Ratio>,
) -> Cycle {
    cyc.extend_time(absolute_time, time_fraction)
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
        let expected = vec![1.0, 1.0];
        let actual = accel_array_for_constant_jerk(n, a0_m_per_s2, k_m_per_s3, dt_s);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(actual[i], expected[i]);
        }
    }
    #[test]
    fn test_average_step_speeds() {
        let cyc = make_triangle_cycle();
        let expected = vec![0.0 * uc::MPS, 2.0 * uc::MPS, 2.0 * uc::MPS, 0.0 * uc::MPS];
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
        let expected = vec![0.0 * uc::M, 20.0 * uc::M, 20.0 * uc::M, 0.0 * uc::M];
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
        let expected = vec![(0.0 * uc::M, (40.0 / 30.0) * uc::MPS)];
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
}
