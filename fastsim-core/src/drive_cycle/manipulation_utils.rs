use crate::imports::*;

#[fastsim_api]
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
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

#[cfg(test)]
mod tests {
    use fastsim_2::cycle::speed_for_constant_jerk;

    use super::*;
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
}
