use crate::imports::*;

#[altrios_api(
    #[new]
    fn __new__(
    force_max_newtons: f64,
    ramp_up_time_seconds: f64,
    ramp_up_coeff: f64,
    // recharge_rate_pa_per_sec: f64,
    state: Option<FricBrakeState>,
    save_interval: Option<usize>,
    ) -> Self {
        Self::new(
            force_max_newtons * uc::N,
            ramp_up_time_seconds * uc::S,
            ramp_up_coeff * uc::R,
            // recharge_rate_pa_per_sec,
            state,
            save_interval,
        )
    }
)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
pub struct FricBrake {
    /// max static force achievable
    pub force_max: si::Force,
    /// time to go from zero to max braking force
    pub ramp_up_time: si::Time,
    /// ramp-up correction factor
    pub ramp_up_coeff: si::Ratio,
    // commented out.  This stuff needs refinement but
    // added complexity is probably worthwhile
    // /// time to go from max braking force to zero braking force
    // pub ramp_down_time: si::Time,
    // /// rate at which brakes can be recovered after full release
    // pub recharge_rate_pa_per_sec: f64,
    // TODO: add in whatever is needed to estimate aux load impact
    pub state: FricBrakeState,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: FricBrakeStateHistoryVec,
    pub save_interval: Option<usize>,
}

impl Default for FricBrake {
    fn default() -> Self {
        Self {
            force_max: 600_000.0 * uc::LBF,
            ramp_up_time: 60.0 * uc::S,
            ramp_up_coeff: 0.5 * uc::R,
            state: Default::default(),
            history: Default::default(),
            save_interval: Default::default(),
        }
    }
}

impl FricBrake {
    pub fn new(
        force_max: si::Force,
        ramp_up_time: si::Time,
        ramp_up_coeff: si::Ratio,
        // recharge_rate_pa_per_sec: f64,
        state: Option<FricBrakeState>,
        save_interval: Option<usize>,
    ) -> Self {
        let mut state = state.unwrap_or_default();
        state.force_max_curr = force_max;
        Self {
            force_max,
            ramp_up_time,
            ramp_up_coeff,
            // recharge_rate_pa_per_sec,
            state,
            history: Default::default(),
            save_interval,
        }
    }

    pub fn set_cur_force_max_out(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // maybe check parameter values here and propagate any errors
        self.state.force_max_curr =
            (self.state.force + self.force_max / self.ramp_up_time * dt).min(self.force_max);
        Ok(())
    }
}

// TODO: figure out a way to make the braking reasonably polymorphic (e.g. for Parallel Systems)

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, HistoryVec)]
#[altrios_api(
    #[new]
    fn __new__(
    ) -> Self {
        Self::new()
    }
)]
pub struct FricBrakeState {
    /// index counter
    pub i: usize,
    // actual applied force of brakes
    pub force: si::Force,
    // time-varying max force of brakes in current time step
    pub force_max_curr: si::Force,
    // pressure: si::Pressure,
}

impl FricBrakeState {
    /// TODO: this method needs to accept arguments
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for FricBrakeState {
    fn default() -> Self {
        Self {
            i: 1,
            force: Default::default(),
            force_max_curr: Default::default(),
        }
    }
}
