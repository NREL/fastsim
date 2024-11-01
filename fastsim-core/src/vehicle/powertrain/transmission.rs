use super::*;

#[fastsim_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
pub struct Transmission {
    /// Transmission mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    pub(crate) mass: Option<si::Mass>,

    /// interpolator for calculating [Self] efficiency as a function of the following variants:  
    /// - 0d -- constant
    #[api(skip_get, skip_set)]
    pub eff_interp: Interpolator,

    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: TransmissionState,
    /// Custom vector of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "TransmissionStateHistoryVec::is_empty")]
    pub history: TransmissionStateHistoryVec,
}

impl Transmission {
    pub fn get_pwr_in_req(&mut self, pwr_out_req: si::Power) -> anyhow::Result<si::Power> {
        let state = &mut self.state;

        state.eff = match self.eff_interp {
            Interpolator::Interp0D(eff) => eff * uc::R,
            _ => unimplemented!(),
        };
        ensure!(
            state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R,
            format!(
                "{}\nTransmission efficiency ({}) must be between 0 and 1",
                format_dbg!(state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R),
                state.eff.get::<si::ratio>()
            )
        );

        state.pwr_out = pwr_out_req;
        state.pwr_in = if state.pwr_out > si::Power::ZERO {
            state.pwr_out / state.eff
        } else {
            state.pwr_out * state.eff
        };
        state.pwr_loss = (state.pwr_in - state.pwr_out).abs();

        Ok(state.pwr_in)
    }
}

impl SerdeAPI for Transmission {}
impl Init for Transmission {}

impl Mass for Transmission {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self.mass)
    }

    // TODO: the side effect doesn't really do anything, hmmm
    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        self.mass = new_mass;

        Ok(())
    }

    // TODO this also doesn't really need to exist, except for the trait's sake
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self.mass)
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
    }
}

impl SetCumulative for Transmission {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
pub struct TransmissionState {
    /// time step index
    pub i: usize,

    pub eff: si::Ratio,

    pub pwr_out: si::Power,
    pub pwr_in: si::Power,
    /// Power loss: [`pwr_in`] - [`pwr_out`]
    pub pwr_loss: si::Power,

    pub energy_out: si::Energy,
    pub energy_loss: si::Energy,
}

impl Default for TransmissionState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            eff: si::Ratio::ZERO,
            pwr_out: si::Power::ZERO,
            pwr_in: si::Power::ZERO,
            pwr_loss: si::Power::ZERO,
            energy_out: si::Energy::ZERO,
            energy_loss: si::Energy::ZERO,
        }
    }
}

impl Init for TransmissionState {}
impl SerdeAPI for TransmissionState {}
