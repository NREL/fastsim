use super::*;

#[fastsim_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct Transmission {
    /// Transmission mass
    #[serde(default)]
    pub(crate) mass: Option<si::Mass>,

    /// interpolator for calculating [Self] efficiency as a function of the following variants:  
    /// - 0d -- constant
    pub eff_interp: Interpolator,

    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    pub state: TransmissionState,
    /// Custom vector of [Self::state]
    #[serde(default, skip_serializing_if = "TransmissionStateHistoryVec::is_empty")]
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
impl SaveInterval for Transmission {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
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

#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[non_exhaustive]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct TransmissionState {
    /// time step index
    pub i: usize,

    pub eff: si::Ratio,

    pub pwr_out: si::Power,
    pub pwr_in: si::Power,
    /// Power loss: [Self::pwr_in] - [Self::pwr_out]
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
