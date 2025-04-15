use super::*;

#[fastsim_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods)]
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

        state.eff.update(
            match self.eff_interp {
                Interpolator::Interp0D(eff) => eff * uc::R,
                _ => unimplemented!(),
            },
            format_dbg!(),
        )?;
        ensure!(
            *state.eff.get(format_dbg!())? >= 0.0 * uc::R
                && *state.eff.get(format_dbg!())? <= 1.0 * uc::R,
            format!(
                "{}\nTransmission efficiency ({}) must be between 0 and 1",
                format_dbg!(
                    *state.eff.get(format_dbg!())? >= 0.0 * uc::R
                        && *state.eff.get(format_dbg!())? <= 1.0 * uc::R
                ),
                state.eff.get(format_dbg!())?.get::<si::ratio>()
            )
        );

        state.pwr_out.update(pwr_out_req, format_dbg!())?;
        state.pwr_in.update(
            if *state.pwr_out.get(format_dbg!())? > si::Power::ZERO {
                *state.pwr_out.get(format_dbg!())? / *state.eff.get(format_dbg!())?
            } else {
                *state.pwr_out.get(format_dbg!())? * *state.eff.get(format_dbg!())?
            },
            format_dbg!(),
        )?;
        state.pwr_loss.update(
            (*state.pwr_in.get(format_dbg!())? - *state.pwr_out.get(format_dbg!())?).abs(),
            format_dbg!(),
        )?;

        Ok(*state.pwr_in.get(format_dbg!())?)
    }
}
impl HistoryMethods for Transmission {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear()
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
#[derive(
    Clone,
    Default,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    HistoryVec,
    SetCumulative,
    StateMethods,
)]
#[non_exhaustive]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct TransmissionState {
    /// time step index
    pub i: TrackedStateWithMemory<usize>,

    pub eff: TrackedState<si::Ratio>,

    pub pwr_out: TrackedState<si::Power>,
    pub pwr_in: TrackedState<si::Power>,
    /// Power loss: [Self::pwr_in] - [Self::pwr_out]
    pub pwr_loss: TrackedState<si::Power>,

    pub energy_out: TrackedStateWithMemory<si::Energy>,
    pub energy_loss: TrackedStateWithMemory<si::Energy>,
}

impl Init for TransmissionState {}
impl SerdeAPI for TransmissionState {}
