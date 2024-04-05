use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    /// Alternator efficiency used to calculate aux mechanical power demand on engine
    pub alt_eff: si::Ratio,
}

impl SaveInterval for ConventionalVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in ConventionalVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.fc.save_interval = save_interval;
        Ok(())
    }
}

impl Powertrain for Box<ConventionalVehicle> {
    fn get_curr_pwr_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        self.fc.get_curr_pwr_out_max(pwr_aux / self.alt_eff, dt)
        // TODO: put transmission efficiency in here somehow
    }
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        let enabled = true; // TODO: replace with a stop/start model
        self.fc
            .solve(pwr_out_req, pwr_aux, enabled, dt, assert_limits)?;
        Ok(())
    }
}
