use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
}

impl Powertrain for Box<ConventionalVehicle> {
    fn get_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<si::Power> {
        self.fc.set_cur_pwr_out_max(dt)?;
        Ok(self.fc.state.pwr_out_max)
    }
    fn solve_powertrain(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        let fc_on = true;
        self.fc
            .solve_energy_consumption(pwr_out_req, pwr_aux, fc_on, dt, assert_limits)?;
        Ok(())
    }
}
