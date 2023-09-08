use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    /// Transmission efficiency
    pub trans_eff: si::Ratio,
}

impl ConventionalVehicle {
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        fc_on: bool,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.fc
            .solve_energy_consumption(pwr_out_req + pwr_aux, dt, fc_on, assert_limits)?;
        Ok(())
    }
}

impl VehicleTrait for Box<ConventionalVehicle> {
    /// returns current max power
    fn set_cur_pwr_max_out(
        &mut self,
        _pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.fc.set_cur_pwr_out_max(dt)?;
        Ok(())
    }

    fn save_state(&mut self) {
        self.fs.save_state();
        self.fc.save_state();
    }

    fn step(&mut self) {
        self.fs.step();
        self.fc.step();
    }
}
