use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
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
            .solve_energy_consumption(pwr_out_req, pwr_aux, fc_on, dt, assert_limits)?;
        Ok(())
    }

    /// # Arguments
    /// - pwr_aux: amount of auxilliary power required from engine
    /// - dt: time step size
    pub fn get_cur_pwr_max_out(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        self.fc.set_cur_pwr_out_max(dt)?;
        Ok((self.fc.state.pwr_out_max - pwr_aux) * self.trans_eff)
    }
}
