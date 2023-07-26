use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub trans: Transmission,
}

impl ConventionalVehicle {
    pub fn new(fuel_converter: FuelConverter, trans: Transmission) -> Self {
        ConventionalVehicle {
            fc: fuel_converter,
            trans: trans,
        }
    }

    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: bool,
        pwr_aux: si::Power,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.trans.set_pwr_in_req(pwr_out_req, dt)?;
        Ok(())
    }
}

impl VehicleTrait for Box<ConventionalVehicle> {
    /// returns current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        todo!();
        Ok(())
    }

    fn save_state(&mut self) {
        self.save_state();
    }

    fn step(&mut self) {
        self.step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss + self.trans.state.energy_loss
    }
}
