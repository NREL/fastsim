use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub e_machine: ElectricMachine,
}

impl ConventionalVehicle {
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: bool,
        pwr_aux: si::Power,
    ) -> anyhow::Result<()> {
        self.e_machine.set_pwr_in_req(pwr_out_req, dt)?;
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
        // TODO
        Ok(())
    }

    fn save_state(&mut self) {
        self.save_state();
    }

    fn step(&mut self) {
        self.step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss + self.e_machine.state.energy_loss
    }
}
