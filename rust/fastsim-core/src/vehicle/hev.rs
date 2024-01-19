use super::*;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
/// Hybrid vehicle with both engine and reversible energy storage (aka battery)  
/// This type of vehicle is not likely to be widely prevalent due to modularity of consists.  
pub struct HybridElectricVehicle {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub e_machine: ElectricMachine,
}

impl SerdeAPI for HybridElectricVehicle {}

impl Powertrain for Box<HybridElectricVehicle> {
    fn get_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<si::Power> {
        todo!();
    }
    fn solve_powertrain(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        todo!()
    }
}
