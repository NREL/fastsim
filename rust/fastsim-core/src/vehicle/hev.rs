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

impl HybridElectricVehicle {
    pub fn get_cur_pwr_max_out(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        todo!();
    }
}
