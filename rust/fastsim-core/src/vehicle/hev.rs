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
    // TODO: add enum for controling fraction of aux pwr handled by battery vs engine
    // TODO: add enum for controling fraction of tractive pwr handled by battery vs engine -- there
    // might be many ways we'd want to do this, especially since there will be thermal models involved
}

impl SaveInterval for HybridElectricVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in HybridElectricVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.res.save_interval = save_interval;
        self.e_machine.save_interval = save_interval;
        Ok(())
    }
}

impl SerdeAPI for HybridElectricVehicle {}

impl Powertrain for Box<HybridElectricVehicle> {
    fn get_curr_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<si::Power> {
        todo!();
    }
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        todo!()
    }
}
