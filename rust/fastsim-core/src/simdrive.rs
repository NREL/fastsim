use super::drive_cycle::Cycle;
use super::vehicle::{Vehicle, VehicleTrait};
use crate::imports::*;

#[pyo3_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
pub struct SimDrive {
    veh: Vehicle,
    cyc: Cycle,
}

impl SimDrive {
    fn walk(&mut self) -> anyhow::Result<()> {
        while self.veh.state.i <= self.cyc.len() {
            self.step();
        }
        Ok(())
    }

    fn step(&mut self) -> anyhow::Result<()> {
        self.veh.step();
        Ok(())
    }
}
