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
        ensure!(self.cyc.len() >= 2, format_dbg!(self.cyc.len() < 2));
        while self.veh.state.i < self.cyc.len() {
            let i = self.veh.state.i;
            let dt = self.cyc.dt_at_i(i);
            self.veh.solve_step(self.cyc.speed[i], dt)?;
            self.veh.step();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sim_drive() {
        let veh = Vehicle::test_conv_veh();
        let cyc = Cycle::from_file(todo!()).unwrap();
        let sd = SimDrive { veh, cyc };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fuel_converter().unwrap().state.energy_fuel > uc::J * 0.);
    }
}
