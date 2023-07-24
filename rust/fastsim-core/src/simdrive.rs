use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::imports::*;

#[pyo3_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
pub struct SimDrive {
    veh: Vehicle,
    cyc: Cycle,
}
