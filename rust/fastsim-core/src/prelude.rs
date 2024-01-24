//! Convenience module for exposing commonly used structs
// TODO: consider exposing more structs here

pub use crate::drive_cycle::{Cycle, CycleElement};
pub use crate::simdrive::{SimDrive, SimParams};
pub use crate::utils::{Pyo3Vec2Wrapper, Pyo3Vec3Wrapper, Pyo3VecBoolWrapper, Pyo3VecWrapper};
pub use crate::vehicle::powertrain::e_machine::{
    ElectricMachine, ElectricMachineState, ElectricMachineStateHistoryVec,
};
pub use crate::vehicle::powertrain::fuel_converter::{
    FuelConverter, FuelConverterState, FuelConverterStateHistoryVec,
};
pub use crate::vehicle::powertrain::reversible_energy_storage::{
    ReversibleEnergyStorage, ReversibleEnergyStorageState, ReversibleEnergyStorageStateHistoryVec,
};
pub use crate::vehicle::Vehicle;
