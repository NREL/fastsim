//! Convenience module for exposing commonly used structs
// NOTE: consider exposing more structs and other stuff here

pub use crate::drive_cycle::{Cycle, CycleElement};
pub use crate::gas_properties::{get_sphere_conv_params, Air, Octane, H_STD, TE_STD_AIR};
pub use crate::simdrive::{SimDrive, SimParams};
pub use crate::utils::{Pyo3Vec2Wrapper, Pyo3Vec3Wrapper, Pyo3VecBoolWrapper, Pyo3VecWrapper};
pub use crate::vehicle::cabin::{
    CabinOption, LumpedCabin, LumpedCabinState, LumpedCabinStateHistoryVec,
};
pub use crate::vehicle::hev::RESGreedyWithDynamicBuffers;
pub use crate::vehicle::hvac::{
    HVACOption, HVACSystemForLumpedCabin, HVACSystemForLumpedCabinAndRES,
    HVACSystemForLumpedCabinAndRESState, HVACSystemForLumpedCabinAndRESStateHistoryVec,
    HVACSystemForLumpedCabinState, HVACSystemForLumpedCabinStateHistoryVec,
};
pub use crate::vehicle::powertrain::electric_machine::{
    ElectricMachine, ElectricMachineState, ElectricMachineStateHistoryVec,
};
pub use crate::vehicle::powertrain::fuel_converter::{
    FuelConverter, FuelConverterState, FuelConverterStateHistoryVec, FuelConverterThermal,
    FuelConverterThermalOption, FuelConverterThermalState, FuelConverterThermalStateHistoryVec,
};
pub use crate::vehicle::powertrain::reversible_energy_storage::{
    RESLumpedThermal, RESLumpedThermalState, RESLumpedThermalStateHistoryVec, RESThermalOption,
    ReversibleEnergyStorage, ReversibleEnergyStorageState, ReversibleEnergyStorageStateHistoryVec,
};
pub use crate::vehicle::PowertrainType;
pub use crate::vehicle::Vehicle;
