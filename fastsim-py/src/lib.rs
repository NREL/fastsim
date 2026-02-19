//! Crate that wraps `fastsim-core` and enables the `pyo3` feature to
//! expose most structs, methods, and functions to Python.

use fastsim_2::simdrive::RustSimDrive as SimDriveF2;
use fastsim_2::vehicle::RustVehicle as VehicleF2;
use fastsim_core::prelude::*;
use pyo3::prelude::*;
pub use pyo3::types::PyType;

#[pymodule]
fn fastsim(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Air>()?;
    m.add_class::<FuelConverter>()?;
    m.add_class::<FuelConverterState>()?;
    m.add_class::<FuelConverterStateHistoryVec>()?;
    m.add_class::<FuelConverterThermal>()?;
    m.add_class::<FuelConverterThermalState>()?;
    m.add_class::<FuelConverterThermalStateHistoryVec>()?;
    m.add_class::<HVACSystemForLumpedCabin>()?;
    m.add_class::<HVACSystemForLumpedCabinState>()?;
    m.add_class::<HVACSystemForLumpedCabinStateHistoryVec>()?;
    m.add_class::<HVACSystemForLumpedCabinAndRES>()?;
    m.add_class::<HVACSystemForLumpedCabinAndRESState>()?;
    m.add_class::<HVACSystemForLumpedCabinAndRESStateHistoryVec>()?;
    m.add_class::<RESLumpedThermal>()?;
    m.add_class::<RESLumpedThermalState>()?;
    m.add_class::<RESLumpedThermalStateHistoryVec>()?;
    m.add_class::<LumpedCabin>()?;
    m.add_class::<LumpedCabinState>()?;
    m.add_class::<LumpedCabinStateHistoryVec>()?;
    m.add_class::<Maneuver>()?;
    m.add_class::<ReversibleEnergyStorage>()?;
    m.add_class::<ReversibleEnergyStorageState>()?;
    m.add_class::<ReversibleEnergyStorageStateHistoryVec>()?;
    m.add_class::<ElectricMachine>()?;
    m.add_class::<ElectricMachineState>()?;
    m.add_class::<ElectricMachineStateHistoryVec>()?;
    m.add_class::<RESGreedyWithDynamicBuffers>()?;
    m.add_class::<Cycle>()?;
    m.add_class::<CycleElement>()?;
    m.add_class::<Vehicle>()?;
    m.add_class::<SimDrive>()?;
    m.add_class::<SimParams>()?;
    m.add_class::<SimDriveF2>()?;
    m.add_class::<VehicleF2>()?;

    // List enabled features
    m.add_function(wrap_pyfunction!(fastsim_core::enabled_features, m)?)?;

    m.add_function(wrap_pyfunction!(fastsim_core::simdrivelabel::get_label_fe_py, m)?)?;

    Ok(())
}
