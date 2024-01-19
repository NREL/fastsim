// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]
//! Module containing submodules for vehicle and powertrain models

use crate::imports::*;

// powertrain types
pub mod bev;
pub mod conv;
pub mod hev;

// powertrain components
pub mod powertrain;

// vehicle model
pub mod powertrain_type;
pub mod vehicle_model;

#[cfg(test)]
/// Module containing tests for consists.
pub mod tests;

pub use bev::BatteryElectricVehicle;
pub use conv::ConventionalVehicle;
pub use hev::HybridElectricVehicle;
pub use powertrain::e_machine::ElectricMachine;
pub use powertrain::fuel_converter::FuelConverter;
pub use powertrain::fuel_storage::FuelStorage;
pub use powertrain::powertrain_traits::{Mass, Powertrain};
pub use powertrain::reversible_energy_storage::ReversibleEnergyStorage;
pub use powertrain_type::PowertrainType;
pub use vehicle_model::Vehicle;
