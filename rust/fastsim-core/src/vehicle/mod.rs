// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]
//! Module containing submodules for consists, locomotives, and powertrain models

use crate::imports::*;

// powertrain types
pub mod bev;
pub mod conv;
pub mod hev;

// components
pub mod electric_drivetrain;
pub mod fuel_converter;
pub mod reversible_energy_storage;

// vehicle model
pub mod vehicle_model;

// traits and utilities
pub mod powertrain_traits;
pub mod vehicle_utils;

#[cfg(test)]
/// Module containing tests for consists.
pub mod tests;

pub use bev::BatteryElectricVehicle;
pub use conv::ConventionalLoco;
pub use electric_drivetrain::ElectricDrivetrain;
pub use fuel_converter::FuelConverter;
pub use hev::HybridElectricVehicle;
pub use powertrain_traits::{ElectricMachine, Mass};
pub use reversible_energy_storage::ReversibleEnergyStorage;
pub use vehicle_utils::LocoTrait;
