// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]
//! Module containing submodules for vehicle and powertrain models

pub(crate) use crate::imports::*;

// powertrain types
pub mod bev;
pub mod conv;
pub mod hev;

// powertrain components
pub mod powertrain;

// vehicle model
pub mod chassis;
pub mod powertrain_type;
pub mod traits;
pub mod vehicle_model;

pub use bev::BatteryElectricVehicle;
pub use chassis::Chassis;
pub use conv::ConventionalVehicle;
pub use hev::HybridElectricVehicle;
pub use powertrain::electric_machine::ElectricMachine;
pub use powertrain::fuel_converter::FuelConverter;
pub use powertrain::fuel_storage::FuelStorage;
pub use powertrain::reversible_energy_storage::ReversibleEnergyStorage;
pub use powertrain::traits::Powertrain;
pub use powertrain_type::PowertrainType;
pub use traits::*;
pub use vehicle_model::{Vehicle, VehicleState};
