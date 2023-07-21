// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]
//! Module containing submodules for consists, locomotives, and powertrain models

use crate::imports::*;

pub mod battery_electric_loco;
pub mod conventional_loco;
pub mod electric_drivetrain;
pub mod fuel_converter;
pub mod hybrid_loco;
pub mod powertrain_traits;
pub mod reversible_energy_storage;
pub mod vehicle_utils;

#[cfg(test)]
/// Module containing tests for consists.
pub mod tests;
