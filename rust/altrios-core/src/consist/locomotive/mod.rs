//! Module containing models for consists, locomotives, and powertrain components

pub mod battery_electric_loco;
pub mod conventional_loco;
pub mod hybrid_loco;
pub mod loco_sim;
pub mod locomotive_model;
pub mod powertrain;
pub use locomotive_model::*;
#[cfg(test)]
pub mod tests;

pub use super::*;
pub use crate::consist::locomotive::battery_electric_loco::BatteryElectricLoco;
pub use crate::consist::locomotive::conventional_loco::ConventionalLoco;
pub use crate::consist::locomotive::hybrid_loco::HybridLoco;
use crate::consist::LocoTrait;
use crate::imports::*;
use proc_macros::{altrios_api, HistoryVec};

use crate::consist::locomotive::powertrain::electric_drivetrain::ElectricDrivetrain;
use crate::consist::locomotive::powertrain::fuel_converter::FuelConverter;
use crate::consist::locomotive::powertrain::generator::Generator;
use crate::consist::locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage;

use anyhow::Result;
