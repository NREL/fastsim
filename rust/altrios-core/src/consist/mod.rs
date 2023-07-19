// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]
//! Module containing submodules for consists, locomotives, and powertrain models

pub mod consist_model;
/// Module containing structs for simulating standalone consist.
pub mod consist_sim;
/// Module containing structs for locomotive models.
pub mod locomotive;
pub use consist_model::*;

use crate::imports::*;

pub mod consist_utils;
pub use consist_utils::*;
#[cfg(test)]
/// Module containing tests for consists.
pub mod tests;

pub(crate) use self::locomotive::powertrain::powertrain_traits::*;
use self::locomotive::BatteryElectricLoco;
use crate::consist::locomotive::{LocoType, Locomotive};
