//! Module containing vehicle struct and related functions.

use lazy_static::lazy_static;
use regex::Regex;
use validator::Validate;

// local
use crate::imports::*;
use crate::params::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
pub use legacy_vehicle::*;

// legacy modules
pub mod legacy_vehicle;
pub mod vehicle_thermal;
pub mod vehicle_utils;

// hierarchical modules
pub mod powertrain;
pub mod vehicle_core;
