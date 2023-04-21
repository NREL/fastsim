//! Module containing vehicle struct and related functions.

use lazy_static::lazy_static;
use regex::Regex;
use validator::Validate;

// local
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::{add_pyo3_api, ApproxEq};
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
pub use legacy::*;

// modules
pub mod legacy;
pub mod powertrain;
pub mod vehicle_thermal;
pub mod vehicle_utils;
