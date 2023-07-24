#![allow(clippy::field_reassign_with_default)]
// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]

//! Crate containing models for second-by-second fuel and energy consumption of simulation
//! of locomotive consists comprising collections of individual locomotives, which comprise
//! various powertrain components (engine, generator/alternator, battery, and electric drivetrain)
//! -- all connected to a detailed train model.  
//! # Features:
//! - pyo3: enable this feature to expose FASTSim structs, methods, and functions to Python

#[macro_use]
pub mod macros;

pub mod drive_cycle;
pub mod imports;
pub mod prelude;
pub mod si;
pub mod simdrive;
pub mod traits;
pub mod uc;
pub mod utils;
pub mod vehicle;

// these might not get used
pub mod combo_error;

#[cfg(feature = "pyo3")]
pub mod pyo3;
