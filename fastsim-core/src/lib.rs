#![allow(clippy::field_reassign_with_default)]
// TODO: uncomment when docs are somewhat mature to check for missing docs
// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]

//! Crate containing models for second-by-second fuel and energy consumption of simulation
//! of vehicles
//! # Feature Flags
#![doc = document_features::document_features!()]

#[macro_use]
pub mod macros;

pub mod air_properties;
pub mod drive_cycle;
pub mod imports;
pub mod prelude;
// #[cfg(feature = "pyo3")] -- feature gate provided inside module
pub mod pyo3;
pub mod resources;
pub mod si;
pub mod simdrive;
pub mod traits;
pub mod uc;
pub mod utils;
pub mod vehicle;

/// List enabled features
#[cfg_attr(feature = "pyo3", imports::pyfunction)]
pub fn enabled_features() -> Vec<String> {
    vec![
        #[cfg(feature = "default")]
        "default".into(),
        #[cfg(feature = "resources")]
        "resources".into(),
        #[cfg(feature = "web")]
        "web".into(),
        #[cfg(feature = "serde-default")]
        "serde-default".into(),
        #[cfg(feature = "bincode")]
        "bincode".into(),
        #[cfg(feature = "csv")]
        "csv".into(),
        #[cfg(feature = "json")]
        "json".into(),
        #[cfg(feature = "toml")]
        "toml".into(),
        #[cfg(feature = "yaml")]
        "yaml".into(),
    ]
}
