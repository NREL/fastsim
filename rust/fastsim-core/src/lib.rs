// This needs to be a square logo to avoid stretching, and can have transparency
#![doc(html_logo_url = "https://www.nrel.gov/transportation/assets/images/icon-fastsim.jpg")]
//! Documentation for the Rust backend of the Future Automotive Systems Technology Simulator (FASTSim).

//! # Overview
//! FASTSim provides a simple way to compare powertrains and estimate the impact of technology
//! improvements on light-, medium-, and heavy-duty vehicle efficiency, performance, cost, and battery life.  
//! More information here: <https://www.nrel.gov/transportation/fastsim.html>
//!
//! # Crate features
//! * **full** - When enabled (which is default), include additional capabilities that
//!   require additional dependencies
//! * **resources** - When enabled (which is triggered by enabling full (thus default)
//!   or enabling this feature directly), compiles commonly used resources (e.g.
//!   standard drive cycles) for faster access.
//!
//! # Python Examples
//! ```python
//! import fastsim
//!
//! ## Load drive cycle by name
//! cyc_py = fastsim.cycle.Cycle.from_file("udds")
//! cyc_rust = cyc_py.to_rust()
//!
//! ## Load vehicle using database vehicle ID number
//! vnum = 1  
//! veh_py = fastsim.vehicle.Vehicle.from_vehdb(vnum)
//! veh_rust = veh_py.to_rust()
//!
//! ## Simulate
//! sd = fastsim.RustSimDrive(cyc_rust, veh_rust)
//! sd.sim_drive()
//! ```

extern crate ndarray;

#[macro_use]
pub mod macros;
pub mod air;
pub mod cycle;
pub mod imports;
pub mod params;
pub mod pyo3imports;
pub mod simdrive;
pub use simdrive::simdrive_impl;
pub mod simdrivelabel;
pub mod thermal;
pub mod traits;
pub mod utils;
pub mod vehicle;
pub mod vehicle_import;
pub mod vehicle_thermal;
pub mod vehicle_utils;

pub use fastsim_proc_macros as proc_macros;

#[cfg_attr(feature = "pyo3", pyo3imports::pyfunction)]
#[allow(clippy::vec_init_then_push)]
pub fn enabled_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut enabled = vec![];

    #[cfg(feature = "default")]
    enabled.push("default".into());

    #[cfg(feature = "bincode")]
    enabled.push("bincode".into());

    #[cfg(feature = "logging")]
    enabled.push("logging".into());

    #[cfg(feature = "resources")]
    enabled.push("resources".into());

    #[cfg(feature = "simdrivelabel")]
    enabled.push("simdrivelabel".into());

    #[cfg(feature = "validation")]
    enabled.push("validation".into());

    #[cfg(feature = "vehicle-import")]
    enabled.push("vehicle-import".into());

    enabled
}
