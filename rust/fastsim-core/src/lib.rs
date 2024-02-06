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
pub mod resources;
pub mod simdrivelabel;
pub mod thermal;
pub mod traits;
pub mod utils;
pub mod vehicle;
pub mod vehicle_import;
pub mod vehicle_thermal;
pub mod vehicle_utils;

#[cfg(feature = "dev-proc-macros")]
pub use dev_proc_macros as proc_macros;
#[cfg(not(feature = "dev-proc-macros"))]
pub use fastsim_proc_macros as proc_macros;

#[cfg_attr(feature = "pyo3", pyo3imports::pyfunction)]
#[allow(clippy::vec_init_then_push)]
pub fn enabled_features() -> Vec<String> {
    let mut enabled = vec![];

    #[cfg(feature = "full")]
    enabled.push("full".into());

    #[cfg(feature = "resources")]
    enabled.push("resources".into());

    #[cfg(feature = "validation")]
    enabled.push("validation".into());

    enabled
}

#[cfg(feature = "pyo3")]
pub mod pyfunctions {
    use super::*;
    use pyo3imports::*;
    use simdrivelabel::*;
    use vehicle_import::*;
    use vehicle_utils::*;

    pub fn add_pyfunctions(m: &PyModule) -> PyResult<()> {
        #[cfg(feature = "full")]
        m.add_function(wrap_pyfunction!(abc_to_drag_coeffs, m)?)?;
        m.add_function(wrap_pyfunction!(make_accel_trace_py, m)?)?;
        m.add_function(wrap_pyfunction!(get_net_accel_py, m)?)?;
        #[cfg(feature = "full")]
        m.add_function(wrap_pyfunction!(get_label_fe_py, m)?)?;
        m.add_function(wrap_pyfunction!(get_label_fe_phev_py, m)?)?;
        #[cfg(feature = "full")]
        m.add_function(wrap_pyfunction!(get_label_fe_conv_py, m)?)?;
        #[cfg(feature = "vehicle-import")]
        m.add_function(wrap_pyfunction!(get_options_for_year_make_model, m)?)?;
        #[cfg(feature = "vehicle-import")]
        m.add_function(wrap_pyfunction!(vehicle_import_by_id_and_year, m)?)?;
        #[cfg(feature = "vehicle-import")]
        m.add_function(wrap_pyfunction!(import_all_vehicles, m)?)?;

        m.add_function(wrap_pyfunction!(enabled_features, m)?)?;
        Ok(())
    }
}
