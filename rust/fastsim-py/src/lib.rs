//! # Crate features
//! * **full** - When enabled (which is default), include additional capabilities that
//!   require additional dependencies
//! * **resources** - When enabled (which is triggered by enabling full (thus default)
//!   or enabling this feature directly), compiles commonly used resources (e.g.
//!   standard drive cycles) for faster access.

use fastsim_core::*;
use pyo3imports::*;

// re-expose pyfunctions for default feature
// pyo3 0.15 requires pyfunctions be defined in the same crate
#[cfg(feature = "default")]
mod default_pyfunctions {
    use super::*;

    #[pyfunction]
    fn abc_to_drag_coeffs(
        veh: &mut vehicle::RustVehicle,
        a_lbf: f64,
        b_lbf__mph: f64,
        c_lbf__mph2: f64,
        custom_rho: Option<bool>,
        custom_rho_temp_degC: Option<f64>,
        custom_rho_elevation_m: Option<f64>,
        simdrive_optimize: Option<bool>,
        _show_plots: Option<bool>,
    ) -> (f64, f64) {
        vehicle_utils::abc_to_drag_coeffs(
            veh,
            a_lbf,
            b_lbf__mph,
            c_lbf__mph2,
            custom_rho,
            custom_rho_temp_degC,
            custom_rho_elevation_m,
            simdrive_optimize,
            _show_plots,
        )
    }
}
#[cfg(feature = "default")]
use default_pyfunctions::*;

// re-expose pyfunctions for vehicle-import feature
// pyo3 0.15 requires pyfunctions be defined in the same crate
#[cfg(feature = "vehicle-import")]
mod vehicle_import_pyfunctions {
    use super::*;

    #[pyfunction]
    fn get_options_for_year_make_model(
        year: &str,
        make: &str,
        model: &str,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> PyResult<Vec<vehicle_import::VehicleDataFE>> {
        Ok(vehicle_import::get_options_for_year_make_model(
            year, make, model, cache_url, data_dir,
        )?)
    }

    #[pyfunction]
    fn get_vehicle_data_for_id(
        id: i32,
        year: &str,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> PyResult<vehicle_import::VehicleDataFE> {
        Ok(vehicle_import::get_vehicle_data_for_id(
            id, year, cache_url, data_dir,
        )?)
    }

    #[pyfunction]
    fn vehicle_import_by_id_and_year(
        vehicle_id: i32,
        year: u32,
        other_inputs: &vehicle_import::OtherVehicleInputs,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> PyResult<vehicle::RustVehicle> {
        Ok(vehicle_import::vehicle_import_by_id_and_year(
            vehicle_id,
            year,
            other_inputs,
            cache_url,
            data_dir,
        )?)
    }

    #[pyfunction]
    fn import_all_vehicles(
        year: u32,
        make: &str,
        model: &str,
        other_inputs: &vehicle_import::OtherVehicleInputs,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> PyResult<Vec<vehicle::RustVehicle>> {
        Ok(vehicle_import::import_all_vehicles(
            year,
            make,
            model,
            other_inputs,
            cache_url,
            data_dir,
        )?)
    }
}
#[cfg(feature = "vehicle-import")]
use vehicle_import_pyfunctions::*;

// re-expose pyfunctions for default feature
// pyo3 0.15 requires pyfunctions be defined in the same crate
#[cfg(feature = "simdrivelabel")]
mod simdrivelabel_pyfunctions {
    use super::*;

    #[pyfunction]
    fn make_accel_trace() -> cycle::RustCycle {
        simdrivelabel::make_accel_trace_py()
    }

    #[pyfunction]
    fn get_net_accel(sd_accel: &mut simdrive::RustSimDrive, scenario_name: &str) -> PyResult<f64> {
        Ok(simdrivelabel::get_net_accel_py(sd_accel, scenario_name)?)
    }

    #[pyfunction]
    fn get_label_fe(
        veh: &vehicle::RustVehicle,
        full_detail: Option<bool>,
        verbose: Option<bool>,
    ) -> PyResult<(
        simdrivelabel::LabelFe,
        Option<std::collections::HashMap<&str, simdrive::RustSimDrive>>,
    )> {
        Ok(simdrivelabel::get_label_fe_py(veh, full_detail, verbose)?)
    }

    #[pyfunction]
    fn get_label_fe_phev(
        veh: &vehicle::RustVehicle,
        full_detail: Option<bool>,
        verbose: Option<bool>,
    ) -> PyResult<(
        simdrivelabel::LabelFe,
        Option<std::collections::HashMap<&str, simdrive::RustSimDrive>>,
    )> {
        Ok(simdrivelabel::get_label_fe_py(veh, full_detail, verbose)?)
    }
}
#[cfg(feature = "simdrivelabel")]
use simdrivelabel_pyfunctions::*;

// re-expose other pyfunctions
// pyo3 0.15 requires pyfunctions be defined in the same crate
#[pyfunction]
fn enabled_features() -> Vec<String> {
    fastsim_core::enabled_features()
}

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "logging")]
    pyo3_log::init();
    m.add_class::<cycle::RustCycle>()?;
    m.add_class::<vehicle::RustVehicle>()?;
    m.add_class::<params::RustPhysicalProperties>()?;
    m.add_class::<params::AdjCoef>()?;
    m.add_class::<params::RustLongParams>()?;
    m.add_class::<utils::Pyo3ArrayU32>()?;
    m.add_class::<utils::Pyo3ArrayF64>()?;
    m.add_class::<utils::Pyo3ArrayBool>()?;
    m.add_class::<utils::Pyo3VecF64>()?;
    m.add_class::<simdrive::RustSimDriveParams>()?;
    m.add_class::<simdrive::RustSimDrive>()?;
    m.add_class::<thermal::SimDriveHot>()?;
    m.add_class::<vehicle_thermal::VehicleThermal>()?;
    m.add_class::<thermal::ThermalState>()?;
    m.add_class::<vehicle_thermal::HVACModel>()?;
    m.add_class::<simdrive::simdrive_iter::SimDriveVec>()?;

    cycle::register(py, m)?;

    // Features
    #[cfg(feature = "default")]
    m.add_function(wrap_pyfunction!(abc_to_drag_coeffs, m)?)?;
    #[cfg(feature = "simdrivelabel")]
    {
        m.add_class::<simdrivelabel::LabelFe>()?;
        m.add_class::<simdrivelabel::LabelFePHEV>()?;
        m.add_class::<simdrivelabel::PHEVCycleCalc>()?;
        m.add_function(wrap_pyfunction!(make_accel_trace, m)?)?;
        m.add_function(wrap_pyfunction!(get_net_accel, m)?)?;
        m.add_function(wrap_pyfunction!(get_label_fe, m)?)?;
        m.add_function(wrap_pyfunction!(get_label_fe_phev, m)?)?;
    }
    #[cfg(feature = "vehicle-import")]
    {
        m.add_class::<vehicle_import::OtherVehicleInputs>()?;
        m.add_function(wrap_pyfunction!(get_options_for_year_make_model, m)?)?;
        m.add_function(wrap_pyfunction!(get_vehicle_data_for_id, m)?)?;
        m.add_function(wrap_pyfunction!(vehicle_import_by_id_and_year, m)?)?;
        m.add_function(wrap_pyfunction!(import_all_vehicles, m)?)?;
    }
    // Function to check what features are enabled from Python
    m.add_function(wrap_pyfunction!(enabled_features, m)?)?;

    Ok(())
}
