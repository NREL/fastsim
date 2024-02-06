//! # Crate features
//! * **full** - When enabled (which is default), include additional capabilities that
//!   require additional dependencies
//! * **resources** - When enabled (which is triggered by enabling full (thus default)
//!   or enabling this feature directly), compiles commonly used resources (e.g.
//!   standard drive cycles) for faster access.

use fastsim_core::*;
use pyo3imports::*;
use std::collections::HashMap;

/// Function for adding Rust structs as Python Classes
#[pymodule]
fn fastsimrust(py: Python, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "full")]
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
    m.add_class::<vehicle_import::OtherVehicleInputs>()?;
    m.add_class::<simdrivelabel::LabelFe>()?;
    m.add_class::<simdrivelabel::LabelFePHEV>()?;
    m.add_class::<simdrivelabel::PHEVCycleCalc>()?;
    m.add_class::<simdrive::simdrive_iter::SimDriveVec>()?;
    m.add_class::<MiscFns>()?;
    cycle::register(py, m)?;
    Ok(())
}

#[pyclass]
/// Miscellaneous functions not otherwise available
struct MiscFns;

#[pymethods]
impl MiscFns {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
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

    #[staticmethod]
    fn make_accel_trace() -> cycle::RustCycle {
        simdrivelabel::make_accel_trace()
    }

    #[cfg(feature = "full")]
    #[staticmethod]
    fn get_net_accel(
        sd_accel: &mut simdrive::RustSimDrive,
        scenario_name: &str,
    ) -> anyhow::Result<f64> {
        simdrivelabel::get_net_accel(sd_accel, scenario_name)
    }

    #[cfg(feature = "full")]
    #[staticmethod]
    fn get_label_fe(
        veh: &vehicle::RustVehicle,
        full_detail: Option<bool>,
        verbose: Option<bool>,
    ) -> anyhow::Result<(
        simdrivelabel::LabelFe,
        Option<HashMap<&str, simdrive::RustSimDrive>>,
    )> {
        simdrivelabel::get_label_fe(veh, full_detail, verbose)
    }

    #[staticmethod]
    fn get_label_fe_phev_py(
        veh: &vehicle::RustVehicle,
        sd: HashMap<&str, simdrive::RustSimDrive>,
        long_params: &params::RustLongParams,
        adj_params: &params::AdjCoef,
        sim_params: &simdrive::RustSimDriveParams,
        props: &params::RustPhysicalProperties,
    ) -> anyhow::Result<simdrivelabel::LabelFePHEV> {
        let mut sd_mut = HashMap::new();
        for (key, value) in sd {
            sd_mut.insert(key, value);
        }
        simdrivelabel::get_label_fe_phev(
            veh,
            &mut sd_mut,
            long_params,
            adj_params,
            sim_params,
            props,
        )
    }

    #[cfg(feature = "vehicle-import")]
    #[staticmethod]
    fn get_options_for_year_make_model(
        year: &str,
        make: &str,
        model: &str,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> anyhow::Result<Vec<vehicle_import::VehicleDataFE>> {
        vehicle_import::get_options_for_year_make_model(year, make, model, cache_url, data_dir)
    }

    #[cfg(feature = "vehicle-import")]
    #[staticmethod]
    fn vehicle_import_by_id_and_year(
        vehicle_id: i32,
        year: u32,
        other_inputs: &vehicle_import::OtherVehicleInputs,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> anyhow::Result<vehicle::RustVehicle> {
        vehicle_import::vehicle_import_by_id_and_year(
            vehicle_id,
            year,
            other_inputs,
            cache_url,
            data_dir,
        )
    }

    #[cfg(feature = "vehicle-import")]
    #[staticmethod]
    fn import_all_vehicles(
        year: u32,
        make: &str,
        model: &str,
        other_inputs: &vehicle_import::OtherVehicleInputs,
        cache_url: Option<String>,
        data_dir: Option<String>,
    ) -> anyhow::Result<Vec<vehicle::RustVehicle>> {
        vehicle_import::import_all_vehicles(year, make, model, other_inputs, cache_url, data_dir)
    }

    #[staticmethod]
    fn enabled_features() -> Vec<String> {
        enabled_features()
    }
}
