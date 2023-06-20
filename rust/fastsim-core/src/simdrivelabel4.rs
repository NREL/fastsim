// Modify the existing code from 1-125 to add pyo3 

//! Module containing classes and methods for calculating label fuel economy.

use ndarray::Array;
use std::collections::HashMap;

// crate local
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;

use crate::proc_macros::{add_pyo3_api, ApproxEq};
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;

use crate::simdrive::{RustSimDrive, RustSimDriveParams};
use crate::vehicle;


#[pyclass]
#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy values
pub struct LabelFe {
    pub veh: vehicle::RustVehicle,
    pub adj_params: AdjCoef,
    pub lab_udds_mpgge: f64,
    pub lab_hwy_mpgge: f64,
    pub lab_comb_mpgge: f64,
    pub lab_udds_kwh_per_mi: f64,
    pub lab_hwy_kwh_per_mi: f64,
    pub lab_comb_kwh_per_mi: f64,
    pub adj_udds_mpgge: f64,
    pub adj_hwy_mpgge: f64,
    pub adj_comb_mpgge: f64,
    pub adj_udds_kwh_per_mi: f64,
    pub adj_hwy_kwh_per_mi: f64,
    pub adj_comb_kwh_per_mi: f64,
    pub adj_udds_ess_kwh_per_mi: f64,
    pub adj_hwy_ess_kwh_per_mi: f64,
    pub adj_comb_ess_kwh_per_mi: f64,
    /// Range for combined city/highway
    pub net_range_miles: f64,
    /// Utility factor
    pub uf: f64,
    pub net_accel: f64,
    pub res_found: String,
    pub phev_calcs: Option<LabelFePHEV>,
    pub adj_cs_comb_mpgge: Option<f64>,
    pub adj_cd_comb_mpgge: Option<f64>,
    pub net_phev_cd_miles: Option<f64>,
    pub trace_miss_speed_mph: f64,
}

#[pyclass]
#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: f64,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[pyclass]
#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy calculations for a specific cycle of a PHEV vehicle
pub struct PHEVCycleCalc {
    /// Charge depletion battery kW-hr
    pub cd_ess_kwh: f64,
    pub cd_ess_kwh_per_mi: f64,
    /// Charge depletion fuel gallons
    pub cd_fs_gal: f64,
    pub cd_fs_kwh: f64,
    pub cd_mpg: f64,
    /// Number of cycles in charge depletion mode, up to transition
    pub cd_cycs: f64,
    pub cd_miles: f64,
    pub cd_lab_mpg: f64,
    pub cd_adj_mpg: f64,
    /// Fraction of transition cycles spent in charge depletion
    pub cd_frac_in_trans: f64,
    /// SOC change during 1 cycle
    pub trans_init_soc: f64,
    /// charge depletion battery kW-hr
    pub trans_ess_kwh: f64,
    pub trans_ess_kwh_per_mi: f64,
    pub trans_fs_gal: f64,
    pub trans_fs_kwh: f64,
    /// charge sustaining battery kW-hr
    pub cs_ess_kwh: f64,
    pub cs_ess_kwh_per_mi: f64,
    /// charge sustaining fuel gallons
    pub cs_fs_gal: f64,
    pub cs_fs_kwh: f64,
    pub cs_mpg: f64,
    pub lab_mpgge: f64,
    pub lab_kwh_per_mi: f64,
    pub lab_uf: f64,
    pub lab_uf_gpm: Array1<f64>,
    pub lab_iter_uf: Array1<f64>,
    pub lab_iter_uf_kwh_per_mi: Array1<f64>,
    pub lab_iter_kwh_per_mi: Array1<f64>,
    pub adj_iter_mpgge: Array1<f64>,
    pub adj_iter_kwh_per_mi: Array1<f64>,
    pub adj_iter_cd_miles: Array1<f64>,
    pub adj_iter_uf: Array1<f64>,
    pub adj_iter_uf_gpm: Vec<f64>,
    pub adj_iter_uf_kwh_per_mi: Array1<f64>,
    pub adj_cd_miles: f64,
    pub adj_cd_mpgge: f64,
    pub adj_cs_mpgge: f64,
    pub adj_uf: f64,
    pub adj_mpgge: f64,
    pub adj_kwh_per_mi: f64,
    pub adj_ess_kwh_per_mi: f64,
    pub delta_soc: f64,
    /// Total number of miles in charge depletion mode, assuming constant kWh_per_mi
    pub total_cd_miles: f64,
}


#[cfg(feature = "pyo3")]
#[pymodule]
fn label_fuel_economy(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the make_accel_trace_py function
    #[pyfn(m, "make_accel_trace")]
    fn make_accel_trace_py() -> PyResult<RustCycle> {
        let accel_cyc_secs = Array::range(0., 300., 0.1);
        let mut accel_cyc_mps = Array::ones(accel_cyc_secs.len()) * 90.0 / MPH_PER_MPS;
        accel_cyc_mps[0] = 0.0;

        let rust_cycle = RustCycle::new(
            accel_cyc_secs.to_vec(),
            accel_cyc_mps.to_vec(),
            Array::zeros(accel_cyc_secs.len()).to_vec(),
            Array::zeros(accel_cyc_secs.len()).to_vec(),
            String::from("accel"),
        );

        Ok(rust_cycle)
    }

    // Add the make_accel_trace_py function to the module
    m.add_function(wrap_pyfunction!(make_accel_trace_py, m)?)?;

    // Return Ok(()) to indicate successful execution
    Ok(())
}
