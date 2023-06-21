// Modify the existing code from 1-56 to add pyo3 

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

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
#[add_pyo3_api]
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
    pub net_range_miles: f64,
    pub uf: f64,
    pub net_accel: f64,
    pub res_found: String,
    pub phev_calcs: Option<LabelFePHEV>,
    pub adj_cs_comb_mpgge: Option<f64>,
    pub adj_cd_comb_mpgge: Option<f64>,
    pub net_phev_cd_miles: Option<f64>,
    pub trace_miss_speed_mph: f64,
}

impl LabelFe {
    #[getter]
    pub fn get_veh(&self) -> &vehicle::RustVehicle {
        &self.veh
    }

    #[setter]
    pub fn set_veh_py(&mut self, new_veh: vehicle::RustVehicle) {
        self.veh = new_veh;
    }

    #[getter]
    pub fn get_adj_params(&self) -> &AdjCoef {
        &self.adj_params
    }

    #[setter]
    pub fn set_adj_params_py(&mut self, new_params: AdjCoef) {
        self.adj_params = new_params;
    }

    #[getter]
    pub fn get_lab_udds_mpgge(&self) -> f64 {
        self.lab_udds_mpgge
    }

    #[setter]
    pub fn set_lab_udds_mpgge_py(&mut self, new_value: f64) {
        self.lab_udds_mpgge = new_value;
    }

    #[getter]
    pub fn get_lab_hwy_mpgge(&self) -> f64 {
        self.lab_hwy_mpgge
    }

    #[setter]
    pub fn set_lab_hwy_mpgge_py(&mut self, new_value: f64) {
        self.lab_hwy_mpgge = new_value;
    }

    #[getter]
    pub fn get_lab_comb_mpgge(&self) -> f64 {
        self.lab_comb_mpgge
    }

    #[setter]
    pub fn set_lab_comb_mpgge_py(&mut self, new_value: f64) {
        self.lab_comb_mpgge = new_value;
    }

    #[getter]
    pub fn get_lab_udds_kwh_per_mi(&self) -> f64 {
        self.lab_udds_kwh_per_mi
    }

    #[setter]
    pub fn set_lab_udds_kwh_per_mi_py(&mut self, new_value: f64) {
        self.lab_udds_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_lab_hwy_kwh_per_mi(&self) -> f64 {
        self.lab_hwy_kwh_per_mi
    }

    #[setter]
    pub fn set_lab_hwy_kwh_per_mi_py(&mut self, new_value: f64) {
        self.lab_hwy_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_lab_comb_kwh_per_mi(&self) -> f64 {
        self.lab_comb_kwh_per_mi
    }

    #[setter]
    pub fn set_lab_comb_kwh_per_mi_py(&mut self, new_value: f64) {
        self.lab_comb_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_udds_mpgge(&self) -> f64 {
        self.adj_udds_mpgge
    }

    #[setter]
    pub fn set_adj_udds_mpgge_py(&mut self, new_value: f64) {
        self.adj_udds_mpgge = new_value;
    }

    #[getter]
    pub fn get_adj_hwy_mpgge(&self) -> f64 {
        self.adj_hwy_mpgge
    }

    #[setter]
    pub fn set_adj_hwy_mpgge_py(&mut self, new_value: f64) {
        self.adj_hwy_mpgge = new_value;
    }

    #[getter]
    pub fn get_adj_comb_mpgge(&self) -> f64 {
        self.adj_comb_mpgge
    }

    #[setter]
    pub fn set_adj_comb_mpgge_py(&mut self, new_value: f64) {
        self.adj_comb_mpgge = new_value;
    }

    #[getter]
    pub fn get_adj_udds_kwh_per_mi(&self) -> f64 {
        self.adj_udds_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_udds_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_udds_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_hwy_kwh_per_mi(&self) -> f64 {
        self.adj_hwy_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_hwy_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_hwy_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_comb_kwh_per_mi(&self) -> f64 {
        self.adj_comb_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_comb_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_comb_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_udds_ess_kwh_per_mi(&self) -> f64 {
        self.adj_udds_ess_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_udds_ess_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_udds_ess_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_hwy_ess_kwh_per_mi(&self) -> f64 {
        self.adj_hwy_ess_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_hwy_ess_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_hwy_ess_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_adj_comb_ess_kwh_per_mi(&self) -> f64 {
        self.adj_comb_ess_kwh_per_mi
    }

    #[setter]
    pub fn set_adj_comb_ess_kwh_per_mi_py(&mut self, new_value: f64) {
        self.adj_comb_ess_kwh_per_mi = new_value;
    }

    #[getter]
    pub fn get_net_range_miles(&self) -> f64 {
        self.net_range_miles
    }

    #[setter]
    pub fn set_net_range_miles_py(&mut self, new_value: f64) {
        self.net_range_miles = new_value;
    }

    #[getter]
    pub fn get_uf(&self) -> f64 {
        self.uf
    }

    #[setter]
    pub fn set_uf_py(&mut self, new_value: f64) {
        self.uf = new_value;
    }

    #[getter]
    pub fn get_net_accel(&self) -> f64 {
        self.net_accel
    }

    #[setter]
    pub fn set_net_accel_py(&mut self, new_value: f64) {
        self.net_accel = new_value;
    }

    #[getter]
    pub fn get_res_found(&self) -> &str {
        &self.res_found
    }

    #[setter]
    pub fn set_res_found_py(&mut self, new_value: String) {
        self.res_found = new_value;
    }

    #[getter]
    pub fn get_phev_calcs(&self) -> Option<&LabelFePHEV> {
        self.phev_calcs.as_ref()
    }

    #[setter]
    pub fn set_phev_calcs_py(&mut self, new_value: Option<LabelFePHEV>) {
        self.phev_calcs = new_value;
    }

    #[getter]
    pub fn get_adj_cs_comb_mpgge(&self) -> Option<f64> {
        self.adj_cs_comb_mpgge
    }

    #[setter]
    pub fn set_adj_cs_comb_mpgge_py(&mut self, new_value: Option<f64>) {
        self.adj_cs_comb_mpgge = new_value;
    }

    #[getter]
    pub fn get_adj_cd_comb_mpgge(&self) -> Option<f64> {
        self.adj_cd_comb_mpgge
    }

    #[setter]
    pub fn set_adj_cd_comb_mpgge_py(&mut self, new_value: Option<f64>) {
        self.adj_cd_comb_mpgge = new_value;
    }

    #[getter]
    pub fn get_net_phev_cd_miles(&self) -> Option<f64> {
        self.net_phev_cd_miles
    }

    #[setter]
    pub fn set_net_phev_cd_miles_py(&mut self, new_value: Option<f64>) {
        self.net_phev_cd_miles = new_value;
    }

    #[getter]
    pub fn get_trace_miss_speed_mph(&self) -> f64 {
        self.trace_miss_speed_mph
    }

    #[setter]
    pub fn set_trace_miss_speed_mph_py(&mut self, new_value: f64) {
        self.trace_miss_speed_mph = new_value;
    }
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
#[add_pyo3_api]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: f64,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[pymethods]
impl LabelFePHEV {
    #[getter]
    pub fn get_regen_soc_buffer(&self) -> f64 {
        self.regen_soc_buffer() 
    }

    #[setter]
    pub fn set_regen_soc_buffer_py(&mut self, new_value: f64) {
        self.regen_soc_buffer = new_value;
    }

    #[getter]
    pub fn get_udds(&self) -> &PHEVCycleCalc {
        &self.udds
    }

    #[setter]
    pub fn set_udds_py(&mut self, new_value: PHEVCycleCalc) {
        self.udds = new_value;
    }

    #[getter]
    pub fn get_hwy(&self) -> &PHEVCycleCalc {
        &self.hwy
    }

    #[setter]
    pub fn set_hwy_py(&mut self, new_value: PHEVCycleCalc) {
        self.hwy = new_value;
    }
}

