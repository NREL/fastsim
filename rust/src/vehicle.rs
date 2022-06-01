//! Module for vehicle attributes and related functions and structs.

extern crate ndarray;
use ndarray::{Array, Array1};
extern crate pyo3;
use pyo3::prelude::*;
// extern crate itertools;
// use itertools::Itertools;
// use numpy::pyo3::Python;
// use numpy::ndarray::array;
// use numpy::{ToPyArray, PyArray};

// local
use super::params::*;
use super::utils::*;

pub const CONV: &str = "Conv";
pub const HEV: &str = "HEV";
pub const PHEV: &str = "PHEV";
pub const BEV: &str = "BEV";
pub const VEH_PT_TYPES: [&str; 4] = [CONV, HEV, PHEV, BEV];

pub const SI: &str = "SI";
pub const ATKINSON: &str = "Atkinson";
pub const DIESEL: &str = "Diesel";
pub const H2FC: &str = "H2FC";
pub const HD_DIESEL: &str = "HD_Diesel";

pub const FC_EFF_TYPES: [&str; 5] = [SI, ATKINSON, DIESEL, H2FC, HD_DIESEL];

#[pyclass]
#[derive(Debug, Clone)]
/// Struct containing vehicle attributes
pub struct RustVehicle {
    #[pyo3(get, set)]
    pub props: RustPhysicalProperties,
    #[pyo3(get, set)]
    pub scenario_name: String,
    #[pyo3(get, set)]
    pub selection: u32,
    #[pyo3(get, set)]
    pub veh_year: u32,
    #[pyo3(get, set)]
    pub veh_pt_type: String,
    #[pyo3(get, set)]
    pub drag_coef: f64,
    /// Frontal area \[m²\]
    #[pyo3(get, set)]
    pub frontal_area_m2: f64,
    #[pyo3(get, set)]
    pub glider_kg: f64,
    #[pyo3(get, set)]
    pub veh_cg_m: f64,
    #[pyo3(get, set)]
    pub drive_axle_weight_frac: f64,
    #[pyo3(get, set)]
    pub wheel_base_m: f64,
    #[pyo3(get, set)]
    pub cargo_kg: f64,
    #[pyo3(get, set)]
    pub veh_override_kg: f64,
    #[pyo3(get, set)]
    pub comp_mass_multiplier: f64,
    #[pyo3(get, set)]
    pub fs_max_kw: f64,
    #[pyo3(get, set)]
    pub fs_secs_to_peak_pwr: f64,
    #[pyo3(get, set)]
    pub fs_kwh: f64,
    #[pyo3(get, set)]
    pub fs_kwh_per_kg: f64,
    #[pyo3(get, set)]
    pub fc_max_kw: f64,
    pub fc_pwr_out_perc: Array1<f64>,
    pub fc_eff_map: Array1<f64>,
    #[pyo3(get, set)]
    pub fc_eff_type: String,
    #[pyo3(get, set)]
    pub fc_sec_to_peak_pwr: f64,
    #[pyo3(get, set)]
    pub fc_base_kg: f64,
    #[pyo3(get, set)]
    pub fc_kw_per_kg: f64,
    #[pyo3(get, set)]
    pub min_fc_time_on: f64,
    #[pyo3(get, set)]
    pub idle_fc_kw: f64,
    #[pyo3(get, set)]
    pub mc_max_kw: f64,
    pub mc_pwr_out_perc: Array1<f64>,
    pub mc_eff_map: Array1<f64>,
    #[pyo3(get, set)]
    pub mc_sec_to_peak_pwr: f64,
    #[pyo3(get, set)]
    pub mc_pe_kg_per_kw: f64,
    #[pyo3(get, set)]
    pub mc_pe_base_kg: f64,
    #[pyo3(get, set)]
    pub ess_max_kw: f64,
    #[pyo3(get, set)]
    pub ess_max_kwh: f64,
    #[pyo3(get, set)]
    pub ess_kg_per_kwh: f64,
    #[pyo3(get, set)]
    pub ess_base_kg: f64,
    #[pyo3(get, set)]
    pub ess_round_trip_eff: f64,
    #[pyo3(get, set)]
    pub ess_life_coef_a: f64,
    #[pyo3(get, set)]
    pub ess_life_coef_b: f64,
    #[pyo3(get, set)]
    pub min_soc: f64,
    #[pyo3(get, set)]
    pub max_soc: f64,
    #[pyo3(get, set)]
    pub ess_dischg_to_fc_max_eff_perc: f64,
    #[pyo3(get, set)]
    pub ess_chg_to_fc_max_eff_perc: f64,
    #[pyo3(get, set)]
    pub wheel_inertia_kg_m2: f64,
    #[pyo3(get, set)]
    pub num_wheels: f64,
    #[pyo3(get, set)]
    pub wheel_rr_coef: f64,
    #[pyo3(get, set)]
    pub wheel_radius_m: f64,
    #[pyo3(get, set)]
    pub wheel_coef_of_fric: f64,
    #[pyo3(get, set)]
    pub max_accel_buffer_mph: f64,
    #[pyo3(get, set)]
    pub max_accel_buffer_perc_of_useable_soc: f64,
    #[pyo3(get, set)]
    pub perc_high_acc_buf: f64,
    #[pyo3(get, set)]
    pub mph_fc_on: f64,
    #[pyo3(get, set)]
    pub kw_demand_fc_on: f64,
    #[pyo3(get, set)]
    pub max_regen: f64,
    #[pyo3(get, set)]
    pub stop_start: bool,
    #[pyo3(get, set)]
    pub force_aux_on_fc: bool,
    #[pyo3(get, set)]
    pub alt_eff: f64,
    #[pyo3(get, set)]
    pub chg_eff: f64,
    #[pyo3(get, set)]
    pub aux_kw: f64,
    #[pyo3(get, set)]
    pub trans_kg: f64,
    #[pyo3(get, set)]
    pub trans_eff: f64,
    #[pyo3(get, set)]
    pub ess_to_fuel_ok_error: f64,
    #[pyo3(get, set)]
    pub small_motor_power_kw: f64,
    #[pyo3(get, set)]
    pub large_motor_power_kw: f64,
    // this and other fixed-size arrays can probably be vectors
    // without any performance penalty with the current implementation
    // of the functions in utils.rs
    pub fc_perc_out_array: Vec<f64>,
    #[pyo3(get, set)]
    pub regen_a: f64,
    #[pyo3(get, set)]
    pub regen_b: f64,
    #[pyo3(get, set)]
    pub charging_on: bool,
    #[pyo3(get, set)]
    pub no_elec_sys: bool,
    #[pyo3(get, set)]
    pub no_elec_aux: bool,
    pub max_roadway_chg_kw: Array1<f64>,
    pub input_kw_out_array: Array1<f64>,
    pub fc_kw_out_array: Vec<f64>,
    pub fc_eff_array: Vec<f64>,
    #[pyo3(get, set)]
    pub modern_max: f64,
    pub mc_eff_array: Array1<f64>,
    pub mc_kw_in_array: Vec<f64>,
    pub mc_kw_out_array: Vec<f64>,
    #[pyo3(get, set)]
    pub mc_max_elec_in_kw: f64,
    pub mc_full_eff_array: Vec<f64>,
    #[pyo3(get, set)]
    pub veh_kg: f64,
    #[pyo3(get, set)]
    pub max_trac_mps2: f64,
    #[pyo3(get, set)]
    pub ess_mass_kg: f64,
    #[pyo3(get, set)]
    pub mc_mass_kg: f64,
    #[pyo3(get, set)]
    pub fc_mass_kg: f64,
    #[pyo3(get, set)]
    pub fs_mass_kg: f64,
    pub mc_perc_out_array: Vec<f64>,
    // these probably don't need to be in rust
    pub val_udds_mpgge: f64,
    pub val_hwy_mpgge: f64,
    pub val_comb_mpgge: f64,
    pub val_udds_kwh_per_mile: f64,
    pub val_hwy_kwh_per_mile: f64,
    pub val_comb_kwh_per_mile: f64,
    pub val_cd_range_mi: f64,
    pub val_const65_mph_kwh_per_mile: f64,
    pub val_const60_mph_kwh_per_mile: f64,
    pub val_const55_mph_kwh_per_mile: f64,
    pub val_const45_mph_kwh_per_mile: f64,
    pub val_unadj_udds_kwh_per_mile: f64,
    pub val_unadj_hwy_kwh_per_mile: f64,
    pub val0_to60_mph: f64,
    pub val_ess_life_miles: f64,
    pub val_range_miles: f64,
    pub val_veh_base_cost: f64,
    pub val_msrp: f64,
}

/// RustVehicle rust methods
impl RustVehicle {
    pub fn max_regen_kwh(&self) -> f64 {
        0.5 * self.veh_kg * (27.0 * 27.0) / (3_600.0 * 1_000.0)
    }

    pub fn mc_peak_eff(&self) -> f64 {
        arrmax(&self.mc_full_eff_array)
    }

    pub fn set_mc_peak_eff_rust(&mut self, new_peak: f64) {
        let mc_max_eff = ndarrmax(&self.mc_eff_array);
        self.mc_eff_array *= new_peak / mc_max_eff;
        let mc_max_full_eff = arrmax(&self.mc_full_eff_array);
        self.mc_full_eff_array = self
            .mc_full_eff_array
            .iter()
            .map(|e: &f64| -> f64 { e * (new_peak / mc_max_full_eff) })
            .collect();
    }

    pub fn max_fc_eff_kw(&self) -> f64 {
        let fc_eff_arr_max_i =
            first_eq(&self.fc_eff_array, arrmax(&self.fc_eff_array)).unwrap_or(0);
        self.fc_kw_out_array[fc_eff_arr_max_i]
    }

    pub fn fc_peak_eff(&self) -> f64 {
        arrmax(&self.fc_eff_array)
    }

    pub fn set_fc_peak_eff_rust(&mut self, new_peak: f64) {
        let old_fc_peak_eff = self.fc_peak_eff();
        let multiplier = new_peak / old_fc_peak_eff;
        self.fc_eff_array = self
            .fc_eff_array
            .iter()
            .map(|eff: &f64| -> f64 { eff * multiplier })
            .collect();
        let new_fc_peak_eff = self.fc_peak_eff();
        let eff_map_multiplier = new_peak / new_fc_peak_eff;
        self.fc_eff_map = self
            .fc_eff_map
            .map(|eff| -> f64 { eff * eff_map_multiplier });
    }

    /// Sets derived parameters.
    /// Arguments:
    /// ----------
    /// mc_peak_eff_override: float (0, 1), if provided, overrides motor peak efficiency
    ///     with proportional scaling.  Default of -1 has no effect.  
    pub fn set_derived(&mut self) {
        if self.scenario_name != "Template Vehicle for setting up data types" {
            if self.veh_pt_type == BEV {
                assert!(
                    self.fs_max_kw == 0.0,
                    "max_fuel_stor_kw must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.fs_kwh == 0.0,
                    "fuel_stor_kwh must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.fc_max_kw == 0.0,
                    "max_fuel_conv_kw must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
            } else if (self.veh_pt_type == CONV) && !self.stop_start {
                assert!(
                    self.mc_max_kw == 0.0,
                    "max_mc_kw must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.ess_max_kw == 0.0,
                    "max_ess_kw must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.ess_max_kwh == 0.0,
                    "max_ess_kwh must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
            }
        }
        // ### Build roadway power lookup table
        self.max_roadway_chg_kw = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        self.charging_on = false;

        // # Checking if a vehicle has any hybrid components
        if (self.ess_max_kwh == 0.0) || (self.ess_max_kw == 0.0) || (self.mc_max_kw == 0.0) {
            self.no_elec_sys = true;
        } else {
            self.no_elec_sys = false;
        }

        // # Checking if aux loads go through an alternator
        if self.no_elec_sys || (self.mc_max_kw <= self.aux_kw) || self.force_aux_on_fc {
            self.no_elec_aux = true;
        } else {
            self.no_elec_aux = false;
        }

        // # discrete array of possible engine power outputs
        self.input_kw_out_array = self.fc_pwr_out_perc.clone() * self.fc_max_kw;
        // # Relatively continuous array of possible engine power outputs
        self.fc_kw_out_array = self
            .fc_perc_out_array
            .iter()
            .map(|n| n * self.fc_max_kw)
            .collect();
        // # Creates relatively continuous array for fc_eff
        self.fc_eff_array = self
            .fc_perc_out_array
            .iter()
            .map(|x: &f64| -> f64 {
                interpolate(
                    x,
                    &Array1::from(self.fc_pwr_out_perc.to_vec()),
                    &self.fc_eff_map,
                    false,
                )
            })
            .collect();

        self.modern_max = MODERN_MAX;

        // NOTE: unused because the first part of if/else commented below is unused
        let modern_diff = self.modern_max - arrmax(&LARGE_BASELINE_EFF);
        let _large_baseline_eff_adj: Vec<f64> =
            LARGE_BASELINE_EFF.iter().map(|x| x + modern_diff).collect();
        // Should the above lines be moved to another file? Or maybe have the outputs hardcoded?
        let _mc_kw_adj_perc = max(
            0.0,
            min(
                (self.mc_max_kw - self.small_motor_power_kw)
                    / (self.large_motor_power_kw - self.small_motor_power_kw),
                1.0,
            ),
        );

        // NOTE: it should not be possible to have `None in self.mc_eff_map` in Rust (although NaN is possible...).
        //       if we want to express that mc_eff_map should be calculated in some cases, but not others,
        //       we may need some sort of option type ?
        //if None in self.mc_eff_map:
        //    self.mc_eff_array = mc_kw_adj_perc * large_baseline_eff_adj + \
        //            (1 - mc_kw_adj_perc) * self.small_baseline_eff
        //    self.mc_eff_map = self.mc_eff_array
        //else:
        //    self.mc_eff_array = self.mc_eff_map
        if false {
            // println!("{:?}",self.mc_eff_map);
            // self.mc_eff_array = mc_kw_adj_perc * large_baseline_eff_adj
            //     + (1.0 - mc_kw_adj_perc) * self.small_baseline_eff.clone();
            // self.mc_eff_map = self.mc_eff_array.clone();
        } else {
            self.mc_eff_array = self.mc_eff_map.clone();
        }

        let mc_kw_out_array: Vec<f64> =
            (Array::linspace(0.0, 1.0, self.mc_perc_out_array.len()) * self.mc_max_kw).to_vec();

        let mc_full_eff_array: Vec<f64> = self
            .mc_perc_out_array
            .iter()
            .enumerate()
            .map(|(idx, &x): (usize, &f64)| -> f64 {
                if idx == 0 {
                    0.0
                } else {
                    interpolate(&x, &self.mc_pwr_out_perc, &self.mc_eff_array, false)
                }
            })
            .collect();

        let mc_kw_in_array: Vec<f64> = [0.0; 101]
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if idx == 0 {
                    0.0
                } else {
                    mc_kw_out_array[idx] / mc_full_eff_array[idx]
                }
            })
            .collect();

        self.mc_kw_in_array = mc_kw_in_array.clone();
        self.mc_kw_out_array = mc_kw_out_array;
        self.mc_max_elec_in_kw = arrmax(&mc_kw_in_array);
        self.mc_full_eff_array = mc_full_eff_array;

        self.mc_max_elec_in_kw = arrmax(&self.mc_kw_in_array);

        // check that efficiencies are not violating the first law of thermo
        assert!(
            arrmin(&self.fc_eff_array) >= 0.0,
            "min MC eff < 0 is not allowed"
        );
        assert!(self.fc_peak_eff() < 1.0, "fcPeakEff >= 1 is not allowed.");
        assert!(
            arrmin(&self.mc_full_eff_array) >= 0.0,
            "min MC eff < 0 is not allowed"
        );
        assert!(self.mc_peak_eff() < 1.0, "mcPeakEff >= 1 is not allowed.");

        self.set_veh_mass();
    }

    pub fn test_veh() -> Self {
        let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
        let selection: u32 = 5;
        let veh_year: u32 = 2016;
        let veh_pt_type = String::from("Conv");
        let drag_coef: f64 = 0.355;
        let frontal_area_m2: f64 = 3.066;
        let glider_kg: f64 = 1359.166;
        let veh_cg_m: f64 = 0.53;
        let drive_axle_weight_frac: f64 = 0.59;
        let wheel_base_m: f64 = 2.6;
        let cargo_kg: f64 = 136.0;
        let veh_override_kg: f64 = f64::NAN;
        let comp_mass_multiplier: f64 = 1.4;
        let fs_max_kw: f64 = 2000.0;
        let fs_secs_to_peak_pwr: f64 = 1.0;
        let fs_kwh: f64 = 504.0;
        let fs_kwh_per_kg: f64 = 9.89;
        let fc_max_kw: f64 = 125.0;
        let fc_pwr_out_perc: Vec<f64> = vec![
            0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
        ];
        let fc_eff_map: Vec<f64> = vec![
            0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
        ];
        let fc_eff_type: String = String::from("SI");
        let fc_sec_to_peak_pwr: f64 = 6.0;
        let fc_base_kg: f64 = 61.0;
        let fc_kw_per_kg: f64 = 2.13;
        let min_fc_time_on: f64 = 30.0;
        let idle_fc_kw: f64 = 2.5;
        let mc_max_kw: f64 = 0.0;
        let mc_pwr_out_perc: Vec<f64> =
            vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mc_eff_map: Vec<f64> = vec![
            0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,
        ];
        let mc_sec_to_peak_pwr: f64 = 4.0;
        let mc_pe_kg_per_kw: f64 = 0.833;
        let mc_pe_base_kg: f64 = 21.6;
        let ess_max_kw: f64 = 0.0;
        let ess_max_kwh: f64 = 0.0;
        let ess_kg_per_kwh: f64 = 8.0;
        let ess_base_kg: f64 = 75.0;
        let ess_round_trip_eff: f64 = 0.97;
        let ess_life_coef_a: f64 = 110.0;
        let ess_life_coef_b: f64 = -0.6811;
        let min_soc: f64 = 0.4;
        let max_soc: f64 = 0.8;
        let ess_dischg_to_fc_max_eff_perc: f64 = 0.0;
        let ess_chg_to_fc_max_eff_perc: f64 = 0.0;
        let wheel_inertia_kg_m2: f64 = 0.815;
        let num_wheels: f64 = 4.0;
        let wheel_rr_coef: f64 = 0.006;
        let wheel_radius_m: f64 = 0.336;
        let wheel_coef_of_fric: f64 = 0.7;
        let max_accel_buffer_mph: f64 = 60.0;
        let max_accel_buffer_perc_of_useable_soc: f64 = 0.2;
        let perc_high_acc_buf: f64 = 0.0;
        let mph_fc_on: f64 = 30.0;
        let kw_demand_fc_on: f64 = 100.0;
        let max_regen: f64 = 0.98;
        let stop_start: bool = false;
        let force_aux_on_fc: bool = false;
        let alt_eff: f64 = 1.0;
        let chg_eff: f64 = 0.86;
        let aux_kw: f64 = 0.7;
        let trans_kg: f64 = 114.0;
        let trans_eff: f64 = 0.92;
        let ess_to_fuel_ok_error: f64 = 0.005;
        let val_udds_mpgge: f64 = 23.0;
        let val_hwy_mpgge: f64 = 32.0;
        let val_comb_mpgge: f64 = 26.0;
        let val_udds_kwh_per_mile: f64 = f64::NAN;
        let val_hwy_kwh_per_mile: f64 = f64::NAN;
        let val_comb_kwh_per_mile: f64 = f64::NAN;
        let val_cd_range_mi: f64 = f64::NAN;
        let val_const65_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const60_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const55_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const45_mph_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_udds_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_hwy_kwh_per_mile: f64 = f64::NAN;
        let val0_to60_mph: f64 = 9.9;
        let val_ess_life_miles: f64 = f64::NAN;
        let val_range_miles: f64 = f64::NAN;
        let val_veh_base_cost: f64 = f64::NAN;
        let val_msrp: f64 = f64::NAN;
        let props = RustPhysicalProperties::__new__();
        // TODO: make large_baseline_eff and small_baseline_eff constanst at the module level
        // pub const LARGE_BASELINE_EFF: &[f64; 11] = [
        // 0.83, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.94, 0.94, 0.93, 0.92,
        // ]
        let small_motor_power_kw: f64 = 7.5;
        let large_motor_power_kw: f64 = 75.0;
        // TODO: make this look more like:
        // fc_perc_out_array = np.r_[np.arange(0, 3.0, 0.1), np.arange(
        //     3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100  # hardcoded ***
        let fc_perc_out_array: Vec<f64> = FC_PERC_OUT_ARRAY.to_vec();
        let max_roadway_chg_kw: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let charging_on: bool = false;
        let no_elec_sys: bool = true;
        let no_elec_aux: bool = true;
        let modern_max: f64 = 0.95;
        let regen_a: f64 = 500.0;
        let regen_b: f64 = 0.99;
        let mc_max_elec_in_kw: f64 = 100.0;
        let ess_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        let mc_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        let fc_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        let fs_mass_kg: f64 = 0.0;
        // DERIVED
        let input_kw_out_array = fc_pwr_out_perc.iter().map(|&x| x * fc_max_kw).collect();
        let fc_kw_out_array = fc_perc_out_array.iter().map(|&x| x * fc_max_kw).collect();
        let fc_eff_array = fc_perc_out_array
            .iter()
            .map(|&x| {
                interpolate(
                    &x,
                    &Array::from(fc_pwr_out_perc.clone()),
                    &Array::from(fc_eff_map.clone()),
                    false,
                )
            })
            .collect::<Vec<_>>();
        let mc_perc_out_array = MC_PERC_OUT_ARRAY.to_vec();
        let mc_kw_out_array =
            (Array::linspace(0.0, 1.0, mc_perc_out_array.len()) * mc_max_kw).to_vec();
        let mc_eff_array = LARGE_BASELINE_EFF
            .iter()
            .map(|&x| {
                interpolate(
                    &x,
                    &Array::from(mc_pwr_out_perc.clone()),
                    &Array::from(mc_eff_map.clone()),
                    false,
                )
            })
            .collect::<Vec<_>>();
        let mc_kw_in_array = Array::ones(mc_kw_out_array.len()).to_vec();
        let mc_full_eff_array = Array::ones(mc_eff_array.len()).to_vec();
        let veh_kg: f64 = cargo_kg
            + glider_kg
            + trans_kg * comp_mass_multiplier
            + ess_mass_kg
            + mc_mass_kg
            + fc_mass_kg
            + fs_mass_kg;
        let max_trac_mps2: f64 =
            (wheel_coef_of_fric * drive_axle_weight_frac * veh_kg * props.a_grav_mps2
                / (1.0 + veh_cg_m * wheel_coef_of_fric / wheel_base_m))
                / (veh_kg * props.a_grav_mps2)
                * props.a_grav_mps2;

        RustVehicle::__new__(
            scenario_name,
            selection,
            veh_year,
            veh_pt_type,
            drag_coef,
            frontal_area_m2,
            glider_kg,
            veh_cg_m,
            drive_axle_weight_frac,
            wheel_base_m,
            cargo_kg,
            veh_override_kg,
            comp_mass_multiplier,
            fs_max_kw,
            fs_secs_to_peak_pwr,
            fs_kwh,
            fs_kwh_per_kg,
            fc_max_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            mc_eff_map,
            mc_sec_to_peak_pwr,
            mc_pe_kg_per_kw,
            mc_pe_base_kg,
            ess_max_kw,
            ess_max_kwh,
            ess_kg_per_kwh,
            ess_base_kg,
            ess_round_trip_eff,
            ess_life_coef_a,
            ess_life_coef_b,
            min_soc,
            max_soc,
            ess_dischg_to_fc_max_eff_perc,
            ess_chg_to_fc_max_eff_perc,
            wheel_inertia_kg_m2,
            num_wheels,
            wheel_rr_coef,
            wheel_radius_m,
            wheel_coef_of_fric,
            max_accel_buffer_mph,
            max_accel_buffer_perc_of_useable_soc,
            perc_high_acc_buf,
            mph_fc_on,
            kw_demand_fc_on,
            max_regen,
            stop_start,
            force_aux_on_fc,
            alt_eff,
            chg_eff,
            aux_kw,
            trans_kg,
            trans_eff,
            ess_to_fuel_ok_error,
            val_udds_mpgge,
            val_hwy_mpgge,
            val_comb_mpgge,
            val_udds_kwh_per_mile,
            val_hwy_kwh_per_mile,
            val_comb_kwh_per_mile,
            val_cd_range_mi,
            val_const65_mph_kwh_per_mile,
            val_const60_mph_kwh_per_mile,
            val_const55_mph_kwh_per_mile,
            val_const45_mph_kwh_per_mile,
            val_unadj_udds_kwh_per_mile,
            val_unadj_hwy_kwh_per_mile,
            val0_to60_mph,
            val_ess_life_miles,
            val_range_miles,
            val_veh_base_cost,
            val_msrp,
            props,
            small_motor_power_kw,
            large_motor_power_kw,
            Some(fc_perc_out_array),
            charging_on,
            no_elec_sys,
            no_elec_aux,
            max_roadway_chg_kw,
            input_kw_out_array,
            fc_kw_out_array,
            fc_eff_array,
            modern_max,
            mc_eff_array,
            mc_kw_in_array,
            mc_kw_out_array,
            mc_max_elec_in_kw,
            Some(mc_full_eff_array),
            regen_a,
            regen_b,
            veh_kg,
            max_trac_mps2,
            ess_mass_kg,
            mc_mass_kg,
            fc_mass_kg,
            fs_mass_kg,
            Some(mc_perc_out_array),
        )
    }
}

/// RustVehicle class for containing:
#[pymethods]
#[allow(clippy::too_many_arguments)]
impl RustVehicle {
    /// Calculate total vehicle mass. Sum up component masses if
    /// positive real number is not specified for self.veh_override_kg
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn set_veh_mass(&mut self) {
        let mut ess_mass_kg = 0.0;
        let mut mc_mass_kg = 0.0;
        let mut fc_mass_kg = 0.0;
        let mut fs_mass_kg = 0.0;

        if !(self.veh_override_kg > 0.0) {
            ess_mass_kg = if self.ess_max_kwh == 0.0 || self.ess_max_kw == 0.0 {
                0.0
            } else {
                ((self.ess_max_kwh * self.ess_kg_per_kwh) + self.ess_base_kg)
                    * self.comp_mass_multiplier
            };
            mc_mass_kg = if self.mc_max_kw == 0.0 {
                0.0
            } else {
                (self.mc_pe_base_kg + (self.mc_pe_kg_per_kw * self.mc_max_kw))
                    * self.comp_mass_multiplier
            };
            fc_mass_kg = if self.fc_max_kw == 0.0 {
                0.0
            } else {
                (1.0 / self.fc_kw_per_kg * self.fc_max_kw + self.fc_base_kg)
                    * self.comp_mass_multiplier
            };
            fs_mass_kg = if self.fs_max_kw == 0.0 {
                0.0
            } else {
                ((1.0 / self.fs_kwh_per_kg) * self.fs_kwh) * self.comp_mass_multiplier
            };
            self.veh_kg = self.cargo_kg
                + self.glider_kg
                + self.trans_kg * self.comp_mass_multiplier
                + ess_mass_kg
                + mc_mass_kg
                + fc_mass_kg
                + fs_mass_kg;
        } else {
            // if positive real number is specified for veh_override_kg, use that
            self.veh_kg = self.veh_override_kg;
        }

        self.max_trac_mps2 = (self.wheel_coef_of_fric
            * self.drive_axle_weight_frac
            * self.veh_kg
            * self.props.a_grav_mps2
            / (1.0 + self.veh_cg_m * self.wheel_coef_of_fric / self.wheel_base_m))
            / (self.veh_kg * self.props.a_grav_mps2)
            * self.props.a_grav_mps2;

        // copying to instance attributes
        self.ess_mass_kg = ess_mass_kg;
        self.mc_mass_kg = mc_mass_kg;
        self.fc_mass_kg = fc_mass_kg;
        self.fs_mass_kg = fs_mass_kg;
    }

    #[new]
    pub fn __new__(
        scenario_name: String,
        selection: u32,
        veh_year: u32,
        veh_pt_type: String,
        drag_coef: f64,
        frontal_area_m2: f64,
        glider_kg: f64,
        veh_cg_m: f64,
        drive_axle_weight_frac: f64,
        wheel_base_m: f64,
        cargo_kg: f64,
        veh_override_kg: f64,
        comp_mass_multiplier: f64,
        fs_max_kw: f64,
        fs_secs_to_peak_pwr: f64,
        fs_kwh: f64,
        fs_kwh_per_kg: f64,
        fc_max_kw: f64,
        fc_pwr_out_perc: Vec<f64>,
        fc_eff_map: Vec<f64>,
        fc_eff_type: String,
        fc_sec_to_peak_pwr: f64,
        fc_base_kg: f64,
        fc_kw_per_kg: f64,
        min_fc_time_on: f64,
        idle_fc_kw: f64,
        mc_max_kw: f64,
        mc_pwr_out_perc: Vec<f64>,
        mc_eff_map: Vec<f64>,
        mc_sec_to_peak_pwr: f64,
        mc_pe_kg_per_kw: f64,
        mc_pe_base_kg: f64,
        ess_max_kw: f64,
        ess_max_kwh: f64,
        ess_kg_per_kwh: f64,
        ess_base_kg: f64,
        ess_round_trip_eff: f64,
        ess_life_coef_a: f64,
        ess_life_coef_b: f64,
        min_soc: f64,
        max_soc: f64,
        ess_dischg_to_fc_max_eff_perc: f64,
        ess_chg_to_fc_max_eff_perc: f64,
        wheel_inertia_kg_m2: f64,
        num_wheels: f64,
        wheel_rr_coef: f64,
        wheel_radius_m: f64,
        wheel_coef_of_fric: f64,
        max_accel_buffer_mph: f64,
        max_accel_buffer_perc_of_useable_soc: f64,
        perc_high_acc_buf: f64,
        mph_fc_on: f64,
        kw_demand_fc_on: f64,
        max_regen: f64,
        stop_start: bool,
        force_aux_on_fc: bool,
        alt_eff: f64,
        chg_eff: f64,
        aux_kw: f64,
        trans_kg: f64,
        trans_eff: f64,
        ess_to_fuel_ok_error: f64,
        val_udds_mpgge: f64,
        val_hwy_mpgge: f64,
        val_comb_mpgge: f64,
        val_udds_kwh_per_mile: f64,
        val_hwy_kwh_per_mile: f64,
        val_comb_kwh_per_mile: f64,
        val_cd_range_mi: f64,
        val_const65_mph_kwh_per_mile: f64,
        val_const60_mph_kwh_per_mile: f64,
        val_const55_mph_kwh_per_mile: f64,
        val_const45_mph_kwh_per_mile: f64,
        val_unadj_udds_kwh_per_mile: f64,
        val_unadj_hwy_kwh_per_mile: f64,
        val0_to60_mph: f64,
        val_ess_life_miles: f64,
        val_range_miles: f64,
        val_veh_base_cost: f64,
        val_msrp: f64,
        props: RustPhysicalProperties,
        small_motor_power_kw: f64,
        large_motor_power_kw: f64,
        fc_perc_out_array: Option<Vec<f64>>,
        charging_on: bool,
        no_elec_sys: bool,
        no_elec_aux: bool,
        max_roadway_chg_kw: Vec<f64>,
        input_kw_out_array: Vec<f64>,
        fc_kw_out_array: Vec<f64>,
        fc_eff_array: Vec<f64>,
        modern_max: f64,
        mc_eff_array: Vec<f64>,
        mc_kw_in_array: Vec<f64>,
        mc_kw_out_array: Vec<f64>,
        mc_max_elec_in_kw: f64,
        mc_full_eff_array: Option<Vec<f64>>,
        regen_a: f64,
        regen_b: f64,
        veh_kg: f64,
        max_trac_mps2: f64,
        ess_mass_kg: f64,
        mc_mass_kg: f64,
        fc_mass_kg: f64,
        fs_mass_kg: f64,
        mc_perc_out_array: Option<Vec<f64>>,
    ) -> Self {
        let fc_pwr_out_perc = Array::from_vec(fc_pwr_out_perc);
        let fc_eff_map = Array::from_vec(fc_eff_map);
        let mc_pwr_out_perc = Array::from_vec(mc_pwr_out_perc);
        let mc_eff_map = Array::from_vec(mc_eff_map);
        let fc_perc_out_array: Vec<f64> =
            fc_perc_out_array.unwrap_or_else(|| FC_PERC_OUT_ARRAY.clone().to_vec());
        let max_roadway_chg_kw = Array::from_vec(max_roadway_chg_kw);
        let input_kw_out_array = Array::from_vec(input_kw_out_array);
        let mc_eff_array = Array::from_vec(mc_eff_array);
        // get mc_full_eff_vec into array form
        let mc_full_eff_array: Vec<f64> = mc_full_eff_array.unwrap_or_else(|| [1.0; 101].to_vec());
        let mc_perc_out_array: Vec<f64> =
            mc_perc_out_array.unwrap_or_else(|| MC_PERC_OUT_ARRAY.clone().to_vec());

        // DERIVED VALUES
        // TODO: correctly implement and re-enable these after Rust does all initialization of inputs

        // let veh_kg: f64 = cargo_kg + glider_kg + trans_kg * comp_mass_multiplier
        //     + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg;
        // let max_trac_mps2: f64 = (
        //     wheel_coef_of_fric * drive_axle_weight_frac * veh_kg * props.a_grav_mps2 /
        //     (1.0 + veh_cg_m * wheel_coef_of_fric / wheel_base_m)
        // ) / (veh_kg * props.a_grav_mps2)  * props.a_grav_mps2;

        let mut veh = RustVehicle {
            scenario_name,
            selection,
            veh_year,
            veh_pt_type,
            drag_coef,
            frontal_area_m2,
            glider_kg,
            veh_cg_m,
            drive_axle_weight_frac,
            wheel_base_m,
            cargo_kg,
            veh_override_kg,
            comp_mass_multiplier,
            fs_max_kw,
            fs_secs_to_peak_pwr,
            fs_kwh,
            fs_kwh_per_kg,
            fc_max_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            mc_eff_map,
            mc_sec_to_peak_pwr,
            mc_pe_kg_per_kw,
            mc_pe_base_kg,
            ess_max_kw,
            ess_max_kwh,
            ess_kg_per_kwh,
            ess_base_kg,
            ess_round_trip_eff,
            ess_life_coef_a,
            ess_life_coef_b,
            min_soc,
            max_soc,
            ess_dischg_to_fc_max_eff_perc,
            ess_chg_to_fc_max_eff_perc,
            wheel_inertia_kg_m2,
            num_wheels,
            wheel_rr_coef,
            wheel_radius_m,
            wheel_coef_of_fric,
            max_accel_buffer_mph,
            max_accel_buffer_perc_of_useable_soc,
            perc_high_acc_buf,
            mph_fc_on,
            kw_demand_fc_on,
            max_regen,
            stop_start,
            force_aux_on_fc,
            alt_eff,
            chg_eff,
            aux_kw,
            trans_kg,
            trans_eff,
            ess_to_fuel_ok_error,
            val_udds_mpgge,
            val_hwy_mpgge,
            val_comb_mpgge,
            val_udds_kwh_per_mile,
            val_hwy_kwh_per_mile,
            val_comb_kwh_per_mile,
            val_cd_range_mi,
            val_const65_mph_kwh_per_mile,
            val_const60_mph_kwh_per_mile,
            val_const55_mph_kwh_per_mile,
            val_const45_mph_kwh_per_mile,
            val_unadj_udds_kwh_per_mile,
            val_unadj_hwy_kwh_per_mile,
            val0_to60_mph,
            val_ess_life_miles,
            val_range_miles,
            val_veh_base_cost,
            val_msrp,
            props,
            small_motor_power_kw,
            large_motor_power_kw,
            fc_perc_out_array,
            regen_a,
            regen_b,
            charging_on,
            no_elec_sys,
            no_elec_aux,
            max_roadway_chg_kw,
            input_kw_out_array,
            fc_kw_out_array,
            fc_eff_array,
            modern_max,
            mc_eff_array,
            mc_kw_in_array,
            mc_kw_out_array,
            mc_max_elec_in_kw,
            mc_full_eff_array,
            veh_kg,
            max_trac_mps2,
            ess_mass_kg,
            mc_mass_kg,
            fc_mass_kg,
            fs_mass_kg,
            mc_perc_out_array,
        };
        veh.set_derived();
        veh
    }

    #[getter]
    pub fn get_fc_peak_eff(&self) -> PyResult<f64> {
        Ok(self.fc_peak_eff())
    }
    #[setter]
    pub fn set_fc_peak_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.set_fc_peak_eff_rust(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_mc_peak_eff(&self) -> PyResult<f64> {
        Ok(self.mc_peak_eff())
    }
    #[setter]
    pub fn set_mc_peak_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.set_mc_peak_eff_rust(new_value);
        Ok(())
    }

    /// An identify function to allow RustVehicle to be used as a python vehicle and respond to this method
    /// Returns a clone of the current object
    pub fn to_rust(&self) -> PyResult<RustVehicle> {
        Ok(self.clone())
    }

    #[getter]
    pub fn get_max_fc_eff_kw(&self) -> PyResult<f64> {
        Ok(self.max_fc_eff_kw())
    }

    #[getter]
    pub fn get_max_regen_kwh(&self) -> PyResult<f64> {
        Ok(self.max_regen_kwh())
    }

    #[getter]
    pub fn get_alt_eff(&self) -> PyResult<f64> {
        Ok(self.alt_eff)
    }
    #[setter]
    pub fn set_alt_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.alt_eff = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_aux_kw(&self) -> PyResult<f64> {
        Ok(self.aux_kw)
    }
    #[setter]
    pub fn set_aux_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.aux_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_cargo_kg(&self) -> PyResult<f64> {
        Ok(self.cargo_kg)
    }
    #[setter]
    pub fn set_cargo_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.cargo_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_charging_on(&self) -> PyResult<bool> {
        Ok(self.charging_on)
    }
    #[setter]
    pub fn set_charging_on(&mut self, new_value: bool) -> PyResult<()> {
        self.charging_on = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_chg_eff(&self) -> PyResult<f64> {
        Ok(self.chg_eff)
    }
    #[setter]
    pub fn set_chg_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.chg_eff = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_comp_mass_multiplier(&self) -> PyResult<f64> {
        Ok(self.comp_mass_multiplier)
    }
    #[setter]
    pub fn set_comp_mass_multiplier(&mut self, new_value: f64) -> PyResult<()> {
        self.comp_mass_multiplier = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_drag_coef(&self) -> PyResult<f64> {
        Ok(self.drag_coef)
    }
    #[setter]
    pub fn set_drag_coef(&mut self, new_value: f64) -> PyResult<()> {
        self.drag_coef = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_drive_axle_weight_frac(&self) -> PyResult<f64> {
        Ok(self.drive_axle_weight_frac)
    }
    #[setter]
    pub fn set_drive_axle_weight_frac(&mut self, new_value: f64) -> PyResult<()> {
        self.drive_axle_weight_frac = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_base_kg(&self) -> PyResult<f64> {
        Ok(self.ess_base_kg)
    }
    #[setter]
    pub fn set_ess_base_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_base_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_chg_to_fc_max_eff_perc(&self) -> PyResult<f64> {
        Ok(self.ess_chg_to_fc_max_eff_perc)
    }
    #[setter]
    pub fn set_ess_chg_to_fc_max_eff_perc(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_chg_to_fc_max_eff_perc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_dischg_to_fc_max_eff_perc(&self) -> PyResult<f64> {
        Ok(self.ess_dischg_to_fc_max_eff_perc)
    }
    #[setter]
    pub fn set_ess_dischg_to_fc_max_eff_perc(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_dischg_to_fc_max_eff_perc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_kg_per_kwh(&self) -> PyResult<f64> {
        Ok(self.ess_kg_per_kwh)
    }
    #[setter]
    pub fn set_ess_kg_per_kwh(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_kg_per_kwh = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_life_coef_a(&self) -> PyResult<f64> {
        Ok(self.ess_life_coef_a)
    }
    #[setter]
    pub fn set_ess_life_coef_a(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_life_coef_a = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_life_coef_b(&self) -> PyResult<f64> {
        Ok(self.ess_life_coef_b)
    }
    #[setter]
    pub fn set_ess_life_coef_b(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_life_coef_b = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_mass_kg(&self) -> PyResult<f64> {
        Ok(self.ess_mass_kg)
    }
    #[setter]
    pub fn set_ess_mass_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_mass_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_round_trip_eff(&self) -> PyResult<f64> {
        Ok(self.ess_round_trip_eff)
    }
    #[setter]
    pub fn set_ess_round_trip_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_round_trip_eff = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_to_fuel_ok_error(&self) -> PyResult<f64> {
        Ok(self.ess_to_fuel_ok_error)
    }
    #[setter]
    pub fn set_ess_to_fuel_ok_error(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_to_fuel_ok_error = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_eff_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.fc_eff_array.to_vec())
    }
    #[setter]
    pub fn set_fc_eff_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.fc_eff_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_eff_map(&self) -> PyResult<Vec<f64>> {
        Ok(self.fc_eff_map.to_vec())
    }
    #[setter]
    pub fn set_fc_eff_map(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.fc_eff_map = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_fc_eff_type(&self) -> PyResult<String> {
        Ok(self.fc_eff_type.clone())
    }
    #[setter]
    pub fn set_fc_eff_type(&mut self, new_value: String) -> PyResult<()> {
        self.fc_eff_type = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_kw_out_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.fc_kw_out_array.to_vec())
    }
    #[setter]
    pub fn set_fc_kw_out_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.fc_kw_out_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_mass_kg(&self) -> PyResult<f64> {
        Ok(self.fc_mass_kg)
    }
    #[setter]
    pub fn set_fc_mass_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.fc_mass_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_perc_out_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.fc_perc_out_array.to_vec())
    }
    #[setter]
    pub fn set_fc_perc_out_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.fc_perc_out_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_pwr_out_perc(&self) -> PyResult<Vec<f64>> {
        Ok(self.fc_pwr_out_perc.to_vec())
    }
    #[setter]
    pub fn set_fc_pwr_out_perc(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.fc_pwr_out_perc = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_force_aux_on_fc(&self) -> PyResult<bool> {
        Ok(self.force_aux_on_fc)
    }
    #[setter]
    pub fn set_force_aux_on_fc(&mut self, new_value: bool) -> PyResult<()> {
        self.force_aux_on_fc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_frontal_area_m2(&self) -> PyResult<f64> {
        Ok(self.frontal_area_m2)
    }
    #[setter]
    pub fn set_frontal_area_m2(&mut self, new_value: f64) -> PyResult<()> {
        self.frontal_area_m2 = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fs_mass_kg(&self) -> PyResult<f64> {
        Ok(self.fs_mass_kg)
    }
    #[setter]
    pub fn set_fs_mass_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.fs_mass_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_base_kg(&self) -> PyResult<f64> {
        Ok(self.fc_base_kg)
    }
    #[setter]
    pub fn set_fc_base_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.fc_base_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_kw_per_kg(&self) -> PyResult<f64> {
        Ok(self.fc_kw_per_kg)
    }
    #[setter]
    pub fn set_fc_kw_per_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.fc_kw_per_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_sec_to_peak_pwr(&self) -> PyResult<f64> {
        Ok(self.fc_sec_to_peak_pwr)
    }
    #[setter]
    pub fn set_fc_sec_to_peak_pwr(&mut self, new_value: f64) -> PyResult<()> {
        self.fc_sec_to_peak_pwr = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fs_kwh(&self) -> PyResult<f64> {
        Ok(self.fs_kwh)
    }
    #[setter]
    pub fn set_fs_kwh(&mut self, new_value: f64) -> PyResult<()> {
        self.fs_kwh = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fs_kwh_per_kg(&self) -> PyResult<f64> {
        Ok(self.fs_kwh_per_kg)
    }
    #[setter]
    pub fn set_fs_kwh_per_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.fs_kwh_per_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fs_secs_to_peak_pwr(&self) -> PyResult<f64> {
        Ok(self.fs_secs_to_peak_pwr)
    }
    #[setter]
    pub fn set_fs_secs_to_peak_pwr(&mut self, new_value: f64) -> PyResult<()> {
        self.fs_secs_to_peak_pwr = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_glider_kg(&self) -> PyResult<f64> {
        Ok(self.glider_kg)
    }
    #[setter]
    pub fn set_glider_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.glider_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_idle_fc_kw(&self) -> PyResult<f64> {
        Ok(self.idle_fc_kw)
    }
    #[setter]
    pub fn set_idle_fc_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.idle_fc_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_input_kw_out_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.input_kw_out_array.to_vec())
    }
    #[setter]
    pub fn set_input_kw_out_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.input_kw_out_array = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_kw_demand_fc_on(&self) -> PyResult<f64> {
        Ok(self.kw_demand_fc_on)
    }
    #[setter]
    pub fn set_kw_demand_fc_on(&mut self, new_value: f64) -> PyResult<()> {
        self.kw_demand_fc_on = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_large_baseline_eff(&self) -> PyResult<Vec<f64>> {
        Ok(LARGE_BASELINE_EFF.to_vec())
    }

    #[getter]
    pub fn get_large_motor_power_kw(&self) -> PyResult<f64> {
        Ok(self.large_motor_power_kw)
    }
    #[setter]
    pub fn set_large_motor_power_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.large_motor_power_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_max_accel_buffer_mph(&self) -> PyResult<f64> {
        Ok(self.max_accel_buffer_mph)
    }
    #[setter]
    pub fn set_max_accel_buffer_mph(&mut self, new_value: f64) -> PyResult<()> {
        self.max_accel_buffer_mph = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_max_accel_buffer_perc_of_useable_soc(&self) -> PyResult<f64> {
        Ok(self.max_accel_buffer_perc_of_useable_soc)
    }
    #[setter]
    pub fn set_max_accel_buffer_perc_of_useable_soc(&mut self, new_value: f64) -> PyResult<()> {
        self.max_accel_buffer_perc_of_useable_soc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_max_kw(&self) -> PyResult<f64> {
        Ok(self.ess_max_kw)
    }
    #[setter]
    pub fn set_ess_max_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_max_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_ess_max_kwh(&self) -> PyResult<f64> {
        Ok(self.ess_max_kwh)
    }
    #[setter]
    pub fn set_ess_max_kwh(&mut self, new_value: f64) -> PyResult<()> {
        self.ess_max_kwh = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fc_max_kw(&self) -> PyResult<f64> {
        Ok(self.fc_max_kw)
    }
    #[setter]
    pub fn set_fc_max_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.fc_max_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_fs_max_kw(&self) -> PyResult<f64> {
        Ok(self.fs_max_kw)
    }
    #[setter]
    pub fn set_fs_max_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.fs_max_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_max_kw(&self) -> PyResult<f64> {
        Ok(self.mc_max_kw)
    }
    #[setter]
    pub fn set_mc_max_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_max_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_max_regen(&self) -> PyResult<f64> {
        Ok(self.max_regen)
    }
    #[setter]
    pub fn set_max_regen(&mut self, new_value: f64) -> PyResult<()> {
        self.max_regen = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_max_roadway_chg_kw(&self) -> PyResult<Vec<f64>> {
        Ok(self.max_roadway_chg_kw.to_vec())
    }
    #[setter]
    pub fn set_max_roadway_chg_kw(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.max_roadway_chg_kw = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_max_soc(&self) -> PyResult<f64> {
        Ok(self.max_soc)
    }
    #[setter]
    pub fn set_max_soc(&mut self, new_value: f64) -> PyResult<()> {
        self.max_soc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_max_trac_mps2(&self) -> PyResult<f64> {
        Ok(self.max_trac_mps2)
    }
    #[setter]
    pub fn set_max_trac_mps2(&mut self, new_value: f64) -> PyResult<()> {
        self.max_trac_mps2 = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_eff_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_eff_array.to_vec())
    }
    #[setter]
    pub fn set_mc_eff_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_eff_array = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_mc_eff_map(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_eff_map.to_vec())
    }
    #[setter]
    pub fn set_mc_eff_map(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_eff_map = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_mc_full_eff_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_full_eff_array.to_vec())
    }
    #[setter]
    pub fn set_mc_full_eff_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_full_eff_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_kw_in_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_kw_in_array.to_vec())
    }
    #[setter]
    pub fn set_mc_kw_in_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_kw_in_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_kw_out_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_kw_out_array.to_vec())
    }
    #[setter]
    pub fn set_mc_kw_out_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_kw_out_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_mass_kg(&self) -> PyResult<f64> {
        Ok(self.mc_mass_kg)
    }
    #[setter]
    pub fn set_mc_mass_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_mass_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_max_elec_in_kw(&self) -> PyResult<f64> {
        Ok(self.mc_max_elec_in_kw)
    }
    #[setter]
    pub fn set_mc_max_elec_in_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_max_elec_in_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_pe_base_kg(&self) -> PyResult<f64> {
        Ok(self.mc_pe_base_kg)
    }
    #[setter]
    pub fn set_mc_pe_base_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_pe_base_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_pe_kg_per_kw(&self) -> PyResult<f64> {
        Ok(self.mc_pe_kg_per_kw)
    }
    #[setter]
    pub fn set_mc_pe_kg_per_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_pe_kg_per_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_perc_out_array(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_perc_out_array.to_vec())
    }
    #[setter]
    pub fn set_mc_perc_out_array(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_perc_out_array = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_pwr_out_perc(&self) -> PyResult<Vec<f64>> {
        Ok(self.mc_pwr_out_perc.to_vec())
    }
    #[setter]
    pub fn set_mc_pwr_out_perc(&mut self, new_value: Vec<f64>) -> PyResult<()> {
        self.mc_pwr_out_perc = Array::from_vec(new_value);
        Ok(())
    }

    #[getter]
    pub fn get_min_fc_time_on(&self) -> PyResult<f64> {
        Ok(self.min_fc_time_on)
    }
    #[setter]
    pub fn set_min_fc_time_on(&mut self, new_value: f64) -> PyResult<()> {
        self.min_fc_time_on = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_min_soc(&self) -> PyResult<f64> {
        Ok(self.min_soc)
    }
    #[setter]
    pub fn set_min_soc(&mut self, new_value: f64) -> PyResult<()> {
        self.min_soc = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_modern_max(&self) -> PyResult<f64> {
        Ok(self.modern_max)
    }
    #[setter]
    pub fn set_modern_max(&mut self, new_value: f64) -> PyResult<()> {
        self.modern_max = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mc_sec_to_peak_pwr(&self) -> PyResult<f64> {
        Ok(self.mc_sec_to_peak_pwr)
    }
    #[setter]
    pub fn set_mc_sec_to_peak_pwr(&mut self, new_value: f64) -> PyResult<()> {
        self.mc_sec_to_peak_pwr = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_mph_fc_on(&self) -> PyResult<f64> {
        Ok(self.mph_fc_on)
    }
    #[setter]
    pub fn set_mph_fc_on(&mut self, new_value: f64) -> PyResult<()> {
        self.mph_fc_on = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_no_elec_aux(&self) -> PyResult<bool> {
        Ok(self.no_elec_aux)
    }
    #[setter]
    pub fn set_no_elec_aux(&mut self, new_value: bool) -> PyResult<()> {
        self.no_elec_aux = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_no_elec_sys(&self) -> PyResult<bool> {
        Ok(self.no_elec_sys)
    }
    #[setter]
    pub fn set_no_elec_sys(&mut self, new_value: bool) -> PyResult<()> {
        self.no_elec_sys = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_num_wheels(&self) -> PyResult<f64> {
        Ok(self.num_wheels)
    }
    #[setter]
    pub fn set_num_wheels(&mut self, new_value: f64) -> PyResult<()> {
        self.num_wheels = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_perc_high_acc_buf(&self) -> PyResult<f64> {
        Ok(self.perc_high_acc_buf)
    }
    #[setter]
    pub fn set_perc_high_acc_buf(&mut self, new_value: f64) -> PyResult<()> {
        self.perc_high_acc_buf = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_props(&self) -> PyResult<RustPhysicalProperties> {
        let new_props = RustPhysicalProperties {
            air_density_kg_per_m3: self.props.air_density_kg_per_m3,
            a_grav_mps2: self.props.a_grav_mps2,
            kwh_per_gge: self.props.kwh_per_gge,
            fuel_rho_kg__L: self.props.fuel_rho_kg__L,
            fuel_afr_stoich: self.props.fuel_afr_stoich,
        };
        Ok(new_props)
    }
    #[setter]
    pub fn set_props(&mut self, new_value: RustPhysicalProperties) -> PyResult<()> {
        self.props = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_regen_a(&self) -> PyResult<f64> {
        Ok(self.regen_a)
    }
    #[setter]
    pub fn set_regen_a(&mut self, new_value: f64) -> PyResult<()> {
        self.regen_a = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_regen_b(&self) -> PyResult<f64> {
        Ok(self.regen_b)
    }
    #[setter]
    pub fn set_regen_b(&mut self, new_value: f64) -> PyResult<()> {
        self.regen_b = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_scenario_name(&self) -> PyResult<String> {
        Ok(self.scenario_name.clone())
    }
    #[setter]
    pub fn set_scenario_name(&mut self, new_value: String) -> PyResult<()> {
        self.scenario_name = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_selection(&self) -> PyResult<u32> {
        Ok(self.selection)
    }
    #[setter]
    pub fn set_selection(&mut self, new_value: u32) -> PyResult<()> {
        self.selection = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_small_baseline_eff(&self) -> PyResult<Vec<f64>> {
        Ok(SMALL_BASELINE_EFF.to_vec())
    }

    #[getter]
    pub fn get_small_motor_power_kw(&self) -> PyResult<f64> {
        Ok(self.small_motor_power_kw)
    }
    #[setter]
    pub fn set_small_motor_power_kw(&mut self, new_value: f64) -> PyResult<()> {
        self.small_motor_power_kw = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_stop_start(&self) -> PyResult<bool> {
        Ok(self.stop_start)
    }
    #[setter]
    pub fn set_stop_start(&mut self, new_value: bool) -> PyResult<()> {
        self.stop_start = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_trans_eff(&self) -> PyResult<f64> {
        Ok(self.trans_eff)
    }
    #[setter]
    pub fn set_trans_eff(&mut self, new_value: f64) -> PyResult<()> {
        self.trans_eff = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_trans_kg(&self) -> PyResult<f64> {
        Ok(self.trans_kg)
    }
    #[setter]
    pub fn set_trans_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.trans_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val0_to60_mph(&self) -> PyResult<f64> {
        Ok(self.val0_to60_mph)
    }
    #[setter]
    pub fn set_val0_to60_mph(&mut self, new_value: f64) -> PyResult<()> {
        self.val0_to60_mph = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_cd_range_mi(&self) -> PyResult<f64> {
        Ok(self.val_cd_range_mi)
    }
    #[setter]
    pub fn set_val_cd_range_mi(&mut self, new_value: f64) -> PyResult<()> {
        self.val_cd_range_mi = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_comb_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_comb_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_comb_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_comb_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_comb_mpgge(&self) -> PyResult<f64> {
        Ok(self.val_comb_mpgge)
    }
    #[setter]
    pub fn set_val_comb_mpgge(&mut self, new_value: f64) -> PyResult<()> {
        self.val_comb_mpgge = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_const45_mph_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_const45_mph_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_const45_mph_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_const45_mph_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_const55_mph_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_const55_mph_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_const55_mph_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_const55_mph_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_const60_mph_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_const60_mph_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_const60_mph_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_const60_mph_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_const65_mph_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_const65_mph_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_const65_mph_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_const65_mph_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_ess_life_miles(&self) -> PyResult<f64> {
        Ok(self.val_ess_life_miles)
    }
    #[setter]
    pub fn set_val_ess_life_miles(&mut self, new_value: f64) -> PyResult<()> {
        self.val_ess_life_miles = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_hwy_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_hwy_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_hwy_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_hwy_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_hwy_mpgge(&self) -> PyResult<f64> {
        Ok(self.val_hwy_mpgge)
    }
    #[setter]
    pub fn set_val_hwy_mpgge(&mut self, new_value: f64) -> PyResult<()> {
        self.val_hwy_mpgge = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_msrp(&self) -> PyResult<f64> {
        Ok(self.val_msrp)
    }
    #[setter]
    pub fn set_val_msrp(&mut self, new_value: f64) -> PyResult<()> {
        self.val_msrp = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_range_miles(&self) -> PyResult<f64> {
        Ok(self.val_range_miles)
    }
    #[setter]
    pub fn set_val_range_miles(&mut self, new_value: f64) -> PyResult<()> {
        self.val_range_miles = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_udds_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_udds_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_udds_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_udds_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_udds_mpgge(&self) -> PyResult<f64> {
        Ok(self.val_udds_mpgge)
    }
    #[setter]
    pub fn set_val_udds_mpgge(&mut self, new_value: f64) -> PyResult<()> {
        self.val_udds_mpgge = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_unadj_hwy_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_unadj_hwy_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_unadj_hwy_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_unadj_hwy_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_unadj_udds_kwh_per_mile(&self) -> PyResult<f64> {
        Ok(self.val_unadj_udds_kwh_per_mile)
    }
    #[setter]
    pub fn set_val_unadj_udds_kwh_per_mile(&mut self, new_value: f64) -> PyResult<()> {
        self.val_unadj_udds_kwh_per_mile = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_val_veh_base_cost(&self) -> PyResult<f64> {
        Ok(self.val_veh_base_cost)
    }
    #[setter]
    pub fn set_val_veh_base_cost(&mut self, new_value: f64) -> PyResult<()> {
        self.val_veh_base_cost = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_veh_cg_m(&self) -> PyResult<f64> {
        Ok(self.veh_cg_m)
    }
    #[setter]
    pub fn set_veh_cg_m(&mut self, new_value: f64) -> PyResult<()> {
        self.veh_cg_m = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_veh_kg(&self) -> PyResult<f64> {
        Ok(self.veh_kg)
    }
    #[setter]
    pub fn set_veh_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.veh_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_veh_override_kg(&self) -> PyResult<f64> {
        Ok(self.veh_override_kg)
    }
    #[setter]
    pub fn set_veh_override_kg(&mut self, new_value: f64) -> PyResult<()> {
        self.veh_override_kg = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_veh_pt_type(&self) -> PyResult<String> {
        Ok(self.veh_pt_type.clone())
    }
    #[setter]
    pub fn set_veh_pt_type(&mut self, new_value: String) -> PyResult<()> {
        self.veh_pt_type = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_veh_year(&self) -> PyResult<u32> {
        Ok(self.veh_year)
    }
    #[setter]
    pub fn set_veh_year(&mut self, new_value: u32) -> PyResult<()> {
        self.veh_year = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_wheel_base_m(&self) -> PyResult<f64> {
        Ok(self.wheel_base_m)
    }
    #[setter]
    pub fn set_wheel_base_m(&mut self, new_value: f64) -> PyResult<()> {
        self.wheel_base_m = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_wheel_coef_of_fric(&self) -> PyResult<f64> {
        Ok(self.wheel_coef_of_fric)
    }
    #[setter]
    pub fn set_wheel_coef_of_fric(&mut self, new_value: f64) -> PyResult<()> {
        self.wheel_coef_of_fric = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_wheel_inertia_kg_m2(&self) -> PyResult<f64> {
        Ok(self.wheel_inertia_kg_m2)
    }
    #[setter]
    pub fn set_wheel_inertia_kg_m2(&mut self, new_value: f64) -> PyResult<()> {
        self.wheel_inertia_kg_m2 = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_wheel_radius_m(&self) -> PyResult<f64> {
        Ok(self.wheel_radius_m)
    }
    #[setter]
    pub fn set_wheel_radius_m(&mut self, new_value: f64) -> PyResult<()> {
        self.wheel_radius_m = new_value;
        Ok(())
    }

    #[getter]
    pub fn get_wheel_rr_coef(&self) -> PyResult<f64> {
        Ok(self.wheel_rr_coef)
    }
    #[setter]
    pub fn set_wheel_rr_coef(&mut self, new_value: f64) -> PyResult<()> {
        self.wheel_rr_coef = new_value;
        Ok(())
    }
}

pub fn load_vehicle() -> RustVehicle {
    let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
    let selection: u32 = 5;
    let veh_year: u32 = 2016;
    let veh_pt_type = String::from("Conv");
    let drag_coef: f64 = 0.355;
    let frontal_area_m2: f64 = 3.066;
    let glider_kg: f64 = 1359.166;
    let veh_cg_m: f64 = 0.53;
    let drive_axle_weight_frac: f64 = 0.59;
    let wheel_base_m: f64 = 2.6;
    let cargo_kg: f64 = 136.0;
    let veh_override_kg: f64 = f64::NAN;
    let comp_mass_multiplier: f64 = 1.4;
    let fs_max_kw: f64 = 2000.0;
    let fs_secs_to_peak_pwr: f64 = 1.0;
    let fs_kwh: f64 = 504.0;
    let fs_kwh_per_kg: f64 = 9.89;
    let fc_max_kw: f64 = 125.0;
    let fc_pwr_out_perc: Vec<f64> = vec![
        0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
    ];
    let fc_eff_map: Vec<f64> = vec![
        0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
    ];
    let fc_eff_type: String = String::from("SI");
    let fc_sec_to_peak_pwr: f64 = 6.0;
    let fc_base_kg: f64 = 61.0;
    let fc_kw_per_kg: f64 = 2.13;
    let min_fc_time_on: f64 = 30.0;
    let idle_fc_kw: f64 = 2.5;
    let mc_max_kw: f64 = 0.0;
    let mc_pwr_out_perc: Vec<f64> = vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
    let mc_eff_map: Vec<f64> = vec![
        0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,
    ];
    let mc_sec_to_peak_pwr: f64 = 4.0;
    let mc_pe_kg_per_kw: f64 = 0.833;
    let mc_pe_base_kg: f64 = 21.6;
    let ess_max_kw: f64 = 0.0;
    let ess_max_kwh: f64 = 0.0;
    let ess_kg_per_kwh: f64 = 8.0;
    let ess_base_kg: f64 = 75.0;
    let ess_round_trip_eff: f64 = 0.97;
    let ess_life_coef_a: f64 = 110.0;
    let ess_life_coef_b: f64 = -0.6811;
    let min_soc: f64 = 0.4;
    let max_soc: f64 = 0.8;
    let ess_dischg_to_fc_max_eff_perc: f64 = 0.0;
    let ess_chg_to_fc_max_eff_perc: f64 = 0.0;
    let wheel_inertia_kg_m2: f64 = 0.815;
    let num_wheels: f64 = 4.0;
    let wheel_rr_coef: f64 = 0.006;
    let wheel_radius_m: f64 = 0.336;
    let wheel_coef_of_fric: f64 = 0.7;
    let max_accel_buffer_mph: f64 = 60.0;
    let max_accel_buffer_perc_of_useable_soc: f64 = 0.2;
    let perc_high_acc_buf: f64 = 0.0;
    let mph_fc_on: f64 = 30.0;
    let kw_demand_fc_on: f64 = 100.0;
    let max_regen: f64 = 0.98;
    let stop_start: bool = false;
    let force_aux_on_fc: bool = true;
    let alt_eff: f64 = 1.0;
    let chg_eff: f64 = 0.86;
    let aux_kw: f64 = 0.7;
    let trans_kg: f64 = 114.0;
    let trans_eff: f64 = 0.92;
    let ess_to_fuel_ok_error: f64 = 0.005;
    let val_udds_mpgge: f64 = 23.0;
    let val_hwy_mpgge: f64 = 32.0;
    let val_comb_mpgge: f64 = 26.0;
    let val_udds_kwh_per_mile: f64 = f64::NAN;
    let val_hwy_kwh_per_mile: f64 = f64::NAN;
    let val_comb_kwh_per_mile: f64 = f64::NAN;
    let val_cd_range_mi: f64 = f64::NAN;
    let val_const65_mph_kwh_per_mile: f64 = f64::NAN;
    let val_const60_mph_kwh_per_mile: f64 = f64::NAN;
    let val_const55_mph_kwh_per_mile: f64 = f64::NAN;
    let val_const45_mph_kwh_per_mile: f64 = f64::NAN;
    let val_unadj_udds_kwh_per_mile: f64 = f64::NAN;
    let val_unadj_hwy_kwh_per_mile: f64 = f64::NAN;
    let val0_to60_mph: f64 = 9.9;
    let val_ess_life_miles: f64 = f64::NAN;
    let val_range_miles: f64 = f64::NAN;
    let val_veh_base_cost: f64 = f64::NAN;
    let val_msrp: f64 = f64::NAN;
    let props = RustPhysicalProperties::__new__();
    let small_motor_power_kw: f64 = 7.5;
    let large_motor_power_kw: f64 = 75.0;
    // TODO: make this look more like:
    // fc_perc_out_array = np.r_[np.arange(0, 3.0, 0.1), np.arange(
    //     3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100  # hardcoded ***
    let fc_perc_out_array: Vec<f64> = FC_PERC_OUT_ARRAY.to_vec();
    let max_roadway_chg_kw: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let charging_on: bool = false;
    let no_elec_sys: bool = true;
    let no_elec_aux: bool = true;
    let modern_max: f64 = 0.95;
    let regen_a: f64 = 500.0;
    let regen_b: f64 = 0.99;
    let mc_max_elec_in_kw: f64 = 100.0;
    let ess_mass_kg: f64 = 0.0;
    // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
    let mc_mass_kg: f64 = 0.0;
    // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
    let fc_mass_kg: f64 = 0.0;
    // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
    let fs_mass_kg: f64 = 0.0;
    // DERIVED
    let input_kw_out_array = fc_pwr_out_perc.iter().map(|&x| x * fc_max_kw).collect();
    let fc_kw_out_array = fc_perc_out_array.iter().map(|&x| x * fc_max_kw).collect();
    let fc_eff_array = fc_perc_out_array
        .iter()
        .map(|&x| {
            interpolate(
                &x,
                &Array::from(fc_pwr_out_perc.clone()),
                &Array::from(fc_eff_map.clone()),
                false,
            )
        })
        .collect::<Vec<_>>();
    let mc_perc_out_array = MC_PERC_OUT_ARRAY.to_vec();
    let mc_kw_out_array = (Array::linspace(0.0, 1.0, mc_perc_out_array.len()) * mc_max_kw).to_vec();
    let mc_eff_array: Vec<f64> = LARGE_BASELINE_EFF
        .iter()
        .map(|&x| {
            interpolate(
                &x,
                &Array::from(mc_pwr_out_perc.clone()),
                &Array::from(mc_eff_map.clone()),
                false,
            )
        })
        .collect();
    let mc_kw_in_array = Array::ones(mc_kw_out_array.len()).to_vec();
    let veh_kg: f64 = 0.0;
    /*
    cargo_kg + glider_kg + trans_kg * comp_mass_multiplier
        + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg;
    */
    let max_trac_mps2: f64 =
        (wheel_coef_of_fric * drive_axle_weight_frac * veh_kg * props.a_grav_mps2
            / (1.0 + veh_cg_m * wheel_coef_of_fric / wheel_base_m))
            / (veh_kg * props.a_grav_mps2)
            * props.a_grav_mps2;

    RustVehicle::__new__(
        scenario_name,
        selection,
        veh_year,
        veh_pt_type,
        drag_coef,
        frontal_area_m2,
        glider_kg,
        veh_cg_m,
        drive_axle_weight_frac,
        wheel_base_m,
        cargo_kg,
        veh_override_kg,
        comp_mass_multiplier,
        fs_max_kw,
        fs_secs_to_peak_pwr,
        fs_kwh,
        fs_kwh_per_kg,
        fc_max_kw,
        fc_pwr_out_perc,
        fc_eff_map,
        fc_eff_type,
        fc_sec_to_peak_pwr,
        fc_base_kg,
        fc_kw_per_kg,
        min_fc_time_on,
        idle_fc_kw,
        mc_max_kw,
        mc_pwr_out_perc,
        mc_eff_map,
        mc_sec_to_peak_pwr,
        mc_pe_kg_per_kw,
        mc_pe_base_kg,
        ess_max_kw,
        ess_max_kwh,
        ess_kg_per_kwh,
        ess_base_kg,
        ess_round_trip_eff,
        ess_life_coef_a,
        ess_life_coef_b,
        min_soc,
        max_soc,
        ess_dischg_to_fc_max_eff_perc,
        ess_chg_to_fc_max_eff_perc,
        wheel_inertia_kg_m2,
        num_wheels,
        wheel_rr_coef,
        wheel_radius_m,
        wheel_coef_of_fric,
        max_accel_buffer_mph,
        max_accel_buffer_perc_of_useable_soc,
        perc_high_acc_buf,
        mph_fc_on,
        kw_demand_fc_on,
        max_regen,
        stop_start,
        force_aux_on_fc,
        alt_eff,
        chg_eff,
        aux_kw,
        trans_kg,
        trans_eff,
        ess_to_fuel_ok_error,
        val_udds_mpgge,
        val_hwy_mpgge,
        val_comb_mpgge,
        val_udds_kwh_per_mile,
        val_hwy_kwh_per_mile,
        val_comb_kwh_per_mile,
        val_cd_range_mi,
        val_const65_mph_kwh_per_mile,
        val_const60_mph_kwh_per_mile,
        val_const55_mph_kwh_per_mile,
        val_const45_mph_kwh_per_mile,
        val_unadj_udds_kwh_per_mile,
        val_unadj_hwy_kwh_per_mile,
        val0_to60_mph,
        val_ess_life_miles,
        val_range_miles,
        val_veh_base_cost,
        val_msrp,
        props,
        small_motor_power_kw,
        large_motor_power_kw,
        None,
        charging_on,
        no_elec_sys,
        no_elec_aux,
        max_roadway_chg_kw,
        input_kw_out_array,
        fc_kw_out_array,
        fc_eff_array,
        modern_max,
        mc_eff_array,
        mc_kw_in_array,
        mc_kw_out_array,
        mc_max_elec_in_kw,
        None,
        regen_a,
        regen_b,
        veh_kg,
        max_trac_mps2,
        ess_mass_kg,
        mc_mass_kg,
        fc_mass_kg,
        fs_mass_kg,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_derived_via_new() {
        let veh = load_vehicle();
        assert!(veh.veh_kg > 0.0);
    }
    // #[test]
    // fn test_veh_get_mc_peak_eff() {
    //     // VEHICLE
    //     let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
    //     let selection: u32 = 5;
    //     let veh_year: u32 = 2016;
    //     let veh_pt_type = String::from("Conv");
    //     let drag_coef: f64 = 0.355;
    //     let frontal_area_m2: f64 = 3.066;
    //     let glider_kg: f64 = 1359.166;
    //     let veh_cg_m: f64 = 0.53;
    //     let drive_axle_weight_frac: f64 = 0.59;
    //     let wheel_base_m: f64 = 2.6;
    //     let cargo_kg: f64 = 136.0;
    //     let veh_override_kg: f64 = f64::NAN;
    //     let comp_mass_multiplier: f64 = 1.4;
    //     let fs_max_kw: f64 = 2000.0;
    //     let fs_secs_to_peak_pwr: f64 = 1.0;
    //     let fs_kwh: f64 = 504.0;
    //     let fs_kwh_per_kg: f64 = 9.89;
    //     let fc_max_kw: f64 = 125.0;
    //     let fc_pwr_out_perc: Vec<f64> = vec![0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0];
    //     let fc_eff_map: Vec<f64> = vec![0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3];
    //     let fc_eff_type: String = String::from("SI");
    //     let fc_sec_to_peak_pwr: f64 = 6.0;
    //     let fc_base_kg: f64 = 61.0;
    //     let fc_kw_per_kg: f64 = 2.13;
    //     let min_fc_time_on: f64 = 30.0;
    //     let idle_fc_kw: f64 = 2.5;
    //     let mc_max_kw: f64 = 0.0;
    //     let mc_pwr_out_perc: Vec<f64> = vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
    //     let mc_eff_map: Vec<f64> = vec![0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92];
    //     let mc_sec_to_peak_pwr: f64 = 4.0;
    //     let mc_pe_kg_per_kw: f64 = 0.833;
    //     let mc_pe_base_kg: f64 = 21.6;
    //     let ess_max_kw: f64 = 0.0;
    //     let ess_max_kwh: f64 = 0.0;
    //     let ess_kg_per_kwh: f64 = 8.0;
    //     let ess_base_kg: f64 = 75.0;
    //     let ess_round_trip_eff: f64 = 0.97;
    //     let ess_life_coef_a: f64 = 110.0;
    //     let ess_life_coef_b: f64 = -0.6811;
    //     let min_soc: f64 = 0.4;
    //     let max_soc: f64 = 0.8;
    //     let ess_dischg_to_fc_max_eff_perc: f64 = 0.0;
    //     let ess_chg_to_fc_max_eff_perc: f64 = 0.0;
    //     let wheel_inertia_kg_m2: f64 = 0.815;
    //     let num_wheels: f64 = 4.0;
    //     let wheel_rr_coef: f64 = 0.006;
    //     let wheel_radius_m: f64 = 0.336;
    //     let wheel_coef_of_fric: f64 = 0.7;
    //     let max_accel_buffer_mph: f64 = 60.0;
    //     let max_accel_buffer_perc_of_useable_soc: f64 = 0.2;
    //     let perc_high_acc_buf: f64 = 0.0;
    //     let mph_fc_on: f64 = 30.0;
    //     let kw_demand_fc_on: f64 = 100.0;
    //     let max_regen: f64 = 0.98;
    //     let stop_start: bool = false;
    //     let force_aux_on_fc: bool = false;
    //     let alt_eff: f64 = 1.0;
    //     let chg_eff: f64 = 0.86;
    //     let aux_kw: f64 = 0.7;
    //     let trans_kg: f64 = 114.0;
    //     let trans_eff: f64 = 0.92;
    //     let ess_to_fuel_ok_error: f64 = 0.005;
    //     let val_udds_mpgge: f64 = 23.0;
    //     let val_hwy_mpgge: f64 = 32.0;
    //     let val_comb_mpgge: f64 = 26.0;
    //     let val_udds_kwh_per_mile: f64 = f64::NAN;
    //     let val_hwy_kwh_per_mile: f64 = f64::NAN;
    //     let val_comb_kwh_per_mile: f64 = f64::NAN;
    //     let val_cd_range_mi: f64 = f64::NAN;
    //     let val_const65_mph_kwh_per_mile: f64 = f64::NAN;
    //     let val_const60_mph_kwh_per_mile: f64 = f64::NAN;
    //     let val_const55_mph_kwh_per_mile: f64 = f64::NAN;
    //     let val_const45_mph_kwh_per_mile: f64 = f64::NAN;
    //     let val_unadj_udds_kwh_per_mile: f64 = f64::NAN;
    //     let val_unadj_hwy_kwh_per_mile: f64 = f64::NAN;
    //     let val0_to60_mph: f64 = 9.9;
    //     let val_ess_life_miles: f64 = f64::NAN;
    //     let val_range_miles: f64 = f64::NAN;
    //     let val_veh_base_cost: f64 = f64::NAN;
    //     let val_msrp: f64 = f64::NAN;
    //     let props = RustPhysicalProperties::__new__();
    //     let small_motor_power_kw: f64 = 7.5;
    //     let large_motor_power_kw: f64 = 75.0;
    //     let fc_perc_out_array: Vec<f64> = vec![
    //       0.0  , 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01 , 0.011,
    //       0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02 , 0.021, 0.022, 0.023,
    //       0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03 , 0.035, 0.04 , 0.045, 0.05 , 0.055,
    //       0.06 , 0.065, 0.07 , 0.08 , 0.09 , 0.1  , 0.11 , 0.12 , 0.13 , 0.14 , 0.15 , 0.16 ,
    //       0.17 , 0.18 , 0.19 , 0.2  , 0.21 , 0.22 , 0.23 , 0.24 , 0.25 , 0.26 , 0.27 , 0.28 ,
    //       0.29 , 0.3  , 0.31 , 0.32 , 0.33 , 0.34 , 0.35 , 0.36 , 0.37 , 0.38 , 0.39 , 0.4  ,
    //       0.41 , 0.42 , 0.43 , 0.44 , 0.45 , 0.46 , 0.47 , 0.48 , 0.49 , 0.5  , 0.51 , 0.52 ,
    //       0.53 , 0.54 , 0.55 , 0.56 , 0.57 , 0.58 , 0.59 , 0.6  , 0.65 , 0.7  , 0.75 , 0.8  ,
    //       0.85 , 0.9  , 0.95 , 1.0
    //     ];
    //     let max_roadway_chg_kw: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    //     let charging_on: bool = false;
    //     let no_elec_sys: bool = true;
    //     let no_elec_aux: bool = true;
    //     let veh = RustVehicle::__new__(
    //       scenario_name,
    //       selection,
    //       veh_year,
    //       veh_pt_type,
    //       drag_coef,
    //       frontal_area_m2,
    //       glider_kg,
    //       veh_cg_m,
    //       drive_axle_weight_frac,
    //       wheel_base_m,
    //       cargo_kg,
    //       veh_override_kg,
    //       comp_mass_multiplier,
    //       fs_max_kw,
    //       fs_secs_to_peak_pwr,
    //       fs_kwh,
    //       fs_kwh_per_kg,
    //       fc_max_kw,
    //       fc_pwr_out_perc,
    //       fc_eff_map,
    //       fc_eff_type,
    //       fc_sec_to_peak_pwr,
    //       fc_base_kg,
    //       fc_kw_per_kg,
    //       min_fc_time_on,
    //       idle_fc_kw,
    //       mc_max_kw,
    //       mc_pwr_out_perc,
    //       mc_eff_map,
    //       mc_sec_to_peak_pwr,
    //       mc_pe_kg_per_kw,
    //       mc_pe_base_kg,
    //       ess_max_kw,
    //       ess_max_kwh,
    //       ess_kg_per_kwh,
    //       ess_base_kg,
    //       ess_round_trip_eff,
    //       ess_life_coef_a,
    //       ess_life_coef_b,
    //       min_soc,
    //       max_soc,
    //       ess_dischg_to_fc_max_eff_perc,
    //       ess_chg_to_fc_max_eff_perc,
    //       wheel_inertia_kg_m2,
    //       num_wheels,
    //       wheel_rr_coef,
    //       wheel_radius_m,
    //       wheel_coef_of_fric,
    //       max_accel_buffer_mph,
    //       max_accel_buffer_perc_of_useable_soc,
    //       perc_high_acc_buf,
    //       mph_fc_on,
    //       kw_demand_fc_on,
    //       max_regen,
    //       stop_start,
    //       force_aux_on_fc,
    //       alt_eff,
    //       chg_eff,
    //       aux_kw,
    //       trans_kg,
    //       trans_eff,
    //       ess_to_fuel_ok_error,
    //       val_udds_mpgge,
    //       val_hwy_mpgge,
    //       val_comb_mpgge,
    //       val_udds_kwh_per_mile,
    //       val_hwy_kwh_per_mile,
    //       val_comb_kwh_per_mile,
    //       val_cd_range_mi,
    //       val_const65_mph_kwh_per_mile,
    //       val_const60_mph_kwh_per_mile,
    //       val_const55_mph_kwh_per_mile,
    //       val_const45_mph_kwh_per_mile,
    //       val_unadj_udds_kwh_per_mile,
    //       val_unadj_hwy_kwh_per_mile,
    //       val0_to60_mph,
    //       val_ess_life_miles,
    //       val_range_miles,
    //       val_veh_base_cost,
    //       val_msrp,
    //       props,
    //       small_motor_power_kw,
    //       large_motor_power_kw,
    //       fc_perc_out_array,
    //       charging_on,
    //       no_elec_sys,
    //       no_elec_aux,
    //       max_roadway_chg_kw,
    //     );

    //     let mc_peak_eff = veh.get_mc_peak_eff_rust();
    //     let expected_mc_peak_eff: f64 = 0.93;
    //     assert_eq!(mc_peak_eff, expected_mc_peak_eff);
    // }
}
