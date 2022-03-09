extern crate ndarray;
//use ndarray::{Array, Array1}; 
use ndarray::{Array1}; 
extern crate pyo3;
use pyo3::prelude::*;
// use numpy::pyo3::Python;
// use numpy::ndarray::array;
// use numpy::{ToPyArray, PyArray};

// local 
use super::params::*;
//use super::utils::*;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustVehicle{
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
    max_fuel_stor_kw: f64,
    fuel_stor_secs_to_peak_pwr: f64,
    fuel_stor_kwh: f64,
    fuel_stor_kwh_per_kg: f64,
    max_fuel_conv_kw: f64,
    fc_pwr_out_perc: f64,
    fc_eff_map: Array1<f64>,
    fc_eff_type: String,
    fuel_conv_secs_to_peak_pwr: f64,
    fuel_conv_base_kg: f64,
    fuel_conv_kw_per_kg: f64,
    min_fc_time_on: f64,
    idle_fc_kw: f64,
    max_motor_kw: f64,
    mc_pwr_out_perc: Array1<f64>,
    mc_eff_map: Array1<f64>,
    motor_secs_to_peak_pwr: f64,
    mc_pe_kg_per_kw: f64,
    mc_pe_base_kg: f64,
    max_ess_kw: f64,
    max_ess_kwh: f64,
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
    max_regen: bool,
    stop_start: bool,
    force_aux_on_fc: f64,
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
    props: RustPhysicalProperties,  // todo: implement this
    large_baseline_eff: Array1<f64>,
    small_baseline_eff: Array1<f64>,
    small_motor_power_kw: f64,
    large_motor_power_kw: f64,
    fc_perc_out_array: Array1<f64>,
    regen_a: f64,
    regen_b: f64,
    charging_on: bool,
    no_elec_sys: bool,
    no_elec_aux: bool,
    max_roadway_chg_kw: Array1<f64>,
    input_kw_out_array: Array1<f64>,
    fc_kw_out_array: Array1<f64>,
    fc_eff_array: Array1<f64>,
    modern_max: f64,
    mc_eff_array: Array1<f64>,
    mc_kw_in_array: Array1<f64>,
    mc_kw_out_array: Array1<f64>,
    mc_max_elec_in_kw: f64,
    mc_full_eff_array: Array1<f64>,
    veh_kg: f64,
    max_trac_mps2: f64,
    ess_mass_kg: f64,
    mc_mass_kg: f64,
    fc_mass_kg: f64,
    fs_mass_kg: f64,
    mc_perc_out_array: Array1<f64>,
}

/// RustVehicle class for containing: 
#[pymethods]
impl RustVehicle{
    #[new]
    pub fn __new__() -> Self{
        let scenario_name = String::from("test");
        let selection: u32 = 1;
        let veh_year: u32 = 2022;
        let veh_pt_type = String::from("Conv");
        let drag_coef: f64 = 0.25;
        let frontal_area_m2: f64 = 2.0;
        let glider_kg: f64 = 1000.0;
        let veh_cg_m: f64 = 0.25;
        let drive_axle_weight_frac: f64 = 0.5;
        let wheel_base_m: f64 = 2.5;
        let cargo_kg: f64 = 200.0;
        let veh_override_kg: f64 = 1200.0;
        let comp_mass_multiplier: f64 = 1.0;
        let max_fuel_stor_kw: f64 = 50.0;
        let fuel_stor_secs_to_peak_pwr: f64 = 1.0;
        let fuel_stor_kwh: f64 = 100.0;
        let fuel_stor_kwh_per_kg: f64 = 1.0;
        let max_fuel_conv_kw: f64 = 100.0;
        let fc_pwr_out_perc: f64 = 0.20;
        let fc_eff_map = Array1::<f64>::range(0.0, 10.0, 1.0);
        let fc_eff_type = String::from("SI");
        let fuel_conv_secs_to_peak_pwr: f64 = 1.0;
        let fuel_conv_base_kg: f64 = 100.0;
        let fuel_conv_kw_per_kg: f64 = 1.0;
        let min_fc_time_on: f64 = 10.0;
        let idle_fc_kw: f64 = 1.0;
        let max_motor_kw: f64 = 100.0;
        let mc_pwr_out_perc = Array1::<f64>::range(0.0, 10.0, 1.0);
        let mc_eff_map = Array1::<f64>::range(0.0, 1.0, 0.01);
        let motor_secs_to_peak_pwr: f64 = 1.0;
        let mc_pe_kg_per_kw: f64 = 1.0;
        let mc_pe_base_kg: f64 = 1.0;
        let max_ess_kw: f64 = 100.0;
        let max_ess_kwh: f64 = 100.0;
        let ess_kg_per_kwh: f64 = 1.0;
        let ess_base_kg: f64 = 10.0;
        let ess_round_trip_eff: f64 = 0.90;
        let ess_life_coef_a: f64 = 1.0;
        let ess_life_coef_b: f64 = 1.0;
        let min_soc: f64 = 0.0;
        let max_soc: f64 = 1.0;
        let ess_dischg_to_fc_max_eff_perc: f64 = 1.0;
        let ess_chg_to_fc_max_eff_perc: f64 = 1.0;
        let wheel_inertia_kg_m2: f64 = 4.0;
        let num_wheels: f64 = 4.0;
        let wheel_rr_coef: f64 = 0.01;
        let wheel_radius_m: f64 = 0.2;
        let wheel_coef_of_fric: f64 = 0.5;
        let max_accel_buffer_mph: f64 = 55.0;
        let max_accel_buffer_perc_of_useable_soc: f64 = 0.9;
        let perc_high_acc_buf: f64 = 0.9;
        let mph_fc_on: f64 = 10.0;
        let kw_demand_fc_on: f64 = 120.0;
        let max_regen = false;
        let stop_start = false;
        let force_aux_on_fc: f64 = 5.0;
        let alt_eff: f64 = 0.8;
        let chg_eff: f64 = 0.95;
        let aux_kw: f64 = 2.0;
        let trans_kg: f64 = 80.0;
        let trans_eff: f64 = 0.98;
        let ess_to_fuel_ok_error: f64 = 1.0;
        let val_udds_mpgge: f64 = 0.0;
        let val_hwy_mpgge: f64 = 0.0;
        let val_comb_mpgge: f64 = 0.0;
        let val_udds_kwh_per_mile: f64 = 0.0;
        let val_hwy_kwh_per_mile: f64 = 0.0;
        let val_comb_kwh_per_mile: f64 = 0.0;
        let val_cd_range_mi: f64 = 0.0;
        let val_const65_mph_kwh_per_mile: f64 = 0.0;
        let val_const60_mph_kwh_per_mile: f64 = 0.0;
        let val_const55_mph_kwh_per_mile: f64 = 0.0;
        let val_const45_mph_kwh_per_mile: f64 = 0.0;
        let val_unadj_udds_kwh_per_mile: f64 = 0.0;
        let val_unadj_hwy_kwh_per_mile: f64 = 0.0;
        let val0_to60_mph: f64 = 0.0;
        let val_ess_life_miles: f64 = 0.0;
        let val_range_miles: f64 = 0.0;
        let val_veh_base_cost: f64 = 0.0;
        let val_msrp: f64 = 0.0;
        let props = RustPhysicalProperties{
            air_density_kg_per_m3:1.2,  // Sea level air density at approximately 20C
            a_grav_mps2: 9.81,          // acceleration due to gravity (m/s2)
            kwh_per_gge: 33.7,          // kWh per gallon of gasoline
            fuel_rho_kg__L: 0.75,       // gasoline density in kg/L https://inchem.org/documents/icsc/icsc/eics1400.htm
            fuel_afr_stoich:14.7,       // gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio
        };
        let large_baseline_eff = Array1::<f64>::range(1.0, 10.0, 1.0);
        let small_baseline_eff = Array1::<f64>::range(1.0, 10.0, 1.0);
        let small_motor_power_kw: f64 = 30.0;
        let large_motor_power_kw: f64 = 120.0;
        let fc_perc_out_array = Array1::<f64>::range(1.0, 10.0, 1.0);
        let regen_a: f64 = 1.0;
        let regen_b: f64 = 1.0;
        let charging_on = true;
        let no_elec_sys = true;
        let no_elec_aux = true;
        let max_roadway_chg_kw = Array1::<f64>::zeros(6);
        let input_kw_out_array = Array1::<f64>::zeros(6);
        let fc_kw_out_array = Array1::<f64>::zeros(6);
        let fc_eff_array = Array1::<f64>::zeros(6);
        let modern_max: f64 = 1.0;
        let mc_eff_array = Array1::<f64>::zeros(6);
        let mc_kw_in_array = Array1::<f64>::zeros(6);
        let mc_kw_out_array = Array1::<f64>::zeros(6);
        let mc_max_elec_in_kw: f64 = 100.0;
        let mc_full_eff_array = Array1::<f64>::zeros(6);
        let veh_kg: f64 = 1200.0;
        let max_trac_mps2: f64 = 1.0;
        let ess_mass_kg: f64 = 20.0;
        let mc_mass_kg: f64 = 20.0;
        let fc_mass_kg: f64 = 80.0;
        let fs_mass_kg: f64 = 15.0;
        let mc_perc_out_array = Array1::<f64>::zeros(6);
        RustVehicle {
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
            max_fuel_stor_kw,
            fuel_stor_secs_to_peak_pwr,
            fuel_stor_kwh,
            fuel_stor_kwh_per_kg,
            max_fuel_conv_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fuel_conv_secs_to_peak_pwr,
            fuel_conv_base_kg,
            fuel_conv_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            max_motor_kw,
            mc_pwr_out_perc,
            mc_eff_map,
            motor_secs_to_peak_pwr,
            mc_pe_kg_per_kw,
            mc_pe_base_kg,
            max_ess_kw,
            max_ess_kwh,
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
            large_baseline_eff,
            small_baseline_eff,
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
        }
    }
}
