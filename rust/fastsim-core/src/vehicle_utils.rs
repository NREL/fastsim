//! Module for utility functions that support the vehicle struct.

use polynomial::Polynomial;

use crate::imports::*;
use crate::air::*;
use crate::params::*;
use crate::cycle::RustCycle;
use crate::simdrive::RustSimDrive;
use crate::vehicle::RustVehicle;

#[allow(non_snake_case)]
pub fn abc_to_drag_coeffs(veh: &mut RustVehicle,
                          a_lbf: f64, b_lbf__mph: f64, c_lbf__mph2: f64,
                          custom_rho: Option<bool>,
                          custom_rho_temp_degC: Option<f64>,
                          custom_rho_elevation_m: Option<f64>,
                          simdrive_optimize: Option<bool>,
                          show_plots: Option<bool>) -> (f64, f64) {
    // For a given vehicle and target A, B, and C coefficients;
    // calculate and return drag and rolling resistance coefficients.
    //
    // Arguments:
    // ----------
    // veh: vehicle.RustVehicle with all parameters correct except for drag and rolling resistance coefficients
    // a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    // custom_rho: if True, use `air::get_rho()` to calculate the current ambient density
    // custom_rho_temp_degC: ambient temperature [degree C] for `get_rho()`; 
    //     will only be used when `custom_rho` is True
    // custom_rho_elevation_m: location elevation [degree C] for `get_rho()`; 
    //     will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
    // simdrive_optimize: if True, use `SimDrive` to optimize the drag and rolling resistance; 
    //     otherwise, directly use target A, B, C to calculate the results
    // show_plots: if True, plots are shown
    
    let air_props: AirProperties = AirProperties::default();
    let props: RustPhysicalProperties = RustPhysicalProperties::default();
    let cur_ambient_air_density_kg__m3: f64 = if custom_rho.unwrap_or(false) 
        {air_props.get_rho(custom_rho_temp_degC.unwrap_or(20.0), custom_rho_elevation_m)}
        else {props.air_density_kg_per_m3};

    let vmax_mph: f64 = 70.0;
    let a_newton: f64 = a_lbf * super::params::N_PER_LBF;
    let b_newton__mps: f64 = b_lbf__mph * super::params::N_PER_LBF * super::params::MPH_PER_MPS;
    let c_newton__mps2: f64 = c_lbf__mph2 * super::params::N_PER_LBF * super::params::MPH_PER_MPS * super::params::MPH_PER_MPS; 
    
    let cd_len: usize = 300;

    let cyc: RustCycle = RustCycle::new((0..cd_len as i32).map(f64::from).collect(),
        Array::linspace(vmax_mph / super::params::MPH_PER_MPS, 0.0, cd_len).to_vec(),
        vec![0.0; cd_len], vec![0.0; cd_len], String::from("cycle"));

    // polynomial function for pounds vs speed
    let dyno_func_lb: Polynomial<f64> = Polynomial::new(vec![a_lbf, b_lbf__mph, c_lbf__mph2]);

    let drag_coef: f64;
    let wheel_rr_coef: f64;

    drag_coef = c_newton__mps2 / (0.5 * veh.frontal_area_m2 * cur_ambient_air_density_kg__m3);
    wheel_rr_coef = a_newton / veh.veh_kg / props.a_grav_mps2;

    veh.drag_coef = drag_coef;
    veh.wheel_rr_coef = wheel_rr_coef;

    return (drag_coef, wheel_rr_coef);
}