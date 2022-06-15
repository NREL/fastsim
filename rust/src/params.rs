/// module containing parameters that are used in fastsim
extern crate pyo3;
use pyo3::prelude::*;

/// Unit conversions
pub const MPH_PER_MPS: f64 = 2.2369;
pub const M_PER_MI: f64 = 1609.00;

/// Misc Constants
pub const MODERN_MAX: f64 = 0.95;

/// Struct containing time trace data
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
#[allow(non_snake_case)]
pub struct RustPhysicalProperties {
    #[pyo3(get, set)] // enables get/set access from python for simple data types
    pub air_density_kg_per_m3: f64, // = 1.2, Sea level air density at approximately 20C
    #[pyo3(get, set)]
    pub a_grav_mps2: f64, // = 9.81
    #[pyo3(get, set)]
    pub kwh_per_gge: f64, // = 33.7 # kWh per gallon of gasoline
    #[pyo3(get, set)]
    pub fuel_rho_kg__L: f64, // = 0.75 # gasoline density in kg/L https://inchem.org/documents/icsc/icsc/eics1400.htm
    #[pyo3(get, set)]
    pub fuel_afr_stoich: f64, // = 14.7 # gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio
}

#[pymethods]
#[allow(non_snake_case)]
impl RustPhysicalProperties {
    #[new]
    pub fn __new__() -> Self {
        let air_density_kg_per_m3: f64 = 1.2;
        let a_grav_mps2: f64 = 9.81;
        let kwh_per_gge: f64 = 33.7;
        let fuel_rho_kg__L: f64 = 0.75;
        let fuel_afr_stoich: f64 = 14.7;
        RustPhysicalProperties {
            air_density_kg_per_m3,
            a_grav_mps2,
            kwh_per_gge,
            fuel_rho_kg__L,
            fuel_afr_stoich,
        }
    }
}

// Vehicle model parameters that should be changed only by advanced users

/// Relatively continuous power out percentages for assigning FC efficiencies
pub const FC_PERC_OUT_ARRAY: [f64; 100] = [
    0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013,
    0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026,
    0.027, 0.028, 0.029, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09, 0.1,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
    0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42,
    0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58,
    0.59, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.,
];

pub const MC_PERC_OUT_ARRAY: [f64; 101] = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31,
    0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
    0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63,
    0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99, 1.,
];

pub const SMALL_MOTOR_POWER_KW: f64 = 7.5;
pub const LARGE_MOTOR_POWER_KW: f64 = 75.0;

pub const LARGE_BASELINE_EFF: [f64; 11] = [
    0.83, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.94, 0.94, 0.93, 0.92,
];

pub const SMALL_BASELINE_EFF: [f64; 11] = [
    0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,
];
