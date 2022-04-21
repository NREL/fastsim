/// module containing parameters that are used in fastsim
extern crate pyo3;
use pyo3::prelude::*;

/// Unit conversions
pub const MPH_PER_MPS:f64 = 2.2369;
pub const M_PER_MI:f64 = 1609.00;

/// Misc Constants
pub const MODERN_MAX:f64 = 0.95;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustPhysicalProperties{
    #[pyo3(get, set)]  // enables get/set access from python for simple data types
    pub air_density_kg_per_m3:f64, // = 1.2, Sea level air density at approximately 20C
    #[pyo3(get, set)]
    pub a_grav_mps2:f64, // = 9.81
    #[pyo3(get, set)]
    pub kwh_per_gge:f64, // = 33.7 # kWh per gallon of gasoline
    #[pyo3(get, set)]
    pub fuel_rho_kg__L:f64, // = 0.75 # gasoline density in kg/L https://inchem.org/documents/icsc/icsc/eics1400.htm
    #[pyo3(get, set)]
    pub fuel_afr_stoich:f64 // = 14.7 # gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio
}

#[pymethods]
impl RustPhysicalProperties{
    #[new]
    pub fn __new__() -> Self{
        let air_density_kg_per_m3: f64 = 1.2;
        let a_grav_mps2: f64 = 9.81;
        let kwh_per_gge: f64 = 33.7;
        let fuel_rho_kg__L: f64 = 0.75;
        let fuel_afr_stoich: f64 = 14.7;
        RustPhysicalProperties{
            air_density_kg_per_m3, a_grav_mps2, kwh_per_gge, fuel_rho_kg__L, fuel_afr_stoich
        }
    }
}