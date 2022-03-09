/// module containing parameters that are used in fastsim
extern crate pyo3;
use pyo3::prelude::*;

/// Unit conversions
pub const MPH_PER_MPS:f64 = 2.2369;
pub const METERS_PER_MILE:f64 = 1609.00;

#[pyclass] 
#[derive(Debug, Clone)]
/// Struct containing time trace data 
pub struct RustPhysicalProperties{
    pub air_density_kg_per_m3:f64, // = 1.2, Sea level air density at approximately 20C
    pub a_grav_mps2:f64, // = 9.81
    pub kwh_per_gge:f64, // = 33.7 # kWh per gallon of gasoline
    pub fuel_rho_kg__L:f64, // = 0.75 # gasoline density in kg/L https://inchem.org/documents/icsc/icsc/eics1400.htm
    pub fuel_afr_stoich:f64 // = 14.7 # gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio
}