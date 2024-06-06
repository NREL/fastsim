use super::imports::*;
use super::*;

/// Returns density of air  
/// Source: <https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html>  
///
/// # Equations used
/// T = 15.04 - .00649 * h  
/// p = 101.29 * [(T + 273.1)/288.08]^5.256  
///
/// # Arguments  
/// * `te_air` - ambient temperature of air, defaults to 22 C
/// * `h` - evelation above sea level, defaults to 180 m
pub fn get_density_air(
    te_air: Option<si::ThermodynamicTemperature>,
    h: Option<si::Length>,
) -> si::MassDensity {
    let te_air = te_air.unwrap_or((22. + 273.15) * uc::KELVIN);
    let h = h.unwrap_or(180. * uc::M);
    let cur_elevation_std_temp = (15.04 - 0.00649 * h.get::<si::meter>() + 273.15) * uc::KELVIN;
    let cur_pressure = (101.29e3 * uc::PASCAL)
        * ((cur_elevation_std_temp / (288.08 * uc::KELVIN))
            .get::<si::ratio>()
            .powf(5.256));
    cur_pressure / (287.0 * uc::M2PS2K) / te_air
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "get_density_air")]
/// Returns density of air [kg/m^3]
/// Source: <https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html>  
///
/// # Equations used
/// T = 15.04 - .00649 * h  
/// p = 101.29 * [(T + 273.1)/288.08]^5.256  
///
/// # Arguments  
/// * `te_air_deg_c` - optional ambient temperature [Celsius] of air, defaults to 22 C
/// * `h_m` - optional elevation [m] above sea level, defaults to 180 m
pub fn get_density_air_py(te_air_deg_c: Option<f64>, h_m: Option<f64>) -> f64 {
    get_density_air(
        te_air_deg_c.map(|te_air_deg_c| (te_air_deg_c + 273.15) * uc::KELVIN),
        h_m.map(|h_m| h_m * uc::M),
    )
    .get::<si::kilogram_per_cubic_meter>()
}
