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
pub fn get_rho_air(
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
