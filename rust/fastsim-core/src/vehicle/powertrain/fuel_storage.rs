use super::*;

pub struct FuelStorage {
    /// Fuel storage max power output, $kW$
    pub max_kw: si::Power,
    /// Fuel storage time to peak power
    pub t_to_peak_pwr: si::Time,
    /// Fuel storage energy capacity
    pub energy_capacity: f64,
    /// Fuel specific energy
    pub specific_energy: si::Pressure,
}
