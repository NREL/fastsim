use super::*;

#[pyo3_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods, SerdeAPI)]
pub struct FuelStorage {
    /// Fuel storage max power output, $kW$
    pub max_kw: si::Power,
    /// Fuel storage time to peak power
    pub t_to_peak_pwr: si::Time,
    /// Fuel storage energy capacity
    pub energy_capacity: si::Energy,
    /// Fuel specific energy  
    ///
    /// Note that this is `si::Ratio` because the poorly named `si::AvailableEnergy` has a bug:  
    /// https://github.com/iliekturtles/uom/issues/435
    pub specific_energy: si::Ratio,
    /// Mass of fuel storage
    pub mass: si::Mass,
}
