use super::*;

#[pyo3_api(
    // #[setter("__mass_kg")]
    // fn set_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
    //     self.set_mass(mass_kg.map(|m| m * uc::KG))?;
    //     Ok(())
    // }

    // #[getter("mass_kg")]
    // fn get_mass_py(&self) -> PyResult<Option<f64>> {
    //     Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    // }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, SerdeAPI)]
pub struct FuelStorage {
    /// max power output
    pub pwr_out_max: si::Power,
    /// time to peak power
    pub pwr_ramp_lag: si::Time,
    /// energy capacity
    pub energy_capacity: si::Energy,
    /// Fuel and tank specific energy
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specific_energy: Option<si::SpecificEnergy>,
    /// Mass of fuel storage
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub(in super::super) mass: Option<si::Mass>,
    // TODO: add state to track fuel level and make sure mass changes propagate up to vehicle level,
    // which should then include vehicle mass in state
}

impl Mass for FuelStorage {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self.derived_mass()?;
        if let (Some(derived_mass), Some(set_mass)) = (derived_mass, self.mass) {
            ensure!(
                utils::almost_eq_uom(&set_mass, &derived_mass, None),
                format!(
                    "{}",
                    format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                )
            );
        }
        Ok(self.mass)
    }

    fn set_mass(&mut self, new_mass: Option<si::Mass>) -> anyhow::Result<()> {
        let derived_mass = self.derived_mass()?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                log::info!(
                    "Derived mass from `self.specific_energy` and `self.energy_capacity` does not match {}",
                    "provided mass, setting `self.specific_energy` to be consistent with provided mass"
                );
                self.specific_energy = Some(self.energy_capacity / new_mass);
            }
        } else if let None = new_mass {
            log::debug!("Provided mass is None, setting `self.specific_energy` to None");
            self.specific_energy = None;
        }
        self.mass = new_mass;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self
            .specific_energy
            .map(|specific_energy| self.energy_capacity / specific_energy))
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
        self.specific_energy = None;
    }
}
