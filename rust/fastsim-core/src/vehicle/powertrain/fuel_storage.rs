use super::*;

#[pyo3_api(
    // #[setter("__mass_kg")]
    // fn update_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
    //     self.update_mass(mass_kg.map(|m| m * uc::KG))?;
    //     Ok(())
    // }

    // #[getter("mass_kg")]
    // fn get_mass_py(&self) -> PyResult<Option<f64>> {
    //     Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    // }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods, SerdeAPI)]
pub struct FuelStorage {
    /// max power output
    pub pwr_out_max: si::Power,
    /// time to peak power
    pub t_to_peak_pwr: si::Time,
    /// energy capacity
    pub energy_capacity: si::Energy,
    /// Fuel and tank specific energy \[J/kg\]
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specific_energy: Option<si::AvailableEnergy>,
    /// Mass of fuel storage
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub mass: Option<si::Mass>,
}

impl Mass for FuelStorage {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.check_mass_consistent()?;
        Ok(self.mass)
    }

    fn update_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()> {
        match mass {
            Some(mass) => {
                self.specific_energy = Some(self.energy_capacity / mass);
                self.mass = Some(mass)
            }
            None => match self.specific_energy {
                Some(e) => self.mass = Some(self.energy_capacity / e),
                None => {
                    bail!(format!(
                        "{}\n{}",
                        format_dbg!(),
                        "Mass must be provided or `self.specific_energy` must be set"
                    ));
                }
            },
        }

        Ok(())
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        match &self.mass {
            Some(mass) => match &self.specific_energy {
                Some(e) => {
                    ensure!(self.energy_capacity / *e == *mass,
                    format!("{}\n{}", format_dbg!(), "ReversibleEnergyStorage `energy_capacity`, `specific_energy` and `mass` are not consistent"))
                }
                None => {}
            },
            None => {}
        }
        Ok(())
    }
}
