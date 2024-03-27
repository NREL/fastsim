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
    pub specific_energy: Option<si::AvailableEnergy>,
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
        self.check_mass_consistent()?;
        Ok(self.mass)
    }

    fn set_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()> {
        self.mass = match mass {
            Some(mass) => {
                self.specific_energy = Some(self.energy_capacity / mass);
                Some(mass)
            },
            None => {
                Some(self.energy_capacity / self.specific_energy.with_context(|| format!(
                    "{}\n{}",
                    format_dbg!(),
                    "`mass` must be provided, or `self.specific_energy` must be set")
                )?)
            },
        };
        Ok(())
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        if self.mass.is_some() && self.specific_energy.is_some() {
            ensure!(
                self.energy_capacity / self.specific_energy.unwrap() == self.mass.unwrap(),
                "{}\n{}",
                format_dbg!(),
                "`energy_capacity`, `specific_energy`, and `mass` fields are not consistent"
            )
        }
        Ok(())
    }
}
