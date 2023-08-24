use super::*;

#[pyo3_api(
    #[setter("__mass_kg")]
    fn update_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
        self.update_mass(mass_kg.map(|m| m * uc::KG))?;
        Ok(())
    }

    #[getter("mass_kg")]
    fn get_mass_py(&self) -> PyResult<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_pwr_kw_per_kg(&self) -> Option<f64> {
        self.specific_pwr_kw_per_kg
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods, SerdeAPI)]
pub struct FuelStorage {
    /// max power output, $kW$
    pub pwr_out_max: si::Power,
    /// time to peak power
    pub t_to_peak_pwr: si::Time,
    /// energy capacity
    pub energy_capacity: si::Energy,
    /// Fuel and tank specific energy  
    ///
    /// Note that this is `si::Ratio` because the poorly named `si::AvailableEnergy` has a bug:  
    /// https://github.com/iliekturtles/uom/issues/435
    #[api(skip_get, skip_set)]
    pub specific_energy: Option<f64>,
    /// specific power
    /// TODO: make this si::specific_power after Geordie's pull request into `uom`
    #[api(skip_get, skip_set)]
    specific_pwr_kw_per_kg: Option<f64>,
    /// Mass of fuel storage
    #[serde(default)]
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
                self.specific_pwr_kw_per_kg =
                    Some(self.pwr_out_max.get::<si::kilowatt>() / mass.get::<si::kilogram>());
                self.mass = Some(mass);
            }
            None => match self.specific_pwr_kw_per_kg {
                Some(spec_pwr_kw_per_kg) => {
                    self.mass = Some(self.pwr_out_max / (spec_pwr_kw_per_kg * uc::KW / uc::KG));
                }
                None => {
                    bail!(format!(
                        "{}\n{}",
                        format_dbg!(),
                        "Mass must be provided or `self.specific_pwr_kw_per_kg` must be set"
                    ));
                }
            },
        }

        Ok(())
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        match &self.mass {
            Some(mass) => match &self.specific_pwr_kw_per_kg {
                Some(spec_pwr_kw_per_kg) => {
                    ensure!(self.pwr_out_max / (*spec_pwr_kw_per_kg * uc::KW / uc::KG) == *mass,
                    format!("{}\n{}", 
                        format_dbg!(),
                        "FuelConverter `pwr_out_max`, `specific_pwr_kw_per_kg` and `mass` are not consistent"))
                }
                None => {}
            },
            None => {}
        }
        Ok(())
    }
}
