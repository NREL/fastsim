use super::*;

// TODO: think about how to incorporate life modeling for Fuel Cells and other tech

const TOL: f64 = 1e-3;

#[altrios_api(
    // optional, custom, struct-specific pymethods
    #[getter("eta_max")]
    fn get_eta_max_py(&self) -> f64 {
        self.get_eta_max()
    }

    #[setter("__eta_max")]
    fn set_eta_max_py(&mut self, eta_max: f64) -> PyResult<()> {
        self.set_eta_max(eta_max).map_err(PyValueError::new_err)
    }

    #[getter("eta_min")]
    fn get_eta_min_py(&self) -> f64 {
        self.get_eta_min()
    }

    #[getter("eta_range")]
    fn get_eta_range_py(&self) -> f64 {
        self.get_eta_range()
    }

    #[setter("__eta_range")]
    fn set_eta_range_py(&mut self, eta_range: f64) -> PyResult<()> {
        self.set_eta_range(eta_range).map_err(PyValueError::new_err)
    }

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
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling Fuel Converter (e.g. engine, fuel cell.)
pub struct FuelConverter {
    #[serde(default)]
    /// struct for tracking current state
    pub state: FuelConverterState,
    /// FuelConverter mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    mass: Option<si::Mass>,
    /// FuelConverter specific power
    /// TODO: make this si::specific_power after Geordie's pull request into `uom`
    #[api(skip_get, skip_set)]
    specific_pwr_kw_per_kg: Option<f64>,
    #[serde(rename = "pwr_out_max_watts")]
    /// max rated brake output power
    pub pwr_out_max: si::Power,
    /// starting/baseline transient power limit
    #[serde(default)]
    pub pwr_out_max_init: si::Power,
    // TODO: consider a ramp down rate, which may be needed for fuel cells
    #[serde(rename(
        serialize = "pwr_ramp_lag_seconds",
        deserialize = "pwr_ramp_lag_seconds"
    ))]
    /// lag time for ramp up
    pub pwr_ramp_lag: si::Time,
    /// Fuel converter brake power fraction array at which efficiencies are evaluated.
    /// This fuel converter efficiency model assumes that speed and load (or voltage and current) will
    /// always be controlled for operating at max possible efficiency for the power demand
    pub pwr_out_frac_interp: Vec<f64>,
    /// fuel converter efficiency array
    pub eta_interp: Vec<f64>,
    /// idle fuel power to overcome internal friction (not including aux load) \[W\]
    #[serde(rename = "pwr_idle_fuel_watts")]
    pub pwr_idle_fuel: si::Power,
    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: FuelConverterStateHistoryVec, // TODO: spec out fuel tank size and track kg of fuel
}

impl Default for FuelConverter {
    fn default() -> Self {
        let file_contents = include_str!("fuel_converter.default.yaml");
        serde_yaml::from_str::<FuelConverter>(file_contents).unwrap()
    }
}

impl SerdeAPI for FuelConverter {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        Ok(())
    }
}

impl Mass for FuelConverter {
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

// non-py methods
impl FuelConverter {
    /// Get fuel converter max power output given time step, dt \[s\]
    pub fn set_cur_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<()> {
        ensure!(
            dt > si::Time::ZERO,
            format!(
                "{}\n dt must always be greater than 0.0",
                format_dbg!(dt > si::Time::ZERO)
            )
        );
        if self.pwr_out_max_init == si::Power::ZERO {
            self.pwr_out_max_init = self.pwr_out_max / 10.
        };
        self.state.pwr_out_max = (self.state.pwr_brake
            + (self.pwr_out_max / self.pwr_ramp_lag) * dt)
            .min(self.pwr_out_max)
            .max(self.pwr_out_max_init);
        Ok(())
    }

    /// Solve for fuel usage for a given required fuel converter power output
    /// (pwr_out_req \[W\]) and time step size (dt_s \[s\])
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: bool,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        if assert_limits {
            ensure!(
                utils::almost_le_uom(&pwr_out_req, &self.pwr_out_max, Some(TOL)),
                format!(
                "{}\nfc pwr_out_req ({:.6} MW) must be less than or equal to static pwr_out_max ({:.6} MW)",
                format_dbg!(utils::almost_le_uom(&pwr_out_req, &self.pwr_out_max, Some(TOL))),
                pwr_out_req.get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()),
            );
            ensure!(
                utils::almost_le_uom(&pwr_out_req, &self.state.pwr_out_max, Some(TOL)),
                format!("{}\nfc pwr_out_req ({:.6} MW) must be less than or equal to current transient pwr_out_max ({:.6} MW)",
                format_dbg!(utils::almost_le_uom(&pwr_out_req, &self.state.pwr_out_max, Some(TOL))),
                pwr_out_req.get::<si::megawatt>(),
                self.state.pwr_out_max.get::<si::megawatt>()),
            );
        }
        ensure!(
            pwr_out_req >= si::Power::ZERO,
            format!(
                "{}\nfc pwr_out_req ({:.6} MW) must be greater than or equal to zero",
                format_dbg!(pwr_out_req >= si::Power::ZERO),
                pwr_out_req.get::<si::megawatt>()
            )
        );
        self.state.pwr_brake = pwr_out_req;
        self.state.eta = uc::R
            * interp1d(
                &(pwr_out_req / self.pwr_out_max).get::<si::ratio>(),
                &self.pwr_out_frac_interp,
                &self.eta_interp,
                false,
            )?;
        ensure!(
            self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R,
            format!(
                "{}\nfc eta ({}) must be between 0 and 1",
                format_dbg!(self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R),
                self.state.eta.get::<si::ratio>()
            )
        );

        self.state.engine_on = engine_on;
        self.state.pwr_idle_fuel = if self.state.engine_on {
            self.pwr_idle_fuel
        } else {
            si::Power::ZERO
        };
        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            self.state.engine_on || pwr_out_req == si::Power::ZERO,
            format!(
                "{}\nEngine is off but pwr_out_req is non-zero",
                format_dbg!(self.state.engine_on || pwr_out_req == si::Power::ZERO)
            )
        );
        self.state.pwr_fuel = pwr_out_req / self.state.eta + self.pwr_idle_fuel;
        self.state.pwr_loss = self.state.pwr_fuel - self.state.pwr_brake;

        self.state.energy_brake += self.state.pwr_brake * dt;
        self.state.energy_fuel += self.state.pwr_fuel * dt;
        self.state.energy_loss += self.state.pwr_loss * dt;
        self.state.energy_idle_fuel += self.state.pwr_idle_fuel * dt;
        ensure!(
            self.state.energy_loss.get::<si::joule>() >= 0.0,
            format!(
                "{}\nEnergy loss must be non-negative",
                format_dbg!(self.state.energy_loss.get::<si::joule>() >= 0.0)
            )
        );
        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct FuelConverterState {
    /// iteration counter
    pub i: usize,
    /// max power fc can produce at current time
    pub pwr_out_max: si::Power,
    /// efficiency evaluated at current demand
    pub eta: si::Ratio,
    /// instantaneous power going to generator
    pub pwr_brake: si::Power,
    /// instantaneous fuel power flow
    pub pwr_fuel: si::Power,
    /// loss power, including idle
    pub pwr_loss: si::Power,
    /// idle fuel flow rate power
    pub pwr_idle_fuel: si::Power,
    /// cumulative propulsion energy fc has produced
    pub energy_brake: si::Energy,
    /// cumulative fuel energy fc has consumed
    pub energy_fuel: si::Energy,
    /// cumulative energy fc has lost due to imperfect efficiency
    pub energy_loss: si::Energy,
    /// cumulative fuel energy fc has lost due to idle
    pub energy_idle_fuel: si::Energy,
    /// If true, engine is on, and if false, off (no idle)
    pub engine_on: bool,
}

impl Default for FuelConverterState {
    fn default() -> Self {
        Self {
            i: 1,
            pwr_out_max: Default::default(),
            eta: Default::default(),
            pwr_fuel: Default::default(),
            pwr_brake: Default::default(),
            pwr_loss: Default::default(),
            pwr_idle_fuel: Default::default(),
            energy_fuel: Default::default(),
            energy_brake: Default::default(),
            energy_loss: Default::default(),
            energy_idle_fuel: Default::default(),
            engine_on: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_fc() -> FuelConverter {
        FuelConverter {
            pwr_out_max: 8_000e3 * uc::W,
            pwr_out_max_init: 800e3 * uc::W,
            pwr_ramp_lag: 25.0 * uc::S,
            pwr_out_frac_interp: Vec::linspace(0.01, 1.0, 5),
            eta_interp: vec![0.2, 0.32, 0.35, 0.4, 0.38],
            pwr_idle_fuel: 500e3 * uc::W,
            save_interval: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_that_fuel_grtr_than_shaft_energy() {
        let mut fc = test_fc();
        fc.state.pwr_out_max = uc::MW * 2.;
        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        assert!(fc.state.energy_fuel > fc.state.energy_brake);
    }

    #[test]
    fn test_default() {
        let _fc = FuelConverter::default();
    }

    #[test]
    fn test_that_max_power_includes_rate() {
        let mut fc = test_fc();
        fc.set_cur_pwr_out_max(uc::S * 1.0).unwrap();
        let pwr_out_max = fc.state.pwr_out_max;
        assert!(pwr_out_max < fc.pwr_out_max);
    }

    #[test]
    fn test_that_i_increments() {
        let mut fc = test_fc();
        fc.step();
        assert_eq!(2, fc.state.i);
    }

    #[test]
    fn test_that_fuel_is_monotonic() {
        let mut fc = test_fc();
        fc.state.pwr_out_max = uc::MW * 2.0;
        fc.save_interval = Some(1);
        fc.save_state();
        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        fc.step();
        fc.save_state();
        fc.solve_energy_consumption(uc::W * 2_000e3, uc::S * 1.0, true, true)
            .unwrap();
        fc.step();
        assert!(fc.history.energy_fuel[1] > fc.history.energy_fuel[0]);
        assert!(fc.history.energy_loss[1] > fc.history.energy_loss[0]);
    }

    #[test]
    fn test_that_history_has_len_1() {
        let mut fc: FuelConverter = FuelConverter::default();
        fc.save_interval = Some(1);
        assert!(fc.history.is_empty());
        fc.save_state();
        assert_eq!(1, fc.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut fc: FuelConverter = FuelConverter::default();
        assert!(fc.history.is_empty());
        fc.save_state();
        assert!(fc.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut fc = test_fc();
        let eta_max = 0.4;
        let eta_min = 0.2;
        let eta_range = 0.2;

        eta_test_body!(fc, eta_max, eta_min, eta_range);
    }
}
