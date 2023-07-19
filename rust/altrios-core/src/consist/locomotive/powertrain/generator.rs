use super::*;
use crate::consist::locomotive::powertrain::ElectricMachine;

#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[altrios_api(
    /// Initialize a fuel converter object
    #[new]
    fn __new__(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            pwr_out_frac_interp,
            eta_interp,
            pwr_out_max_watts,
            save_interval,
        )
    }

    #[setter]
    pub fn set_eta_interp(&mut self, new_value: Vec<f64>) -> anyhow::Result<()> {
        self.eta_interp = new_value;
        self.set_pwr_in_frac_interp()
    }

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
    fn get_mass_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_pwr_kw_per_kg(&self) -> Option<f64> {
        self.specific_pwr_kw_per_kg
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling generator/alternator.
pub struct Generator {
    #[serde(default)]
    /// struct for tracking current state
    pub state: GeneratorState,
    /// Generator mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    mass: Option<si::Mass>,
    /// Generator specific power
    /// TODO: make this si::specific_power after Geordie's pull request into `uom`
    #[api(skip_get, skip_set)]
    specific_pwr_kw_per_kg: Option<f64>,
    // no macro-generated setter because derived parameters would get messed up
    /// Generator brake power fraction array at which efficiencies are evaluated.
    #[api(skip_set)]
    pub pwr_out_frac_interp: Vec<f64>,
    // no macro-generated setter because derived parameters would get messed up
    /// Generator efficiency array correpsonding to [Self::pwr_out_frac_interp] and [Self::pwr_in_frac_interp].
    #[api(skip_set)]
    pub eta_interp: Vec<f64>,
    /// Mechanical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    #[serde(skip)]
    #[api(skip_set)]
    pub pwr_in_frac_interp: Vec<f64>,
    /// Generator max power out \[W\]
    #[serde(rename = "pwr_out_max_watts")]
    pub pwr_out_max: si::Power,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: GeneratorStateHistoryVec,
}

impl SerdeAPI for Generator {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        Ok(())
    }
}

impl Mass for Generator {
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
                    format!("{}\n{}", format_dbg!(), "Generator `pwr_out_max`, `specific_pwr_kw_per_kg` and `mass` are not consistent"))
                }
                None => {}
            },
            None => {}
        }
        Ok(())
    }
}

impl Generator {
    pub fn new(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            eta_interp.len() == pwr_out_frac_interp.len(),
            format!(
                "{}\ngen eta_interp and pwr_out_frac_interp must be the same length",
                format_dbg!(eta_interp.len() == pwr_out_frac_interp.len())
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x >= 0.0),
            format!(
                "{}\ngen pwr_out_frac_interp must be non-negative",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x >= 0.0))
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x <= 1.0),
            format!(
                "{}\ngen pwr_out_frac_interp must be less than or equal to 1.0",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x <= 1.0))
            )
        );

        let history = GeneratorStateHistoryVec::new();
        let pwr_out_max = uc::W * pwr_out_max_watts;
        let state = GeneratorState::default();

        let mut gen = Generator {
            state,
            pwr_out_frac_interp,
            eta_interp,
            pwr_in_frac_interp: Vec::new(),
            pwr_out_max,
            save_interval,
            history,
            ..Default::default()
        };
        gen.set_pwr_in_frac_interp()?;
        Ok(gen)
    }

    pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
        // make sure vector has been created
        self.pwr_in_frac_interp = self
            .pwr_out_frac_interp
            .iter()
            .zip(self.eta_interp.iter())
            .map(|(x, y)| x / y)
            .collect();
        // verify monotonicity
        ensure!(
            self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1]),
            format!(
                "{}\ngen pwr_in_frac_interp ({:?}) must be monotonically increasing",
                format_dbg!(self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1])),
                self.pwr_in_frac_interp
            )
        );
        Ok(())
    }

    pub fn set_pwr_in_req(
        &mut self,
        pwr_prop_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // generator cannot regen
        ensure!(
            pwr_prop_req >= si::Power::ZERO,
            format!(
                "{}\ngen propulsion power is negative",
                format_dbg!(pwr_prop_req >= si::Power::ZERO)
            )
        );
        ensure!(
            pwr_prop_req + pwr_aux <= self.pwr_out_max,
            format!(
                "{}\ngen required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_prop_req + pwr_aux <= self.pwr_out_max),
                (pwr_prop_req + pwr_aux).get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        self.state.eta = uc::R
            * interp1d(
                &(pwr_prop_req / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_out_frac_interp,
                &self.eta_interp,
                false,
            )?;

        ensure!(
            self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R,
            format!(
                "{}\ngen eta ({}) must be between 0 and 1",
                format_dbg!(self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R),
                self.state.eta.get::<si::ratio>()
            )
        );

        self.state.pwr_elec_prop_out = pwr_prop_req;
        self.state.energy_elec_prop_out += self.state.pwr_elec_prop_out * dt;

        self.state.pwr_elec_aux = pwr_aux;
        self.state.energy_elec_aux += self.state.pwr_elec_aux * dt;

        self.state.pwr_mech_in =
            (self.state.pwr_elec_prop_out + self.state.pwr_elec_aux) / self.state.eta;
        self.state.energy_mech_in += self.state.pwr_mech_in * dt;

        self.state.pwr_loss =
            self.state.pwr_mech_in - (self.state.pwr_elec_prop_out + self.state.pwr_elec_aux);
        self.state.energy_loss += self.state.pwr_loss * dt;
        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();
}

impl Default for Generator {
    fn default() -> Self {
        let file_contents = include_str!("generator.default.yaml");
        serde_yaml::from_str::<Generator>(file_contents).unwrap()
    }
}

impl ElectricMachine for Generator {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        if self.pwr_in_frac_interp.is_empty() {
            // make sure vector has been populated
            self.set_pwr_in_frac_interp()?;
        }
        let eta = uc::R
            * interp1d(
                &(pwr_in_max / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_in_frac_interp,
                &self.eta_interp,
                false,
            )?;
        self.state.pwr_elec_out_max = (pwr_in_max * eta).min(self.pwr_out_max);
        self.state.pwr_elec_prop_out_max = self.state.pwr_elec_out_max - pwr_aux.unwrap();

        Ok(())
    }

    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) {
        self.state.pwr_rate_out_max = pwr_rate_in_max
            * if self.state.eta.get::<si::ratio>() > 0.0 {
                self.state.eta
            } else {
                uc::R * 1.0
            };
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct GeneratorState {
    /// iteration counter
    pub i: usize,
    /// efficiency evaluated at current power demand
    pub eta: si::Ratio,
    /// max possible power output for propulsion
    pub pwr_elec_prop_out_max: si::Power,
    /// max possible power output total
    pub pwr_elec_out_max: si::Power,
    /// max possible power output rate
    pub pwr_rate_out_max: si::PowerRate,
    /// mechanical power input
    pub pwr_mech_in: si::Power,
    /// electrical power output to propulsion
    pub pwr_elec_prop_out: si::Power,
    /// electrical power output to aux loads
    pub pwr_elec_aux: si::Power,
    /// power lost due to conversion inefficiency
    pub pwr_loss: si::Power,
    /// cumulative mech energy in from fc
    pub energy_mech_in: si::Energy,
    /// cumulative elec energy out to propulsion
    pub energy_elec_prop_out: si::Energy,
    /// cumulative elec energy to aux loads
    pub energy_elec_aux: si::Energy,
    /// cumulative energy has lost due to imperfect efficiency
    pub energy_loss: si::Energy,
}

impl Default for GeneratorState {
    fn default() -> Self {
        Self {
            i: 1,
            eta: Default::default(),
            pwr_rate_out_max: Default::default(),
            pwr_elec_out_max: Default::default(),
            pwr_elec_prop_out_max: Default::default(),
            pwr_mech_in: Default::default(),
            pwr_elec_prop_out: Default::default(),
            pwr_elec_aux: Default::default(),
            pwr_loss: Default::default(),
            energy_mech_in: Default::default(),
            energy_elec_prop_out: Default::default(),
            energy_elec_aux: Default::default(),
            energy_loss: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_gen() -> Generator {
        Generator::new(vec![0.0, 1.0], vec![0.9, 0.8], 8e6, None).unwrap()
    }

    #[test]
    fn test_that_i_increments() {
        let mut gen = test_gen();
        gen.step();
        assert_eq!(2, gen.state.i);
    }

    #[test]
    fn test_that_loss_is_monotonic() {
        let mut gen = test_gen();
        gen.save_interval = Some(1);
        gen.save_state();
        gen.set_pwr_in_req(uc::W * 2_000e3, uc::W * 500e3, uc::S * 1.0)
            .unwrap();
        gen.step();
        gen.save_state();
        gen.set_pwr_in_req(uc::W * 2_000e3, uc::W * 500e3, uc::S * 1.0)
            .unwrap();
        gen.step();
        gen.save_state();
        gen.set_pwr_in_req(uc::W * 1_500e3, uc::W * 500e3, uc::S * 1.0)
            .unwrap();
        gen.step();
        gen.save_state();
        gen.set_pwr_in_req(uc::W * 1_500e3, uc::W * 500e3, uc::S * 1.0)
            .unwrap();
        gen.step();
        let energy_loss_j = gen
            .history
            .energy_loss
            .iter()
            .map(|x| x.get::<si::joule>())
            .collect::<Vec<_>>();
        for i in 1..energy_loss_j.len() {
            assert!(energy_loss_j[i] >= energy_loss_j[i - 1]);
        }
    }

    #[test]
    fn test_that_history_has_len_1() {
        let mut gen: Generator = Generator::default();
        gen.save_interval = Some(1);
        assert!(gen.history.is_empty());
        gen.save_state();
        assert_eq!(1, gen.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut gen: Generator = Generator::default();
        assert!(gen.history.is_empty());
        gen.save_state();
        assert!(gen.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = test_gen();
        let eta_max = 0.9;
        let eta_min = 0.8;
        let eta_range = 0.1;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
