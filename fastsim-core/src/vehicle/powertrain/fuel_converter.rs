use super::*;

// TODO: think about how to incorporate life modeling for Fuel Cells and other tech

const TOL: f64 = 1e-3;

#[pyo3_api(
    // optional, custom, struct-specific pymethods
    #[getter("eff_max")]
    fn get_eff_max_py(&self) -> f64 {
        self.get_eff_max()
    }

    #[setter("__eff_max")]
    fn set_eff_max_py(&mut self, eff_max: f64) -> PyResult<()> {
        self.set_eff_max(eff_max).map_err(PyValueError::new_err)
    }

    #[getter("eff_min")]
    fn get_eff_min_py(&self) -> f64 {
        self.get_eff_min()
    }

    #[getter("eff_range")]
    fn get_eff_range_py(&self) -> f64 {
        self.get_eff_range()
    }

    #[setter("__eff_range")]
    fn set_eff_range_py(&mut self, eff_range: f64) -> PyResult<()> {
        self.set_eff_range(eff_range).map_err(PyValueError::new_err)
    }

    // TODO: handle `side_effects` and uncomment
    // #[setter("__mass_kg")]
    // fn set_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
    //     self.set_mass(mass_kg.map(|m| m * uc::KG))?;
    //     Ok(())
    // }

    #[getter("mass_kg")]
    fn get_mass_py(&self) -> PyResult<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_pwr_kw_per_kg(&self) -> Option<f64> {
        self.specific_pwr.map(|x| x.get::<si::kilowatt_per_kilogram>())
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling Fuel Converter (e.g. engine, fuel cell.)
pub struct FuelConverter {
    #[serde(default)]
    /// struct for tracking current state
    #[serde(skip_serializing_if = "IsDefault::is_default")]
    pub state: FuelConverterState,
    /// FuelConverter mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    pub(in super::super) mass: Option<si::Mass>,
    /// FuelConverter specific power
    #[api(skip_get, skip_set)]
    pub(in super::super) specific_pwr: Option<si::SpecificPower>,
    #[serde(rename = "pwr_out_max_watts")]
    /// max rated brake output power
    pub pwr_out_max: si::Power,
    /// starting/baseline transient power limit
    #[serde(default)]
    pub pwr_out_max_init: si::Power,
    // TODO: consider a ramp down rate, which may be needed for fuel cells
    #[serde(rename = "pwr_ramp_lag_seconds")]
    /// lag time for ramp up
    pub pwr_ramp_lag: si::Time,
    /// Fuel converter brake power fraction array at which efficiencies are evaluated.
    /// This fuel converter efficiency model assumes that speed and load (or voltage and current) will
    /// always be controlled for operating at max possible efficiency for the power demand
    pub pwr_out_frac_interp: Vec<f64>,
    /// fuel converter efficiency array
    pub eff_interp: Vec<f64>,
    /// idle fuel power to overcome internal friction (not including aux load) \[W\]
    #[serde(rename = "pwr_idle_fuel_watts")]
    pub pwr_idle_fuel: si::Power,
    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "FuelConverterStateHistoryVec::is_empty")]
    pub history: FuelConverterStateHistoryVec, // TODO: spec out fuel tank size and track kg of fuel
}

impl SetCumulative for FuelConverter {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

impl SerdeAPI for FuelConverter {}
impl Init for FuelConverter {
    fn init(&mut self) -> anyhow::Result<()> {
        let _ = self.mass()?;
        Ok(())
    }
}

impl Mass for FuelConverter {
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

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        let derived_mass = self.derived_mass()?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                log::info!(
                    "Derived mass from `self.specific_pwr` and `self.pwr_out_max` does not match {}",
                    "provided mass. Updating based on `side_effect`"
                );
                match side_effect {
                    MassSideEffect::Extensive => {
                        self.pwr_out_max = self.specific_pwr.ok_or_else(|| {
                            anyhow!(
                                "{}\nExpected `self.specific_pwr` to be `Some`.",
                                format_dbg!()
                            )
                        })? * new_mass;
                    }
                    MassSideEffect::Intensive => {
                        self.specific_pwr = Some(self.pwr_out_max / new_mass);
                    }
                    MassSideEffect::None => {
                        self.specific_pwr = None;
                    }
                }
            }
        } else if let None = new_mass {
            log::debug!("Provided mass is None, setting `self.specific_pwr` to None");
            self.specific_pwr = None;
        }
        self.mass = new_mass;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self
            .specific_pwr
            .map(|specific_pwr| self.pwr_out_max / specific_pwr))
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
        self.specific_pwr = None;
    }
}

impl SaveInterval for FuelConverter {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
    }
}

// non-py methods

impl FuelConverter {
    /// Returns maximum possible traction-related power [FuelConverter]
    /// can produce, accounting for any aux-related power required.
    /// # Arguments
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    pub fn get_cur_pwr_tract_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        if self.pwr_out_max_init == si::Power::ZERO {
            // TODO: think about how to initialize power
            self.pwr_out_max_init = self.pwr_out_max / 10.
        };
        self.state.pwr_aux = pwr_aux;
        self.state.pwr_out_max = (self.state.pwr_tractive
            + (self.pwr_out_max / self.pwr_ramp_lag) * dt)
            .min(self.pwr_out_max)
            .max(self.pwr_out_max_init);
        Ok(self.pwr_out_max)
    }

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: tractive power output required to achieve presribed speed
    /// - `pwr_aux`: component-specific aux power demand (e.g. mechanical power if from engine/FC)
    /// - `enabled`: whether component is actively running
    /// - `dt`: time step size
    pub fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            pwr_out_req >= si::Power::ZERO,
            format!(
                "{}\n`pwr_out_req` must be >= 0",
                format_dbg!(pwr_out_req >= si::Power::ZERO),
            )
        );
        ensure!(
            pwr_aux >= si::Power::ZERO,
            format!(
                "{}\n`pwr_aux` must be >= 0",
                format_dbg!(pwr_aux >= si::Power::ZERO),
            )
        );
        self.state.pwr_tractive = pwr_out_req;
        self.state.pwr_aux = pwr_aux;
        self.state.eff = uc::R
            * interp1d(
                &((pwr_out_req + pwr_aux) / self.pwr_out_max).get::<si::ratio>(),
                &self.pwr_out_frac_interp,
                &self.eff_interp,
                Default::default(),
            )?;
        ensure!(
            self.state.eff >= 0.0 * uc::R || self.state.eff <= 1.0 * uc::R,
            format!(
                "{}\nfc efficiency ({}) must be between 0 and 1",
                format_dbg!(self.state.eff >= 0.0 * uc::R || self.state.eff <= 1.0 * uc::R),
                self.state.eff.get::<si::ratio>()
            )
        );

        self.state.fc_on = enabled;
        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            self.state.fc_on || (pwr_out_req == si::Power::ZERO && pwr_aux == si::Power::ZERO),
            format!(
                "{}\nEngine is off but pwr_out_req + pwr_aux is non-zero",
                format_dbg!(
                    self.state.fc_on
                        || (pwr_out_req == si::Power::ZERO && pwr_aux == si::Power::ZERO)
                )
            )
        );
        // TODO: consider how idle is handled.  The goal is to make it so that even if `pwr_aux` is
        // zero, there will be fuel consumption to overcome internal dissipation.
        self.state.pwr_fuel = ((pwr_out_req + pwr_aux) / self.state.eff).max(self.pwr_idle_fuel);
        self.state.pwr_loss = self.state.pwr_fuel - self.state.pwr_tractive;

        // TODO: put this in `SetCumulative::set_custom_cumulative`
        // ensure!(
        //     self.state.energy_loss.get::<si::joule>() >= 0.0,
        //     format!(
        //         "{}\nEnergy loss must be non-negative",
        //         format_dbg!(self.state.energy_loss.get::<si::joule>() >= 0.0)
        //     )
        // );
        Ok(())
    }
}

impl FuelConverter {
    impl_get_set_eff_max_min!();
    impl_get_set_eff_range!();
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative,
)]
#[pyo3_api]
pub struct FuelConverterState {
    /// time step index
    pub i: usize,
    /// max power fc can produce at current time
    pub pwr_out_max: si::Power,
    /// efficiency evaluated at current demand
    pub eff: si::Ratio,
    /// instantaneous power going to drivetrain, not including aux
    pub pwr_tractive: si::Power,
    /// integral of [Self::pwr_tractive]
    pub energy_tractive: si::Energy,
    /// power going to auxiliaries
    pub pwr_aux: si::Power,
    /// Integral of [Self::pwr_aux]
    pub energy_aux: si::Energy,
    /// instantaneous fuel power flow
    pub pwr_fuel: si::Power,
    /// Integral of [Self::pwr_fuel]
    pub energy_fuel: si::Energy,
    /// loss power, including idle
    pub pwr_loss: si::Power,
    /// Integral of [Self::pwr_loss]
    pub energy_loss: si::Energy,
    /// If true, engine is on, and if false, off (no idle)
    pub fc_on: bool,
}

impl SerdeAPI for FuelConverterState {}
impl Init for FuelConverterState {}

impl FuelConverterState {
    pub fn new() -> Self {
        Self {
            i: 1,
            fc_on: true,
            ..Default::default()
        }
    }
}
