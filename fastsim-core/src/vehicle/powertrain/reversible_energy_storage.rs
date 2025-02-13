use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

const TOL: f64 = 1e-3;

#[fastsim_api(
    #[getter("eff_max")]
    fn get_eff_max_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_max()?)
    }

    #[setter("__eff_max")]
    fn set_eff_max_py(&mut self, eff_max: f64) -> PyResult<()> {
        self.set_eff_max(eff_max)?;
        Ok(())
    }

    #[getter("eff_min")]
    fn get_eff_min_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_min()?)
    }

    #[getter("eff_range")]
    fn get_eff_range_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_range()?)
    }

    #[setter("__eff_range")]
    fn set_eff_range_py(&mut self, eff_range: f64) -> PyResult<()> {
        self.set_eff_range(eff_range)?;
        Ok(())
    }

    // TODO: decide on way to deal with `side_effect` coming after optional arg and uncomment
    #[pyo3(name = "set_mass")]
    #[pyo3(signature = (mass_kg=None, side_effect=None))]
    fn set_mass_py(&mut self, mass_kg: Option<f64>, side_effect: Option<String>) -> anyhow::Result<()> {
        let side_effect = side_effect.unwrap_or_else(|| "Intensive".into());
        self.set_mass(
            mass_kg.map(|m| m * uc::KG),
            MassSideEffect::try_from(side_effect)?
        )?;
        Ok(())
    }

    #[getter("mass_kg")]
    fn get_mass_kg_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_energy_kjoules_per_kg(&self) -> Option<f64> {
        self.specific_energy.map(|se| se.get::<si::kilojoule_per_kilogram>())
    }

    #[getter]
    fn get_energy_capacity_usable_joules(&self) -> f64 {
        self.energy_capacity_usable().get::<si::joule>()
    }

    #[pyo3(name = "set_default_pwr_interp")]
    fn set_default_pwr_interp_py(&mut self) -> anyhow::Result<()> {
        self.set_default_pwr_interp()
    }

    #[pyo3(name = "set_default_pwr_and_soc_interp")]
    fn set_default_pwr_and_soc_interp_py(&mut self) -> anyhow::Result<()> {
        self.set_default_pwr_and_soc_interp()
    }

    #[pyo3(name = "set_default_pwr_and_temp_interp")]
    fn set_default_pwr_and_temp_interp_py(&mut self) -> anyhow::Result<()> {
        self.set_default_pwr_and_temp_interp()
    }

    #[pyo3(name = "set_default_pwr_soc_and_temp_interp")]
    fn set_default_pwr_soc_and_temp_interp_py(&mut self) -> anyhow::Result<()> {
        self.set_default_pwr_soc_and_temp_interp()
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
#[non_exhaustive]
/// Struct for modeling technology-naive Reversible Energy Storage (e.g. battery, flywheel).
pub struct ReversibleEnergyStorage {
    /// [Self] Thermal plant, including thermal management controls
    #[serde(default, skip_serializing_if = "RESThermalOption::is_none")]
    #[has_state]
    pub thrml: RESThermalOption,
    /// ReversibleEnergyStorage mass
    #[serde(default)]
    pub(in super::super) mass: Option<si::Mass>,
    /// ReversibleEnergyStorage specific energy
    pub(in super::super) specific_energy: Option<si::SpecificEnergy>,
    /// Max output (and input) power battery can produce (accept)
    pub pwr_out_max: si::Power,

    /// Total energy capacity of battery of full discharge SOC of 0.0 and 1.0
    pub energy_capacity: si::Energy,

    /// interpolator for calculating [Self] efficiency
    pub eff_interp: Interpolator,

    /// what state variables to use in calculating efficiency
    pub eff_interp_inputs: RESEffInterpInputs,

    /// Hard limit on minimum SOC, e.g. 0.05
    pub min_soc: si::Ratio,
    /// Hard limit on maximum SOC, e.g. 0.95
    pub max_soc: si::Ratio,

    /// Time step interval at which history is saved
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    pub state: ReversibleEnergyStorageState,
    /// Custom vector of [Self::state]
    #[serde(
        default,
        skip_serializing_if = "ReversibleEnergyStorageStateHistoryVec::is_empty"
    )]
    pub history: ReversibleEnergyStorageStateHistoryVec,
}

impl ReversibleEnergyStorage {
    pub fn solve(&mut self, pwr_out_req: si::Power, dt: si::Time) -> anyhow::Result<()> {
        let te_res = self.temperature();
        let state = &mut self.state;

        ensure!(
            state.soc <= self.max_soc,
            format_dbg!(state.soc.get::<si::ratio>())
        );
        ensure!(
            almost_ge_uom(&state.soc, &self.min_soc, Some(1e-3)),
            "{}\n{}\n{}",
            format_dbg!(state.soc.get::<si::ratio>()),
            format_dbg!(state.soc_disch_buffer.get::<si::ratio>()),
            format_dbg!(state.pwr_aux.get::<si::watt>())
        );

        state.pwr_out_prop = pwr_out_req;
        state.pwr_out_electrical = state.pwr_out_prop + state.pwr_aux;

        if pwr_out_req + state.pwr_aux >= si::Power::ZERO {
            // discharging
            ensure!(
                utils::almost_le_uom(&(pwr_out_req + state.pwr_aux), &self.pwr_out_max, Some(TOL)),
                "{}\nres required power ({:.6} kW) exceeds static max discharge power ({:.6} kW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(
                    &(pwr_out_req + state.pwr_aux),
                    &self.pwr_out_max,
                    Some(TOL)
                )),
                (pwr_out_req + state.pwr_aux).get::<si::kilowatt>(),
                state.pwr_disch_max.get::<si::kilowatt>(),
                state.soc.get::<si::ratio>()
            );
            ensure!(
                utils::almost_le_uom(&(pwr_out_req + state.pwr_aux), &state.pwr_disch_max, Some(TOL)),
                "{}\nres required power ({:.6} kW) exceeds current max discharge power ({:.6} kW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(&(pwr_out_req + state.pwr_aux), &state.pwr_disch_max, Some(TOL))),
                (pwr_out_req + state.pwr_aux).get::<si::kilowatt>(),
                state.pwr_disch_max.get::<si::kilowatt>(),
                state.soc.get::<si::ratio>()
            );
        } else {
            // charging
            ensure!(
                utils::almost_ge_uom(
                    &(pwr_out_req + state.pwr_aux),
                    &-self.pwr_out_max,
                    Some(TOL)
                ),
                format!(
                    "{}\nres required power ({:.6} kW) exceeds static max power ({:.6} kW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_out_req + state.pwr_aux),
                        &-self.pwr_out_max,
                        Some(TOL)
                    )),
                    (pwr_out_req + state.pwr_aux).get::<si::kilowatt>(),
                    state.pwr_charge_max.get::<si::kilowatt>()
                )
            );
            ensure!(
                utils::almost_ge_uom(
                    &(pwr_out_req + state.pwr_aux),
                    &-state.pwr_charge_max,
                    Some(TOL)
                ),
                format!(
                    "{}\nres required power ({:.6} kW) exceeds current max power ({:.6} kW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_out_req + state.pwr_aux),
                        &-state.pwr_charge_max,
                        Some(TOL)
                    )),
                    (pwr_out_req + state.pwr_aux).get::<si::kilowatt>(),
                    state.pwr_charge_max.get::<si::kilowatt>()
                )
            );
        }
        let interp_pt: &[f64] = match (&self.eff_interp, &self.eff_interp_inputs) {
            (Interpolator::Interp0D(..), RESEffInterpInputs::Constant) => &[],
            (Interpolator::Interp1D(..), RESEffInterpInputs::CRate) => {
                &[state.pwr_out_electrical.get::<si::watt>()
                    / self.energy_capacity.get::<si::watt_hour>()]
            }
            (Interpolator::Interp2D(..), RESEffInterpInputs::CRateSOC) => &[
                state.pwr_out_electrical.get::<si::watt>()
                    / self.energy_capacity.get::<si::watt_hour>(),
                state.soc.get::<si::ratio>(),
            ],
            (Interpolator::Interp2D(..), RESEffInterpInputs::CRateTemperature) => &[
                state.pwr_out_electrical.get::<si::watt>()
                    / self.energy_capacity.get::<si::watt_hour>(),
                te_res
                    .with_context(|| format_dbg!("Expected thermal model to be configured"))?
                    .get::<si::degree_celsius>(),
            ],
            (Interpolator::Interp3D(..), RESEffInterpInputs::CRateSOCTemperature) => &[
                state.pwr_out_electrical.get::<si::watt>()
                    / self.energy_capacity.get::<si::watt_hour>(),
                state.soc.get::<si::ratio>(),
                te_res
                    .with_context(|| format_dbg!("Expected thermal model to be configured"))?
                    .get::<si::degree_celsius>(),
            ],
            _ => bail!(
                "
Invalid or not yet enabled interpolator config.
See docs for `ReversibleEnergyStorage::eff_interp` an `ReversibleEnergyStorage::eff_interp_inputs`"
            ),
        };
        state.eff = self.eff_interp.interpolate(interp_pt)? * uc::R;
        ensure!(
            state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R,
            format!(
                "{}\nres efficiency ({}) must be between 0 and 1",
                format_dbg!(state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R),
                state.eff.get::<si::ratio>()
            )
        );

        state.pwr_out_chemical = if state.pwr_out_electrical > si::Power::ZERO {
            // if positive, chemical power must be greater than electrical power
            // i.e. not all chemical power can be converted to electrical power
            state.pwr_out_electrical / state.eff
        } else {
            // if negative, chemical power, must be less than electrical power
            // i.e. not all electrical power can be converted back to chemical power
            state.pwr_out_electrical * state.eff
        };

        state.pwr_loss = (state.pwr_out_chemical - state.pwr_out_electrical).abs();

        state.soc -= state.pwr_out_chemical * dt / self.energy_capacity;

        Ok(())
    }

    /// Solve change in temperature and other thermal effects
    /// # Arguments
    /// - `fc_state`: [ReversibleEnergyStorage] state
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_hvac_to_res`: thermal power flowing from [Vehicle::hvac] system to [Self::thrml]
    /// - `te_cab`: cabin temperature for heat transfer interaction with
    ///    [Self], required if [Self::thrml] is `Some`
    /// - `dt`: simulation time step size
    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_hvac_to_res: si::Power,
        te_cab: Option<si::Temperature>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.thrml
            .solve(self.state, te_amb, pwr_thrml_hvac_to_res, te_cab, dt)
            .with_context(|| format_dbg!())
    }

    /// Sets and returns max output and max regen power based on current state
    /// # Arguments
    /// - `dt`: simulation time step size
    /// - `disch_buffer`: buffer offset from static SOC limit at which discharging is not allowed
    /// - `chrg_buffer`: buffer offset from static SOC limit at which charging is not allowed
    pub fn set_curr_pwr_out_max(
        &mut self,
        dt: si::Time,
        disch_buffer: si::Energy,
        chrg_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        self.set_pwr_disch_max(dt, disch_buffer)?;
        self.set_pwr_charge_max(dt, chrg_buffer)?;

        Ok(())
    }

    /// # Arguments
    /// - `dt`: simulation time step size
    /// - `buffer`: buffer below static maximum SOC above which charging is disabled
    pub fn set_pwr_charge_max(
        &mut self,
        dt: si::Time,
        chrg_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive topping off of the battery
        let soc_buffer_delta = (chrg_buffer
            / (self.energy_capacity * (self.max_soc - self.min_soc)))
            .max(si::Ratio::ZERO);
        ensure!(soc_buffer_delta >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state.soc_regen_buffer = self.max_soc - soc_buffer_delta;
        let pwr_max_for_dt =
            ((self.max_soc - self.state.soc) * self.energy_capacity / dt).max(si::Power::ZERO);
        self.state.pwr_charge_max = if self.state.soc <= self.state.soc_regen_buffer {
            self.pwr_out_max
        } else if self.state.soc < self.max_soc && soc_buffer_delta > si::Ratio::ZERO {
            self.pwr_out_max * (self.max_soc - self.state.soc) / soc_buffer_delta
        } else {
            // current SOC is less than both
            si::Power::ZERO
        }
        .min(pwr_max_for_dt);

        ensure!(
            self.state.pwr_charge_max >= si::Power::ZERO,
            "{}\n`{}` ({} W) must be greater than or equal to zero\n{}",
            format_dbg!(),
            stringify!(self.state.pwr_charge_max),
            self.state.pwr_charge_max.get::<si::watt>().format_eng(None),
            format_dbg!(soc_buffer_delta)
        );

        Ok(())
    }

    /// # Arguments
    /// - `dt`: simulation time step size
    /// - `buffer`: buffer above static minimum SOC above which charging is disabled
    pub fn set_pwr_disch_max(
        &mut self,
        dt: si::Time,
        disch_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive bottoming out of the battery
        let soc_buffer_delta = (disch_buffer / self.energy_capacity_usable()).max(si::Ratio::ZERO);
        ensure!(soc_buffer_delta >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state.soc_disch_buffer = self.min_soc + soc_buffer_delta;
        let pwr_max_for_dt =
            ((self.state.soc - self.min_soc) * self.energy_capacity / dt).max(si::Power::ZERO);
        self.state.pwr_disch_max = if self.state.soc > self.state.soc_disch_buffer {
            self.pwr_out_max
        } else if self.state.soc > self.min_soc && soc_buffer_delta > si::Ratio::ZERO {
            self.pwr_out_max * (self.state.soc - self.min_soc) / soc_buffer_delta
        } else {
            // current SOC is less than both
            si::Power::ZERO
        }
        .min(pwr_max_for_dt);

        ensure!(
            self.state.pwr_disch_max >= si::Power::ZERO,
            "{}\n`{}` ({} W) must be greater than or equal to zero\n{}",
            format_dbg!(),
            stringify!(self.state.pwr_disch_max),
            self.state.pwr_disch_max.get::<si::watt>().format_eng(None),
            format_dbg!(soc_buffer_delta)
        );

        Ok(())
    }

    /// Set current maximum power available for propulsion
    /// # Arguments
    /// - `pwr_aux`: aux power demand on `ReversibleEnergyStorage`
    pub fn set_curr_pwr_prop_max(&mut self, pwr_aux: si::Power) -> anyhow::Result<()> {
        let state = &mut self.state;
        state.pwr_aux = pwr_aux;
        state.pwr_prop_max = state.pwr_disch_max - pwr_aux;
        state.pwr_regen_max = state.pwr_charge_max + pwr_aux;

        ensure!(
            pwr_aux <= state.pwr_disch_max,
            "`{}` ({} W) must always be less than or equal to {} ({} W)\nsoc:{}",
            stringify!(pwr_aux),
            pwr_aux.get::<si::watt>().format_eng(None),
            stringify!(state.pwr_disch_max),
            state.pwr_disch_max.get::<si::watt>().format_eng(None),
            state.soc.get::<si::ratio>()
        );
        ensure!(
            state.pwr_prop_max >= si::Power::ZERO,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_prop_max),
            state.pwr_prop_max.get::<si::watt>().format_eng(None)
        );
        ensure!(
            state.pwr_regen_max >= si::Power::ZERO,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_regen_max),
            state.pwr_regen_max.get::<si::watt>().format_eng(None)
        );

        Ok(())
    }

    /// Sets specific energy and either mass or energy capacity of battery
    /// # Arguments
    /// - `specific_energy`: specific energy of battery
    /// - `side_effect`: whether to update mass or energy capacity
    pub fn set_specific_energy(
        mut self,
        specific_energy: si::SpecificEnergy,
        side_effect: SpecificEnergySideEffect,
    ) -> anyhow::Result<()> {
        self.specific_energy = Some(specific_energy);
        match side_effect {
            SpecificEnergySideEffect::Mass => self.set_mass(
                Some(self.energy_capacity / specific_energy),
                MassSideEffect::Intensive,
            )?,
            SpecificEnergySideEffect::Energy => {
                self.energy_capacity = specific_energy
                    * self.mass.with_context(|| {
                        format_dbg!("Expected `ReversibleEnergyStorage::mass` to have been set.")
                    })?;
            }
        }
        Ok(())
    }

    /// Returns max value of [Self::eff_interp]
    pub fn get_eff_max(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp
            .f_x()?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }

    /// Scales eff_interp by ratio of new `eff_max` per current calculated
    /// max linearly, such that `eff_min` is untouched
    pub fn set_eff_max(&mut self, eff_max: f64) -> anyhow::Result<()> {
        if (self.get_eff_min()?..=1.0).contains(&eff_max) {
            let old_max = self.get_eff_max()?;
            let f_x = self.eff_interp.f_x()?.to_owned();
            match &mut self.eff_interp {
                interp @ Interpolator::Interp1D(..) => {
                    interp.set_f_x(f_x.iter().map(|x| x * eff_max / old_max).collect())?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
            Ok(())
        } else {
            Err(anyhow!(
                "`eta_max` ({:.3}) must be between `eta_min` ({:.3}) and 1.0",
                eff_max,
                self.get_eff_min()?
            ))
        }
    }

    /// Returns min value of [Self::eff_interp]
    pub fn get_eff_min(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp
            .f_x()?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.min(*curr)))
    }

    /// Max value of `eff_interp` minus min value of `eff_interp`.
    pub fn get_eff_range(&self) -> anyhow::Result<f64> {
        Ok(self.get_eff_max()? - self.get_eff_min()?)
    }

    /// Scales values of `eff_interp` without changing max such that max - min
    /// is equal to new range.  Will change max if needed to ensure no values are
    /// less than zero.
    pub fn set_eff_range(&mut self, eff_range: f64) -> anyhow::Result<()> {
        let eff_max = self.get_eff_max()?;
        if eff_range == 0.0 {
            let f_x = vec![
                eff_max;
                self.eff_interp
                    .f_x()
                    .with_context(|| "eff_interp_fwd does not have f_x field")?
                    .len()
            ];
            self.eff_interp.set_f_x(f_x)?;
            Ok(())
        } else if (0.0..=1.0).contains(&eff_range) {
            let old_min = self.get_eff_min()?;
            let old_range = self.get_eff_max()? - old_min;
            if old_range == 0.0 {
                return Err(anyhow!(
                    "`eff_range` is already zero so it cannot be modified."
                ));
            }
            let f_x_fwd = self.eff_interp.f_x()?.to_owned();
            match &mut self.eff_interp {
                interp @ Interpolator::Interp1D(..) => {
                    interp.set_f_x(
                        f_x_fwd
                            .iter()
                            .map(|x| eff_max + (x - eff_max) * eff_range / old_range)
                            .collect(),
                    )?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
            if self.get_eff_min()? < 0.0 {
                let x_neg = self.get_eff_min()?;
                let f_x_fwd = self.eff_interp.f_x()?.to_owned();
                match &mut self.eff_interp {
                    interp @ Interpolator::Interp1D(..) => {
                        interp.set_f_x(f_x_fwd.iter().map(|x| x - x_neg).collect())?;
                    }
                    _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
                }
            }
            if self.get_eff_max()? > 1.0 {
                return Err(anyhow!(format!(
                    "`eff_max` ({:.3}) must be no greater than 1.0",
                    self.get_eff_max()?
                )));
            }
            Ok(())
        } else {
            Err(anyhow!(format!(
                "`eff_range` ({:.3}) must be between 0.0 and 1.0",
                eff_range,
            )))
        }
    }

    /// Usable energy capacity, accounting for SOC limits
    pub fn energy_capacity_usable(&self) -> si::Energy {
        self.energy_capacity * (self.max_soc - self.min_soc)
    }

    /// Sets the ReversibleEnergyStorage eff_interp Interpolator to be a 1D
    /// interpolator with the default x and f_x arrays  
    /// # Source of default efficiency values  
    /// - `x`: values in the third sub-array (corresponding to power) in ALTRIOS's
    ///    eta_interp_grid  
    /// - `f_x`: efficiency array as a function of power at constant 50% SOC and 23
    ///    °C corresponds to `eta_interp_values[0][5]` in ALTRIOS
    #[cfg(all(feature = "yaml", feature = "resources"))]
    pub fn set_default_pwr_interp(&mut self) -> anyhow::Result<()> {
        self.eff_interp_inputs = RESEffInterpInputs::CRate;
        self.eff_interp = ninterp::Interpolator::from_resource("res/default_pwr.yaml", false)?;
        Ok(())
    }

    /// Sets the ReversibleEnergyStorage eff_interp Interpolator to be a 2D
    /// interpolator with the default x, y, and f_xy arrays  
    /// # Source of default efficiency values  
    /// - `x`: values in the third sub-array (corresponding to power) in ALTRIOS's
    ///    eta_interp_grid  
    /// - `y`: values in the second sub-array (corresponding to SOC) in
    ///    ALTRIOS's eta_interp_grid  
    /// - `f_xy`: efficiency array as a function of power and SOC at constant 23
    ///    °C corresponds to `eta_interp_values[0]` in ALTRIOS, transposed so
    ///    that the outermost layer is now power and the innermost layer SOC (in
    ///    ALTRIOS, the outermost layer is SOC and innermost is power)
    #[cfg(all(feature = "yaml", feature = "resources"))]
    pub fn set_default_pwr_and_soc_interp(&mut self) -> anyhow::Result<()> {
        self.eff_interp_inputs = RESEffInterpInputs::CRateSOC;
        self.eff_interp =
            ninterp::Interpolator::from_resource("res/default_pwr_and_soc.yaml", false)?;
        Ok(())
    }

    /// - `f_xy`: efficiency array as a function of power and temperature at
    ///    constant 50% SOC
    #[cfg(all(feature = "yaml", feature = "resources"))]
    pub fn set_default_pwr_and_temp_interp(&mut self) -> anyhow::Result<()> {
        self.eff_interp_inputs = RESEffInterpInputs::CRateTemperature;
        self.eff_interp =
            ninterp::Interpolator::from_resource("res/default_pwr_and_temp.yaml", false)?;
        Ok(())
    }

    /// Sets the ReversibleEnergyStorage eff_interp Interpolator to be a 3D
    /// interpolator with the default x, y, z, and f_xyz arrays  
    /// # Source of default efficiency values  
    /// - `x`: values in the third sub-array (corresponding to power) in ALTRIOS's
    ///    eta_interp_grid  
    /// - `y`: values in the second sub-array (corresponding to SOC) in ALTRIOS's
    ///    eta_interp_grid  
    /// - `z`: values in the first sub-array (corresponding to temperature) in
    ///    ALTRIOS's eta_interp_grid  
    /// - `f_xyz`: efficiency array as a function of power, SOC, and temperature
    ///    corresponds to eta_interp_values in ALTRIOS, transposed so that the
    ///    outermost layer is now power, and the innermost layer temperature (in
    ///    ALTRIOS, the outermost layer is temperature and innermost is power)
    #[cfg(all(feature = "yaml", feature = "resources"))]
    pub fn set_default_pwr_soc_and_temp_interp(&mut self) -> anyhow::Result<()> {
        self.eff_interp_inputs = RESEffInterpInputs::CRateSOCTemperature;
        self.eff_interp =
            ninterp::Interpolator::from_resource("res/default_pwr_soc_and_temp.yaml", false)?;
        Ok(())
    }

    /// If thermal model is appropriately configured, returns current lumped [Self] temperature
    pub fn temperature(&self) -> Option<si::Temperature> {
        match &self.thrml {
            RESThermalOption::RESLumpedThermal(rest) => Some(rest.state.temperature),
            RESThermalOption::None => None,
        }
    }

    /// If thermal model is appropriately configured, returns lumped [Self]
    /// temperature at previous time step
    pub fn temp_prev(&self) -> Option<si::Temperature> {
        match &self.thrml {
            RESThermalOption::RESLumpedThermal(rest) => Some(rest.state.temp_prev),
            RESThermalOption::None => None,
        }
    }
}

impl SetCumulative for ReversibleEnergyStorage {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

impl Mass for ReversibleEnergyStorage {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
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
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                match side_effect {
                    MassSideEffect::Extensive => {
                        self.energy_capacity = self.specific_energy.ok_or_else(|| {
                            anyhow!(
                                "{}\nExpected `self.specific_energy` to be `Some`.",
                                format_dbg!()
                            )
                        })? * new_mass;
                    }
                    MassSideEffect::Intensive => {
                        self.specific_energy = Some(self.energy_capacity / new_mass);
                    }
                    MassSideEffect::None => {
                        self.specific_energy = None;
                    }
                }
            }
        } else if new_mass.is_none() {
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

impl SerdeAPI for ReversibleEnergyStorage {}
impl Init for ReversibleEnergyStorage {
    fn init(&mut self) -> anyhow::Result<()> {
        let _ = self.mass().with_context(|| anyhow!(format_dbg!()))?;
        self.state.init().with_context(|| anyhow!(format_dbg!()))?;
        // TODO: make some kind of data validation framework to replace this code.
        ensure!(
            self.max_soc > self.min_soc,
            format!(
                "{}\n`max_soc`: {} must be greater than `min_soc`: {}`",
                format_dbg!(),
                self.max_soc.get::<si::ratio>(),
                self.min_soc.get::<si::ratio>(),
            )
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, IsVariant, From, TryInto)]
/// Controls which parameter to update when setting specific energy
pub enum SpecificEnergySideEffect {
    /// update mass
    Mass,
    /// update energy
    Energy,
}

#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[non_exhaustive]
#[serde(default)]
/// ReversibleEnergyStorage state variables
pub struct ReversibleEnergyStorageState {
    // limits
    /// max output power for propulsion during positive traction
    pub pwr_prop_max: si::Power,
    /// max regen power for propulsion during negative traction
    pub pwr_regen_max: si::Power,
    /// max discharge power total
    pub pwr_disch_max: si::Power,
    /// max charge power on the output side
    pub pwr_charge_max: si::Power,

    /// time step index
    pub i: usize,

    /// state of charge (SOC)
    pub soc: si::Ratio,
    /// SOC at which [ReversibleEnergyStorage] regen power begins linearly
    /// derating as it approaches maximum SOC
    pub soc_regen_buffer: si::Ratio,
    /// SOC at which [ReversibleEnergyStorage] discharge power begins linearly
    /// derating as it approaches minimum SOC
    pub soc_disch_buffer: si::Ratio,
    /// Chemical <-> Electrical conversion efficiency based on current power demand
    pub eff: si::Ratio,
    /// State of Health (SOH)
    pub soh: f64,

    // TODO: add `pwr_out_neg_electrical` and `pwr_out_pos_electrical` and corresponding energies
    // powers to separately pin negative- and positive-power operation
    /// total electrical power; positive is discharging
    pub pwr_out_electrical: si::Power,
    /// electrical power going to propulsion
    pub pwr_out_prop: si::Power,
    /// electrical power going to aux loads
    pub pwr_aux: si::Power,
    /// power dissipated as loss
    pub pwr_loss: si::Power,
    /// chemical power; positive is discharging
    pub pwr_out_chemical: si::Power,

    // cumulative energies
    /// cumulative total electrical energy; positive is discharging
    pub energy_out_electrical: si::Energy,
    /// cumulative electrical energy going to propulsion
    pub energy_out_prop: si::Energy,
    /// cumulative electrical energy going to aux loads
    pub energy_aux: si::Energy,
    /// cumulative energy dissipated as loss
    pub energy_loss: si::Energy,
    /// cumulative chemical energy; positive is discharging
    pub energy_out_chemical: si::Energy,
}

impl Default for ReversibleEnergyStorageState {
    fn default() -> Self {
        Self {
            pwr_prop_max: si::Power::ZERO,
            pwr_regen_max: si::Power::ZERO,
            pwr_disch_max: si::Power::ZERO,
            pwr_charge_max: si::Power::ZERO,
            i: Default::default(),
            soc: uc::R * 0.5,
            soc_regen_buffer: uc::R * 1.,
            soc_disch_buffer: si::Ratio::ZERO,
            eff: si::Ratio::ZERO,
            soh: 0.,
            pwr_out_electrical: si::Power::ZERO,
            pwr_out_prop: si::Power::ZERO,
            pwr_aux: si::Power::ZERO,
            pwr_loss: si::Power::ZERO,
            pwr_out_chemical: si::Power::ZERO,
            energy_out_electrical: si::Energy::ZERO,
            energy_out_prop: si::Energy::ZERO,
            energy_aux: si::Energy::ZERO,
            energy_loss: si::Energy::ZERO,
            energy_out_chemical: si::Energy::ZERO,
        }
    }
}
impl Init for ReversibleEnergyStorageState {}
impl SerdeAPI for ReversibleEnergyStorageState {}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, IsVariant, From, TryInto)]
pub enum RESThermalOption {
    /// Basic thermal plant for [ReversibleEnergyStorage]
    RESLumpedThermal(Box<RESLumpedThermal>),
    /// no thermal plant for [ReversibleEnergyStorage]
    #[default]
    None,
}
impl SetCumulative for RESThermalOption {
    fn set_cumulative(&mut self, dt: si::Time) {
        match self {
            Self::RESLumpedThermal(rlt) => rlt.set_cumulative(dt),
            Self::None => {}
        }
    }
}
impl SaveState for RESThermalOption {
    fn save_state(&mut self) {
        match self {
            Self::RESLumpedThermal(rlt) => rlt.save_state(),
            Self::None => {}
        }
    }
}
impl Step for RESThermalOption {
    fn step(&mut self) {
        match self {
            Self::RESLumpedThermal(rlt) => rlt.step(),
            Self::None => {}
        }
    }
}
impl Init for RESThermalOption {
    fn init(&mut self) -> anyhow::Result<()> {
        match self {
            Self::RESLumpedThermal(rest) => rest.init()?,
            Self::None => {}
        }
        Ok(())
    }
}
impl SerdeAPI for RESThermalOption {}
impl RESThermalOption {
    /// Solve change in temperature and other thermal effects
    /// # Arguments
    /// - `res_state`: [ReversibleEnergyStorage] state
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_hvac_to_res`: thermal power flowing from [Vehicle::hvac]
    ///    system to [Self], required if [Self::is_none] is false
    /// - `dt`: simulation time step size
    fn solve(
        &mut self,
        res_state: ReversibleEnergyStorageState,
        te_amb: si::Temperature,
        pwr_thrml_hvac_to_res: si::Power,
        te_cab: Option<si::Temperature>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        match self {
            Self::RESLumpedThermal(rest) => rest
                .solve(
                    res_state,
                    te_amb,
                    pwr_thrml_hvac_to_res,
                    te_cab.with_context(|| {
                        format_dbg!(
                            "`te_cab` must be `Some` for [RESThermalOption::RESLumpedThermal]"
                        )
                    })?,
                    dt,
                )
                .with_context(|| format_dbg!())?,
            Self::None => {
                // TODO: make sure this triggers error if appropriate
            }
        }
        Ok(())
    }
}

#[fastsim_api(
    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Default::default()
    }
)]
#[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling [ReversibleEnergyStorage] (e.g. battery) thermal plant
pub struct RESLumpedThermal {
    /// [ReversibleEnergyStorage] thermal capacitance
    pub heat_capacitance: si::HeatCapacity,
    /// parameter for heat transfer coeff from [ReversibleEnergyStorage::thrml] to ambient
    pub conductance_to_amb: si::ThermalConductance,
    /// parameter for heat transfer coeff from [ReversibleEnergyStorage::thrml] to cabin
    pub conductance_to_cab: si::ThermalConductance,
    /// current state
    #[serde(default)]
    pub state: RESLumpedThermalState,
    /// history of state
    #[serde(
        default,
        skip_serializing_if = "RESLumpedThermalStateHistoryVec::is_empty"
    )]
    pub history: RESLumpedThermalStateHistoryVec,
    // TODO: add `save_interval` and associated methods
}
impl SetCumulative for RESLumpedThermal {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}
impl SerdeAPI for RESLumpedThermal {}
impl Init for RESLumpedThermal {}
impl RESLumpedThermal {
    fn solve(
        &mut self,
        res_state: ReversibleEnergyStorageState,
        te_amb: si::Temperature,
        pwr_thrml_hvac_to_res: si::Power,
        te_cab: si::Temperature,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.state.temp_prev = self.state.temperature;
        // TODO: make sure this impacts cabin temperature
        self.state.pwr_thrml_from_cabin = self.conductance_to_cab
            * (te_cab.get::<si::degree_celsius>()
                - self.state.temperature.get::<si::degree_celsius>())
            * uc::KELVIN_INT;
        self.state.pwr_thrml_hvac_to_res = pwr_thrml_hvac_to_res;
        self.state.pwr_thrml_from_amb = self.conductance_to_amb
            * (te_amb.get::<si::degree_celsius>()
                - self.state.temperature.get::<si::degree_celsius>())
            * uc::KELVIN_INT;
        self.state.pwr_thrml_loss =
            res_state.pwr_out_electrical.abs() * (1.0 * uc::R - res_state.eff);
        self.state.temperature += (self.state.pwr_thrml_hvac_to_res
            + self.state.pwr_thrml_loss
            + self.state.pwr_thrml_from_cabin
            + self.state.pwr_thrml_from_amb)
            / self.heat_capacitance
            * dt;
        Ok(())
    }
}

#[fastsim_api(
    #[pyo3(name = "default")]
    #[staticmethod]
    fn default_py() -> Self {
        Self::default()
    }
)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[serde(default)]
pub struct RESLumpedThermalState {
    /// time step index
    pub i: usize,
    /// Current thermal mass temperature
    pub temperature: si::Temperature,
    /// Thermal mass temperature at previous time step
    pub temp_prev: si::Temperature,
    /// Thermal power flow to [RESLumpedThermal] from cabin
    pub pwr_thrml_from_cabin: si::Power,
    /// Cumulative thermal energy flow to [RESLumpedThermal] from cabin
    pub energy_thrml_from_cabin: si::Energy,
    /// Thermal power flow to [RESLumpedThermal] from ambient
    pub pwr_thrml_from_amb: si::Power,
    /// Cumulative thermal energy flow to [RESLumpedThermal] from ambient
    pub energy_thrml_from_amb: si::Energy,
    /// Thermal power flow to [RESLumpedThermal] from HVAC
    pub pwr_thrml_hvac_to_res: si::Power,
    /// Cumulative thermal energy flow to [RESLumpedThermal] from HVAC
    pub energy_thrml_hvac_to_res: si::Energy,
    /// Thermal generation due to losses
    pub pwr_thrml_loss: si::Power,
    /// Cumulative thermal energy generation due to losses
    pub energy_thrml_loss: si::Energy,
}

impl Init for RESLumpedThermalState {}
impl SerdeAPI for RESLumpedThermalState {}
impl Default for RESLumpedThermalState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            temperature: *TE_STD_AIR,
            temp_prev: *TE_STD_AIR,
            pwr_thrml_from_cabin: Default::default(),
            energy_thrml_from_cabin: Default::default(),
            pwr_thrml_from_amb: Default::default(),
            energy_thrml_from_amb: Default::default(),
            pwr_thrml_hvac_to_res: Default::default(),
            energy_thrml_hvac_to_res: Default::default(),
            pwr_thrml_loss: Default::default(),
            energy_thrml_loss: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, IsVariant, From, TryInto)]
/// Determines what [ReversibleEnergyStorage] state variables to use in calculating efficiency
pub enum RESEffInterpInputs {
    /// Efficiency is constant
    Constant,
    /// Efficiency = f(C-rate)
    CRate,
    /// Efficiency = f(C-rate, temperature)
    CRateSOCTemperature,
    /// Efficiency = f(C-rate, soc, temperature)
    CRateTemperature,
    /// Efficiency = f(C-rate, soc)
    CRateSOC,
    // TODO: finish adding possible variants
}
