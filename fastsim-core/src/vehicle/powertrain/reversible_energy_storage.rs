use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

const TOL: f64 = 1e-3;

#[fastsim_api(
   #[allow(clippy::too_many_arguments)]
    #[new]
    fn __new__(
        pwr_out_max_watts: f64,
        energy_capacity_joules: f64,
        min_soc: f64,
        max_soc: f64,
        initial_soc: f64,
        initial_temperature_celcius: f64,
        soc_hi_ramp_start: Option<f64>,
        soc_lo_ramp_start: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            pwr_out_max_watts,
            energy_capacity_joules,
            min_soc,
            max_soc,
            initial_soc,
            initial_temperature_celcius,
            soc_hi_ramp_start,
            soc_lo_ramp_start,
            save_interval,
        )
    }

    /// pyo3 getter for soc_lo_ramp_start
    #[getter]
    pub fn get_soc_lo_ramp_start(&self) -> PyResult<f64> {
        Ok(self.soc_lo_ramp_start.unwrap().get::<si::ratio>())
    }
    /// pyo3 setter for soc_lo_ramp_start
    #[setter("__soc_lo_ramp_start")]
    pub fn set_soc_lo_ramp_start(&mut self, new_value: f64) -> PyResult<()> {
        self.soc_lo_ramp_start = Some(new_value * uc::R);
        Ok(())
    }
    /// pyo3 getter for soc_hi_ramp_start
    #[getter]
    pub fn get_soc_hi_ramp_start(&self) -> PyResult<f64> {
        Ok(self.soc_hi_ramp_start.unwrap().get::<si::ratio>())
    }
    /// pyo3 setter for soc_hi_ramp_start
    #[setter("__soc_hi_ramp_start")]
    pub fn set_soc_hi_ramp_start(&mut self, new_value: f64) -> PyResult<()> {
        self.soc_hi_ramp_start = Some(new_value * uc::R);
        Ok(())
    }

    // #[getter("eff_max")]
    // fn get_eff_max_py(&self) -> f64 {
    //     self.get_eff_max()
    // }

    // #[setter("__eff_max")]
    // fn set_eff_max_py(&mut self, eff_max: f64) -> PyResult<()> {
    //     self.set_eff_max(eff_max).map_err(PyValueError::new_err)
    // }

    // #[getter("eff_min")]
    // fn get_eff_min_py(&self) -> f64 {
    //     self.get_eff_min()
    // }

    #[getter("eff_range")]
    fn get_eff_range_py(&self) -> f64 {
        self.get_eff_range()
    }

    // #[setter("__eff_range")]
    // fn set_eff_range_py(&mut self, eff_range: f64) -> anyhow::Result<()> {
    //     self.set_eff_range(eff_range)
    // }

    // TODO: decide on way to deal with `side_effect` coming after optional arg and uncomment
    #[pyo3(name = "set_mass")]
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
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling technology-naive Reversible Energy Storage (e.g. battery, flywheel).
pub struct ReversibleEnergyStorage {
    /// ReversibleEnergyStorage mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    pub(in super::super) mass: Option<si::Mass>,
    /// ReversibleEnergyStorage specific energy
    #[api(skip_get, skip_set)]
    pub(in super::super) specific_energy: Option<si::SpecificEnergy>,
    #[serde(rename = "pwr_out_max_watts")]
    /// Max output (and input) power battery can produce (accept)
    pub pwr_out_max: si::Power,

    /// Total energy capacity of battery of full discharge SOC of 0.0 and 1.0
    #[serde(rename = "energy_capacity_joules")]
    pub energy_capacity: si::Energy,

    /// interpolator for calculating [Self] efficiency as a function of the following variants:  
    /// - 0d -- constant -- handled on a round trip basis
    /// - 1d -- linear w.r.t. power
    /// - 2d -- linear w.r.t. power and SOC
    /// - 3d -- linear w.r.t. power, SOC, and temperature
    #[api(skip_get, skip_set)]
    pub eff_interp: Interpolator,

    /// Hard limit on minimum SOC, e.g. 0.05
    pub min_soc: si::Ratio,
    /// Hard limit on maximum SOC, e.g. 0.95
    pub max_soc: si::Ratio,
    /// SOC at which negative/charge power begins to ramp down.
    /// Should always be slightly below [Self::max_soc].
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soc_hi_ramp_start: Option<si::Ratio>,
    /// SOC at which positive/discharge power begins to ramp down.
    /// Should always be slightly above [Self::min_soc].
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soc_lo_ramp_start: Option<si::Ratio>,
    /// Time step interval at which history is saved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: ReversibleEnergyStorageState,
    /// Custom vector of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "ReversibleEnergyStorageStateHistoryVec::is_empty")]
    pub history: ReversibleEnergyStorageStateHistoryVec,
}

impl ReversibleEnergyStorage {
    pub fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let state = &mut self.state;

        ensure!(
            state.soc <= state.max_soc || pwr_out_req >= si::Power::ZERO,
            "{}\npwr_out_req must be greater than or equal to 0 if SOC is over max SOC\nstate.soc = {}",
            format_dbg!(state.soc <= state.max_soc || pwr_out_req >= si::Power::ZERO),
            state.soc.get::<si::ratio>()
        );
        ensure!(
            state.soc >= state.min_soc || pwr_out_req <= si::Power::ZERO,
            "{}\npwr_out_req must be less than 0 or equal to zero if SOC is below min SOC\nstate.soc = {}",
            format_dbg!(state.soc >= state.min_soc || pwr_out_req <= si::Power::ZERO),
            state.soc.get::<si::ratio>()
        );

        if pwr_out_req + pwr_aux >= si::Power::ZERO {
            ensure!(
                utils::almost_le_uom(&(pwr_out_req + pwr_aux), &self.pwr_out_max, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds static max discharge power ({:.6} MW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(
                    &(pwr_out_req + pwr_aux),
                    &self.pwr_out_max,
                    Some(TOL)
                )),
                (pwr_out_req + pwr_aux).get::<si::megawatt>(),
                state.pwr_disch_max.get::<si::megawatt>(),
                state.soc.get::<si::ratio>()
            );
            ensure!(
                utils::almost_le_uom(&(pwr_out_req + pwr_aux), &state.pwr_disch_max, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds transient max discharge power ({:.6} MW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(&(pwr_out_req + pwr_aux), &state.pwr_disch_max, Some(TOL))),
                (pwr_out_req + pwr_aux).get::<si::megawatt>(),
                state.pwr_disch_max.get::<si::megawatt>(),
                state.soc.get::<si::ratio>()
            );
        } else {
            ensure!(
                utils::almost_ge_uom(&(pwr_out_req + pwr_aux), &-self.pwr_out_max, Some(TOL)),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_out_req + pwr_aux),
                        &-self.pwr_out_max,
                        Some(TOL)
                    )),
                    (pwr_out_req + pwr_aux).get::<si::megawatt>(),
                    state.pwr_charge_max.get::<si::megawatt>()
                )
            );
            ensure!(
                utils::almost_ge_uom(&(pwr_out_req + pwr_aux), &-state.pwr_charge_max, Some(TOL)),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds transient max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_out_req + pwr_aux),
                        &-state.pwr_charge_max,
                        Some(TOL)
                    )),
                    (pwr_out_req + pwr_aux).get::<si::megawatt>(),
                    state.pwr_charge_max.get::<si::megawatt>()
                )
            );
        }

        state.pwr_out_propulsion = pwr_out_req;
        state.pwr_aux = pwr_aux;

        state.pwr_out_electrical = state.pwr_out_propulsion + state.pwr_aux;

        // TODO: replace this with something correct.
        // This should trip the `ensure` below
        state.eff = match self.eff_interp {
            Interpolator::Interp0D(round_trip_eff) => round_trip_eff * uc::R,
            Interpolator::Interp1D(interp1d) => {
                interp1d.interpolate(&[state.pwr_out_electrical.get::<si::watt>()]) * uc::R
            }
            Interpolator::Interp2D(interp2d) => {
                interp2d.interpolate(&[
                    state.pwr_out_electrical.get::<si::watt>(),
                    state.soc.get::<si::ratio>(),
                ])? * uc::R
            }
            Interpolator::Interp3D(interp3d) => {
                interp3d.interpolate(&[
                    state.pwr_out_electrical.get::<si::watt>(),
                    state.soc.get::<si::ratio>(),
                    state.temperature_celsius,
                ])? * uc::R
            }
            _ => bail!("Invalid interpolator.  See docs for `ReversibleEnergyStorage::eff_interp`"),
        };
        ensure!(
            state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R,
            format!(
                "{}\nres efficiency ({}) must be between 0 and 1",
                format_dbg!(state.eff >= 0.0 * uc::R || state.eff <= 1.0 * uc::R),
                state.eff.get::<si::ratio>()
            )
        );

        // TODO: figure out how to handle round trip efficiency calculation in fastsim-3 style
        if state.pwr_out_electrical > si::Power::ZERO {
            // if positive, chemical power must be greater than electrical power
            // i.e. not all chemical power can be converted to electrical power
            state.pwr_out_chemical = state.pwr_out_electrical / state.eff;
        } else {
            // if negative, chemical power, must be less than electrical power
            // i.e. not all electrical power can be converted back to chemical power
            state.pwr_out_chemical = state.pwr_out_electrical * state.eff;
        }

        state.pwr_loss = (state.pwr_out_chemical - state.pwr_out_electrical).abs();

        state.soc -= state.pwr_out_chemical * dt / self.energy_capacity;

        Ok(())
    }

    /// Sets and returns max output and max regen power based on current state
    /// #  Arguments:
    /// - `pwr_aux`: aux power demand on `ReversibleEnergyStorage`
    /// - `charge_buffer`: buffer below max SOC to allow for anticipated future
    ///    charging (i.e. decelerating while exiting a highway)
    /// - `discharge_buffer`: buffer above min SOC to allow for anticipated
    ///    future discharging (i.e. accelerating to enter a highway)
    pub fn set_cur_pwr_out_max(
        &mut self,
        pwr_aux: si::Power,
        charge_buffer: Option<si::Energy>,
        discharge_buffer: Option<si::Energy>,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        if self.soc_hi_ramp_start.is_none() {
            self.soc_hi_ramp_start = Some(self.soc_hi_ramp_start_default());
        }
        if self.soc_lo_ramp_start.is_none() {
            self.soc_lo_ramp_start = Some(self.soc_lo_ramp_start_default());
        }

        let state = &mut self.state;
        // TODO: consider having the buffer affect the max and min but not the ramp???
        // operating lo_ramp_start and min_soc, allowing for buffer
        // Set the dynamic minimum SOC to be the static min SOC plus the charge buffer
        state.min_soc = (self.min_soc + charge_buffer.unwrap_or_default() / self.energy_capacity)
            .min(self.max_soc);
        // Set the dynamic maximum SOC to be the static max SOC minus the discharge buffer
        state.max_soc = (self.max_soc
            - discharge_buffer.unwrap_or_default() / self.energy_capacity)
            .max(self.min_soc);

        state.pwr_disch_max =
            // current SOC is greater than or equal to current min and ramp down threshold
            if state.soc >= state.min_soc
            && state.soc >= self.soc_lo_ramp_start.with_context(|| format_dbg!())?
        {
            self.pwr_out_max
        } // current SOC is less than ramp down threshold and ramp down threshold is greater than min soc
        else if state.soc < self.soc_lo_ramp_start.unwrap()
            && self.soc_lo_ramp_start.unwrap() > state.min_soc
        {
            uc::W
                * interp1d(
                    &state.soc.get::<si::ratio>(),
                    &[
                        state.min_soc.get::<si::ratio>(),
                        self.soc_lo_ramp_start
                            .with_context(|| format_dbg!())?
                            .get::<si::ratio>(),
                    ],
                    &[0.0, self.pwr_out_max.get::<si::watt>()],
                    Extrapolate::No, // don't extrapolate
                )
                .with_context(|| {
                    anyhow!(
                        "{}\n failed to calculate {}",
                        format_dbg!(),
                        stringify!(state.pwr_disch_max)
                    )
                })?
        }
        // current SOC is greater than ramp down threshold but less than current min or current SOC is less than both
        else {
            uc::W * 0.
        };

        state.pwr_charge_max =
            // current SOC is less than or equal to current max and ramp down threshold
            if state.soc <= state.max_soc
            && state.soc <= self.soc_hi_ramp_start.with_context(|| format_dbg!())?
        {
            self.pwr_out_max
        } // current SOC is greater than ramp down threshold and ramp down threshold is less than max soc
        else if state.soc > self.soc_lo_ramp_start.unwrap()
            && self.soc_hi_ramp_start.unwrap() < state.max_soc
        {
            uc::W
                * interp1d(
                    &state.soc.get::<si::ratio>(),
                    &[
                        state.max_soc.get::<si::ratio>(),
                        self.soc_hi_ramp_start
                            .with_context(|| format_dbg!())?
                            .get::<si::ratio>(),
                    ],
                    &[0.0, self.pwr_out_max.get::<si::watt>()],
                    Extrapolate::No, // don't extrapolate
                )
                .with_context(|| {
                    anyhow!(
                        "{}\n failed to calculate {}",
                        format_dbg!(),
                        stringify!(state.pwr_disch_max)
                    )
                })?
        }
        // current SOC is less than ramp down threshold but greater than current
        // max or current SOC is greater than both
        else {
            uc::W * 0.
        };

        ensure!(
            state.pwr_disch_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_disch_max),
            state.pwr_disch_max.get::<si::watt>().format_eng(None)
        );
        ensure!(
            state.pwr_charge_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_charge_max),
            state.pwr_charge_max.get::<si::watt>().format_eng(None)
        );

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
            state.pwr_prop_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_prop_max),
            state.pwr_prop_max.get::<si::watt>().format_eng(None)
        );
        ensure!(
            state.pwr_regen_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero",
            stringify!(state.pwr_regen_max),
            state.pwr_regen_max.get::<si::watt>().format_eng(None)
        );

        Ok((state.pwr_prop_max, state.pwr_regen_max))
    }

    fn soc_hi_ramp_start_default(&self) -> si::Ratio {
        self.max_soc - 0.05 * uc::R
    }

    fn soc_lo_ramp_start_default(&self) -> si::Ratio {
        self.min_soc + 0.05 * uc::R
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

    pub fn get_eff_max(&self) -> f64 {
        todo!("adapt from ALTRIOS");
    }

    /// Scales eff_interp by ratio of new `eff_max` per current calculated
    /// max linearly, such that `eff_min` is untouched
    pub fn set_eff_max(&mut self, _eff_max: f64) -> Result<(), String> {
        todo!("adapt from ALTRIOS");
    }

    pub fn get_eff_min(&self) -> f64 {
        todo!("adapt from ALTRIOS");
    }

    /// Max value of `eff_interp` minus min value of `eff_interp`.
    pub fn get_eff_range(&self) -> f64 {
        self.get_eff_max() - self.get_eff_min()
    }

    /// Scales values of `eff_interp` without changing max such that max - min
    /// is equal to new range
    pub fn set_eff_range(&mut self, _eff_range: f64) -> anyhow::Result<()> {
        todo!("adapt from ALTRIOS");
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
                #[cfg(feature = "logging")]
                log::info!(
                    "Derived mass from `self.specific_energy` and `self.energy_capacity` does not match {}",
                    "provided mass. Updating based on `side_effect`"
                );
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
            #[cfg(feature = "logging")]
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

impl SerdeAPI for ReversibleEnergyStorage {}
impl Init for ReversibleEnergyStorage {
    fn init(&mut self) -> anyhow::Result<()> {
        let _ = self.mass().with_context(|| anyhow!(format_dbg!()))?;
        self.state.init().with_context(|| anyhow!(format_dbg!()))?;
        // TODO: make some kind of data validation framework to replace this code.
        ensure!(
            self.max_soc > self.min_soc
                && match self.soc_hi_ramp_start {
                    Some(soc_hi_ramp_start) => soc_hi_ramp_start <= self.max_soc,
                    None => true,
                }
                && match self.soc_lo_ramp_start {
                    Some(soc_lo_ramp_start) => soc_lo_ramp_start >= self.min_soc,
                    None => true,
                },
            format!(
                "{}\n`max_soc`: {} must be greater than `soc_hi_ramp_start`: {:?}, which must be greater than `soc_lo_ramp_start`: {:?}`, which must be greater than `min_soc`: {}`",
                format_dbg!(),
                self.max_soc.get::<si::ratio>(),
                self.soc_hi_ramp_start.map(|x| x.get::<si::ratio>()),
                self.soc_lo_ramp_start.map(|x| x.get::<si::ratio>()),
                self.min_soc.get::<si::ratio>(),
            )
        );
        Ok(())
    }
}

/// Controls which parameter to update when setting specific energy
pub enum SpecificEnergySideEffect {
    /// update mass
    Mass,
    /// update energy
    Energy,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[fastsim_api]
// component limits
/// ReversibleEnergyStorage state variables
pub struct ReversibleEnergyStorageState {
    // limits
    // TODO: create separate binning for cat power and
    /// maximum catenary power capability
    pub pwr_cat_max: si::Power,
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
    /// Chemical <-> Electrical conversion efficiency based on current power demand
    pub eff: si::Ratio,
    /// State of Health (SOH)
    pub soh: f64,

    // TODO: add `pwr_out_neg_electrical` and `pwr_out_pos_electrical` and corresponding energies
    // powers to separately pin negative- and positive-power operation
    /// total electrical power; positive is discharging
    pub pwr_out_electrical: si::Power,
    /// electrical power going to propulsion
    pub pwr_out_propulsion: si::Power,
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
    pub energy_out_propulsion: si::Energy,
    /// cumulative electrical energy going to aux loads
    pub energy_aux: si::Energy,
    /// cumulative energy dissipated as loss
    pub energy_loss: si::Energy,
    /// cumulative chemical energy; positive is discharging
    pub energy_out_chemical: si::Energy,

    /// dynamically updated max SOC limit
    pub max_soc: si::Ratio,
    /// dynamically updated min SOC limit
    pub min_soc: si::Ratio,

    /// component temperature
    // TODO: make this uom or figure out why it's not!
    pub temperature_celsius: f64,
}

impl Default for ReversibleEnergyStorageState {
    fn default() -> Self {
        Self {
            pwr_cat_max: si::Power::ZERO,
            pwr_prop_max: si::Power::ZERO,
            pwr_regen_max: si::Power::ZERO,
            pwr_disch_max: si::Power::ZERO,
            pwr_charge_max: si::Power::ZERO,
            i: Default::default(),
            soc: uc::R * 0.5,
            eff: si::Ratio::ZERO,
            soh: 0.,
            pwr_out_electrical: si::Power::ZERO,
            pwr_out_propulsion: si::Power::ZERO,
            pwr_aux: si::Power::ZERO,
            pwr_loss: si::Power::ZERO,
            pwr_out_chemical: si::Power::ZERO,
            energy_out_electrical: si::Energy::ZERO,
            energy_out_propulsion: si::Energy::ZERO,
            energy_aux: si::Energy::ZERO,
            energy_loss: si::Energy::ZERO,
            energy_out_chemical: si::Energy::ZERO,
            max_soc: si::Ratio::ZERO,
            min_soc: si::Ratio::ZERO,
            temperature_celsius: 22.,
        }
    }
}

impl Init for ReversibleEnergyStorageState {}
impl SerdeAPI for ReversibleEnergyStorageState {}
