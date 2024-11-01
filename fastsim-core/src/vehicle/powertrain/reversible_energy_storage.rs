use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

const TOL: f64 = 1e-3;

#[fastsim_api(
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

    #[getter]
    fn get_energy_capacity_usable_joules(&self) -> f64 {
        self.energy_capacity_usable().get::<si::joule>()
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
    /// Max output (and input) power battery can produce (accept)
    pub pwr_out_max: si::Power,

    /// Total energy capacity of battery of full discharge SOC of 0.0 and 1.0
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
    pub fn solve(&mut self, pwr_out_req: si::Power, dt: si::Time) -> anyhow::Result<()> {
        let state = &mut self.state;

        ensure!(
            state.soc <= self.max_soc || (pwr_out_req + state.pwr_aux) >= si::Power::ZERO,
            "{}\n{}",
            format_dbg!(pwr_out_req + state.pwr_aux),
            state.soc.get::<si::ratio>()
        );
        ensure!(
            state.soc >= self.min_soc || (pwr_out_req + state.pwr_aux) <= si::Power::ZERO,
            "{}\n{}",
            format_dbg!(pwr_out_req + state.pwr_aux),
            state.soc.get::<si::ratio>()
        );

        state.pwr_out_propulsion = pwr_out_req;
        state.pwr_out_electrical = state.pwr_out_propulsion + state.pwr_aux;

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
        state.eff = match self.eff_interp.to_owned() {
            Interpolator::Interp0D(eff) => eff * uc::R,
            Interpolator::Interp1D(interp1d) => {
                interp1d.interpolate(&[state.pwr_out_electrical.get::<si::watt>()])? * uc::R
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
                format_dbg!(state.eff >= 0.0 * uc::R && state.eff <= 1.0 * uc::R),
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
    /// # Arguments
    /// - `dt`: time step size
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
    /// - `dt`: time step size
    /// - `buffer`: buffer below static maximum SOC above which charging is disabled
    pub fn set_pwr_charge_max(
        &mut self,
        dt: si::Time,
        chrg_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive topping off of the battery
        let soc_buffer = (chrg_buffer / (self.energy_capacity * (self.max_soc - self.min_soc)))
            .max(si::Ratio::ZERO);
        ensure!(soc_buffer >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state.soc_regen_buffer = self.max_soc - soc_buffer;
        let pwr_max_for_dt =
            ((self.max_soc - self.state.soc) * self.energy_capacity / dt).max(si::Power::ZERO);
        self.state.pwr_charge_max = if self.state.soc <= self.max_soc - soc_buffer {
            self.pwr_out_max
        } else if self.state.soc < self.max_soc && soc_buffer > si::Ratio::ZERO {
            self.pwr_out_max * (self.state.soc - self.min_soc) / soc_buffer
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
            format_dbg!(soc_buffer)
        );

        Ok(())
    }

    /// # Arguments
    /// - `dt`: time step size
    /// - `buffer`: buffer above static minimum SOC above which charging is disabled
    pub fn set_pwr_disch_max(
        &mut self,
        dt: si::Time,
        disch_buffer: si::Energy,
    ) -> anyhow::Result<()> {
        // to protect against excessive bottoming out of the battery
        let soc_buffer = (disch_buffer / self.energy_capacity_usable()).max(si::Ratio::ZERO);
        ensure!(soc_buffer >= si::Ratio::ZERO, "{}", format_dbg!());
        self.state.soc_accel_buffer = self.min_soc + soc_buffer;
        let pwr_max_for_dt =
            ((self.state.soc - self.min_soc) * self.energy_capacity / dt).max(si::Power::ZERO);
        self.state.pwr_disch_max = if self.state.soc > self.min_soc + soc_buffer {
            self.pwr_out_max
        } else if self.state.soc > self.min_soc && soc_buffer > si::Ratio::ZERO {
            self.pwr_out_max * (self.state.soc - self.min_soc) / soc_buffer
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
            format_dbg!(soc_buffer)
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

    /// Usable energy capacity, accounting for SOC limits
    pub fn energy_capacity_usable(&self) -> si::Energy {
        self.energy_capacity * (self.max_soc - self.min_soc)
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

/// Controls which parameter to update when setting specific energy
pub enum SpecificEnergySideEffect {
    /// update mass
    Mass,
    /// update energy
    Energy,
}

#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
// component limits
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
    /// max state of charge (SOC) buffer for regen
    pub soc_regen_buffer: si::Ratio,
    /// min state of charge (SOC) buffer for acceleration
    pub soc_accel_buffer: si::Ratio,
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

    /// component temperature
    // TODO: make this uom or figure out why it's not!
    pub temperature_celsius: f64,
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
            soc_accel_buffer: si::Ratio::ZERO,
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
            temperature_celsius: 22.,
        }
    }
}

impl Init for ReversibleEnergyStorageState {}
impl SerdeAPI for ReversibleEnergyStorageState {}
