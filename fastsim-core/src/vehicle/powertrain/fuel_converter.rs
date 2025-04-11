use super::*;
use crate::prelude::*;
use std::f64::consts::PI;

// TODO: think about how to incorporate life modeling for Fuel Cells and other tech

#[fastsim_api(
    // optional, custom, struct-specific pymethods
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

    // #[getter("eff_range")]
    // fn get_eff_range_py(&self) -> PyResult<f64> {
    //     self.get_eff_range()?;
    //     Ok(())
    // }

    #[setter("__eff_range")]
    fn set_eff_range_py(&mut self, eff_range: f64) -> PyResult<()> {
        self.set_eff_range(eff_range)?;
        Ok(())
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
/// Struct for modeling [FuelConverter] (e.g. engine, fuel cell.) thermal plant
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct FuelConverter {
    /// [Self] Thermal plant, including thermal management controls
    #[serde(default, skip_serializing_if = "FuelConverterThermalOption::is_none")]
    #[has_state]
    pub thrml: FuelConverterThermalOption,
    /// [Self] mass
    #[serde(default)]
    pub(in super::super) mass: Option<si::Mass>,
    /// FuelConverter specific power
    pub(in super::super) specific_pwr: Option<si::SpecificPower>,
    /// max rated brake output power
    pub pwr_out_max: si::Power,
    /// starting/baseline transient power limit
    #[serde(default)]
    pub pwr_out_max_init: si::Power,
    // TODO: consider a ramp down rate, which may be needed for fuel cells
    /// lag time for ramp up
    pub pwr_ramp_lag: si::Time,
    /// interpolator for calculating [Self] efficiency as a function of output power
    pub eff_interp_from_pwr_out: Interpolator,
    /// power at which peak efficiency occurs
    #[serde(skip)]
    pub(crate) pwr_for_peak_eff: si::Power,
    /// idle fuel power to overcome internal friction (not including aux load) \[W\]
    pub pwr_idle_fuel: si::Power,
    /// time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    pub state: FuelConverterState,
    /// Custom vector of [Self::state]
    #[serde(
        default,
        skip_serializing_if = "FuelConverterStateHistoryVec::is_empty"
    )]
    pub history: FuelConverterStateHistoryVec,
}

impl SetCumulative for FuelConverter {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

impl SerdeAPI for FuelConverter {}
impl Init for FuelConverter {
    fn init(&mut self) -> Result<(), Error> {
        let _ = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.thrml.init()?;
        self.state
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        let eff_max = self
            .eff_max()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.pwr_for_peak_eff = *self
            .eff_interp_from_pwr_out
            .x()
            .map_err(|err| Error::InitError(format_dbg!(err)))?
            .get(
                self.eff_interp_from_pwr_out
                    .f_x()
                    .unwrap()
                    .iter()
                    .position(|&eff| eff * uc::R == eff_max)
                    .ok_or_else(|| Error::InitError(format_dbg!()))?,
            )
            .ok_or_else(|| Error::InitError(format_dbg!()))?
            * self.pwr_out_max;
        Ok(())
    }
}
impl HistoryMethods for FuelConverter {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        self.thrml.set_save_interval(save_interval)?;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear();
        self.thrml.clear();
    }
}

impl Mass for FuelConverter {
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
        } else if new_mass.is_none() {
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

// non-py methods
impl FuelConverter {
    /// Sets maximum possible total power [FuelConverter]
    /// can produce.
    /// # Arguments
    /// - `dt`: simulation time step size
    pub fn set_curr_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<()> {
        if self.pwr_out_max_init == si::Power::ZERO {
            // TODO: think about how to initialize power
            self.pwr_out_max_init = self.pwr_out_max / 10.
        };
        self.state.pwr_out_max.update(
            (self.state.pwr_prop + self.state.pwr_aux + self.pwr_out_max / self.pwr_ramp_lag * dt)
                .min(self.pwr_out_max)
                .max(self.pwr_out_max_init),
        )?;
        Ok(())
    }

    /// Sets maximum possible propulsion-related power [FuelConverter]
    /// can produce, accounting for any aux-related power required.
    /// # Arguments
    /// - `pwr_aux`: aux-related power required from this component
    pub fn set_curr_pwr_prop_max(&mut self, pwr_aux: si::Power) -> anyhow::Result<()> {
        ensure!(
            pwr_aux >= si::Power::ZERO,
            format!(
                "{}\n`pwr_aux` must be >= 0",
                format_dbg!(pwr_aux >= si::Power::ZERO),
            )
        );
        self.state.pwr_aux = pwr_aux;
        self.state.pwr_prop_max = *self.state.pwr_out_max.get()? - pwr_aux;
        Ok(())
    }

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: tractive power output required to achieve presribed speed
    /// - `fc_on`: whether component is actively running
    /// - `dt`: simulation time step size
    pub fn solve(
        &mut self,
        pwr_out_req: si::Power,
        fc_on: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.state.fc_on = fc_on;
        if fc_on {
            self.state.time_on += dt;
        } else {
            self.state.time_on = si::Time::ZERO;
        }
        // NOTE: think about the possibility of engine braking, not urgent
        ensure!(
            pwr_out_req >= si::Power::ZERO,
            format!(
                "{}\n`pwr_out_req` must be >= 0",
                format_dbg!(pwr_out_req >= si::Power::ZERO),
            )
        );
        // if the engine is not on, `pwr_out_req` should be 0.0
        ensure!(
            fc_on || (pwr_out_req == si::Power::ZERO && self.state.pwr_aux == si::Power::ZERO),
            format!(
                "{}\nEngine is off but pwr_out_req + pwr_aux is non-zero\n`pwr_out_req`: {} kW\n`self.state.pwr_aux`: {} kW",
                format_dbg!(
                    fc_on
                        || (pwr_out_req == si::Power::ZERO
                            && self.state.pwr_aux == si::Power::ZERO)
                ),
               pwr_out_req.get::<si::kilowatt>(),
               self.state.pwr_aux.get::<si::kilowatt>()
            )
        );
        self.state.pwr_prop = pwr_out_req;
        self.state.eff = if fc_on {
            uc::R
                * self
                    .eff_interp_from_pwr_out
                    .interpolate(&[
                        ((pwr_out_req + self.state.pwr_aux) / self.pwr_out_max).get::<si::ratio>()
                    ])
                    .with_context(|| {
                        anyhow!(
                            "{}\n failed to calculate {}",
                            format_dbg!(),
                            stringify!(self.state.eff)
                        )
                    })?
        } else {
            si::Ratio::ZERO
        } * self.thrml.temp_eff_coeff().unwrap_or(1.0 * uc::R);
        ensure!(
            (self.state.eff >= 0.0 * uc::R && self.state.eff <= 1.0 * uc::R),
            format!(
                "fc efficiency ({}) must be either between 0 and 1",
                self.state.eff.get::<si::ratio>()
            )
        );

        self.state.pwr_fuel = if self.state.fc_on {
            ((pwr_out_req + self.state.pwr_aux) / self.state.eff).max(self.pwr_idle_fuel)
        } else {
            si::Power::ZERO
        };
        self.state.pwr_loss = self.state.pwr_fuel - self.state.pwr_prop;

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

    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: Option<si::Power>,
        veh_state: &mut VehicleState,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let veh_speed = veh_state.speed_ach;
        self.thrml
            .solve(&self.state, te_amb, pwr_thrml_fc_to_cab, veh_speed, dt)
            .with_context(|| format_dbg!())
    }

    pub fn eff_max(&self) -> anyhow::Result<si::Ratio> {
        Ok(self
            .eff_interp_from_pwr_out
            .f_x()
            .with_context(|| format_dbg!())?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &curr| acc.max(curr))
            * uc::R)
    }

    /// If thermal model is appropriately configured, returns current lumped [Self] temperature
    pub fn temperature(&self) -> Option<si::Temperature> {
        match &self.thrml {
            FuelConverterThermalOption::FuelConverterThermal(fct) => Some(fct.state.temperature),
            FuelConverterThermalOption::None => None,
        }
    }

    /// If thermal model is appropriately configured, returns previous time step
    /// lumped [Self] temperature
    pub fn temp_prev(&self) -> Option<si::Temperature> {
        match &self.thrml {
            FuelConverterThermalOption::FuelConverterThermal(fct) => Some(fct.state.temp_prev),
            FuelConverterThermalOption::None => None,
        }
    }

    /// Returns max value of [Self::eff_interp_from_pwr_out]
    pub fn get_eff_max(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_from_pwr_out
            .f_x()?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }

    /// Returns min value of [Self::eff_interp_from_pwr_out]
    pub fn get_eff_min(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_from_pwr_out
            .f_x()?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.min(*curr)))
    }

    /// Scales eff_interp_fwd and eff_interp_bwd by ratio of new `eff_max` per current calculated max
    pub fn set_eff_max(&mut self, eff_max: f64) -> anyhow::Result<()> {
        if (0.0..=1.0).contains(&eff_max) {
            let old_max = self.get_eff_max()?;
            let f_x = self.eff_interp_from_pwr_out.f_x()?.to_owned();
            match &mut self.eff_interp_from_pwr_out {
                interp @ Interpolator::Interp1D(..) => {
                    interp.set_f_x(f_x.iter().map(|x| x * eff_max / old_max).collect())?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
        } else {
            return Err(anyhow!(
                "`eff_max` ({:.3}) must be between 0.0 and 1.0",
                eff_max,
            ));
        }
        // to update any dependent fields
        self.init()
            .map_err(|err| anyhow!("{}\n{err}", format_dbg!()))?;
        Ok(())
    }

    /// Scales values of `eff_interp_fwd.f_x` and `eff_interp_bwd.f_x` without changing max such that max - min
    /// is equal to new range.  Will change max if needed to ensure no values are
    /// less than zero.
    pub fn set_eff_range(&mut self, eff_range: f64) -> anyhow::Result<()> {
        let eff_max = self.get_eff_max()?;
        if eff_range == 0.0 {
            let f_x = vec![
                eff_max;
                self.eff_interp_from_pwr_out
                    .f_x()
                    .with_context(|| "eff_interp_fwd does not have f_x field")?
                    .len()
            ];
            self.eff_interp_from_pwr_out.set_f_x(f_x)?;
        } else if (0.0..=1.0).contains(&eff_range) {
            let old_min = self.get_eff_min()?;
            let old_range = self.get_eff_max()? - old_min;
            if old_range == 0.0 {
                return Err(anyhow!(
                    "`eff_range` is already zero so it cannot be modified."
                ));
            }
            let f_x_fwd = self.eff_interp_from_pwr_out.f_x()?.to_owned();
            match &mut self.eff_interp_from_pwr_out {
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
                let f_x_fwd = self.eff_interp_from_pwr_out.f_x()?.to_owned();
                match &mut self.eff_interp_from_pwr_out {
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
        } else {
            return Err(anyhow!(format!(
                "`eff_range` ({:.3}) must be between 0.0 and 1.0",
                eff_range,
            )));
        }
        // to update any dependent fields
        self.init()
            .map_err(|err| anyhow!("{}\n{err}", format_dbg!()))?;

        Ok(())
    }
}

// impl FuelConverter {
//     impl_get_set_eff_max_min!();
//     impl_get_set_eff_range!();
// }

#[fastsim_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[non_exhaustive]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct FuelConverterState {
    /// time step index
    pub i: usize,
    /// max total output power fc can produce at current time
    pub pwr_out_max: TrackedState<si::Power>,
    /// max propulsion power fc can produce at current time
    pub pwr_prop_max: si::Power,
    /// efficiency evaluated at current demand
    pub eff: si::Ratio,
    /// instantaneous power going to drivetrain, not including aux
    pub pwr_prop: si::Power,
    /// integral of [Self::pwr_prop]
    pub energy_prop: si::Energy,
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
    /// Time the engine has been on
    pub time_on: si::Time,
}

impl SerdeAPI for FuelConverterState {}
impl Init for FuelConverterState {}

/// Options for handling [FuelConverter] thermal model
#[derive(
    Clone, Default, Debug, Serialize, Deserialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum FuelConverterThermalOption {
    /// Basic thermal plant for [FuelConverter]
    FuelConverterThermal(Box<FuelConverterThermal>),
    /// no thermal plant for [FuelConverter]
    #[default]
    None,
}

impl StateMethods for FuelConverterThermalOption {
    fn save_state(&mut self) {
        match self {
            Self::FuelConverterThermal(fct) => fct.save_state(),
            Self::None => {}
        }
    }
    fn check_and_reset(&mut self) -> anyhow::Result<()> {
        match self {
            Self::FuelConverterThermal(fct) => fct.check_and_reset()?,
            Self::None => {}
        }
        Ok(())
    }
}
impl Step for FuelConverterThermalOption {
    fn step(&mut self) {
        match self {
            Self::FuelConverterThermal(fct) => fct.step(),
            Self::None => {}
        }
    }
}
impl Init for FuelConverterThermalOption {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::FuelConverterThermal(fct) => fct.init()?,
            Self::None => {}
        }
        Ok(())
    }
}
impl SerdeAPI for FuelConverterThermalOption {}
impl SetCumulative for FuelConverterThermalOption {
    fn set_cumulative(&mut self, dt: si::Time) {
        match self {
            Self::FuelConverterThermal(fct) => fct.set_cumulative(dt),
            Self::None => {}
        }
    }
}
impl HistoryMethods for FuelConverterThermalOption {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        match self {
            FuelConverterThermalOption::FuelConverterThermal(fct) => fct.save_interval(),
            FuelConverterThermalOption::None => Ok(None),
        }
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        match self {
            FuelConverterThermalOption::FuelConverterThermal(fct) => {
                fct.set_save_interval(save_interval)
            }
            FuelConverterThermalOption::None => Ok(()),
        }
    }
    fn clear(&mut self) {
        match self {
            FuelConverterThermalOption::FuelConverterThermal(fct) => {
                fct.clear();
            }
            FuelConverterThermalOption::None => {}
        }
    }
}
impl FuelConverterThermalOption {
    /// Solve change in temperature and other thermal effects
    /// # Arguments
    /// - `fc_state`: [FuelConverter] state
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_fc_to_cab`: heat demand from [Vehicle::hvac] system -- zero if `None` is passed
    /// - `veh_speed`: current achieved speed
    fn solve(
        &mut self,
        fc_state: &FuelConverterState,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: Option<si::Power>,
        veh_speed: si::Velocity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        match self {
            Self::FuelConverterThermal(fct) => fct
                .solve(
                    fc_state,
                    te_amb,
                    pwr_thrml_fc_to_cab.unwrap_or_default(),
                    veh_speed,
                    dt,
                )
                .with_context(|| format_dbg!())?,
            Self::None => {
                ensure!(
                    pwr_thrml_fc_to_cab.is_none(),
                    format_dbg!(
                        "`FuelConverterThermal needs to be configured to provide heat demand`"
                    )
                );
            }
        }
        Ok(())
    }

    /// If appropriately configured, returns temperature-dependent efficiency coefficient
    fn temp_eff_coeff(&self) -> Option<si::Ratio> {
        match self {
            Self::FuelConverterThermal(fct) => Some(fct.state.eff_coeff),
            Self::None => None,
        }
    }
}

#[fastsim_api(
    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Default::default()
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
/// Struct for modeling Fuel Converter (e.g. engine, fuel cell.)
pub struct FuelConverterThermal {
    /// [FuelConverter] thermal capacitance
    pub heat_capacitance: si::HeatCapacity,
    /// parameter for engine characteristic length for heat transfer calcs
    pub length_for_convection: si::Length,
    /// parameter for heat transfer coeff from [FuelConverter] to ambient during vehicle stop
    pub htc_to_amb_stop: si::HeatTransferCoeff,

    /// Heat transfer coefficient between adiabatic flame temperature and [FuelConverterThermal] temperature
    pub conductance_from_comb: si::ThermalConductance,
    /// Max coefficient for fraction of combustion heat that goes to [FuelConverter]
    /// (engine) thermal mass. Remainder goes to environment (e.g. via tailpipe).
    pub max_frac_from_comb: si::Ratio,
    /// parameter for temperature at which thermostat starts to open
    pub tstat_te_sto: Option<si::Temperature>,
    /// temperature delta over which thermostat is partially open
    pub tstat_te_delta: Option<si::TemperatureInterval>,
    #[serde(default = "tstat_interp_default")]
    pub tstat_interp: Interpolator,
    /// Radiator effectiveness -- ratio of active heat rejection from
    /// radiator to passive heat rejection, always greater than 1
    pub radiator_effectiveness: si::Ratio,
    /// Model for [FuelConverter] dependence on efficiency
    pub fc_eff_model: FCTempEffModel,
    /// struct for tracking current state
    #[serde(default)]
    pub state: FuelConverterThermalState,
    /// Custom vector of [Self::state]
    #[serde(
        default,
        skip_serializing_if = "FuelConverterThermalStateHistoryVec::is_empty"
    )]
    pub history: FuelConverterThermalStateHistoryVec,
    pub save_interval: Option<usize>,
}

impl HistoryMethods for FuelConverterThermal {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear();
    }
}

/// Dummy interpolator that will be overridden in [FuelConverterThermal::init]
fn tstat_interp_default() -> Interpolator {
    Interpolator::new_1d(
        vec![85.0, 90.0],
        vec![0.0, 1.0],
        Strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap()
}

lazy_static! {
    /// gasoline stoichiometric air-fuel ratio https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio
    pub static ref AFR_STOICH_GASOLINE: si::Ratio = uc::R * 14.7;
    /// gasoline density in https://inchem.org/documents/icsc/icsc/eics1400.htm
    /// This is reasonably constant with respect to temperature and pressure
    pub static ref GASOLINE_DENSITY: si::MassDensity = 0.75 * uc::KG / uc::L;
    /// TODO: find a source for this value
    pub static ref GASOLINE_LHV: si::SpecificEnergy = 33.7 * uc::KWH / uc::GALLON / *GASOLINE_DENSITY;
    pub static ref TE_ADIABATIC_STD: si::Temperature = Air::get_te_from_u(
            Air::get_specific_energy(*TE_STD_AIR).with_context(|| format_dbg!()).unwrap()
                + (Octane::get_specific_energy(*TE_STD_AIR).with_context(|| format_dbg!()).unwrap()
                    + *GASOLINE_LHV)
                    / *AFR_STOICH_GASOLINE,
        )
        .with_context(|| format_dbg!()).unwrap();
}

impl FuelConverterThermal {
    /// Solve change in temperature and other thermal effects
    /// # Arguments
    /// - `fc_state`: [FuelConverter] state
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_fc_to_cab`: heat demand from [Vehicle::hvac] system
    /// - `veh_speed`: current achieved speed
    /// - `dt`: simulation time step size
    fn solve(
        &mut self,
        fc_state: &FuelConverterState,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: si::Power,
        veh_speed: si::Velocity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.state.pwr_thrml_fc_to_cab = pwr_thrml_fc_to_cab;
        // film temperature for external convection calculations
        let te_air_film: si::Temperature = 0.5
            * (self.state.temperature.get::<si::kelvin_abs>() + te_amb.get::<si::kelvin_abs>())
            * uc::KELVIN;
        // Reynolds number = density * speed * diameter / dynamic viscosity
        // NOTE: might be good to pipe in elevation
        let fc_air_film_re =
            Air::get_density(Some(te_air_film), None) * veh_speed * self.length_for_convection
                / Air::get_dyn_visc(te_air_film).with_context(|| format_dbg!())?;

        // calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
        self.state.htc_to_amb = if veh_speed < 1.0 * uc::MPS {
            // if stopped, scale based on thermostat opening and constant convection
            self.state.tstat_open_frac = self
                .tstat_interp
                .interpolate(&[self.state.temperature.get::<si::degree_celsius>()])
                .with_context(|| format_dbg!())?;
            (uc::R + self.state.tstat_open_frac * self.radiator_effectiveness)
                * self.htc_to_amb_stop
        } else {
            // Calculate heat transfer coefficient for sphere,
            // from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            let sphere_conv_params = get_sphere_conv_params(fc_air_film_re.get::<si::ratio>());
            let htc_to_amb_sphere: si::HeatTransferCoeff = sphere_conv_params.0
                * fc_air_film_re.get::<si::ratio>().powf(sphere_conv_params.1)
                * Air::get_pr(te_air_film)
                    .with_context(|| format_dbg!())?
                    .get::<si::ratio>()
                    .powf(1.0 / 3.0)
                * Air::get_therm_cond(te_air_film).with_context(|| format_dbg!())?
                / self.length_for_convection;
            // if stopped, scale based on thermostat opening and constant convection
            self.state.tstat_open_frac = self
                .tstat_interp
                .interpolate(&[self.state.temperature.get::<si::degree_celsius>()])
                .with_context(|| format_dbg!())?;
            self.state.tstat_open_frac * htc_to_amb_sphere
        };

        self.state.pwr_thrml_to_amb =
            self.state.htc_to_amb * PI * self.length_for_convection.powi(typenum::P2::new()) / 4.0
                * (self.state.temperature.get::<si::degree_celsius>()
                    - te_amb.get::<si::degree_celsius>())
                * uc::KELVIN_INT;

        // let heat_to_amb = ;
        // assumes fuel/air mixture is entering combustion chamber at block temperature
        // assumes stoichiometric combustion
        self.state.te_adiabatic = Air::get_te_from_u(
            Air::get_specific_energy(self.state.temperature).with_context(|| format_dbg!())?
                + (Octane::get_specific_energy(self.state.temperature)
                    .with_context(|| format_dbg!())?
                    // TODO: make config. for other fuels -- e.g. with enum for specific fuels and/or fuel properties
                    + *GASOLINE_LHV)
                    / *AFR_STOICH_GASOLINE,
        )
        .with_context(|| format_dbg!())?;
        // heat that will go both to the block and out the exhaust port
        self.state.pwr_fuel_as_heat = fc_state.pwr_fuel - (fc_state.pwr_prop + fc_state.pwr_aux);
        self.state.pwr_thrml_to_tm = (self.conductance_from_comb
            * (self.state.te_adiabatic.get::<si::degree_celsius>()
                - self.state.temperature.get::<si::degree_celsius>())
            * uc::KELVIN_INT)
            .min(self.max_frac_from_comb * self.state.pwr_fuel_as_heat);
        let delta_temp: si::TemperatureInterval = ((self.state.pwr_thrml_to_tm
            - self.state.pwr_thrml_fc_to_cab
            - self.state.pwr_thrml_to_amb)
            * dt)
            / self.heat_capacitance;
        self.state.temp_prev = self.state.temperature;
        // Interestingly, it seems to be ok to add a `TemperatureInterval` to a `Temperature` here
        self.state.temperature += delta_temp;

        self.state.eff_coeff = match self.fc_eff_model {
            FCTempEffModel::Linear(FCTempEffModelLinear {
                offset,
                slope_per_kelvin: slope,
                minimum,
            }) => minimum.max(
                {
                    let calc_unbound: si::Ratio =
                        offset + slope * uc::R / uc::KELVIN * self.state.temperature;
                    calc_unbound
                }
                .min(1.0 * uc::R),
            ),
            FCTempEffModel::Exponential(FCTempEffModelExponential {
                offset,
                lag,
                minimum,
            }) => {
                let dte: si::TemperatureInterval = (self.state.temperature.get::<si::kelvin_abs>()
                    - offset.get::<si::kelvin_abs>())
                    * uc::KELVIN_INT;
                ((1.0 - f64::exp((-dte / lag).get::<si::ratio>())) * uc::R).max(minimum)
            }
        };
        Ok(())
    }
}
impl SerdeAPI for FuelConverterThermal {}
impl SetCumulative for FuelConverterThermal {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}
impl Init for FuelConverterThermal {
    fn init(&mut self) -> Result<(), Error> {
        self.tstat_te_sto = self
            .tstat_te_sto
            .or(Some((85. + uc::CELSIUS_TO_KELVIN) * uc::KELVIN));
        self.tstat_te_delta = self.tstat_te_delta.or(Some(5. * uc::KELVIN_INT));
        self.tstat_interp = Interpolator::new_1d(
            vec![
                self.tstat_te_sto.unwrap().get::<si::degree_celsius>(),
                self.tstat_te_sto.unwrap().get::<si::degree_celsius>()
                    + self.tstat_te_delta.unwrap().get::<si::kelvin>(),
            ],
            vec![0.0, 1.0],
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .map_err(|err| {
            Error::InitError(format!(
                "{}\n{}\n{}",
                err,
                format_dbg!(self.tstat_te_sto),
                format_dbg!(self.tstat_te_delta)
            ))
        })?;
        Ok(())
    }
}
impl Default for FuelConverterThermal {
    fn default() -> Self {
        let mut fct = Self {
            heat_capacitance: Default::default(),
            length_for_convection: Default::default(),
            htc_to_amb_stop: Default::default(),
            conductance_from_comb: Default::default(),
            max_frac_from_comb: Default::default(),
            tstat_te_sto: None,
            tstat_te_delta: None,
            tstat_interp: tstat_interp_default(),
            radiator_effectiveness: Default::default(),
            fc_eff_model: Default::default(),
            state: Default::default(),
            history: Default::default(),
            save_interval: Some(1),
        };
        fct.init().unwrap();
        fct
    }
}

#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct FuelConverterThermalState {
    /// time step index
    pub i: usize,
    /// Adiabatic flame temperature assuming complete (i.e. all fuel is consumed
    /// if fuel lean or stoich or all air is consumed if fuel rich) combustion
    pub te_adiabatic: si::Temperature,
    /// Current engine thermal mass temperature (lumped engine block and coolant)
    pub temperature: si::Temperature,
    /// Engine thermal mass temperature (lumped engine block and coolant) at previous time step
    pub temp_prev: si::Temperature,
    /// thermostat open fraction (1 = fully open, 0 = fully closed)
    pub tstat_open_frac: f64,
    /// Current heat transfer coefficient from [FuelConverter] to ambient
    pub htc_to_amb: si::HeatTransferCoeff,
    /// Current heat transfer power to ambient
    pub pwr_thrml_to_amb: si::Power,
    /// Cumulative heat transfer energy to ambient
    pub energy_thrml_to_amb: si::Energy,
    /// Efficency coefficient, used to modify [FuelConverter] effciency based on temperature
    pub eff_coeff: si::Ratio,
    /// Thermal power flowing from fuel converter to cabin
    pub pwr_thrml_fc_to_cab: si::Power,
    /// Cumulative thermal energy flowing from fuel converter to cabin
    pub energy_thrml_fc_to_cab: si::Energy,
    /// Fuel power that is not converted to mechanical work
    pub pwr_fuel_as_heat: si::Power,
    /// Cumulative fuel energy that is not converted to mechanical work
    pub energy_fuel_as_heat: si::Energy,
    /// Thermal power flowing from combustion to [FuelConverter] thermal mass
    pub pwr_thrml_to_tm: si::Power,
    /// Cumulative thermal energy flowing from combustion to [FuelConverter] thermal mass
    pub energy_thrml_to_tm: si::Energy,
}

impl Init for FuelConverterThermalState {}
impl SerdeAPI for FuelConverterThermalState {}
impl Default for FuelConverterThermalState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            te_adiabatic: *TE_ADIABATIC_STD,
            temperature: *TE_STD_AIR,
            temp_prev: *TE_STD_AIR,
            tstat_open_frac: Default::default(),
            htc_to_amb: Default::default(),
            eff_coeff: uc::R,
            pwr_thrml_fc_to_cab: Default::default(),
            energy_thrml_fc_to_cab: Default::default(),
            pwr_thrml_to_amb: Default::default(),
            energy_thrml_to_amb: Default::default(),
            pwr_fuel_as_heat: Default::default(),
            energy_fuel_as_heat: Default::default(),
            pwr_thrml_to_tm: Default::default(),
            energy_thrml_to_tm: Default::default(),
        }
    }
}

/// Model variants for how FC efficiency depends on temperature
#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum FCTempEffModel {
    /// Linear temperature dependence
    Linear(FCTempEffModelLinear),
    /// Exponential temperature dependence
    Exponential(FCTempEffModelExponential),
}

impl Default for FCTempEffModel {
    fn default() -> Self {
        FCTempEffModel::Exponential(FCTempEffModelExponential::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FCTempEffModelLinear {
    pub offset: si::Ratio,
    /// Change in efficiency factor per change in temperature /[K/]
    pub slope_per_kelvin: f64,
    pub minimum: si::Ratio,
}

impl Default for FCTempEffModelLinear {
    fn default() -> Self {
        Self {
            offset: 0.0 * uc::R,
            slope_per_kelvin: 25.0,
            minimum: 0.2 * uc::R,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FCTempEffModelExponential {
    /// temperature at which `fc_eta_temp_coeff` begins to grow
    pub offset: si::Temperature,
    /// exponential lag parameter [K^-1]
    pub lag: si::TemperatureInterval,
    /// minimum value that `fc_eta_temp_coeff` can take
    pub minimum: si::Ratio,
}

impl Default for FCTempEffModelExponential {
    fn default() -> Self {
        Self {
            // TODO: update after reasonable calibration
            offset: 0.0 * uc::KELVIN,
            lag: 25.0 * uc::KELVIN_INT,
            minimum: 0.2 * uc::R,
        }
    }
}
