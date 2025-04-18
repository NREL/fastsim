use super::*;
// TODO: add parameters and/or cabin model variant for solar heat load

/// Options for handling cabin thermal model
#[derive(
    Clone, Default, Debug, Serialize, Deserialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum CabinOption {
    /// Basic single thermal capacitance cabin thermal model, including HVAC
    /// system and controls
    LumpedCabin(Box<LumpedCabin>),
    /// Cabin with interior and shell capacitances
    LumpedCabinWithShell,
    /// no cabin thermal model
    #[default]
    None,
}
impl SaveState for CabinOption {
    fn save_state(&mut self) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.save_state()?,
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::None => {}
        }
        Ok(())
    }
}
impl TrackedStateMethods for CabinOption {
    fn check_and_reset(&mut self, loc: String) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.check_and_reset(format!("{}\n{loc}", format_dbg!()))?,
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::None => {}
        }
        Ok(())
    }
}
impl Step for CabinOption {
    fn step(&mut self) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.step(),
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::None => Ok(()),
        }
    }
}
impl Init for CabinOption {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::LumpedCabin(scc) => scc.init()?,
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::None => {}
        }
        Ok(())
    }
}
impl SerdeAPI for CabinOption {}
impl HistoryMethods for CabinOption {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        match self {
            CabinOption::LumpedCabin(lc) => lc.save_interval(),
            CabinOption::LumpedCabinWithShell => todo!(),
            CabinOption::None => Ok(None),
        }
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        match self {
            CabinOption::LumpedCabin(lc) => lc.set_save_interval(save_interval),
            CabinOption::LumpedCabinWithShell => todo!(),
            CabinOption::None => Ok(()),
        }
    }
    fn clear(&mut self) {
        match self {
            CabinOption::LumpedCabin(lc) => lc.clear(),
            CabinOption::LumpedCabinWithShell => todo!(),
            CabinOption::None => {}
        }
    }
}
impl SetCumulative for CabinOption {
    fn set_cumulative(&mut self, dt: si::Time) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.set_cumulative(dt)?,
            Self::LumpedCabinWithShell => todo!(),
            Self::None => {}
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
#[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
/// Basic single thermal capacitance cabin thermal model, including HVAC
/// system and controls
pub struct LumpedCabin {
    /// Inverse of cabin shell thermal resistance
    pub cab_shell_htc_to_amb: si::HeatTransferCoeff,
    /// parameter for heat transfer coeff from cabin outer surface to ambient
    /// during vehicle stop
    pub cab_htc_to_amb_stop: si::HeatTransferCoeff,
    /// cabin thermal capacitance
    pub heat_capacitance: si::HeatCapacity,
    /// cabin length, modeled as a flat plate
    pub length: si::Length,
    /// cabin width, modeled as a flat plate
    pub width: si::Length,
    #[serde(default)]
    pub state: LumpedCabinState,
    #[serde(default, skip_serializing_if = "LumpedCabinStateHistoryVec::is_empty")]
    pub history: LumpedCabinStateHistoryVec,
    /// Time step interval at which history is saved
    pub save_interval: Option<usize>,
}
impl SetCumulative for LumpedCabin {
    fn set_cumulative(&mut self, dt: si::Time) -> anyhow::Result<()> {
        self.state.set_cumulative(dt)
    }
}
impl SerdeAPI for LumpedCabin {}
impl Init for LumpedCabin {}
impl HistoryMethods for LumpedCabin {
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

impl LumpedCabin {
    /// Solve temperatures, HVAC powers, and cumulative energies of cabin and HVAC system
    /// Arguments:
    /// - `te_amb_air`: ambient air temperature
    /// - `veh_state`: current [VehicleState]
    /// - 'pwr_thrml_from_hvac`: power to cabin from [Vehicle::hvac] system
    /// - `dt`: simulation time step size
    /// # Returns
    /// - `te_cab`: current cabin temperature, after solving cabin for current
    ///     simulation time step
    pub fn solve(
        &mut self,
        te_amb_air: si::Temperature,
        veh_state: &VehicleState,
        pwr_thrml_from_hvac: si::Power,
        pwr_thrml_to_res: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Temperature> {
        self.state
            .pwr_thrml_from_hvac
            .update(pwr_thrml_from_hvac, format_dbg!())?;
        self.state
            .pwr_thrml_to_res
            .update(pwr_thrml_to_res, format_dbg!())?;
        // flat plate model for isothermal, mixed-flow from Incropera and deWitt, Fundamentals of Heat and Mass
        // Transfer, 7th Edition
        let cab_te_film_ext: si::Temperature = 0.5
            * (self
                .state
                .temperature
                .get_fresh(format_dbg!())?
                .get::<si::kelvin_abs>()
                + te_amb_air.get::<si::kelvin_abs>())
            * uc::KELVIN;
        self.state.reynolds_for_plate.update(
            Air::get_density(
                Some(cab_te_film_ext),
                Some(*veh_state.elev_curr.get_fresh(format_dbg!())?),
            ) * *veh_state.speed_ach.get_fresh(format_dbg!())?
                * self.length
                / Air::get_dyn_visc(cab_te_film_ext).with_context(|| format_dbg!())?,
            format_dbg!(),
        )?;
        let re_l_crit = 5.0e5 * uc::R; // critical Re for transition to turbulence

        let nu_l_bar: si::Ratio = if *self.state.reynolds_for_plate.get_fresh(format_dbg!())? < re_l_crit
        {
            // equation 7.30
            0.664
                * self
                    .state
                    .reynolds_for_plate
                    .get_fresh(format_dbg!())?
                    .get::<si::ratio>()
                    .powf(0.5)
                * Air::get_pr(cab_te_film_ext)
                    .with_context(|| format_dbg!())?
                    .get::<si::ratio>()
                    .powf(1.0 / 3.0)
                * uc::R
        } else {
            // equation 7.38
            let a = 871.0; // equation 7.39
            (0.037
                * self
                    .state
                    .reynolds_for_plate
                    .get_fresh(format_dbg!())?
                    .get::<si::ratio>()
                    .powf(0.8)
                - a)
                * Air::get_pr(cab_te_film_ext).with_context(|| format_dbg!())?
        };

        self.state.pwr_thrml_from_amb.update(
            if *veh_state.speed_ach.get_fresh(format_dbg!())? > 2.0 * uc::MPH {
                let htc_overall_moving: si::HeatTransferCoeff = 1.0
                    / (1.0
                        / (nu_l_bar
                            * Air::get_therm_cond(cab_te_film_ext)
                                .with_context(|| format_dbg!())?
                            / self.length)
                        + 1.0 / self.cab_shell_htc_to_amb);
                (self.length * self.width)
                    * htc_overall_moving
                    * (te_amb_air.get::<si::degree_celsius>()
                        - self
                            .state
                            .temperature
                            .get_fresh(format_dbg!())?
                            .get::<si::degree_celsius>())
                    * uc::KELVIN_INT
            } else {
                (self.length * self.width)
                    / (1.0 / self.cab_htc_to_amb_stop + 1.0 / self.cab_shell_htc_to_amb)
                    * (te_amb_air.get::<si::degree_celsius>()
                        - self
                            .state
                            .temperature
                            .get_fresh(format_dbg!())?
                            .get::<si::degree_celsius>())
                    * uc::KELVIN_INT
            },
            format_dbg!(),
        )?;

        self.state.temperature.update(
            *self.state.temperature.get_prev_or_curr(format_dbg!())?
                + (*self.state.pwr_thrml_from_hvac.get_fresh(format_dbg!())?
                    + *self.state.pwr_thrml_from_amb.get_fresh(format_dbg!())?
                    - *self.state.pwr_thrml_to_res.get_fresh(format_dbg!())?)
                    / self.heat_capacitance
                    * dt,
            format_dbg!(),
        )?;
        Ok(*self.state.temperature.get_fresh(format_dbg!())?)
    }
}

#[fastsim_api]
#[derive(
    Clone, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative, StateMethods,
)]
#[serde(deny_unknown_fields)]
pub struct LumpedCabinState {
    /// time step counter
    pub i: TrackedState<usize>,
    /// lumped cabin temperature
    pub temperature: TrackedState<si::Temperature>,
    /// Thermal power coming to cabin from [Vehicle::hvac] system.  Positive indicates
    /// heating, and negative indicates cooling.
    pub pwr_thrml_from_hvac: TrackedState<si::Power>,
    /// Cumulative thermal energy coming to cabin from [Vehicle::hvac] system.
    /// Positive indicates heating, and negative indicates cooling.
    pub energy_thrml_from_hvac: TrackedState<si::Energy>,
    /// Thermal power coming to cabin from ambient air.  Positive indicates
    /// heating, and negative indicates cooling.
    pub pwr_thrml_from_amb: TrackedState<si::Power>,
    /// Cumulative thermal energy coming to cabin from ambient air.  Positive indicates
    /// heating, and negative indicates cooling.
    pub energy_thrml_from_amb: TrackedState<si::Energy>,
    /// Thermal power flowing from [Cabin] to [ReversibleEnergyStorage] (zero if
    /// not equipped) due to temperature delta
    pub pwr_thrml_to_res: TrackedState<si::Power>,
    /// Cumulative thermal energy flowing from [Cabin] to
    /// [ReversibleEnergyStorage] due to temperature delta
    pub energy_thrml_to_res: TrackedState<si::Energy>,
    /// Reynolds number for flow over cabin, treating cabin as a flat plate
    pub reynolds_for_plate: TrackedState<si::Ratio>,
}

impl Default for LumpedCabinState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            temperature: TrackedState::new(*TE_STD_AIR),
            pwr_thrml_from_hvac: Default::default(),
            energy_thrml_from_hvac: Default::default(),
            pwr_thrml_from_amb: Default::default(),
            energy_thrml_from_amb: Default::default(),
            pwr_thrml_to_res: Default::default(),
            energy_thrml_to_res: Default::default(),
            reynolds_for_plate: Default::default(),
        }
    }
}
impl Init for LumpedCabinState {}
impl SerdeAPI for LumpedCabinState {}
