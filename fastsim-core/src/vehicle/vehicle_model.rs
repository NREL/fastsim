use super::{hev::HEVPowertrainControls, hev::HEVStopStartControl, *};
use crate::{
    prelude::*,
    vehicle::conv::{ConvPowertrainControls, ConvStopStartControl},
};
pub mod fastsim2_interface;

/// Possible aux load power sources
#[derive(
    Clone, Debug, Serialize, Deserialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum AuxSource {
    /// Aux load power provided by ReversibleEnergyStorage with help from FuelConverter, if present
    /// and needed
    ReversibleEnergyStorage,
    /// Aux load power provided by FuelConverter with help from ReversibleEnergyStorage, if present
    /// and needed
    FuelConverter,
}

impl SerdeAPI for AuxSource {}
impl Init for AuxSource {}

#[serde_api]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, StateMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
/// Struct for simulating vehicle
pub struct Vehicle {
    /// Vehicle name
    pub name: String,
    /// Documentation (e.g. how this file was generated, calibration details)]
    pub doc: Option<String>,
    /// Year manufactured
    pub year: u32,
    #[has_state]
    /// type of vehicle powertrain including contained type-specific parameters and variables
    pub pt_type: PowertrainType,

    /// Chassis model with various chassis-related parameters
    pub chassis: Chassis,

    /// Cabin thermal model
    #[has_state]
    #[serde(default)]
    pub cabin: CabinOption,

    /// HVAC model
    #[has_state]
    #[serde(default)]
    pub hvac: HVACOption,

    /// Total vehicle mass
    pub(crate) mass: Option<si::Mass>,

    /// Baseline power required by auxilliary systems
    pub pwr_aux_base: si::Power,

    /// time step interval at which `state` is saved into `history`
    save_interval: Option<usize>,
    /// current state of vehicle
    #[serde(default)]
    pub state: VehicleState,
    /// Vector-like history of [Self::state]
    #[serde(default)]
    pub history: VehicleStateHistoryVec,
}

#[pyo3_api]
impl Vehicle {
    #[staticmethod]
    fn try_from_fastsim2(veh: fastsim_2::vehicle::RustVehicle) -> PyResult<Vehicle> {
        Ok(Self::try_from(veh.clone())?)
    }

    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> PyResult<()> {
        self.set_save_interval(save_interval)
            .map_err(|e| PyAttributeError::new_err(e.to_string()))
    }

    // despite having `getter` here, this seems to work as a function
    #[getter("save_interval")]
    /// Set save interval and cascade to nested components.
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        self.save_interval()
    }

    #[getter]
    fn get_fc(&self) -> Option<FuelConverter> {
        self.fc().cloned()
    }

    #[getter]
    fn get_res(&self) -> Option<ReversibleEnergyStorage> {
        self.res().cloned()
    }

    #[getter]
    fn get_em(&self) -> Option<ElectricMachine> {
        self.em().cloned()
    }

    fn veh_type(&self) -> String {
        self.pt_type.to_string()
    }

    // #[getter]
    // fn get_pwr_rated_kilowatts(&self) -> f64 {
    //     self.get_pwr_rated().get::<si::kilowatt>()
    // }

    // #[getter]
    // fn get_mass_kg(&self) -> PyResult<Option<f64>> {
    //     Ok(self.mass()?.map(|m| m))
    // }

    /// Load vehicle from file saved in fastsim-2 format
    #[pyo3(name = "from_f2_file")]
    #[staticmethod]
    fn from_f2_file_py(file: PathBuf) -> anyhow::Result<Self> {
        Self::from_f2_file(file)
    }

    #[pyo3(name = "to_fastsim2")]
    fn to_fastsim2_py(&self) -> anyhow::Result<fastsim_2::vehicle::RustVehicle> {
        self.to_fastsim2()
    }

    #[pyo3(name = "reset_py")]
    /// Combines [Self::reset_cumulative], [Self::reset_step], [Self::clear]
    fn reset_py(&mut self) -> anyhow::Result<()> {
        self.reset_cumulative(|| format_dbg!())?;
        self.reset_step(|| format_dbg!())?;
        self.clear();
        Ok(())
    }

    #[pyo3(name = "clear")]
    fn clear_py(&mut self) {
        self.clear()
    }

    #[pyo3(name = "reset_step")]
    fn reset_step_py(&mut self) -> anyhow::Result<()> {
        self.reset_step(|| format_dbg!())
    }

    #[pyo3(name = "reset_cumulative")]
    fn reset_cumulative_py(&mut self) -> anyhow::Result<()> {
        self.reset_cumulative(|| format_dbg!())
    }

    #[pyo3(name = "use_stop_start_controller")]
    fn use_stop_start_controller_py(&mut self) -> anyhow::Result<()> {
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(veh) => {
                match veh.pt_cntrl {
                    ConvPowertrainControls::Normal => {
                        let save_interval = veh.save_interval().unwrap_or(Option::None);
                        veh.pt_cntrl =
                            ConvPowertrainControls::StopStart(Box::new(ConvStopStartControl::new(
                                Option::None, // fc_min_time_on
                                Option::None, // temp_fc_forced_on
                                Option::None, // temp_fc_allowed_off
                                Option::None, // time_delay_after_stop_until_fc_can_turn_off
                                save_interval,
                            )?));
                    }
                    ConvPowertrainControls::StopStart(_) => (),
                }
            }
            PowertrainType::HybridElectricVehicle(veh) => match veh.pt_cntrl {
                HEVPowertrainControls::RGWDB(_) => {
                    let save_interval = veh.save_interval().unwrap_or(Option::None);
                    veh.pt_cntrl =
                        HEVPowertrainControls::StopStart(Box::new(HEVStopStartControl::new(
                            Option::None, // fc_min_time_on
                            Option::None, // soc_fc_forced_on
                            Option::None, // frac_of_most_eff_pwr_to_run_fc
                            Option::None, // temp_fc_forced_on
                            Option::None, // temp_fc_allowed_off
                            Option::None, // time_delay_after_stop_until_fc_can_turn_off
                            Option::None, // em_can_regen
                            save_interval,
                        )?));
                }
                HEVPowertrainControls::StopStart(_) => (),
            },
            _ => (),
        }
        Ok(())
    }

    #[pyo3(name = "use_normal_controller")]
    fn use_normal_controller_py(&mut self) -> anyhow::Result<()> {
        let save_interval = self.save_interval().unwrap_or(Option::None);
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(conv) => match &conv.pt_cntrl {
                ConvPowertrainControls::StopStart(_) => {
                    conv.pt_cntrl = ConvPowertrainControls::Normal;
                }
                ConvPowertrainControls::Normal => (),
            },
            PowertrainType::HybridElectricVehicle(hev) => {
                match &hev.pt_cntrl {
                    HEVPowertrainControls::StopStart(_) => {
                        hev.pt_cntrl = HEVPowertrainControls::RGWDB(Box::new(
                            RESGreedyWithDynamicBuffers::new(
                                Option::None, // speed_soc_disch_buffer
                                Option::None, // speed_soc_disch_buffer_coeff
                                Option::None, // speed_soc_fc_on_buffer
                                Option::None, // speed_soc_fc_on_buffer_coeff
                                Option::None, // speed_soc_regen_buffer
                                Option::None, // speed_soc_regen_buffer_coeff
                                Option::None, // fc_min_time_on
                                Option::None, // speed_fc_forced_on
                                Option::None, // frac_pwr_demand_fc_forced_on
                                Option::None, // frac_of_most_eff_pwr_to_run_fc
                                Option::None, // temp_fc_forced_on
                                Option::None, // temp_fc_allowed_off
                                save_interval,
                            )?,
                        ))
                    }
                    HEVPowertrainControls::RGWDB(_) => (),
                }
            }
            _ => (),
        }
        Ok(())
    }

    #[pyo3(name = "set_dfco_params")]
    fn set_dfco_params_py(
        &mut self,
        enabled: bool,
        min_dfco_speed_m_per_s: f64,
        max_accel_for_dfco_m_per_s2: f64,
    ) -> anyhow::Result<()> {
        let min_dfco_speed_m_per_s = min_dfco_speed_m_per_s.max(0.0);
        let max_accel_for_dfco_m_per_s2 = max_accel_for_dfco_m_per_s2.min(0.0);
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.dfco_cntrl.dfco_enabled = enabled;
                conv.dfco_cntrl.minimum_dfco_speed = min_dfco_speed_m_per_s * uc::MPS;
                conv.dfco_cntrl.minimum_dfco_deceleration = max_accel_for_dfco_m_per_s2 * uc::MPS2;
                conv.dfco_cntrl.save_interval = self.save_interval;
            }
            _ => (),
        }
        Ok(())
    }
}

/// implementing constructor function for Vehicle
impl Vehicle {
    /// Create new Vehicle with specified parameters
    pub fn new(
        name: String,
        doc: Option<String>,
        year: u32,
        pt_type: PowertrainType,
        chassis: Chassis,
        cabin: CabinOption,
        hvac: HVACOption,
        mass: Option<si::Mass>,
        pwr_aux_base: si::Power,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut veh = Self {
            name,
            doc,
            year,
            pt_type,
            chassis,
            cabin,
            hvac,
            mass,
            pwr_aux_base,
            state: VehicleState::default(),
            history: VehicleStateHistoryVec::default(),
            save_interval,
        };
        veh.init()?;
        Ok(veh)
    }
}

impl Mass for Vehicle {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        match (derived_mass, self.mass) {
            (Some(derived_mass), Some(set_mass)) => {
                ensure!(
                    utils::almost_eq_uom(&set_mass, &derived_mass, None),
                    format!(
                        "{}",
                        format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                    )
                );
                Ok(Some(set_mass))
            }
            (None, None) => bail!(
                "Not all mass fields in `{}` are set and no mass was previously set.",
                stringify!(Vehicle)
            ),
            _ => Ok(self.mass.or(derived_mass)),
        }
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        ensure!(
            side_effect == MassSideEffect::None,
            "At the vehicle level, only `MassSideEffect::None` is allowed"
        );
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.mass = match (new_mass, derived_mass) {
            // Set using provided `new_mass`, setting constituent mass fields to `None` to match if inconsistent
            (Some(new_mass), Some(dm)) => {
                if dm != new_mass {
                    self.expunge_mass_fields();
                }
                Some(new_mass)
            }
            (Some(new_mass), None) => Some(new_mass),
            (None, Some(dm)) => Some(dm),
            (None, None) => bail!(
                "Not all mass fields in `{}` are set and no mass was provided.",
                stringify!(Vehicle)
            ),
        };
        ensure!(
            self.mass > Some(0.0 * uc::KG),
            "{} mass must be positive",
            stringify!(Vehicle)
        );
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let chassis_mass = self
            .chassis
            .mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        let pt_mass = match &self.pt_type {
            PowertrainType::ConventionalVehicle(conv) => conv.mass()?,
            PowertrainType::HybridElectricVehicle(hev) => hev.mass()?,
            PowertrainType::PlugInHybridElectricVehicle(phev) => phev.mass()?,
            PowertrainType::BatteryElectricVehicle(bev) => bev.mass()?,
        };
        if let (Some(pt_mass), Some(chassis_mass)) = (pt_mass, chassis_mass) {
            Ok(Some(pt_mass + chassis_mass))
        } else {
            Ok(None)
        }
    }

    fn expunge_mass_fields(&mut self) {
        self.chassis.expunge_mass_fields();
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(conv) => conv.expunge_mass_fields(),
            PowertrainType::HybridElectricVehicle(hev) => hev.expunge_mass_fields(),
            PowertrainType::PlugInHybridElectricVehicle(phev) => phev.expunge_mass_fields(),
            PowertrainType::BatteryElectricVehicle(bev) => bev.expunge_mass_fields(),
        };
    }
}

impl SerdeAPI for Vehicle {
    #[cfg(feature = "resources")]
    const RESOURCES_SUBDIR: &'static str = "vehicles";
}
impl Init for Vehicle {
    fn init(&mut self) -> Result<(), Error> {
        let _mass = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.calculate_wheel_radius()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.pt_type
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        let mass = self
            .mass()
            .unwrap_or(Some(0.0 * uc::KG))
            .unwrap_or(0.0 * uc::KG);
        let _ = match &self.pt_type {
            PowertrainType::HybridElectricVehicle(hev) => hev.check_buffers(mass),
            PowertrainType::PlugInHybridElectricVehicle(hev) => hev.check_buffers(mass),
            _ => Ok(()),
        };
        Ok(())
    }
}

impl HistoryMethods for Vehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        self.pt_type.set_save_interval(save_interval)?;
        self.cabin.set_save_interval(save_interval)?;
        self.hvac.set_save_interval(save_interval)?;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear();
        self.pt_type.clear();
        self.cabin.clear();
        self.hvac.clear();
    }
}

/// TODO: update this constant to match fastsim-2 for gasoline
pub(super) const FUEL_LHV_MJ_PER_KG: f64 = 43.2;
const CONV: &str = "Conv";
const HEV: &str = "HEV";
const PHEV: &str = "PHEV";
const BEV: &str = "BEV";

impl SetCumulative for Vehicle {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        self.state
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.pt_type
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.cabin
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        self.hvac
            .set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?;
        // this does not get handled by the `SetCumulative` derive macro
        self.state.dist.increment(
            *self.state.speed_ach.get_fresh(|| format_dbg!())? * dt,
            || format_dbg!(),
        )?;
        Ok(())
    }

    fn reset_cumulative<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.state
            .reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.pt_type
            .reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.cabin
            .reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?;
        self.hvac
            .reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?;
        // this does not get handled by the `SetCumulative` derive macro
        self.state.dist.mark_stale();
        self.state.dist.update(si::Length::ZERO, || format_dbg!())?;
        self.state.time.mark_stale();
        self.state.time.update(si::Time::ZERO, || format_dbg!())?;
        self.state.speed_ach.mark_stale();
        self.state
            .speed_ach
            .update(si::Velocity::ZERO, || format_dbg!())?;
        Ok(())
    }
}

impl Vehicle {
    /// # Assumptions
    /// - peak power of all components can be produced concurrently.
    pub fn get_pwr_rated(&self) -> si::Power {
        match (self.fc(), self.res()) {
            (Some(fc), Some(res)) => fc.pwr_out_max + res.pwr_out_max,
            (Some(fc), None) => fc.pwr_out_max,
            (None, Some(res)) => res.pwr_out_max,
            (None, None) => unreachable!(),
        }
    }

    pub fn conv(&self) -> Option<&ConventionalVehicle> {
        self.pt_type.conv()
    }

    pub fn hev(&self) -> Option<&HybridElectricVehicle> {
        self.pt_type.hev()
    }

    // pub fn phev(&self) -> Option<&HybridElectricVehicle> {
    //     self.pt_type.phev()
    // }

    pub fn bev(&self) -> Option<&BatteryElectricVehicle> {
        self.pt_type.bev()
    }

    pub fn conv_mut(&mut self) -> Option<&mut ConventionalVehicle> {
        self.pt_type.conv_mut()
    }

    pub fn hev_mut(&mut self) -> Option<&mut HybridElectricVehicle> {
        self.pt_type.hev_mut()
    }

    // pub fn phev_mut(&mut self) -> Option<&mut HybridElectricVehicle> {
    //     self.pt_type.phev_mut()
    // }

    pub fn bev_mut(&mut self) -> Option<&mut BatteryElectricVehicle> {
        self.pt_type.bev_mut()
    }

    pub fn fc(&self) -> Option<&FuelConverter> {
        self.pt_type.fc()
    }

    pub fn fc_mut(&mut self) -> Option<&mut FuelConverter> {
        self.pt_type.fc_mut()
    }

    pub fn set_fc(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        self.pt_type.set_fc(fc)
    }

    pub fn fs(&self) -> Option<&FuelStorage> {
        self.pt_type.fs()
    }

    pub fn fs_mut(&mut self) -> Option<&mut FuelStorage> {
        self.pt_type.fs_mut()
    }

    pub fn set_fs(&mut self, fs: FuelStorage) -> anyhow::Result<()> {
        self.pt_type.set_fs(fs)
    }

    pub fn res(&self) -> Option<&ReversibleEnergyStorage> {
        self.pt_type.res()
    }

    pub fn res_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        self.pt_type.res_mut()
    }

    pub fn set_res(&mut self, res: ReversibleEnergyStorage) -> anyhow::Result<()> {
        self.pt_type.set_res(res)
    }

    pub fn em(&self) -> Option<&ElectricMachine> {
        self.pt_type.em()
    }

    pub fn em_mut(&mut self) -> Option<&mut ElectricMachine> {
        self.pt_type.em_mut()
    }

    pub fn set_em(&mut self, em: ElectricMachine) -> anyhow::Result<()> {
        self.pt_type.set_em(em)
    }

    pub fn trans(&self) -> Option<&Transmission> {
        self.pt_type.trans()
    }

    pub fn trans_mut(&mut self) -> Option<&mut Transmission> {
        self.pt_type.trans_mut()
    }

    pub fn set_trans(&mut self, trans: Transmission) -> anyhow::Result<()> {
        self.pt_type.set_trans(trans)
    }

    /// Calculate wheel radius from tire code, if applicable
    fn calculate_wheel_radius(&mut self) -> anyhow::Result<()> {
        ensure!(
            self.chassis.wheel_radius.is_some() || self.chassis.tire_code.is_some(),
            "Either `wheel_radius` or `tire_code` must be supplied"
        );
        if self.chassis.wheel_radius.is_none() {
            self.chassis.wheel_radius =
                Some(utils::tire_code_to_radius(self.chassis.tire_code.as_ref().unwrap())? * uc::M)
        }
        Ok(())
    }

    /// Solves for energy consumption
    pub fn solve_powertrain(&mut self, dt: si::Time) -> anyhow::Result<()> {
        self.pt_type
            .solve(
                *self.state.pwr_tractive.get_fresh(|| format_dbg!())?,
                true, // `enabled` should always be true at the powertrain level
                dt,
            )
            .map_err(|err| {
                anyhow::anyhow!(
                    "solve() failed at line {} with originating error [{}]",
                    format_dbg!(),
                    err
                )
            })?;
        self.state.pwr_brake.update(
            -self
                .state
                .pwr_tractive
                .get_fresh(|| format_dbg!())?
                .max(si::Power::ZERO)
                - self.pt_type.pwr_regen().with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    pub fn set_curr_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // Calculate traction limits
        let mass = self
            .mass
            .with_context(|| format!("{}\nMass should have been set before now", format_dbg!()))?;
        let max_trac_accel = self.chassis.wheel_fric_coef
            * self.chassis.drive_axle_weight_frac
            * uc::ACC_GRAV
            / (1.0 * uc::R
                + self.chassis.cg_height * self.chassis.wheel_fric_coef / self.chassis.wheel_base);
        let prev_speed = *self.state.speed_ach.get_stale(|| format_dbg!())?;
        let max_trac_speed = prev_speed + (max_trac_accel * dt);
        let max_trac_power = self.chassis.wheel_fric_coef
            * self.chassis.drive_axle_weight_frac
            * mass
            * uc::ACC_GRAV
            / (1.0 * uc::R
                + self.chassis.cg_height * self.chassis.wheel_fric_coef / self.chassis.wheel_base)
            * max_trac_speed;
        // Calculate powertrain limits
        self.pt_type
            .set_curr_pwr_prop_out_max(
                (si::Power::ZERO, si::Power::ZERO),
                *self.state.pwr_aux.get_fresh(|| format_dbg!())?,
                dt,
                &self.state,
            )
            .with_context(|| anyhow!(format_dbg!()))?;
        let pwr_prop_maxes = self
            .pt_type
            .get_curr_pwr_prop_out_max()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.state.pwr_prop_fwd_max.update(
            if pwr_prop_maxes.0 > max_trac_power {
                max_trac_power
            } else {
                pwr_prop_maxes.0
            },
            || format_dbg!(),
        )?;
        self.state
            .pwr_prop_bwd_max
            .update(pwr_prop_maxes.1, || format_dbg!())?;

        Ok(())
    }

    pub fn solve_thermal(
        &mut self,
        te_amb_air: si::Temperature,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let te_fc: Option<si::Temperature> = self
            .fc()
            .and_then(|fc| fc.temperature().map(|fct| fct.get_stale(|| format_dbg!())))
            .transpose()
            .with_context(|| {
                format!(
                    "{}\nfuel converter temperature has not been properly set",
                    format_dbg!()
                )
            })?
            .copied();
        let pwr_thrml_cab_to_res: si::Power = match self.res() {
            Some(res) => match &res.thrml {
                RESThermalOption::RESLumpedThermal(rlt) => {
                    *rlt.state.pwr_thrml_from_cabin.get_stale(|| format_dbg!())?
                }
                RESThermalOption::None => si::Power::ZERO,
            },
            None => si::Power::ZERO,
        };

        let (pwr_thrml_fc_to_cabin, pwr_thrml_hvac_to_res, te_cab) = self
            .solve_hvac_cab_res(te_amb_air, dt, te_fc, pwr_thrml_cab_to_res)
            .with_context(|| format_dbg!())?;

        self.pt_type
            .solve_thermal(
                te_amb_air,
                pwr_thrml_fc_to_cabin,
                &mut self.state,
                pwr_thrml_hvac_to_res,
                te_cab,
                dt,
            )
            .with_context(|| format_dbg!())?;
        Ok(())
    }

    fn solve_hvac_cab_res(
        &mut self,
        te_amb_air: si::Temperature,
        dt: si::Time,
        te_fc: Option<si::Temperature>,
        pwr_thrml_cab_to_res: si::Power,
    ) -> anyhow::Result<(
        Option<si::Power>,
        Option<si::Power>,
        Option<si::Temperature>,
    )> {
        let res_thrml_state = self.pt_type.res_mut().and_then(|rm| rm.res_thrml_state());
        let (pwr_thrml_fc_to_cabin, pwr_thrml_hvac_to_res, te_cab): (
            Option<si::Power>,
            Option<si::Power>,
            Option<si::Temperature>,
        ) = match (&mut self.cabin, &mut self.hvac, res_thrml_state) {
            (CabinOption::None, HVACOption::None, None) => {
                self.state
                    .pwr_aux
                    .update(self.pwr_aux_base, || format_dbg!())?;
                (None, None, None)
            }
            (CabinOption::LumpedCabin(cab), HVACOption::LumpedCabin(hvac), None) => {
                let (pwr_thrml_hvac_to_cabin, pwr_thrml_fc_to_cab) = hvac
                    .solve(te_amb_air, te_fc, &cab.state, cab.heat_capacitance, dt)
                    .with_context(|| format_dbg!())?;
                let te_cab = cab
                    .solve(
                        te_amb_air,
                        &self.state,
                        pwr_thrml_hvac_to_cabin,
                        Default::default(),
                        dt,
                    )
                    .with_context(|| format_dbg!())?;
                self.state.pwr_aux.update(
                    self.pwr_aux_base
                        + *hvac
                            .state
                            .pwr_aux_for_hvac
                            .get_fresh(|| format_dbg!("hvac.state.pwr_aux_for_hvac"))?,
                    || format_dbg!(),
                )?;
                (Some(pwr_thrml_fc_to_cab), None, Some(te_cab))
            }
            (
                CabinOption::LumpedCabin(cab),
                HVACOption::LumpedCabinAndRES(hvac),
                Some(res_thrml_state),
            ) => {
                let (pwr_thrml_hvac_to_cabin, pwr_thrml_fc_to_cab, pwr_thrml_hvac_to_res) = hvac
                    .solve(
                        te_amb_air,
                        te_fc,
                        &cab.state,
                        cab.heat_capacitance,
                        res_thrml_state,
                        dt,
                    )
                    .with_context(|| format_dbg!())?;
                let te_cab = cab
                    .solve(
                        te_amb_air,
                        &self.state,
                        pwr_thrml_hvac_to_cabin,
                        pwr_thrml_cab_to_res,
                        dt,
                    )
                    .with_context(|| format_dbg!())?;
                self.state.pwr_aux.update(
                    self.pwr_aux_base
                        + *hvac
                            .state
                            .pwr_aux_for_cab_hvac
                            .get_fresh(|| format_dbg!())?
                        + *hvac
                            .state
                            .pwr_aux_for_res_hvac
                            .get_fresh(|| format_dbg!())?,
                    || format_dbg!(),
                )?;
                ensure!(
                    *self.state.pwr_aux.get_fresh(|| format_dbg!())? > si::Power::ZERO,
                    format!(
                        "{}\n{}\n{}",
                        format_dbg!(self.state.pwr_aux),
                        format_dbg!(hvac.state.pwr_aux_for_res_hvac),
                        format_dbg!(hvac.state.pwr_aux_for_cab_hvac)
                    )
                );
                (
                    Some(pwr_thrml_fc_to_cab),
                    Some(pwr_thrml_hvac_to_res),
                    Some(te_cab),
                )
            }
            (CabinOption::LumpedCabin(cab), HVACOption::LumpedCabin(hvac), Some(_)) => {
                let (pwr_thrml_hvac_to_cabin, pwr_thrml_fc_to_cab) = hvac
                    .solve(te_amb_air, te_fc, &cab.state, cab.heat_capacitance, dt)
                    .with_context(|| format_dbg!())?;
                let te_cab = cab
                    .solve(
                        te_amb_air,
                        &self.state,
                        pwr_thrml_hvac_to_cabin,
                        Default::default(),
                        dt,
                    )
                    .with_context(|| format_dbg!())?;
                self.state.pwr_aux.update(
                    self.pwr_aux_base
                        + *hvac
                            .state
                            .pwr_aux_for_hvac
                            .get_fresh(|| format_dbg!("hvac.state.pwr_aux_for_hvac"))?,
                    || format_dbg!(),
                )?;
                (Some(pwr_thrml_fc_to_cab), None, Some(te_cab))
            }
            (CabinOption::LumpedCabin(cab), HVACOption::None, Some(_)) => {
                let te_cab = cab
                    .solve(
                        te_amb_air,
                        &self.state,
                        si::Power::ZERO,
                        si::Power::ZERO,
                        dt,
                    )
                    .with_context(|| format_dbg!())?;
                self.state
                    .pwr_aux
                    .update(self.pwr_aux_base, || format_dbg!())?;
                (None, None, Some(te_cab))
            }
            (_, _, _) => {
                bail!(
                    "{}\nCabin, HVAC, and RESThermal configuration is either invalid or not yet implemented.\n{} - {} - {}",
                    format_dbg!(),
                    format!("{}", self.hvac),
                    format!("{}", self.cabin),
                    format!(
                        "`res.res_thrml_state().is_some()`: {}",
                        self.pt_type.res().and_then(|res| res.res_thrml_state()).is_some()
                    ),
                );
            }
        };
        Ok((pwr_thrml_fc_to_cabin, pwr_thrml_hvac_to_res, te_cab))
    }

    #[allow(dead_code)]
    fn from_f2_file(file: PathBuf) -> anyhow::Result<Self> {
        use fastsim_2::traits::SerdeAPI;
        let f2veh = fastsim_2::vehicle::RustVehicle::from_file(file, false)
            .with_context(|| format_dbg!())?;
        Self::try_from(f2veh)
    }

    pub(crate) fn mark_non_thermal_fresh(&mut self) -> Result<(), anyhow::Error> {
        self.state.i.mark_stale();
        self.state.time.mark_stale();
        self.state.pwr_aux.mark_stale();
        self.state.mass.mark_stale();
        self.state.mark_fresh(|| format_dbg!())?;
        self.state.energy_tractive.mark_stale();
        self.state.energy_aux.mark_stale();
        self.state.energy_drag.mark_stale();
        self.state.energy_accel.mark_stale();
        self.state.energy_ascent.mark_stale();
        self.state.energy_rr.mark_stale();
        self.state.energy_whl_inertia.mark_stale();
        self.state.energy_brake.mark_stale();
        self.state.dist.mark_stale();
        if let Some(fc) = self.fc_mut() {
            fc.state.i.mark_stale();
            fc.state.mark_fresh(|| format_dbg!())?;
            fc.state.energy_prop.mark_stale();
            fc.state.energy_aux.mark_stale();
            fc.state.energy_fuel.mark_stale();
            fc.state.energy_loss.mark_stale();
        }
        if let Some(res) = self.res_mut() {
            res.state.i.mark_stale();
            res.state.soh.mark_stale();
            res.state.mark_fresh(|| format_dbg!())?;
            res.state.energy_out_electrical.mark_stale();
            res.state.energy_out_prop.mark_stale();
            res.state.energy_aux.mark_stale();
            res.state.energy_loss.mark_stale();
            res.state.energy_out_chemical.mark_stale();
        }

        if let Some(em) = self.em_mut() {
            em.state.i.mark_stale();
            em.state.mark_fresh(|| format_dbg!())?;
            em.state.energy_out_req.mark_stale();
            em.state.energy_elec_prop_in.mark_stale();
            em.state.energy_mech_prop_out.mark_stale();
            em.state.energy_mech_dyn_brake.mark_stale();
            em.state.energy_elec_dyn_brake.mark_stale();
            em.state.energy_loss.mark_stale();
        }
        if let Some(trans) = self.trans_mut() {
            trans.state.i.mark_stale();
            trans.state.mark_fresh(|| format_dbg!())?;
            trans.state.energy_out.mark_stale();
            trans.state.energy_in.mark_stale();
            trans.state.energy_loss.mark_stale();
        }
        if let PowertrainType::HybridElectricVehicle(hev) = &mut self.pt_type {
            match &mut hev.pt_cntrl {
                HEVPowertrainControls::RGWDB(rgwdb) => rgwdb.state.i.mark_stale(),
                HEVPowertrainControls::StopStart(ctrl) => ctrl.state.i.mark_stale(),
            }
            hev.pt_cntrl.mark_fresh(|| format_dbg!())?
        }
        Ok(())
    }
}

/// Vehicle state for current time step
#[serde_api]
#[derive(
    Clone, Debug, Deserialize, Serialize, PartialEq, HistoryVec, StateMethods, SetCumulative,
)]
#[non_exhaustive]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct VehicleState {
    /// time step index
    pub i: TrackedState<usize>,

    /// elapsed simulation time since start
    pub time: TrackedState<si::Time>,

    // power and energy fields
    /// maximum forward propulsive power vehicle can produce
    pub pwr_prop_fwd_max: TrackedState<si::Power>,
    /// pwr exerted on wheels by powertrain
    /// maximum backward propulsive power (e.g. regenerative braking) vehicle can produce
    pub pwr_prop_bwd_max: TrackedState<si::Power>,
    /// Tractive power for achieved speed
    pub pwr_tractive: TrackedState<si::Power>,
    /// Tractive power required for prescribed speed
    pub pwr_tractive_for_cyc: TrackedState<si::Power>,
    /// integral of [Self::pwr_tractive]
    pub energy_tractive: TrackedState<si::Energy>,
    /// time varying aux load
    pub pwr_aux: TrackedState<si::Power>,
    /// integral of [Self::pwr_aux]
    pub energy_aux: TrackedState<si::Energy>,
    /// Power applied to aero drag
    pub pwr_drag: TrackedState<si::Power>,
    /// integral of [Self::pwr_drag]
    pub energy_drag: TrackedState<si::Energy>,
    /// Power applied to acceleration (includes deceleration)
    pub pwr_accel: TrackedState<si::Power>,
    /// integral of [Self::pwr_accel]
    pub energy_accel: TrackedState<si::Energy>,
    /// Power applied to grade ascent
    pub pwr_ascent: TrackedState<si::Power>,
    /// integral of [Self::pwr_ascent]
    pub energy_ascent: TrackedState<si::Energy>,
    /// Power applied to rolling resistance
    pub pwr_rr: TrackedState<si::Power>,
    /// integral of [Self::pwr_rr]
    pub energy_rr: TrackedState<si::Energy>,
    /// Power applied to wheel and tire inertia
    pub pwr_whl_inertia: TrackedState<si::Power>,
    /// integral of [Self::pwr_whl_inertia]
    pub energy_whl_inertia: TrackedState<si::Energy>,
    /// Total braking power including regen
    pub pwr_brake: TrackedState<si::Power>,
    /// integral of [Self::pwr_brake]
    pub energy_brake: TrackedState<si::Energy>,
    /// whether powertrain can achieve power demand to achieve prescribed speed
    /// in current time step
    // because it should be assumed true in the first time step
    pub cyc_met: TrackedState<bool>,
    /// whether powertrain can achieve power demand to achieve prescribed speed
    /// in entire cycle
    pub cyc_met_overall: TrackedState<bool>,
    /// actual achieved speed
    pub speed_ach: TrackedState<si::Velocity>,
    /// cumulative distance traveled, integral of [Self::speed_ach]
    pub dist: TrackedState<si::Length>,
    /// current grade
    pub grade_curr: TrackedState<si::Ratio>,
    /// current grade
    // will be overridden during simulation anyway
    pub elev_curr: TrackedState<si::Length>,
    /// current air density
    pub air_density: TrackedState<si::MassDensity>,
    /// current mass
    // TODO: make sure this gets updated appropriately
    pub mass: TrackedState<si::Mass>,
}

impl SerdeAPI for VehicleState {}
impl Init for VehicleState {}
impl Default for VehicleState {
    fn default() -> Self {
        Self {
            i: TrackedState::new(Default::default()),
            time: Default::default(),
            pwr_prop_fwd_max: Default::default(),
            pwr_prop_bwd_max: Default::default(),
            pwr_tractive: Default::default(),
            pwr_tractive_for_cyc: Default::default(),
            energy_tractive: Default::default(),
            pwr_aux: Default::default(),
            energy_aux: Default::default(),
            pwr_drag: Default::default(),
            energy_drag: Default::default(),
            pwr_accel: Default::default(),
            energy_accel: Default::default(),
            pwr_ascent: Default::default(),
            energy_ascent: Default::default(),
            pwr_rr: Default::default(),
            energy_rr: Default::default(),
            pwr_whl_inertia: Default::default(),
            energy_whl_inertia: Default::default(),
            pwr_brake: Default::default(),
            energy_brake: Default::default(),
            cyc_met: TrackedState::new(true),
            cyc_met_overall: TrackedState::new(true),
            speed_ach: Default::default(),
            dist: Default::default(),
            // note that this value will be overwritten
            grade_curr: Default::default(),
            // note that this value will be overwritten
            elev_curr: Default::default(),
            air_density: Default::default(),
            mass: TrackedState::new(uc::KG * f64::NAN),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::vehicle::conv::{ConvPowertrainControls, ConvStopStartControl};
    use crate::vehicle::hev::{HEVAuxControls, HEVSimulationParams, HEVStopStartControl};
    use crate::vehicle::powertrain::reversible_energy_storage::EffInterp;

    use super::*;

    #[allow(dead_code)]
    fn vehicles_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/vehicles")
    }

    #[cfg(feature = "yaml")]
    /// Load representative conv from fastsim-2, convert to fastsim-3 format, and
    /// save to file in the resources folder
    pub(crate) fn mock_conv_veh() -> Vehicle {
        let file_contents = include_str!("fastsim-2_2012_Ford_Fusion.yaml");
        use fastsim_2::traits::SerdeAPI;
        let veh = {
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
            let veh = Vehicle::try_from(f2veh);
            veh.unwrap()
        };

        veh.to_file(vehicles_dir().join("2012_Ford_Fusion.yaml"))
            .unwrap();
        assert!(veh.pt_type.is_conventional_vehicle());
        veh
    }

    #[cfg(feature = "yaml")]
    /// Load representative HEV from fastsim-2, convert to fastsim-3 format, and
    /// save to file in the resources folder
    pub(crate) fn mock_hev() -> Vehicle {
        let file_contents = include_str!("fastsim-2_2016_TOYOTA_Prius_Two.yaml");
        use fastsim_2::traits::SerdeAPI;
        let veh = {
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
            let veh = Vehicle::try_from(f2veh);
            veh.unwrap()
        };

        veh.to_file(vehicles_dir().join("2016_TOYOTA_Prius_Two.yaml"))
            .unwrap();
        assert!(veh.pt_type.is_hybrid_electric_vehicle());
        veh
    }

    #[cfg(feature = "yaml")]
    /// Load representative BEV from fastsim-2, convert to fastsim-3 format, and
    /// save to file in the resources folder
    pub(crate) fn mock_bev() -> Vehicle {
        let file_contents = include_str!("fastsim-2_2022_Renault_Zoe_ZE50_R135.yaml");
        use fastsim_2::traits::SerdeAPI;
        let veh = {
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
            let veh = Vehicle::try_from(f2veh);
            veh.unwrap()
        };

        veh.to_file(vehicles_dir().join("2022_Renault_Zoe_ZE50_R135.yaml"))
            .unwrap();
        assert!(veh.pt_type.is_battery_electric_vehicle());
        veh
    }

    #[test]
    #[cfg(feature = "yaml")]
    pub(crate) fn test_conv_veh_init() {
        use pretty_assertions::assert_eq;
        let veh = mock_conv_veh();
        let mut veh1 = veh.clone();
        // NOTE: eventually figure out why the following assertions fail if
        // `.to_yaml().uwrap()` is removed.  It's probably related to f64::NAN
        assert_eq!(veh.to_yaml().unwrap(), veh1.to_yaml().unwrap());
        veh1.init().unwrap();
        assert_eq!(veh.to_yaml().unwrap(), veh1.to_yaml().unwrap());
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_conv() {
        let veh = mock_conv_veh();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive::new(veh, cyc, Default::default());
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_hev() {
        let veh = mock_hev();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive::new(veh, cyc, Default::default());
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_bev() {
        let veh = mock_bev();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive::new(veh, cyc, Default::default());
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }

    type StructWithResources = Vehicle;

    #[test]
    fn test_resources() {
        let mut time_to_panic = false;

        let resource_list = StructWithResources::list_resources().unwrap();
        assert!(!resource_list.is_empty());

        // verify that resources can all load
        for resource in resource_list {
            if let Err(e) = StructWithResources::from_resource(resource.clone(), false) {
                time_to_panic = true;
                eprintln!("Error loading {resource:?}: {e}\n");
            }
        }
        if time_to_panic {
            panic!()
        }
    }

    #[test]
    fn test_calibrated_vehicles() {
        let mut time_to_panic = false;

        // check that calibrated vehicles can load
        let paths: Vec<_> = std::fs::read_dir("../cal_and_val/f3-vehicles")
            .unwrap()
            .collect();
        assert!(!paths.is_empty());
        for path in paths {
            let p = path.unwrap().path();
            if let Err(e) = StructWithResources::from_file(p.clone(), false) {
                time_to_panic = true;
                eprintln!("Error loading {p:?}: {e}\n");
            }
        }

        // check that calibrated thermal-equipped vehicles can load
        let paths: Vec<_> = std::fs::read_dir("../cal_and_val/thermal/f3-vehicles")
            .unwrap()
            .collect();
        assert!(!paths.is_empty());
        for path in paths {
            let p = path.unwrap().path();
            if let Err(e) = StructWithResources::from_file(p.clone(), false) {
                time_to_panic = true;
                eprintln!("Error loading {p:?}: {e}\n");
            }
        }

        if time_to_panic {
            panic!()
        }
    }

    fn make_conv_pacifica(with_conv_stop_start: bool, with_dfco: bool) -> anyhow::Result<Vehicle> {
        let fs = FuelStorage::new(
            2000000.0 * uc::W,
            1.1 * uc::S,
            2305080000.0 * uc::J,
            Option::None,
            Option::None,
        )?;
        let fc = FuelConverter::new(
            FuelConverterThermalOption::None, // thrml
            Option::None,                     // mass
            Option::None,                     // specific_pwr
            211088.0 * uc::W,                 // pwr_out_max
            34604.59016393443 * uc::W,        // pwr_out_max_init
            6.1 * uc::S,                      // pwr_ramp_lag
            InterpolatorEnum::new_1d(
                vec![
                    0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
                ]
                .into(),
                vec![
                    0.0,
                    0.0875106035,
                    0.143482108,
                    0.216273855,
                    0.252599848,
                    0.301508117,
                    0.33,
                    0.34,
                    0.35,
                    0.34,
                    0.32,
                    0.3,
                ]
                .into(),
                strategy::Linear,
                Extrapolate::Error,
            )?, // eff_interp_from_pwr_out
            0.4 * 211088.0 * uc::W,           // pwr_for_peak_eff
            0.0 * uc::W,                      // pwr_idle_fuel
            Option::None,
        )?;
        let tx = Transmission::new(
            Option::None,                   // mass
            InterpolatorEnum::new_0d(0.95), // eff_interp
            Option::None,                   // save_interval
        )?;
        let pt_controls = {
            if with_conv_stop_start {
                let ctrl = ConvStopStartControl::new(
                    Option::None, // fc_min_time_on
                    Option::None, // temp_fc_forced_on
                    Option::None, // temp_fc_allowed_off
                    Option::None, // time_delay_after_stop_until_fc_can_turn_off
                    Option::None, // save_interval
                )
                .map_err(|err| {
                    assert!(
                        false,
                        "Unable to create stop/start control for conv: {}",
                        err
                    );
                });
                let ctrl = ctrl.ok().unwrap();
                ConvPowertrainControls::StopStart(Box::new(ctrl))
            } else {
                ConvPowertrainControls::Normal
            }
        };
        let dfco_controls = conv::DfcoControls::new(
            with_dfco,       // dfco_enabled
            25.0 * uc::MPH,  // minimum_dfco_speed
            -0.2 * uc::MPS2, // minimum_dfco_deceleration
            Option::None,    // save_interval
        )?;
        let conv = ConventionalVehicle::new(
            fs,            // fs
            fc,            // fc
            tx,            // transmission
            Option::None,  // mass
            pt_controls,   // powertrain control
            dfco_controls, // dfco_cntrl
            1.0 * uc::R,   // alt_eff
        )?;
        let chassis = Chassis {
            drag_coef: 0.3303036837542712 * uc::R,
            frontal_area: 3.05124164 * uc::M2,
            wheel_rr_coef: 0.0064798953284486704 * uc::R,
            wheel_inertia: 0.815 * uc::KGM2,
            num_wheels: 4,
            wheel_radius: Option::Some(0.36865 * uc::M),
            tire_code: Option::None,
            cg_height: 0.53 * uc::M,
            wheel_fric_coef: 0.8 * uc::R,
            drive_type: chassis::DriveTypes::FWD,
            drive_axle_weight_frac: 0.61 * uc::R,
            wheel_base: 3.08864 * uc::M,
            mass: Option::None,
            glider_mass: Option::None,
            cargo_mass: Option::None,
        };
        let boxed_conv = Box::new(conv);
        let mut veh = Vehicle::new(
            String::from("2026 Chrysler Pacifica Select"),   // name
            Option::None,                                    // doc
            2026,                                            // year
            PowertrainType::ConventionalVehicle(boxed_conv), // pt_type
            chassis,                                         // chassis
            CabinOption::None,                               // cabin
            HVACOption::None,                                // hvac
            Option::Some(2154.564 * uc::KG),                 // mass
            700.0 * uc::W,                                   // pwr_aux_base
            Option::None,                                    // save_interval
        )?;
        veh.set_save_interval(Option::Some(1))?;
        Ok(veh)
    }

    fn make_microhybrid_pacifica() -> anyhow::Result<Vehicle> {
        let res = ReversibleEnergyStorage::new(
            RESThermalOption::None,             // thrml
            Option::None,                       // mass
            Option::None,                       // specific_energy
            5.0 * uc::KW,                       // pwr_out_max
            1.0 * uc::KWH,                      // energy_capacity
            EffInterp::Constant(Interp0D(0.9)), // eff_interp
            0.0 * uc::R,                        // min_soc
            1.0 * uc::R,                        // max_soc
            Option::None,
        )?;
        let fs = FuelStorage::new(
            2000000.0 * uc::W,
            1.1 * uc::S,
            2305080000.0 * uc::J,
            Option::None,
            Option::None,
        )?;
        let fc = FuelConverter::new(
            FuelConverterThermalOption::None, // thrml
            Option::None,                     // mass
            Option::None,                     // specific_pwr
            211088.0 * uc::W,                 // pwr_out_max
            34604.59016393443 * uc::W,        // pwr_out_max_init
            6.1 * uc::S,                      // pwr_ramp_lag
            InterpolatorEnum::new_1d(
                vec![
                    0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
                ]
                .into(),
                vec![
                    0.0,
                    0.0875106035,
                    0.143482108,
                    0.216273855,
                    0.252599848,
                    0.301508117,
                    0.33,
                    0.34,
                    0.35,
                    0.34,
                    0.32,
                    0.3,
                ]
                .into(),
                strategy::Linear,
                Extrapolate::Error,
            )?, // eff_interp_from_pwr_out
            0.4 * 211088.0 * uc::W,           // pwr_for_peak_eff
            0.0 * uc::W,                      // pwr_idle_fuel
            Option::None,
        )?;
        let em = ElectricMachine::new(
            InterpolatorEnum::new_1d(
                vec![0.0, 1.0].into(),
                vec![0.95, 0.95].into(),
                strategy::Linear,
                Extrapolate::Error,
            )?, // eff_interp_achieved
            Option::None, // eff_interp_at_max_input
            5.0 * uc::KW, // pwr_out_max
            Option::None, // specific_pwr
            Option::None, // mass
            Option::None, // save_interval
        )?;
        let tx = Transmission::new(
            Option::None,                   // mass
            InterpolatorEnum::new_0d(0.95), // eff_interp
            Option::None,                   // save_interval
        )?;
        let ctrl = HEVStopStartControl::new(
            Option::None, // fc_min_time_on
            Option::None, // soc_fc_forced_on
            Option::None, // frac_of_most_eff_pwr_to_run_fc
            Option::None, // temp_fc_forced_on
            Option::None, // temp_fc_allowed_off
            Option::None, // time_delay_after_stop_until_fc_can_turn_off
            Option::None, // em_can_regen
            Option::None, // save_interval
        )?;
        let pt_ctrl = HEVPowertrainControls::StopStart(Box::new(ctrl));
        let aux_ctrl = HEVAuxControls::AuxOnResPriority;
        let sim_params = HEVSimulationParams::new(
            0.05 * uc::R, // res_per_fuel_lim
            5,            // soc_balance_iter_err
            false,        // balance_soc
            false,        // save_soc_bal_iters
        )?;
        let hev = HybridElectricVehicle::new(
            res,          // res
            fs,           // fs
            fc,           // fc
            em,           // em
            tx,           // transmission
            pt_ctrl,      // pt_cntrl
            aux_ctrl,     // aux_cntrl
            Option::None, // mass
            sim_params,   // sim_params
        )?;
        let chassis = Chassis {
            drag_coef: 0.3303036837542712 * uc::R,
            frontal_area: 3.05124164 * uc::M2,
            wheel_rr_coef: 0.0064798953284486704 * uc::R,
            wheel_inertia: 0.815 * uc::KGM2,
            num_wheels: 4,
            wheel_radius: Option::Some(0.36865 * uc::M),
            tire_code: Option::None,
            cg_height: 0.53 * uc::M,
            wheel_fric_coef: 0.8 * uc::R,
            drive_type: chassis::DriveTypes::FWD,
            drive_axle_weight_frac: 0.61 * uc::R,
            wheel_base: 3.08864 * uc::M,
            mass: Option::None,
            glider_mass: Option::None,
            cargo_mass: Option::None,
        };
        let boxed_hev = Box::new(hev);
        let mut veh = Vehicle::new(
            String::from("2026 Chrysler Pacifica Select (uHEV Test)"), // name
            Option::None,                                              // doc
            2026,                                                      // year
            PowertrainType::HybridElectricVehicle(boxed_hev),          // pt_type
            chassis,                                                   // chassis
            CabinOption::None,                                         // cabin
            HVACOption::None,                                          // hvac
            Option::Some(2154.564 * uc::KG),                           // mass
            700.0 * uc::W,                                             // pwr_aux_base
            Option::None,                                              // save_interval
        )?;
        veh.set_save_interval(Option::Some(1))?;
        Ok(veh)
    }

    #[test]
    fn we_can_create_and_simulate_a_micro_hybrid_vehicle() {
        let veh_result = make_microhybrid_pacifica();
        assert!(veh_result.is_ok());
        let veh = veh_result.unwrap();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = crate::simdrive::SimDrive::new(veh, cyc, Default::default());
        let walk_result = sd.walk();
        if let Err(err) = walk_result {
            panic!("Error: {}", err);
        }
        assert!(walk_result.is_ok());
    }

    fn accumulate_for_zero_speed(speeds_mps: &[f64], fuels_mj: &[f64]) -> f64 {
        let shortest_idx = speeds_mps.len().min(fuels_mj.len());
        let mut result_mj = 0.0;
        for idx in 0..shortest_idx {
            if speeds_mps[idx] == 0.0 {
                result_mj += fuels_mj[idx];
            }
        }
        result_mj
    }

    #[test]
    fn micro_hybrid_saves_more_fuel_than_conventional() {
        let veh_uhev_result = make_microhybrid_pacifica();
        assert!(veh_uhev_result.is_ok());
        let veh_uhev = veh_uhev_result.unwrap();
        let veh_conv_result = make_conv_pacifica(false, false);
        assert!(veh_conv_result.is_ok());
        let veh_conv = veh_conv_result.unwrap();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd_uhev = crate::simdrive::SimDrive::new(veh_uhev, cyc.clone(), Default::default());
        let result_uhev = sd_uhev.walk();
        if let Err(err) = result_uhev {
            panic!("Error: {}", err);
        }
        assert!(result_uhev.is_ok());
        let mut sd_conv = crate::simdrive::SimDrive::new(veh_conv, cyc.clone(), Default::default());
        let result_conv = sd_conv.walk();
        assert!(result_conv.is_ok());
        let speeds_mps: Vec<f64> = cyc
            .speed
            .iter()
            .map(|spd| spd.get::<si::meter_per_second>())
            .collect();
        let fc_uhev = sd_uhev.veh.pt_type.fc().unwrap();
        let fc_conv = sd_conv.veh.pt_type.fc().unwrap();
        let fuels_uhev_mj: Vec<f64> = fc_uhev
            .history
            .energy_fuel
            .iter()
            .map(|ef| ef.get_fresh(|| format_dbg!()).unwrap().get::<si::joule>() / 1e6)
            .collect();
        let fuel_uhev_mj: f64 = fuels_uhev_mj.iter().sum();
        let fuels_conv_mj: Vec<f64> = fc_conv
            .history
            .energy_fuel
            .iter()
            .map(|ef| ef.get_fresh(|| format_dbg!()).unwrap().get::<si::joule>() / 1e6)
            .collect();
        let fuel_conv_mj: f64 = fuels_conv_mj.iter().sum();
        assert_eq!(speeds_mps.len(), fuels_uhev_mj.len());
        assert_eq!(speeds_mps.len(), fuels_conv_mj.len());
        eprintln!("fuel_uhev: {} MJ", fuel_uhev_mj);
        eprintln!("fuel_conv: {} MJ", fuel_conv_mj);
        assert!(
            fuel_uhev_mj < fuel_conv_mj,
            "Expected uHEV fuel ({fuel_uhev_mj}) to be less than conventional ({fuel_conv_mj})"
        );
        let fuel_stopped_uhev_mj = accumulate_for_zero_speed(&speeds_mps, &fuels_uhev_mj);
        let fuel_stopped_conv_mj = accumulate_for_zero_speed(&speeds_mps, &fuels_conv_mj);
        eprintln!("fuel_stopped_uhev: {} MJ", fuel_stopped_uhev_mj);
        eprintln!("fuel_stopped_conv: {} MJ", fuel_stopped_conv_mj);
        assert!(
            fuel_stopped_uhev_mj < fuel_stopped_conv_mj,
            "Expected stopped uHEV fuel ({fuel_stopped_uhev_mj}) to be less than stopped conventional ({fuel_stopped_conv_mj})"
        );
    }

    #[test]
    fn stop_start_conv_saves_more_fuel_than_normal_conventional() {
        let veh_ss_result = make_conv_pacifica(true, false);
        assert!(veh_ss_result.is_ok());
        let veh_ss = veh_ss_result.unwrap();
        let veh_conv_result = make_conv_pacifica(false, false);
        assert!(veh_conv_result.is_ok());
        let veh_conv = veh_conv_result.unwrap();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd_ss = crate::simdrive::SimDrive::new(veh_ss, cyc.clone(), Default::default());
        let result_ss = sd_ss.walk();
        if let Err(err) = result_ss {
            panic!("Error: {}", err);
        }
        assert!(result_ss.is_ok());
        let mut sd_conv = crate::simdrive::SimDrive::new(veh_conv, cyc.clone(), Default::default());
        let result_conv = sd_conv.walk();
        assert!(result_conv.is_ok());
        let speeds_mps: Vec<f64> = cyc
            .speed
            .iter()
            .map(|spd| spd.get::<si::meter_per_second>())
            .collect();
        let fc_ss = sd_ss.veh.pt_type.fc().unwrap();
        let fc_conv = sd_conv.veh.pt_type.fc().unwrap();
        let fuels_ss_mj: Vec<f64> = fc_ss
            .history
            .energy_fuel
            .iter()
            .map(|ef| ef.get_fresh(|| format_dbg!()).unwrap().get::<si::joule>() / 1e6)
            .collect();
        let fuel_ss_mj: f64 = fuels_ss_mj.iter().sum();
        let fuels_conv_mj: Vec<f64> = fc_conv
            .history
            .energy_fuel
            .iter()
            .map(|ef| ef.get_fresh(|| format_dbg!()).unwrap().get::<si::joule>() / 1e6)
            .collect();
        let fuel_conv_mj: f64 = fuels_conv_mj.iter().sum();
        assert_eq!(speeds_mps.len(), fuels_ss_mj.len());
        assert_eq!(speeds_mps.len(), fuels_conv_mj.len());
        eprintln!("fuel_ss: {} MJ", fuel_ss_mj);
        eprintln!("fuel_conv: {} MJ", fuel_conv_mj);
        assert!(
            fuel_ss_mj < fuel_conv_mj,
            "Expected ss fuel ({fuel_ss_mj}) to be less than conventional ({fuel_conv_mj})"
        );
        let fuel_stopped_ss_mj = accumulate_for_zero_speed(&speeds_mps, &fuels_ss_mj);
        let fuel_stopped_conv_mj = accumulate_for_zero_speed(&speeds_mps, &fuels_conv_mj);
        eprintln!("fuel_stopped_ss: {} MJ", fuel_stopped_ss_mj);
        eprintln!("fuel_stopped_conv: {} MJ", fuel_stopped_conv_mj);
        assert!(
            fuel_stopped_ss_mj < fuel_stopped_conv_mj,
            "Expected stopped ss fuel ({fuel_stopped_ss_mj}) to be less than stopped conventional ({fuel_stopped_conv_mj})"
        );
    }

    #[test]
    fn that_use_stop_start_switches_the_conv_controller() {
        let veh_result = make_conv_pacifica(false, false);
        assert!(veh_result.is_ok());
        let mut veh = veh_result.unwrap();
        let use_result = veh.use_stop_start_controller_py();
        assert!(use_result.is_ok());
        match &veh.pt_type {
            PowertrainType::ConventionalVehicle(conv) => match conv.pt_cntrl {
                ConvPowertrainControls::Normal => {
                    assert!(false, "Powertrain controls didn't change");
                }
                _ => (),
            },
            _ => {
                assert!(false, "Unexpected powertrain type");
            }
        }
        let normal_result = veh.use_normal_controller_py();
        assert!(normal_result.is_ok());
        match &veh.pt_type {
            PowertrainType::ConventionalVehicle(conv) => match conv.pt_cntrl {
                ConvPowertrainControls::StopStart(_) => {
                    assert!(false, "Powertrain controls didn't change");
                }
                _ => (),
            },
            _ => {
                assert!(false, "Unexpected powertrain type");
            }
        }
    }

    #[test]
    fn that_use_stop_start_switches_the_hev_controller() {
        let veh_result = make_microhybrid_pacifica();
        assert!(veh_result.is_ok());
        let mut veh = veh_result.unwrap();
        let use_result = veh.use_normal_controller_py();
        assert!(use_result.is_ok());
        match &veh.pt_type {
            PowertrainType::HybridElectricVehicle(hev) => match &hev.pt_cntrl {
                HEVPowertrainControls::StopStart(_) => {
                    assert!(false, "Powertrain controls didn't change");
                }
                HEVPowertrainControls::RGWDB(_) => (),
            },
            _ => {
                assert!(false, "Unexpected powertrain type");
            }
        }
        let use_ss_result = veh.use_stop_start_controller_py();
        assert!(use_ss_result.is_ok());
        match &veh.pt_type {
            PowertrainType::HybridElectricVehicle(hev) => match &hev.pt_cntrl {
                HEVPowertrainControls::RGWDB(_) => {
                    assert!(
                        false,
                        "Powertrain controls didn't change: RGWDB => StopStart"
                    );
                }
                HEVPowertrainControls::StopStart(_) => (),
            },
            _ => {
                assert!(false, "Unexpected powertrain type");
            }
        }
    }

    fn sum_fuel_in_mj(fc: &FuelConverter) -> f64 {
        let fuels_mj: Vec<f64> = fc
            .history
            .energy_fuel
            .iter()
            .map(|ef| ef.get_fresh(|| format_dbg!()).unwrap().get::<si::joule>() / 1e6)
            .collect();
        fuels_mj.iter().sum()
    }

    #[test]
    fn that_a_vehicle_with_dfco_enabled_uses_less_fuel() {
        let veh_result = make_conv_pacifica(false, false);
        assert!(veh_result.is_ok());
        let veh = veh_result.unwrap();
        let veh_dfco_result = make_conv_pacifica(false, true);
        assert!(veh_dfco_result.is_ok());
        let veh_dfco = veh_dfco_result.unwrap();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let mut sd = crate::simdrive::SimDrive::new(veh, cyc.clone(), Default::default());
        let sd_result = sd.walk();
        if let Err(err) = sd_result {
            panic!("Error: {}", err);
        }
        assert!(sd_result.is_ok());
        let mut sd_dfco = crate::simdrive::SimDrive::new(veh_dfco, cyc.clone(), Default::default());
        let sd_dfco_result = sd_dfco.walk();
        assert!(sd_dfco_result.is_ok());
        let fc = sd.veh.pt_type.fc().unwrap();
        let fc_dfco = sd_dfco.veh.pt_type.fc().unwrap();
        let fuel_mj: f64 = sum_fuel_in_mj(&fc);
        let fuel_dfco_mj: f64 = sum_fuel_in_mj(&fc_dfco);
        eprintln!("fuel     : {} MJ", fuel_mj);
        eprintln!("fuel_dfco: {} MJ", fuel_dfco_mj);
        let percent_reduction = ((fuel_mj - fuel_dfco_mj) * 100.0) / fuel_mj;
        eprintln!("percent reduction: {}", percent_reduction);
        assert!(
            fuel_dfco_mj < fuel_mj,
            "Expected DFCO fuel ({fuel_dfco_mj}) to be less than conventional ({fuel_mj})"
        );
    }
}
