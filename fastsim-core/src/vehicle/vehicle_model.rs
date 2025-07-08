use crate::prelude::*;

use super::{hev::HEVPowertrainControls, *};
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
    /// Year manufactured
    pub year: u32,
    #[has_state]
    /// type of vehicle powertrain including contained type-specific parameters and variables
    pub pt_type: PowertrainType,

    /// Chassis model with various chassis-related parameters
    pub chassis: Chassis,

    /// Cabin thermal model
    #[serde(default, skip_serializing_if = "CabinOption::is_none")]
    #[has_state]
    pub cabin: CabinOption,

    /// HVAC model
    #[serde(default, skip_serializing_if = "HVACOption::is_none")]
    #[has_state]
    pub hvac: HVACOption,

    /// Total vehicle mass
    pub(crate) mass: Option<si::Mass>,

    /// Baseline power required by auxilliary systems
    pub pwr_aux_base: si::Power,

    /// Transmission efficiency
    pub trans_eff: si::Ratio,

    /// time step interval at which `state` is saved into `history`
    save_interval: Option<usize>,
    /// current state of vehicle
    #[serde(default)]
    pub state: VehicleState,
    /// Vector-like history of [Self::state]
    #[serde(default, skip_serializing_if = "VehicleStateHistoryVec::is_empty")]
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

    #[pyo3(name = "clear")]
    fn clear_py(&mut self) {
        self.clear()
    }

    #[pyo3(name = "reset_step")]
    fn reset_step_py(&mut self) -> anyhow::Result<()> {
        self.reset_step(|| format_dbg!())
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
        self.mass = match new_mass {
            // Set using provided `new_mass`, setting constituent mass fields to `None` to match if inconsistent
            Some(new_mass) => {
                if let Some(dm) = derived_mass {
                    if dm != new_mass {
                        self.expunge_mass_fields();
                    }
                }
                Some(new_mass)
            }
            // Set using `derived_mass()`, failing if it returns `None`
            None => Some(derived_mass.with_context(|| {
                format!(
                    "Not all mass fields in `{}` are set and no mass was provided.",
                    stringify!(Vehicle)
                )
            })?),
        };
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
const FUEL_LHV_MJ_PER_KG: f64 = 43.2;
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
            .with_context(|| anyhow!(format_dbg!()))?;
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
        // TODO: account for traction limits here
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
        self.state
            .pwr_prop_fwd_max
            .update(pwr_prop_maxes.0, || format_dbg!())?;
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
                )
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
    use super::*;

    #[allow(dead_code)]
    fn vehicles_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/vehicles")
    }

    #[cfg(feature = "yaml")]
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
}
