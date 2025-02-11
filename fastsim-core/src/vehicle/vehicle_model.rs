use crate::prelude::*;
#[cfg(feature = "resources")]
use crate::resources;

use super::{hev::HEVPowertrainControls, *};
pub mod fastsim2_interface;

/// Possible aux load power sources
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, IsVariant, From, TryInto)]
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

#[fastsim_api(
    #[staticmethod]
    fn try_from_fastsim2(veh: fastsim_2::vehicle::RustVehicle) -> PyResult<Vehicle> {
        Ok(Self::try_from(veh.clone())?)
    }

    #[pyo3(name = "set_save_interval")]
    #[pyo3(signature = (save_interval=None))]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> PyResult<()> {
        self.set_save_interval(save_interval).map_err(|e| PyAttributeError::new_err(e.to_string()))
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

    fn veh_type(&self) -> PyResult<String> {
        Ok(self.pt_type.to_string())
    }

    // #[getter]
    // fn get_pwr_rated_kilowatts(&self) -> f64 {
    //     self.get_pwr_rated().get::<si::kilowatt>()
    // }

    // #[getter]
    // fn get_mass_kg(&self) -> PyResult<Option<f64>> {
    //     Ok(self.mass()?.map(|m| m))
    // }

    #[cfg(feature = "resources")]
    #[pyo3(name = "list_resources")]
    #[staticmethod]
    /// list available vehicle resources
    fn list_resources_py() -> Vec<String> {
        resources::list_resources(Self::RESOURCE_PREFIX)
    }

    /// Load vehicle from file saved in fastsim-2 format
    #[pyo3(name = "from_f2_file")]
    #[staticmethod]
    fn from_f2_file_py(file: PathBuf) -> anyhow::Result<Self> {
        Self::from_f2_file(file)
    }
)]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, HistoryMethods)]
#[non_exhaustive]
/// Struct for simulating vehicle
pub struct Vehicle {
    /// Vehicle name
    name: String,
    /// Year manufactured
    year: u32,
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

    /// transmission efficiency
    // TODO: check if `trans_eff` is redundant (most likely) and fix
    // TODO: make `transmission::{Transmission, TransmissionState}` and
    // `Transmission` should have field `efficency: Efficiency`.
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
            PowertrainType::BatteryElectricVehicle(bev) => bev.expunge_mass_fields(),
        };
    }
}

impl SerdeAPI for Vehicle {
    #[cfg(feature = "resources")]
    const RESOURCE_PREFIX: &'static str = "vehicles";
}
impl Init for Vehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        let _mass = self.mass().with_context(|| anyhow!(format_dbg!()))?;
        self.calculate_wheel_radius()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.pt_type
            .init()
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }
}

impl SaveInterval for Vehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        self.pt_type.set_save_interval(save_interval)
    }
}

/// TODO: update this constant to match fastsim-2 for gasoline
const FUEL_LHV_MJ_PER_KG: f64 = 43.2;
const CONV: &str = "Conv";
const HEV: &str = "HEV";
const PHEV: &str = "PHEV";
const BEV: &str = "BEV";

impl SetCumulative for Vehicle {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
        if let Some(fc) = self.fc_mut() {
            fc.set_cumulative(dt);
        }
        if let Some(res) = self.res_mut() {
            res.set_cumulative(dt);
        }
        if let Some(em) = self.em_mut() {
            em.set_cumulative(dt);
        }
        self.cabin.set_cumulative(dt);
        self.state.dist += self.state.speed_ach * dt;
    }
}

impl Vehicle {
    /// # Assumptions
    /// - peak power of all components can be produced concurrently.
    pub fn get_pwr_rated(&self) -> si::Power {
        if self.fc().is_some() && self.res().is_some() {
            self.fc().unwrap().pwr_out_max + self.res().unwrap().pwr_out_max
        } else if self.fc().is_some() {
            self.fc().unwrap().pwr_out_max
        } else {
            self.res().unwrap().pwr_out_max
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
                self.state.pwr_tractive,
                self.state,
                true, // `enabled` should always be true at the powertrain level
                dt,
            )
            .with_context(|| anyhow!(format_dbg!()))?;
        self.state.pwr_brake =
            -self.state.pwr_tractive.max(si::Power::ZERO) - self.pt_type.pwr_regen();
        Ok(())
    }

    pub fn set_curr_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // TODO: make transmission field in vehicle and make it be able to produce an efficiency
        // TODO: account for traction limits here or somewhere?

        self.pt_type
            .set_curr_pwr_prop_out_max(self.state.pwr_aux, dt, self.state)
            .with_context(|| anyhow!(format_dbg!()))?;

        (self.state.pwr_prop_fwd_max, self.state.pwr_prop_bwd_max) = self
            .pt_type
            .get_curr_pwr_prop_out_max()
            .with_context(|| anyhow!(format_dbg!()))?;

        Ok(())
    }

    pub fn solve_thermal(
        &mut self,
        te_amb_air: si::Temperature,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let te_fc: Option<si::Temperature> = self.fc().and_then(|fc| fc.temperature());
        let res_temp = self.res().and_then(|res| res.temperature());
        let res_temp_prev = self.res().and_then(|res| res.temp_prev());
        let pwr_thrml_cab_to_res: si::Power = self
            .res()
            .and_then(|res| match &res.thrml {
                RESThermalOption::RESLumpedThermal(rlt) => Some(rlt.state.pwr_thrml_from_cabin),
                RESThermalOption::None => None,
            })
            .unwrap_or_default();
        let (pwr_thrml_fc_to_cabin, pwr_thrml_hvac_to_res, te_cab): (
            Option<si::Power>,
            Option<si::Power>,
            Option<si::Temperature>,
        ) = match (&mut self.cabin, &mut self.hvac) {
            (CabinOption::None, HVACOption::None) => {
                self.state.pwr_aux = self.pwr_aux_base;
                (None, None, None)
            }
            (CabinOption::LumpedCabin(cab), HVACOption::LumpedCabin(hvac)) => {
                let (pwr_thrml_hvac_to_cabin, pwr_thrml_fc_to_cab) = hvac
                    .solve(te_amb_air, te_fc, cab.state, cab.heat_capacitance, dt)
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
                self.state.pwr_aux = self.pwr_aux_base + hvac.state.pwr_aux_for_hvac;
                (Some(pwr_thrml_fc_to_cab), None, Some(te_cab))
            }
            (CabinOption::LumpedCabin(cab), HVACOption::LumpedCabinAndRES(hvac)) => {
                let (pwr_thrml_hvac_to_cabin, pwr_thrml_fc_to_cab, pwr_thrml_hvac_to_res) = hvac
                    .solve(
                        te_amb_air,
                        te_fc,
                        cab.state,
                        cab.heat_capacitance,
                        (
                            res_temp
                            .with_context(
                                || "{}\n[HVACOption::LumpedCabinAndRES] requires [ReversibleEnergyStorage::thrml] to be `Some`"
                            )?,
                            res_temp_prev.unwrap()
                        ),
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
                self.state.pwr_aux = self.pwr_aux_base + hvac.state.pwr_aux_for_hvac;
                (
                    Some(pwr_thrml_fc_to_cab),
                    Some(pwr_thrml_hvac_to_res),
                    Some(te_cab),
                )
            }
            (CabinOption::LumpedCabinWithShell, HVACOption::LumpedCabinWithShell) => {
                bail!("{}\nNot yet implemented.", format_dbg!())
            }
            (CabinOption::None, HVACOption::ReversibleEnergyStorageOnly) => {
                bail!("{}\nNot yet implemented.", format_dbg!())
            }
            (CabinOption::None, _) => {
                bail!(
                    "{}\n`CabinOption::is_none` must be true if `HVACOption::is_none` is true.",
                    format_dbg!()
                )
            }
            (_, HVACOption::None) => {
                bail!(
                    "{}\n`CabinOption::is_none` must be true if `HVACOption::is_none` is true.",
                    format_dbg!()
                )
            }
            _ => todo!(
                "This match needs more match arms to be fully correct in validating model config."
            ),
        };

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

    fn from_f2_file(file: PathBuf) -> anyhow::Result<Self> {
        use fastsim_2::traits::SerdeAPI;
        let f2veh = fastsim_2::vehicle::RustVehicle::from_file(file, false)
            .with_context(|| format_dbg!())?;
        Self::try_from(f2veh)
    }
}

/// Vehicle state for current time step
#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[non_exhaustive]
#[serde(default)]
pub struct VehicleState {
    /// time step index
    pub i: usize,

    /// elapsed simulation time since start
    pub time: si::Time,

    // power and energy fields
    /// maximum forward propulsive power vehicle can produce
    pub pwr_prop_fwd_max: si::Power,
    /// pwr exerted on wheels by powertrain
    /// maximum backward propulsive power (e.g. regenerative braking) vehicle can produce
    pub pwr_prop_bwd_max: si::Power,
    /// Tractive power for achieved speed
    pub pwr_tractive: si::Power,
    /// Tractive power required for prescribed speed
    pub pwr_tractive_for_cyc: si::Power,
    /// integral of [Self::pwr_tractive]
    pub energy_tractive: si::Energy,
    /// time varying aux load
    pub pwr_aux: si::Power,
    /// integral of [Self::pwr_aux]
    pub energy_aux: si::Energy,
    /// Power applied to aero drag
    pub pwr_drag: si::Power,
    /// integral of [Self::pwr_drag]
    pub energy_drag: si::Energy,
    /// Power applied to acceleration (includes deceleration)
    pub pwr_accel: si::Power,
    /// integral of [Self::pwr_accel]
    pub energy_accel: si::Energy,
    /// Power applied to grade ascent
    pub pwr_ascent: si::Power,
    /// integral of [Self::pwr_ascent]
    pub energy_ascent: si::Energy,
    /// Power applied to rolling resistance
    pub pwr_rr: si::Power,
    /// integral of [Self::pwr_rr]
    pub energy_rr: si::Energy,
    /// Power applied to wheel and tire inertia
    pub pwr_whl_inertia: si::Power,
    /// integral of [Self::pwr_whl_inertia]
    pub energy_whl_inertia: si::Energy,
    /// Total braking power including regen
    pub pwr_brake: si::Power,
    /// integral of [Self::pwr_brake]
    pub energy_brake: si::Energy,
    /// whether powertrain can achieve power demand to achieve prescribed speed
    /// in current time step
    // because it should be assumed true in the first time step
    pub cyc_met: bool,
    /// whether powertrain can achieve power demand to achieve prescribed speed
    /// in entire cycle
    pub cyc_met_overall: bool,
    /// actual achieved speed
    pub speed_ach: si::Velocity,
    /// cumulative distance traveled, integral of [Self::speed_ach]
    pub dist: si::Length,
    /// current grade
    pub grade_curr: si::Ratio,
    /// current grade
    // will be overridden during simulation anyway
    pub elev_curr: si::Length,
    /// current air density
    pub air_density: si::MassDensity,
    /// current mass
    // TODO: make sure this gets updated appropriately
    pub mass: si::Mass,
}

impl SerdeAPI for VehicleState {}
impl Init for VehicleState {}
impl Default for VehicleState {
    fn default() -> Self {
        Self {
            i: Default::default(),
            time: si::Time::ZERO,
            pwr_prop_fwd_max: si::Power::ZERO,
            pwr_prop_bwd_max: si::Power::ZERO,
            pwr_tractive: si::Power::ZERO,
            pwr_tractive_for_cyc: si::Power::ZERO,
            energy_tractive: si::Energy::ZERO,
            pwr_aux: si::Power::ZERO,
            energy_aux: si::Energy::ZERO,
            pwr_drag: si::Power::ZERO,
            energy_drag: si::Energy::ZERO,
            pwr_accel: si::Power::ZERO,
            energy_accel: si::Energy::ZERO,
            pwr_ascent: si::Power::ZERO,
            energy_ascent: si::Energy::ZERO,
            pwr_rr: si::Power::ZERO,
            energy_rr: si::Energy::ZERO,
            pwr_whl_inertia: si::Power::ZERO,
            energy_whl_inertia: si::Energy::ZERO,
            pwr_brake: si::Power::ZERO,
            energy_brake: si::Energy::ZERO,
            cyc_met: true,
            cyc_met_overall: true,
            speed_ach: si::Velocity::ZERO,
            dist: si::Length::ZERO,
            // note that this value will be overwritten
            grade_curr: si::Ratio::ZERO,
            // note that this value will be overwritten
            elev_curr: *H_STD,
            air_density: Air::get_density(None, None),
            mass: uc::KG * f64::NAN,
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
}
