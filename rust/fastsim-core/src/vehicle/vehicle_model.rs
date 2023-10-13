use super::*;
use crate::air_properties::*;

/// Possible aux load power sources
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub enum AuxSource {
    /// Aux load power provided by ReversibleEnergyStorage with help from FuelConverter, if present
    /// and needed
    ReversibleEnergyStorage,
    /// Aux load power provided by FuelConverter with help from ReversibleEnergyStorage, if present
    /// and needed
    FuelConverter,
}

/// Possible drive wheel configurations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub enum DriveTypes {
    /// Rear-wheel drive
    RWD,
    /// Front-wheel drive
    FWD,
    /// All-wheel drive
    AWD,
    /// 4-wheel drive
    FourWD,
}

#[pyo3_api(
    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> PyResult<()> {
        self.set_save_interval(save_interval);
        Ok(())
    }

    // #[pyo3(name = "get_save_interval")]
    // /// Set save interval and cascade to nested components.
    // fn get_save_interval_py(&self) -> PyResult<Option<usize>> {
    //     Ok(self.get_save_interval())
    // }

    // #[getter]
    // fn get_fc(&self) -> Option<FuelConverter> {
    //     self.fuel_converter().cloned()
    // }
    // #[setter]
    // fn set_fc(&mut self, _fc: FuelConverter) -> PyResult<()> {
    //     Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    // }

    // #[setter(__fc)]
    // fn set_fc_hidden(&mut self, fc: FuelConverter) -> PyResult<()> {
    //     self.set_fuel_converter(fc).map_err(|e| PyAttributeError::new_err(e.to_string()))
    // }

    // #[getter]
    // fn get_res(&self) -> Option<ReversibleEnergyStorage> {
    //     self.reversible_energy_storage().cloned()
    // }
    // #[setter]
    // fn set_res(&mut self, _res: ReversibleEnergyStorage) -> PyResult<()> {
    //     Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    // }

    // #[setter(__res)]
    // fn set_res_hidden(&mut self, res: ReversibleEnergyStorage) -> PyResult<()> {
    //     self.set_reversible_energy_storage(res).map_err(|e| PyAttributeError::new_err(e.to_string()))
    // }
    // #[getter]
    // fn get_e_machine(&self) -> ElectricMachine {
    //     self.e_machine().clone()
    // }

    // #[setter]
    // fn set_e_machine_py(&mut self, _e_machine: ElectricMachine) -> PyResult<()> {
    //     Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    // }
    // #[setter(__e_machine)]
    // fn set_e_machine_hidden(&mut self, e_machine: ElectricMachine) -> PyResult<()> {
    //     self.set_e_machine(e_machine).map_err(|e| PyAttributeError::new_err(e.to_string()))
    // }

    // fn veh_type(&self) -> PyResult<String> {
    //     Ok(self.pt_type.to_string())
    // }

    // #[getter]
    // fn get_pwr_rated_kilowatts(&self) -> f64 {
    //     self.get_pwr_rated().get::<si::kilowatt>()
    // }

    // #[getter]
    // fn get_mass_kg(&self) -> PyResult<Option<f64>> {
    //     Ok(self.mass()?.map(|m| m))
    // }
)]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, HistoryMethods)]
/// Struct for simulating vehicle
pub struct Vehicle {
    /// Vehicle name
    name: String,
    /// Year manufactured
    year: u32,
    #[has_state]
    #[api(skip_get, skip_set)]
    /// type of vehicle powertrain including contained type-specific parameters and variables
    pub pt_type: PowertrainType,
    /// Aerodynamic drag coefficient
    pub drag_coef: si::Ratio,
    /// Projected frontal area for drag calculations
    pub frontal_area: si::Area,
    /// Vehicle center of mass height
    pub cg_height: si::Length,
    #[api(skip_get, skip_set)]
    /// TODO: make getters and setters for this.
    /// Drive wheel configuration
    pub drive_type: DriveTypes,
    /// Fraction of vehicle weight on drive action when stationary
    /// #[fsim2_name = "drive_axle_weight_frac"]
    pub drive_axle_weight_frac: si::Ratio,
    /// Wheel base length
    /// #[fsim2_name = "wheel_base_m"]
    pub wheel_base: si::Length,
    /// Total vehicle mass
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mass: Option<si::Mass>,
    /// Vehicle mass excluding cargo, passengers, and powertrain components
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    glider_mass: Option<si::Mass>,
    /// Cargo mass including passengers
    /// #[fsim2_name: "cargo_kg"]
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub cargo_mass: Option<si::Mass>,
    // `veh_override_kg` in fastsim-2 is getting deprecated in fastsim-3
    /// Component mass multiplier for vehicle mass calculation
    /// #[fsim2_name = "comp_mass_multiplier"]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub comp_mass_multiplier: Option<si::Ratio>,

    /// power required by auxilliary systems (e.g. HVAC, stereo)
    pub pwr_aux: si::Power,

    /// current state of vehicle
    #[serde(default)]
    pub state: VehicleState,
    /// time step interval at which `state` is saved into `history`
    #[api(skip_set, skip_get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(skip_serializing_if = "VehicleStateHistoryVec::is_empty")]
    pub history: VehicleStateHistoryVec,
}

impl SerdeAPI for Vehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        self.update_mass(None)?;
        Ok(())
    }
}

impl Mass for Vehicle {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.check_mass_consistent()?;
        let mass = match self.mass {
            Some(mass) => Some(mass),
            None => self.derived_mass()?,
        };
        Ok(mass)
    }

    fn update_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()> {
        match mass {
            Some(mass) => {
                // set component masses to None if they aren't consistent
                self.mass = Some(mass);
                if self.check_mass_consistent().is_err() {
                    self.fuel_converter_mut().map(|fc| fc.update_mass(None));
                    self.reversible_energy_storage_mut()
                        .map(|res| res.update_mass(None));
                }
            }
            None => {
                self.mass = Some(
                    self.derived_mass()?
                        .ok_or_else(|| anyhow!("`mass` must be provided or set."))?,
                )
            }
        }
        Ok(())
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        if let (Some(mass_deriv), Some(mass_set)) = (self.derived_mass()?, self.mass) {
            ensure!(
                utils::almost_eq_uom(&mass_set, &mass_deriv, None),
                format!(
                    "{}\n{}",
                    format_dbg!(utils::almost_eq_uom(&mass_set, &mass_deriv, None)),
                    "Try running `update_mass` method."
                )
            )
        }

        Ok(())
    }
}

/// TODO: update this constant to match fastsim-2 for gasoline
const FUEL_LHV_MJ_PER_KG: f64 = 43.2;
const CONV: &str = "Conv";
const HEV: &str = "HEV";
const PHEV: &str = "PHEV";
const BEV: &str = "BEV";

/// Returns fastsim-3 vehicle given fastsim-2 vehicle
///
/// # Arguments
/// * `f2veh` - fastsim-2 vehicle
fn get_pt_type_from_fsim2_veh(
    f2veh: &fastsim_2::vehicle::RustVehicle,
) -> anyhow::Result<PowertrainType> {
    if f2veh.veh_pt_type == CONV {
        let conv = ConventionalVehicle {
            fs: {
                let mut fs = FuelStorage {
                    pwr_out_max: f2veh.fs_max_kw * uc::KW,
                    t_to_peak_pwr: f2veh.fs_secs_to_peak_pwr * uc::S,
                    energy_capacity: f2veh.fs_kwh * 3.6 * uc::MJ,
                    specific_energy: Some(FUEL_LHV_MJ_PER_KG * uc::MJ / uc::KG),
                    mass: None,
                };
                fs.update_mass(None)?;
                fs
            },
            fc: {
                let mut fc = FuelConverter {
                    state: Default::default(),
                    mass: None,
                    specific_pwr: Some(f2veh.fc_kw_per_kg * uc::KW / uc::KG),
                    pwr_out_max: f2veh.fc_max_kw * uc::KW,
                    // assumes 1 s time step
                    pwr_out_max_init: f2veh.fc_max_kw * uc::KW / f2veh.fc_sec_to_peak_pwr,
                    pwr_ramp_lag: f2veh.fc_sec_to_peak_pwr * uc::S,
                    pwr_out_frac_interp: f2veh.fc_pwr_out_perc.to_vec(),
                    eta_interp: f2veh.fc_eff_map.to_vec(),
                    // TODO: verify this
                    pwr_idle_fuel: f2veh.aux_kw
                        / f2veh
                            .fc_eff_map
                            .to_vec()
                            .first()
                            .ok_or_else(|| anyhow!(format_dbg!(f2veh.fc_eff_map)))?
                        * uc::KW,
                    save_interval: Some(1),
                    history: Default::default(),
                };
                fc.update_mass(None)?;
                fc
            },
            trans_eff: f2veh.trans_eff * uc::R,
        };
        Ok(PowertrainType::ConventionalVehicle(Box::new(conv)))
    } else {
        bail!(
            "Invalid powertrain type: {}.
                Expected one of {}",
            f2veh.veh_pt_type,
            [CONV, HEV, PHEV, BEV].join(", "),
        )
    }
}

impl TryFrom<fastsim_2::vehicle::RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_2::vehicle::RustVehicle) -> Result<Self, Self::Error> {
        let mut veh = f2veh.clone();
        veh.set_derived()?;
        let save_interval = Some(1);
        let pt_type = get_pt_type_from_fsim2_veh(&veh)?;

        // TODO: check this sign convention
        let drive_type = if veh.veh_cg_m < 0. {
            DriveTypes::RWD
        } else {
            DriveTypes::FWD
        };

        Ok(Self {
            name: veh.scenario_name,
            year: veh.veh_year,
            pt_type,
            drag_coef: veh.drag_coef * uc::R,
            frontal_area: veh.frontal_area_m2 * uc::M2,
            glider_mass: Some(veh.glider_kg * uc::KG),
            cg_height: veh.veh_cg_m * uc::M,
            drive_type,
            drive_axle_weight_frac: veh.drive_axle_weight_frac * uc::R,
            wheel_base: veh.wheel_base_m * uc::M,
            cargo_mass: Some(veh.cargo_kg * uc::KG),
            comp_mass_multiplier: Some(veh.comp_mass_multiplier * uc::R),
            pwr_aux: f2veh.aux_kw * uc::KW,
            state: Default::default(),
            save_interval,
            history: Default::default(),
            mass: None,
        })
    }
}

impl Vehicle {
    /// # Assumptions
    /// - peak power of all components can be produced concurrently.
    pub fn get_pwr_rated(&self) -> si::Power {
        if self.fuel_converter().is_some() && self.reversible_energy_storage().is_some() {
            self.fuel_converter().unwrap().pwr_out_max
                + self.reversible_energy_storage().unwrap().pwr_out_max
        } else if self.fuel_converter().is_some() {
            self.fuel_converter().unwrap().pwr_out_max
        } else {
            self.reversible_energy_storage().unwrap().pwr_out_max
        }
    }

    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(veh) => {
                veh.fc.save_interval = save_interval;
            }
            PowertrainType::HybridElectricVehicle(veh) => {
                veh.fc.save_interval = save_interval;
                veh.res.save_interval = save_interval;
                veh.e_machine.save_interval = save_interval;
            }
            PowertrainType::BatteryElectricVehicle(veh) => {
                veh.res.save_interval = save_interval;
                veh.e_machine.save_interval = save_interval;
            }
        }
    }

    pub fn fuel_converter(&self) -> Option<&FuelConverter> {
        self.pt_type.fuel_converter()
    }

    pub fn fuel_converter_mut(&mut self) -> Option<&mut FuelConverter> {
        self.pt_type.fuel_converter_mut()
    }

    pub fn set_fuel_converter(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        self.pt_type.set_fuel_converter(fc)
    }

    pub fn reversible_energy_storage(&self) -> Option<&ReversibleEnergyStorage> {
        self.pt_type.reversible_energy_storage()
    }

    pub fn reversible_energy_storage_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        self.pt_type.reversible_energy_storage_mut()
    }

    pub fn set_reversible_energy_storage(
        &mut self,
        res: ReversibleEnergyStorage,
    ) -> anyhow::Result<()> {
        self.pt_type.set_reversible_energy_storage(res)
    }

    pub fn e_machine(&self) -> Option<&ElectricMachine> {
        self.pt_type.e_machine()
    }

    pub fn e_machine_mut(&mut self) -> Option<&mut ElectricMachine> {
        self.pt_type.e_machine_mut()
    }

    /// Calculate mass from components.
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        if let Some(_glider_mass) = self.glider_mass {
            match self.pt_type {
                PowertrainType::ConventionalVehicle(_) => {
                    // TODO: add the other component and vehicle level masses here
                    if let Some(fc) = self.fuel_converter().unwrap().mass()? {
                        Ok(Some(fc))
                    } else {
                        bail!(
                            "TODO: fix this error message\n{}\n{}",
                            "so `fc` and `gen` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::HybridElectricVehicle(_) => {
                    if let (Some(fc), Some(res)) = (
                        self.fuel_converter().unwrap().mass()?,
                        self.reversible_energy_storage().unwrap().mass()?,
                    ) {
                        // TODO: add the other component and vehicle level masses here
                        Ok(Some(fc + res))
                    } else {
                        // TODO: update error message
                        bail!(
                            "TODO: fix this error message\n{}\n{}",
                            "so `fc`, `gen`, and `res` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                PowertrainType::BatteryElectricVehicle(_) => {
                    if let Some(res) = self.reversible_energy_storage().unwrap().mass()? {
                        Ok(Some(res))
                    } else {
                        bail!(
                            "TODO: fix this error message\n{}\n{}",
                            "so `res` mass must also be specified.",
                            format_dbg!()
                        )
                    }
                }
            }
            // TODO: probably need more `else ... if` branches here
        } else {
            bail!(
                "Both `baseline` and `ballast` masses must be either `Some` or `None`\n{}",
                format_dbg!()
            )
        }
    }

    /// Given required power output and time step, solves for energy consumption
    /// # Arguments
    /// - `pwr_out_req`: float, output brake power required from fuel converter.
    /// - `dt`: current time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: think carefully about whether `self.state.pwr_out` ought to include
        // `self.state.pwr_aux` and document accordingly in the places
        self.state.pwr_out = pwr_out_req + pwr_aux;
        self.state.pwr_aux = pwr_aux;
        match &mut self.pt_type {
            PowertrainType::ConventionalVehicle(conv) => {
                // TODO: put logic for toggling `fc_on` here
                let fc_on = true;
                // TODO: propagate this
                let assert_limits = true;
                conv.solve_energy_consumption(
                    pwr_out_req,
                    self.state.pwr_aux,
                    fc_on,
                    dt,
                    assert_limits,
                )?;
                self.state.pwr_out = conv.fc.state.pwr_out * conv.trans_eff;
            }
            PowertrainType::HybridElectricVehicle(_hev) => {
                todo!()
            }
            PowertrainType::BatteryElectricVehicle(_bev) => {
                todo!()
            }
        }
        self.state.energy_out += self.state.pwr_out * dt;
        self.state.energy_aux += self.state.pwr_aux * dt;
        Ok(())
    }

    pub fn set_cur_pwr_max_out(&mut self, pwr_aux: si::Power, dt: si::Time) -> anyhow::Result<()> {
        self.state.pwr_out_max = self.pt_type.get_cur_pwr_max_out(pwr_aux, dt)?;
        Ok(())
    }
}

/// Vehicle state for current time step
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, Default)]
#[pyo3_api]
pub struct VehicleState {
    /// time step index
    pub i: usize,

    // power and fields
    /// maximum forward propulsive power vehicle can produce
    pub pwr_out_max: si::Power,
    /// maximum regen power vehicle can absorb at the wheel
    pub pwr_regen_max: si::Power,
    /// actual wheel power achieved
    pub pwr_out: si::Power,
    /// integral of [Self::pwr_out]
    pub energy_out: si::Energy,
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
    /// Power applied to wheel inertia
    pub pwr_whl_inertia: si::Power,
    /// integral of [Self::pwr_whl_inertia]
    pub energy_whl_inertia: si::Energy,
    /// Total braking power
    pub pwr_brake: si::Power,
    /// integral of [Self::pwr_brake]
    pub energy_brake: si::Energy,
    /// whether powertrain can achieve power demand
    pub cyc_met: bool,
    /// actual achieved speed
    pub speed_ach: si::Velocity,
    /// actual achieved speed in previous time step
    pub speed_ach_prev: si::Velocity,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn test_load_f2_fusion() {
        _ = mock_f2_conv_veh();
    }

    pub(crate) fn mock_f2_conv_veh() -> Vehicle {
        let file_contents = include_str!("fastsim-2_2012_Ford_Fusion.yaml");
        use fastsim_2::traits::SerdeAPI;
        let veh =
            Vehicle::try_from(fastsim_2::vehicle::RustVehicle::from_yaml(file_contents).unwrap())
                .unwrap();
        // uncomment this if the fastsim-3 version needs to be rewritten
        // veh.to_file("2012_Ford_Fusion.yaml").unwrap();
        #[allow(clippy::let_and_return)]
        veh
    }
}
