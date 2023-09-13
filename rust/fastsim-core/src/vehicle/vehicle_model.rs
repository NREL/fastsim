use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub enum PowertrainType {
    ConventionalVehicle(Box<ConventionalVehicle>),
    HybridElectricVehicle(Box<HybridElectricVehicle>),
    BatteryElectricVehicle(Box<BatteryElectricVehicle>),
    // TODO: add PHEV here
}

impl PowertrainType {
    pub fn fuel_converter(&self) -> Option<&FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.fc),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn fuel_converter_mut(&mut self) -> Option<&mut FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.fc),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn set_fuel_converter(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.fc = fc;
                Ok(())
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.fc = fc;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(_) => bail!("BEL has no FuelConverter."),
        }
    }

    pub fn reversible_energy_storage(&self) -> Option<&ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.res),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.res),
        }
    }

    pub fn reversible_energy_storage_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.res),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&mut bev.res),
        }
    }

    pub fn set_reversible_energy_storage(
        &mut self,
        res: ReversibleEnergyStorage,
    ) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(_) => {
                bail!("Conventional has no ReversibleEnergyStorage.")
            }
            PowertrainType::HybridElectricVehicle(veh) => {
                veh.res = res;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(veh) => {
                veh.res = res;
                Ok(())
            }
        }
    }

    pub fn e_machine(&self) -> Option<&ElectricMachine> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.e_machine),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.e_machine),
        }
    }

    pub fn e_machine_mut(&mut self) -> Option<&mut ElectricMachine> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.e_machine),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&mut bev.e_machine),
        }
    }

    pub fn set_e_machine(&mut self, e_machine: ElectricMachine) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => {
                Err(anyhow!("ConventionalVehicle has no `e_machine`"))
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.e_machine = e_machine;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(bev) => {
                bev.e_machine = e_machine;
                Ok(())
            }
        }
    }
}

impl VehicleTrait for PowertrainType {
    fn set_cur_pwr_max_out(&mut self, pwr_aux: si::Power, dt: si::Time) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => conv.set_cur_pwr_max_out(pwr_aux, dt)?,
            Self::HybridElectricVehicle(hev) => hev.set_cur_pwr_max_out(pwr_aux, dt)?,
            Self::BatteryElectricVehicle(bev) => bev.set_cur_pwr_max_out(pwr_aux, dt)?,
        }
        Ok(())
    }

    fn save_state(&mut self) {
        match self {
            Self::ConventionalVehicle(conv) => conv.save_state(),
            Self::HybridElectricVehicle(hev) => hev.save_state(),
            Self::BatteryElectricVehicle(bev) => bev.save_state(),
        }
    }

    fn step(&mut self) {
        match self {
            Self::ConventionalVehicle(conv) => conv.step(),
            Self::HybridElectricVehicle(hev) => hev.step(),
            Self::BatteryElectricVehicle(bev) => bev.step(),
        }
    }
}

impl std::string::ToString for PowertrainType {
    fn to_string(&self) -> String {
        match self {
            PowertrainType::ConventionalVehicle(_) => String::from("Conv"),
            PowertrainType::HybridElectricVehicle(_) => String::from("HEV"),
            PowertrainType::BatteryElectricVehicle(_) => String::from("BEV"),
        }
    }
}

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
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
/// Struct for simulating vehicle
pub struct Vehicle {
    /// Vehicle name
    name: String,
    /// Year manufactured
    year: u32,
    #[api(skip_get, skip_set)]
    /// type of vehicle powertrain including contained type-specific parameters and variables
    pub pt_type: PowertrainType,
    /// Aerodynamic drag coefficient
    pub drag_coeff: si::Ratio,
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
            drag_coeff: veh.drag_coef * uc::R,
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

    /// Solves current time step
    /// # Arguments
    /// - speed - prescribed speed
    /// - dt - time step size
    pub fn solve_step(&mut self, speed: si::Velocity, dt: si::Time) -> anyhow::Result<()> {
        self.set_cur_pwr_max_out(self.pwr_aux, dt)?;
        self.get_req_pwr(speed, dt)?;
        self.set_ach_speed(dt)?;
        Ok(())
    }

    /// Sets power required for given prescribed speed
    /// # Arguments
    /// - speed - prescribed or achieved speed
    /// - dt - time step size
    pub fn get_req_pwr(&mut self, speed: si::Velocity, dt: si::Time) -> anyhow::Result<si::Power> {
        Ok(uc::W * 666.)
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - dt - time step size
    pub fn set_ach_speed(&mut self, dt: si::Time) -> anyhow::Result<()> {
        todo!();
        Ok(())
    }

    /// Given required power output and time step, solves for energy consumption
    /// # Arguments
    /// * pwr_out_req: float, output brake power required from fuel converter.
    /// * dt: current time step size
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

    #[allow(unused)]
    pub(crate) fn test_conv_veh() -> Self {
        let file_contents = include_str!("2012_Ford_Fusion.yaml");
        Self::from_yaml(file_contents).unwrap()
    }
}

impl VehicleTrait for Vehicle {
    fn step(&mut self) {
        self.pt_type.step();
        self.state.i += 1;
    }

    fn save_state(&mut self) {
        self.pt_type.save_state();
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || self.state.i == 1 {
                self.history.push(self.state);
            }
        }
    }

    fn set_cur_pwr_max_out(&mut self, pwr_aux: si::Power, dt: si::Time) -> anyhow::Result<()> {
        self.pt_type.set_cur_pwr_max_out(pwr_aux, dt)?;

        Ok(())
    }
}

/// Vehicle state for current time step
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[pyo3_api]
pub struct VehicleState {
    pub i: usize,
    /// maximum forward propulsive power vehicle can produce
    pub pwr_out_max: si::Power,
    /// maximum regen power vehicle can absorb at the wheel
    pub pwr_regen_max: si::Power,
    /// actual wheel power achieved
    pub pwr_out: si::Power,
    /// time varying aux load
    pub pwr_aux: si::Power,
    //todo: add variable for statemachine pwr_out_prev,
    //time_at_or_below_idle, time_in_engine_state
    /// integral of [Self::pwr_out]
    pub energy_out: si::Energy,
    /// integral of [Self::pwr_aux]
    pub energy_aux: si::Energy,
    // pub force_max: si::Mass,
}

impl Default for VehicleState {
    fn default() -> Self {
        Self {
            i: 1,
            pwr_out_max: Default::default(),
            pwr_out: Default::default(),
            pwr_regen_max: Default::default(),
            energy_out: Default::default(),
            pwr_aux: Default::default(),
            energy_aux: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_fusion() {
        let veh = Vehicle::test_conv_veh();
    }
}
