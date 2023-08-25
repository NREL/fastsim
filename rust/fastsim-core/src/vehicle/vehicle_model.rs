use super::*;

#[enum_dispatch(VehicleTrait)]
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
            PowertrainType::HybridElectricVehicle(loco) => {
                loco.res = res;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(loco) => {
                loco.res = res;
                Ok(())
            }
        }
    }

    pub fn e_machine(&self) -> &ElectricMachine {
        match self {
            PowertrainType::ConventionalVehicle(conv) => &conv.e_machine,
            PowertrainType::HybridElectricVehicle(hev) => &hev.e_machine,
            PowertrainType::BatteryElectricVehicle(bev) => &bev.e_machine,
        }
    }

    pub fn e_machine_mut(&mut self) -> &mut ElectricMachine {
        match self {
            PowertrainType::ConventionalVehicle(conv) => &mut conv.e_machine,
            PowertrainType::HybridElectricVehicle(hev) => &mut hev.e_machine,
            PowertrainType::BatteryElectricVehicle(bev) => &mut bev.e_machine,
        }
    }

    pub fn set_e_machine(&mut self, e_machine: ElectricMachine) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.e_machine = e_machine;
                Ok(())
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

impl std::string::ToString for PowertrainType {
    fn to_string(&self) -> String {
        match self {
            PowertrainType::ConventionalVehicle(_) => String::from("Conv"),
            PowertrainType::HybridElectricVehicle(_) => String::from("HEV"),
            PowertrainType::BatteryElectricVehicle(_) => String::from("BEV"),
        }
    }
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

    // fn loco_type(&self) -> PyResult<String> {
    //     Ok(self.powertrain_type.to_string())
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
/// Struct for simulating any type of locomotive
pub struct Vehicle {
    /// Vehicle name
    name: String,
    /// Year manufactured
    year: u32,
    #[api(skip_get, skip_set)]
    /// type of locomotive including contained type-specific parameters
    /// and variables
    pub powertrain_type: PowertrainType,
    /// Aerodynamic drag coefficient
    drag_coeff: si::Ratio,
    /// Projected frontal area for drag calculations
    frontal_area: si::Area,
    /// Vehicle mass excluding cargo, passengers, and powertrain components
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    glider_mass: Option<si::Mass>,
    /// Vehicle center of mass height
    cg_height: si::Length,
    #[api(skip_get, skip_set)]
    /// TODO: make getters and setters for this.
    /// Drive wheel configuration
    drive_type: DriveTypes,
    /// Fraction of vehicle weight on drive action when stationary
    /// #[fsim2_name = "drive_axle_weight_frac"]
    drive_axle_weight_frac: si::Ratio,
    /// Wheel base length
    /// #[fsim2_name = "wheel_base_m"]
    wheel_base: si::Length,
    /// Total vehicle mass
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    pub mass: Option<si::Mass>,
    /// Cargo mass including passengers
    /// #[fsim2_name: "cargo_kg"]
    #[serde(default)]
    pub cargo_mass: si::Mass,
    // `veh_override_kg` in fastsim-2 is getting deprecated in fastsim-3
    /// Component mass multiplier for vehicle mass calculation
    /// #[fsim2_name = "comp_mass_multiplier"]
    pub comp_mass_multiplier: si::Ratio,

    /// current state of vehicle
    #[serde(default)]
    pub state: VehicleState,
    /// time step interval at which `state` is saved into `history`
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
    /// Custom vector of [Self::state]
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

impl TryFrom<fastsim_2::vehicle::RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(veh: fastsim_2::vehicle::RustVehicle) -> Result<Self, Self::Error> {
        let mut veh = veh.clone();
        veh.set_derived();
        let save_interval = Some(1);
        let powertrain_type: PowertrainType = if &veh.veh_pt_type == CONV {
            let conv = ConventionalVehicle {
                fs: {
                    let mut fs = FuelStorage {
                        pwr_out_max: veh.fs_max_kw * uc::KW,
                        t_to_peak_pwr: veh.fs_secs_to_peak_pwr * uc::S,
                        energy_capacity: veh.fs_kwh * 3.6 * uc::MJ,
                        specific_energy: Some(FUEL_LHV_MJ_PER_KG * uc::MJ / uc::KG),
                        mass: None,
                    };
                    fs.update_mass(None);
                    fs
                },
                fc: {
                    let mut fc = FuelConverter {
                        state: Default::default(),
                        mass: todo!(),
                        specific_pwr_kw_per_kg: todo!(),
                        pwr_out_max: todo!(),
                        pwr_out_max_init: todo!(),
                        pwr_ramp_lag: todo!(),
                        pwr_out_frac_interp: todo!(),
                        eta_interp: todo!(),
                        pwr_idle_fuel: todo!(),
                        save_interval: todo!(),
                        history: Default::default(),
                    };
                    fc.update_mass(None);
                    fc
                },
                e_machine: ElectricMachine {
                    state: Default::default(),
                    pwr_out_frac_interp: veh.mc_pwr_out_perc.to_vec(),
                    eta_interp: veh.mc_eff_map.to_vec(),
                    pwr_in_frac_interp: vec![],
                    pwr_out_max: veh.mc_max_kw * uc::KW,
                    save_interval,
                    history: Default::default(),
                },
            };
            Some(PowertrainType::ConventionalVehicle(Box::new(conv)))
        } else {
            None
        }
        .expect(&format!(
            "Invalid powertrain type: {}.
            Expected one of {}",
            veh.veh_pt_type,
            [CONV, HEV, PHEV, BEV].join(", ")
        ));

        let drive_type = if veh.veh_cg_m < 0. {
            DriveTypes::RWD
        } else {
            DriveTypes::FWD
        };

        Ok(Self {
            name: veh.scenario_name,
            year: veh.veh_year,
            powertrain_type,
            drag_coeff: veh.drag_coef * uc::R,
            frontal_area: veh.frontal_area_m2 * uc::M2,
            glider_mass: Some(veh.glider_kg * uc::KG),
            cg_height: veh.veh_cg_m * uc::M,
            // TODO: check the logic here!
            drive_type,
            drive_axle_weight_frac: veh.drive_axle_weight_frac * uc::R,
            wheel_base: veh.wheel_base_m * uc::M,
            cargo_mass: veh.cargo_kg * uc::KG,
            comp_mass_multiplier: veh.comp_mass_multiplier * uc::R,
            state: Default::default(),
            save_interval,
            history: Default::default(),
            mass: None,
        })
    }
}

impl Vehicle {
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
        match &mut self.powertrain_type {
            PowertrainType::ConventionalVehicle(loco) => {
                loco.fc.save_interval = save_interval;
                loco.e_machine.save_interval = save_interval;
            }
            PowertrainType::HybridElectricVehicle(loco) => {
                loco.fc.save_interval = save_interval;
                loco.res.save_interval = save_interval;
                loco.e_machine.save_interval = save_interval;
            }
            PowertrainType::BatteryElectricVehicle(loco) => {
                loco.res.save_interval = save_interval;
                loco.e_machine.save_interval = save_interval;
            }
        }
    }

    pub fn fuel_converter(&self) -> Option<&FuelConverter> {
        self.powertrain_type.fuel_converter()
    }

    pub fn fuel_converter_mut(&mut self) -> Option<&mut FuelConverter> {
        self.powertrain_type.fuel_converter_mut()
    }

    pub fn set_fuel_converter(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        self.powertrain_type.set_fuel_converter(fc)
    }

    pub fn reversible_energy_storage(&self) -> Option<&ReversibleEnergyStorage> {
        self.powertrain_type.reversible_energy_storage()
    }

    pub fn reversible_energy_storage_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        self.powertrain_type.reversible_energy_storage_mut()
    }

    pub fn set_reversible_energy_storage(
        &mut self,
        res: ReversibleEnergyStorage,
    ) -> anyhow::Result<()> {
        self.powertrain_type.set_reversible_energy_storage(res)
    }

    pub fn e_machine(&self) -> &ElectricMachine {
        self.powertrain_type.e_machine()
    }

    pub fn e_machine_mut(&mut self) -> &mut ElectricMachine {
        self.powertrain_type.e_machine_mut()
    }

    /// Calculate mass from components.
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        if let Some(_glider_mass) = self.glider_mass {
            match self.powertrain_type {
                PowertrainType::ConventionalVehicle(_) => {
                    // TODO: add the other component and vehicle level masses here
                    if let Some(fc) = self.fuel_converter().unwrap().mass()? {
                        Ok(Some(fc))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
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
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
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
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
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

    /// Given required power output and time step, solves for energy
    /// consumption Arguments:
    /// ----------
    /// pwr_out_req: float, output brake power required from fuel
    /// converter. dt: current time step size engine_on: whether or not
    /// locomotive is active
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: Option<bool>,
    ) -> anyhow::Result<()> {
        // maybe put logic for toggling `engine_on` here

        self.state.pwr_out = pwr_out_req;
        match &mut self.powertrain_type {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.solve_energy_consumption(
                    pwr_out_req,
                    dt,
                    engine_on.unwrap_or(true),
                    self.state.pwr_aux,
                )?;
                self.state.pwr_out = conv.e_machine.state.pwr_mech_prop_out
                    - conv.e_machine.state.pwr_mech_dyn_brake;
            }
            PowertrainType::HybridElectricVehicle(_hev) => {}
            PowertrainType::BatteryElectricVehicle(_bev) => {}
        }
        self.state.energy_out += self.state.pwr_out * dt;
        self.state.energy_aux += self.state.pwr_aux * dt;
        Ok(())
    }
}

fn set_pwr_lims(state: &mut VehicleState, e_machine: &ElectricMachine) {
    state.pwr_out_max = e_machine.state.pwr_mech_out_max;
    state.pwr_rate_out_max = e_machine.state.pwr_rate_out_max;
    state.pwr_regen_max = e_machine.state.pwr_mech_regen_max;
}

impl VehicleTrait for Vehicle {
    fn step(&mut self) {
        self.powertrain_type.step();
        self.state.i += 1;
    }

    fn save_state(&mut self) {
        self.powertrain_type.save_state();
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || self.state.i == 1 {
                self.history.push(self.state);
            }
        }
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.powertrain_type.get_energy_loss()
    }

    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            pwr_aux.is_none(),
            format!(
                "{}\ntime step: {}",
                format_dbg!(pwr_aux.is_none()),
                self.state.i
            )
        );

        self.powertrain_type
            .set_cur_pwr_max_out(Some(self.state.pwr_aux), dt)?;
        match &self.powertrain_type {
            PowertrainType::ConventionalVehicle(loco) => {
                // TODO: Coordinate with Geordie on the rate
                set_pwr_lims(&mut self.state, &loco.e_machine);
                assert_eq!(self.state.pwr_regen_max, si::Power::ZERO);
            }
            PowertrainType::HybridElectricVehicle(loco) => {
                set_pwr_lims(&mut self.state, &loco.e_machine);
                // TODO: Coordinate with Geordie on rate
            }
            PowertrainType::BatteryElectricVehicle(loco) => {
                set_pwr_lims(&mut self.state, &loco.e_machine);
                // TODO: Coordinate with Geordie on rate; INCOMPLETE ON
                // RATE (Jinghu as of 06/06/2022)
            }
        }
        Ok(())
    }
}

/// Locomotive state for current time step
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[pyo3_api]
pub struct VehicleState {
    pub i: usize,
    /// maximum forward propulsive power locomotive can produce
    pub pwr_out_max: si::Power,
    /// maximum rate of increase of forward propulsive power locomotive
    /// can produce
    pub pwr_rate_out_max: si::PowerRate,
    /// maximum regen power locomotive can absorb at the wheel
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
            pwr_rate_out_max: Default::default(),
            pwr_out: Default::default(),
            pwr_regen_max: Default::default(),
            energy_out: Default::default(),
            pwr_aux: Default::default(),
            energy_aux: Default::default(),
        }
    }
}
