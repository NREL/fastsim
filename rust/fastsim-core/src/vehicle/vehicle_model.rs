use super::*;

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
    // TODO: expose `try_from` method so python users can load fastsim-2 vehicles
    
    #[setter("save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.set_save_interval(save_interval)
    }

    #[getter("save_interval")]
    /// Set save interval and cascade to nested components.
    fn get_save_interval_py(&self) -> anyhow::Result<Option<usize>> {
        self.save_interval()
    }

    #[getter]
    fn get_fc(&self) -> Option<FuelConverter> {
        self.fc().cloned()
    }
    #[setter("fc")]
    fn set_fc_py(&mut self, _fc: FuelConverter) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

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
    /// Wheel rolling resistance coefficient for the vehicle (i.e. all wheels included)
    pub wheel_rr_coef: si::Ratio,
    /// Wheel inertia per wheel
    pub wheel_inertia: si::MomentOfInertia,
    /// Number of wheels
    pub num_wheels: u8,
    /// Wheel radius
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub wheel_radius: Option<si::Length>,
    /// Tire code (optional method of calculating wheel radius)
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub tire_code: Option<String>,
    /// Vehicle center of mass height
    pub cg_height: si::Length,
    #[api(skip_get, skip_set)]
    /// TODO: make getters and setters for this.
    /// Drive wheel configuration
    pub drive_type: DriveTypes,
    /// Fraction of vehicle weight on drive action when stationary
    pub drive_axle_weight_frac: si::Ratio,
    /// Wheel base length
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

    /// transmission efficiency
    pub trans_eff: si::Ratio,

    /// current state of vehicle
    #[serde(default)]
    pub state: VehicleState,
    /// time step interval at which `state` is saved into `history`
    #[api(skip_set, skip_get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    save_interval: Option<usize>,
    /// Vector-like history of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "VehicleStateHistoryVec::is_empty")]
    pub history: VehicleStateHistoryVec,
}

impl SerdeAPI for Vehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        self.update_mass(None)?;
        self.calculate_wheel_radius()?;
        Ok(())
    }
}

impl Mass for Vehicle {
    // TODO: make the Option go away and throw error if None is returned
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
                    self.fc_mut().map(|fc| fc.update_mass(None));
                    self.res_mut()
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
                    pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
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
        let mut f2veh = f2veh.clone();
        f2veh.set_derived()?;
        let save_interval = Some(1);
        let pt_type = get_pt_type_from_fsim2_veh(&f2veh)?;

        // TODO: check this sign convention
        let drive_type = if f2veh.veh_cg_m < 0. {
            DriveTypes::RWD
        } else {
            DriveTypes::FWD
        };

        let mut f3veh = Self {
            name: f2veh.scenario_name,
            year: f2veh.veh_year,
            pt_type,
            drag_coef: f2veh.drag_coef * uc::R,
            frontal_area: f2veh.frontal_area_m2 * uc::M2,
            glider_mass: Some(f2veh.glider_kg * uc::KG),
            cg_height: f2veh.veh_cg_m * uc::M,
            drive_type,
            drive_axle_weight_frac: f2veh.drive_axle_weight_frac * uc::R,
            wheel_base: f2veh.wheel_base_m * uc::M,
            wheel_inertia: f2veh.wheel_inertia_kg_m2 * uc::KGM2,
            wheel_rr_coef: f2veh.wheel_rr_coef * uc::R,
            num_wheels: f2veh.num_wheels as u8,
            wheel_radius: Some(f2veh.wheel_radius_m * uc::M),
            tire_code: None,
            cargo_mass: Some(f2veh.cargo_kg * uc::KG),
            comp_mass_multiplier: Some(f2veh.comp_mass_multiplier * uc::R),
            pwr_aux: f2veh.aux_kw * uc::KW,
            trans_eff: f2veh.trans_eff * uc::R,
            state: Default::default(),
            save_interval,
            history: Default::default(),
            mass: None,
        };
        f3veh.init()?;

        Ok(f3veh)
    }
}

impl Vehicle {
    /// # Assumptions
    /// - peak power of all components can be produced concurrently.
    pub fn get_pwr_rated(&self) -> si::Power {
        if self.fc().is_some() && self.res().is_some() {
            self.fc().unwrap().pwr_out_max
                + self.res().unwrap().pwr_out_max
        } else if self.fc().is_some() {
            self.fc().unwrap().pwr_out_max
        } else {
            self.res().unwrap().pwr_out_max
        }
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

    pub fn set_res(
        &mut self,
        res: ReversibleEnergyStorage,
    ) -> anyhow::Result<()> {
        self.pt_type.set_res(res)
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
                    if let Some(fc) = self.fc().unwrap().mass()? {
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
                        self.fc().unwrap().mass()?,
                        self.res().unwrap().mass()?,
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
                    if let Some(res) = self.res().unwrap().mass()? {
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

    /// Calculate wheel radius from tire code, if applicable
    fn calculate_wheel_radius(&mut self) -> anyhow::Result<()> {
        ensure!(self.wheel_radius.is_some() || self.tire_code.is_some(), "Either `wheel_radius` or `tire_code` must be supplied");
        if self.wheel_radius.is_none() {
            self.wheel_radius = Some(utils::tire_code_to_radius(self.tire_code.as_ref().unwrap())? * uc::M)
        }
        Ok(())
    }

    /// Solves for energy consumption
    pub fn solve_powertrain(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // TODO: do something more sophisticated with pwr_aux
        self.state.pwr_aux = self.pwr_aux;
        // TODO: put logic for toggling `fc_on` here, after moving this comment to the right place
        // let fc_on = true;
        // TODO: propagate this
        self.pt_type.solve_powertrain(
            self.state.pwr_tractive,
            self.pwr_aux,
            dt,
            true, // TODO: parameterize this
        )?;
        Ok(())
    }

    pub fn set_cur_pwr_max_out(&mut self, dt: si::Time) -> anyhow::Result<()> {
        self.state.pwr_tractive_max = self.get_pwr_out_max(dt)?;
        Ok(())
    }

    pub fn get_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<si::Power> {
        Ok(self.pt_type.get_pwr_out_max(dt)? * self.trans_eff)
    }

    pub fn to_fastsim2(&self) -> anyhow::Result<fastsim_2::vehicle::RustVehicle> {
        let mut veh = fastsim_2::vehicle::RustVehicle{
            alt_eff: todo!(),
            alt_eff_doc: None,
            aux_kw: todo!(),
            aux_kw_doc: None,
            cargo_kg: self.cargo_mass.unwrap_or_default().get::<si::kilogram>(),
            cargo_kg_doc: None,
            charging_on: false,
            chg_eff: todo!(),
            chg_eff_doc: None,
            comp_mass_multiplier: 1.4,
            comp_mass_multiplier_doc: None,
            // TODO: replace with `doc` field once implemented in fastsim-3
            doc: None,
            drag_coef: self.drag_coef.get::<si::ratio>(),
            drag_coef_doc: None,
            drive_axle_weight_frac: self.drive_axle_weight_frac.get::<si::ratio>(),
            drive_axle_weight_frac_doc: None,
            ess_base_kg: todo!(),
            ess_base_kg_doc: None,
            ess_chg_to_fc_max_eff_perc: todo!(),
            ess_chg_to_fc_max_eff_perc_doc: None,
            ess_dischg_to_fc_max_eff_perc: todo!(),
            ess_dischg_to_fc_max_eff_perc_doc: None,
            ess_kg_per_kwh: todo!(),
            ess_kg_per_kwh_doc: None,
            ess_life_coef_a: todo!(),
            ess_life_coef_a_doc: None,
            ess_life_coef_b: todo!(),
            ess_life_coef_b_doc: None,
            ess_mass_kg: todo!(),
            ess_max_kw: self.res().map(|res| res.pwr_out_max.get::<si::kilowatt>()).unwrap_or_default(),
            ess_max_kw_doc: None,
            ess_max_kwh: self.res().map(|res| res.energy_capacity.get::<si::kilowatt_hour>()).unwrap_or_default(),
            ess_max_kwh_doc: None,
            ess_round_trip_eff: todo!(), // SOC is not time-varying in fastsim-2 // self.res().map(|res| res.eta.get::<si::ratio>().powi(2)).unwrap_or_default()
            ess_round_trip_eff_doc: None,
            ess_to_fuel_ok_error: todo!(),
            ess_to_fuel_ok_error_doc: None,
            fc_base_kg: todo!(),
            fc_base_kg_doc: None,
            fc_eff_array: todo!(),
            fc_eff_map: todo!(),
            fc_eff_map_doc: None,
            fc_eff_type: todo!(),
            fc_eff_type_doc: None,
            fc_kw_out_array: todo!(),
            fc_kw_per_kg: todo!(),
            fc_kw_per_kg_doc: None,
            fc_mass_kg: todo!(),
            fc_max_kw: self.fc().map(|fc| fc.pwr_out_max.get::<si::kilowatt>()).unwrap_or_default(),
            fc_max_kw_doc: None,
            fc_peak_eff_override: todo!(),
            fc_peak_eff_override_doc: None,
            fc_perc_out_array: todo!(),
            fc_pwr_out_perc: todo!(),
            fc_pwr_out_perc_doc: None,
            fc_sec_to_peak_pwr: self.fc().map(|fc| fc.pwr_ramp_lag.get::<si::second>()).unwrap_or_default(),
            fc_sec_to_peak_pwr_doc: None,
            force_aux_on_fc: todo!(),
            force_aux_on_fc_doc: None,
            frontal_area_m2: self.frontal_area.get::<si::square_meter>(),
            frontal_area_m2_doc: None,
            fs_kwh: self.fs().map(|fs| fs.energy_capacity.get::<si::kilowatt_hour>()).unwrap_or_default(),
            fs_kwh_doc: None,
            fs_kwh_per_kg: self.fs().and_then(|fs| fs.specific_energy).map(|specific_energy| specific_energy.get::<si::kilojoule_per_kilogram>() / 3600.).unwrap_or_default(),
            fs_kwh_per_kg_doc: None,
            fs_mass_kg: todo!(),
            fs_max_kw: self.fs().map(|fs| fs.pwr_out_max.get::<si::kilowatt>()).unwrap_or_default(),
            fs_max_kw_doc: None,
            fs_secs_to_peak_pwr: self.fs().map(|fs| fs.pwr_ramp_lag.get::<si::second>()).unwrap_or_default(),
            fs_secs_to_peak_pwr_doc: None,
            glider_kg: self.glider_mass.unwrap_or_default().get::<si::kilogram>(),
            glider_kg_doc: None,
            idle_fc_kw: 0.,
            idle_fc_kw_doc: None,
            input_kw_out_array: todo!(),
            kw_demand_fc_on: todo!(),
            kw_demand_fc_on_doc: None,
            large_motor_power_kw: todo!(),
            max_accel_buffer_mph: todo!(),
            max_accel_buffer_mph_doc: None,
            max_accel_buffer_perc_of_useable_soc: todo!(),
            max_accel_buffer_perc_of_useable_soc_doc: None,
            max_regen: todo!(),
            max_regen_doc: None,
            max_regen_kwh: todo!(),
            max_roadway_chg_kw: todo!(),
            max_soc: self.res().map(|res| res.max_soc.get::<si::ratio>()).unwrap_or(1.0),
            max_soc_doc: None,
            max_trac_mps2: todo!(),
            mc_eff_array: todo!(),
            mc_eff_map: todo!(),
            mc_eff_map_doc: None,
            mc_full_eff_array: todo!(),
            mc_kw_in_array: todo!(),
            mc_kw_out_array: todo!(),
            mc_mass_kg: todo!(),
            mc_max_elec_in_kw: todo!(),
            mc_max_kw: todo!(),
            mc_max_kw_doc: None,
            mc_pe_base_kg: todo!(),
            mc_pe_base_kg_doc: None,
            mc_pe_kg_per_kw: todo!(),
            mc_pe_kg_per_kw_doc: None,
            mc_peak_eff_override: todo!(),
            mc_peak_eff_override_doc: None,
            mc_perc_out_array: todo!(),
            mc_pwr_out_perc: todo!(),
            mc_pwr_out_perc_doc: None,
            mc_sec_to_peak_pwr: todo!(),
            mc_sec_to_peak_pwr_doc: None,
            min_fc_time_on: todo!(),
            min_fc_time_on_doc: None,
            min_soc: self.res().map(|res| res.min_soc.get::<si::ratio>()).unwrap_or_default(),
            min_soc_doc: None,
            modern_max: todo!(),
            mph_fc_on: todo!(),
            mph_fc_on_doc: None,
            no_elec_aux: todo!(),
            no_elec_sys: todo!(),
            num_wheels: self.num_wheels as f64,
            num_wheels_doc: None,
            orphaned: false,
            perc_high_acc_buf: todo!(),
            perc_high_acc_buf_doc: None,
            props: fastsim_2::params::RustPhysicalProperties::default(),
            regen_a: todo!(),
            regen_b: todo!(),
            scenario_name: self.name,
            selection: 0, // there is no equivalent in fastsim-3
            small_motor_power_kw: todo!(),
            stop_start: todo!(),
            stop_start_doc: None,
            trans_eff: self.trans_eff.get::<si::ratio>(),
            trans_eff_doc: None,
            trans_kg: todo!(),
            trans_kg_doc: None,
            val0_to60_mph: f64::NAN,
            val_cd_range_mi: f64::NAN,
            val_comb_kwh_per_mile: f64::NAN,
            val_comb_mpgge: f64::NAN,
            val_const45_mph_kwh_per_mile: f64::NAN,
            val_const55_mph_kwh_per_mile: f64::NAN,
            val_const60_mph_kwh_per_mile: f64::NAN,
            val_const65_mph_kwh_per_mile: f64::NAN,
            val_ess_life_miles: f64::NAN,
            val_hwy_kwh_per_mile: f64::NAN,
            val_hwy_mpgge: f64::NAN,
            val_msrp: f64::NAN,
            val_range_miles: f64::NAN,
            val_udds_kwh_per_mile: f64::NAN,
            val_udds_mpgge: f64::NAN,
            val_unadj_hwy_kwh_per_mile: f64::NAN,
            val_unadj_udds_kwh_per_mile: f64::NAN,
            val_veh_base_cost: f64::NAN,
            veh_cg_m: self.cg_height.get::<si::meter>(),
            veh_cg_m_doc: None,
            veh_kg: self.mass()?.context("Vehicle mass is `None`")?.get::<si::kilogram>(),
            veh_override_kg: self.mass()?.map(|m| m.get::<si::kilogram>()),
            veh_override_kg_doc: None,
            veh_pt_type: match &self.pt_type {
                PowertrainType::ConventionalVehicle(_) => "Conv".into(),
                PowertrainType::HybridElectricVehicle(_) => "HEV".into(),
                PowertrainType::BatteryElectricVehicle(_) => "BEV".into(),
            },
            veh_year: self.year,
            wheel_base_m: self.wheel_base.get::<si::meter>(),
            wheel_base_m_doc: None,
            wheel_coef_of_fric: todo!(),
            wheel_coef_of_fric_doc: None,
            wheel_inertia_kg_m2: self.wheel_inertia.get::<si::kilogram_square_meter>(),
            wheel_inertia_kg_m2_doc: None,
            wheel_radius_m: self.wheel_radius.get::<si::meter>(),
            wheel_radius_m_doc: None,
            wheel_rr_coef: self.wheel_rr_coef.get::<si::ratio>(),
            wheel_rr_coef_doc: None
        };
        veh.set_derived();
        Ok(veh)
    }
}

/// Vehicle state for current time step
#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, Default, EnergyMethod,
)]
#[pyo3_api]
pub struct VehicleState {
    /// time step index
    pub i: usize,

    // power and fields
    /// maximum forward propulsive power vehicle can produce
    pub pwr_tractive_max: si::Power,
    /// pwr exerted on wheels by powertrain
    pub pwr_tractive: si::Power,
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
    /// Power applied to wheel and tire inertia
    pub pwr_whl_inertia: si::Power,
    /// integral of [Self::pwr_whl_inertia]
    pub energy_whl_inertia: si::Energy,
    /// Total braking power
    pub pwr_brake: si::Power,
    /// integral of [Self::pwr_brake]
    pub energy_brake: si::Energy,
    /// integral of [Self::pwr_tractive]
    pub energy_tractive: si::Energy,
    /// whether powertrain can achieve power demand
    pub cyc_met: bool,
    /// actual achieved speed
    pub speed_ach: si::Velocity,
    /// cumulative distance traveled, integral of [Self::speed_ach]
    pub dist: si::Length,
    /// [Self::speed_ach] from previous time step
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
        veh.to_file(
            project_root::get_project_root()
                .unwrap()
                .join("tests/assets/2012_Ford_Fusion.yaml"),
        )
        .unwrap();
        #[allow(clippy::let_and_return)]
        veh
    }
}
