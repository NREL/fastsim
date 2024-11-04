use super::{hev::HEVPowertrainControls, *};

/// Possible aux load power sources
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

    // despite having `setter` here, this seems to work as a function
    #[setter("save_interval")]
    fn set_save_interval_py(&mut self, _save_interval: Option<usize>) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

    #[setter("__save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_hidden(&mut self, save_interval: Option<usize>) -> PyResult<()> {
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
    #[setter("fc")]
    fn set_fc_py(&mut self, _fc: FuelConverter) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter("__fc")]
    fn set_fc_hidden(&mut self, fc: FuelConverter) -> PyResult<()> {
        self.set_fc(fc).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }

    #[getter]
    fn get_res(&self) -> Option<ReversibleEnergyStorage> {
        self.res().cloned()
    }
    #[setter("res")]
    fn set_res_py(&mut self, _res: ReversibleEnergyStorage) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter("__res")]
    fn set_res_hidden(&mut self, res: ReversibleEnergyStorage) -> PyResult<()> {
        self.set_res(res).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }

    #[getter]
    fn get_em(&self) -> Option<ElectricMachine> {
        self.em().cloned()
    }

    #[setter("em")]
    fn set_em_py(&mut self, _em: ElectricMachine) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter("__em")]
    fn set_em_hidden(&mut self, em: ElectricMachine) -> PyResult<()> {
        self.set_em(em).map_err(|e| PyAttributeError::new_err(e.to_string()))
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

    #[getter("pt_type_json")]
    fn get_pt_type_json_py(&self) -> anyhow::Result<String >{
        self.pt_type.to_str("json")
    }
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

    /// Chassis model with various chassis-related parameters
    pub chassis: Chassis,

    /// Total vehicle mass
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mass: Option<si::Mass>,

    /// power required by auxilliary systems (e.g. HVAC, stereo)  
    /// TODO: make this an enum to allow for future variations
    pub pwr_aux: si::Power,

    /// transmission efficiency
    // TODO: make `transmission::{Transmission, TransmissionState}` and
    // `Transmission` should have field `efficency: Efficiency`.
    pub trans_eff: si::Ratio,

    /// time step interval at which `state` is saved into `history`
    #[api(skip_set, skip_get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    save_interval: Option<usize>,
    /// current state of vehicle
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: VehicleState,
    /// Vector-like history of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "VehicleStateHistoryVec::is_empty")]
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

impl TryFrom<&fastsim_2::vehicle::RustVehicle> for PowertrainType {
    type Error = anyhow::Error;
    /// Returns fastsim-3 vehicle given fastsim-2 vehicle
    ///
    /// # Arguments
    /// * `f2veh` - fastsim-2 vehicle
    fn try_from(f2veh: &fastsim_2::vehicle::RustVehicle) -> anyhow::Result<PowertrainType> {
        // TODO: implement the `_doc` fields in fastsim-3 and make sure they get carried over from fastsim-2
        // see https://github.com/NREL/fastsim/blob/fastsim-2/rust/fastsim-core/fastsim-proc-macros/src/doc_field.rs and do something similar
        match f2veh.veh_pt_type.as_str() {
            CONV => {
                let conv = ConventionalVehicle {
                    fs: {
                        let mut fs = FuelStorage {
                            pwr_out_max: f2veh.fs_max_kw * uc::KW,
                            pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
                            energy_capacity: f2veh.fs_kwh * 3.6 * uc::MJ,
                            specific_energy: Some(FUEL_LHV_MJ_PER_KG * uc::MJ / uc::KG),
                            mass: None,
                        };
                        fs.set_mass(None, MassSideEffect::None)
                            .with_context(|| anyhow!(format_dbg!()))?;
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
                            eff_interp_from_pwr_out: Interpolator::Interp1D(Interp1D::new(
                                f2veh.fc_perc_out_array.to_vec(),
                                f2veh.fc_eff_array.to_vec(),
                                Strategy::LeftNearest,
                                Extrapolate::Error,
                            )?),
                            pwr_for_peak_eff: uc::KW * f64::NAN, // this gets updated in `init`
                            // this means that aux power must include idle fuel
                            pwr_idle_fuel: si::Power::ZERO,
                            save_interval: Some(1),
                            history: Default::default(),
                            _phantom: PhantomData,
                        };
                        fc.init()?;
                        fc.set_mass(None, MassSideEffect::None)
                            .with_context(|| anyhow!(format_dbg!()))?;
                        fc
                    },
                    transmission: Transmission {
                        mass: None,
                        eff_interp: Interpolator::Interp0D(f2veh.trans_eff),
                        save_interval: Some(1),
                        state: Default::default(),
                        history: Default::default(),
                    },
                    mass: None,
                    alt_eff: f2veh.alt_eff * uc::R,
                };
                Ok(PowertrainType::ConventionalVehicle(Box::new(conv)))
            }
            HEV => {
                let pt_cntrl = HEVPowertrainControls::Fastsim2(hev::RESGreedyWithDynamicBuffers {
                    speed_soc_fc_on_buffer: None,
                    speed_soc_accel_buffer: None,
                    speed_soc_accel_buffer_coeff: None,
                    speed_soc_regen_buffer: None,
                    speed_soc_regen_buffer_coeff: None,
                    // note that this exists in `fastsim-2` but has no apparent effect!
                    fc_min_time_on: None,
                    speed_fc_forced_on: Some(f2veh.mph_fc_on * uc::MPH),
                    frac_pwr_demand_fc_forced_on: Some(
                        f2veh.kw_demand_fc_on
                            / (f2veh.fc_max_kw + f2veh.ess_max_kw.min(f2veh.mc_max_kw))
                            * uc::R,
                    ),
                    frac_of_most_eff_pwr_to_run_fc: None,
                    // TODO: make sure these actually do something, if deemed worthwhile
                    frac_res_chrg_for_fc: f2veh.ess_chg_to_fc_max_eff_perc * uc::R,
                    frac_res_dschrg_for_fc: f2veh.ess_dischg_to_fc_max_eff_perc * uc::R,
                });
                let mut hev = HybridElectricVehicle {
                    fs: {
                        let mut fs = FuelStorage {
                            pwr_out_max: f2veh.fs_max_kw * uc::KW,
                            pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
                            energy_capacity: f2veh.fs_kwh * 3.6 * uc::MJ,
                            specific_energy: None,
                            mass: None,
                        };
                        fs.set_mass(None, MassSideEffect::None)
                            .with_context(|| anyhow!(format_dbg!()))?;
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
                            eff_interp_from_pwr_out: Interpolator::Interp1D(Interp1D::new(
                                f2veh.fc_perc_out_array.to_vec(),
                                f2veh.fc_eff_array.to_vec(),
                                Strategy::LeftNearest,
                                Extrapolate::Error,
                            )?),
                            pwr_for_peak_eff: uc::KW * f64::NAN, // this gets updated in `init`
                            // this means that aux power must include idle fuel
                            pwr_idle_fuel: si::Power::ZERO,
                            save_interval: Some(1),
                            history: Default::default(),
                            _phantom: PhantomData,
                        };
                        fc.init()?;
                        fc.set_mass(None, MassSideEffect::None)
                            .with_context(|| anyhow!(format_dbg!()))?;
                        fc
                    },
                    res: ReversibleEnergyStorage {
                        state: Default::default(),
                        mass: None,
                        specific_energy: None,
                        pwr_out_max: f2veh.ess_max_kw * uc::KW,
                        energy_capacity: f2veh.ess_max_kwh * uc::KWH,
                        eff_interp: Interpolator::Interp0D(f2veh.ess_round_trip_eff.sqrt()),
                        min_soc: f2veh.min_soc * uc::R,
                        max_soc: f2veh.max_soc * uc::R,
                        save_interval: Some(1),
                        history: Default::default(),
                    },
                    em: ElectricMachine {
                        state: Default::default(),
                        eff_interp_fwd: (Interpolator::Interp1D(
                            Interp1D::new(
                                f2veh.mc_perc_out_array.to_vec(),
                                {
                                    let mut mc_full_eff_vec = f2veh.mc_full_eff_array.to_vec();
                                    ensure!(mc_full_eff_vec.len() > 1);
                                    mc_full_eff_vec[0] = mc_full_eff_vec[1];
                                    mc_full_eff_vec
                                },
                                Strategy::LeftNearest,
                                Extrapolate::Error,
                            )
                            .unwrap(),
                        )),
                        eff_interp_at_max_input: None,
                        // pwr_in_frac_interp: Default::default(),
                        pwr_out_max: f2veh.mc_max_kw * uc::KW,
                        specific_pwr: None,
                        mass: None,
                        save_interval: Some(1),
                        history: Default::default(),
                    },
                    pt_cntrl,
                    mass: None,
                    sim_params: Default::default(),
                    aux_cntrl: Default::default(),
                    state: Default::default(),
                    history: Default::default(),
                    soc_bal_iter_history: Default::default(),
                };
                hev.init()?;
                Ok(PowertrainType::HybridElectricVehicle(Box::new(hev)))
            }
            BEV => {
                let bev = BatteryElectricVehicle {
                    res: ReversibleEnergyStorage {
                        state: Default::default(),
                        mass: None,
                        specific_energy: None,
                        pwr_out_max: f2veh.ess_max_kw * uc::KW,
                        energy_capacity: f2veh.ess_max_kwh * uc::KWH,
                        eff_interp: Interpolator::Interp0D(f2veh.ess_round_trip_eff.sqrt()),
                        min_soc: f2veh.min_soc * uc::R,
                        max_soc: f2veh.max_soc * uc::R,
                        save_interval: Some(1),
                        history: Default::default(),
                    },
                    em: ElectricMachine {
                        state: Default::default(),
                        eff_interp_fwd: (Interpolator::Interp1D(Interp1D::new(
                            f2veh.mc_pwr_out_perc.to_vec(),
                            f2veh.mc_eff_array.to_vec(),
                            Strategy::LeftNearest,
                            Extrapolate::Error,
                        )?)),
                        eff_interp_at_max_input: Some(Interpolator::Interp1D(Interp1D::new(
                            // before adding the interpolator, pwr_in_frac_interp was set as Default::default(), can this
                            // be transferred over as done here, or does a new defualt need to be defined?
                            f2veh
                                .mc_pwr_out_perc
                                .to_vec()
                                .iter()
                                .zip(f2veh.mc_eff_array.to_vec().iter())
                                .map(|(x, y)| x / y)
                                .collect(),
                            f2veh.mc_eff_array.to_vec(),
                            Strategy::LeftNearest,
                            Extrapolate::Error,
                        )?)),
                        pwr_out_max: f2veh.mc_max_kw * uc::KW,
                        specific_pwr: None,
                        mass: None,
                        save_interval: Some(1),
                        history: Default::default(),
                    },
                    transmission: Transmission {
                        mass: None,
                        eff_interp: Interpolator::Interp0D(f2veh.trans_eff),
                        save_interval: Some(1),
                        state: Default::default(),
                        history: Default::default(),
                    },
                    mass: None,
                };
                Ok(PowertrainType::BatteryElectricVehicle(Box::new(bev)))
            }
            _ => {
                bail!(
                    "Invalid powertrain type: {}.
Expected one of {}",
                    f2veh.veh_pt_type,
                    [CONV, HEV, PHEV, BEV].join(", "),
                )
            }
        }
    }
}

impl TryFrom<fastsim_2::vehicle::RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_2::vehicle::RustVehicle) -> anyhow::Result<Self> {
        let mut f2veh = f2veh.clone();
        f2veh
            .set_derived()
            .with_context(|| anyhow!(format_dbg!()))?;
        let save_interval = Some(1);
        let pt_type = PowertrainType::try_from(&f2veh).with_context(|| anyhow!(format_dbg!()))?;

        let mut f3veh = Self {
            name: f2veh.scenario_name.clone(),
            year: f2veh.veh_year,
            pt_type,
            chassis: Chassis::try_from(&f2veh)?,
            pwr_aux: f2veh.aux_kw * uc::KW,
            trans_eff: f2veh.trans_eff * uc::R,
            state: Default::default(),
            save_interval,
            history: Default::default(),
            mass: Some(f2veh.veh_kg * uc::KG),
        };
        f3veh.expunge_mass_fields();
        f3veh.init().with_context(|| anyhow!(format_dbg!()))?;

        Ok(f3veh)
    }
}

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
        self.state.dist += self.state.speed_ach * dt;
    }
}

impl Vehicle {
    // TODO: run this assumption by Robin: peak power of all components can be produced concurrently.
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
        // TODO: do something more sophisticated with pwr_aux
        self.state.pwr_aux = self.pwr_aux;
        self.pt_type
            .solve(
                self.state.pwr_tractive,
                &self.state,
                true, // `enabled` should always be true at the powertrain level
                dt,
            )
            .with_context(|| anyhow!(format_dbg!()))?;
        self.state.pwr_brake =
            -self.state.pwr_tractive.max(si::Power::ZERO) - self.pt_type.pwr_regen();
        Ok(())
    }

    pub fn set_curr_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<()> {
        // TODO: when a fancier model for `pwr_aux` is implemented, put it here
        // TODO: make transmission field in vehicle and make it be able to produce an efficiency
        // TODO: account for traction limits here

        self.pt_type
            .set_curr_pwr_prop_out_max(self.pwr_aux, dt, &self.state)
            .with_context(|| anyhow!(format_dbg!()))?;

        (self.state.pwr_prop_fwd_max, self.state.pwr_prop_bwd_max) = self
            .pt_type
            .get_curr_pwr_prop_out_max()
            .with_context(|| anyhow!(format_dbg!()))?;

        Ok(())
    }

    pub fn to_fastsim2(&self) -> anyhow::Result<fastsim_2::vehicle::RustVehicle> {
        let mut veh = fastsim_2::vehicle::RustVehicle {
            alt_eff: match &self.pt_type {
                PowertrainType::ConventionalVehicle(conv) => conv.alt_eff.get::<si::ratio>(),
                _ => 1.0,
            },
            alt_eff_doc: None,
            aux_kw: self.pwr_aux.get::<si::kilowatt>(),
            aux_kw_doc: None,
            cargo_kg: self
                .chassis
                .cargo_mass
                .unwrap_or_default()
                .get::<si::kilogram>(),
            cargo_kg_doc: None,
            charging_on: false,
            chg_eff: 0.86, // TODO: revisit?
            chg_eff_doc: None,
            comp_mass_multiplier: 1.4,
            comp_mass_multiplier_doc: None,
            // TODO: replace with `doc` field once implemented in fastsim-3
            doc: None,
            drag_coef: self.chassis.drag_coef.get::<si::ratio>(),
            drag_coef_doc: None,
            drive_axle_weight_frac: self.chassis.drive_axle_weight_frac.get::<si::ratio>(),
            drive_axle_weight_frac_doc: None,
            ess_base_kg: 75.0, // TODO: revisit
            ess_base_kg_doc: None,
            ess_chg_to_fc_max_eff_perc: 0.0, // TODO: ??? update later
            ess_chg_to_fc_max_eff_perc_doc: None,
            ess_dischg_to_fc_max_eff_perc: 0.0, // TODO: ??? update later
            ess_dischg_to_fc_max_eff_perc_doc: None,
            ess_kg_per_kwh: 8.0, // TODO: revisit
            ess_kg_per_kwh_doc: None,
            ess_life_coef_a: 110.,
            ess_life_coef_a_doc: None,
            ess_life_coef_b: -0.6811,
            ess_life_coef_b_doc: None,
            ess_mass_kg: self.res().map_or(anyhow::Ok(0.), |res| {
                Ok(res.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            ess_max_kw: self
                .res()
                .map(|res| res.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            ess_max_kw_doc: None,
            ess_max_kwh: self
                .res()
                .map(|res| res.energy_capacity.get::<si::kilowatt_hour>())
                .unwrap_or_default(),
            ess_max_kwh_doc: None,
            ess_round_trip_eff: self
                .res()
                .map(|res| {
                    if let Interpolator::Interp0D(eff) = res.eff_interp {
                        Ok(eff.powi(2))
                    } else {
                        bail!("`to_fastsim2` is not implemented for non-0D `res.eff_interp`")
                    }
                })
                .transpose()?
                .unwrap_or(f64::NAN),
            ess_round_trip_eff_doc: None,
            ess_to_fuel_ok_error: 0.005, // TODO: update when hybrid logic is implemented
            ess_to_fuel_ok_error_doc: None,
            fc_base_kg: 61.0, // TODO: revisit
            fc_base_kg_doc: None,
            fc_eff_array: Default::default(),
            fc_eff_map: self
                .fc()
                .map(|fc| match &fc.eff_interp_from_pwr_out {
                    utils::interp::Interpolator::Interp1D(_interp1d) => {
                        Ok(fc.eff_interp_from_pwr_out.f_x()?.clone().into())
                    }
                    _ => bail!(
                        "{}\nOnly 1-D interpolators can be converted to FASTSim 2",
                        format_dbg!()
                    ),
                })
                .transpose()?
                .unwrap_or_else(|| array![0., 0.]),
            fc_eff_map_doc: None,
            fc_eff_type: "SI".into(), // TODO: placeholder, revisit and update if needed
            fc_eff_type_doc: None,
            fc_kw_out_array: Default::default(),
            fc_kw_per_kg: 2.13, // TODO: revisit
            fc_kw_per_kg_doc: None,
            fc_mass_kg: self.fc().map_or(anyhow::Ok(0.), |fc| {
                Ok(fc.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            fc_max_kw: self
                .fc()
                .map(|fc| fc.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            fc_max_kw_doc: None,
            fc_peak_eff_override: None,
            fc_peak_eff_override_doc: None,
            fc_perc_out_array: Default::default(),
            fc_pwr_out_perc: self
                .fc()
                .map(|fc| match &fc.eff_interp_from_pwr_out {
                    utils::interp::Interpolator::Interp1D(_interp) => {
                        Ok(fc.eff_interp_from_pwr_out.x()?.clone().into())
                    }
                    _ => bail!(
                        "{}\nOnly 1-D interpolators can be converted to FASTSim 2",
                        format_dbg!()
                    ),
                })
                .transpose()?
                .unwrap_or_else(|| array![0., 1.]),
            fc_pwr_out_perc_doc: None,
            fc_sec_to_peak_pwr: self
                .fc()
                .map(|fc| fc.pwr_ramp_lag.get::<si::second>())
                .unwrap_or_default(),
            fc_sec_to_peak_pwr_doc: None,
            force_aux_on_fc: matches!(self.pt_type, PowertrainType::ConventionalVehicle(_)),
            force_aux_on_fc_doc: None,
            frontal_area_m2: self.chassis.frontal_area.get::<si::square_meter>(),
            frontal_area_m2_doc: None,
            fs_kwh: self
                .fs()
                .map(|fs| fs.energy_capacity.get::<si::kilowatt_hour>())
                .unwrap_or_default(),
            fs_kwh_doc: None,
            fs_kwh_per_kg: self
                .fs()
                .and_then(|fs| fs.specific_energy)
                .map(|specific_energy| specific_energy.get::<si::kilojoule_per_kilogram>() / 3600.)
                .unwrap_or_default(),
            fs_kwh_per_kg_doc: None,
            fs_mass_kg: self.fs().map_or(anyhow::Ok(0.), |fs| {
                Ok(fs.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            fs_max_kw: self
                .fs()
                .map(|fs| fs.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            fs_max_kw_doc: None,
            fs_secs_to_peak_pwr: self
                .fs()
                .map(|fs| fs.pwr_ramp_lag.get::<si::second>())
                .unwrap_or_default(),
            fs_secs_to_peak_pwr_doc: None,
            glider_kg: self
                .chassis
                .glider_mass
                .unwrap_or_default()
                .get::<si::kilogram>(),
            glider_kg_doc: None,
            idle_fc_kw: 0.,
            idle_fc_kw_doc: None,
            input_kw_out_array: Default::default(), // calculated in `set_derived()`
            kw_demand_fc_on: 100.0,                 // TODO: placeholder, revisit
            kw_demand_fc_on_doc: None,
            large_motor_power_kw: 75.0,
            max_accel_buffer_mph: 60.0, // TODO: placeholder, revisit
            max_accel_buffer_mph_doc: None,
            max_accel_buffer_perc_of_useable_soc: 0.2, // TODO: placeholder, revisit
            max_accel_buffer_perc_of_useable_soc_doc: None,
            max_regen: 0.98, // TODO: placeholder, revisit
            max_regen_doc: None,
            max_regen_kwh: Default::default(),
            max_roadway_chg_kw: Default::default(),
            max_soc: self
                .res()
                .map(|res| res.max_soc.get::<si::ratio>())
                .unwrap_or_else(|| 1.0),
            max_soc_doc: None,
            max_trac_mps2: Default::default(),
            mc_eff_array: Default::default(), // calculated in `set_derived`
            mc_eff_map: self
                .em()
                .map(|em| em.eff_interp_fwd.f_x())
                .transpose()?
                .unwrap_or_else(|| vec![0., 1.])
                .into(),
            mc_eff_map_doc: None,
            mc_full_eff_array: Default::default(), // calculated in `set_derived`
            mc_kw_in_array: Default::default(),    // calculated in `set_derived`
            mc_kw_out_array: Default::default(),   // calculated in `set_derived`
            mc_mass_kg: self.em().map_or(anyhow::Ok(0.), |em| {
                Ok(em.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            mc_max_elec_in_kw: Default::default(), // calculated in `set_derived`
            mc_max_kw: self
                .em()
                .map(|em| em.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            mc_max_kw_doc: None,
            mc_pe_base_kg: 0.0, // placeholder, TODO: review when implementing xEVs
            mc_pe_base_kg_doc: None,
            mc_pe_kg_per_kw: 0.833, // placeholder, TODO: review when implementing xEVs
            mc_pe_kg_per_kw_doc: None,
            mc_peak_eff_override: Default::default(),
            mc_peak_eff_override_doc: None,
            mc_perc_out_array: Default::default(),
            // short array that can use xEV when implented.  TODO: fix this when implementing xEV
            mc_pwr_out_perc: self
                .em()
                .map(|em| em.eff_interp_fwd.x())
                .transpose()?
                .unwrap_or_else(|| vec![0., 1.])
                .into(),
            mc_pwr_out_perc_doc: None,
            mc_sec_to_peak_pwr: Default::default(), // placeholder, TODO: revisit when implementing xEVs
            mc_sec_to_peak_pwr_doc: None,
            min_fc_time_on: 30.0, // TODO: implement this when doing HEV
            min_fc_time_on_doc: None,
            min_soc: self
                .res()
                .map(|res| res.min_soc.get::<si::ratio>())
                .unwrap_or_default(),
            min_soc_doc: None,
            modern_max: 0.95,
            // TODO: revisit when implemementing HEV
            mph_fc_on: 70.0,
            mph_fc_on_doc: None,
            no_elec_aux: false, // TODO: revisit when implemementing HEV
            no_elec_sys: false, // TODO: revisit when implemementing HEV
            num_wheels: self.chassis.num_wheels as f64,
            num_wheels_doc: None,
            orphaned: false,
            perc_high_acc_buf: Default::default(), // TODO: revisit when implemementing HEV
            perc_high_acc_buf_doc: None,
            props: fastsim_2::params::RustPhysicalProperties::default(),
            regen_a: 500.0, //TODO: placeholder
            regen_b: 0.99,  //TODO: placeholder
            scenario_name: self.name.clone(),
            selection: 0, // there is no equivalent in fastsim-3
            small_motor_power_kw: 7.5,
            stop_start: false, // TODO: revisit when implemementing mild hybrids and stop/start vehicles
            stop_start_doc: None,
            trans_eff: self.trans_eff.get::<si::ratio>(),
            trans_eff_doc: None,
            trans_kg: 114.0, // TODO: replace with actual transmission mass
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
            veh_cg_m: self.chassis.cg_height.get::<si::meter>()
                * match self.chassis.drive_type {
                    chassis::DriveTypes::FWD => 1.0,
                    chassis::DriveTypes::RWD
                    | chassis::DriveTypes::AWD
                    | chassis::DriveTypes::FourWD => -1.0,
                },
            veh_cg_m_doc: None,
            veh_kg: self
                .mass()?
                .context("Vehicle mass is `None`")?
                .get::<si::kilogram>(),
            veh_override_kg: self.mass()?.map(|m| m.get::<si::kilogram>()),
            veh_override_kg_doc: None,
            veh_pt_type: match &self.pt_type {
                PowertrainType::ConventionalVehicle(_) => "Conv".into(),
                PowertrainType::HybridElectricVehicle(_) => "HEV".into(),
                PowertrainType::BatteryElectricVehicle(_) => "BEV".into(),
            },
            veh_year: self.year,
            wheel_base_m: self.chassis.wheel_base.get::<si::meter>(),
            wheel_base_m_doc: None,
            wheel_coef_of_fric: self.chassis.wheel_fric_coef.get::<si::ratio>(),
            wheel_coef_of_fric_doc: None,
            wheel_inertia_kg_m2: self
                .chassis
                .wheel_inertia
                .get::<si::kilogram_square_meter>(),
            wheel_inertia_kg_m2_doc: None,
            wheel_radius_m: self.chassis.wheel_radius.unwrap().get::<si::meter>(),
            wheel_radius_m_doc: None,
            wheel_rr_coef: self.chassis.wheel_rr_coef.get::<si::ratio>(),
            wheel_rr_coef_doc: None,
        };
        veh.set_derived().with_context(|| anyhow!(format_dbg!()))?;
        Ok(veh)
    }
}

/// Vehicle state for current time step
#[fastsim_api]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
pub struct VehicleState {
    /// time step index
    pub i: usize,

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
    /// integral of [Self::pwr_out]
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
            grade_curr: si::Ratio::ZERO,
            air_density: crate::air_properties::get_density_air(None, None),
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
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents).unwrap();
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
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents).unwrap();
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
            let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents).unwrap();
            let veh = Vehicle::try_from(f2veh);
            veh.unwrap()
        };

        // veh.to_file(vehicles_dir().join("2022_Renault_Zoe_ZE50_R135.yaml"))
        //     .unwrap();
        assert!(veh.pt_type.is_battery_electric_vehicle());
        veh
    }

    /// tests that vehicle can be initialized and that repeating has no net effect
    // TODO: fix this test from the python side.  Use `deepdiff` or some such
    // #[test]
    // #[cfg(feature = "yaml")]
    // pub(crate) fn test_conv_veh_init() {
    //     let veh = mock_conv_veh();
    //     let mut veh1 = veh.clone();
    //     assert!(veh == veh1);
    //     veh1.init().unwrap();
    //     assert!(veh == veh1);
    // }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_conv() {
        let veh = mock_conv_veh();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive {
            veh,
            cyc,
            sim_params: Default::default(),
        };
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_hev() {
        let veh = mock_hev();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive {
            veh,
            cyc,
            sim_params: Default::default(),
        };
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }

    #[test]
    #[cfg(all(feature = "csv", feature = "resources"))]
    fn test_to_fastsim2_bev() {
        let veh = mock_bev();
        let cyc = crate::drive_cycle::Cycle::from_resource("udds.csv", false).unwrap();
        let sd = crate::simdrive::SimDrive {
            veh,
            cyc,
            sim_params: Default::default(),
        };
        let mut sd2 = sd.to_fastsim2().unwrap();
        sd2.sim_drive(None, None).unwrap();
    }
}
