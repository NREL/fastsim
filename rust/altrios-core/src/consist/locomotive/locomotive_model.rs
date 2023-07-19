use super::*;

#[enum_dispatch(LocoTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub enum LocoType {
    ConventionalLoco,
    HybridLoco(Box<HybridLoco>),
    BatteryElectricLoco,
    /// Dummy locomotive with infinite power and free energy, used for
    /// working with train performance calculator with
    /// [crate::train::SetSpeedTrainSim] with no effort to ensure loads
    /// on locomotive are realistic.
    Dummy,
}

impl Default for LocoType {
    fn default() -> Self {
        Self::ConventionalLoco(Default::default())
    }
}

impl std::string::ToString for LocoType {
    fn to_string(&self) -> String {
        match self {
            LocoType::ConventionalLoco(_) => String::from("Conventional"),
            LocoType::HybridLoco(_) => String::from("Hybrid"),
            LocoType::BatteryElectricLoco(_) => String::from("Battery Electric"),
            LocoType::Dummy(_) => String::from("Dummy"),
        }
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct Dummy {}

impl LocoTrait for Dummy {
    fn set_cur_pwr_max_out(
        &mut self,
        _pwr_aux: Option<si::Power>,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        Ok(())
    }
    fn save_state(&mut self) {}
    fn step(&mut self) {}
    fn get_energy_loss(&self) -> si::Energy {
        si::Energy::ZERO
    }
}

#[altrios_api(
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    fn build_conventional_loco(
        _cls: &PyType,
        fuel_converter: FuelConverter,
        generator: Generator,
        drivetrain: ElectricDrivetrain,
        pwr_aux_offset_watts: f64,
        pwr_aux_traction_coeff_ratio: f64,
        force_max_newtons: f64,
        save_interval: Option<usize>,
    ) -> PyResult<Self> {
        let mut loco = Self {
            loco_type: LocoType::ConventionalLoco(ConventionalLoco::new(
                fuel_converter,
                generator,
                drivetrain,
            )),
            state: Default::default(),
            save_interval,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: pwr_aux_offset_watts * uc::W,
            pwr_aux_traction_coeff: pwr_aux_traction_coeff_ratio * uc::R,
            force_max: Some(force_max_newtons * uc::N),
            ..Default::default()
        };
        // make sure save_interval is propagated
        loco.set_save_interval(save_interval);
        Ok(loco)
    }

    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    fn build_hybrid_loco(
        _cls: &PyType,
        fuel_converter: FuelConverter,
        generator: Generator,
        reversible_energy_storage: ReversibleEnergyStorage,
        drivetrain: ElectricDrivetrain,
        pwr_aux_offset_watts: f64,
        pwr_aux_traction_coeff_ratio: f64,
        force_max_newtons: f64,
        fuel_res_split: Option<f64>,
        fuel_res_ratio: Option<f64>,
        gss_interval: Option<usize>,
        save_interval: Option<usize>,

    ) -> PyResult<Self> {
        let mut loco = Self {
            loco_type: LocoType::HybridLoco(Box::new(HybridLoco::new(
                fuel_converter,
                generator,
                reversible_energy_storage,
                drivetrain,
                fuel_res_split,
                fuel_res_ratio,
                gss_interval,
            ))),
            state: Default::default(),
            save_interval,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: pwr_aux_offset_watts * uc::W,
            pwr_aux_traction_coeff: pwr_aux_traction_coeff_ratio * uc::R,
            force_max: Some(force_max_newtons * uc::N),
            ..Default::default()
        };
        // make sure save_interval is propagated
        loco.set_save_interval(save_interval);
        Ok(loco)
    }

    #[classmethod]
    #[pyo3(name = "default_battery_electic_loco")]
    fn default_battery_electic_loco_py (_cls: &PyType) -> PyResult<Self> {
        Ok(Self::default_battery_electric_loco())
    }

    #[classmethod]
    fn build_battery_electric_loco (
        _cls: &PyType,
        reversible_energy_storage: ReversibleEnergyStorage,
        drivetrain: ElectricDrivetrain,
        pwr_aux_offset_watts: f64,
        pwr_aux_traction_coeff_ratio: f64,
        force_max_newtons: f64,
        save_interval: Option<usize>,

    ) -> PyResult<Self> {
        let mut loco = Self {
            loco_type: LocoType::BatteryElectricLoco(BatteryElectricLoco::new(
                reversible_energy_storage,
                drivetrain,
            )),
            state: Default::default(),
            save_interval,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: pwr_aux_offset_watts * uc::W,
            pwr_aux_traction_coeff: pwr_aux_traction_coeff_ratio * uc::R,
            force_max: Some(force_max_newtons * uc::N),
            ..Default::default()

        };
        // make sure save_interval is propagated
        loco.set_save_interval(save_interval);
        Ok(loco)
    }

    #[classmethod]
    fn build_dummy_loco(_cls: &PyType) -> Self {
        let mut dummy  = Self {
            loco_type: LocoType::Dummy(Dummy::default()),
            state: LocomotiveState::default(),
            save_interval: None,
            history: LocomotiveStateHistoryVec::new(),
            assert_limits: true,
            pwr_aux_offset: 50e3 * uc::W,
            pwr_aux_traction_coeff: 0.01 * uc::R,
            force_max: Some(50e6 * uc::N),
            ..Default::default()
        };
        dummy.update_mass(None).unwrap();
        dummy
    }

    #[getter]
    fn get_fuel_res_split(&self) -> PyResult<Option<f64>> {
        match &self.loco_type {
            LocoType::HybridLoco(loco) => Ok(Some(loco.fuel_res_split)),
            _ => Ok(None),
        }
    }

    #[getter]
    fn get_fuel_res_ratio(&self) -> PyResult<Option<f64>> {
        match &self.loco_type {
            LocoType::HybridLoco(loco) => Ok(loco.fuel_res_ratio),
            _ => Ok(None),
        }
    }

    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> PyResult<()> {
        self.set_save_interval(save_interval);
        Ok(())
    }

    #[pyo3(name = "get_save_interval")]
    /// Set save interval and cascade to nested components.
    fn get_save_interval_py(&self) -> PyResult<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[getter]
    fn get_fc(&self) -> Option<FuelConverter> {
        self.fuel_converter().cloned()
    }
    #[setter]
    fn set_fc(&mut self, _fc: FuelConverter) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

    #[setter(__fc)]
    fn set_fc_hidden(&mut self, fc: FuelConverter) -> PyResult<()> {
        self.set_fuel_converter(fc).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }
    #[getter]
    fn get_gen(&self) -> Option<Generator> {
        self.generator().cloned()
    }

    #[setter]
    fn set_gen(&mut self, _gen: Generator) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter(__gen)]
    fn set_gen_hidden(&mut self, gen: Generator) -> PyResult<()> {
        self.set_generator(gen).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }
    #[getter]
    fn get_res(&self) -> Option<ReversibleEnergyStorage> {
        self.reversible_energy_storage().cloned()
    }
    #[setter]
    fn set_res(&mut self, _res: ReversibleEnergyStorage) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }

    #[setter(__res)]
    fn set_res_hidden(&mut self, res: ReversibleEnergyStorage) -> PyResult<()> {
        self.set_reversible_energy_storage(res).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }
    #[getter]
    fn get_edrv(&self) -> Option<ElectricDrivetrain> {
        self.electric_drivetrain()
    }
    #[setter]
    fn set_edrv(&mut self, _edrv: ElectricDrivetrain) -> PyResult<()> {
        Err(PyAttributeError::new_err(DIRECT_SET_ERR))
    }
    #[setter(__edrv)]
    fn set_edrv_hidden(&mut self, edrv: ElectricDrivetrain) -> PyResult<()> {
        self.set_electric_drivetrain(edrv).map_err(|e| PyAttributeError::new_err(e.to_string()))
    }

    fn loco_type(&self) -> PyResult<String> {
        Ok(self.loco_type.to_string())
    }

    #[getter]
    fn get_pwr_rated_kilowatts(&self) -> f64 {
        self.get_pwr_rated().get::<si::kilowatt>()
    }

    #[getter("force_max_pounds")]
    fn get_force_max_pounds_py(&self) -> PyResult<Option<f64>> {
        Ok(self.force_max()?.map(|f| f.get::<si::pound_force>()))
    }

    #[getter("force_max_newtons")]
    fn get_force_max_newtons_py(&self) -> PyResult<Option<f64>> {
        Ok(
            self.force_max()?.map(|f| f.get::<si::newton>())
        )
    }

    #[getter]
    fn get_mass_kg(&self) -> PyResult<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_ballast_mass_kg(&self) -> PyResult<Option<f64>> {
        Ok(self.ballast_mass.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_baseline_mass_kg(&self) -> PyResult<Option<f64>> {
        Ok(self.baseline_mass.map(|m| m.get::<si::kilogram>()))
    }
)]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
/// Struct for simulating any type of locomotive
pub struct Locomotive {
    #[api(skip_get, skip_set)]
    /// type of locomotive including contained type-specific parameters
    /// and variables
    pub loco_type: LocoType,
    /// current state of locomotive
    #[serde(default)]
    pub state: LocomotiveState,
    #[api(skip_get, skip_set)]
    #[serde(default)]
    /// Locomotive mass
    mass: Option<si::Mass>,
    /// Locomotive coefficient of friction between wheels and rail when
    /// stopped
    #[api(skip_get, skip_set)]
    mu: Option<si::Ratio>,
    /// Ballast mass, any mass that must be added to achieve nominal
    /// locomotive weight of 432,000 lb.
    #[api(skip_get, skip_set)]
    ballast_mass: Option<si::Mass>,
    /// Baseline mass, which comprises any non-differentiating
    /// components between technologies, e.g. chassis, motors, trucks,
    /// cabin
    #[api(skip_get, skip_set)]
    baseline_mass: Option<si::Mass>,
    /// time step interval between saves.  1 is a good option.  If None,
    /// no saving occurs.
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: LocomotiveStateHistoryVec,
    #[serde(default = "utils::return_true")]
    /// If true, requires power demand to not exceed consist
    /// capabilities.  May be deprecated soon.
    pub assert_limits: bool,
    /// constant aux load
    pub pwr_aux_offset: si::Power,
    /// gain for linear model on traciton hp use to compute linear aux
    /// load
    pub pwr_aux_traction_coeff: si::Ratio,
    /// maximum tractive force
    #[api(skip_get, skip_set)]
    force_max: Option<si::Force>,
}

impl Default for Locomotive {
    fn default() -> Self {
        let mut loco = Self {
            loco_type: LocoType::ConventionalLoco(ConventionalLoco::default()),
            pwr_aux_offset: 8554.15 * uc::W, // pwr_aux_offset
            pwr_aux_traction_coeff: 0.000539638 * uc::R, // pwr_aux_traction_coeff
            force_max: None,
            state: Default::default(),
            mass: Default::default(),
            ballast_mass: Default::default(),
            baseline_mass: Default::default(),
            save_interval: Some(1),
            history: Default::default(),
            assert_limits: true,
            mu: Default::default(),
        };
        loco.update_mass(
            // Steve Fritz said 432,000 lbs is expected
            Some(432e3 * uc::LB),
        )
        .unwrap();
        loco.update_force_max(
            // 150,000 pounds of force = 667.3e3 N
            // TODO: track down source for this
            Some(667.2e3 * uc::N),
        )
        .unwrap();
        loco
    }
}

impl SerdeAPI for Locomotive {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        self.update_mass(None)?;
        Ok(())
    }
}

impl Mass for Locomotive {
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
                    self.generator_mut().map(|gen| gen.update_mass(None));
                    self.reversible_energy_storage_mut()
                        .map(|res| res.update_mass(None));
                    self.baseline_mass = None;
                    self.ballast_mass = None;
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

impl Locomotive {
    /// Sets force max based on provided value or previously set
    /// `self.mu`.
    ///
    /// Arugments:
    /// * `force_max` - option for setting `self.force_max` directly
    pub fn update_force_max(&mut self, force_max: Option<si::Force>) -> anyhow::Result<()> {
        match force_max {
            Some(force_max) => {
                self.force_max = Some(force_max);
                self.mu = self.mass.map(|mass| force_max / (mass * uc::ACC_GRAV))
            }
            // derive force_max from other parameters
            None => {
                self.force_max = match self.mu {
                    Some(mu) => match self.mass {
                        Some(mass) => Some(mass * uc::ACC_GRAV * mu),
                        None => {
                            bail!("Must set `self.mass`")
                        }
                    },
                    None => match self.mu {
                        Some(_mu) => bail!("Must set `self.mu`"),
                        None => bail!("Must set `self.mu` and `self.mass`"),
                    },
                }
            }
        };
        Ok(())
    }

    pub fn force_max(&self) -> anyhow::Result<Option<si::Force>> {
        self.check_force_max()?; // TODO: might want to copy this to `from_file` method
        Ok(self.force_max)
    }

    pub fn check_force_max(&self) -> anyhow::Result<()> {
        if let (Some(f), Some(mu), Some(mass)) = (self.force_max, self.mu, self.mass) {
            ensure!(utils::almost_eq_uom(&f, &(mu * mass * uc::ACC_GRAV), None));
        }
        Ok(())
    }

    pub fn default_battery_electric_loco() -> Self {
        // TODO: make need to add `pwr_aux_offset` and
        // `pwr_aux_traction_coeff` based on calibration
        let bel_type = LocoType::BatteryElectricLoco(BatteryElectricLoco::default());
        let mut bel = Locomotive::default();
        bel.loco_type = bel_type;
        bel
    }

    pub fn default_hybrid_electric_loco() -> Self {
        // TODO: make need to add `pwr_aux_offset` and
        // `pwr_aux_traction_coeff` based on calibration
        let hel_type = LocoType::HybridLoco(Box::default());
        let mut hel = Locomotive::default();
        hel.loco_type = hel_type;
        hel
    }

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
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.fc.save_interval = save_interval;
                loco.gen.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            LocoType::HybridLoco(loco) => {
                loco.fc.save_interval = save_interval;
                loco.gen.save_interval = save_interval;
                loco.res.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            LocoType::BatteryElectricLoco(loco) => {
                loco.res.save_interval = save_interval;
                loco.edrv.save_interval = save_interval;
            }
            LocoType::Dummy(_) => { /* maybe return an error for this in the future */ }
        }
    }

    pub fn fuel_converter(&self) -> Option<&FuelConverter> {
        match &self.loco_type {
            LocoType::ConventionalLoco(loco) => Some(&loco.fc),
            LocoType::HybridLoco(loco) => Some(&loco.fc),
            LocoType::BatteryElectricLoco(_) => None,
            LocoType::Dummy(_) => None,
        }
    }

    pub fn fuel_converter_mut(&mut self) -> Option<&mut FuelConverter> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => Some(&mut loco.fc),
            LocoType::HybridLoco(loco) => Some(&mut loco.fc),
            LocoType::BatteryElectricLoco(_) => None,
            LocoType::Dummy(_) => None,
        }
    }

    pub fn set_fuel_converter(&mut self, fc: FuelConverter) -> Result<()> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.fc = fc;
                Ok(())
            }
            LocoType::HybridLoco(loco) => {
                loco.fc = fc;
                Ok(())
            }
            LocoType::BatteryElectricLoco(_) => bail!("BEL has no FuelConverter."),
            LocoType::Dummy(_) => bail!("Dummy locomotive has no FuelConverter."),
        }
    }

    pub fn generator(&self) -> Option<&Generator> {
        match &self.loco_type {
            LocoType::ConventionalLoco(loco) => Some(&loco.gen),
            LocoType::HybridLoco(loco) => Some(&loco.gen),
            LocoType::BatteryElectricLoco(_) => None,
            LocoType::Dummy(_) => None,
        }
    }

    pub fn generator_mut(&mut self) -> Option<&mut Generator> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => Some(&mut loco.gen),
            LocoType::HybridLoco(loco) => Some(&mut loco.gen),
            LocoType::BatteryElectricLoco(_) => None,
            LocoType::Dummy(_) => None,
        }
    }

    pub fn set_generator(&mut self, gen: Generator) -> Result<()> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.gen = gen;
                Ok(())
            }
            LocoType::HybridLoco(loco) => {
                loco.gen = gen;
                Ok(())
            }
            LocoType::BatteryElectricLoco(_) => bail!("BEL has no Generator."),
            LocoType::Dummy(_) => bail!("Dummy locomotive has no Generator."),
        }
    }

    pub fn reversible_energy_storage(&self) -> Option<&ReversibleEnergyStorage> {
        match &self.loco_type {
            LocoType::ConventionalLoco(_) => None,
            LocoType::HybridLoco(loco) => Some(&loco.res),
            LocoType::BatteryElectricLoco(loco) => Some(&loco.res),
            LocoType::Dummy(_) => None,
        }
    }

    pub fn reversible_energy_storage_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(_) => None,
            LocoType::HybridLoco(loco) => Some(&mut loco.res),
            LocoType::BatteryElectricLoco(loco) => Some(&mut loco.res),
            LocoType::Dummy(_) => None,
        }
    }

    pub fn set_reversible_energy_storage(&mut self, res: ReversibleEnergyStorage) -> Result<()> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(_) => {
                bail!("Conventional has no ReversibleEnergyStorage.")
            }
            LocoType::HybridLoco(loco) => {
                loco.res = res;
                Ok(())
            }
            LocoType::BatteryElectricLoco(loco) => {
                loco.res = res;
                Ok(())
            }
            LocoType::Dummy(_) => bail!("Dummy locomotive has no RES."),
        }
    }

    pub fn electric_drivetrain(&self) -> Option<ElectricDrivetrain> {
        match &self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            LocoType::HybridLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            LocoType::BatteryElectricLoco(loco) => {
                let edrv = loco.edrv.clone();
                Some(edrv)
            }
            LocoType::Dummy(_) => None,
        }
    }

    pub fn set_electric_drivetrain(&mut self, edrv: ElectricDrivetrain) -> Result<()> {
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            LocoType::HybridLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            LocoType::BatteryElectricLoco(loco) => {
                loco.edrv = edrv;
                Ok(())
            }
            LocoType::Dummy(_) => bail!("Dummy locomotive has no ElectricDrivetrain."),
        }
    }

    /// Calculate mass from components.
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        if let (Some(baseline), Some(ballast)) = (self.baseline_mass, self.ballast_mass) {
            match self.loco_type {
                LocoType::ConventionalLoco(_) => {
                    if let (Some(fc), Some(gen)) = (
                        self.fuel_converter().unwrap().mass()?,
                        self.generator().unwrap().mass()?,
                    ) {
                        Ok(Some(fc + gen + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `fc` and `gen` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::HybridLoco(_) => {
                    if let (Some(fc), Some(gen), Some(res)) = (
                        self.fuel_converter().unwrap().mass()?,
                        self.generator().unwrap().mass()?,
                        self.reversible_energy_storage().unwrap().mass()?,
                    ) {
                        Ok(Some(fc + gen + res + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `fc`, `gen`, and `res` masses must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::BatteryElectricLoco(_) => {
                    if let Some(res) = self.reversible_energy_storage().unwrap().mass()? {
                        Ok(Some(res + baseline + ballast))
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both specified\n{}\n{}",
                            "so `res` mass must also be specified.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::Dummy(_) => {
                    bail!(
                        "`baseline` and `ballast` mass must be `None` with Dummy locomotive.\n{}",
                        format_dbg!()
                    )
                }
            }
        } else if self.baseline_mass.is_none() && self.ballast_mass.is_none() {
            match self.loco_type {
                LocoType::ConventionalLoco(_) => {
                    if self.fuel_converter().unwrap().mass()?.is_none()
                        && self.generator().unwrap().mass()?.is_none()
                    {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `fc` and `gen` masses must also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::HybridLoco(_) => {
                    if self.fuel_converter().unwrap().mass()?.is_none()
                        && self.generator().unwrap().mass()?.is_none()
                        && self.reversible_energy_storage().unwrap().mass()?.is_none()
                    {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `fc`, `gen`, and `res` masses must also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::BatteryElectricLoco(_) => {
                    if self.reversible_energy_storage().unwrap().mass()?.is_none() {
                        Ok(None)
                    } else {
                        bail!(
                            "Locomotive fields baseline and ballast masses are both `None`\n{}\n{}",
                            "so `res` mass must also also be `None`.",
                            format_dbg!()
                        )
                    }
                }
                LocoType::Dummy(_) => Ok(Some(0.0 * uc::KG)),
            }
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
        match &mut self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.solve_energy_consumption(
                    pwr_out_req,
                    dt,
                    engine_on.unwrap_or(true),
                    self.state.pwr_aux,
                    self.assert_limits,
                )?;
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            LocoType::HybridLoco(loco) => {
                loco.solve_energy_consumption(pwr_out_req, dt, self.assert_limits)?;
                // TODO: add `engine_on` and `pwr_aux` here as inputs
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            LocoType::BatteryElectricLoco(loco) => {
                //todo: put something in hear for deep sleep that is the
                //equivalent of engine_on in conventional loco
                loco.solve_energy_consumption(pwr_out_req, dt, self.state.pwr_aux)?;
                self.state.pwr_out =
                    loco.edrv.state.pwr_mech_prop_out - loco.edrv.state.pwr_mech_dyn_brake;
            }
            LocoType::Dummy(_) => { /* maybe put an error error in the future */ }
        }
        self.state.energy_out += self.state.pwr_out * dt;
        self.state.energy_aux += self.state.pwr_aux * dt;

        Ok(())
    }

    pub fn set_pwr_aux(&mut self, engine_on: Option<bool>) {
        self.state.pwr_aux = if engine_on.unwrap_or(true) {
            self.pwr_aux_offset + self.pwr_aux_traction_coeff * self.state.pwr_out.abs()
        } else {
            si::Power::ZERO
        };
    }
}

fn set_pwr_lims(state: &mut LocomotiveState, edrv: &ElectricDrivetrain) {
    state.pwr_out_max = edrv.state.pwr_mech_out_max;
    state.pwr_rate_out_max = edrv.state.pwr_rate_out_max;
    state.pwr_regen_max = edrv.state.pwr_mech_regen_max;
}

impl LocoTrait for Locomotive {
    fn step(&mut self) {
        self.loco_type.step();
        self.state.i += 1;
    }

    fn save_state(&mut self) {
        self.loco_type.save_state();
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || self.state.i == 1 {
                self.history.push(self.state);
            }
        }
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.loco_type.get_energy_loss()
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

        self.loco_type
            .set_cur_pwr_max_out(Some(self.state.pwr_aux), dt)?;
        match &self.loco_type {
            LocoType::ConventionalLoco(loco) => {
                // TODO: Coordinate with Geordie on the rate
                set_pwr_lims(&mut self.state, &loco.edrv);
                assert_eq!(self.state.pwr_regen_max, si::Power::ZERO);
            }
            LocoType::HybridLoco(loco) => {
                set_pwr_lims(&mut self.state, &loco.edrv);
                // TODO: Coordinate with Geordie on rate
            }
            LocoType::BatteryElectricLoco(loco) => {
                set_pwr_lims(&mut self.state, &loco.edrv);
                // TODO: Coordinate with Geordie on rate; INCOMPLETE ON
                // RATE (Jinghu as of 06/06/2022)
            }
            LocoType::Dummy(_) => {
                // this locomotive has the power of 1,000 suns and more
                // power absorption ability than really big numbers that
                // are not inf to avoid null in json
                self.state.pwr_out_max = uc::W * 1e15;
                self.state.pwr_rate_out_max = uc::WPS * 1e15;
                self.state.pwr_regen_max = uc::W * 1e15;
            }
        }
        Ok(())
    }
}

/// Locomotive state for current time step
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct LocomotiveState {
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

impl Default for LocomotiveState {
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
