use super::*;

#[altrios_api(
    #[new]
    fn __new__(
        loco_vec: Vec<Locomotive>,
        save_interval: Option<usize>
    ) -> PyResult<Self> {
        Ok(Self::new(loco_vec, save_interval, PowerDistributionControlType::default()))
    }

    #[getter("loco_vec")]
    fn get_loco_vec_py(&self) -> PyResult<Pyo3VecLocoWrapper> {
        Ok(Pyo3VecLocoWrapper(self.loco_vec.clone()))
    }

    #[setter("loco_vec")]
    fn set_loco_vec_py(&mut self, loco_vec: Vec<Locomotive>) -> PyResult<()> {
        self.set_loco_vec(loco_vec);
        Ok(())
    }

    #[pyo3(name="drain_loco_vec")]
    fn drain_loco_vec_py(&mut self, start: usize, end: usize) -> PyResult<Pyo3VecLocoWrapper> {
        Ok(Pyo3VecLocoWrapper(self.drain_loco_vec(start, end)))
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

    // methods setting values for hct, which is not directly exposed to python because enums
    // with fields are not supported by pyo3.

    /// Set hct to PowerDistributionControlType::Proportional
    fn set_pdct_prop(&mut self) {
        self.pdct = PowerDistributionControlType::Proportional(Proportional);
    }
    /// Set hct to PowerDistributionControlType::Greedy
    fn set_pdct_resgreedy(&mut self) {
        self.pdct = PowerDistributionControlType::RESGreedy(RESGreedy);
    }
    /// Set hct to PowerDistributionControlType::GoldenSectionSearch(fuel_res_ratio, gss_interval)
    fn set_pdct_gss(&mut self, fuel_res_ratio: f64, gss_interval: usize) {
        self.pdct = PowerDistributionControlType::GoldenSectionSearch(
            GoldenSectionSearch{fuel_res_ratio, gss_interval}
        );
    }

    fn get_hct(&self) -> String {
        // make a `describe` function
        match &self.pdct {
            PowerDistributionControlType::RESGreedy(val) => format!("{val:?}"),
            PowerDistributionControlType::Proportional(val) => format!("{val:?}"),
            PowerDistributionControlType::GoldenSectionSearch(val) => format!("{val:?}"),
            PowerDistributionControlType::FrontAndBack(val) => format!("{val:?}"),
        }
    }

    #[setter("__assert_limits")]
    fn set_assert_limits_py(&mut self, val: bool) {
        self.set_assert_limits(val);
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    fn get_net_energy_res_py(&self) -> f64 {
        self.get_net_energy_res().get::<si::joule>()
    }

    #[pyo3(name = "get_energy_fuel_joules")]
    fn get_energy_fuel_py(&self) -> f64 {
        self.get_energy_fuel().get::<si::joule>()
    }

    #[getter("force_max_lbs")]
    fn get_force_max_pounds_py(&self) -> PyResult<f64> {
        Ok(self.force_max()?.get::<si::pound_force>())
    }

    #[getter("force_max_newtons")]
    fn get_force_max_newtons_py(&self) -> PyResult<f64> {
        Ok(self.force_max()?.get::<si::newton>())
    }

    #[getter("mass_kg")]
    fn get_mass_kg_py(&self) -> PyResult<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }
)]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
/// Struct for simulating power distribution controls and energy usage of locomotive consist.  
pub struct Consist {
    // pretty sure these won't get automatically generated correctly
    #[api(skip_get, skip_set)]
    /// vector of locomotives, must be private to allow for side effects when setting
    pub loco_vec: Vec<Locomotive>,
    #[api(skip_set, skip_get)]
    /// power distribution control type
    pub pdct: PowerDistributionControlType,
    #[serde(default = "utils::return_true")]
    #[api(skip_set)] // setter needs to also apply to individual locomotives
    /// whether to panic if TPC requires more power than consist can deliver
    assert_limits: bool,
    pub state: ConsistState,
    /// Custom vector of [Self::state]
    pub history: ConsistStateHistoryVec,
    #[api(skip_set, skip_get)] // custom needed for this
    save_interval: Option<usize>,
    #[serde(skip)]
    #[api(skip_get, skip_set)]
    n_res_equipped: Option<u8>,
}

impl SerdeAPI for Consist {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        self.update_mass(None)?;
        Ok(())
    }
}

impl Consist {
    pub fn new(
        loco_vec: Vec<Locomotive>,
        save_interval: Option<usize>,
        pdct: PowerDistributionControlType,
    ) -> Self {
        let mut consist = Self {
            state: Default::default(),
            loco_vec,
            history: Default::default(),
            save_interval,
            pdct,
            assert_limits: true,
            n_res_equipped: None,
        };
        let _ = consist.n_res_equipped();
        consist.set_save_interval(save_interval);
        consist
    }

    /// Returns number of RES-equipped locomotives
    fn n_res_equipped(&mut self) -> u8 {
        match self.n_res_equipped {
            Some(n_res_equipped) => n_res_equipped,
            None => {
                self.n_res_equipped = Some(self.loco_vec.iter().fold(0, |acc, loco| {
                    acc + if loco.reversible_energy_storage().is_some() {
                        1
                    } else {
                        0
                    }
                }));
                self.n_res_equipped.unwrap()
            }
        }
    }

    pub fn set_assert_limits(&mut self, val: bool) {
        self.assert_limits = val;
        for loco in self.loco_vec.iter_mut() {
            loco.assert_limits = val;
        }
    }

    pub fn force_max(&self) -> anyhow::Result<si::Force> {
        self.loco_vec.iter().enumerate().try_fold(
            0. * uc::N,
            |f_sum, (i, loco)| -> anyhow::Result<si::Force> {
                Ok(loco
                    .force_max()?
                    .ok_or_else(|| anyhow!("Locomotive {i} does not have `force_max` set"))?
                    + f_sum)
            },
        )
    }

    pub fn get_loco_vec(&self) -> Vec<Locomotive> {
        self.loco_vec.clone()
    }

    pub fn set_loco_vec(&mut self, loco_vec: Vec<Locomotive>) {
        self.loco_vec = loco_vec;
    }

    pub fn drain_loco_vec(&mut self, start: usize, end: usize) -> Vec<Locomotive> {
        let loco_vec = self.loco_vec.drain(start..end).collect();
        loco_vec
    }

    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        for loco in self.loco_vec.iter_mut() {
            loco.set_save_interval(save_interval);
        }
    }

    /// Set catenary charging/discharging power limit
    pub fn set_cat_power_limit(&mut self, path_tpc: &crate::track::PathTpc, offset: si::Length) {
        for cpl in path_tpc.cat_power_limits() {
            if offset < cpl.offset_start {
                break;
            } else if offset <= cpl.offset_end {
                self.state.pwr_cat_lim = cpl.power_limit;
                return;
            }
        }
        self.state.pwr_cat_lim = si::Power::ZERO;
    }

    pub fn get_energy_fuel(&self) -> si::Energy {
        self.loco_vec
            .iter()
            .map(|loco| match loco.loco_type {
                LocoType::BatteryElectricLoco(_) => si::Energy::ZERO,
                _ => loco.fuel_converter().unwrap().state.energy_fuel,
            })
            .sum::<si::Energy>()
    }

    pub fn get_net_energy_res(&self) -> si::Energy {
        self.loco_vec
            .iter()
            .map(|lt| match &lt.loco_type {
                LocoType::BatteryElectricLoco(loco) => loco.res.state.energy_out_chemical,
                LocoType::HybridLoco(loco) => loco.res.state.energy_out_chemical,
                _ => si::Energy::ZERO,
            })
            .sum::<si::Energy>()
    }

    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: Option<bool>,
    ) -> anyhow::Result<()> {
        // TODO: account for catenary in here
        if self.assert_limits {
            ensure!(
                -pwr_out_req <= self.state.pwr_dyn_brake_max,
                "{}\n{}",
                format_dbg!(-pwr_out_req),
                format_dbg!(self.state.pwr_dyn_brake_max),
            );
            ensure!(
                pwr_out_req <= self.state.pwr_out_max,
                "{}\n{}",
                format_dbg!(pwr_out_req),
                format_dbg!(self.state.pwr_out_max)
            );
        }

        self.state.pwr_out_req = pwr_out_req;
        self.state.pwr_out_deficit =
            (pwr_out_req - self.state.pwr_out_max_reves).max(si::Power::ZERO);
        self.state.pwr_regen_deficit =
            (-pwr_out_req - self.state.pwr_regen_max).max(si::Power::ZERO);

        // Sum of dynamic braking capability, including regenerative capability
        self.state.pwr_dyn_brake_max = self
            .loco_vec
            .iter()
            .map(|loco| match &loco.loco_type {
                LocoType::ConventionalLoco(conv) => conv.edrv.pwr_out_max,
                LocoType::HybridLoco(hel) => hel.edrv.pwr_out_max,
                LocoType::BatteryElectricLoco(bel) => bel.edrv.pwr_out_max,
                // really big number that is not inf to avoid null in json
                LocoType::Dummy(_) => uc::W * 1e15,
            })
            .sum();

        let pwr_out_vec: Vec<si::Power> = if pwr_out_req > si::Power::ZERO {
            // positive tractive power `pwr_out_vec`
            self.pdct
                .solve_positive_traction(&self.loco_vec, &self.state)?
        } else if pwr_out_req < si::Power::ZERO {
            // negative tractive power `pwr_out_vec`
            self.pdct
                .solve_negative_traction(&self.loco_vec, &self.state)?
        } else {
            // zero tractive power `pwr_out_vec`
            vec![si::Power::ZERO; self.loco_vec.len()]
        };

        self.state.pwr_out = pwr_out_vec
            .iter()
            .fold(si::Power::ZERO, |acc, &curr| acc + curr);

        if self.assert_limits {
            ensure!(
                utils::almost_eq_uom(&self.state.pwr_out_req, &self.state.pwr_out, None),
                format!(
                    "{}
                    self.state.pwr_out_req: {:.6} MW 
                    self.state.pwr_out: {:.6} MW
                    self.state.pwr_out_deficit: {:.6} MW 
                    pwr_out_vec: {:?}",
                    format_dbg!(),
                    &self.state.pwr_out_req.get::<si::megawatt>(),
                    &self.state.pwr_out.get::<si::megawatt>(),
                    &self.state.pwr_out_deficit.get::<si::megawatt>(),
                    &pwr_out_vec,
                )
            );
        }

        // maybe put logic for toggling `engine_on` here

        for (i, (loco, pwr_out)) in self.loco_vec.iter_mut().zip(pwr_out_vec.iter()).enumerate() {
            loco.solve_energy_consumption(*pwr_out, dt, engine_on)
                .map_err(|err| {
                    err.context(format!(
                        "loco idx: {}, loco type: {}",
                        i,
                        loco.loco_type.to_string()
                    ))
                })?;
        }

        self.state.pwr_fuel = self
            .loco_vec
            .iter()
            .map(|loco| match &loco.loco_type {
                LocoType::ConventionalLoco(cl) => cl.fc.state.pwr_fuel,
                LocoType::HybridLoco(hel) => hel.fc.state.pwr_fuel,
                LocoType::BatteryElectricLoco(_) => si::Power::ZERO,
                LocoType::Dummy(_) => f64::NAN * uc::W,
            })
            .sum();

        self.state.pwr_reves = self
            .loco_vec
            .iter()
            .map(|loco| match &loco.loco_type {
                LocoType::ConventionalLoco(_cl) => si::Power::ZERO,
                LocoType::HybridLoco(hel) => hel.res.state.pwr_out_chemical,
                LocoType::BatteryElectricLoco(bel) => bel.res.state.pwr_out_chemical,
                LocoType::Dummy(_) => f64::NAN * uc::W,
            })
            .sum();

        self.state.energy_out += self.state.pwr_out * dt;
        if self.state.pwr_out >= 0. * uc::W {
            self.state.energy_out_pos += self.state.pwr_out * dt;
        } else {
            self.state.energy_out_neg -= self.state.pwr_out * dt;
        }
        self.state.energy_fuel += self.state.pwr_fuel * dt;
        self.state.energy_res += self.state.pwr_reves * dt;
        Ok(())
    }
}

impl Default for Consist {
    fn default() -> Self {
        let bel_type = LocoType::BatteryElectricLoco(BatteryElectricLoco::default());
        let mut bel = Locomotive::default();
        bel.loco_type = bel_type;
        bel.set_save_interval(Some(1));
        // TODO: change PowerDistributionControlType to whatever ends up being best
        let mut consist = Self {
            state: Default::default(),
            history: Default::default(),
            loco_vec: vec![
                Locomotive::default(),
                bel,
                Locomotive::default(),
                Locomotive::default(),
                Locomotive::default(),
            ],
            assert_limits: true,
            save_interval: Some(1),
            n_res_equipped: Default::default(),
            pdct: Default::default(),
        };
        // ensure propagation to nested components
        consist.set_save_interval(Some(1));
        consist.check_mass_consistent().unwrap();
        consist
    }
}

impl LocoTrait for Consist {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: this will need to account for catenary power
        // TODO: need to be able to configure regen to go to catenary or not
        // TODO: make sure that self.state includes catenary effects so that `solve_energy_consumption`
        // is operating with the same catenary power availability at the train position for which this
        // method is called
        ensure!(pwr_aux.is_none(), format_dbg!(pwr_aux.is_none()));
        for (i, loco) in self.loco_vec.iter_mut().enumerate() {
            loco.set_cur_pwr_max_out(None, dt).map_err(|err| {
                err.context(format!(
                    "loco idx: {} loco type: {}",
                    i,
                    loco.loco_type.to_string()
                ))
            })?;
        }
        self.state.pwr_out_max = self
            .loco_vec
            .iter()
            .fold(si::Power::ZERO, |acc, loco| acc + loco.state.pwr_out_max);
        self.state.pwr_rate_out_max =
            self.loco_vec.iter().fold(si::PowerRate::ZERO, |acc, loco| {
                acc + loco.state.pwr_rate_out_max
            });
        self.state.pwr_regen_max = self
            .loco_vec
            .iter()
            .fold(si::Power::ZERO, |acc, loco| acc + loco.state.pwr_regen_max);
        self.state.pwr_out_max_reves = self
            .loco_vec
            .iter()
            .map(|loco| match &loco.loco_type {
                LocoType::ConventionalLoco(_) => si::Power::ZERO,
                LocoType::HybridLoco(_) => loco.state.pwr_out_max,
                LocoType::BatteryElectricLoco(_) => loco.state.pwr_out_max,
                // really big number that is not inf to avoid null in json
                LocoType::Dummy(_) => 1e15 * uc::W,
            })
            .sum();
        self.state.pwr_out_max_non_reves = self.state.pwr_out_max - self.state.pwr_out_max_reves;

        Ok(())
    }

    fn step(&mut self) {
        for loco in self.loco_vec.iter_mut() {
            loco.step();
        }
        self.state.i += 1;
    }

    fn save_state(&mut self) {
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || self.state.i == 1 {
                self.history.push(self.state);
                for loco in self.loco_vec.iter_mut() {
                    loco.save_state();
                }
            }
        }
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.loco_vec
            .iter()
            .map(|loco| loco.get_energy_loss())
            .sum()
    }
}

impl Mass for Consist {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let mass = self.loco_vec.iter().enumerate().try_fold(
            0. * uc::KG,
            |m_acc, (i, loco)| -> anyhow::Result<si::Mass> {
                let loco_mass = loco
                    .mass()?
                    .ok_or_else(|| anyhow!("Locomotive {i} does not have `mass` set"))?;
                let new_mass: si::Mass = loco_mass + m_acc;
                Ok(new_mass)
            },
        )?;
        Ok(Some(mass))
    }

    fn update_mass(&mut self, _mass: Option<si::Mass>) -> anyhow::Result<()> {
        self.loco_vec
            .iter_mut()
            .enumerate()
            .try_for_each(|(i, loco)| -> anyhow::Result<()> {
                loco.update_mass(None).map_err(|e| {
                    anyhow!("{e}").context(format!("{}\nfailed at loco: {}", format_dbg!(), i))
                })
            })
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        for (i, loco) in self.loco_vec.iter().enumerate() {
            match loco.check_mass_consistent() {
                Ok(res) => res,
                Err(e) => bail!(
                    "{e}\n{}",
                    format!(
                        "{}\nfailed at loco: {}\n{}",
                        format_dbg!(),
                        i,
                        "Try running `update_mass` method."
                    )
                ),
            };
        }

        Ok(())
    }
}
/// Locomotive State
/// probably reusable across all powertrain types
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct ConsistState {
    /// current time index
    pub i: usize,

    /// maximum forward propulsive power consist can produce
    pub pwr_out_max: si::Power,
    /// maximum rate of increase of forward propulsive power consist can produce
    pub pwr_rate_out_max: si::PowerRate,
    /// maximum regen power consist can absorb at the wheel
    pub pwr_regen_max: si::Power,

    // limit variables
    /// maximum power that can be produced by
    /// [RES](locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage)-equppped locomotives
    pub pwr_out_max_reves: si::Power,
    /// power demand not fulfilled by
    /// [RES](locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage)-equppped locomotives
    pub pwr_out_deficit: si::Power,
    /// max power demand from
    /// non-[RES](locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage)-equppped locomotives
    pub pwr_out_max_non_reves: si::Power,
    /// braking power demand not fulfilled as regen by [RES](locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage)-equppped locomotives
    pub pwr_regen_deficit: si::Power,
    /// Total dynamic braking power of consist, based on sum of
    /// [electric-drivetrain](locomotive::powertrain::electric_drivetrain::ElectricDrivetrain)
    /// static limits across all locomotives (including regen).
    pub pwr_dyn_brake_max: si::Power,
    /// consist power output requested by [SpeedLimitTrainSim](crate::train::SpeedLimitTrainSim) or
    /// [SetSpeedTrainSim](crate::train::SetSpeedTrainSim)
    pub pwr_out_req: si::Power,
    /// Current consist/train-level catenary power limit
    pub pwr_cat_lim: si::Power,

    // achieved values
    /// Total tractive power of consist.
    /// Should always match [pwr_out_req](Self::pwr_out_req)] if `assert_limits == true`.  
    pub pwr_out: si::Power,
    /// Total battery power of [RES](locomotive::powertrain::reversible_energy_storage::ReversibleEnergyStorage)-equppped locomotives
    pub pwr_reves: si::Power,
    /// Total fuel power of [FC](locomotive::powertrain::fuel_converter::FuelConverter)-equppped locomotives
    pub pwr_fuel: si::Power,

    /// Time-integrated energy form of [pwr_out](Self::pwr_out)
    pub energy_out: si::Energy,
    /// Energy out during positive or zero traction
    pub energy_out_pos: si::Energy,
    /// Energy out during negative traction (positive value means negative traction)
    pub energy_out_neg: si::Energy,
    /// Time-integrated energy form of [pwr_reves](Self::pwr_reves)
    pub energy_res: si::Energy,
    /// Time-integrated energy form of [pwr_fuel](Self::pwr_fuel)
    pub energy_fuel: si::Energy,
}

impl Default for ConsistState {
    fn default() -> Self {
        Self {
            i: 1,
            pwr_out_max: Default::default(),
            pwr_rate_out_max: Default::default(),
            pwr_regen_max: Default::default(),

            // limit variables
            pwr_out_max_reves: Default::default(),
            pwr_out_deficit: Default::default(),
            pwr_out_max_non_reves: Default::default(),
            pwr_regen_deficit: Default::default(),
            pwr_dyn_brake_max: Default::default(),
            pwr_out_req: Default::default(),
            pwr_cat_lim: Default::default(),

            // achieved values
            pwr_out: Default::default(),
            pwr_reves: Default::default(),
            pwr_fuel: Default::default(),

            energy_out: Default::default(),
            energy_out_pos: Default::default(),
            energy_out_neg: Default::default(),

            energy_res: Default::default(),
            energy_fuel: Default::default(),
        }
    }
}
