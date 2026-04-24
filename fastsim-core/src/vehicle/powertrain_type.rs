use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, IsVariant, TryInto)]
pub enum PowertrainType {
    #[serde(rename = "Conv")]
    #[serde(alias = "ConventionalVehicle")]
    ConventionalVehicle(Box<ConventionalVehicle>),
    #[serde(rename = "HEV")]
    #[serde(alias = "HybridElectricVehicle")]
    HybridElectricVehicle(Box<HybridElectricVehicle>),
    #[serde(rename = "PHEV")]
    #[serde(alias = "PlugInHybridElectricVehicle")]
    PlugInHybridElectricVehicle(Box<HybridElectricVehicle>),
    #[serde(rename = "BEV")]
    #[serde(alias = "BatteryElectricVehicle")]
    BatteryElectricVehicle(Box<BatteryElectricVehicle>),
}

impl SerdeAPI for PowertrainType {}
impl Init for PowertrainType {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::ConventionalVehicle(conv) => conv.init(),
            Self::HybridElectricVehicle(hev) => hev.init(),
            Self::PlugInHybridElectricVehicle(phev) => phev.init(),
            Self::BatteryElectricVehicle(bev) => bev.init(),
        }
    }
}

impl SetCumulative for PowertrainType {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => {
                conv.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::HybridElectricVehicle(hev) => {
                hev.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))
            }
        }
    }

    fn reset_cumulative<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => {
                conv.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::HybridElectricVehicle(hev) => {
                hev.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))
            }
        }
    }
}

impl Powertrain for PowertrainType {
    fn set_curr_pwr_prop_out_max(
        &mut self,
        _pwr_upstream: (si::Power, si::Power),
        pwr_aux: si::Power,
        dt: si::Time,
        veh_state: &VehicleState,
    ) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(v) => v.set_curr_pwr_prop_out_max(
                (si::Power::ZERO, si::Power::ZERO),
                pwr_aux,
                dt,
                veh_state,
            ),
            Self::HybridElectricVehicle(v) => v.set_curr_pwr_prop_out_max(
                (si::Power::ZERO, si::Power::ZERO),
                pwr_aux,
                dt,
                veh_state,
            ),
            Self::PlugInHybridElectricVehicle(v) => v.set_curr_pwr_prop_out_max(
                (si::Power::ZERO, si::Power::ZERO),
                pwr_aux,
                dt,
                veh_state,
            ),
            Self::BatteryElectricVehicle(v) => v.set_curr_pwr_prop_out_max(
                (si::Power::ZERO, si::Power::ZERO),
                pwr_aux,
                dt,
                veh_state,
            ),
        }
    }

    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<Option<si::Power>> {
        match self {
            Self::ConventionalVehicle(v) => v.solve(pwr_out_req.max(si::Power::ZERO), enabled, dt),
            Self::HybridElectricVehicle(v) => v.solve(pwr_out_req, enabled, dt),
            Self::PlugInHybridElectricVehicle(v) => v.solve(pwr_out_req, enabled, dt),
            Self::BatteryElectricVehicle(v) => v.solve(pwr_out_req, enabled, dt),
        }
    }

    /// Returns max power for forward direction and backward direction
    fn get_curr_pwr_prop_out_max(&self) -> anyhow::Result<(si::Power, si::Power)> {
        match self {
            Self::ConventionalVehicle(v) => v.get_curr_pwr_prop_out_max(),
            Self::HybridElectricVehicle(v) => v.get_curr_pwr_prop_out_max(),
            Self::PlugInHybridElectricVehicle(v) => v.get_curr_pwr_prop_out_max(),
            Self::BatteryElectricVehicle(v) => v.get_curr_pwr_prop_out_max(),
        }
    }

    fn pwr_regen(&self) -> anyhow::Result<si::Power> {
        match self {
            Self::ConventionalVehicle(v) => v.pwr_regen(),
            Self::HybridElectricVehicle(v) => v.pwr_regen(),
            Self::PlugInHybridElectricVehicle(v) => v.pwr_regen(),
            Self::BatteryElectricVehicle(v) => v.pwr_regen(),
        }
    }
}

impl HistoryMethods for PowertrainType {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        match self {
            PowertrainType::ConventionalVehicle(v) => v.save_interval(),
            PowertrainType::HybridElectricVehicle(v) => v.save_interval(),
            PowertrainType::PlugInHybridElectricVehicle(v) => v.save_interval(),
            PowertrainType::BatteryElectricVehicle(v) => v.save_interval(),
        }
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(v) => v.set_save_interval(save_interval),
            PowertrainType::HybridElectricVehicle(v) => v.set_save_interval(save_interval),
            PowertrainType::PlugInHybridElectricVehicle(v) => v.set_save_interval(save_interval),
            PowertrainType::BatteryElectricVehicle(v) => v.set_save_interval(save_interval),
        }
    }
    fn clear(&mut self) {
        match self {
            PowertrainType::ConventionalVehicle(v) => v.clear(),
            PowertrainType::HybridElectricVehicle(v) => v.clear(),
            PowertrainType::PlugInHybridElectricVehicle(v) => v.clear(),
            PowertrainType::BatteryElectricVehicle(v) => v.clear(),
        }
    }
}

impl PowertrainType {
    /// # Arguments:
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_fc_to_cab`: thermal power flow from [FuelConverter::thrml] to [Vehicle::cabin], if both are equipped
    /// - `veh_state`: current state of vehicle
    /// - `pwr_thrml_hvac_to_res`: thermal power flow from [Vehicle::hvac]
    ///   system, if equipped, to [ReversibleEnergyStorage::thrml] -- zero if `None` is
    ///   passed
    /// - `te_cab`: [Vehicle::cabin] temperature, if equipped
    /// - `dt`: simulation time step size
    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: Option<si::Power>,
        veh_state: &mut VehicleState,
        pwr_thrml_hvac_to_res: Option<si::Power>,
        te_cab: Option<si::Temperature>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(v) => {
                v.solve_thermal(te_amb, pwr_thrml_fc_to_cab, veh_state, dt)
            }
            Self::HybridElectricVehicle(v) => v.solve_thermal(
                te_amb,
                pwr_thrml_fc_to_cab,
                veh_state,
                pwr_thrml_hvac_to_res,
                te_cab,
                dt,
            ),
            Self::PlugInHybridElectricVehicle(v) => v.solve_thermal(
                te_amb,
                pwr_thrml_fc_to_cab,
                veh_state,
                pwr_thrml_hvac_to_res,
                te_cab,
                dt,
            ),
            Self::BatteryElectricVehicle(v) => v.solve_thermal(
                te_amb,
                pwr_thrml_hvac_to_res.unwrap_or_default(),
                te_cab,
                dt,
            ),
        }
    }

    pub fn conv_mut(&mut self) -> Option<&mut ConventionalVehicle> {
        match self {
            Self::ConventionalVehicle(conv) => Some(conv),
            _ => None,
        }
    }

    pub fn hev_mut(&mut self) -> Option<&mut HybridElectricVehicle> {
        match self {
            Self::HybridElectricVehicle(hev) => Some(hev),
            _ => None,
        }
    }

    // pub fn phev_mut(&mut self) -> Option<&mut> {
    //     self.pt_type.phev()
    // }

    pub fn bev_mut(&mut self) -> Option<&mut BatteryElectricVehicle> {
        match self {
            Self::BatteryElectricVehicle(bev) => Some(bev),
            _ => None,
        }
    }

    pub fn conv(&self) -> Option<&ConventionalVehicle> {
        match self {
            Self::ConventionalVehicle(conv) => Some(conv),
            _ => None,
        }
    }

    pub fn hev(&self) -> Option<&HybridElectricVehicle> {
        match self {
            Self::HybridElectricVehicle(hev) => Some(hev),
            _ => None,
        }
    }

    // pub fn phev(&self) -> Option<&> {
    //     self.pt_type.phev()
    // }

    pub fn bev(&self) -> Option<&BatteryElectricVehicle> {
        match self {
            Self::BatteryElectricVehicle(bev) => Some(bev),
            _ => None,
        }
    }

    pub fn fc(&self) -> Option<&FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.fc),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&hev.fc),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn fc_mut(&mut self) -> Option<&mut FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.fc),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&mut hev.fc),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn set_fc(&mut self, fc: FuelConverter) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.fc = fc;
                Ok(())
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.fc = fc;
                Ok(())
            }
            PowertrainType::PlugInHybridElectricVehicle(phev) => {
                phev.fc = fc;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(_) => bail!("BEL has no FuelConverter."),
        }
    }

    pub fn fs(&self) -> Option<&FuelStorage> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.fs),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.fs),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&hev.fs),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn fs_mut(&mut self) -> Option<&mut FuelStorage> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.fs),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.fs),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&mut hev.fs),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn set_fs(&mut self, fs: FuelStorage) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.fs = fs;
                Ok(())
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.fs = fs;
                Ok(())
            }
            PowertrainType::PlugInHybridElectricVehicle(phev) => {
                phev.fs = fs;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(_) => bail!("BEL has no FuelConverter."),
        }
    }

    pub fn res(&self) -> Option<&ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.res),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&hev.res),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.res),
        }
    }

    pub fn res_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.res),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&mut hev.res),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&mut bev.res),
        }
    }

    pub fn set_res(&mut self, res: ReversibleEnergyStorage) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(_) => {
                bail!("Conventional has no ReversibleEnergyStorage.")
            }
            PowertrainType::HybridElectricVehicle(veh) => {
                veh.res = res;
                Ok(())
            }
            PowertrainType::PlugInHybridElectricVehicle(veh) => {
                veh.res = res;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(veh) => {
                veh.res = res;
                Ok(())
            }
        }
    }

    pub fn em(&self) -> Option<&ElectricMachine> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.em),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&hev.em),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.em),
        }
    }

    pub fn em_mut(&mut self) -> Option<&mut ElectricMachine> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.em),
            PowertrainType::PlugInHybridElectricVehicle(hev) => Some(&mut hev.em),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&mut bev.em),
        }
    }

    pub fn set_em(&mut self, em: ElectricMachine) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(_conv) => {
                Err(anyhow!("ConventionalVehicle has no `ElectricMachine`"))
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.em = em;
                Ok(())
            }
            PowertrainType::PlugInHybridElectricVehicle(phev) => {
                phev.em = em;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(bev) => {
                bev.em = em;
                Ok(())
            }
        }
    }

    pub fn trans(&self) -> Option<&Transmission> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.transmission),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.transmission),
            PowertrainType::PlugInHybridElectricVehicle(phev) => Some(&phev.transmission),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.transmission),
        }
    }

    pub fn trans_mut(&mut self) -> Option<&mut Transmission> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.transmission),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.transmission),
            PowertrainType::PlugInHybridElectricVehicle(phev) => Some(&mut phev.transmission),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&mut bev.transmission),
        }
    }

    pub fn set_trans(&mut self, trans: Transmission) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => {
                conv.transmission = trans;
                Ok(())
            }
            PowertrainType::HybridElectricVehicle(hev) => {
                hev.transmission = trans;
                Ok(())
            }
            PowertrainType::PlugInHybridElectricVehicle(phev) => {
                phev.transmission = trans;
                Ok(())
            }
            PowertrainType::BatteryElectricVehicle(bev) => {
                bev.transmission = trans;
                Ok(())
            }
        }
    }

    pub fn variant_as_str(&self) -> String {
        match self {
            Self::ConventionalVehicle(_) => String::from("ConventionalVehicle"),
            Self::PlugInHybridElectricVehicle(_) => String::from("PlugInHybridElectricVehicle"),
            Self::HybridElectricVehicle(_) => String::from("HybridElectricVehicle"),
            Self::BatteryElectricVehicle(_) => String::from("BatteryElectricVehicle"),
        }
    }
}

impl StateMethods for PowertrainType {}

impl SaveState for PowertrainType {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::ConventionalVehicle(conv) => {
                conv.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::HybridElectricVehicle(hev) => {
                hev.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}
impl TrackedStateMethods for PowertrainType {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::ConventionalVehicle(conv) => {
                conv.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::HybridElectricVehicle(hev) => {
                hev.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }

    fn mark_fresh<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => {
                conv.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::HybridElectricVehicle(hev) => {
                hev.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
        }
        Ok(())
    }
}

impl Step for PowertrainType {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => {
                conv.step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::HybridElectricVehicle(hev) => {
                hev.step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
        }
    }

    fn reset_step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(conv) => {
                conv.reset_step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::HybridElectricVehicle(hev) => {
                hev.reset_step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::PlugInHybridElectricVehicle(phev) => {
                phev.reset_step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::BatteryElectricVehicle(bev) => {
                bev.reset_step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
        }
    }
}

#[allow(clippy::to_string_trait_impl)]
impl std::string::ToString for PowertrainType {
    fn to_string(&self) -> String {
        match self {
            PowertrainType::ConventionalVehicle(_) => String::from("Conv"),
            PowertrainType::HybridElectricVehicle(_) => String::from("HEV"),
            PowertrainType::PlugInHybridElectricVehicle(_) => String::from("PHEV"),
            PowertrainType::BatteryElectricVehicle(_) => String::from("BEV"),
        }
    }
}
