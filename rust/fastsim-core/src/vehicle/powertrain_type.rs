use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub enum PowertrainType {
    ConventionalVehicle(Box<ConventionalVehicle>),
    HybridElectricVehicle(Box<HybridElectricVehicle>),
    BatteryElectricVehicle(Box<BatteryElectricVehicle>),
    // TODO: add PHEV here
}

impl Powertrain for PowertrainType {
    fn get_curr_pwr_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        match self {
            Self::ConventionalVehicle(v) => v.get_curr_pwr_out_max(pwr_aux, dt),
            Self::HybridElectricVehicle(v) => v.get_curr_pwr_out_max(pwr_aux, dt),
            Self::BatteryElectricVehicle(v) => v.get_curr_pwr_out_max(pwr_aux, dt),
        }
    }

    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        match self {
            Self::ConventionalVehicle(v) => {
                v.solve(pwr_out_req, pwr_aux, enabled, dt, assert_limits)
            }
            Self::HybridElectricVehicle(v) => {
                v.solve(pwr_out_req, pwr_aux, enabled, dt, assert_limits)
            }
            Self::BatteryElectricVehicle(v) => {
                v.solve(pwr_out_req, pwr_aux, enabled, dt, assert_limits)
            }
        }
    }
}

impl SaveInterval for PowertrainType {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        match self {
            PowertrainType::ConventionalVehicle(v) => v.save_interval(),
            PowertrainType::HybridElectricVehicle(v) => v.save_interval(),
            PowertrainType::BatteryElectricVehicle(v) => v.save_interval(),
        }
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        match self {
            PowertrainType::ConventionalVehicle(v) => v.set_save_interval(save_interval),
            PowertrainType::HybridElectricVehicle(v) => v.set_save_interval(save_interval),
            PowertrainType::BatteryElectricVehicle(v) => v.set_save_interval(save_interval),
        }
    }
}

impl PowertrainType {
    pub fn fc(&self) -> Option<&FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.fc),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn fc_mut(&mut self) -> Option<&mut FuelConverter> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.fc),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.fc),
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
            PowertrainType::BatteryElectricVehicle(_) => bail!("BEL has no FuelConverter."),
        }
    }

    pub fn fs(&self) -> Option<&FuelStorage> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&conv.fs),
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.fs),
            PowertrainType::BatteryElectricVehicle(_) => None,
        }
    }

    pub fn fs_mut(&mut self) -> Option<&mut FuelStorage> {
        match self {
            PowertrainType::ConventionalVehicle(conv) => Some(&mut conv.fs),
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.fs),
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
            PowertrainType::BatteryElectricVehicle(_) => bail!("BEL has no FuelConverter."),
        }
    }

    pub fn res(&self) -> Option<&ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&hev.res),
            PowertrainType::BatteryElectricVehicle(bev) => Some(&bev.res),
        }
    }

    pub fn res_mut(&mut self) -> Option<&mut ReversibleEnergyStorage> {
        match self {
            PowertrainType::ConventionalVehicle(_) => None,
            PowertrainType::HybridElectricVehicle(hev) => Some(&mut hev.res),
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

impl SaveState for PowertrainType {
    fn save_state(&mut self) {
        match self {
            Self::ConventionalVehicle(conv) => conv.save_state(),
            Self::HybridElectricVehicle(hev) => hev.save_state(),
            Self::BatteryElectricVehicle(bev) => bev.save_state(),
        }
    }
}

impl Step for PowertrainType {
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
