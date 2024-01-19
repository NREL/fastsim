use super::*;

#[enum_dispatch(Powertrain)]
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
