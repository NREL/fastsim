use super::*;

pub mod hvac_utils;
pub use hvac_utils::*;

pub mod hvac_sys_for_lumped_cabin;
pub use hvac_sys_for_lumped_cabin::*;

pub mod hvac_sys_for_lumped_cabin_and_res;
pub use hvac_sys_for_lumped_cabin_and_res::*;

/// Options for handling HVAC system
#[derive(
    Clone, Default, Debug, Serialize, Deserialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
pub enum HVACOption {
    /// HVAC system for [LumpedCabin]
    LumpedCabin(Box<HVACSystemForLumpedCabin>),
    /// HVAC system for [LumpedCabin] and [ReversibleEnergyStorage]
    LumpedCabinAndRES(Box<HVACSystemForLumpedCabinAndRES>),
    /// Cabin with interior and shell capacitances
    LumpedCabinWithShell,
    /// [ReversibleEnergyStorage] thermal management with no cabin
    ReversibleEnergyStorageOnly,
    /// no cabin thermal model
    #[default]
    None,
}
impl Init for HVACOption {
    fn init(&mut self) -> Result<(), Error> {
        match self {
            Self::LumpedCabin(cab) => cab.init()?,
            Self::LumpedCabinAndRES(cab) => cab.init()?,
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => {
                todo!()
            }
            Self::None => {}
        }
        Ok(())
    }
}
impl SerdeAPI for HVACOption {}
impl HistoryMethods for HVACOption {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        match self {
            HVACOption::LumpedCabin(lc) => lc.save_interval(),
            HVACOption::LumpedCabinAndRES(lcr) => lcr.save_interval(),
            HVACOption::LumpedCabinWithShell => todo!(),
            HVACOption::ReversibleEnergyStorageOnly => todo!(),
            HVACOption::None => Ok(None),
        }
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        match self {
            HVACOption::LumpedCabin(lc) => lc.set_save_interval(save_interval),
            HVACOption::LumpedCabinAndRES(lcr) => lcr.set_save_interval(save_interval),
            HVACOption::LumpedCabinWithShell => todo!(),
            HVACOption::ReversibleEnergyStorageOnly => todo!(),
            HVACOption::None => Ok(()),
        }
    }
    fn clear(&mut self) {
        match self {
            HVACOption::LumpedCabin(lc) => lc.clear(),
            HVACOption::LumpedCabinAndRES(lcr) => lcr.clear(),
            HVACOption::LumpedCabinWithShell => todo!(),
            HVACOption::ReversibleEnergyStorageOnly => todo!(),
            HVACOption::None => {}
        }
    }
}
impl SaveState for HVACOption {
    fn save_state(&mut self) {
        match self {
            Self::LumpedCabin(lc) => lc.save_state(),
            Self::LumpedCabinAndRES(lcr) => lcr.save_state(),
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => {}
        }
    }
}
impl Step for HVACOption {
    fn step(&mut self) {
        match self {
            Self::LumpedCabin(lc) => lc.step(),
            Self::LumpedCabinAndRES(lcr) => lcr.step(),
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => {}
        }
    }
}
