use super::*;

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
    fn init(&mut self) -> Result<(), FsimError> {
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
