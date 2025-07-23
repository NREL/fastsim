use super::*;

pub mod hvac_utils;
pub use hvac_utils::*;

pub mod hvac_sys_for_lumped_cabin;
pub use hvac_sys_for_lumped_cabin::*;

pub mod hvac_sys_for_lumped_cabin_and_res;
pub use hvac_sys_for_lumped_cabin_and_res::*;

/// Options for handling HVAC system
#[derive(
    Clone,
    Default,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    IsVariant,
    derive_more::From,
    TryInto,
    derive_more::Display,
)]
pub enum HVACOption {
    /// HVAC system for [LumpedCabin]
    #[display("LumpedCabin")]
    LumpedCabin(Box<HVACSystemForLumpedCabin>),
    /// HVAC system for [LumpedCabin] and [ReversibleEnergyStorage]
    #[display("LumpedCabinAndRES")]
    LumpedCabinAndRES(Box<HVACSystemForLumpedCabinAndRES>),
    /// Cabin with interior and shell capacitances
    #[display("LumpedCabinWithShell")]
    LumpedCabinWithShell,
    /// [ReversibleEnergyStorage] thermal management with no cabin
    #[display("ReversibleEnergyStorageOnly")]
    ReversibleEnergyStorageOnly,
    /// no cabin thermal model
    #[default]
    #[display("None")]
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
impl SetCumulative for HVACOption {
    fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
        match self {
            HVACOption::LumpedCabin(lc) => {
                lc.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?
            }
            HVACOption::LumpedCabinAndRES(lcr) => {
                lcr.set_cumulative(dt, || format!("{}\n{}", loc(), format_dbg!()))?
            }
            HVACOption::LumpedCabinWithShell => todo!(),
            HVACOption::ReversibleEnergyStorageOnly => todo!(),
            HVACOption::None => {}
        }
        Ok(())
    }

    fn reset_cumulative<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            HVACOption::LumpedCabin(lc) => {
                lc.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            HVACOption::LumpedCabinAndRES(lcr) => {
                lcr.reset_cumulative(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            HVACOption::LumpedCabinWithShell => todo!(),
            HVACOption::ReversibleEnergyStorageOnly => todo!(),
            HVACOption::None => {}
        }
        Ok(())
    }
}
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

impl StateMethods for HVACOption {}

impl SaveState for HVACOption {
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.save_state(loc)?,
            Self::LumpedCabinAndRES(lcr) => lcr.save_state(loc)?,
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => {}
        }
        Ok(())
    }
}
impl TrackedStateMethods for HVACOption {
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => {
                lc.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::LumpedCabinAndRES(lcr) => {
                lcr.check_and_reset(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => {}
        }
        Ok(())
    }

    fn mark_fresh<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?,
            Self::LumpedCabinAndRES(lcr) => {
                lcr.mark_fresh(|| format!("{}\n{}", loc(), format_dbg!()))?
            }
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => {}
        }
        Ok(())
    }
}
impl Step for HVACOption {
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.step(|| format!("{}\n{}", loc(), format_dbg!())),
            Self::LumpedCabinAndRES(lcr) => lcr.step(|| format!("{}\n{}", loc(), format_dbg!())),
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => Ok(()),
        }
    }

    fn reset_step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        match self {
            Self::LumpedCabin(lc) => lc.reset_step(|| format!("{}\n{}", loc(), format_dbg!())),
            Self::LumpedCabinAndRES(lcr) => {
                lcr.reset_step(|| format!("{}\n{}", loc(), format_dbg!()))
            }
            Self::LumpedCabinWithShell => {
                todo!()
            }
            Self::ReversibleEnergyStorageOnly => todo!(),
            Self::None => Ok(()),
        }
    }
}
