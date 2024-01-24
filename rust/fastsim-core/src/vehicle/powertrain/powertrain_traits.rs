//! Traits defining power flow interfaces for electric machines
use super::*;

// TODO: fix and uncomment or delete this
// pub trait ElectricMachine {
//     /// Sets current max power output given `pwr_in_max` from upstream component
//     fn set_cur_pwr_max_out(
//         &mut self,
//         pwr_in_max: si::Power,
//         pwr_aux: Option<si::Power>,
//     ) -> anyhow::Result<()>;
//     /// Sets current max power output rate given `pwr_rate_in_max` from upstream component
//     fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate);
// }

pub trait Mass {
    /// Returns mass of Self, including contribution from any fields that implement `Mass`
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based
    /// on other component parameters (e.g. power and power density)
    fn update_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()>;

    /// Checks if mass is consistent with other parameters
    fn check_mass_consistent(&self) -> anyhow::Result<()>;
}

/// Provides functions for solving powertrain
#[enum_dispatch]
pub trait Powertrain {
    /// # Arguments
    /// - dt: time step size
    fn get_pwr_out_max(&mut self, dt: si::Time) -> anyhow::Result<si::Power>;
    /// Solves for power flow in powertrain
    /// # Arguments
    /// - dt: time step size
    fn solve_powertrain(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()>;
}
