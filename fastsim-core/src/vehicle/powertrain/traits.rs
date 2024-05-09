//! Traits defining power flow interfaces for various powertrain components
use super::*;

/// Provides functions for solving powertrain through component, i.e. / any
/// component that accepts input power and provides output power but is not
/// itself a power source, e.g. `ElectricMachine`.
pub trait PowertrainThrough {
    /// Returns maximum possible positive and negative traction-related powers
    /// this component/system can produce, accounting for any aux-related power
    /// required.
    /// # Arguments
    /// - `pwr_in_pos_max`: positive-traction-related power available to this
    /// component
    /// - `pwr_in_neg_max`: negative-traction-related power available to this
    /// component
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    fn get_cur_pwr_tract_out_max(
        &mut self,
        pwr_in_pos_max: si::Power,
        pwr_in_neg_max: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<(si::Power, si::Power)>;

    /// Solves for this powertrain system/component efficiency and sets/returns power input required.
    /// # Arguments
    /// - `pwr_out_req`: traction-related power output required
    /// - `pwr_aux`: component-specific aux power demand (e.g. mechanical power if from engine/FC)
    /// - `dt`: time step size
    fn get_pwr_in_req(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power>;
}

pub trait Powertrain {
    /// Returns maximum possible positive and negative traction-related powers
    /// this component/system can produce, accounting for any aux-related power
    /// required.
    /// # Arguments
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    fn get_cur_pwr_tract_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<(si::Power, si::Power)>;

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: traction-related power output required
    /// - `pwr_aux`: component-specific aux power demand (e.g. mechanical power if from engine/FC)
    /// - `enabled`: whether the component is active in current time step (e.g. engine idling v. shut off)
    /// - `dt`: time step size
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()>;
}
