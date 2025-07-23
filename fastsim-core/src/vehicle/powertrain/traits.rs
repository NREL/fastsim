//! Traits defining power flow interfaces for various powertrain components
use self::vehicle_model::VehicleState;

use super::*;

pub trait Powertrain {
    /// Sets maximum possible positive and negative propulsion-related powers
    /// this component/system can produce, accounting for any aux-related power
    /// required.
    /// # Arguments
    /// - `pwr_upstream`: power (in forward and backward/regen directions) available from upstream component, where applicable
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: simulation time step size
    /// - `veh_state`: the vehicle state
    fn set_curr_pwr_prop_out_max(
        &mut self,
        pwr_upstream: (si::Power, si::Power),
        pwr_aux: si::Power,
        dt: si::Time,
        veh_state: &VehicleState,
    ) -> anyhow::Result<()>;

    /// Returns maximum achievable positive and negative propulsion powers after
    /// [Powertrain::set_curr_pwr_prop_out_max] has been called.
    fn get_curr_pwr_prop_out_max(&self) -> anyhow::Result<(si::Power, si::Power)>;

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: propulsion-related power output required
    /// - `veh_state`: state of vehicle
    /// - `enabled`: whether the component is active in current time step (e.g. engine idling v. shut off)
    /// - `dt`: simulation time step size
    /// # Returns
    /// - Some(si::Power) if this is a pass-through component (e.g. [ElectricMachine])
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<Option<si::Power>>;

    /// Returns regen power after `Powertrain::solve` has been called
    fn pwr_regen(&self) -> anyhow::Result<si::Power>;
}
