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
    /// Returns mass of Self, either from `self.mass` or
    /// the derived from fields that store mass data. `Mass::mass` also checks that
    /// derived mass, if Some, is same as `self.mass`.
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based on other component parameters (e.g. power and power
    /// density, sum of fields containing mass)
    fn set_mass(&mut self, new_mass: Option<si::Mass>) -> anyhow::Result<()>;

    /// Returns derived mass (e.g. sum of mass fields, or
    /// calculation involving mass specific properties).  If
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets all fields that are used in calculating derived mass to `None`.
    /// Does not touch `self.mass`.
    fn expunge_mass_fields(&mut self);
}

/// Provides functions for solving powertrain source component -- e.g. [FuelConverter], [super::ReversibleEnergyStorage]
pub trait PowertrainSource {
    /// Returns maximum possible traction-related power this component/system
    /// can produce, accounting for any aux-related power required.
    /// # Arguments
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    fn get_curr_pwr_tract_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power>;

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: tractive power output required to achieve presribed speed
    /// - `pwr_aux`: component-specific aux power demand (e.g. mechanical power if from engine/FC)
    /// - `enabled`: whether component is actively running
    /// - `dt`: time step size
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()>;
}

/// Provides functions for solving powertrain through component -- e.g. TODO: put something here  
/// Any component that accepts input power and provides output power but is not itself a power source.  
pub trait PowertrainThrough {
    /// Returns maximum possible traction-related power this component/system
    /// can produce, accounting for any aux-related power required.
    /// # Arguments
    /// - `pwr_in_pos_max`: positive-traction-related power available to this component
    /// - `pwr_in_neg_max`: negative-traction-related power available to this component
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    fn get_curr_pwr_tract_out_max(
        &mut self,
        pwr_in_pos_max: si::Power,
        pwr_in_neg_max: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power>;

    /// Solves for this powertrain system/component efficiency and sets/returns power output values.
    /// # Arguments
    /// - `pwr_out_req`: traction-related power output required
    /// - `pwr_aux`: component-specific aux power demand (e.g. mechanical power if from engine/FC)
    /// - `dt`: time step size
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()>;
}
