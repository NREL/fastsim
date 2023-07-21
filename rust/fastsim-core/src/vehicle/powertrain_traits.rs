use super::*;

/// Traits defining power flow interfaces for electric machines
pub trait ElectricMachine {
    /// Sets current max power output given `pwr_in_max` from upstream component
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()>;
    /// Sets current max power output rate given `pwr_rate_in_max` from upstream component
    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate);
}

pub trait Mass {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based
    /// on other component parameters (e.g. power and power density)
    fn update_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()>;

    /// Checks if mass is consistent with other parameters
    fn check_mass_consistent(&self) -> anyhow::Result<()>;
}
