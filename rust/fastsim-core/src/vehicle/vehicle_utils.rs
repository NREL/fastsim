use super::*;

/// Trait for ensuring consistency among locomotives and consists
pub trait VehicleTrait {
    /// Sets current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery.
    ///
    /// # Arguments
    /// * `pwr_aux` - power required by auxilliary systems (e.g. HVAC, stereo)
    /// * `dt` - time step size
    fn set_cur_pwr_max_out(&mut self, pwr_aux: si::Power, dt: si::Time) -> anyhow::Result<()>;
    /// Save current state and propagate to fields that implement this trait
    fn save_state(&mut self) {
        unimplemented!();
    }
    /// Step counter
    fn step(&mut self) {
        unimplemented!();
    }
}
