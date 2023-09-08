use super::*;

/// Trait for ensuring consistency among locomotives and consists
#[enum_dispatch]
pub trait VehicleTrait {
    /// returns current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()>;
    /// Save current state
    fn save_state(&mut self) {
        unimplemented!();
    }
    /// Step counter
    fn step(&mut self) {
        unimplemented!();
    }
}
