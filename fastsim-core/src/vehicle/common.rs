use crate::imports::*;
use crate::{si, utils::TrackedState};

pub fn handle_fc_on_causes_for_propulsion_request(
    has_traction_power_request: &mut TrackedState<bool>,
    pwr_in_transmission: si::Power,
) -> anyhow::Result<()> {
    has_traction_power_request.update(pwr_in_transmission > si::Power::ZERO, || format_dbg!())?;
    Ok(())
}
