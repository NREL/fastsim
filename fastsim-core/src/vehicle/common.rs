use crate::imports::*;
use crate::vehicle::vehicle_model::VehicleState;
use crate::{si, utils::TrackedState};

pub fn handle_fc_on_causes_for_propulsion_request(
    has_traction_power_request: &mut TrackedState<bool>,
    pwr_in_transmission: si::Power,
) -> anyhow::Result<()> {
    has_traction_power_request.update(pwr_in_transmission > si::Power::ZERO, || format_dbg!())?;
    Ok(())
}

pub fn handle_fc_on_causes_for_stopped_time(
    time_vehicle_stopped: &mut TrackedState<si::Time>,
    vehicle_not_stopped_long_enough: &mut TrackedState<bool>,
    veh_state: &VehicleState,
    dt: si::Time,
    time_delay_after_stop_until_fc_can_turn_off: Option<si::Time>,
) -> anyhow::Result<()> {
    let v_prev = *veh_state.speed_ach.get_stale(|| format_dbg!())?;
    let dt_stopped = *time_vehicle_stopped.get_stale(|| format_dbg!())?;
    let new_dt_stopped = if v_prev == si::Velocity::ZERO {
        dt_stopped + dt
    } else {
        0.0 * uc::S
    };
    time_vehicle_stopped.update(new_dt_stopped, || format_dbg!())?;
    let dt_delay = time_delay_after_stop_until_fc_can_turn_off.unwrap_or(0.0 * uc::S);
    vehicle_not_stopped_long_enough.update(new_dt_stopped < dt_delay, || format_dbg!())?;
    Ok(())
}
