use crate::imports::*;
use crate::vehicle::powertrain::FuelConverter;
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

pub fn handle_fc_on_causes_for_temp(
    fc: &FuelConverter,
    temp_fc_forced_on: Option<si::Temperature>,
    temp_fc_allowed_off: Option<si::Temperature>,
    fc_temperature_too_low: &mut TrackedState<bool>,
) -> anyhow::Result<()> {
    let fc_temperature = if let Some(temp_ts) = fc.temperature() {
        Some(*temp_ts.get_fresh(|| format_dbg!())?)
    } else {
        None
    };
    let key = (
        fc_temperature,
        fc_temperature,
        temp_fc_forced_on,
        temp_fc_allowed_off,
    );
    match key {
        (None, None, None, None) => {
            fc_temperature_too_low.update(false, || format_dbg!())?;
        }
        (
            Some(temperature),
            Some(temp_prev),
            Some(temp_fc_forced_on),
            Some(temp_fc_allowed_off),
        ) => {
            fc_temperature_too_low.update(
                temperature < temp_fc_forced_on
                    || (temp_prev < temp_fc_forced_on && temperature < temp_fc_allowed_off),
                || format_dbg!(),
            )?;
        }
        _ => {
            bail!("{}\n`fc.temperature()`, `fc.temp_prev()`, `self.temp_fc_forced_on`, `self.temp_fc_allowed_off` must all be `None` or `Some`",
                    format_dbg!((
                        fc.temperature(),
                        temp_fc_forced_on,
                        temp_fc_allowed_off,
                    ))
                );
        }
    }
    Ok(())
}

pub fn handle_fc_on_causes_for_on_time(
    fc: &FuelConverter,
    fc_min_time_on: Option<si::Time>,
    on_time_too_short: &mut TrackedState<bool>,
) -> Result<(), anyhow::Error> {
    on_time_too_short.update(
        *fc.state.fc_on.get_stale(|| format_dbg!())?
            && *fc.state.time_on.get_stale(|| format_dbg!())?
                < fc_min_time_on.with_context(|| format_dbg!())?,
        || format_dbg!(),
    )?;
    Ok(())
}

pub fn handle_fc_on_causes_for_speed(
    vehicle_not_stopped: &mut TrackedState<bool>,
    speed: si::Velocity,
) -> anyhow::Result<()> {
    vehicle_not_stopped.update(speed.get::<si::meter_per_second>() > 1e-6, || format_dbg!())?;
    Ok(())
}

pub struct VehicleDynamicState {
    pub prev_speed: si::Velocity,
    pub speed: si::Velocity,
    pub dt: si::Time,
    pub dfco_allowed: bool,
    pub minimum_dfco_speed: si::Velocity,
    pub minimum_dfco_deceleration: si::Acceleration,
}

/// Determine if decel fuel cut-off (DFCO) is disabled based on vehicle
/// dynamics considerations (i.e., speed, acceleration). Note: considerations
/// related to whether the engine is too cold and such would be handled
/// elsewhere.
pub fn is_dfco_disabled_due_to_veh_dynamics(dynamic_state: &VehicleDynamicState) -> bool {
    let decel = (dynamic_state.speed - dynamic_state.prev_speed) / dynamic_state.dt;
    let is_accel = decel > si::Acceleration::ZERO;
    if !dynamic_state.dfco_allowed {
        true
    } else if dynamic_state.speed < dynamic_state.minimum_dfco_speed {
        true
    } else if dynamic_state.speed <= 1e-6 * uc::MPS {
        true
    } else if is_accel || decel > dynamic_state.minimum_dfco_deceleration {
        true
    } else {
        // NOTE: we **can** apply DFCO
        false
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn make_favorable_dfco_conditions() -> VehicleDynamicState {
        VehicleDynamicState {
            prev_speed: 40.0 * uc::MPH,
            speed: 36.0 * uc::MPH,
            dt: 1.0 * uc::S,
            dfco_allowed: true,
            minimum_dfco_speed: 20.0 * uc::MPH,
            minimum_dfco_deceleration: 0.0 * uc::MPS2,
        }
    }

    #[test]
    fn dfco_activates_when_all_conditions_are_good() {
        let s = make_favorable_dfco_conditions();
        let result = is_dfco_disabled_due_to_veh_dynamics(&s);
        assert_eq!(false, result);
    }

    #[test]
    fn dfco_cannot_be_active_if_speed_too_low() {
        let mut s = make_favorable_dfco_conditions();
        s.prev_speed = 10.0 * uc::MPH;
        s.speed = 8.0 * uc::MPH;
        let result = is_dfco_disabled_due_to_veh_dynamics(&s);
        assert_eq!(result, true);
    }

    #[test]
    fn dfco_cannot_be_active_if_not_decelerating() {
        let mut s = make_favorable_dfco_conditions();
        s.speed = s.prev_speed + 2.0 * uc::MPH;
        let result = is_dfco_disabled_due_to_veh_dynamics(&s);
        assert_eq!(true, result);
    }

    #[test]
    fn dfco_cannot_be_active_if_not_allowed() {
        let mut s = make_favorable_dfco_conditions();
        s.dfco_allowed = false;
        let result = is_dfco_disabled_due_to_veh_dynamics(&s);
        assert_eq!(true, result);
    }
}
