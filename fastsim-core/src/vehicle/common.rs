use crate::imports::*;
use crate::vehicle::powertrain::FuelConverter;
use crate::vehicle::vehicle_model::VehicleState;
use crate::{si, utils::TrackedState};

pub trait StartStopControl {
    fn handle_fc_on_causes_for_propulsion_request(
        has_traction_power_request: &mut TrackedState<bool>,
        pwr_in_transmission: si::Power,
    ) -> anyhow::Result<()> {
        has_traction_power_request
            .update(pwr_in_transmission > si::Power::ZERO, || format_dbg!())?;
        Ok(())
    }

    fn handle_fc_on_causes_for_stopped_time(
        time_vehicle_stopped: &mut TrackedState<si::Time>,
        vehicle_not_stopped_long_enough: &mut TrackedState<bool>,
        veh_state: &VehicleState,
        dt: si::Time,
        time_delay_after_stop_until_fc_can_turn_off: Option<si::Time>,
        stopped_speed_threshold: si::Velocity,
    ) -> anyhow::Result<()> {
        let v_prev = *veh_state.speed_ach.get_stale(|| format_dbg!())?;
        let dt_stopped = *time_vehicle_stopped.get_stale(|| format_dbg!())?;
        let new_dt_stopped = if v_prev <= stopped_speed_threshold {
            dt_stopped + dt
        } else {
            0.0 * uc::S
        };
        time_vehicle_stopped.update(new_dt_stopped, || format_dbg!())?;
        let dt_delay = time_delay_after_stop_until_fc_can_turn_off.unwrap_or(0.0 * uc::S);
        vehicle_not_stopped_long_enough.update(new_dt_stopped < dt_delay, || format_dbg!())?;
        Ok(())
    }

    fn handle_fc_on_causes_for_temp(
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

    fn handle_fc_on_causes_for_on_time(
        fc: &FuelConverter,
        fc_min_time_on: Option<si::Time>,
        on_time_too_short: &mut TrackedState<bool>,
    ) -> anyhow::Result<()> {
        on_time_too_short.update(
            *fc.state.fc_on.get_stale(|| format_dbg!())?
                && *fc.state.time_on.get_stale(|| format_dbg!())?
                    < fc_min_time_on.with_context(|| format_dbg!())?,
            || format_dbg!(),
        )?;
        Ok(())
    }

    fn handle_fc_on_causes_for_speed(
        vehicle_not_stopped: &mut TrackedState<bool>,
        speed: si::Velocity,
        stopped_speed_threshold: si::Velocity,
    ) -> anyhow::Result<()> {
        vehicle_not_stopped.update(speed > stopped_speed_threshold, || format_dbg!())?;
        Ok(())
    }
}
