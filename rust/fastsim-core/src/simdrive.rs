use super::drive_cycle::Cycle;
use super::vehicle::Vehicle;
use crate::air_properties::*;
use crate::imports::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI, HistoryMethods)]
pub struct SimDrive {
    #[has_state]
    veh: Vehicle,
    cyc: Cycle,
}

impl SimDrive {
    pub fn walk(&mut self) -> anyhow::Result<()> {
        ensure!(self.cyc.len() >= 2, format_dbg!(self.cyc.len() < 2));
        while self.veh.state.i < self.cyc.len() {
            self.solve_step()?;
            self.step();
        }
        Ok(())
    }

    /// Solves current time step
    /// # Arguments
    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        let i = self.veh.state.i;
        let dt = self.cyc.dt_at_i(i);
        self.veh.set_cur_pwr_max_out(self.veh.pwr_aux, dt)?;
        self.set_req_pwr(self.cyc.speed[i], dt)?;
        self.set_ach_speed(dt)?;
        Ok(())
    }

    /// Sets power required for given prescribed speed
    /// # Arguments
    /// - `speed`: prescribed or achieved speed
    /// - `speed_prev`: achieved speed at previous time step
    /// - `dt`: time step size
    pub fn set_req_pwr(&mut self, speed: si::Velocity, dt: si::Time) -> anyhow::Result<()> {
        // unwrap on `self.mass` is ok because any method of creating the vehicle should
        // automatically called `SerdeAPI::init`, which will ensure mass is some
        let i = self.veh.state.i;
        let speed_prev = self.veh.state.speed_ach_prev;
        self.veh.state.pwr_accel = self.veh.mass.unwrap() / (2.0 * dt)
            * (speed.powi(typenum::P2::new()) - speed_prev.powi(typenum::P2::new()));
        self.veh.state.pwr_ascent = uc::ACC_GRAV
            * match &self.cyc.grade {
                Some(grade) => grade[i],
                None => uc::R * 0.,
            }
            * self.veh.mass.unwrap()
            * (speed_prev + speed)
            / 2.0;
        self.veh.state.pwr_drag = 0.5
            * get_rho_air(None, None)
            * self.veh.drag_coef
            * self.veh.frontal_area
            * ((speed + speed_prev) / 2.0).powi(typenum::P3::new());

        Ok(())
    }

    /// Sets achieved speed based on known current max power
    /// # Arguments
    /// - `dt`: time step size
    pub fn set_ach_speed(&mut self, dt: si::Time) -> anyhow::Result<()> {
        todo!();
        Ok(())
    }
}

#[allow(dead_code)]
pub struct TraceMissTolerance {
    tol_dist: si::Length,
    tol_dist_frac: si::Ratio,
    tol_speed: si::Velocity,
    tol_speed_frac: si::Ratio,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::vehicle_model::tests::mock_f2_conv_veh;
    #[test]
    fn test_sim_drive() {
        let veh = mock_f2_conv_veh();
        let cyc = Cycle::from_file(todo!()).unwrap();
        let sd = SimDrive { veh, cyc };
        sd.walk().unwrap();
        assert!(sd.veh.state.i == sd.cyc.len());
        assert!(sd.veh.fuel_converter().unwrap().state.energy_fuel > uc::J * 0.);
    }
}
