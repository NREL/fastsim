use super::{braking_point::BrakingPoints, friction_brakes::*, train_imports::*};
use crate::imports::*;
use crate::track::{LinkPoint, Location};

#[altrios_api(
    #[new]
    fn __new__(
        link_idx: LinkIdx,
        time_seconds: f64,
    ) -> Self {
        Self::new(
            link_idx,
            time_seconds * uc::S
        )
    }
)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize, SerdeAPI)]
pub struct LinkIdxTime {
    pub link_idx: LinkIdx,
    pub time: si::Time,
}

impl LinkIdxTime {
    pub fn new(link_idx: LinkIdx, time: si::Time) -> Self {
        Self { link_idx, time }
    }
}

#[altrios_api(
    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> PyResult<Option<usize>> {
        Ok(self.get_save_interval())
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool)  -> f64 {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> f64 {
        self.get_net_energy_res(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> f64 {
        self.get_energy_fuel(annualize).get::<si::joule>()
    }

    #[pyo3(name = "walk")]
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[classmethod]
    #[pyo3(name = "valid")]
    fn valid_py(_cls: &PyType) -> Self {
        Self::valid()
    }

    #[pyo3(name = "extend_path")]
    pub fn extend_path_py(&mut self, network_file_path: String, link_path: Vec<LinkIdx>) -> anyhow::Result<()> {
        let network = Vec::<Link>::from_file(&network_file_path).unwrap();
        network.validate().unwrap();

        self.extend_path(&network, &link_path)?;
        Ok(())
    }
)]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct SpeedLimitTrainSim {
    #[api(skip_set)]
    pub train_id: String,
    pub origs: Vec<Location>,
    pub dests: Vec<Location>,
    pub loco_con: Consist,
    pub state: TrainState,
    #[api(skip_set, skip_get)]
    pub train_res: TrainRes,
    #[api(skip_set)]
    pub path_tpc: PathTpc,
    #[api(skip_set)]
    pub braking_points: BrakingPoints,
    pub fric_brake: FricBrake,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: TrainStateHistoryVec,
    #[api(skip_set, skip_get)]
    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
}

impl SpeedLimitTrainSim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        train_id: String,
        origs: &[Location],
        dests: &[Location],
        loco_con: Consist,
        state: TrainState,
        train_res: TrainRes,
        path_tpc: PathTpc,
        fric_brake: FricBrake,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> Self {
        let mut train_sim = Self {
            train_id,
            origs: origs.to_vec(),
            dests: dests.to_vec(),
            loco_con,
            state,
            train_res,
            path_tpc,
            braking_points: Default::default(),
            fric_brake,
            history: Default::default(),
            save_interval,
            simulation_days,
            scenario_year,
        };
        train_sim.set_save_interval(save_interval);

        train_sim
    }

    /// Returns the scaling factor to be used when converting partial-year
    /// simulations to a full year of output metrics.
    pub fn get_scaling_factor(&self, annualize: bool) -> f64 {
        if annualize {
            match self.simulation_days {
                Some(val) => 365.25 / val as f64,
                None => 365.25,
            }
        } else {
            1.0
        }
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> f64 {
        self.state.mass_freight.get::<si::megagram>()
            * self.state.total_dist.get::<si::kilometer>()
            * self.get_scaling_factor(annualize)
    }

    pub fn get_energy_fuel(&self, annualize: bool) -> si::Energy {
        self.loco_con.get_energy_fuel() * self.get_scaling_factor(annualize)
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> si::Energy {
        self.loco_con.get_net_energy_res() * self.get_scaling_factor(annualize)
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.save_interval = save_interval;
        self.loco_con.set_save_interval(save_interval);
        self.fric_brake.save_interval = save_interval;
    }
    pub fn get_save_interval(&self) -> Option<usize> {
        self.save_interval
    }

    pub fn extend_path(&mut self, network: &[Link], link_path: &[LinkIdx]) -> anyhow::Result<()> {
        self.path_tpc.extend(network, link_path)?;
        self.recalc_braking_points()?;
        Ok(())
    }
    pub fn clear_path(&mut self) {
        // let link_point_del = self.path_tpc.clear(self.state.offset_back);
        // self.train_res.fix_cache(&link_point_del);
    }

    pub fn finish(&mut self) {
        self.path_tpc.finish()
    }
    pub fn is_finished(&self) -> bool {
        self.path_tpc.is_finished()
    }
    pub fn offset_begin(&self) -> si::Length {
        self.path_tpc.offset_begin()
    }
    pub fn offset_end(&self) -> si::Length {
        self.path_tpc.offset_end()
    }
    pub fn link_idx_last(&self) -> Option<&LinkIdx> {
        self.path_tpc.link_idx_last()
    }
    pub fn link_points(&self) -> &[LinkPoint] {
        self.path_tpc.link_points()
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        self.solve_step()
            .map_err(|err| err.context(format!("time step: {}", self.state.i)))?;
        self.save_state();
        self.loco_con.step();
        self.fric_brake.step();
        self.state.i += 1;
        Ok(())
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        self.loco_con
            .set_cat_power_limit(&self.path_tpc, self.state.offset);
        self.loco_con.set_cur_pwr_max_out(None, self.state.dt)?;
        self.train_res
            .update_res::<{ Dir::Fwd }>(&mut self.state, &self.path_tpc)?;
        self.solve_required_pwr()?;
        self.loco_con.solve_energy_consumption(
            self.state.pwr_whl_out,
            self.state.dt,
            Some(true),
        )?;
        Ok(())
    }

    fn save_state(&mut self) {
        if let Some(interval) = self.save_interval {
            if self.state.i % interval == 0 || 1 == self.state.i {
                self.history.push(self.state);
                self.loco_con.save_state();
                self.fric_brake.save_state();
            }
        }
    }

    /// Walks until getting to the end of the path
    fn walk_internal(&mut self) -> anyhow::Result<()> {
        while self.state.offset < self.path_tpc.offset_end() - 1000.0 * uc::FT
            || (self.state.offset < self.path_tpc.offset_end()
                && self.state.velocity != si::Velocity::ZERO)
        {
            self.step()?;
        }
        Ok(())
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward until it reaches destination.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state();
        self.walk_internal()
    }

    /// Iterates `save_state` and `step` until offset >= final offset --
    /// i.e. moves train forward and extends path TPC until it reaches destination.
    pub fn walk_timed_path(
        &mut self,
        network: &[Link],
        timed_path: &[LinkIdxTime],
    ) -> anyhow::Result<()> {
        if timed_path.is_empty() {
            bail!("Timed path cannot be empty!");
        }

        self.save_state();
        let mut idx_prev = 0;
        while idx_prev != timed_path.len() - 1 {
            let mut idx_next = idx_prev + 1;
            while idx_next + 1 < timed_path.len() - 1 && timed_path[idx_next].time < self.state.time
            {
                idx_next += 1;
            }
            let time_extend = timed_path[idx_next - 1].time;
            self.extend_path(
                network,
                &timed_path[idx_prev..idx_next]
                    .iter()
                    .map(|x| x.link_idx)
                    .collect::<Vec<LinkIdx>>(),
            )?;
            idx_prev = idx_next;
            while self.state.time < time_extend {
                self.step()?;
            }
        }

        self.walk_internal()
    }

    /// Sets power requirements based on:
    /// - rolling resistance
    /// - drag
    /// - inertia
    /// - target acceleration
    pub fn solve_required_pwr(&mut self) -> anyhow::Result<()> {
        let res_net = self.state.res_net();

        // Verify that train can slow down
        if self.fric_brake.force_max + res_net <= si::Force::ZERO {
            bail!("Train [TODO: put train id here] does not have sufficient braking to slow down at time{:?}.
            Fric brake force = {:?}.
            Net resistance = {:?}",
            self.state.time,
            self.fric_brake.force_max,
            res_net
        );
        }

        // TODO: Validate that this makes sense considering friction brakes
        let (speed_limit, speed_target) = self.braking_points.calc_speeds(
            self.state.offset,
            self.state.velocity,
            self.fric_brake.ramp_up_time * self.fric_brake.ramp_up_coeff,
        );
        self.state.speed_limit = speed_limit;
        self.state.speed_target = speed_target;

        let f_applied_target =
            res_net + self.state.mass_static * (speed_target - self.state.velocity) / self.state.dt;

        let pwr_pos_max =
            self.loco_con.state.pwr_out_max.min(si::Power::ZERO.max(
                self.state.pwr_whl_out + self.loco_con.state.pwr_rate_out_max * self.state.dt,
            ));
        let pwr_neg_max = self.loco_con.state.pwr_dyn_brake_max.max(si::Power::ZERO);
        ensure!(
            pwr_pos_max >= si::Power::ZERO,
            format_dbg!(pwr_pos_max >= si::Power::ZERO)
        );
        let time_per_mass = self.state.dt / self.state.mass_static;

        // Concept: calculate the final speed such that the worst case
        // (i.e. maximum) acceleration force does not exceed `power_max`
        // Base equation: m * (v_max - v_curr) / dt = p_max / v_max – f_res
        let v_max = 0.5
            * (self.state.velocity - res_net * time_per_mass
                + ((self.state.velocity - res_net * time_per_mass)
                    * (self.state.velocity - res_net * time_per_mass)
                    + 4.0 * time_per_mass * pwr_pos_max)
                    .sqrt());

        // Final v_max value should also be bounded by speed_target
        // maximum achievable positive tractive force
        let f_pos_max = self
            .loco_con
            .force_max()?
            .min(pwr_pos_max / speed_target.min(v_max));
        // Verify that train has sufficient power to move
        if self.state.velocity < uc::MPH * 0.1 && f_pos_max <= res_net {
            bail!(
                "{}\nTrain does not have sufficient power to move!\nforce_max={:?},\nres_net={:?},\ntrain_state={:?}", // ,\nlink={:?}
                format_dbg!(),
                f_pos_max,
                res_net,
                self.state,
                // self.path_tpc
            );
        }

        self.fric_brake.set_cur_force_max_out(self.state.dt)?;

        // Transition speed between force and power limited negative traction
        let v_neg_trac_lim: si::Velocity =
            self.loco_con.state.pwr_dyn_brake_max / self.loco_con.force_max()?;

        // TODO: Make sure that train handling rules consist dynamic braking force limit is respected!
        let f_max_consist_regen_dyn = if self.state.velocity > v_neg_trac_lim {
            // If there is enough braking to slow down at v_max
            let f_max_dyn_fast = self.loco_con.state.pwr_dyn_brake_max / v_max;
            if res_net + self.fric_brake.state.force_max_curr + f_max_dyn_fast >= si::Force::ZERO {
                self.loco_con.state.pwr_dyn_brake_max / v_max //self.state.velocity
            } else {
                f_max_dyn_fast
            }
        } else {
            self.loco_con.force_max()?
        };

        // total impetus force applied to control train speed
        let f_applied = f_pos_max.min(
            f_applied_target.max(-self.fric_brake.state.force_max_curr - f_max_consist_regen_dyn),
        );

        let vel_change = time_per_mass * (f_applied - res_net);
        let vel_avg = self.state.velocity + 0.5 * vel_change;

        self.state.pwr_res = res_net * vel_avg;
        self.state.pwr_accel = self.state.mass_adj / (2.0 * self.state.dt)
            * ((self.state.velocity + vel_change) * (self.state.velocity + vel_change)
                - self.state.velocity * self.state.velocity);

        self.state.time += self.state.dt;
        self.state.offset += self.state.dt * vel_avg;
        self.state.total_dist += (self.state.dt * vel_avg).abs();
        self.state.velocity += vel_change;
        if utils::almost_eq_uom(&self.state.velocity, &speed_target, None) {
            self.state.velocity = speed_target;
        }

        // Questions:
        // - do we need to update the brake points model to account for ramp-up time?
        // - do we need a brake model?
        //     - how do we respect ramp up rates of friction brakes?
        //     - how do we respect ramp down rates of friction brakes?
        // - how do we control the friction brakes?
        // - how do we make sure that the consist does not exceed e-drive braking capability?
        // - does friction braking get sent from both ends of the train?  Yes, per Tyler and
        //     Nathan, maybe from locomotive and end-of-train (EOT) device.  BNSF says
        //     their EOTs don't send a signal from the end of the train.  There may or may not
        //     be locomotives at the end of the train. Sometimes they run 4 at head and none on rear,
        //     sometimes 2 in front and 2 in rear.  Could base it on tonnage and car count.

        // TODO: figure out some reasonable split of regen and friction braking
        // and make sure it's applied here

        // Dynamic vs friction brake apportioning controls
        // These controls need to make sure that adequate braking force is achieved
        // which means they need to respect the dynamics of the friction braking system.
        // They also need to ensure that a reasonable amount of energy is available for
        // regenerative braking.

        // Whenever braking starting happens, do as much as possible with dyn/regen (respecting limitations of track, traction, etc.),
        // and when braking needs exceed consist capabilities, then add in friction braking.
        // Retain level of friction braking until braking is no longer needed and won't be
        // needed for some time.

        let f_consist = if f_applied >= si::Force::ZERO {
            self.fric_brake.state.force = si::Force::ZERO;
            f_applied
        } else {
            let f_consist = f_applied + self.fric_brake.state.force;
            // If the friction brakes should be released, don't add power
            if f_consist >= si::Force::ZERO {
                si::Force::ZERO
            }
            // If the current friction brakes and consist regen + dyn can handle things, don't add friction braking
            else if f_consist + f_max_consist_regen_dyn >= si::Force::ZERO {
                f_consist
            }
            // If the friction braking must increase, max out the regen dyn first
            else {
                self.fric_brake.state.force = -(f_applied + f_max_consist_regen_dyn);
                ensure!(
                    utils::almost_le_uom(
                        &self.fric_brake.state.force,
                        &self.fric_brake.state.force_max_curr,
                        None
                    ),
                    "Too much force requested from friction brake! Req={:?}, max={:?}",
                    self.fric_brake.state.force,
                    self.fric_brake.state.force_max_curr,
                );
                -f_max_consist_regen_dyn
            }
        };

        self.state.pwr_whl_out = f_consist * self.state.velocity;

        // this allows for float rounding error overshoot
        ensure!(
            utils::almost_le_uom(&self.state.pwr_whl_out, &pwr_pos_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max positive power! pwr_whl_out={:?}, pwr_pos_max={:?}",
            format_dbg!(utils::almost_le_uom(&self.state.pwr_whl_out, &pwr_pos_max, Some(1.0e-7))),
            self.state.pwr_whl_out,
            pwr_pos_max)
        );
        ensure!(
            utils::almost_le_uom(&-self.state.pwr_whl_out, &pwr_neg_max, Some(1.0e-7)),
            format!("{}\nPower wheel out is larger than max negative power! pwr_whl_out={:?}, pwr_neg_max={:?}
            {:?}\n{:?}\n{:?}\n{:?}",
            format_dbg!(utils::almost_le_uom(&-self.state.pwr_whl_out, &pwr_neg_max, Some(1.0e-7))),
            -self.state.pwr_whl_out,
            pwr_neg_max,
            self.state.velocity,
            self.fric_brake.state.force * self.state.velocity,
            vel_change,
        res_net)
        );
        self.state.pwr_whl_out = self.state.pwr_whl_out.max(-pwr_neg_max).min(pwr_pos_max);

        self.state.energy_whl_out += self.state.pwr_whl_out * self.state.dt;
        if self.state.pwr_whl_out >= 0. * uc::W {
            self.state.energy_whl_out_pos += self.state.pwr_whl_out * self.state.dt;
        } else {
            self.state.energy_whl_out_neg -= self.state.pwr_whl_out * self.state.dt;
        }

        Ok(())
    }

    fn recalc_braking_points(&mut self) -> anyhow::Result<()> {
        self.braking_points.recalc(
            &self.state,
            &self.fric_brake,
            &self.train_res,
            &self.path_tpc,
        )
    }
}

impl Default for SpeedLimitTrainSim {
    fn default() -> Self {
        let mut slts = Self {
            train_id: Default::default(),
            origs: Default::default(),
            dests: Default::default(),
            loco_con: Default::default(),
            state: TrainState::valid(),
            train_res: TrainRes::valid(),
            path_tpc: PathTpc::default(),
            braking_points: Default::default(),
            fric_brake: Default::default(),
            history: Default::default(),
            save_interval: None,
            simulation_days: None,
            scenario_year: None,
        };
        slts.set_save_interval(None);
        slts
    }
}

impl Valid for SpeedLimitTrainSim {
    fn valid() -> Self {
        let mut train_sim = Self::default();
        train_sim.path_tpc = PathTpc::valid();
        train_sim.recalc_braking_points().unwrap();
        train_sim
    }
}

pub fn speed_limit_train_sim_fwd() -> SpeedLimitTrainSim {
    let mut speed_limit_train_sim = SpeedLimitTrainSim::valid();
    speed_limit_train_sim.path_tpc = PathTpc::new(TrainParams::valid());
    speed_limit_train_sim.origs = vec![
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(96),
            is_front_end: Default::default(),
            grid_region: "CAMXc".into(),
        },
        Location {
            location_id: "Barstow".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(634),
            is_front_end: Default::default(),
            grid_region: "CAMXc".into(),
        },
    ];
    speed_limit_train_sim.dests = vec![
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(288),
            is_front_end: Default::default(),
            grid_region: "CAMXc".into(),
        },
        Location {
            location_id: "Stockton".into(),
            offset: si::Length::ZERO,
            link_idx: LinkIdx::new(826),
            is_front_end: Default::default(),
            grid_region: "CAMXc".into(),
        },
    ];
    speed_limit_train_sim
}

pub fn speed_limit_train_sim_rev() -> SpeedLimitTrainSim {
    let mut sltsr = speed_limit_train_sim_fwd();
    std::mem::swap(&mut sltsr.origs, &mut sltsr.dests);
    sltsr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    // TODO: Add more SpeedLimitTrainSim cases
    impl Cases for SpeedLimitTrainSim {}

    #[test]
    fn test_speed_limit_train_sim() {
        let mut train_sim = SpeedLimitTrainSim::valid();
        train_sim.walk().unwrap();
    }
}
