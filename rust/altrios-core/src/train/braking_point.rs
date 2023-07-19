use super::{friction_brakes::FricBrake, train_imports::*};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct BrakingPoint {
    pub offset: si::Length,
    pub speed_limit: si::Velocity,
    pub speed_target: si::Velocity,
}

impl ObjState for BrakingPoint {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_fin(&mut errors, &self.speed_limit, "Speed limit");
        si_chk_num_fin(&mut errors, &self.speed_target, "Speed target");
        errors.make_err()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct BrakingPoints {
    points: Vec<BrakingPoint>,
    idx_curr: usize,
}

impl BrakingPoints {
    /// TODO: complete this doc string
    /// Arguments:
    /// - offset: si::Length -- ???
    /// - velocity: si::Velocity -- ???
    /// - adj_ramp_up_time: si::Time -- corrected ramp up time to account
    ///     for approximately linear brake build up

    pub fn calc_speeds(
        &mut self,
        offset: si::Length,
        velocity: si::Velocity,
        adj_ramp_up_time: si::Time,
    ) -> (si::Velocity, si::Velocity) {
        if self.points.first().unwrap().offset <= offset {
            self.idx_curr = 0;
        } else {
            while self.points[self.idx_curr - 1].offset <= offset {
                self.idx_curr -= 1;
            }
        }
        assert!(
            velocity <= self.points[self.idx_curr].speed_limit,
            "Speed limit violated! velocity={velocity:?}, speed_limit={:?}",
            self.points[self.idx_curr].speed_limit
        );

        // need to make a way for this to never decrease until a stop happens or maybe never at all
        // need to maybe save `offset_far`
        let offset_far = offset + velocity * adj_ramp_up_time;
        let mut speed_target = self.points[self.idx_curr].speed_target;
        let mut idx = self.idx_curr;
        while idx >= 1 && self.points[idx - 1].offset <= offset_far {
            speed_target = speed_target.min(self.points[idx - 1].speed_target);
            idx -= 1;
        }

        (self.points[self.idx_curr].speed_limit, speed_target)
    }
    pub fn recalc(
        &mut self,
        train_state: &TrainState,
        fric_brake: &FricBrake,
        train_res: &TrainRes,
        path_tpc: &PathTpc,
    ) -> anyhow::Result<()> {
        self.points.clear();
        self.points.push(BrakingPoint {
            offset: path_tpc.offset_end(),
            ..Default::default()
        });

        let mut train_state = *train_state;
        let mut train_res = train_res.clone();
        train_state.offset = path_tpc.offset_end();
        train_state.velocity = si::Velocity::ZERO;
        train_res.update_res::<{ Dir::Unk }>(&mut train_state, path_tpc)?;
        let speed_points = path_tpc.speed_points();
        let mut idx = path_tpc.speed_points().len();

        //Iterate backwards through all the speed points
        while 0 < idx {
            idx -= 1;
            if speed_points[idx].speed_limit.abs() > self.points.last().unwrap().speed_limit {
                // Iterate until breaking through the speed limit curve
                loop {
                    let bp_curr = *self.points.last().unwrap();

                    // Update speed limit
                    while bp_curr.offset <= speed_points[idx].offset {
                        idx -= 1;
                    }
                    let speed_limit = speed_points[idx].speed_limit.abs();

                    train_state.offset = bp_curr.offset;
                    train_state.velocity = bp_curr.speed_limit;
                    train_res.update_res::<{ Dir::Bwd }>(&mut train_state, path_tpc)?;

                    assert!(fric_brake.force_max + train_state.res_net() > si::Force::ZERO);
                    let vel_change = train_state.dt
                        * (fric_brake.force_max + train_state.res_net())
                        / train_state.mass_static;

                    // Exit after adding a couple of points if the next braking curve point will exceed the speed limit
                    if speed_limit < bp_curr.speed_limit + vel_change {
                        self.points.push(BrakingPoint {
                            offset: bp_curr.offset - train_state.dt * speed_limit,
                            speed_limit,
                            speed_target: bp_curr.speed_target,
                        });
                        if bp_curr.speed_limit == speed_points[idx].speed_limit.abs() {
                            break;
                        }
                    } else {
                        // Add normal point to braking curve
                        self.points.push(BrakingPoint {
                            offset: bp_curr.offset
                                - train_state.dt * (bp_curr.speed_limit + 0.5 * vel_change),
                            speed_limit: bp_curr.speed_limit + vel_change,
                            speed_target: bp_curr.speed_target,
                        });
                    }

                    // Exit if the braking point passed the beginning of the path
                    if self.points.last().unwrap().offset < path_tpc.offset_begin() {
                        break;
                    }
                }
            }
            self.points.push(BrakingPoint {
                offset: speed_points[idx].offset,
                speed_limit: speed_points[idx].speed_limit.abs(),
                speed_target: speed_points[idx].speed_limit.abs(),
            });
        }

        self.idx_curr = self.points.len() - 1;
        Ok(())
    }
}
