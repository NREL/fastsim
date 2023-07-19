use super::super::disp_imports::*;

#[readonly::make]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
pub(super) struct EstJoinPath {
    pub link_idx_match: LinkIdx,
    pub est_idx_next: EstIdx,
}

impl EstJoinPath {
    pub fn new(link_idx_match: LinkIdx, est_idx_next: EstIdx) -> Self {
        Self {
            link_idx_match,
            est_idx_next,
        }
    }
    pub fn has_space_match(&self) -> bool {
        self.link_idx_match.is_fake()
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
pub(super) struct SimpleState {
    pub time: si::Time,
    pub offset: si::Length,
    pub velocity: si::Velocity,
}

impl SimpleState {
    pub fn from_train_state(train_state: &TrainState) -> Self {
        Self {
            time: train_state.time,
            offset: train_state.offset,
            velocity: train_state.velocity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, SerdeAPI)]
pub(super) struct SavedSim {
    pub train_sim: Box<SpeedLimitTrainSim>,
    pub join_paths: Vec<EstJoinPath>,
    pub est_alt: EstTime,
}

impl SavedSim {
    /// Step the train sim forward and save appropriate state data in the movement
    pub fn update_movement(&mut self, movement: &mut Vec<SimpleState>) -> anyhow::Result<()> {
        let condition = |train_sim: &SpeedLimitTrainSim| -> bool {
            train_sim.state.offset < train_sim.offset_end() - uc::MI * 5.0
                || (
                    train_sim.is_finished()
                    // this needs to be reconsidered.  The issue is determining when SpeedLimitTrainSim is finished.
                        && train_sim.state.velocity > si::Velocity::ZERO
                    // train_sim.state.offset
                    //     < train_sim.path_tpc.offset_end() + train_sim.state.length
                )
        };

        movement.clear();
        movement.push(SimpleState::from_train_state(&self.train_sim.state));
        // TODO: Tighten up this bound using braking points.
        while condition(&self.train_sim) {
            self.train_sim.step()?;
            movement.push(SimpleState::from_train_state(&self.train_sim.state));
        }
        self.train_sim.clear_path();
        Ok(())
    }

    /// Check destinations and finish the path tpc if one is reached
    pub fn check_dests(&mut self, dests: &[Location]) {
        let link_idx_last = self.train_sim.link_idx_last().unwrap();
        for dest in dests {
            if *link_idx_last == dest.link_idx {
                self.train_sim.finish();
                return;
            }
        }
    }
}
