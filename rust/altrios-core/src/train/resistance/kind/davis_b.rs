use crate::imports::*;
use crate::train::TrainState;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Basic {
    davis_b: si::InverseVelocity,
}
impl Basic {
    pub fn new(davis_b: si::InverseVelocity) -> Self {
        Self { davis_b }
    }
    pub fn calc_res(&mut self, state: &TrainState) -> si::Force {
        self.davis_b * state.velocity * state.weight_static
    }
}
