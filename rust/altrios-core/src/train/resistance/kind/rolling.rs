use crate::imports::*;
use crate::train::TrainState;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Basic {
    ratio: si::Ratio,
}

impl Basic {
    pub fn new(ratio: si::Ratio) -> Self {
        Self { ratio }
    }
    pub fn calc_res(&mut self, state: &TrainState) -> si::Force {
        self.ratio * state.weight_static
    }
}
