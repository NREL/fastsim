use crate::imports::*;
use crate::train::TrainState;

// TODO implement method for elevation-dependent air density
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Basic {
    drag_area: si::Area,
}
impl Basic {
    pub fn new(drag_area: si::Area) -> Self {
        Self { drag_area }
    }
    pub fn calc_res(&mut self, state: &TrainState) -> si::Force {
        self.drag_area * uc::rho_air() * state.velocity * state.velocity
    }
}
