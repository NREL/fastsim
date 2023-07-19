use crate::imports::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Basic {
    force: si::Force,
}
impl Basic {
    pub fn new(force: si::Force) -> Self {
        Self { force }
    }
    pub fn calc_res(&mut self) -> si::Force {
        self.force
    }
}
