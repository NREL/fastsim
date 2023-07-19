use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
#[altrios_api]
pub struct LinkIdxTime {
    time: si::Time,
    link_idx: LinkIdx,
}
