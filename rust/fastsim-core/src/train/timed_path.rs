use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, SerdeAPI)]
#[pyo3_api]
pub struct LinkIdxTime {
    time: si::Time,
    link_idx: LinkIdx,
}
