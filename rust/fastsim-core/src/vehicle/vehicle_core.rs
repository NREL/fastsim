use super::*;
use proc_macros::{ApproxEq};

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, Validate)]
pub struct Vehicle {
    pt: PowerTrainType,
}

#[enum_dispatch]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum PowerTrainType {
    Conventional,
    // HEV,
    // PHEV,
    // BEV,
}

impl Default for PowerTrainType {
    fn default() -> Self {
        Self::Conventional(Default::default())
    }
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate)]
pub struct Conventional{
    fc: super::powertrain::fuel_converter::FuelConverter,
}

