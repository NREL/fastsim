use super::super::*;
#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate)]
pub struct FuelConverter {
    pub pwr_max: f64,
    orphaned: bool,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate, HistoryVec)]
pub struct FuelConverterState {
    pub pwr_max: f64,
    orphaned: bool
}
