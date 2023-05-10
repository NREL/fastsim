use super::super::*;
#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate)]
pub struct FuelConverter {
    pwr_max: f64,
}
