use super::*;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, IsVariant)]
/// HVAC operating mode
pub enum HvacMode {
    /// Heating, i.e. greater than setpoint temperature plus deadband
    Heating,
    /// Cooling, i.e. less than setpoint temperature minus deadband
    Cooling,
    /// Inside deadband, i.e. greater than or equal to setpoint temperature
    /// minus deadband and less than or equal to setpoint temperature plus
    /// deadband
    InsideDeadband,
    #[default]
    /// Inactive
    Inactive,
}
