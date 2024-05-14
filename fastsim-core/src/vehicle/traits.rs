//! Traits to be used in vehicle and nested subsystems and components

use super::*;

#[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq)]
/// Governs which side effect to trigger when setting mass
pub enum MassSideEffect {
    /// To be used when [MassSideEffect] is not applicable
    #[default]
    None,
    /// Set the extensive parameter -- e.g. energy, power -- as a side effect
    Extensive,
    /// Set the intensive parameter -- e.g. specific power, specific energy -- as a side effect
    Intensive,
}

impl TryFrom<String> for MassSideEffect {
    type Error = anyhow::Error;
    fn try_from(value: String) -> anyhow::Result<MassSideEffect> {
        let mass_side_effect = match value.as_str() {
            "None" => MassSideEffect::None,
            "Extensive" => MassSideEffect::Extensive,
            "Intensive" => MassSideEffect::Intensive,
            _ => {
                bail!(format!(
                    "`MassSideEffect` must be 'Intensive', 'Extensive', or 'None'. "
                ))
            }
        };
        Ok(mass_side_effect)
    }
}

pub trait Mass {
    /// Returns mass of Self, either from `self.mass` or
    /// the derived from fields that store mass data. `Mass::mass` also checks that
    /// derived mass, if Some, is same as `self.mass`.
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based on other component parameters (e.g. power and power
    /// density, sum of fields containing mass)
    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()>;

    /// Returns derived mass (e.g. sum of mass fields, or
    /// calculation involving mass specific properties).  If
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets all fields that are used in calculating derived mass to `None`.
    /// Does not touch `self.mass`.
    fn expunge_mass_fields(&mut self);
}
