use super::*;

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, Validate)]
pub struct Vehicle {
    pt: PowerTrainType,
}

#[enum_dispatch]
pub trait GetSetFC {
    fn get_fc(&self) -> Option<&powertrain::fuel_converter::FuelConverter>;
    fn set_fc(&mut self, fc: Option<powertrain::fuel_converter::FuelConverter>) -> anyhow::Result<()>;
}

impl GetSetFC for Conventional {
    fn get_fc(&self) -> Option<&powertrain::fuel_converter::FuelConverter> {
        Some(&self.fc)
    }
    fn set_fc(&mut self, fc: Option<powertrain::fuel_converter::FuelConverter>) -> anyhow::Result<()> {
        match fc {
            Some(fc) => {
                self.fc = fc;
                Ok(())
            }
            None => bail!("Expected `Some(fc)`."),
        }
    }
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
pub struct Conventional {
    fc: powertrain::fuel_converter::FuelConverter,
}
