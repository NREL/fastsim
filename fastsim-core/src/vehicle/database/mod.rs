use super::*;
use serde::{Deserialize, Serialize};

mod schema_v1;

pub use schema_v1::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Schema {
    V1(DatabaseSchemaV1),
}

pub const DEFAULT_DB_URL: &str =
    "https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main";
