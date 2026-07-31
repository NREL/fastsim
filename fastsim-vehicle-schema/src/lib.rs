use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};
use std::str::FromStr;

pub mod v1;

pub use v1::*;

pub const DEFAULT_DB_URL: &str =
    "https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main";

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("invalid schema {raw:?}: unknown schema version prefix {version:?}. Expected a schema path starting with 'v1/'")]
    UnknownVersion { raw: String, version: String },

    #[error(transparent)]
    V1(#[from] SchemaV1Error),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Schema {
    V1(SchemaV1),
}

impl FromStr for Schema {
    type Err = SchemaError;
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let version = raw.split('/').next().unwrap_or_default();
        match version {
            "v1" => Ok(Schema::V1(SchemaV1::from_str(raw)?)),
            _ => Err(SchemaError::UnknownVersion {
                raw: raw.to_string(),
                version: version.to_string(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        raw.parse::<Schema>().map_err(|err| {
            let msg = match &err {
                SchemaError::UnknownVersion { .. } => err.to_string(), // already self-contained
                SchemaError::V1(_) => {
                    let mut msg = format!(
                        "{err}. Expected format: v1/fastsim-{{N}}/{{powertrain}}/{{make}}/{{model}}/{{year}}/{{variant}}/r{{N}}"
                    );
                    if let Some((current, suggested)) = SchemaV1::suggest_normalized_path(&raw) {
                        msg.push_str(&format!(
                            "\nCurrent path: {current}\nSuggested normalized path: {suggested}"
                        ));
                    }
                    msg
                }
            };
            D::Error::custom(msg)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_deserializes_valid_v1_path() {
        let raw = "\"v1/fastsim-3/conv/ford/fusion/2012/base/r1\"";
        let schema: Schema = serde_json::from_str(raw).unwrap();
        match schema {
            Schema::V1(v1) => {
                assert_eq!(v1.fastsim_version, 3);
                assert_eq!(v1.powertrain, "conv");
            }
        }
    }

    #[test]
    fn schema_deserialize_error_is_actionable() {
        let raw = "\"v1/bad-schema\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        println!("Error: {err}");
        assert!(err.contains("expected 8 path segments"));
        assert!(err.contains("Expected format:"));
    }

    #[test]
    fn schema_deserialize_error_suggests_normalized_path() {
        let raw = "\"v1/fastsim-3/BEV/Ford/F-150 Lightning/2024/Base/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        assert!(err.contains("Current path:"));
        assert!(err.contains("Suggested normalized path:"));
        assert!(err.contains("v1/fastsim-3/bev/ford/f-150-lightning/2024/base/r1"));
    }

    #[test]
    fn schema_deserialize_error_for_unknown_version() {
        let raw = "\"v9/fastsim-3/conv/ford/fusion/2012/base/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        assert!(err.contains("unknown schema version prefix"));
        assert!(err.contains("v1/"));
    }

    #[test]
    fn schema_rejects_unknown_schema_version() {
        let raw = "\"v2/fastsim-3/conv/ford/fusion/2012/base/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        assert!(err.contains("unknown schema version prefix"));
    }

    #[test]
    fn schema_deserialize_error_suggests_punctuation_cleanup() {
        let raw = "\"v1/fastsim-3/conv/ford/f--usio..n/2022/thermal_DFCO/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        assert!(err.contains("v1/fastsim-3/conv/ford/f-usio.n/2022/thermal-dfco/r1"));
    }
}
