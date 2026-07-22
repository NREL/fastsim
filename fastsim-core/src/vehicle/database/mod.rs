use super::*;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};

mod schema_v1;

pub use schema_v1::*;

pub const DEFAULT_DB_URL: &str =
    "https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Schema {
    V1(DatabaseSchemaV1),
}
impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let version = raw.split('/').next().unwrap_or_default();
        match version {
            "v1" => match deserialize_v1_schema(&raw) {
                Ok(schema) => Ok(schema),
                Err(err) => {
                    let mut msg = format!(
                        "invalid db_path schema {raw:?}: {err}. Expected format: v1/fastsim-{{N}}/{{powertrain}}/{{make}}/{{model}}/{{year}}/{{variant}}/r{{N}}"
                    );
                    if let Some((current, suggested)) = normalized_path_suggestion(&raw) {
                        msg.push_str(&format!(
                            "\nCurrent path: {current}\nSuggested normalized path: {suggested}"
                        ));
                    }
                    Err(D::Error::custom(msg))
                }
            },
            _ => Err(D::Error::custom(format!(
                "invalid db_path schema {raw:?}: unknown schema version prefix {version:?}. Expected a schema path starting with 'v1/'"
            ))),
        }
    }
}

fn deserialize_v1_schema(raw: &str) -> anyhow::Result<Schema> {
    let schema = DatabaseSchemaV1::from_str(raw)?;
    Ok(Schema::V1(schema))
}

fn normalized_path_suggestion(raw: &str) -> Option<(String, String)> {
    let (current_path, suggested_path) = match raw.split('/').next()? {
        "v1" => {
            // Intentionally bypass `DatabaseSchemaV1::new` validation to surface a
            // helpful before/after path suggestion for invalid schema strings.
            let current = DatabaseSchemaV1::parse_structural(raw).ok()?;

            let suggested = DatabaseSchemaV1 {
                fastsim_version: current.fastsim_version,
                powertrain: DatabaseSchemaV1::normalize_identifier(&current.powertrain),
                make: DatabaseSchemaV1::normalize_identifier(&current.make),
                model: DatabaseSchemaV1::normalize_identifier(&current.model),
                year: DatabaseSchemaV1::normalize_identifier(&current.year),
                variant: DatabaseSchemaV1::normalize_identifier(&current.variant),
                revision: current.revision,
            };
            (current.to_string(), suggested.to_string())
        }
        _ => return None,
    };

    if current_path != suggested_path {
        Some((current_path, suggested_path))
    } else {
        None
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
        assert!(err.contains("invalid db_path schema"));
        assert!(err.contains("Expected format:"));
    }

    #[test]
    fn schema_deserialize_error_suggests_slugified_path() {
        let raw = "\"v1/fastsim-3/Conv/Ford/F-150 Raptor/2012/Base Trim/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();
        assert!(err.contains("Current path:"));
        assert!(err.contains("Suggested normalized path:"));
        assert!(err.contains("v1/fastsim-3/conv/ford/f-150-raptor/2012/base-trim/r1"));
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
        let raw = "\"v1/fastsim-3/conv/ford/model.3/2022/base_variant/r1\"";
        let err = serde_json::from_str::<Schema>(raw).unwrap_err().to_string();

        assert!(err.contains("v1/fastsim-3/conv/ford/model.3/2022/base-variant/r1"));
    }
}
