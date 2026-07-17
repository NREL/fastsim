use super::*;

/// Database organizational schema version 1 for the `fastsim-vehicles` repository.
///
/// Serializes to/from the path-segment string:
/// `v1/fastsim-{fastsim_version}/{powertrain}/{make}/{model}/{year}/{variant}/v{revision}`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct DatabaseSchemaV1 {
    /// FASTSim version
    pub fastsim_version: u32,
    /// Powertrain type (e.g., "conv", "hev", "phev", "bev")
    pub powertrain: String,
    /// Vehicle make
    pub make: String,
    /// Vehicle model (may include trim information)
    pub model: String,
    /// Vehicle model year (or range of years)
    pub year: String,
    /// Vehicle variant (describes what modeling features are active, etc.)
    pub variant: String,
    /// Model revision/version for correcting model-level issues over time
    pub revision: u32,
}

impl std::fmt::Display for DatabaseSchemaV1 {
    /// Display the schema as its path-segment string representation.
    ///
    /// Formats as: `v1/fastsim-{N}/make/model/year/variant/v{N}`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path_segments().join("/"))
    }
}

impl From<DatabaseSchemaV1> for String {
    fn from(s: DatabaseSchemaV1) -> Self {
        s.to_string()
    }
}

impl std::str::FromStr for DatabaseSchemaV1 {
    type Err = anyhow::Error;
    /// Parse a schema from its path-segment string representation.
    ///
    /// Expected format: `v1/fastsim-{N}/make/model/year/variant/v{N}`
    ///
    /// # Errors
    /// Returns an error if the string doesn't have exactly 7 segments,
    /// is missing required prefixes ("v1", "fastsim-", "v"), or has invalid numbers.
    fn from_str(s: &str) -> anyhow::Result<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        ensure!(
            parts.len() == 8,
            "expected 8 path segments, got {}: {s:?}",
            parts.len()
        );
        ensure!(
            parts[0] == "v1",
            "expected schema prefix 'v1', got {:?}",
            parts[0]
        );
        let fastsim_version = parts[1]
            .strip_prefix("fastsim-")
            .ok_or_else(|| anyhow!("expected 'fastsim-N' segment, got {:?}", parts[1]))?
            .parse::<u32>()
            .with_context(|| format!("invalid FASTSim version in {:?}", parts[1]))?;
        let powertrain = parts[2].to_string();
        let make = parts[3].to_string();
        let model = parts[4].to_string();
        let year = parts[5].to_string();
        let variant = parts[6].to_string();
        let revision = parts[7]
            .strip_prefix('v')
            .ok_or_else(|| anyhow!("expected 'vN' revision segment, got {:?}", parts[7]))?
            .parse::<u32>()
            .with_context(|| format!("invalid revision in {:?}", parts[7]))?;
        Ok(Self {
            fastsim_version,
            powertrain,
            make,
            model,
            year,
            variant,
            revision,
        })
    }
}

impl TryFrom<String> for DatabaseSchemaV1 {
    type Error = anyhow::Error;
    fn try_from(s: String) -> anyhow::Result<Self> {
        s.parse()
    }
}

impl DatabaseSchemaV1 {
    /// Construct a schema, validating that no field contains a `/`.
    ///
    /// # Errors
    /// Returns an error if `powertrain`, `make`, `model`, `year`, or `variant` contains a `/`,
    /// since that would corrupt the path-segment serialization and cause it to
    /// misparse (or fail to parse) on the way back in.
    pub fn new(
        fastsim_version: u32,
        powertrain: String,
        make: String,
        model: String,
        year: String,
        variant: String,
        revision: u32,
    ) -> anyhow::Result<Self> {
        for (field, value) in [
            ("powertrain", &powertrain),
            ("make", &make),
            ("model", &model),
            ("year", &year),
            ("variant", &variant),
        ] {
            ensure!(
                !value.contains('/'),
                "{field} must not contain '/', got {value:?}"
            );
        }
        Ok(Self {
            fastsim_version,
            powertrain,
            make,
            model,
            year,
            variant,
            revision,
        })
    }

    /// Build the ordered path segments as an 8-element array (without file extension).
    ///
    /// Returns: `["v1", "fastsim-{N}", powertrain, make, model, year, variant, "v{N}"]`
    fn path_segments(&self) -> [String; 8] {
        [
            "v1".to_string(),
            format!("fastsim-{}", self.fastsim_version),
            self.powertrain.clone(),
            self.make.clone(),
            self.model.clone(),
            self.year.clone(),
            self.variant.clone(),
            format!("v{}", self.revision),
        ]
    }

    /// Build a local file path with the given extension.
    pub fn build_filepath(
        &self,
        base_dir: &std::path::Path,
        extension: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        Ok(base_dir.join(format!("{}.{}", self, extension)))
    }

    /// Build a remote URL with the given extension.
    pub fn build_url(&self, base_url: &str, extension: &str) -> anyhow::Result<String> {
        Ok(format!(
            "{}/{}.{}",
            base_url.trim_end_matches('/'),
            self,
            extension
        ))
    }
}

impl Vehicle {
    /// Load a vehicle from a local database directory.
    pub fn from_db_local_v1(
        base_dir: &std::path::Path,
        fastsim_version: u32,
        powertrain: &str,
        make: &str,
        model: &str,
        year: &str,
        variant: &str,
        revision: u32,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Vehicle> {
        let schema = DatabaseSchemaV1::new(
            fastsim_version,
            powertrain.to_string(),
            make.to_string(),
            model.to_string(),
            year.to_string(),
            variant.to_string(),
            revision,
        )?;
        let path = schema.build_filepath(base_dir, extension)?;
        let mut veh = Self::from_file(path.clone(), skip_init).map_err(|err| {
            anyhow!(
                "{}: from_db_local_v1 failed for path '{}': {err}",
                format_dbg!(),
                path.display()
            )
        })?;
        if !skip_init {
            veh.init()?;
        }
        Ok(veh)
    }

    /// Load a vehicle from a remote database URL (requires `web` feature).
    #[cfg(feature = "web")]
    pub fn from_db_remote_v1(
        url: Option<&str>,
        fastsim_version: u32,
        powertrain: &str,
        make: &str,
        model: &str,
        year: &str,
        variant: &str,
        revision: u32,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Vehicle> {
        let schema = DatabaseSchemaV1::new(
            fastsim_version,
            powertrain.to_string(),
            make.to_string(),
            model.to_string(),
            year.to_string(),
            variant.to_string(),
            revision,
        )?;
        let resolved_url = schema.build_url(url.unwrap_or(DEFAULT_DB_URL), extension)?;
        let mut veh = Self::from_url(resolved_url.clone(), skip_init).map_err(|err| {
            anyhow!(
                "{}: from_db_remote_v1 failed for URL '{}': {err}",
                format_dbg!(),
                resolved_url
            )
        })?;
        if !skip_init {
            veh.init()?;
        }
        Ok(veh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> DatabaseSchemaV1 {
        DatabaseSchemaV1::new(
            3,
            "conv".to_string(),
            "ford".to_string(),
            "fusion".to_string(),
            "2012".to_string(),
            "base".to_string(),
            1,
        )
        .unwrap()
    }

    #[test]
    fn test_serde_round_trip() {
        let schema = sample_schema();
        let serialized = serde_json::to_string(&schema).unwrap();
        assert_eq!(serialized, "\"v1/fastsim-3/conv/ford/fusion/2012/base/v1\"");
        let deserialized: DatabaseSchemaV1 = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, schema);
    }

    #[test]
    fn test_to_string() {
        let schema = sample_schema();
        assert_eq!(
            String::from(schema),
            "v1/fastsim-3/conv/ford/fusion/2012/base/v1"
        );
    }

    #[test]
    fn test_from_str() {
        let s = "v1/fastsim-3/conv/ford/fusion/2012/base/v1";
        let schema = DatabaseSchemaV1::from_str(s).unwrap();
        assert_eq!(schema, sample_schema());
    }

    #[test]
    fn test_from_str_errors() {
        assert!(DatabaseSchemaV1::from_str("v2/fastsim-3/conv/ford/fusion/2012/base/v1").is_err());
        assert!(DatabaseSchemaV1::from_str("v1/fastsim-3/conv/ford/fusion/2012/base").is_err());
        assert!(DatabaseSchemaV1::from_str("v1/bad-3/conv/ford/fusion/2012/base/v1").is_err());
        assert!(DatabaseSchemaV1::from_str("v1/fastsim-3/conv/ford/fusion/2012/base/1").is_err());
    }

    #[test]
    fn test_build_filepath_output() {
        let base = std::path::Path::new("/tmp/vehicles-db");
        let schema = sample_schema();
        let actual = schema.build_filepath(base, "yaml").unwrap();
        let expected = base.join("v1/fastsim-3/conv/ford/fusion/2012/base/v1.yaml");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_build_url_output() {
        let schema = sample_schema();
        let actual = schema.build_url(DEFAULT_DB_URL, "yaml").unwrap();
        let expected =
            "https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main/v1/fastsim-3/conv/ford/fusion/2012/base/v1.yaml"
                .to_string();
        assert_eq!(actual, expected);
    }

    #[test]
    #[cfg(feature = "web")]
    fn test_from_db_remote_v1() {
        let schema = sample_schema();
        assert!(Vehicle::from_db_remote_v1(
            None,
            schema.fastsim_version,
            &schema.powertrain,
            &schema.make,
            &schema.model,
            &schema.year,
            &schema.variant,
            schema.revision,
            "yaml",
            false,
        )
        .is_ok());
    }
}
