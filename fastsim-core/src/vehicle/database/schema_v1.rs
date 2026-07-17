use super::*;

/// Database organizational schema version 1 for the `fastsim-vehicles` repository.
///
/// Serializes to/from an 8-segment path-segment string, e.g. `"v1/fastsim-3/conv/ford/fusion/2012/base/v1"`
///
/// Segments in order:
/// 1. `v1` — schema version marker
/// 2. `fastsim-{N}` — FASTSim version
/// 3. `{powertrain}` — powertrain type (e.g., "conv", "hev", "phev", "bev")
/// 4. `{make}` — vehicle make
/// 5. `{model}` — vehicle model (may include trim information)
/// 6. `{year}` — model year or year range
/// 7. `{variant}` — variant/feature configuration (e.g., "base")
/// 8. `v{N}` — model revision/version for corrections
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
    /// Formats as: `v1/fastsim-{N}/{powertrain}/{make}/{model}/{year}/{variant}/v{N}`
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
    /// Expected format (8 segments):
    /// `v1/fastsim-{N}/{powertrain}/{make}/{model}/{year}/{variant}/v{N}`
    ///
    /// # Errors
    /// Returns an error if:
    /// - String doesn't have exactly 8 segments
    /// - Missing required prefixes ("v1" at start, "fastsim-" in segment 2, "v" in segment 8)
    /// - Version numbers (fastsim_version or revision) are not valid u32 values
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
        Self::new(
            fastsim_version,
            powertrain,
            make,
            model,
            year,
            variant,
            revision,
        )
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
    /// Returns in order:
    /// 1. "v1" — schema version marker
    /// 2. "fastsim-{N}" — FASTSim version
    /// 3. powertrain — e.g., "conv", "hev", "phev", "bev"
    /// 4. make — e.g., "ford", "tesla"
    /// 5. model — e.g., "fusion", "model-3"
    /// 6. year — e.g., "2012", "2020"
    /// 7. variant — e.g., "base", "trim-package"
    /// 8. "v{N}" — revision/version number
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

    /// Build a local file path by joining the schema's path-segment string with a base directory and extension.
    ///
    /// Example: `base_dir/v1/fastsim-3/conv/ford/fusion/2012/base/v1.yaml`
    pub fn build_filepath<P: AsRef<Path>>(
        &self,
        base_dir: P,
        extension: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        Ok(base_dir.as_ref().join(format!("{}.{}", self, extension)))
    }

    /// Build a remote URL by joining the schema's path-segment string with a base URL and extension.
    ///
    /// If `base_url` is None, uses the default GitHub raw content URL.
    /// Example: `https://raw.githubusercontent.com/.../v1/fastsim-3/conv/ford/fusion/2012/base/v1.yaml`
    pub fn build_url(&self, base_url: Option<&str>, extension: &str) -> anyhow::Result<String> {
        Ok(format!(
            "{}/{}.{}",
            base_url
                .map(|s| s.trim_end_matches('/'))
                .unwrap_or(DEFAULT_DB_URL),
            self,
            extension
        ))
    }
}

impl Vehicle {
    /// Load a vehicle from a database schema (v1).
    ///
    /// This is the unified core loading function that handles both local and remote databases.
    /// Local/remote detection is automatic based on the `db_path_or_url` parameter:
    /// - `None` or URLs starting with `http://`/`https://` → remote loading
    /// - File paths or other strings → local file loading
    ///
    /// The schema determines which vehicle file to load:
    /// - Schema path: `v1/fastsim-{ver}/{powertrain}/{make}/{model}/{year}/{variant}/v{rev}`
    /// - Local file: `{db_path_or_url}/v1/fastsim-.../v{rev}.{extension}`
    /// - Remote URL: `{db_path_or_url}/v1/fastsim-.../v{rev}.{extension}`
    ///
    /// # Parameters
    /// - `db_path_or_url`: Database root path (local) or base URL (remote), or `None` for default remote
    /// - `schema`: Parsed or constructed `DatabaseSchemaV1` object
    /// - `extension`: File extension (typically "yaml" or "json")
    /// - `skip_init`: If false, runs vehicle initialization after loading
    pub fn from_db_schema_v1(
        db_path_or_url: Option<&str>,
        schema: DatabaseSchemaV1,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let is_remote = db_path_or_url
            .map(|u| u.starts_with("http://") || u.starts_with("https://"))
            .unwrap_or(true);

        let mut veh = if is_remote {
            #[cfg(feature = "web")]
            {
                let resolved_url = schema.build_url(db_path_or_url, extension)?;
                Self::from_url(resolved_url, skip_init)?
            }
            #[cfg(not(feature = "web"))]
            {
                anyhow::bail!("Remote DB loading requires FASTSim built with the `web` feature")
            }
        } else {
            let path = schema.build_filepath(db_path_or_url.unwrap(), extension)?;
            Self::from_file(path, skip_init)?
        };

        if !skip_init {
            veh.init()?;
        }
        Ok(veh)
    }

    /// Load a vehicle using a pre-serialized schema path string.
    ///
    /// Convenience wrapper that parses the path string into a `DatabaseSchemaV1` schema object
    /// and delegates to `from_db_schema_v1` for loading.
    ///
    /// # Parameters
    /// - `db_path_or_url`: Database root (local path or remote base URL)
    /// - `path`: Full schema path string: `v1/fastsim-{ver}/{powertrain}/{make}/{model}/{year}/{variant}/v{rev}`
    /// - `extension`: File extension (typically "yaml")
    /// - `skip_init`: If false, runs vehicle initialization
    ///
    /// # Errors
    /// Returns an error if the path string cannot be parsed as a valid schema.
    pub fn from_db_path_v1(
        db_path_or_url: Option<&str>,
        path: &str,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let schema = DatabaseSchemaV1::from_str(path)?;
        Self::from_db_schema_v1(db_path_or_url, schema, extension, skip_init)
    }

    /// Load a vehicle using individual schema fields.
    ///
    /// Convenience wrapper that constructs a `DatabaseSchemaV1` schema object from individual
    /// fields and delegates to `from_db_schema_v1` for loading.
    ///
    /// # Parameters
    /// - `db_path_or_url`: Database root (local path or remote base URL)
    /// - `fastsim_version`: FASTSim version (e.g., 3)
    /// - `powertrain`: Powertrain type (e.g., "conv", "hev", "phev", "bev")
    /// - `make`: Vehicle make (e.g., "ford", "tesla")
    /// - `model`: Vehicle model (e.g., "fusion", "model-3")
    /// - `year`: Model year or range (e.g., "2012", "2020-2023")
    /// - `variant`: Variant descriptor (e.g., "base")
    /// - `revision`: Model revision number
    /// - `extension`: File extension (typically "yaml")
    /// - `skip_init`: If false, runs vehicle initialization
    ///
    /// # Errors
    /// Returns an error if any field contains a `/` character or if file/URL loading fails.
    pub fn from_db_fields_v1(
        db_path_or_url: Option<&str>,
        fastsim_version: u32,
        powertrain: &str,
        make: &str,
        model: &str,
        year: &str,
        variant: &str,
        revision: u32,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let schema = DatabaseSchemaV1::new(
            fastsim_version,
            powertrain.to_string(),
            make.to_string(),
            model.to_string(),
            year.to_string(),
            variant.to_string(),
            revision,
        )?;
        Self::from_db_schema_v1(db_path_or_url, schema, extension, skip_init)
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
    fn test_new_rejects_slash_in_fields() {
        assert!(DatabaseSchemaV1::new(
            3,
            "conv".to_string(),
            "ford".to_string(),
            "f-150/raptor".to_string(),
            "2012".to_string(),
            "base".to_string(),
            1,
        )
        .is_err());
        assert!(DatabaseSchemaV1::new(
            3,
            "conv".to_string(),
            "ford".to_string(),
            "fusion".to_string(),
            "2012".to_string(),
            "base/trim".to_string(),
            1,
        )
        .is_err());
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
        let actual = schema.build_url(None, "yaml").unwrap();
        let expected =
            "https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main/v1/fastsim-3/conv/ford/fusion/2012/base/v1.yaml"
                .to_string();
        assert_eq!(actual, expected);
    }

    #[test]
    #[cfg(feature = "web")]
    fn test_from_db_remote_v1() {
        let schema = sample_schema();
        assert!(Vehicle::from_db_fields_v1(
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
