use super::*;

pub use fastsim_schema::{VehicleSchema, VehicleSchemaV1};

impl Vehicle {
    /// Load a vehicle using a parsed v1 database schema from the schema crate.
    ///
    /// This is the core adapter that converts a `VehicleSchemaV1` into either a
    /// local file path or remote URL and then loads the vehicle with
    /// `Vehicle::from_file` or `Vehicle::from_url`.
    ///
    /// `db_path_or_url` controls local vs remote mode:
    /// - `None` or values starting with `http://`/`https://` use remote loading.
    /// - Other values are treated as local filesystem roots.
    ///
    /// The schema-to-path/URL mapping is delegated to crate methods:
    /// - `VehicleSchemaV1::build_filepath` for local loading.
    /// - `VehicleSchemaV1::build_url` for remote loading (defaults to
    ///   `DEFAULT_DB_URL` when `db_path_or_url` is `None`).
    ///
    /// # Errors
    /// Returns an error if URL/file loading fails, or if remote loading is
    /// requested without the `web` feature.
    pub fn from_db_schema_v1(
        db_path_or_url: Option<&str>,
        schema: VehicleSchemaV1,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let is_remote = db_path_or_url
            .map(|u| u.starts_with("http://") || u.starts_with("https://"))
            .unwrap_or(true);

        if is_remote {
            #[cfg(feature = "web")]
            {
                let resolved_url = schema.build_url(db_path_or_url, extension);
                Ok(Self::from_url(resolved_url, skip_init)?)
            }
            #[cfg(not(feature = "web"))]
            {
                anyhow::bail!("Remote DB loading requires FASTSim built with the `web` feature")
            }
        } else {
            let path = schema.build_filepath(db_path_or_url.unwrap(), extension);
            Ok(Self::from_file(path, skip_init)?)
        }
    }

    /// Load a vehicle from a serialized v1 schema path string.
    ///
    /// This convenience wrapper parses `path` with `VehicleSchemaV1::from_str`
    /// and delegates loading to `from_db_schema_v1`.
    ///
    /// # Errors
    /// Returns an error if schema parsing fails or delegated loading fails.
    pub fn from_db_path_v1(
        db_path_or_url: Option<&str>,
        path: &str,
        extension: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let schema: VehicleSchemaV1 = path.parse()?;
        Self::from_db_schema_v1(db_path_or_url, schema, extension, skip_init)
    }

    /// Load a vehicle from explicit v1 schema fields.
    ///
    /// This convenience wrapper validates/builds a `VehicleSchemaV1` via
    /// `VehicleSchemaV1::new` and delegates loading to `from_db_schema_v1`.
    ///
    /// # Errors
    /// Returns an error if any schema field fails validation or delegated
    /// loading fails.
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
        let schema = VehicleSchemaV1::new(
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

    fn sample_schema() -> VehicleSchemaV1 {
        VehicleSchemaV1::new(
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
    #[cfg(feature = "web")]
    fn test_from_db_remote_v1() {
        if std::env::var("FASTSIM_RUN_NETWORK_TESTS").ok().as_deref() != Some("1") {
            return;
        }

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
