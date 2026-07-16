use super::*;

/// Database schema for vehicle files in the `fastsim-vehicles` repository.
#[derive(Debug)]
pub struct DatabaseSchemaV1 {
    /// FASTSim version namespace to prevent cross-major collisions.
    pub fastsim_version: u32,
    /// Vehicle make.
    pub make: String,
    /// Vehicle model (may include trim information).
    pub model: String,
    /// Vehicle model year.
    pub year: String,
    /// Vehicle variant (describes what modeling features are active, etc).
    pub variant: String,
    /// Model revision/version for correcting model-level issues over time.
    pub revision: u32,
}

impl DatabaseSchemaV1 {
    fn relative_segments(&self, extension: &str) -> Vec<String> {
        vec![
            "v1".to_string(),
            format!("fastsim-{}", self.fastsim_version),
            self.make.clone(),
            self.model.clone(),
            self.year.clone(),
            self.variant.clone(),
            format!("v{}.{}", self.revision, extension),
        ]
    }

    pub fn build_filepath(
        &self,
        base_dir: &std::path::Path,
        extension: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let path = self
            .relative_segments(extension)
            .into_iter()
            .fold(base_dir.to_path_buf(), |acc, segment| acc.join(segment));
        Ok(path)
    }

    pub fn build_url(&self, base_url: &str, extension: &str) -> anyhow::Result<String> {
        let base = base_url.trim_end_matches('/');
        let rel = self.relative_segments(extension).join("/");
        Ok(format!("{base}/{rel}"))
    }
}

impl Vehicle {
    pub fn from_db_local_v1(
        base_dir: &std::path::Path,
        fastsim_version: u32,
        make: &str,
        model: &str,
        year: &str,
        variant: &str,
        model_version: u32,
        skip_init: bool,
    ) -> anyhow::Result<Vehicle> {
        let schema = DatabaseSchemaV1 {
            fastsim_version,
            make: make.to_string(),
            model: model.to_string(),
            year: year.to_string(),
            variant: variant.to_string(),
            revision: model_version,
        };
        let path = schema.build_filepath(base_dir, "yaml")?;
        let mut veh = Self::from_file(path.clone(), false).map_err(|err| {
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

    #[cfg(feature = "web")]
    pub fn from_db_remote_v1(
        url: Option<&str>,
        fastsim_version: u32,
        make: &str,
        model: &str,
        year: &str,
        variant: &str,
        model_version: u32,
        skip_init: bool,
    ) -> anyhow::Result<Vehicle> {
        let schema = DatabaseSchemaV1 {
            fastsim_version,
            make: make.to_string(),
            model: model.to_string(),
            year: year.to_string(),
            variant: variant.to_string(),
            revision: model_version,
        };
        let resolved_url = schema.build_url(url.unwrap_or(DEFAULT_DB_URL), "yaml")?;
        let mut veh = Self::from_url(resolved_url.clone(), false).map_err(|err| {
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
        DatabaseSchemaV1 {
            fastsim_version: 3,
            make: "Ford".to_string(),
            model: "F-150".to_string(),
            year: "2022".to_string(),
            variant: "base".to_string(),
            revision: 1,
        }
    }

    #[test]
    fn test_build_filepath_output() {
        let base = std::path::Path::new("/tmp/vehicles-db");
        let schema = sample_schema();
        let actual = schema.build_filepath(base, "yaml").unwrap();
        let expected = base
            .join("v1")
            .join("fastsim-v3")
            .join("Ford")
            .join("F-150")
            .join("2022")
            .join("v1.yaml");

        eprintln!("build_filepath output: {}", actual.display());
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_build_url_output() {
        let schema = sample_schema();
        let actual = schema
            .build_url("https://example.com/fastsim-vehicles/", "yaml")
            .unwrap();
        let expected =
            "https://example.com/fastsim-vehicles/v1/fastsim-v3/Ford/F-150/2022/base/v1.yaml"
                .to_string();

        eprintln!("build_url output: {actual}");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_from_db_local_output_string() {
        let base = std::path::Path::new("/tmp/vehicles-db");
        let schema = sample_schema();
        let err = Vehicle::from_db_local_v1(
            base,
            schema.fastsim_version,
            &schema.make,
            &schema.model,
            &schema.year,
            &schema.variant,
            schema.revision,
            false,
        )
        .unwrap_err();
        let output = err.to_string();

        eprintln!("from_db_local output: {output}");
        assert!(
            output.contains("/tmp/vehicles-db/v1/fastsim-v3/Ford/F-150/2022/base/v1.yaml"),
            "unexpected output: {output}"
        );
    }

    #[test]
    #[cfg(feature = "web")]
    fn test_from_db_remote_output_string() {
        let schema = sample_schema();
        let err = Vehicle::from_db_remote_v1(
            None,
            schema.fastsim_version,
            &schema.make,
            &schema.model,
            &schema.year,
            &schema.variant,
            schema.revision,
            false,
        )
        .unwrap_err();
        let output = err.to_string();

        eprintln!("from_db_remote output: {output}");
        assert!(
            output.contains(
                "https://example.com/fastsim-vehicles/v1/fastsim-v3/Ford/F-150/2022/base/v1.yaml"
            ),
            "unexpected output: {output}"
        );
    }
}
