use crate::imports::*;
use std::collections::HashMap;
use tempfile::tempdir;

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json"];

    /// Runs any initialization steps that might be needed
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Save current data structure to file. Method adaptively calls serialization methods
    /// dependent on the suffix of the filepath.
    fn to_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        match extension.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::to_writer(&File::create(filepath)?, self)?,
            "json" => serde_json::to_writer(&File::create(filepath)?, self)?,
            "bin" => bincode::serialize_into(&File::create(filepath)?, self)?,
            _ => bail!(
                "Unsupported format {extension:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    /// Read from file and return instantiated struct. Method adaptively calls deserialization
    /// methods dependent on the suffix of the file name given as str.
    /// Function returns a dynamic Error Result if it fails.
    ///
    /// # Argument:
    ///
    /// * `filepath`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result wrapping data structure if method is called successfully; otherwise a dynamic
    /// Error.
    fn from_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = File::open(filepath).with_context(|| {
            if !filepath.exists() {
                format!("File not found: {filepath:?}")
            } else {
                format!("Could not open file: {filepath:?}")
            }
        })?;
        // deserialized file
        let mut deserialized = Self::from_reader(file, extension)?;
        deserialized.init()?;
        Ok(deserialized)
    }

    fn from_resource<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?
            .to_lowercase();
        ensure!(
            Self::ACCEPTED_BYTE_FORMATS.contains(&extension.as_str()),
            "Unsupported format {extension:?}, must be one of {:?}",
            Self::ACCEPTED_BYTE_FORMATS
        );
        let file = crate::resources::RESOURCES_DIR
            .get_file(filepath)
            .with_context(|| format!("File not found in resources: {filepath:?}"))?;
        let mut deserialized = match extension.as_str() {
            "bin" => Self::from_bincode(include_dir::File::contents(file))?,
            _ => Self::from_str(
                include_dir::File::contents_utf8(file)
                    .with_context(|| format!("File could not be parsed to UTF-8: {filepath:?}"))?,
                &extension,
            )?,
        };
        deserialized.init()?;
        Ok(deserialized)
    }

    fn from_reader<R: std::io::Read>(rdr: R, format: &str) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
                "json" => serde_json::from_reader(rdr)?,
                "bin" => bincode::deserialize_from(rdr)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_BYTE_FORMATS
                ),
            },
        )
    }

    fn to_str(&self, format: &str) -> anyhow::Result<String> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => self.to_yaml(),
            "json" => self.to_json(),
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        }
    }

    fn from_str(contents: &str, format: &str) -> anyhow::Result<Self> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => Self::from_yaml(contents),
            "json" => Self::from_json(contents),
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        }
    }

    /// JSON serialization method
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// JSON deserialization method
    fn from_json(json_str: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json_str)?)
    }

    /// YAML serialization method
    fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(&self)?)
    }

    /// YAML deserialization method
    fn from_yaml(yaml_str: &str) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_str(yaml_str)?)
    }

    /// bincode serialization method
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// bincode deserialization method
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(encoded)?)
    }

    /// instantiates an object from a url
    fn from_url<S: AsRef<str>>(url: S) -> anyhow::Result<Self> {
        let url = url.as_ref();
        let temp_dir = tempdir()?;
        let mut file_path = PathBuf::new();
        // do these file types need to be specific to the object?
        // TODO: either make funciton work for csv files, or remove from supported file list
        if url.ends_with("yaml"){
            let file_path = temp_dir.path().join("temporary_object.yaml");
        } else if url.ends_with("csv"){
            let file_path = temp_dir.path().join("temporary_object.csv");
        } else if url.ends_with("json"){
            let file_path = temp_dir.path().join("temporary_object.json");
        } else {
            bail!("Unsupported file type, must be a yaml, json, or csv file.");
        }
        download_file_from_url(url, &file_path);
        // only works for json and yaml
        // seems like I might also be able to use from_reader instead -- which one is preferable?
        Self::from_file(file_path)
    }

    /// takes an object from a url and saves it in the fastsim data directory in a rust_objects folder
    /// WARNING: if there is a file already in the data subdirectory with the same name, it will be replaced by the new file
    /// to save to a folder other than rust_objects for a specific object type, override this default
    /// implementation for the Rust object, and in the object-specific implementation, replace
    /// "rust_objects" with your choice of folder name
    fn to_cache<S: AsRef<str>>(url: S) {
        let url = url.as_ref();
        let url_parts: Vec<&str> = url.split("/").collect();
        let file_name = url_parts.last().unwrap_or_else(||panic!("Could not determine file name/type."));
        let data_subdirectory = create_project_subdir("rust_objects").unwrap_or_else(|_|panic!("Could not find or create Fastsim data subdirectory."));
        let file_path = data_subdirectory.join(file_name);
        // I believe this will overwrite any existing files with the same name -- is this preferable, or should we add
        // a bool argument so user can determine whether the file should overwrite an existing file or not
        download_file_from_url(url, &file_path);
    }
}

pub trait ApproxEq<Rhs = Self> {
    fn approx_eq(&self, other: &Rhs, tol: f64) -> bool;
}

macro_rules! impl_approx_eq_for_strict_eq_types {
    ($($strict_eq_type: ty),*) => {
        $(
            impl ApproxEq for $strict_eq_type {
                fn approx_eq(&self, other: &$strict_eq_type, _tol: f64) -> bool {
                    return self == other;
                }
            }
        )*
    }
}

impl_approx_eq_for_strict_eq_types!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, bool, &str, String
);

macro_rules! impl_approx_eq_for_floats {
    ($($float_type: ty),*) => {
        $(
            impl ApproxEq for $float_type {
                fn approx_eq(&self, other: &$float_type, tol: f64) -> bool {
                    return (((other - self) / (self + other)).abs() as f64) < tol || ((other - self).abs() as f64) < tol;
                }
            }
        )*
    }
}

impl_approx_eq_for_floats!(f32, f64);

impl<T> ApproxEq for Vec<T>
where
    T: ApproxEq,
{
    fn approx_eq(&self, other: &Vec<T>, tol: f64) -> bool {
        return self
            .iter()
            .zip(other.iter())
            .all(|(x, y)| x.approx_eq(y, tol));
    }
}

impl<T> ApproxEq for Array1<T>
where
    T: ApproxEq + std::clone::Clone,
{
    fn approx_eq(&self, other: &Array1<T>, tol: f64) -> bool {
        self.to_vec().approx_eq(&other.to_vec(), tol)
    }
}

impl<T> ApproxEq for Option<T>
where
    T: ApproxEq,
{
    fn approx_eq(&self, other: &Option<T>, tol: f64) -> bool {
        if self.is_none() && other.is_none() {
            true
        } else if self.is_some() && other.is_some() {
            self.as_ref()
                .unwrap()
                .approx_eq(other.as_ref().unwrap(), tol)
        } else {
            false
        }
    }
}

impl<K, V, S> ApproxEq for HashMap<K, V, S>
where
    K: Eq + std::hash::Hash,
    V: ApproxEq,
    S: std::hash::BuildHasher,
{
    fn approx_eq(&self, other: &HashMap<K, V, S>, tol: f64) -> bool {
        if self.len() != other.len() {
            return false;
        }
        return self
            .iter()
            .all(|(key, value)| other.get(key).map_or(false, |v| value.approx_eq(v, tol)));
    }
}
