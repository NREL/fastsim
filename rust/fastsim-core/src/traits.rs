use crate::imports::*;
use std::collections::HashMap;
use ureq;

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json"];
    const CACHE_FOLDER: &'static str = &"";

    /// Specialized code to execute upon initialization
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
    ///
    /// # Arguments:
    ///
    /// * `filepath` - Filepath, relative to the top of the `resources` folder, from which to read the object
    ///
    fn from_resource<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = crate::resources::RESOURCES_DIR
            .get_file(filepath)
            .with_context(|| format!("File not found in resources: {filepath:?}"))?;
        Self::from_reader(file.contents(), extension)
    }

    /// Write (serialize) an object to a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    /// Creates a new file if it does not already exist, otherwise truncates the existing file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The filepath at which to write the object
    ///
    fn to_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        self.to_writer(File::create(filepath)?, extension)
    }

    fn to_writer<W: std::io::Write>(&self, wtr: W, format: &str) -> anyhow::Result<()> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)?,
            "json" => serde_json::to_writer(wtr, self)?,
            "bin" => bincode::serialize_into(wtr, self)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    /// Read (deserialize) an object from a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    ///
    /// # Arguments:
    ///
    /// * `filepath`: The filepath from which to read the object
    ///
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
        Self::from_reader(file, extension)
    }

    /// Write (serialize) an object into a string
    ///
    /// # Arguments:
    ///
    /// * `format` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
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

    /// Read (deserialize) an object from a string
    ///
    /// # Arguments:
    ///
    /// * `contents` - The string containing the object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
    fn from_str<S: AsRef<str>>(contents: S, format: &str) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => Self::from_yaml(contents)?,
                "json" => Self::from_json(contents)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_STR_FORMATS
                ),
            },
        )
    }

    /// Deserialize an object from anything that implements [`std::io::Read`]
    ///
    /// # Arguments:
    ///
    /// * `rdr` - The reader from which to read object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn from_reader<R: std::io::Read>(rdr: R, format: &str) -> anyhow::Result<Self> {
        let mut deserialized: Self = match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            "json" => serde_json::from_reader(rdr)?,
            "bin" => bincode::deserialize_from(rdr)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        };
        deserialized.init()?;
        Ok(deserialized)
    }

    /// Write (serialize) an object to a JSON string
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// Read (deserialize) an object to a JSON string
    ///
    /// # Arguments
    ///
    /// * `json_str` - JSON-formatted string to deserialize from
    ///
    fn from_json<S: AsRef<str>>(json_str: S) -> anyhow::Result<Self> {
        let mut json_de: Self = serde_json::from_str(json_str.as_ref())?;
        json_de.init()?;
        Ok(json_de)
    }

    /// Write (serialize) an object to a YAML string
    fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(&self)?)
    }

    /// Read (deserialize) an object from a YAML string
    ///
    /// # Arguments
    ///
    /// * `yaml_str` - YAML-formatted string to deserialize from
    ///
    fn from_yaml<S: AsRef<str>>(yaml_str: S) -> anyhow::Result<Self> {
        let mut yaml_de: Self = serde_yaml::from_str(yaml_str.as_ref())?;
        yaml_de.init()?;
        Ok(yaml_de)
    }

    /// Write (serialize) an object to bincode-encoded bytes
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// Read (deserialize) an object from bincode-encoded bytes
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded bytes to deserialize from
    ///
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        let mut bincode_de: Self = bincode::deserialize(encoded)?;
        bincode_de.init()?;
        Ok(bincode_de)
    }

    /// Instantiates an object from a url.  Accepts yaml and json file types  
    /// # Arguments  
    /// - url: URL (either as a string or url type) to object  
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL.
    fn from_url<S: AsRef<str>>(url: S) -> anyhow::Result<Self> {
        let url = url::Url::parse(url.as_ref())?;
        let format = url
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|filename| Path::new(filename).extension())
            .and_then(OsStr::to_str)
            .with_context(|| "Could not parse file format from URL: {url:?}")?;
        let response = ureq::get(url.as_ref()).call()?.into_reader();
        Self::from_reader(response, format)
    }

    /// Takes an instantiated Rust object and saves it in the FASTSim data directory in
    /// a rust_objects folder.  
    /// WARNING: If there is a file already in the data subdirectory with the
    /// same name, it will be replaced by the new file.  
    /// # Arguments  
    /// - self (rust object)  
    /// - file_path: path to file within subdirectory. If only the file name is
    /// listed, file will sit directly within the subdirectory of
    /// the FASTSim data directory. If a path is given, the file will live
    /// within the path specified, within the subdirectory CACHE_FOLDER of the
    /// FASTSim data directory.
    fn to_cache<P: AsRef<Path>>(&self, file_path: P) -> anyhow::Result<()> {
        let file_name = file_path
            .as_ref()
            .file_name()
            .with_context(|| "Could not determine file name")?
            .to_str()
            .context("Could not determine file name.")?;
        let file_path_internal = file_path
            .as_ref()
            .to_str()
            .context("Could not determine file name.")?;
        let subpath = if file_name == file_path_internal {
            PathBuf::from(Self::CACHE_FOLDER)
        } else {
            Path::new(Self::CACHE_FOLDER).join(
                file_path_internal
                    .strip_suffix(file_name)
                    .context("Could not determine path to subdirectory.")?,
            )
        };
        let data_subdirectory = create_project_subdir(subpath)
            .with_context(|| "Could not find or build Fastsim data subdirectory.")?;
        let file_path = data_subdirectory.join(file_name);
        self.to_file(file_path)
    }

    /// Instantiates a Rust object from the subdirectory within the FASTSim data
    /// directory corresponding to the Rust Object ("vehices" for a RustVehice,
    /// "cycles" for a RustCycle, and the root folder of the data directory for
    /// all other objects).  
    /// # Arguments  
    /// - file_path: subpath to object, including file name, within subdirectory.
    ///   If the file sits directly in the subdirectory, this will just be the
    ///   file name.  
    /// Note: This function will work for all objects cached using the
    /// to_cache() method. If a file has been saved manually to a different
    /// subdirectory than the correct one for the object type (for instance a
    /// RustVehicle saved within a subdirectory other than "vehicles" using the
    /// utils::url_to_cache() function), then from_cache() will not be able to
    /// find and instantiate the object. Instead, use the from_file method, and
    /// use the utils::path_to_cache() to find the FASTSim data directory
    /// location if needed.
    fn from_cache<P: AsRef<Path>>(file_path: P) -> anyhow::Result<Self> {
        let full_file_path = Path::new(Self::CACHE_FOLDER).join(file_path);
        let path_including_directory = path_to_cache()?.join(full_file_path);
        Self::from_file(path_including_directory)
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
