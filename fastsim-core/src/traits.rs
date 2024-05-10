use crate::imports::*;

pub trait Linspace {
    /// Generate linearly spaced vec
    /// # Arguments
    /// - `start` - starting point
    /// - `stop` - stopping point, inclusive
    /// - `n_elements` - number of array elements
    fn linspace(start: f64, stop: f64, n_elements: usize) -> Vec<f64> {
        let n_steps = n_elements - 1;
        let step_size = (stop - start) / n_steps as f64;
        let v_norm: Vec<f64> = (0..=n_steps)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let v = v_norm.iter().map(|x| (x * step_size) + start).collect();
        v
    }
}

impl Linspace for Vec<f64> {}

pub trait Init {
    /// Specialized code to execute upon initialization
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

// TODO: only call `init` once per deserialization -- @Kyle, has this been solved?
pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> + Init {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json"];

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

    #[allow(clippy::wrong_self_convention)]
    /// Write (serialize) an object to a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    /// Creates a new file if it does not already exist, otherwise truncates the existing file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The filepath at which write the object
    ///
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
        let mut deserialized = Self::from_reader(file, extension)?;
        deserialized.init()?;
        Ok(deserialized)
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
    fn from_str(contents: &str, format: &str) -> anyhow::Result<Self> {
        let mut deserialized = match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => Self::from_yaml(contents)?,
            "json" => Self::from_json(contents)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        };
        deserialized.init()?;
        Ok(deserialized)
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
    fn from_json(json_str: &str) -> anyhow::Result<Self> {
        let mut json_de: Self = serde_json::from_str(json_str)?;
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
    fn from_yaml(yaml_str: &str) -> anyhow::Result<Self> {
        let mut yaml_de: Self = serde_yaml::from_str(yaml_str)?;
        yaml_de.init()?;
        Ok(yaml_de)
    }

    /// Write (serialize) an object to a bincode-encoded byte array
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// Read (deserialize) an object from a bincode-encoded byte array
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded byte array to deserialize from
    ///
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        let mut bincode_de: Self = deserialize(encoded)?;
        bincode_de.init()?;
        Ok(bincode_de)
    }
}

impl<T: SerdeAPI> SerdeAPI for Vec<T> {}
impl<T: Init> Init for Vec<T> {
    fn init(&mut self) -> anyhow::Result<()> {
        for val in self {
            val.init()?
        }
        Ok(())
    }
}

pub trait Diff {
    /// Returns vec of length `self.len() - 1` where each element in the returned vec at index i is
    /// `self[i + 1] - self[i]`
    fn diff(&self) -> Vec<f64>;
}

impl Diff for Vec<f64> {
    fn diff(&self) -> Vec<f64> {
        self.windows(2)
            .map(|vs| {
                let [x, y] = vs else { unreachable!() };
                y - x
            })
            .collect()
    }
}

/// Provides method that saves `self.state` to `self.history` and propagates to any fields with
/// `state`
pub trait SaveState {
    /// Saves `self.state` to `self.history` and propagates to any fields with `state`
    fn save_state(&mut self) {}
}

/// Provides methods for getting and setting the save interval
pub trait SaveInterval {
    /// Recursively sets save interval
    /// # Arguments
    /// - `save_interval`: time step interval at which to save `self.state` to `self.history`
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()>;
    /// Returns save interval for `self` but does not guarantee recursive consistency in nested
    /// objects
    fn save_interval(&self) -> anyhow::Result<Option<usize>>;
}

/// Trait that provides method for incrementing `i` field of this and all contained structs,
/// recursively
pub trait Step {
    /// Increments `i` field of this and all contained structs, recursively
    fn step(&mut self) {}
}

/// Provides method for checking if struct is default
pub trait IsDefault: std::default::Default + PartialEq {
    /// If `self` is default, returns true
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl<T: Default + PartialEq> IsDefault for T {}

/// Trait for setting cumulative values based on rate values
pub trait SetCumulative {
    /// Sets cumulative values based on rate values
    fn set_cumulative(&mut self, dt: si::Time);
    /// Sets any cumulative values that won't be handled by the macro
    fn set_custom_cumu_vals(&mut self, dt: si::Time) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace() {
        assert_eq!(Vec::linspace(0., 2., 3), vec![0., 1., 2.]);
    }

    #[test]
    fn test_diff() {
        let diff = Vec::linspace(0., 2., 3).diff();
        let ref_diff = vec![1., 1.];
        assert_eq!(diff, ref_diff);
    }
}
