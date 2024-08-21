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

pub trait Min {
    fn min(&self) -> anyhow::Result<f64>;
}
impl Min for &[f64] {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }
}
impl Min for Vec<f64> {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }
}
impl Min for Vec<&f64> {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(**curr)))
    }
}

pub trait Max {
    fn max(&self) -> anyhow::Result<f64>;
}
impl Max for &[f64] {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }
}
impl Max for Vec<f64> {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }
}
impl Max for Vec<&f64> {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(**curr)))
    }
}

pub trait Init {
    /// Specialized code to execute upon initialization.  For any struct with fields
    /// implement `Init`, this should propagate down the hierarchy.
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> + Init {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &[
        #[cfg(feature = "yaml")]
        "yaml",
        #[cfg(feature = "json")]
        "json",
        #[cfg(feature = "toml")]
        "toml",
        #[cfg(feature = "bincode")]
        "bin",
    ];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &[
        #[cfg(feature = "yaml")]
        "yaml",
        #[cfg(feature = "json")]
        "json",
        #[cfg(feature = "toml")]
        "toml",
    ];
    #[cfg(feature = "resources")]
    const RESOURCE_PREFIX: &'static str = "";

    /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
    ///
    /// # Arguments:
    ///
    /// * `filepath` - Filepath, relative to the top of the `resources` folder (excluding any relevant prefix), from which to read the object
    #[cfg(feature = "resources")]
    fn from_resource<P: AsRef<Path>>(filepath: P, skip_init: bool) -> anyhow::Result<Self> {
        let filepath = Path::new(Self::RESOURCE_PREFIX).join(filepath);
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = crate::resources::RESOURCES_DIR
            .get_file(&filepath)
            .with_context(|| format!("File not found in resources: {filepath:?}"))?;
        Self::from_reader(file.contents(), extension, skip_init)
    }

    /// Instantiates an object from a url.  Accepts yaml and json file types  
    /// # Arguments  
    /// - url: URL (either as a string or url type) to object  
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL.
    #[cfg(feature = "web")]
    fn from_url<S: AsRef<str>>(url: S, skip_init: bool) -> anyhow::Result<Self> {
        let url = url::Url::parse(url.as_ref())?;
        let format = url
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|filename| Path::new(filename).extension())
            .and_then(OsStr::to_str)
            .with_context(|| "Could not parse file format from URL: {url:?}")?;
        let response = ureq::get(url.as_ref()).call()?.into_reader();
        Self::from_reader(response, format, skip_init)
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

    /// Read (deserialize) an object from a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    ///
    /// # Arguments:
    ///
    /// * `filepath`: The filepath from which to read the object
    ///
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> anyhow::Result<Self> {
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
        Self::from_reader(file, extension, skip_init)
    }

    /// Write (serialize) an object into anything that implements [`std::io::Write`]
    ///
    /// # Arguments:
    ///
    /// * `wtr` - The writer into which to write object data
    /// * `format` - The target format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn to_writer<W: std::io::Write>(&self, mut wtr: W, format: &str) -> anyhow::Result<()> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)?,
            #[cfg(feature = "json")]
            "json" => serde_json::to_writer(wtr, self)?,
            #[cfg(feature = "toml")]
            "toml" => {
                let toml_string = self.to_toml()?;
                wtr.write_all(toml_string.as_bytes())?;
            }
            #[cfg(feature = "bincode")]
            "bin" => bincode::serialize_into(wtr, self)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    /// Deserialize an object from anything that implements [`std::io::Read`]
    ///
    /// # Arguments:
    ///
    /// * `rdr` - The reader from which to read object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn from_reader<R: std::io::Read>(
        mut rdr: R,
        format: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let mut deserialized: Self = match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            #[cfg(feature = "json")]
            "json" => serde_json::from_reader(rdr)?,
            #[cfg(feature = "tonl")]
            "toml" => {
                let mut buf = String::new();
                rdr.read_to_string(&mut buf)?;
                Self::from_toml(buf, skip_init)?
            }
            #[cfg(feature = "bincode")]
            "bin" => bincode::deserialize_from(rdr)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        };
        if !skip_init {
            deserialized.init()?;
        }
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
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => self.to_yaml(),
            #[cfg(feature = "json")]
            "json" => self.to_json(),
            #[cfg(feature = "toml")]
            "toml" => self.to_toml(),
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
    fn from_str<S: AsRef<str>>(contents: S, format: &str, skip_init: bool) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                #[cfg(feature = "yaml")]
                "yaml" | "yml" => Self::from_yaml(contents, skip_init)?,
                #[cfg(feature = "json")]
                "json" => Self::from_json(contents, skip_init)?,
                #[cfg(feature = "toml")]
                "toml" => Self::from_toml(contents, skip_init)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_STR_FORMATS
                ),
            },
        )
    }

    /// Write (serialize) an object to bincode-encoded bytes
    #[cfg(feature = "bincode")]
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// Read (deserialize) an object from bincode-encoded bytes
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded bytes to deserialize from
    ///
    #[cfg(feature = "bincode")]
    fn from_bincode(encoded: &[u8], skip_init: bool) -> anyhow::Result<Self> {
        let mut bincode_de: Self = bincode::deserialize(encoded)?;
        if !skip_init {
            bincode_de.init()?;
        }
        Ok(bincode_de)
    }

    /// Write (serialize) an object to a JSON string
    #[cfg(feature = "json")]
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// Read (deserialize) an object from a JSON string
    ///
    /// # Arguments
    ///
    /// * `json_str` - JSON-formatted string to deserialize from
    ///
    #[cfg(feature = "json")]
    fn from_json<S: AsRef<str>>(json_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut json_de: Self = serde_json::from_str(json_str.as_ref())?;
        if !skip_init {
            json_de.init()?;
        }
        Ok(json_de)
    }

    /// Write (serialize) an object to a TOML string
    #[cfg(feature = "toml")]
    fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string(&self)?)
    }

    /// Read (deserialize) an object from a TOML string
    ///
    /// # Arguments
    ///
    /// * `toml_str` - TOML-formatted string to deserialize from
    ///
    #[cfg(feature = "toml")]
    fn from_toml<S: AsRef<str>>(toml_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut toml_de: Self = toml::from_str(toml_str.as_ref())?;
        if !skip_init {
            toml_de.init()?;
        }
        Ok(toml_de)
    }

    /// Write (serialize) an object to a YAML string
    #[cfg(feature = "yaml")]
    fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(&self)?)
    }

    /// Read (deserialize) an object from a YAML string
    ///
    /// # Arguments
    ///
    /// * `yaml_str` - YAML-formatted string to deserialize from
    ///
    #[cfg(feature = "yaml")]
    fn from_yaml<S: AsRef<str>>(yaml_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut yaml_de: Self = serde_yaml::from_str(yaml_str.as_ref())?;
        if !skip_init {
            yaml_de.init()?;
        }
        Ok(yaml_de)
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

pub trait Diff<T> {
    /// Returns vec of length `self.len() - 1` where each element in the returned vec at index i is
    /// `self[i + 1] - self[i]`
    fn diff(&self) -> Vec<T>;
}

impl<T: Copy + Sub<T, Output = T>> Diff<T> for Vec<T> {
    fn diff(&self) -> Vec<T> {
        self.windows(2)
            .map(|vs| {
                let [x, y] = vs else { unreachable!() };
                *y - *x
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
pub trait EqDefault: std::default::Default + PartialEq {
    /// If `self` is default, returns true
    fn eq_default(&self) -> bool {
        *self == Self::default()
    }
}

impl<T: Default + PartialEq> EqDefault for T {}

/// Trait for setting cumulative values based on rate values
pub trait SetCumulative {
    /// Sets cumulative values based on rate values
    fn set_cumulative(&mut self, dt: si::Time);
    /// Sets any cumulative values that won't be handled by the macro
    #[allow(unused_variables)]
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
    fn test_max_for_vec_f64() {
        assert_eq!(Vec::linspace(-10., 12., 5).max().unwrap(), 12.);
    }
    #[test]
    fn test_min_for_vec_f64() {
        assert_eq!(Vec::linspace(-10., 12., 5).min().unwrap(), -10.);
    }

    #[test]
    fn test_diff() {
        let diff = Vec::linspace(0., 2., 3).diff();
        let ref_diff = vec![1., 1.];
        assert_eq!(diff, ref_diff);
    }
}
