use crate::error::Error;
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
        self.as_slice().min()
    }
}
impl Min for &[&f64] {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(**curr)))
    }
}
impl Min for Vec<&f64> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for &[Vec<f64>] {
    fn min(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.min())
            .try_fold(f64::INFINITY, |acc, x| Ok(acc.min(x?)))
    }
}
impl Min for Vec<Vec<f64>> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for &[Vec<Vec<f64>>] {
    fn min(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.min())
            .try_fold(f64::INFINITY, |acc, x| Ok(acc.min(x?)))
    }
}
impl Min for Vec<Vec<Vec<f64>>> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for Interpolator {
    fn min(&self) -> anyhow::Result<f64> {
        match self {
            Interpolator::Interp0D(value) => Ok(*value),
            Interpolator::Interp1D(..) => self.f_x()?.min(),
            Interpolator::Interp2D(..) => self.f_xy()?.min(),
            Interpolator::Interp3D(..) => self.f_xyz()?.min(),
            Interpolator::InterpND(..) => Ok(self
                .values()?
                .iter()
                .fold(f64::INFINITY, |acc, x| acc.min(*x))),
        }
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
        self.as_slice().max()
    }
}
impl Max for &[&f64] {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(**curr)))
    }
}
impl Max for Vec<&f64> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for &[Vec<f64>] {
    fn max(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.max())
            .try_fold(f64::NEG_INFINITY, |acc, x| Ok(acc.max(x?)))
    }
}
impl Max for Vec<Vec<f64>> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for &[Vec<Vec<f64>>] {
    fn max(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.max())
            .try_fold(f64::NEG_INFINITY, |acc, x| Ok(acc.max(x?)))
    }
}
impl Max for Vec<Vec<Vec<f64>>> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for Interpolator {
    fn max(&self) -> anyhow::Result<f64> {
        match self {
            Interpolator::Interp0D(value) => Ok(*value),
            Interpolator::Interp1D(..) => self.f_x()?.max(),
            Interpolator::Interp2D(..) => self.f_xy()?.max(),
            Interpolator::Interp3D(..) => self.f_xyz()?.max(),
            Interpolator::InterpND(..) => Ok(self
                .values()?
                .iter()
                .fold(f64::NEG_INFINITY, |acc, x| acc.max(*x))),
        }
    }
}

pub trait Init {
    /// Specialized code to execute upon initialization.  For any struct with fields
    /// that implement `Init`, this should propagate down the hierarchy.
    fn init(&mut self) -> Result<(), Error> {
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
    fn from_resource<P: AsRef<Path>>(filepath: P, skip_init: bool) -> Result<Self, Error> {
        let filepath = Path::new(Self::RESOURCE_PREFIX).join(filepath);
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        let file = crate::resources::RESOURCES_DIR
            .get_file(&filepath)
            .ok_or_else(|| {
                Error::SerdeError(format!("File not found in resources: {filepath:?}"))
            })?;
        Self::from_reader(&mut file.contents(), extension, skip_init)
    }

    /// Instantiates an object from a url.  Accepts yaml and json file types  
    /// # Arguments  
    /// - url: URL (either as a string or url type) to object  
    ///
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL.
    #[cfg(feature = "web")]
    fn from_url<S: AsRef<str>>(url: S, skip_init: bool) -> Result<Self, Error> {
        let url = url::Url::parse(url.as_ref())?;
        let format = url
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|filename| Path::new(filename).extension())
            .and_then(OsStr::to_str)
            .with_context(|| "Could not parse file format from URL: {url:?}")?;
        let mut response = ureq::get(url.as_ref()).call()?.into_reader();
        Self::from_reader(&mut response, format, skip_init)
    }

    /// Write (serialize) an object to a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    /// Creates a new file if it does not already exist, otherwise truncates the existing file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The filepath at which to write the object
    ///
    fn to_file<P: AsRef<Path>>(&self, filepath: P) -> Result<(), Error> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        self.to_writer(
            File::create(filepath).map_err(|err| Error::SerdeError(format!("{err}")))?,
            extension,
        )
    }

    /// Read (deserialize) an object from a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    ///
    /// # Arguments:
    ///
    /// * `filepath`: The filepath from which to read the object
    ///
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> Result<Self, Error> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        let mut file = File::open(filepath)
            .with_context(|| {
                if !filepath.exists() {
                    format!("File not found: {filepath:?}")
                } else {
                    format!("Could not open file: {filepath:?}")
                }
            })
            .map_err(|err| Error::SerdeError(format!("{err}")))?;
        Self::from_reader(&mut file, extension, skip_init)
    }

    /// Write (serialize) an object into anything that implements [`std::io::Write`]
    ///
    /// # Arguments:
    ///
    /// * `wtr` - The writer into which to write object data
    /// * `format` - The target format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn to_writer<W: std::io::Write>(&self, mut wtr: W, format: &str) -> Result<(), Error> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            #[cfg(feature = "yaml")]
            "yaml" | "yml" => serde_yaml::to_writer(wtr, self)
                .map_err(|err| Error::SerdeError(format!("{err}")))?,
            #[cfg(feature = "json")]
            "json" => serde_json::to_writer(wtr, self)
                .map_err(|err| Error::SerdeError(format!("{err}")))?,
            #[cfg(feature = "toml")]
            "toml" => {
                let toml_string = self
                    .to_toml()
                    .map_err(|err| Error::SerdeError(format!("{err}")))?;
                wtr.write_all(toml_string.as_bytes())
                    .map_err(|err| Error::SerdeError(format!("{err}")))?;
            }
            _ => Err(Error::SerdeError(format!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            )))?,
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
        rdr: &mut R,
        format: &str,
        skip_init: bool,
    ) -> Result<Self, Error> {
        let mut deserialized: Self =
            match format.trim_start_matches('.').to_lowercase().as_str() {
                #[cfg(feature = "yaml")]
                "yaml" | "yml" => serde_yaml::from_reader(rdr)
                    .map_err(|err| Error::SerdeError(format!("{err}")))?,
                #[cfg(feature = "json")]
                "json" => serde_json::from_reader(rdr)
                    .map_err(|err| Error::SerdeError(format!("{err}")))?,
                #[cfg(feature = "msgpack")]
                "msgpack" => rmp_serde::decode::from_read(rdr)
                    .map_err(|err| Error::SerdeError(format!("{err}")))?,
                #[cfg(feature = "toml")]
                "toml" => {
                    let mut buf = String::new();
                    rdr.read_to_string(&mut buf)
                        .map_err(|err| Error::SerdeError(format!("{err}")))?;
                    Self::from_toml(buf, skip_init)
                        .map_err(|err| Error::SerdeError(format!("{err}")))?
                }
                _ => Err(Error::SerdeError(format!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_BYTE_FORMATS,
                )))?,
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

    /// Write (serialize) an object to a message pack
    #[cfg(feature = "msgpack")]
    fn to_msg_pack(&self) -> anyhow::Result<Vec<u8>> {
        Ok(rmp_serde::encode::to_vec_named(&self)?)
    }

    /// Read (deserialize) an object from a message pack
    ///
    /// # Arguments
    ///
    /// * `msg_pack` - message pack object
    ///
    #[cfg(feature = "msgpack")]
    fn from_msg_pack(msg_pack: &[u8], skip_init: bool) -> anyhow::Result<Self> {
        let mut msg_pack_de: Self = rmp_serde::decode::from_slice(msg_pack)?;
        if !skip_init {
            msg_pack_de.init()?;
        }
        Ok(msg_pack_de)
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
    fn init(&mut self) -> Result<(), Error> {
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

impl<T: Clone + Sub<T, Output = T> + Default> Diff<T> for Vec<T> {
    fn diff(&self) -> Vec<T> {
        let mut v_diff: Vec<T> = vec![Default::default()];
        v_diff.extend::<Vec<T>>(
            self.windows(2)
                .map(|vs| {
                    let x = &vs[0];
                    let y = &vs[1];
                    y.clone() - x.clone()
                })
                .collect(),
        );
        v_diff
    }
}

/// Provides method that saves `self.state` to `self.history` and propagates to any fields with
/// `state`
pub trait SaveState {
    /// Saves `self.state` to `self.history` and propagates to any fields with `state`
    fn save_state(&mut self) {}
}
/// Provides methods to guarantee that states are updated once and only once per time step
pub trait TrackedStateMethods {
    /// checks that all tracked state variables have been updated and resets for next time step
    fn check_and_reset(&mut self) -> anyhow::Result<()>;
}

/// Provides methods for getting and setting the save interval
pub trait HistoryMethods: SaveState {
    /// Recursively sets save interval
    /// # Arguments
    /// - `save_interval`: time step interval at which to save `self.state` to `self.history`
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()>;
    /// Returns save interval for `self` but does not guarantee recursive consistency in nested
    /// objects
    fn save_interval(&self) -> anyhow::Result<Option<usize>>;
    /// Remove all history
    fn clear(&mut self);
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
        let ref_diff = vec![0., 1., 1.];
        assert_eq!(diff, ref_diff);
    }
}
