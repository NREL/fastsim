use crate::imports::*;
use std::collections::HashMap;

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json"];

    /// Specialized code to execute upon initialization
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
    ///
    /// # Arguments:
    ///
    /// * `filepath` - Filepath, relative to the top of the `resources` folder, from which to read the object
    #[cfg(feature = "resources")]
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
        match extension.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::to_writer(&File::create(filepath)?, self)?,
            "json" => serde_json::to_writer(&File::create(filepath)?, self)?,
            #[cfg(feature = "bincode")]
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
            #[cfg(feature = "bincode")]
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
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        let mut bincode_de: Self = bincode::deserialize(encoded)?;
        bincode_de.init()?;
        Ok(bincode_de)
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

/// This trait was heavily inspired by `ndarray-stats` crate
pub trait IterMaxMin<A: PartialOrd> {
    fn max(&self) -> anyhow::Result<&A>;
    fn min(&self) -> anyhow::Result<&A>;
}

#[allow(clippy::manual_try_fold)] // `try_fold` is apparently not implemented
impl IterMaxMin<f64> for Array1<f64> {
    fn max(&self) -> anyhow::Result<&f64> {
        let first = self.first().ok_or(anyhow!("empty input"))?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(anyhow!("undefined order"))? {
                cmp::Ordering::Greater => Ok(elem),
                _ => Ok(acc),
            }
        })
    }
    fn min(&self) -> anyhow::Result<&f64> {
        let first = self.first().ok_or(anyhow!("empty input"))?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(anyhow!("undefined order"))? {
                cmp::Ordering::Less => Ok(elem),
                _ => Ok(acc),
            }
        })
    }
}
