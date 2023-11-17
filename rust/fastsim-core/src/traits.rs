use crate::imports::*;
use std::collections::HashMap;

pub(crate) const ACCEPTED_FILE_FORMATS: [&str; 2] = ["yaml", "json"];

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    /// Runs any initialization steps that might be needed
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
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
                "Unsupported file format {extension:?}, must be one of {ACCEPTED_FILE_FORMATS:?}"
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
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = crate::resources::RESOURCES_DIR
            .get_file(filepath)
            .with_context(|| format!("File not found in resources: {filepath:?}"))?;
        let mut deserialized = match extension.trim_start_matches('.').to_lowercase().as_str() {
            "bin" => Self::from_bincode(include_dir::File::contents(file))?,
            _ => Self::from_str(
                include_dir::File::contents_utf8(file)
                    .with_context(|| format!("File could not be parsed to UTF-8: {filepath:?}"))?,
                extension,
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
                    "Unsupported file format {format:?}, must be one of {ACCEPTED_FILE_FORMATS:?}"
                ),
            },
        )
    }

    fn to_str(&self, format: &str) -> anyhow::Result<String> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => self.to_yaml(),
            "json" => self.to_json(),
            _ => bail!(
                "Unsupported file format {format:?}, must be one of {ACCEPTED_FILE_FORMATS:?}"
            ),
        }
    }

    fn from_str(contents: &str, format: &str) -> anyhow::Result<Self> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => Self::from_yaml(contents),
            "json" => Self::from_json(contents),
            _ => bail!(
                "Unsupported file format {format:?}, must be one of {ACCEPTED_FILE_FORMATS:?}"
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
