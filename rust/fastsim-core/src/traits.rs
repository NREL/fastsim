use crate::imports::*;
use std::collections::HashMap;

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    /// runs any initialization steps that might be needed
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    /// Save current data structure to file. Method adaptively calls serialization methods
    /// dependent on the suffix of the file given as str.
    ///
    /// # Argument:
    ///
    /// * `filepath`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result
    fn to_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| {
                format!(
                    "File extension could not be parsed: \"{}\"",
                    filepath.display()
                )
            })?;
        match extension {
            "json" => serde_json::to_writer(&File::create(filepath)?, self)?,
            "yaml" => serde_yaml::to_writer(&File::create(filepath)?, self)?,
            // TODO: do we want a default behavior for this?
            _ => serde_json::to_writer(&File::create(filepath)?, self)?,
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
            .with_context(|| {
                format!(
                    "File extension could not be parsed: \"{}\"",
                    filepath.display()
                )
            })?;
        let file = File::open(filepath).with_context(|| {
            if !filepath.exists() {
                format!("File not found: \"{}\"", filepath.display())
            } else {
                format!("Could not open file: \"{}\"", filepath.display())
            }
        })?;
        // deserialized file
        let mut deserialized = Self::from_reader(file, extension)?;
        deserialized.init()?;
        Ok(deserialized)
    }

    fn from_resource<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let contents = crate::resources::RESOURCES_DIR
            .get_file(filepath)
            .and_then(include_dir::File::contents_utf8)
            .with_context(|| format!("File not found in resources: \"{}\"", filepath.display()))?;
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| {
                format!(
                    "File extension could not be parsed: \"{}\"",
                    filepath.display()
                )
            })?;
        let mut deserialized = Self::from_str(contents, extension)?;
        deserialized.init()?;
        Ok(deserialized)
    }

    fn from_reader<R: std::io::Read>(rdr: R, format: &str) -> anyhow::Result<Self> {
        Ok(match format {
            "yaml" => serde_yaml::from_reader(rdr)?,
            "json" => serde_json::from_reader(rdr)?,
            _ => bail!("Unsupported file format: {format:?}"),
        })
    }

    fn from_str(contents: &str, format: &str) -> anyhow::Result<Self> {
        match format {
            "yaml" => Self::from_yaml(contents),
            "json" => Self::from_json(contents),
            _ => bail!("Unsupported file format: {format:?}"),
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
