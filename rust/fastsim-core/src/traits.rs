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
    /// * `filename`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result
    fn to_file(&self, filename: &str) -> Result<(), anyhow::Error> {
        let file = PathBuf::from(filename);
        match file.extension().unwrap().to_str().unwrap() {
            "json" => serde_json::to_writer(&File::create(file)?, self)?,
            "yaml" => serde_yaml::to_writer(&File::create(file)?, self)?,
            _ => serde_json::to_writer(&File::create(file)?, self)?,
        };
        Ok(())
    }

    /// Read from file and return instantiated struct. Method adaptively calls deserialization
    /// methods dependent on the suffix of the file name given as str.
    /// Function returns a dynamic Error Result if it fails.
    ///
    /// # Argument:
    ///
    /// * `filename`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result wrapping data structure if method is called successfully; otherwise a dynamic
    /// Error.
    fn from_file(filename: &str) -> Result<Self, anyhow::Error>
    where
        Self: std::marker::Sized,
        for<'de> Self: Deserialize<'de>,
    {
        let extension = Path::new(filename)
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("");

        let file = File::open(filename)?;
        // deserialized file
        let mut file_de: Self = match extension {
            "yaml" => serde_yaml::from_reader(file)?,
            "json" => serde_json::from_reader(file)?,
            _ => bail!("Unsupported file extension {}", extension),
        };
        file_de.init()?;
        Ok(file_de)
    }

    /// json serialization method.
    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    /// json deserialization method.
    fn from_json(json_str: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(json_str)?)
    }

    /// yaml serialization method.
    fn to_yaml(&self) -> String {
        serde_yaml::to_string(&self).unwrap()
    }

    /// yaml deserialization method.
    fn from_yaml(yaml_str: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_yaml::from_str(yaml_str)?)
    }

    /// bincode serialization method.
    fn to_bincode(&self) -> Vec<u8> {
        serialize(&self).unwrap()
    }

    /// bincode deserialization method.
    fn from_bincode(encoded: &[u8]) -> Result<Self, anyhow::Error> {
        Ok(deserialize(encoded)?)
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
