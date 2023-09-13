use crate::imports::*;

pub trait Linspace {
    /// Generate linearly spaced vec
    /// # Arguments
    /// - start - starting point
    /// - stop - stopping point, inclusive
    /// - n_elements - number of array elements
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

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    /// runs any initialization steps that may be needed.
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
            .unwrap_or_default();

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

impl<T: SerdeAPI> SerdeAPI for Vec<T> {
    fn init(&mut self) -> anyhow::Result<()> {
        for val in self {
            val.init()?
        }
        Ok(())
    }
}

pub trait Diff {
    fn diff(&self) -> Vec<f64>;
}

impl Diff for Vec<f64> {
    fn diff(&self) -> Vec<f64> {
        self.windows(2)
            .map(|vs| {
                let [x, y] = vs else {unreachable!()};
                y - x
            })
            .collect()
    }
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
        assert_eq!(Vec::linspace(0., 2., 3).diff(), vec![0., 1., 1.]);
    }
}
