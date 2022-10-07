use crate::imports::*;

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    fn to_file(&self, filename: &str) {
        let file = PathBuf::from(filename);
        let c = match file.extension().unwrap().to_str().unwrap() {
            "json" => serde_json::to_writer(&File::create(file).unwrap(), self).unwrap(),
            "yaml" => serde_yaml::to_writer(&File::create(file).unwrap(), self).unwrap(),
            _ => serde_json::to_writer(&File::create(file).unwrap(), self).unwrap(),
        };
    }

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
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }

    /// json serialization method.
    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    /// json deserialization method.
    fn from_json(json_str: &str) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(serde_json::from_str(json_str)?)
    }

    /// yaml serialization method.
    fn to_yaml(&self) -> String {
        serde_yaml::to_string(&self).unwrap()
    }

    /// yaml deserialization method.
    fn from_yaml(yaml_str: &str) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(serde_yaml::from_str(yaml_str)?)
    }

    /// bincode serialization method.
    fn to_bincode(&self) -> Vec<u8> {
        serialize(&self).unwrap()
    }

    /// bincode deserialization method.
    fn from_bincode(encoded: &[u8]) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(deserialize(encoded)?)
    }
}

impl<T> SerdeAPI for T where T: Serialize + for<'a> Deserialize<'a> {}
