use super::*;
use include_dir::{include_dir, Dir};

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> + Init {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &[
        #[cfg(feature = "yaml")]
        "yaml",
        #[cfg(feature = "json")]
        "json",
        #[cfg(feature = "msgpack")]
        "msgpack",
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
    const RESOURCES_DIR: &'static Dir<'_> = &include_dir!("$CARGO_MANIFEST_DIR/resources");
    #[cfg(feature = "resources")]
    const RESOURCES_SUBDIR: &'static str = "";

    /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
    ///
    /// # Arguments:
    ///
    /// * `filepath` - Filepath, relative to the top of the `resources` folder (excluding any relevant prefix), from which to read the object
    #[cfg(feature = "resources")]
    fn from_resource<P: AsRef<Path>>(filepath: P, skip_init: bool) -> Result<Self, Error> {
        let filepath = Path::new(Self::RESOURCES_SUBDIR).join(filepath);
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                Error::SerdeError(format!("File extension could not be parsed: {filepath:?}"))
            })?;
        let file = Self::RESOURCES_DIR.get_file(&filepath).ok_or_else(|| {
            Error::SerdeError(format!("File not found in resources: {filepath:?}"))
        })?;
        Self::from_reader(&mut file.contents(), extension, skip_init)
    }

    /// List the available resources in the resources directory
    ///
    /// RETURNS: a vector of strings for resources that can be loaded
    #[cfg(feature = "resources")]
    fn list_resources() -> Result<Vec<PathBuf>, Error> {
        // Recursive function to walk the directory
        fn collect_paths(dir: &Dir, paths: &mut Vec<PathBuf>) {
            for entry in dir.entries() {
                match entry {
                    include_dir::DirEntry::Dir(subdir) => {
                        // Recursively process subdirectory
                        collect_paths(subdir, paths);
                    }
                    include_dir::DirEntry::File(file) => {
                        // Add file path
                        paths.push(file.path().to_path_buf());
                    }
                }
            }
        }

        let mut paths = Vec::new();
        if let Some(resources_subdir) = Self::RESOURCES_DIR.get_dir(Self::RESOURCES_SUBDIR) {
            collect_paths(resources_subdir, &mut paths);
            for p in paths.iter_mut() {
                *p = p
                    .strip_prefix(Self::RESOURCES_SUBDIR)
                    .map_err(|err| Error::SerdeError(format!("{err}")))?
                    .to_path_buf();
            }
            paths.sort();
        }
        Ok(paths)
    }

    /// Instantiates an object from a url.  Accepts yaml and json file types  
    /// # Arguments  
    /// - url: URL (either as a string or url type) to object  
    ///
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL.
    #[cfg(feature = "web")]
    fn from_url<S: AsRef<str>>(url: S, skip_init: bool) -> Result<Self, Error> {
        let url =
            url::Url::parse(url.as_ref()).map_err(|err| Error::SerdeError(format!("{err}")))?;
        let format = url
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|filename| Path::new(filename).extension())
            .and_then(OsStr::to_str)
            .with_context(|| "Could not parse file format from URL: {url:?}")
            .map_err(|err| Error::SerdeError(format!("{err}")))?;
        let mut response = ureq::get(url.as_ref())
            .call()
            .map_err(|err| Error::SerdeError(format!("{err}")))?
            .into_reader();
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
            #[cfg(feature = "msgpack")]
            "msgpack" => rmp_serde::encode::write(&mut wtr, self)
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
