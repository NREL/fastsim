#[macro_export]
macro_rules! impl_serde {
    ($self: ident, $component:ident, $default_folder:expr, $flag: literal) => {
        pub fn to_file(&self, filename: &str) -> Result<(),Box<dyn Error>> {
            let file = PathBuf::from(filename);
            let c = match file.extension().unwrap().to_str().unwrap() {
                "json" => {serde_json::to_writer(&File::create(file)?, self)?},
                "yaml" => {serde_yaml::to_writer(&File::create(file)?, self)?},
                _ => {serde_json::to_writer(&File::create(file)?, self)?},
            };
            Ok(c)
        }
    
        fn from_file_parser(filename: &str) -> Result<$component, Box<dyn Error>> {
            let mut pathbuf = PathBuf::from(filename);
            if !pathbuf.exists() {
                // if file doesn't exist, try to find it in the resources folder
                let mut root =  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .to_path_buf();
                root.push($default_folder);

                if [root.to_owned().canonicalize()?, pathbuf.clone()]
                    .iter()
                    .collect::<PathBuf>()
                    .exists()
                {
                    pathbuf = [root.to_owned(), pathbuf].iter().collect::<PathBuf>();
                }
            }
            let file =  File::open(&pathbuf)?;
            let c = match pathbuf.extension().unwrap().to_str().unwrap() {
                "yaml" => {serde_yaml::from_reader(file)?},
                "json" => {serde_json::from_reader(file)?},
                _ => {serde_json::from_reader(file)?},
            };
            Ok(c)
        }

        pub fn from_file(filename: &str) -> Self {
            let mut veh = Self::from_file_parser(filename).unwrap();
            veh.set_derived();
            veh
        }
    
    };

    ($self: ident, $component:ident, $default_folder:expr) => {
        pub fn to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
            let file = PathBuf::from(filename);
            let _ = match file.extension().unwrap().to_str().unwrap() {
                "json" => {serde_json::to_writer(&File::create(file)?, self)?},
                "yaml" => {serde_yaml::to_writer(&File::create(file)?, self)?},
                _ => {serde_json::to_writer(&File::create(file)?, self)?},
            };
            Ok(())
        }
    
        fn from_file_parser(filename: &str) -> Result<$component, Box<dyn Error>> {
            let mut pathbuf = PathBuf::from(filename);
            if !pathbuf.exists() {
                // if file doesn't exist, try to find it in the resources folder
                let mut root =  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .to_path_buf();
                root.push($default_folder);

                if [root.to_owned().canonicalize()?, pathbuf.clone()]
                    .iter()
                    .collect::<PathBuf>()
                    .exists()
                {
                    pathbuf = [root.to_owned(), pathbuf].iter().collect::<PathBuf>();
                }
            }
            let file =  File::open(&pathbuf)?;
            let c = match pathbuf.extension().unwrap().to_str().unwrap() {
                "yaml" => {serde_yaml::from_reader(file)?},
                "json" => {serde_json::from_reader(file)?},
                _ => {serde_json::from_reader(file)?},
            };
            Ok(c)
        }

        pub fn from_file(filename: &str) -> Self {
            Self::from_file_parser(filename).unwrap()
        }
    };
}