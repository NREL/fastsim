/// Generates `to_file` and `from_file_parser` methods for `$component` with `$default_folder`.
#[macro_export]
macro_rules! impl_serde {
    ($component:ident, $default_folder:expr) => {
        pub fn to_file(&self, filename: &str) -> Result<(), Box<dyn Error>> {
            let file = PathBuf::from(filename);
            let c = match file.extension().unwrap().to_str().unwrap() {
                "json" => serde_json::to_writer(&File::create(file)?, self)?,
                "yaml" => serde_yaml::to_writer(&File::create(file)?, self)?,
                _ => serde_json::to_writer(&File::create(file)?, self)?,
            };
            Ok(c)
        }

        fn from_file_parser(filename: &str) -> Result<$component, Box<dyn Error>> {
            let mut pathbuf = PathBuf::from(filename);
            if !pathbuf.exists() {
                // if file doesn't exist, try to find it in the resources folder
                let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
            let file = File::open(&pathbuf)?;
            let c = match pathbuf.extension().unwrap().to_str().unwrap() {
                "yaml" => serde_yaml::from_reader(file)?,
                "json" => serde_json::from_reader(file)?,
                _ => serde_json::from_reader(file)?,
            };
            Ok(c)
        }
    };
}

/// Generates `from_file` method
#[macro_export]
macro_rules! impl_from_file {
    () => {
        pub fn from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
            Self::from_file_parser(filename)
        }
    };
}

/// Given pairs of arbitrary keys and values, prints "key: value" to python intepreter.  
/// Given str, prints str.  
/// Using this will break `cargo test` but work with `maturin develop`.  
#[macro_export]
macro_rules! print_to_py {
    ( $( $x:expr, $y:expr ),* ) => {
        {
            pyo3::Python::with_gil(|py| {
                let locals = pyo3::types::PyDict::new(py);
                $(
                    locals.set_item($x, $y).unwrap();
                    py.run(
                        &format!("print(f\"{}: {{{}:.3g}}\")", $x, $x),
                        None,
                        Some(locals),
                    )
                    .expect(&format!("printing `{}` failed", $x));
                )*
            });
        };
    };
    ( $x:expr ) => {
        {
            // use pyo3::py_run;
            pyo3::Python::with_gil(|py| {
                    py.run(
                        &format!("print({})", $x),
                        None,
                        None,
                    )
                    .expect(&format!("printing `{}` failed", $x));
            });
        };
    }
}

#[macro_export]
macro_rules! check_orphaned_and_set {
    ($struct_self: ident, $field: ident, $value: expr) => {
        if !$struct_self.orphaned {
            $struct_self.$field = $value;
            Ok(())
        } else {
            Err(PyAttributeError::new_err(utils::NESTED_STRUCT_ERR))
        }
    };
}
