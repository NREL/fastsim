use crate::imports::*;
mod fastsim_api_utils;
use crate::utilities::parse_ts_as_fn_defs;
use fastsim_api_utils::*;

pub(crate) fn fastsim_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    let ident = &ast.ident;
    // println!("{}", String::from("*").repeat(30));
    // println!("struct: {}", ast.ident.to_string());

    let mut py_impl_block = TokenStream2::default();
    let mut impl_block = TokenStream2::default();

    py_impl_block.extend::<TokenStream2>(parse_ts_as_fn_defs(attr, vec![], false, vec![]));

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        // struct with named fields
        for field in named.iter_mut() {
            // if attr.tokens.to_string().contains("skip_get"){
            // for (i, idx_del) in idxs_del.into_iter().enumerate() {
            //     attr_vec.remove(*idx_del - i);
            // }

            // this is my quick and dirty attempt at emulating:
            // https://github.com/PyO3/pyo3/blob/48690525e19b87818b59f99be83f1e0eb203c7d4/pyo3-macros-backend/src/pyclass.rs#L220

            let mut opts = FieldOptions::default();
            // find attributes that need to be retained for other macros
            // and handle the `api` attributes
            let keep: Vec<bool> = field
                .attrs
                .iter()
                .map(|attr| {
                    if let Meta::List(ml) = &attr.meta {
                        // catch the `api` in `#[api(skip_get)]`
                        if ml.path.is_ident("api") {
                            let opt_str = ml.tokens.to_string();
                            let opt_split = opt_str.as_str().split(",");
                            let mut opt_vec =
                                opt_split.map(|opt| opt.trim()).collect::<Vec<&str>>();

                            // find the `skip_get` option
                            let mut idx_skip_get: Option<usize> = None;
                            opt_vec.iter().enumerate().for_each(|(i, opt)| {
                                if *opt == "skip_get" {
                                    idx_skip_get = Some(i);
                                    opts.skip_get = true;
                                }
                            });
                            if let Some(idx_skip_get) = idx_skip_get {
                                let _ = opt_vec.remove(idx_skip_get);
                            }

                            // find the `skip_set` option
                            let mut idx_skip_set: Option<usize> = None;
                            opt_vec.iter().enumerate().for_each(|(i, opt)| {
                                if *opt == "skip_set" {
                                    idx_skip_set = Some(i);
                                    opts.skip_set = true;
                                }
                            });
                            if let Some(idx_skip_set) = idx_skip_set {
                                let _ = opt_vec.remove(idx_skip_set);
                            }
                            if !opt_vec.is_empty() {
                                emit_error!(ml.span(), "Invalid option(s): {:?}", opt_vec);
                            }
                            false // this attribute should not be retained because it is handled solely by this proc macro
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                })
                .collect();
            // println!("options in {}: {:?}", field.ident.to_token_stream(), opts);
            let mut iter = keep.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());

            impl_getters_and_setters(&mut py_impl_block, field, &opts);
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut ast.fields {
        // tuple struct
        assert!(unnamed.len() == 1);
        for field in unnamed.iter() {
            let ftype = field.ty.clone();
            if let syn::Type::Path(type_path) = ftype.clone() {
                let type_str = type_path.clone().into_token_stream().to_string();
                if type_str.contains("Vec") {
                    let re = Regex::new(r"Vec < (.+) >").unwrap();
                    // println!("{}", type_str);
                    // println!("{}", &re.captures(&type_str).unwrap()[1]);
                    let contained_dtype: TokenStream2 = re.captures(&type_str).unwrap()[1]
                        .to_string()
                        .parse()
                        .unwrap();
                    py_impl_block.extend::<TokenStream2>(
                            quote! {
                                #[new]
                                /// Rust-defined `__new__` magic method for Python used exposed via PyO3.
                                fn __new__(v: Vec<#contained_dtype>) -> Self {
                                    Self(v)
                                }
                                /// Rust-defined `__repr__` magic method for Python used exposed via PyO3.
                                fn __repr__(&self) -> String {
                                    format!("Pyo3Vec({:?})", self.0)
                                }
                                /// Rust-defined `__str__` magic method for Python used exposed via PyO3.
                                fn __str__(&self) -> String {
                                    format!("{:?}", self.0)
                                }
                                /// Rust-defined `__getitem__` magic method for Python used exposed via PyO3.
                                /// Prevents the Python user getting item directly using indexing.
                                fn __getitem__(&self, _idx: usize) -> PyResult<()> {
                                    Err(PyNotImplementedError::new_err(
                                        "Getting Rust vector value at index is not implemented.
                                        Run `tolist` method to convert to standalone Python list.",
                                    ))
                                }
                                /// Rust-defined `__setitem__` magic method for Python used exposed via PyO3.
                                /// Prevents the Python user setting item using indexing.
                                fn __setitem__(&mut self, _idx: usize, _new_value: #contained_dtype) -> PyResult<()> {
                                    Err(PyNotImplementedError::new_err(
                                        "Setting list value at index is not implemented.
                                        Run `tolist` method, modify value at index, and
                                        then set entire list.",
                                    ))
                                }
                                /// PyO3-exposed method to convert vec-containing struct to Python list.
                                fn tolist(&self) -> PyResult<Vec<#contained_dtype>> {
                                    Ok(self.0.clone())
                                }
                                /// Rust-defined `__len__` magic method for Python used exposed via PyO3.
                                /// Returns the length of the Rust vector.
                                fn __len__(&self) -> usize {
                                    self.0.len()
                                }
                                /// PyO3-exposed method to check if the vec-containing struct is empty.
                                fn is_empty(&self) -> bool {
                                    self.0.is_empty()
                                }
                            }
                        );
                    impl_block.extend::<TokenStream2>(quote! {
                        impl #ident{
                            /// Implement the non-Python `new` method.
                            pub fn new(value: Vec<#contained_dtype>) -> Self {
                                Self(value)
                            }
                        }
                    });
                }
            }
        }
    } else {
        abort_call_site!(
            "Invalid use of `fastsim_api` macro.  Expected tuple struct or C-style struct."
        );
    };

    py_impl_block.extend::<TokenStream2>(quote! {
        pub fn copy(&self) -> Self {self.clone()}
        pub fn __copy__(&self) -> Self {self.clone()}
        pub fn __deepcopy__(&self, _memo: &PyDict) -> Self {self.clone()}

        /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - Filepath, relative to the top of the `resources` folder (excluding any relevant prefix), from which to read the object
        ///
        #[cfg(feature = "resources")]
        #[staticmethod]
        #[pyo3(name = "from_resource")]
        pub fn from_resource_py(filepath: &PyAny, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_resource(PathBuf::extract(filepath)?, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
        ///
        /// # Arguments:
        ///
        /// * `url`: `str` - URL from which to read the object
        ///
        #[cfg(feature = "web")]
        #[staticmethod]
        #[pyo3(name = "from_url")]
        pub fn from_url_py(url: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_url(url, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a file.
        /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
        /// Creates a new file if it does not already exist, otherwise truncates the existing file.
        ///
        /// # Arguments
        ///
        /// * `filepath`: `str | pathlib.Path` - The filepath at which to write the object
        ///
        #[pyo3(name = "to_file")]
        pub fn to_file_py(&self, filepath: &PyAny) -> PyResult<()> {
           self.to_file(PathBuf::extract(filepath)?).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a file.
        /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - The filepath from which to read the object
        ///
        #[staticmethod]
        #[pyo3(name = "from_file")]
        pub fn from_file_py(filepath: &PyAny, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_file(PathBuf::extract(filepath)?, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object into a string
        ///
        /// # Arguments:
        ///
        /// * `format`: `str` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[pyo3(name = "to_str")]
        pub fn to_str_py(&self, format: &str) -> PyResult<String> {
            self.to_str(format).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a string
        ///
        /// # Arguments:
        ///
        /// * `contents`: `str` - The string containing the object data
        /// * `format`: `str` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[staticmethod]
        #[pyo3(name = "from_str")]
        pub fn from_str_py(contents: &str, format: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_str(contents, format, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to bincode-encoded `bytes`
        #[cfg(feature = "bincode")]
        #[pyo3(name = "to_bincode")]
        pub fn to_bincode_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            PyResult::Ok(PyBytes::new(py, &self.to_bincode()?)).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from bincode-encoded `bytes`
        ///
        /// # Arguments
        ///
        /// * `encoded`: `bytes` - Encoded bytes to deserialize from
        ///
        #[cfg(feature = "bincode")]
        #[staticmethod]
        #[pyo3(name = "from_bincode")]
        pub fn from_bincode_py(encoded: &PyBytes, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_bincode(encoded.as_bytes(), skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a JSON string
        #[cfg(feature = "json")]
        #[pyo3(name = "to_json")]
        pub fn to_json_py(&self) -> PyResult<String> {
            self.to_json().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object to a JSON string
        ///
        /// # Arguments
        ///
        /// * `json_str`: `str` - JSON-formatted string to deserialize from
        ///
        #[cfg(feature = "json")]
        #[staticmethod]
        #[pyo3(name = "from_json")]
        pub fn from_json_py(json_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_json(json_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a TOML string
        #[cfg(feature = "toml")]
        #[pyo3(name = "to_toml")]
        pub fn to_toml_py(&self) -> PyResult<String> {
            self.to_toml().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object to a TOML string
        ///
        /// # Arguments
        ///
        /// * `toml_str`: `str` - TOML-formatted string to deserialize from
        ///
        #[cfg(feature = "toml")]
        #[staticmethod]
        #[pyo3(name = "from_toml")]
        pub fn from_toml_py(toml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_toml(toml_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a YAML string
        #[cfg(feature = "yaml")]
        #[pyo3(name = "to_yaml")]
        pub fn to_yaml_py(&self) -> PyResult<String> {
            self.to_yaml().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a YAML string
        ///
        /// # Arguments
        ///
        /// * `yaml_str`: `str` - YAML-formatted string to deserialize from
        ///
        #[cfg(feature = "yaml")]
        #[staticmethod]
        #[pyo3(name = "from_yaml")]
        pub fn from_yaml_py(yaml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_yaml(yaml_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }
    });

    let py_impl_block = quote! {
        #[allow(non_snake_case)]
        #[pymethods]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block
        }
    };
    let mut final_output = TokenStream2::default();
    final_output.extend::<TokenStream2>(quote! {
        #[cfg_attr(feature="pyo3", pyclass(module = "fastsim", subclass))]
    });
    let mut output: TokenStream2 = ast.to_token_stream();
    output.extend(impl_block);
    output.extend(py_impl_block);
    // println!("{}", output.to_string());
    final_output.extend::<TokenStream2>(output);
    final_output.into()
}
