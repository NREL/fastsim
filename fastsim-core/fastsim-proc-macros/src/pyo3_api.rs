use crate::imports::*;

pub(crate) fn named_struct_pyo3_api(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let py_impl_block = syn::parse_macro_input!(item as syn::ItemImpl);
    let ident = match *py_impl_block.self_ty {
        syn::Type::Path(type_path) if type_path.path.segments.len() == 1 => {
            let first_seg = type_path.path.segments.first().unwrap();
            first_seg.ident.clone()
        }
        _ => abort_call_site!(String::from("Invalid usage")),
    };
    let mut py_impl_block_body: TokenStream2 = Default::default();
    for item in py_impl_block.items {
        if let syn::ImplItem::Fn(item_fn) = item {
            py_impl_block_body.extend(quote! {#item_fn});
        }
    }
    add_serde_methods(&mut py_impl_block_body);
    let mut new_py_impl_block: TokenStream2 = Default::default();
    new_py_impl_block.extend(quote! {
        #[allow(non_snake_case)]
        #[pymethods]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block_body

            fn __str__(&self) -> String {
                format!("{self:?}")
            }
        }
    });

    new_py_impl_block.into()
}

fn add_serde_methods(py_impl_block: &mut TokenStream2) {
    // NOTE: may be helpful to add an `init_py` method
    py_impl_block.extend::<TokenStream2>(quote! {
        pub fn copy(&self) -> Self {self.clone()}
        pub fn __copy__(&self) -> Self {self.clone()}
        pub fn __deepcopy__(&self, _memo: &Bound<PyDict>) -> Self {self.clone()}

        /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - Filepath, relative to the top of the `resources` folder (excluding any relevant prefix), from which to read the object
        ///
        #[cfg(feature = "resources")]
        #[staticmethod]
        #[pyo3(name = "from_resource")]
        #[pyo3(signature = (filepath, skip_init=None))]
        pub fn from_resource_py(filepath: &Bound<PyAny>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_resource(
                PathBuf::extract_bound(filepath)?,
                skip_init.unwrap_or_default()
            ).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
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
        #[pyo3(signature = (url, skip_init=None))]
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
        pub fn to_file_py(&self, filepath: &Bound<PyAny>) -> PyResult<()> {
           self.to_file(PathBuf::extract_bound(filepath)?).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
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
        #[pyo3(signature = (filepath, skip_init=None))]
        pub fn from_file_py(filepath: &Bound<PyAny>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_file(PathBuf::extract_bound(filepath)?, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
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
        #[pyo3(signature = (contents, format, skip_init=None))]
        pub fn from_str_py(contents: &str, format: &str, skip_init: Option<bool>) -> PyResult<Self> {
            SerdeAPI::from_str(contents, format, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a JSON string
        #[cfg(feature = "json")]
        #[pyo3(name = "to_json")]
        pub fn to_json_py(&self) -> PyResult<String> {
            self.to_json().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a JSON string
        ///
        /// # Arguments
        ///
        /// * `json_str`: `str` - JSON-formatted string to deserialize from
        ///
        #[cfg(feature = "json")]
        #[staticmethod]
        #[pyo3(name = "from_json")]
        #[pyo3(signature = (json_str, skip_init=None))]
        pub fn from_json_py(json_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_json(json_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a message pack
        #[cfg(feature = "msgpack")]
        #[pyo3(name = "to_msg_pack")]
        pub fn to_msg_pack_py(&self) -> PyResult<Vec<u8>> {
            self.to_msg_pack().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a message pack
        ///
        /// # Arguments
        ///
        /// * `msg_pack`: message pack
        ///
        #[cfg(feature = "msgpack")]
        #[staticmethod]
        #[pyo3(name = "from_msg_pack")]
        #[pyo3(signature = (msg_pack, skip_init=None))]
        pub fn from_msg_pack_py(msg_pack: &Bound<PyBytes>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_msg_pack(
                msg_pack.as_bytes(),
                skip_init.unwrap_or_default()
            ).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
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
        #[pyo3(signature = (toml_str, skip_init=None))]
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
        #[pyo3(signature = (yaml_str, skip_init=None))]
        pub fn from_yaml_py(yaml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_yaml(yaml_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }
    });
}

pub(crate) fn tuple_struct_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast_attr = syn::parse_macro_input!(attr as syn::ItemStruct);
    let ident = &ast_attr.ident;

    let py_impl_block = syn::parse_macro_input!(item as syn::ItemImpl);
    let mut py_impl_block_body: TokenStream2 = Default::default();
    for item in py_impl_block.items {
        if let syn::ImplItem::Fn(item_fn) = item {
            py_impl_block_body.extend(quote! {#item_fn});
        }
    }
    let impl_block = TokenStream2::default();
    if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut ast_attr.fields {
        process_tuple_struct(ident, &mut py_impl_block_body, impl_block, unnamed);
    } else {
        abort_call_site!(
            "Invalid use of `fastsim_api` macro.  Expected tuple struct or C-style struct."
        );
    };
    add_serde_methods(&mut py_impl_block_body);
    let mut new_py_impl_block: TokenStream2 = Default::default();
    new_py_impl_block.extend(quote! {
        #[allow(non_snake_case)]
        #[pymethods]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block_body
        }
    });

    new_py_impl_block.into()
}

fn process_tuple_struct(
    ident: &Ident,
    py_impl_block: &mut TokenStream2,
    mut impl_block: TokenStream2,
    unnamed: &mut syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) {
    // tuple struct
    assert!(unnamed.len() == 1);
    let re = Regex::new(r"Vec < (.+) >").unwrap();
    for field in unnamed.iter() {
        let ftype = field.ty.clone();
        if let syn::Type::Path(type_path) = ftype.clone() {
            let type_str = type_path.clone().into_token_stream().to_string();
            if type_str.contains("Vec") {
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
                        #[pyo3(name = "is_empty")]
                        fn is_empty_py(&self) -> bool {
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
}
