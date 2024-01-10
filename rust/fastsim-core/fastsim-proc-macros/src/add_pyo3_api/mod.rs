//! Module that implements [super::add_pyo3_api]

#[macro_use]
mod pyo3_api_utils;

use crate::imports::*;

pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;
    let _is_state_or_history: bool =
        ident.to_string().contains("State") || ident.to_string().contains("HistoryVec");

    let mut impl_block = TokenStream2::default();
    let mut py_impl_block = TokenStream2::default();
    py_impl_block.extend::<TokenStream2>(crate::utilities::parse_ts_as_fn_defs(
        attr,
        vec![],
        false,
        vec![],
    ));

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        let field_names: Vec<String> = named
            .iter()
            .map(|f| f.ident.as_ref().unwrap().to_string())
            .collect();
        let has_orphaned: bool = field_names.iter().any(|f| f == "orphaned");

        for field in named.iter_mut() {
            let ident = field.ident.as_ref().unwrap();
            let ftype = field.ty.clone();

            // if attr.tokens.to_string().contains("skip_get"){
            // for (i, idx_del) in idxs_del.into_iter().enumerate() {
            //     attr_vec.remove(*idx_del - i);
            // }

            // this is my quick and dirty attempt at emulating:
            // https://github.com/PyO3/pyo3/blob/48690525e19b87818b59f99be83f1e0eb203c7d4/pyo3-macros-backend/src/pyclass.rs#L220

            let mut opts = FieldOptions::default();
            // attributes to retain, i.e. attributes that are not handled by this macro
            let keep: Vec<bool> = field
                .attrs
                .iter()
                .map(|x| match x.path.segments[0].ident.to_string().as_str() {
                    "api" => {
                        let meta = x.parse_meta().unwrap();
                        if let Meta::List(list) = meta {
                            for nested in list.nested {
                                if let syn::NestedMeta::Meta(opt) = nested {
                                    // println!("opt_path{:?}", opt.path().segments[0].ident.to_string().as_str());;
                                    let opt_name = opt.path().segments[0].ident.to_string();
                                    match opt_name.as_str() {
                                        "skip_get" => opts.skip_get = true,
                                        "skip_set" => opts.skip_set = true,
                                        "has_orphaned" => opts.field_has_orphaned = true,
                                        // todo: figure out how to use span to have rust-analyzer highlight the exact code
                                        // where this gets messed up
                                        _ => {
                                            abort!(
                                                x.span(),
                                                format!(
                                                    "Invalid api option: {}.\nValid options are: `skip_get`, `skip_set`, and `has_orphaned`.",
                                                    opt_name
                                                )
                                            )
                                        }
                                    }
                                }
                            }
                        }
                        false
                    }
                    _ => true,
                })
                .collect();
            // println!("options {:?}", opts);
            // this drops attrs matching `#[pyo3_api(...)]`, removing the field attribute from the struct def
            let new_attrs: (Vec<&syn::Attribute>, Vec<bool>) = field
                .attrs
                .iter()
                .zip(keep.iter())
                .filter(|(_a, k)| **k)
                .unzip();
            field.attrs = new_attrs.0.iter().cloned().cloned().collect();

            if let syn::Type::Path(type_path) = ftype.clone() {
                // println!(
                //     "{:?}",
                //     ident.to_string().as_str(),
                // type_path.clone().into_token_stream().to_string().as_str(),
                //     // attr_vec.clone().into_iter().collect::<Vec<syn::Attribute>>()
                // );
                impl_getters_and_setters(
                    type_path,
                    &mut py_impl_block,
                    ident,
                    opts,
                    has_orphaned,
                    ftype,
                );
            }
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut ast.fields {
        // tuple struct
        if ast.ident.to_string().contains("Vec") || ast.ident.to_string().contains("Array") {
            assert!(unnamed.len() == 1);
            for field in unnamed.iter() {
                let ftype = field.ty.clone();
                if let syn::Type::Path(type_path) = ftype.clone() {
                    let type_str = type_path.clone().into_token_stream().to_string();
                    let (re, container, py_new_body, tolist_body) = if type_str.contains("Vec") {
                        (
                            Regex::new(r"Vec < (.+) >").unwrap(),
                            "Vec".parse::<TokenStream2>().unwrap(),
                            "Self(v)".parse::<TokenStream2>().unwrap(),
                            "self.0.clone()".parse::<TokenStream2>().unwrap(),
                        )
                    } else if type_str.contains("Array1") {
                        (
                            Regex::new(r"Array1 < (.+) >").unwrap(),
                            "Array1".parse::<TokenStream2>().unwrap(),
                            "Self(Array1::from_vec(v))".parse::<TokenStream2>().unwrap(),
                            "self.0.to_vec()".parse::<TokenStream2>().unwrap(),
                        )
                    } else {
                        abort!(
                            ftype.span(),
                            "Invalid container type.  Must be Array1 or Vec."
                        )
                        // replace with proc_macro_error::abort macro
                    };

                    // println!("{}", type_str);
                    // println!("{}", &re.captures(&type_str).unwrap()[1]);
                    let contained_dtype: TokenStream2 = re.captures(&type_str).unwrap()[1]
                        .to_string()
                        .parse()
                        .unwrap();
                    py_impl_block.extend::<TokenStream2>(quote! {
                        #[new]
                        pub fn __new__(v: Vec<#contained_dtype>) -> Self {
                            #py_new_body
                        }

                        pub fn __repr__(&self) -> String {
                            format!("RustArray({:?})", self.0)
                        }
                        pub fn __str__(&self) -> String {
                            format!("{:?}", self.0)
                        }
                        pub fn __getitem__(&self, idx: i32) -> anyhow::Result<#contained_dtype> {
                            if idx >= self.0.len() as i32 {
                                bail!(PyIndexError::new_err("Index is out of bounds"))
                            } else if idx >= 0 {
                                Ok(self.0[idx as usize].clone())
                            } else {
                                Ok(self.0[self.0.len() + idx as usize].clone())
                            }
                        }
                        pub fn __setitem__(&mut self, _idx: usize, _new_value: #contained_dtype
                            ) -> anyhow::Result<()> {
                            bail!(PyNotImplementedError::new_err(
                                "Setting value at index is not implemented.
                                Run `tolist` method, modify value at index, and
                                then set entire vector.",
                            ))
                        }
                        pub fn tolist(&self) -> anyhow::Result<Vec<#contained_dtype>> {
                            Ok(#tolist_body)
                        }
                        pub fn __list__(&self) -> anyhow::Result<Vec<#contained_dtype>> {
                            Ok(#tolist_body)
                        }
                        pub fn __len__(&self) -> usize {
                            self.0.len()
                        }
                        pub fn is_empty(&self) -> bool {
                            self.0.is_empty()
                        }
                    });
                    impl_block.extend::<TokenStream2>(quote! {
                        pub fn new(value: #container<#contained_dtype>) -> Self {
                            Self(value)
                        }
                    });
                }
            }
        }
    } else {
        abort_call_site!("`add_pyo3_api` works only on named and tuple structs.");
    };

    // py_impl_block.extend::<TokenStream2>(quote! {
    //     #[staticmethod]
    //     #[pyo3(name = "default")]
    //     pub fn default_py() -> Self {
    //         Self::default()
    //     }
    // });

    py_impl_block.extend::<TokenStream2>(quote! {
        pub fn copy(&self) -> Self {self.clone()}
        pub fn __copy__(&self) -> Self {self.clone()}
        pub fn __deepcopy__(&self, _memo: &PyDict) -> Self {self.clone()}

        /// Read (deserialize) an object from a resource file packaged with the `fastsim-core` crate
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - Filepath, relative to the top of the `resources` folder, from which to read the object
        ///
        #[staticmethod]
        #[pyo3(name = "from_resource")]
        pub fn from_resource_py(filepath: &PyAny) -> anyhow::Result<Self> {
            Self::from_resource(PathBuf::extract(filepath)?)
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
        pub fn to_file_py(&self, filepath: &PyAny) -> anyhow::Result<()> {
           self.to_file(PathBuf::extract(filepath)?)
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
        pub fn from_file_py(filepath: &PyAny) -> anyhow::Result<Self> {
            Self::from_file(PathBuf::extract(filepath)?)
        }

        /// Write (serialize) an object into a string
        ///
        /// # Arguments:
        ///
        /// * `format`: `str` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[pyo3(name = "to_str")]
        pub fn to_str_py(&self, format: &str) -> anyhow::Result<String> {
            self.to_str(format)
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
        pub fn from_str_py(contents: &str, format: &str) -> anyhow::Result<Self> {
            Self::from_str(contents, format)
        }

        /// Write (serialize) an object to a JSON string
        #[pyo3(name = "to_json")]
        pub fn to_json_py(&self) -> anyhow::Result<String> {
            self.to_json()
        }

        /// Read (deserialize) an object to a JSON string
        ///
        /// # Arguments
        ///
        /// * `json_str`: `str` - JSON-formatted string to deserialize from
        ///
        #[staticmethod]
        #[pyo3(name = "from_json")]
        pub fn from_json_py(json_str: &str) -> anyhow::Result<Self> {
            Self::from_json(json_str)
        }

        /// Write (serialize) an object to a YAML string
        #[pyo3(name = "to_yaml")]
        pub fn to_yaml_py(&self) -> anyhow::Result<String> {
            self.to_yaml()
        }

        /// Read (deserialize) an object from a YAML string
        ///
        /// # Arguments
        ///
        /// * `yaml_str`: `str` - YAML-formatted string to deserialize from
        ///
        #[staticmethod]
        #[pyo3(name = "from_yaml")]
        pub fn from_yaml_py(yaml_str: &str) -> anyhow::Result<Self> {
            Self::from_yaml(yaml_str)
        }

        /// Write (serialize) an object to bincode-encoded `bytes`
        #[pyo3(name = "to_bincode")]
        pub fn to_bincode_py<'py>(&self, py: Python<'py>) -> anyhow::Result<&'py PyBytes> {
            Ok(PyBytes::new(py, &self.to_bincode()?))
        }

        /// Read (deserialize) an object from bincode-encoded `bytes`
        ///
        /// # Arguments
        ///
        /// * `encoded`: `bytes` - Encoded bytes to deserialize from
        ///
        #[staticmethod]
        #[pyo3(name = "from_bincode")]
        pub fn from_bincode_py(encoded: &PyBytes) -> anyhow::Result<Self> {
            Self::from_bincode(encoded.as_bytes())
        }
    });

    let impl_block = quote! {
        impl #ident {
            #impl_block
        }

        #[pymethods]
        #[cfg(feature="pyo3")]
        impl #ident {
            #py_impl_block
        }
    };

    let mut final_output = TokenStream2::default();
    // add pyclass attribute
    final_output.extend::<TokenStream2>(quote! {
        #[cfg_attr(feature="pyo3", pyclass(module = "fastsimrust", subclass))]
    });
    let mut output: TokenStream2 = ast.to_token_stream();
    output.extend(impl_block);
    // if ast.ident.to_string() == "RustSimDrive" {
    //     println!("{}", output.to_string());
    // }
    // println!("{}", output.to_string());
    final_output.extend::<TokenStream2>(output);
    final_output.into()
}

#[derive(Debug, Default, Clone)]
pub struct FieldOptions {
    /// if true, getters are not generated for a field
    pub skip_get: bool,
    /// if true, setters are not generated for a field
    pub skip_set: bool,
    /// if true, current field is itself a struct with `orphaned` field
    pub field_has_orphaned: bool,
}

pub fn impl_getters_and_setters(
    type_path: syn::TypePath,
    impl_block: &mut TokenStream2,
    ident: &proc_macro2::Ident,
    opts: FieldOptions,
    has_orphaned: bool,
    ftype: syn::Type,
) {
    let type_str = type_path.into_token_stream().to_string();
    match type_str.as_str() {
        "Array1 < f64 >" => {
            impl_vec_get_set!(opts, ident, impl_block, f64, Pyo3ArrayF64, has_orphaned);
        }
        "Array1 < u32 >" => {
            impl_vec_get_set!(opts, ident, impl_block, u32, Pyo3ArrayU32, has_orphaned);
        }
        "Array1 < i32 >" => {
            impl_vec_get_set!(opts, ident, impl_block, i32, Pyo3ArrayI32, has_orphaned);
        }
        "Array1 < bool >" => {
            impl_vec_get_set!(opts, ident, impl_block, bool, Pyo3ArrayBool, has_orphaned);
        }
        "Vec < f64 >" => {
            impl_vec_get_set!(opts, ident, impl_block, f64, Pyo3VecF64, has_orphaned);
        }
        _ => match ident.to_string().as_str() {
            "orphaned" => {
                impl_block.extend::<TokenStream2>(quote! {
                    #[getter]
                    pub fn get_orphaned(&self) -> bool {
                        self.orphaned
                    }
                    /// Reset the orphaned flag to false.
                    pub fn reset_orphaned(&mut self) {
                        self.orphaned = false;
                    }
                })
            }
            _ => {
                impl_get_body!(ftype, ident, impl_block, opts);
                impl_set_body!(ftype, ident, impl_block, has_orphaned, opts);
            }
        },
    }
}
