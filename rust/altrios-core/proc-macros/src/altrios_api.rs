use crate::imports::*;
mod altrios_api_utils;
use crate::utilities::parse_ts_as_fn_defs;
use altrios_api_utils::*;

pub(crate) fn altrios_api(attr: TokenStream, item: TokenStream) -> TokenStream {
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
            let ident = field.ident.as_ref().unwrap();
            let ftype = field.ty.clone();

            // if attr.tokens.to_string().contains("skip_get"){
            // for (i, idx_del) in idxs_del.into_iter().enumerate() {
            //     attr_vec.remove(*idx_del - i);
            // }

            // this is my quick and dirty attempt at emulating:
            // https://github.com/PyO3/pyo3/blob/48690525e19b87818b59f99be83f1e0eb203c7d4/pyo3-macros-backend/src/pyclass.rs#L220

            let mut opts = FieldOptions::default();
            let keep: Vec<bool> = field
                .attrs
                .iter()
                .map(|x| match x.path.segments[0].ident.to_string().as_str() { // todo: check length of segments for robustness
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
                                        // todo: figure out how to use span to have rust-analyzer highlight the exact code
                                        // where this gets messed up
                                        _ => {
                                            abort!(
                                                x.span(),
                                                format!(
                                                    "Invalid api option: {opt_name}.\nValid options are: `skip_get`, `skip_set`."
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
            let mut iter = keep.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());

            impl_getters_and_setters(&mut py_impl_block, ident, &opts, &ftype);
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut ast.fields {
        // tuple struct
        if ast.ident.to_string().contains("Vec") {
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
                "Invalid use of `altrios_api` macro.  Expected tuple struct with Vec in name."
            );
        }
    } else {
        abort_call_site!(
            "Invalid use of `altrios_api` macro.  Expected tuple struct or C-style struct."
        );
    };

    py_impl_block.extend::<TokenStream2>(quote! {
        #[classmethod]
        #[pyo3(name = "default")]
        /// Exposes `default` to python.
        fn default_py(_cls: &PyType) -> PyResult<Self> {
            Ok(Self::default())
        }

        /// json serialization method.
        #[pyo3(name = "to_json")]
        fn to_json_py(&self) -> PyResult<String> {
            Ok(self.to_json())
        }

        #[classmethod]
        /// json deserialization method.
        #[pyo3(name = "from_json")]
        fn from_json_py(_cls: &PyType, json_str: &str) -> PyResult<Self> {
            Ok(Self::from_json(json_str)?)
        }

        /// yaml serialization method.
        #[pyo3(name = "to_yaml")]
        fn to_yaml_py(&self) -> PyResult<String> {
            Ok(self.to_yaml())
        }

        #[classmethod]
        /// yaml deserialization method.
        #[pyo3(name = "from_yaml")]
        fn from_yaml_py(_cls: &PyType, yaml_str: &str) -> PyResult<Self> {
            Ok(Self::from_yaml(yaml_str)?)
        }

        /// bincode serialization method.
        #[pyo3(name = "to_bincode")]
        fn to_bincode_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            Ok(PyBytes::new(py, &self.to_bincode()))
        }

        #[classmethod]
        /// bincode deserialization method.
        #[pyo3(name = "from_bincode")]
        fn from_bincode_py(_cls: &PyType, encoded: &PyBytes) -> PyResult<Self> {
            Ok(Self::from_bincode(encoded.as_bytes())?)
        }

        /// `__copy__` magic method that uses `clone`.
        fn __copy__(&self) -> Self {
            self.clone()
        }

        /// `__deepcopy__` magic method that uses `clone`.
        fn __deepcopy__(&self) -> Self {
            self.clone()
        }

        #[pyo3(name = "clone")]
        /// calls Rust's `clone`.
        fn clone_py(&self) -> Self {
            self.clone()
        }
    });

    let py_impl_block = quote! {
        #[pymethods]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block

            #[classmethod]
            #[pyo3(name = "from_file")]
            /// Exposes `from_file` to Python.
            fn from_file_py(_cls: &PyType, filename: String) -> PyResult<Self> {
                Ok(Self::from_file(&filename)?)
            }

            #[pyo3(name = "to_file")]
            /// Exposes `to_file` to Python.
            fn to_file_py(&self, filename: &str) -> PyResult<()> {
                Ok(self.to_file(filename)?)
            }
        }
    };
    let mut final_output = TokenStream2::default();
    final_output.extend::<TokenStream2>(quote! {
        #[cfg_attr(feature="pyo3", pyclass)]
    });
    let mut output: TokenStream2 = ast.to_token_stream();
    output.extend(impl_block);
    output.extend(py_impl_block);
    // println!("{}", output.to_string());
    final_output.extend::<TokenStream2>(output);
    final_output.into()
}
