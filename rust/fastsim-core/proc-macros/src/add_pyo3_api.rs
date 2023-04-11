use crate::imports::*;
use crate::utilities::*;

pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;
    let is_state_or_history: bool =
        ident.to_string().contains("State") || ident.to_string().contains("HistoryVec");

    let mut impl_block = TokenStream2::default();
    let mut py_impl_block = TokenStream2::default();
    py_impl_block.extend::<TokenStream2>(crate::utilities::parse_ts_as_fn_defs(
        attr.into(),
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
            let mut iter = keep.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());

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

        if !is_state_or_history {
            py_impl_block.extend::<TokenStream2>(quote! {
                #[pyo3(name = "to_file")]
                pub fn to_file_py(&self, filename: &str) -> PyResult<()> {
                   Ok(self.to_file(filename)?)
                }

                #[classmethod]
                #[pyo3(name = "from_file")]
                pub fn from_file_py(_cls: &PyType, json_str:String) -> PyResult<Self> {
                    // unwrap is ok here because it makes sense to stop execution if a file is not loadable
                    Ok(Self::from_file(&json_str)?)
                }
            });
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
                        panic!("Invalid container type.  Must be Array1 or Vec.")
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
                        pub fn __getitem__(&self, idx: i32) -> PyResult<#contained_dtype> {
                            if idx >= self.0.len() as i32 {
                                Err(PyIndexError::new_err("Index is out of bounds"))
                            } else if idx >= 0 {
                                Ok(self.0[idx as usize])
                            } else {
                                Ok(self.0[self.0.len() + idx as usize])
                            }
                        }
                        pub fn __setitem__(&mut self, _idx: usize, _new_value: #contained_dtype
                            ) -> PyResult<()> {
                            Err(PyNotImplementedError::new_err(
                                "Setting value at index is not implemented.
                                Run `tolist` method, modify value at index, and
                                then set entire vector.",
                            ))
                        }
                        pub fn tolist(&self) -> PyResult<Vec<#contained_dtype>> {
                            Ok(#tolist_body)
                        }
                        pub fn __list__(&self) -> PyResult<Vec<#contained_dtype>> {
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
        } else {
            panic!("Likely invalid use of `add_pyo3_api` macro.");
        }
    } else {
        panic!("Likely invalid use of `add_pyo3_api` macro.");
    };

    // py_impl_block.extend::<TokenStream2>(quote! {
    //     #[classmethod]
    //     #[pyo3(name = "default")]
    //     pub fn default_py(_cls: &PyType) -> PyResult<Self> {
    //         Ok(Self::default())
    //     }
    // });

    py_impl_block.extend::<TokenStream2>(quote! {
        pub fn copy(&self) -> Self {self.clone()}
        pub fn __copy__(&self) -> Self {self.clone()}
        pub fn __deepcopy__(&self, _memo: &PyDict) -> Self {self.clone()}

        /// json serialization method.
        #[pyo3(name = "to_json")]
        pub fn to_json_py(&self) -> PyResult<String> {
            Ok(self.to_json())
        }

        #[classmethod]
        /// json deserialization method.
        #[pyo3(name = "from_json")]
        pub fn from_json_py(_cls: &PyType, json_str: &str) -> PyResult<Self> {
            Ok(Self::from_json(json_str)?)
        }

        /// yaml serialization method.
        #[pyo3(name = "to_yaml")]
        pub fn to_yaml_py(&self) -> PyResult<String> {
            Ok(self.to_yaml())
        }

        #[classmethod]
        /// yaml deserialization method.
        #[pyo3(name = "from_yaml")]
        pub fn from_yaml_py(_cls: &PyType, yaml_str: &str) -> PyResult<Self> {
            Ok(Self::from_yaml(yaml_str)?)
        }

        /// bincode serialization method.
        #[pyo3(name = "to_bincode")]
        pub fn to_bincode_py<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            Ok(PyBytes::new(py, &self.to_bincode()))
        }

        #[classmethod]
        /// bincode deserialization method.
        #[pyo3(name = "from_bincode")]
        pub fn from_bincode_py(_cls: &PyType, encoded: &PyBytes) -> PyResult<Self> {
            Ok(Self::from_bincode(encoded.as_bytes())?)
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
