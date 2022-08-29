use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort, proc_macro_error};
use quote::{quote, ToTokens, TokenStreamExt}; // ToTokens is implicitly used as a trait
use regex::Regex;
use syn::{spanned::Spanned, DeriveInput, Ident, Meta};

mod utilities;
use utilities::{impl_getters_and_setters, FieldOptions};

/// macro for creating appropriate setters and getters for pyo3 struct attributes
#[proc_macro_error]
#[proc_macro_attribute]
pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;
    let is_state_or_history: bool = ident.to_string().contains("State") || ident.to_string().contains("HistoryVec");

    let mut impl_block = TokenStream2::default();
    let mut py_impl_block = TokenStream2::default();
    py_impl_block.extend::<TokenStream2>(attr.into());

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
                    self.to_file(filename).unwrap();
                    Ok(())
                }

                #[classmethod]
                #[pyo3(name = "from_file")]
                pub fn from_file_py(_cls: &PyType, json_str:String) -> PyResult<Self> {
                    let obj: #ident = Self::from_file(&json_str);
                    Ok(obj)
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
                    let contained_dtype: TokenStream2 = (&re.captures(&type_str).unwrap()[1])
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

        pub fn to_bincode<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
            Ok(PyBytes::new(py, &serialize(&self).unwrap()))
        }
        #[classmethod]
        pub fn from_bincode(_cls: &PyType, encoded: &PyBytes) -> PyResult<Self> {
            Ok(deserialize(encoded.as_bytes()).unwrap())
        }

        pub fn to_json(&self) -> PyResult<String> {
            Ok(serde_json::to_string(&self).unwrap())
        }
        #[classmethod]
        pub fn from_json(_cls: &PyType, json_str: &str) -> PyResult<Self> {
            Ok(serde_json::from_str(json_str).unwrap())
        }

        pub fn to_yaml(&self) -> PyResult<String> {
            Ok(serde_yaml::to_string(&self).unwrap())
        }
        #[classmethod]
        pub fn from_yaml(_cls: &PyType, yaml_str: &str) -> PyResult<Self> {
            Ok(serde_yaml::from_str(yaml_str).unwrap())
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

// taken from https://github.com/lumol-org/soa-derive/blob/master/soa-derive-internal/src/input.rs
pub(crate) trait TokenStreamIterator {
    fn concat_by(
        self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream;
    fn concat(self) -> proc_macro2::TokenStream;
}

impl<T: Iterator<Item = proc_macro2::TokenStream>> TokenStreamIterator for T {
    fn concat_by(
        mut self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self.next() {
            Some(first) => self.fold(first, f),
            None => quote! {},
        }
    }

    fn concat(self) -> proc_macro2::TokenStream {
        self.concat_by(|a, b| quote! { #a #b })
    }
}

#[proc_macro_derive(HistoryVec)]
pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    let ast: DeriveInput = syn::parse(input).unwrap();
    let original_name = &ast.ident;
    let new_name = Ident::new(
        &format!("{}HistoryVec", original_name.to_token_stream()),
        original_name.span(),
    );
    let mut fields = Vec::new();
    match ast.data {
        syn::Data::Struct(s) => {
            for field in s.fields.iter() {
                fields.push(field.clone());
            }
        }
        _ => panic!("#[derive(HistoryVec)] only works on structs"),
    }
    let field_names = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    let field_names_no_orphaned = field_names
        .iter()
        .filter(|f| f.to_string() != "orphaned")
        .collect::<Vec<_>>();

    let first_field = &field_names[0];

    let vec_fields = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            if *ident == "orphaned" {
                quote! {
                    pub orphaned: #ty,
                }
            } else {
                quote! {
                    pub #ident: Vec<#ty>,
                }
            }
        })
        .concat();

    let vec_new = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            if *ident == "orphaned" {
                quote! {
                    orphaned: false,
                }
            } else {
                quote! {
                    #ident: Vec::new(),
                }
            }
        })
        .concat();

    let mut generated = TokenStream2::new();
    generated.append_all(quote! {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[add_pyo3_api]
        pub struct #new_name {
            #vec_fields
        }

        impl #new_name {
            pub fn new() -> #new_name {
                #new_name {
                    #vec_new
                }
            }

            /// push fields of state to vec fields in history
            pub fn push(&mut self, value: #original_name) {
                #(self.#field_names_no_orphaned.push(value.#field_names_no_orphaned);)*
            }

            /// clear all history vecs
            pub fn clear(&mut self) {
                #(self.#field_names_no_orphaned.clear();)*
            }

            pub fn pop(&mut self) -> Option<#original_name> {
                if self.is_empty() {
                    None
                } else {
                    #(
                        let #field_names_no_orphaned = self.#field_names_no_orphaned.pop().unwrap();
                    )*
                    let orphaned = self.orphaned;
                    Some(#original_name{#(#field_names: #field_names),*})
                }
            }

            pub fn len(&self) -> usize {
                self.#first_field.len()
            }

            pub fn is_empty(&self) -> bool {
                self.#first_field.is_empty()
            }
        }

        impl Default for #new_name {
            fn default() -> #new_name {
                #new_name::new()
            }
        }
    });
    generated.into()
}
