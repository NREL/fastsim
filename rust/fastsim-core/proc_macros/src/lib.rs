use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens, TokenStreamExt}; // ToTokens is implicitly used as a trait
use syn::{DeriveInput, Ident, Meta, spanned::Spanned};
use proc_macro_error::{abort, proc_macro_error};

mod utilities;
use utilities::{impl_getters_and_setters, FieldOptions};

/// macro for creating appropriate setters and getters for pyo3 struct attributes
#[proc_macro_error]
#[proc_macro_attribute]
pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;
    let is_state: bool = ident.to_string().contains("State");

    let mut impl_block = TokenStream2::default();
    impl_block.extend::<TokenStream2>(attr.into());

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        let field_names: Vec<String> = named
            .iter()
            .map(|f| f.ident.clone().unwrap().to_string())
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
                    &mut impl_block,
                    ident,
                    opts,
                    has_orphaned,
                    ftype,
                );
            }
        }
    } else {
        panic!("Invalid use of `add_pyo3_api` macro.  Works on structs with named fields only.")
    };

    // impl_block.extend::<TokenStream2>(quote! {
    //     #[classmethod]
    //     #[pyo3(name = "default")]
    //     pub fn default_py(_cls: &PyType) -> PyResult<Self> {
    //         Ok(Self::default())
    //     }
    // });

    impl_block.extend::<TokenStream2>(quote! {
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

        pub fn to_bincode(&self) -> PyResult<Vec<u8>> {
            Ok(bincode::serialize(&self).unwrap())
        }

        #[classmethod]
        pub fn from_bincode(_cls: &PyType, bytes_vec: Vec<u8>) -> PyResult<Self> {
           Ok(bincode::deserialize(&bytes_vec).unwrap())
        }
    });

    if !is_state {
        impl_block.extend::<TokenStream2>(quote! {
            #[pyo3(name = "to_file")]
            pub fn to_file_py(&self, filename: &str) -> PyResult<()> {
                self.to_file(filename).unwrap();
                Ok(())
            }
        });
        impl_block.extend::<TokenStream2>(quote! {
             #[classmethod]
             #[pyo3(name = "from_file")]
             pub fn from_file_py(_cls: &PyType, json_str:String) -> PyResult<Self> {
                 let obj: #ident = Self::from_file(&json_str);
                 Ok(obj)
             }
        });
    }

    let impl_block = quote! {
        #[pymethods]
        #[cfg(feature="pyo3")]
        impl #ident {
            #impl_block
        }
    };

    let mut final_output = TokenStream2::default();
    // add pyclass attribute
    final_output.extend::<TokenStream2>(quote! {
        #[cfg_attr(feature="pyo3", pyclass)]
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
