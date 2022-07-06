use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens}; // ToTokens is implicitly used as a trait
use syn::Meta;

mod utilities;
use utilities::{impl_getters_and_setters, FieldOptions};

/// macro for creating appropriate setters and getters for pyo3 struct attributes
#[proc_macro_attribute]
pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;

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
                .map(|x| match x.path.segments[0].ident.to_string().as_str() { // todo: check length of segments for robustness
                    "api" => {
                        let meta = x.parse_meta().unwrap();
                        if let Meta::List(list) = meta {
                            for nested in list.nested {
                                if let syn::NestedMeta::Meta(opt) = nested {
                                    // println!("opt_path{:?}", opt.path().segments[0].ident.to_string().as_str());;
                                    match opt.path().segments[0].ident.to_string().as_str() {
                                        "skip_get" => opts.skip_get = true,
                                        "skip_set" => opts.skip_set = true,
                                        "has_orphaned" => opts.field_has_orphaned = true,
                                        // todo: figure out how to use span to have rust-analyzer highlight the exact code
                                        // where this gets messed up
                                        _ => {panic!("Invalid api option. Valid options are: `skip_get`, `skip_set`, and `has_orphaned`")}
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
    });

    impl_block.extend::<TokenStream2>(quote! {
        #[classmethod]
        pub fn from_json(_cls: &PyType, json_str: &str) -> PyResult<Self> {
            Ok(serde_json::from_str(json_str).unwrap())
        }
    });

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

    let impl_block = quote! {
        #[pymethods]
        impl #ident {
            #impl_block
        }
    };

    let mut output: TokenStream2 = ast.to_token_stream();

    output.extend(impl_block);
    // if ast.ident.to_string() == "RustSimDrive" {
    //     println!("{}", output.to_string());
    // }
    output.into()
}
