use crate::imports::*;
use crate::utilities::*;

pub fn doc_field(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());
    let ident = &ast.ident;

    // add field `doc: Option<String>`

    let is_state_or_history: bool =
        ident.to_string().contains("State") || ident.to_string().contains("HistoryVec");

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        let field_names: Vec<String> = named
            .iter()
            .map(|f| f.ident.as_ref().unwrap().to_string())
            .collect();
        let has_orphaned: bool = field_names.iter().any(|f| f == "orphaned");

        for field in named.iter_mut() {
            let ident = field.ident.as_ref().unwrap();

            let mut skip_doc = false;
            let fields_to_doc: Vec<bool> = field
                .attrs
                .iter()
                .map(|x| match x.path.segments[0].ident.to_string().as_str() {
                    "doc" => {
                        let meta = x.parse_meta().unwrap();
                        if let Meta::List(list) = meta {
                            for nested in list.nested {
                                if let syn::NestedMeta::Meta(opt) = nested {
                                    // println!("opt_path{:?}", opt.path().segments[0].ident.to_string().as_str());;
                                    let opt_name = opt.path().segments[0].ident.to_string();
                                    match opt_name.as_str() {
                                        "skip_doc" => skip_doc = true,
                                        _ => {
                                            abort!(
                                                x.span(),
                                                format!(
                                                    "Invalid doc option: {}.\nValid option is: `skip_doc`.",
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
            // println!("options {:?}", skip_doc);
            let mut iter = fields_to_doc.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());
        }

        if !is_state_or_history {
            todo!()
        }
    } else {
        abort_call_site!("`doc_field` proc macro works only on named structs.");
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
