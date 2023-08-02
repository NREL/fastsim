use crate::imports::*;

///
pub fn doc_field(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        let mut new_doc_fields = String::from("{\n");

        for field in named.iter_mut() {
            let mut skip_doc = false;
            let keep = get_attrs_to_keep(field, &mut skip_doc);
            // println!("options {:?}", opts);
            let mut iter = keep.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());
            if !skip_doc {
                // create new doc field as string
                new_doc_fields.push_str(&format!(
                    "/// {} documentation -- e.g. info about calibration/validation of vehicle,
                    /// links to reports or other long-form documentation.
                    pub {}_doc: Option<String>,\n",
                    field.ident.as_ref().unwrap(),
                    field.ident.as_ref().unwrap()
                ));
            }
        }

        new_doc_fields.push_str(
            "/// Vehicle level documentation -- e.g. info about calibration/validation of this parameter,
            /// links to reports or other long-form documentation.
        doc: Option<String>\n}",
        );
        let new_doc_fields: syn::FieldsNamed = syn::parse_str::<syn::FieldsNamed>(&new_doc_fields)
            .unwrap_or_else(|e| abort_call_site!("{}", e));

        named.extend(new_doc_fields.named);
    } else {
        abort_call_site!("`doc_field` proc macro works only on named structs.");
    };

    ast.into_token_stream().into()
}

/// Returns attributes to retain, i.e. attributes that are not handled by the [doc_field] macro
fn get_attrs_to_keep(field: &mut syn::Field, skip_doc: &mut bool) -> Vec<bool> {
    let keep: Vec<bool> = field
        .attrs
        .iter()
        .map(|x| match x.path.segments[0].ident.to_string().as_str() {
            "doc_field" => {
                let meta = x.parse_meta().unwrap();
                if let Meta::List(list) = meta {
                    for nested in list.nested {
                        if let syn::NestedMeta::Meta(opt) = nested {
                            // println!("opt_path{:?}", opt.path().segments[0].ident.to_string().as_str());;
                            let opt_name = opt.path().segments[0].ident.to_string();
                            match opt_name.as_str() {
                                "skip_doc" => *skip_doc = true,
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
    keep
}
