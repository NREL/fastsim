use crate::imports::*;

pub fn doc_field(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    // println!("{}", ast.ident.to_string());

    // add field `doc: Option<String>`

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        let field_names: Vec<String> = named
            .iter()
            .map(|f| f.ident.as_ref().unwrap().to_string())
            .collect();
        let has_orphaned: bool = field_names.iter().any(|f| f == "orphaned");

        let mut new_doc_fields: Vec<syn::Field> = Vec::new();

        for field in named.iter_mut() {
            let mut skip_doc = false;
            let keep = get_attrs_to_keep(field, &mut skip_doc);
            // println!("options {:?}", opts);
            let mut iter = keep.iter();
            // this drops attrs with api, removing the field attribute from the struct def
            field.attrs.retain(|_| *iter.next().unwrap());
            if !skip_doc {
                // create new doc field as string
                let new_doc_field = format!("{}_doc", field.ident.as_ref().unwrap().to_string());
                // convert it to named field
                let mut new_doc_field = syn::parse_str::<syn::FieldsNamed>(&new_doc_field)
                    .or_else(|e| abort_call_site!(e))
                    .unwrap();
                // give it a type
                for named in new_doc_field.named.iter_mut() {
                    named.ty = syn::parse_str::<syn::Type>("Option<String>")
                        .or_else(|_| abort_call_site!())
                        .unwrap();
                    // TODO: figure out how to add a doc string here
                }
                new_doc_fields.push(
                    new_doc_field
                        .named
                        .first()
                        .or_else(|| abort_call_site!())
                        .unwrap()
                        .clone(),
                );
            }
        }

        named.extend(new_doc_fields);
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
            "doc" => {
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
