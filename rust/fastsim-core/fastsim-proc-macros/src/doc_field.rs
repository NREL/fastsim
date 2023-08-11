//! Module that implements [super::doc_field]

use crate::imports::*;

pub fn doc_field(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);

    let new_doc_fields: FieldsNamed = if let syn::Fields::Named(FieldsNamed { named, .. }) =
        &mut ast.fields
    {
        let mut new_doc_fields = String::from("{\n");

        for field in named.iter_mut() {
            let mut skip_doc = false;
            remove_handled_attrs(field, &mut skip_doc);

            if !skip_doc {
                // create new doc field as string
                new_doc_fields.push_str(&format!(
                    "
                    /// {} documentation -- e.g. info about calibration/validation.
                    pub {}_doc: Option<String>,\n",
                    field.ident.as_ref().unwrap(),
                    field.ident.as_ref().unwrap()
                ));
            }
        }

        new_doc_fields.push_str(
            "
            /// Vehicle level documentation -- e.g. info about calibration/validation of vehicle
            /// and/or links to reports or other long-form documentation.
            pub doc: Option<String>\n}",
        );
        syn::parse_str::<FieldsNamed>(&new_doc_fields).unwrap_or_else(|e| abort_call_site!("{}", e))
    } else {
        abort_call_site!("Expected use on struct with named fields.")
    };

    if let syn::Fields::Named(FieldsNamed { named, .. }) = &mut ast.fields {
        // named.extend(new_doc_fields.named);

        let old_named = named.clone();
        named.clear();
        for old in old_named.into_iter() {
            named.push(old.clone());
            if let Some(i) = new_doc_fields.named.iter().position(|x| {
                let re = Regex::new(&old.ident.to_token_stream().to_string())
                    .unwrap_or_else(|e| abort!(old.span(), e));
                re.is_match(&x.ident.to_token_stream().to_string())
            }) {
                dbg!(i);
                // named.extend::<Punctuated<Field, Comma>>(
                //     syn::parse_str::<FieldsNamed>(&format!(
                //         "{}_doc",
                //         new_doc_fields.named[i].clone().to_token_stream()
                //     ))
                //     .unwrap()
                //     .named,
                // );
                // might be good to also remove field `i` from `new_doc_fields`
                // dbg!(named.iter().last().to_token_stream().to_string());
            }
        }
    } else {
        abort_call_site!("`doc_field` proc macro works only on named structs.");
    };

    ast.into_token_stream().into()
}

/// Remove field attributes, i.e. attributes that are not handled by the [doc_field] macro
fn remove_handled_attrs(field: &mut syn::Field, skip_doc: &mut bool) {
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
    // println!("options {:?}", opts);
    // this drops attrs matching `#[doc_field]`, removing the field attribute from the struct def
    let new_attrs: (Vec<&syn::Attribute>, Vec<bool>) = field
        .attrs
        .iter()
        .zip(keep.iter())
        .filter(|(_a, k)| **k)
        .unzip();
    field.attrs = new_attrs.0.iter().cloned().cloned().collect();
}
