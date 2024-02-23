//! Module that implements [super::doc_field]

use crate::imports::*;

pub fn doc_field(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_struct = syn::parse_macro_input!(item as syn::ItemStruct);

    let new_fields = if let syn::Fields::Named(FieldsNamed { named, .. }) = &mut item_struct.fields
    {
        let mut new_doc_fields: Vec<TokenStream2> = Vec::new();
        for field in named.iter_mut() {
            let mut skip_doc = false;
            remove_handled_attrs(field, &mut skip_doc);

            if !skip_doc {
                let field_name: TokenStream2 = format!("{}_doc", field.ident.to_token_stream())
                    .parse()
                    .unwrap();
                new_doc_fields.push(quote! {
                    /// String for documentation -- e.g. info about calibration/validation.
                    #[serde(skip_serializing_if = "Option::is_none")]
                    pub #field_name: Option<String>,
                });
            }
        }

        let mut new_fields = TokenStream2::new();
        new_fields.extend(quote! {
            /// Vehicle level documentation -- e.g. info about calibration/validation of vehicle
            /// and/or links to reports or other long-form documentation.
            #[serde(skip_serializing_if = "Option::is_none")]
            pub doc: Option<String>,
        });

        for orig_field in named.iter() {
            // fields from `orig_field` need to have comma added
            new_fields.extend(
                format!("{},", orig_field.to_token_stream())
                    .parse::<TokenStream2>()
                    .unwrap(),
            );
            if let Some(i) = new_doc_fields.iter().position(|x| {
                let re = Regex::new(&orig_field.ident.to_token_stream().to_string())
                    .unwrap_or_else(|err| abort!(orig_field.span(), err));
                re.is_match(&x.to_token_stream().to_string())
            }) {
                new_fields.extend(new_doc_fields[i].clone());
                new_doc_fields.remove(i);
            }
        }
        new_fields
    } else {
        abort_call_site!("Expected use on struct with named fields.")
    };

    let struct_ident = item_struct.ident.to_token_stream();
    let struct_vis = item_struct.vis;
    let struct_attrs = item_struct.attrs;

    let output: TokenStream2 = quote! {
        #(#struct_attrs)*
        #struct_vis struct #struct_ident {
            #new_fields
        }
    };

    output.into_token_stream().into()
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
    // This drops attrs matching `#[doc_field]`, removing the field attribute from the struct def.
    let new_attrs: (Vec<&syn::Attribute>, Vec<bool>) = field
        .attrs
        .iter()
        .zip(keep.iter())
        .filter(|(_a, k)| **k)
        .unzip();
    field.attrs = new_attrs.0.iter().cloned().cloned().collect();
}
