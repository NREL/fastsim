//! Module that implements [super::doc_field]

use crate::imports::*;

pub fn doc_field(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);

    if let syn::Fields::Named(FieldsNamed { named, .. }) = &mut ast.fields {
        let mut new_doc_fields = TokenStream2::new();
        for field in named.iter_mut() {
            let mut skip_doc = false;
            remove_handled_attrs(field, &mut skip_doc);

            if !skip_doc {
                let field_name = format!("{}_doc", field.ident.to_token_stream());
                new_doc_fields.extend::<TokenStream2>(quote! {
                    ///  documentation -- e.g. info about calibration/validation.
                    pub #field_name: Option<String>,
                });
            }
        }

        new_doc_fields.extend::<TokenStream2>(quote! {
            /// Vehicle level documentation -- e.g. info about calibration/validation of vehicle
            /// and/or links to reports or other long-form documentation.
            pub doc: Option<String>,
        });
        dbg!(&new_doc_fields);
        let new_doc_fields: TokenStream2 = quote! {
            /// Dummy struct that will not be used anywhere
            pub struct Dummy {
                #new_doc_fields
            }
        };
        dbg!(&new_doc_fields);

        let new_doc_fields: FieldsNamed = syn::parse2(new_doc_fields)
            .map_err(|e| format!("[{}:{}] `parse2` failed. {}", file!(), line!(), e))
            .unwrap();

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
                named.push(new_doc_fields.named[i].clone());
                // might be good to also remove field `i` from `new_doc_fields`
            }
        }
    } else {
        abort_call_site!("Expected use on struct with named fields.")
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
