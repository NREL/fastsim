use crate::imports::*;
mod serde_utils;
use serde_utils::*;

pub(crate) fn serde_api(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // TODO: put this in the right place
    let mut impl_block = TokenStream2::default();
    let mut output = TokenStream2::default();

    let mut struct_ast = syn::parse_macro_input!(item as syn::ItemStruct);
    let ident = &struct_ast.ident;

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut struct_ast.fields {
        // struct with named fields
        for field in named.iter_mut() {
            serde_attrs_for_si_fields(field);
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut struct_ast.fields
    {
        process_tuple_struct(unnamed, &mut impl_block, ident);
    } else {
        abort_call_site!(
            "Invalid use of `fastsim_api` macro.  Expected tuple struct or C-style struct."
        );
    };

    output.extend(struct_ast.to_token_stream());
    output.extend::<TokenStream2>(impl_block);
    output.into()
}

fn process_tuple_struct(
    unnamed: &mut syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    impl_block: &mut TokenStream2,
    ident: &Ident,
) {
    // tuple struct
    assert!(unnamed.len() == 1);
    let re = Regex::new(r"Vec < (.+) >").unwrap();
    for field in unnamed.iter() {
        let ftype = field.ty.clone();
        if let syn::Type::Path(type_path) = ftype.clone() {
            let type_str = type_path.clone().into_token_stream().to_string();
            if type_str.contains("Vec") {
                // println!("{}", type_str);
                // println!("{}", &re.captures(&type_str).unwrap()[1]);
                let contained_dtype: TokenStream2 = re.captures(&type_str).unwrap()[1]
                    .to_string()
                    .parse()
                    .unwrap();
                impl_block.extend::<TokenStream2>(quote! {
                    impl #ident{
                        /// Implement the non-Python `new` method.
                        pub fn new(value: Vec<#contained_dtype>) -> Self {
                            Self(value)
                        }
                    }
                });
            }
        }
    }
}
