use crate::imports::*;
mod serde_utils;
use serde_utils::*;

pub(crate) fn serde_api(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // TODO: put this in the right place
    let impl_block = TokenStream2::default();
    let mut output = TokenStream2::default();

    let mut struct_ast = syn::parse_macro_input!(item as syn::ItemStruct);

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut struct_ast.fields {
        // struct with named fields
        for field in named.iter_mut() {
            serde_attrs_for_si_fields(field);
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut struct_ast.fields
    {
        for field in unnamed.iter_mut() {
            serde_attrs_for_si_fields(field);
        }
    } else {
        abort_call_site!(
            "Invalid use of `fastsim_api` macro.  Expected tuple struct or C-style struct."
        );
    };

    output.extend(struct_ast.to_token_stream());
    output.extend::<TokenStream2>(impl_block);
    output.into()
}
