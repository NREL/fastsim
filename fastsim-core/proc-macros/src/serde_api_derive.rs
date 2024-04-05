use crate::imports::*;

pub(crate) fn serde_api_derive(input: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(input as syn::Item);
    let ident = match &item {
        syn::Item::Struct(item_struct) => &item_struct.ident,
        syn::Item::Enum(item_enum) => &item_enum.ident,
        _ => abort_call_site!("Unsupported item type. Only structs and enums are supported."),
    };

    let mut impl_block = TokenStream2::default();

    impl_block.extend::<TokenStream2>(quote! {
        impl SerdeAPI for #ident { }
    });

    impl_block.into()
}
