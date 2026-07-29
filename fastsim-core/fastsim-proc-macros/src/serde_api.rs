use crate::imports::*;
mod serde_utils;
use serde_utils::*;

pub(crate) fn serde_api(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut output = TokenStream2::default();

    let mut struct_ast = syn::parse_macro_input!(item as syn::ItemStruct);
    let struct_name = struct_ast.ident.clone();
    let helper_name = syn::Ident::new(
        &format!("{}DeserializeHelper", struct_name),
        struct_name.span(),
    );

    // Collect SI field data
    let mut si_fields = vec![];

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut struct_ast.fields {
        // struct with named fields
        for field in named.iter_mut() {
            // Collect SI field data before modifying
            if let Some(data) = serde_utils::collect_si_field_data(field) {
                // Only include SI fields that have unit definitions
                if !data.units.is_empty() {
                    si_fields.push(data);
                }
            }
            serde_attrs_for_si_fields(field);
        }
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut struct_ast.fields
    {
        for field in unnamed.iter_mut() {
            serde_attrs_for_si_fields(field);
        }
    }

    // Add serde(from = "Helper") if we have SI fields
    if !si_fields.is_empty() {
        let helper_name_str = helper_name.to_string();
        let from_attr: syn::Attribute = syn::parse_quote! {
            #[serde(from = #helper_name_str)]
        };
        struct_ast.attrs.push(from_attr);
    }

    output.extend(struct_ast.to_token_stream());

    // Generate helper struct and From impl
    if !si_fields.is_empty() {
        let helper_struct =
            serde_utils::generate_helper_struct(&helper_name, &struct_ast, &si_fields);
        let from_impl =
            serde_utils::generate_from_impl(&struct_name, &helper_name, &struct_ast, &si_fields);
        output.extend(helper_struct);
        output.extend(from_impl);
    }

    output.into()
}
