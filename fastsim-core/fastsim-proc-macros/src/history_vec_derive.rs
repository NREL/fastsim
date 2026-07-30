use crate::imports::*;
use crate::utilities::TokenStreamIterator;

/// Rewrite a `#[serde(..., serialize_with = "mod::fn_name", ...)]` attribute so
/// the serialize_with path uses the `vec_` variant of the helper function.
/// All other serde keys are forwarded unchanged.
///
/// Example: `serialize_with = "fastsim_core::utils::serde_helpers::power_as_kilowatts"`
///       →  `serialize_with = "fastsim_core::utils::serde_helpers::vec_power_as_kilowatts"`
fn rewrite_serialize_with_for_vec(attr: &syn::Attribute) -> TokenStream2 {
    let mut parts: Vec<TokenStream2> = vec![];
    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("serialize_with") {
            let _: syn::Token![=] = meta.input.parse()?;
            let lit: syn::LitStr = meta.input.parse()?;
            let base = lit.value();
            let vec_path = if let Some(sep) = base.rfind("::") {
                let (head, tail_with_sep) = base.split_at(sep);
                let tail = &tail_with_sep[2..];
                if tail.starts_with("vec_") {
                    base.clone()
                } else {
                    format!("{head}::vec_{tail}")
                }
            } else if base.starts_with("vec_") {
                base.clone()
            } else {
                format!("vec_{base}")
            };
            let vec_lit = syn::LitStr::new(&vec_path, lit.span());
            parts.push(quote! { serialize_with = #vec_lit });
        } else {
            let path = &meta.path;
            if meta.input.peek(syn::Token![=]) {
                let _: syn::Token![=] = meta.input.parse()?;
                let val: proc_macro2::TokenTree = meta.input.parse()?;
                parts.push(quote! { #path = #val });
            } else {
                parts.push(quote! { #path });
            }
        }
        Ok(())
    });
    quote! { #[serde(#(#parts),*)] }
}

pub(crate) fn history_vec_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let original_name = &item_struct.ident;
    let original_name_str: String = original_name.to_string();

    // `#[api(no_pyo3)]` on the state struct opts out of pyclass / pyo3_api / Init /
    // SerdeAPI generation.  Use this when deriving HistoryVec outside fastsim-core
    // (e.g. integration tests) where those items are not in scope.
    let no_pyo3 = item_struct.attrs.iter().any(|attr| {
        attr.path().is_ident("history_vec")
            && attr.to_token_stream().to_string().contains("no_pyo3")
    });
    let new_name = Ident::new(
        &format!("{}HistoryVec", original_name.to_token_stream()),
        original_name.span(),
    );
    let new_name_str: String = new_name.to_string();
    let fields = item_struct.fields;
    let field_names = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();
    let first_field = &field_names[0];
    let vec_fields = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            // Copy serde attributes; for any that contain `serialize_with`, rewrite
            // the path to its `vec_` variant so history fields serialize with the
            // same unit as the main struct (e.g. `power_as_kilowatts` →
            // `vec_power_as_kilowatts`).
            let attrs = f
                .attrs
                .iter()
                .filter_map(|a| {
                    if !a.path().is_ident("serde") {
                        return Some(quote! { #a });
                    }
                    if !a.to_token_stream().to_string().contains("serialize_with") {
                        return Some(quote! { #a });
                    }
                    Some(rewrite_serialize_with_for_vec(a))
                })
                .collect::<Vec<_>>();
            quote! {
                #(#attrs)*
                pub #ident: Vec<#ty>,
            }
        })
        .concat();
    let vec_new = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            quote! {
                #ident: Vec::new(),
            }
        })
        .concat();
    let mut generated = TokenStream2::new();
    let struct_doc: TokenStream2 = format!("/// Stores history of {original_name_str}")
        .parse()
        .unwrap();
    let push_doc: TokenStream2 =
        format!("/// Pushes fields of {original_name_str} to {new_name_str}")
            .parse()
            .unwrap();
    let pop_doc: TokenStream2 =
        format!("/// Remove and return last element as {original_name_str}")
            .parse()
            .unwrap();
    let state_vec_doc: TokenStream2 = format!("/// Return history as vec of {original_name_str}")
        .parse()
        .unwrap();

    // Conditionally emit pyo3/Init/SerdeAPI code at macro-expansion time.
    // When no_pyo3 is true these are entirely absent from the generated tokens,
    // so the HistoryVec can be used outside fastsim-core without those items in scope.
    let pyclass_attr = if no_pyo3 {
        quote! {}
    } else {
        quote! { #[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))] }
    };
    let pyo3_impls = if no_pyo3 {
        quote! {}
    } else {
        quote! {
            #[cfg(feature = "pyo3")]
            #[pyo3_api]
            impl #new_name {
                #[pyo3(name = "len")]
                fn len_py(&self) -> usize { self.len() }
                fn __len__(&self) -> usize { self.len() }
            }

            #[cfg(feature = "pyo3")]
            impl Init for #new_name {}
            #[cfg(feature = "pyo3")]
            impl SerdeAPI for #new_name {}
        }
    };

    generated.append_all(quote! {
        #[serde_api]
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[serde(default)]
        #pyclass_attr
        #struct_doc
        pub struct #new_name {
            #vec_fields
        }

        #pyo3_impls

        impl #new_name {
            /// Creates new emtpy vec container
            pub fn new() -> #new_name {
                #new_name {
                    #vec_new
                }
            }

            #push_doc
            pub fn push(&mut self, state: #original_name) {
                #(self.#field_names.push(state.#field_names.clone());)*
            }

            /// clear all history vecs
            pub fn clear(&mut self) {
                #(self.#field_names.clear();)*
            }

            #pop_doc
            pub fn pop(&mut self) -> Option<#original_name> {
                if self.is_empty() {
                    None
                } else {
                    #(
                        let #field_names = self.#field_names.pop().unwrap();
                    )*
                    Some(#original_name{#(#field_names: #field_names.clone()),*})
                }
            }

            /// Returns len of contained vectors
            pub fn len(&self) -> usize {
                self.#first_field.len()
            }

            /// Returns True if contained vecs are empty
            pub fn is_empty(&self) -> bool {
                self.#first_field.is_empty()
            }

            #state_vec_doc
            pub fn state_vec(&self) -> Vec<#original_name> {
                let mut state_vec: Vec<#original_name> = Vec::new();
                for i in 0..self.len() {
                    state_vec.push(
                        #original_name{
                            #(#field_names: self.#field_names[i].clone(),)*
                        }
                    )
                }
                state_vec
            }

            // TODO: flesh this out
            // /// Returns fieldnames of any fields that are constant throughout history
            // pub fn names_of_static_fields(&self) -> Vec<String> {

            // }
        }

        impl Default for #new_name {
            fn default() -> #new_name {
                #new_name::new()
            }
        }
    });
    generated.into()
}
