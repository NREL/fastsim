use crate::imports::*;
use crate::utilities::TokenStreamIterator;

pub(crate) fn history_vec_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let original_name = &item_struct.ident;
    let original_name_str: String = original_name.to_string();
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
            let attrs = &f.attrs.iter().collect::<Vec<&syn::Attribute>>();
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
    generated.append_all(quote! {
        #[serde_api]
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
        #struct_doc
        pub struct #new_name {
            #vec_fields
        }

        #[pyo3_api]
        impl #new_name {
            #[pyo3(name = "len")]
            fn len_py(&self) -> usize {
                self.len()
            }

            fn __len__(&self) -> usize {
                self.len()
            }
        }

        impl Init for #new_name { }
        impl SerdeAPI for #new_name { }

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
