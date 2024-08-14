use crate::imports::*;
use crate::utilities::TokenStreamIterator;

#[allow(unused)]
pub(crate) fn cycle_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let original_name = &item_struct.ident;
    let original_name_str: String = original_name.to_string();
    let new_name = Ident::new(
        &format!("{}", original_name.to_token_stream()).replace("Element", ""),
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
            quote! {
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
    let struct_doc: TokenStream2 = format!("/// Vector version of {original_name_str}")
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
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[fastsim_api(
            #[pyo3(name = "len")]
            fn len_py(&self) -> usize {
                self.len()
            }

            fn __len__(&self) -> usize {
                self.len()
            }
        )]
        #struct_doc
        pub struct #new_name {
            #vec_fields
        }

        impl #new_name {
            /// Creates new emtpy vec container
            pub fn new() -> #new_name {
                #new_name {
                    #vec_new
                }
            }

            #push_doc
            pub fn push(&mut self, value: #original_name) {
                #(self.#field_names.push(value.#field_names);)*
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
                    Some(#original_name{#(#field_names: #field_names),*})
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
                            #(#field_names: self.#field_names[i],)*
                        }
                    )
                }
                state_vec
            }
        }

        impl Default for #new_name {
            fn default() -> #new_name {
                #new_name::new()
            }
        }

        impl SerdeAPI for #original_name { }
    });
    generated.into()
}
