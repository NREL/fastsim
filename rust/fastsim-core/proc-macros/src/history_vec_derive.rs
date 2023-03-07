use crate::imports::*;
use crate::utilities::*;

pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    let ast: DeriveInput = syn::parse(input).unwrap();
    let original_name = &ast.ident;
    let new_name = Ident::new(
        &format!("{}HistoryVec", original_name.to_token_stream()),
        original_name.span(),
    );
    let mut fields = Vec::new();
    match ast.data {
        syn::Data::Struct(s) => {
            for field in s.fields.iter() {
                fields.push(field.clone());
            }
        }
        _ => panic!("#[derive(HistoryVec)] only works on structs"),
    }
    let field_names = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    let field_names_no_orphaned = field_names
        .iter()
        .filter(|f| f.to_string() != "orphaned")
        .collect::<Vec<_>>();

    let first_field = &field_names[0];

    let vec_fields = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            if *ident == "orphaned" {
                quote! {
                    pub orphaned: #ty,
                }
            } else {
                quote! {
                    pub #ident: Vec<#ty>,
                }
            }
        })
        .concat();

    let vec_new = fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            if *ident == "orphaned" {
                quote! {
                    orphaned: false,
                }
            } else {
                quote! {
                    #ident: Vec::new(),
                }
            }
        })
        .concat();

    let mut generated = TokenStream2::new();
    generated.append_all(quote! {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
        #[add_pyo3_api]
        pub struct #new_name {
            #vec_fields
        }

        impl #new_name {
            pub fn new() -> #new_name {
                #new_name {
                    #vec_new
                }
            }

            /// push fields of state to vec fields in history
            pub fn push(&mut self, value: #original_name) {
                #(self.#field_names_no_orphaned.push(value.#field_names_no_orphaned);)*
            }

            /// clear all history vecs
            pub fn clear(&mut self) {
                #(self.#field_names_no_orphaned.clear();)*
            }

            pub fn pop(&mut self) -> Option<#original_name> {
                if self.is_empty() {
                    None
                } else {
                    #(
                        let #field_names_no_orphaned = self.#field_names_no_orphaned.pop().unwrap();
                    )*
                    let orphaned = self.orphaned;
                    Some(#original_name{#(#field_names: #field_names),*})
                }
            }

            pub fn len(&self) -> usize {
                self.#first_field.len()
            }

            pub fn is_empty(&self) -> bool {
                self.#first_field.is_empty()
            }
        }

        impl Default for #new_name {
            fn default() -> #new_name {
                #new_name::new()
            }
        }
    });
    generated.into()
}
