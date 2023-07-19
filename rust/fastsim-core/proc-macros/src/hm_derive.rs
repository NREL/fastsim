use crate::imports::*;

pub(crate) fn history_methods_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let mut impl_block = TokenStream2::default();

    let fields = item_struct.fields;

    let struct_has_state = fields.iter().any(|x| *x.ident.as_ref().unwrap() == "state");

    let struct_has_save_interval = fields
        .iter()
        .any(|x| *x.ident.as_ref().unwrap() == "save_interval");

    let fields_with_state_vec: Vec<bool> = fields
        .iter()
        .map(|field| {
            field
                .attrs
                .iter()
                .any(|attr| attr.path.is_ident("has_state"))
        })
        .collect();

    let fields_with_state = fields
        .iter()
        .zip(fields_with_state_vec)
        .filter(|(_f, hsv)| *hsv)
        .map(|(f, _hsv)| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    if struct_has_state {
        impl_block.extend::<TokenStream2>(quote! {
            impl #ident {
                /// Increments `self.state.i`
                pub fn step(&mut self) {
                    self.state.i += 1;
                    #(self.#fields_with_state.step();)*
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            impl #ident {
                /// Increments `self.state.i`
                pub fn step(&mut self) {
                    #(self.#fields_with_state.step();)*
                }
            }
        });
    }

    let self_save_state: TokenStream2 = if struct_has_state {
        quote! {self.history.push(self.state);}
    } else {
        quote! {}
    };

    if struct_has_save_interval {
        impl_block.extend::<TokenStream2>(quote! {
            impl #ident {
                /// Saves `self.state` to `self.history` and propagates to any fields with `state`
                pub fn save_state(&mut self) {
                    if let Some(interval) = self.save_interval {
                        if self.state.i % interval == 0 || self.state.i == 1 {
                            #self_save_state
                            #(self.#fields_with_state.save_state();)*
                        }
                    }
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            impl #ident {
                /// Saves `self.state` to `self.history` and propagates to any fields with `state`
                pub fn save_state(&mut self) {
                    #self_save_state
                    #(self.#fields_with_state.save_state();)*
                }
            }
        });
    }

    impl_block.into()
}
