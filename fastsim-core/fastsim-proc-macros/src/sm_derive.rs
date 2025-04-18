use crate::imports::*;

pub(crate) fn state_methods_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let mut impl_block = TokenStream2::default();

    let fields = item_struct.fields;

    let struct_has_state = fields.iter().any(|x| *x.ident.as_ref().unwrap() == "state");
    let struct_is_state = ident.to_string().contains("State");
    let struct_has_save_interval = fields
        .iter()
        .any(|x| *x.ident.as_ref().unwrap() == "save_interval");
    let fields_with_state_vec: Vec<bool> = fields
        .iter()
        .map(|field| {
            field
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("has_state"))
        })
        .collect();

    // fields that contain nested `state` fields
    let fields_with_state = fields
        .iter()
        .zip(fields_with_state_vec)
        .filter(|(_f, hsv)| *hsv)
        .map(|(f, _hsv)| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    let all_fields = fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    let self_step: TokenStream2 = if struct_has_state {
        quote! {
            self.state.step()?;
        }
    } else if struct_is_state {
        quote! {
            self.i.update(self.i.get_stale(format_dbg!()) + 1, format_dbg!())?;
        }
    } else {
        quote! {}
    };

    impl_block.extend::<TokenStream2>(quote! {
        impl Step for #ident {
            fn step(&mut self) -> anyhow::Result<()> {
                #self_step
                #(self.#fields_with_state.step()?;)*
                Ok(())
            }
        }
    });

    let self_save_state: TokenStream2 = if struct_has_state {
        quote! {self.history.push(self.state.clone());}
    } else {
        quote! {}
    };

    if struct_is_state {
        impl_block.extend::<TokenStream2>(quote! {
            impl TrackedStateMethods for #ident {
                fn check_and_reset(&mut self, loc: String) -> anyhow::Result<()> {
                    #(self.#all_fields.ensure_fresh(format!("{loc}\n{}", stringify!(#all_fields)));)*
                    #(self.#all_fields.reset(format!("{loc}\n{}", stringify!(#all_fields)))?;)*
                    Ok(())
                }
            }
        });
    } else if struct_has_state {
        impl_block.extend::<TokenStream2>(quote! {
            impl TrackedStateMethods for #ident {
                fn check_and_reset(&mut self, loc: String) -> anyhow::Result<()> {
                    self.state.check_and_reset(format!("{loc}\n{}", format_dbg!()))?;
                    #(self.#fields_with_state.ensure_fresh(format!("{loc}\n{}", stringify!(#fields_with_state)));)*
                    #(self.#fields_with_state.reset(format!("{loc}\n{}", stringify!(#fields_with_state)))?;)*
                    Ok(())
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            impl #ident {
                fn check_and_reset(&mut self, loc: String) -> anyhow::Result<()> {
                    #(self.#fields_with_state.ensure_fresh(format!("{loc}\n{}", stringify!(#fields_with_state)));)*
                    #(self.#fields_with_state.reset(format!("{loc}\n{}", stringify!(#fields_with_state)))?;)*
                    Ok(())
                }
            }
        });
    }

    if struct_has_save_interval {
        impl_block.extend::<TokenStream2>(quote! {
            impl SaveState for #ident {
                /// Implementation for structs with `save_interval`
                fn save_state(&mut self) -> anyhow::Result<()> {
                    if let Some(interval) = self.save_interval {
                        if *self.state.i.get_fresh(format_dbg!())? % interval == (0 as usize)
                            || *self.state.i.get_fresh(format_dbg!())? == (1 as usize)
                        {
                            #self_save_state
                            #(self.#fields_with_state.save_state()?;)*
                        }
                    }
                    Ok(())
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            impl SaveState for #ident {
                /// Implementation for objects without `save_interval`
                fn save_state(&mut self) -> anyhow::Result<()> {
                    #self_save_state
                    #(self.#fields_with_state.save_state()?;)*
                    Ok(())
                }
            }
        });
    }

    impl_block.into()
}
