use crate::imports::*;

lazy_static! {
    static ref ENERGY_REGEX: Regex = Regex::new(r"energy_(\w+)").unwrap();
}

pub(crate) fn state_methods_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let mut impl_block = TokenStream2::default();

    let fields = if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = item_struct.fields {
        named
    } else {
        abort_call_site!("`StateMethods` works only on Named Field structs.")
    };

    let struct_has_state = fields.iter().any(|x| *x.ident.as_ref().unwrap() == "state");
    let ident_str = ident.to_string();
    let struct_is_state = ident_str.contains("State");
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
            self.state.step(|| format!("{}\n{}", loc(), #ident_str))?;
        }
    } else if struct_is_state {
        quote! {
            self.i.increment(1, || format_dbg!())?;
        }
    } else {
        quote! {}
    };

    impl_block.extend::<TokenStream2>(quote! {
        impl Step for #ident {
            fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                #self_step
                #(self.#fields_with_state.step(|| format!("{}\n{}", loc(), stringify!(#fields_with_state)))?;)*
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
            #[automatically_derived]
            impl CheckAndResetState for #ident {
                fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                    #(
                        self.#all_fields.check_and_reset(|| format!("{}\n    `{}` has not been updated", loc(), stringify!(#all_fields)))?;
                    )*
                    Ok(())
                }
            }
        });
    } else if struct_has_state {
        impl_block.extend::<TokenStream2>(quote! {
            #[automatically_derived]
            impl CheckAndResetState for #ident {
                fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                    self.state.check_and_reset(|| format!("{}", loc()))?;
                    #(
                        self.#fields_with_state.check_and_reset(|| format!("{}\n    `{}`", loc(), stringify!(#fields_with_state)))?;
                    )*
                    Ok(())
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            #[automatically_derived]
            impl CheckAndResetState for #ident {
                fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                    #(
                        self.#fields_with_state.check_and_reset(|| format!("{}\n    `{}`", loc(), stringify!(#fields_with_state)))?;
                    )*
                    Ok(())
                }
            }
        });
    }

    impl_block.extend::<TokenStream2>(quote! {
        impl StateMethods for #ident {}
    });

    if struct_has_save_interval {
        impl_block.extend::<TokenStream2>(quote! {
            #[automatically_derived]
            impl SaveState for #ident {
                /// Implementation for structs with `save_interval`
                fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                    if let Some(interval) = self.save_interval {
                        if *self.state.i.get_fresh(|| format_dbg!())? % interval == (0 as usize)
                            || *self.state.i.get_fresh(|| format_dbg!())? == (1 as usize)
                        {
                            #self_save_state
                            #(self.#fields_with_state.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?;)*
                        }
                    }
                    Ok(())
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            #[automatically_derived]
            impl SaveState for #ident {
                /// Implementation for objects without `save_interval`
                fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
                    #self_save_state
                    #(self.#fields_with_state.save_state(|| format!("{}\n{}", loc(), format_dbg!()))?;)*
                    Ok(())
                }
            }
        });
    }

    impl_block.into()
}
