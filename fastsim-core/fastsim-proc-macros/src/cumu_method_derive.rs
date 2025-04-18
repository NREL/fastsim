use crate::imports::*;

pub(crate) fn cumu_method_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let fields = if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = item_struct.fields {
        named
    } else {
        abort_call_site!("SetCumulative works only on Named Field structs.")
    };
    let re = Regex::new(r"energy_(\w+)").unwrap();

    let (pwr_fields, energy_fields): (Vec<TokenStream2>, Vec<TokenStream2>) = fields
        .iter()
        .filter_map(|x| {
            let field_str = &x.ident.as_ref().unwrap().to_string();
            if re.is_match(field_str) {
                let key = re.captures(field_str).unwrap()[1].to_string();
                if fields
                    .iter()
                    .any(|x| *x.ident.as_ref().unwrap() == format!("pwr_{}", key))
                {
                    Some((
                        format!("pwr_{}", key).parse().unwrap(),
                        field_str.clone().parse().unwrap(),
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unzip();

    let impl_block: TokenStream2 = quote! {
        // this tells the compiler that the `SetCumulative` trait is not manually derived
        #[automatically_derived]
        impl SetCumulative for #ident {
            fn set_cumulative(&mut self, dt: si::Time) -> anyhow::Result<()> {
                #(self
                    .#energy_fields
                    .increment(
                        *self.#pwr_fields.get_fresh(|| format_dbg!())? * dt,
                        || format_dbg!()
                    )?;
                )*
                Ok(())
            }
        }
    };

    impl_block.into()
}
