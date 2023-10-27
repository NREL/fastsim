use crate::imports::*;

pub(crate) fn energy_method_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let fields = item_struct.fields;
    let re = Regex::new(r"energy_(\w*)").unwrap();

    let (pwr_fields, energy_fields): (Vec<String>, Vec<String>) = fields
        .iter()
        .filter_map(|x| {
            let field_str = &x.ident.as_ref().unwrap().to_string();
            if re.is_match(field_str) {
                let key = re.captures(field_str).unwrap()[0].to_string();
                if fields
                    .iter()
                    .any(|x| *x.ident.as_ref().unwrap() == format!("pwr_{}", key))
                {
                    Some((format!("pwr_{}", key), field_str.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unzip();

    let impl_block: TokenStream2 = quote! {
        impl #ident {
            /// For all `energy_*` fields that have matching `pwr_*` fields, set cumulative energy based on power and `dt`
            /// # Arugments
            /// - `dt`: time step size
            pub fn set_energy_fields(&mut self, dt: si::Time) -> anyhow::Result<()> {
                #(self.#energy_fields = self.#pwr_fields * dt)*;
                Ok(())
            }
        }
    };

    impl_block.into()
}
