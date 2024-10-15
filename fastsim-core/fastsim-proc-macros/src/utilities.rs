use crate::imports::*;

// taken from https://github.com/lumol-org/soa-derive/blob/master/soa-derive-internal/src/input.rs
pub(crate) trait TokenStreamIterator {
    fn concat_by(
        self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream;
    fn concat(self) -> proc_macro2::TokenStream;
}

impl<T: Iterator<Item = proc_macro2::TokenStream>> TokenStreamIterator for T {
    fn concat_by(
        mut self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self.next() {
            Some(first) => self.fold(first, f),
            None => quote! {},
        }
    }

    fn concat(self) -> proc_macro2::TokenStream {
        self.concat_by(|a, b| quote! { #a #b })
    }
}

const ONLY_FN_MSG: &str = "Only function definitions allowed here.";

/// accepts `attr` TokenStream from attribute-like proc macro and returns
/// TokenStream2 of fn defs that are in `expected_fn_names` and/or not in `forbidden_fn_names`.  
/// If `expected_exlusive` is true, only values in `expected_fn_names` are allowed.  
/// Raises locationally useful errors if mistakes are made in formatting or whatnot.  
pub fn parse_ts_as_fn_defs(
    attr: TokenStream,
    mut expected_fn_names: Vec<String>,
    expected_exclusive: bool,
    forbidden_fn_names: Vec<String>,
) -> TokenStream2 {
    let attr = TokenStream2::from(attr);
    let impl_block = quote! {
        impl Dummy { // this name doesn't really matter as it won't get used
            #attr
        }
    }
    .into();
    // let item_impl = syn::parse_macro_input!(impl_block as syn::ItemImpl);
    let item_impl = syn::parse::<syn::ItemImpl>(impl_block)
        .expect("failed to parse `item_impl` in `parse_ts_as_fn_defs`");

    let mut fn_from_attr = TokenStream2::new();

    for impl_item in item_impl.items {
        match &impl_item {
            syn::ImplItem::Fn(item_fn) => {
                let sig_str = &item_fn.sig.ident.to_token_stream().to_string();
                fn_from_attr.extend(item_fn.clone().to_token_stream());
                // check signature
                if expected_exclusive
                    && (forbidden_fn_names.contains(sig_str)
                        || !expected_fn_names.contains(sig_str))
                {
                    emit_error!(
                        &item_fn.sig.ident.span(),
                        format!("Function name `{}` is forbidden", sig_str)
                    )
                }

                let index = expected_fn_names.iter().position(|x| x == sig_str);

                if let Some(i) = index {
                    expected_fn_names.remove(i);
                }
                // remove the matching name from the vec to avoid checking again
                // at the end of iteration, this vec should be empty
            }
            _ => abort_call_site!(ONLY_FN_MSG),
        }
    }

    if !expected_fn_names.is_empty() {
        abort_call_site!(format!(
            "Expected fn def for `{}`",
            expected_fn_names.join(", ")
        ));
    }
    fn_from_attr
}
