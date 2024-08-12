//! Module containing proc macro function for timing stuff

use crate::imports::*;

pub fn timer(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut timed = TokenStream2::new();
    let ast = syn::parse_macro_input!(item as syn::ItemFn);
    let body = &ast.block;

    timed.extend::<TokenStream2>(quote! {
        #[cfg(feature = "logging")]
        let now = std::time::Instant::now();
        #body
        #[cfg(feature = "logging")]
        let elapsed_time = now.elapsed();
        #[cfg(feature = "logging")]
        log::debug!("{}", elapsed_time);
    });

    timed.into()
}
