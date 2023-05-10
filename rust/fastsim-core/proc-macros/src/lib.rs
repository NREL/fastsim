mod imports;
use crate::imports::*;
mod api;
mod approx_eq_derive;
mod history_vec_derive;
mod legacy_api;
mod utilities;

/// legacy macro for creating appropriate setters and getters for pyo3 struct attributes
#[proc_macro_error]
#[proc_macro_attribute]
pub fn legacy_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    legacy_api::legacy_api(attr, item)
}

#[proc_macro_error]
#[proc_macro_attribute]
/// macro for creating appropriate setters and getters for pyo3 struct attributes
/// and other, non-python API functionality
pub fn api(attr: TokenStream, item: TokenStream) -> TokenStream {
    api::api(attr, item)
}

#[proc_macro_derive(HistoryVec)]
pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    history_vec_derive::history_vec_derive(input)
}

#[proc_macro_derive(ApproxEq)]
pub fn approx_eq_derive(input: TokenStream) -> TokenStream {
    approx_eq_derive::approx_eq_derive(input)
}
