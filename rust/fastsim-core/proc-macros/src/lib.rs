mod imports;
use crate::imports::*;
mod add_pyo3_api;
mod approx_eq_derive;
mod history_vec_derive;
mod utilities;

/// macro for creating appropriate setters and getters for pyo3 struct attributes
#[proc_macro_error]
#[proc_macro_attribute]
pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    add_pyo3_api::add_pyo3_api(attr, item)
}

#[proc_macro_derive(HistoryVec)]
pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    history_vec_derive::history_vec_derive(input)
}

#[proc_macro_derive(ApproxEq)]
pub fn approx_eq_derive(input: TokenStream) -> TokenStream {
    approx_eq_derive::approx_eq_derive(input)
}
