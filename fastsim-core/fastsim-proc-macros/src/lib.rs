mod imports;
use imports::*;
mod cumu_method_derive;
mod cycle_derive;
mod history_vec_derive;
mod pyo3_api;
mod serde_api;
mod sm_derive;
mod timer;
mod utilities;

#[proc_macro_error]
#[proc_macro_attribute]
/// macro for creating appropriate setters and getters for pyo3 struct attributes
/// and other, non-python API functionality
pub fn serde_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    serde_api::serde_api(attr, item)
}

#[proc_macro_error]
#[proc_macro_attribute]
/// macro for creating appropriate setters and getters for pyo3 struct attributes
/// and other, non-python API functionality
pub fn pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    pyo3_api::pyo3_api(attr, item)
}

#[proc_macro_error]
#[proc_macro_attribute]
/// macro for creating timing harness
pub fn timer(attr: TokenStream, item: TokenStream) -> TokenStream {
    timer::timer(attr, item)
}

#[proc_macro_derive(HistoryVec, attributes(api))]
/// generate HistoryVec that acts like a vec of states but
/// stores each field of state as a vec field.
pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    history_vec_derive::history_vec_derive(input)
}

#[proc_macro_derive(StateMethods, attributes(has_state))]
/// Generate `step` and `save_state` methods that work for struct and any
/// nested fields with the `#[has_state]` attribute.
pub fn state_methods_derive(input: TokenStream) -> TokenStream {
    sm_derive::state_methods_derive(input)
}

#[proc_macro_derive(SetCumulative, attributes(has_state))]
/// Generate `step` and `save_state` methods that work for struct and any
/// nested fields with the `#[has_state]` attribute.
pub fn cumu_method_derive(input: TokenStream) -> TokenStream {
    cumu_method_derive::cumu_method_derive(input)
}
