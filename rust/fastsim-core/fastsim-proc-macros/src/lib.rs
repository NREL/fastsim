//! Crate that provides procedural macros for [fastsim-core](https://crates.io/crates/fastsim-core)

// modules
mod imports;
// modules - macros
mod add_pyo3_api;
mod approx_eq_derive;
mod doc_field;
mod history_vec_derive;

// modules - other
mod utilities;

// imports
use crate::imports::*;

/// Adds pyo3 getters and setters for all fields, unless skip attribute is present.  
#[proc_macro_error]
#[proc_macro_attribute]
pub fn add_pyo3_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    add_pyo3_api::add_pyo3_api(attr, item)
}

/// Adds an equivelent `*_doc: Option<String>` field for each field
#[proc_macro_error]
#[proc_macro_attribute]
pub fn doc_field(attr: TokenStream, item: TokenStream) -> TokenStream {
    doc_field::doc_field(attr, item)
}

/// Derive macro that creates `*HistoryVec` struct from `*State` struct,
/// with a corresponding Vec for each field in `*State`.
#[proc_macro_derive(HistoryVec)]
pub fn history_vec_derive(input: TokenStream) -> TokenStream {
    history_vec_derive::history_vec_derive(input)
}

/// Derive implementation of ApproxEq trait
#[proc_macro_derive(ApproxEq)]
pub fn approx_eq_derive(input: TokenStream) -> TokenStream {
    approx_eq_derive::approx_eq_derive(input)
}
