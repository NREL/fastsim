#![cfg(feature = "wasm")]

use crate::{IndexEntryV1, VehicleSchemaV1};
use std::str::FromStr;
use wasm_bindgen::prelude::*;

/// Build a download URL for a vehicle id, defaulting to the GitHub raw-content
/// base URL for `fastsim-vehicles` when `base_url` is not provided.
#[wasm_bindgen]
pub fn build_download_url(
    id: &str,
    extension: &str,
    base_url: Option<String>,
) -> Result<String, JsError> {
    let schema = VehicleSchemaV1::from_str(id).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(schema.build_url(base_url.as_deref(), extension))
}

/// Parse `vehicles.jsonl` text (as fetched client-side) into index entries.
///
/// This is the wasm-facing counterpart to `IndexEntryV1::read_jsonl` — the
/// browser widget should fetch the raw file text and pass it here rather
/// than re-implementing line-splitting/JSON-parsing in JS.
#[wasm_bindgen]
pub fn parse_vehicles_jsonl(text: &str) -> Result<JsValue, JsError> {
    let entries = IndexEntryV1::read_jsonl(text).map_err(|e| JsError::new(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&entries).map_err(|e| JsError::new(&e.to_string()))
}

// /// Filter a JSON array of `IndexEntryV1` objects against a JSON-encoded
// /// `Query`. Returns matching entries as a JS array, preserving original order.
// ///
// /// Typical usage: call `parse_vehicles_jsonl` once on page load, keep the
// /// resulting entries in JS, and call this on every filter-input change with
// /// `JSON.stringify(entries)` and the current query.
// #[wasm_bindgen]
// pub fn search_entries(entries_json: &str, query_json: &str) -> Result<JsValue, JsError> {
//     let entries: Vec<IndexEntryV1> =
//         serde_json::from_str(entries_json).map_err(|e| JsError::new(&e.to_string()))?;
//     let query: Query =
//         serde_json::from_str(query_json).map_err(|e| JsError::new(&e.to_string()))?;

//     let results = search(&entries, &query);
//     serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
// }

#[cfg(test)]
mod tests {
    use super::*;

    // wasm_bindgen-exported functions can't be exercised as plain Rust unit
    // tests when compiled for a non-wasm target (JsValue/JsError aren't
    // constructible outside a wasm runtime). The functions here are thin
    // wrappers with no independent logic — their behavior is covered by the
    // underlying VehicleSchemaV1/IndexEntryV1/search tests in v1/mod.rs,
    // v1/index.rs, and v1/search.rs. Actual wasm-facing behavior should be
    // exercised with `wasm-bindgen-test` in a headless browser/node runner
    // if regressions specifically at the JS boundary become a concern.
}
