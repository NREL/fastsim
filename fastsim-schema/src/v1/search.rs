use super::IndexEntryV1;
use serde::{Deserialize, Serialize};

/// Structured search constraints over an `IndexEntryV1` slice.
///
/// Every field is an exact-match constraint against the corresponding
/// `IndexEntryV1` field; `None` means "no constraint on this field." `model`
/// additionally supports substring matching (case-insensitive) since model
/// names are the field users are most likely to search partially
/// (e.g. "f-150" should match "f-150-raptor").
///
/// All fields left as `None` (the `Default`) matches every entry.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub fastsim_version: Option<u32>,
    pub powertrain: Option<String>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub year: Option<String>,
    pub variant: Option<String>,
}

/// Filter `entries` against `query`, returning references to matches in their
/// original order.
///
/// This is intentionally I/O-free and allocation-light (no cloning of entries)
/// so it can back both `fastsim-core`'s `Vehicle::from_db` and the wasm-bound
/// browser search widget from the same implementation.
pub fn search<'a>(entries: &'a [IndexEntryV1], query: &Query) -> Vec<&'a IndexEntryV1> {
    entries.iter().filter(|e| matches(e, query)).collect()
}

fn matches(entry: &IndexEntryV1, query: &Query) -> bool {
    query
        .fastsim_version
        .is_none_or(|v| entry.fastsim_version == v)
        && query
            .powertrain
            .as_deref()
            .is_none_or(|v| entry.powertrain.eq_ignore_ascii_case(v))
        && query
            .make
            .as_deref()
            .is_none_or(|v| entry.make.eq_ignore_ascii_case(v))
        && query.model.as_deref().is_none_or(|v| {
            entry
                .model
                .to_ascii_lowercase()
                .contains(&v.to_ascii_lowercase())
        })
        && query.year.as_deref().is_none_or(|v| entry.year == v)
        && query
            .variant
            .as_deref()
            .is_none_or(|v| entry.variant.eq_ignore_ascii_case(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VehicleSchemaV1;
    use std::str::FromStr;

    fn entry(id: &str) -> IndexEntryV1 {
        IndexEntryV1::new(format!("{id}.yaml"), "2026-01-01T00:00:00Z").unwrap()
    }

    fn sample_entries() -> Vec<IndexEntryV1> {
        vec![
            entry("v1/fastsim-3/conv/ford/fusion/2012/base/r1"),
            entry("v1/fastsim-3/bev/ford/f-150-lightning/2024/base/r1"),
            entry("v1/fastsim-3/bev/tesla/model-3/2020/base/r1"),
            entry("v1/fastsim-3/conv/ford/f-150/2018/base/r1"),
        ]
    }

    #[test]
    fn empty_query_matches_everything() {
        let entries = sample_entries();
        let results = search(&entries, &Query::default());
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn filters_by_make_case_insensitively() {
        let entries = sample_entries();
        let query = Query {
            make: Some("FORD".to_string()),
            ..Default::default()
        };
        let results = search(&entries, &query);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|e| e.make == "ford"));
    }

    #[test]
    fn filters_by_powertrain_and_make_together() {
        let entries = sample_entries();
        let query = Query {
            powertrain: Some("bev".to_string()),
            make: Some("ford".to_string()),
            ..Default::default()
        };
        let results = search(&entries, &query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model, "f-150-lightning");
    }

    #[test]
    fn model_substring_match_is_case_insensitive() {
        let entries = sample_entries();
        let query = Query {
            model: Some("F-150".to_string()),
            ..Default::default()
        };
        let results = search(&entries, &query);
        assert_eq!(results.len(), 2); // f-150 and f-150-lightning
    }

    #[test]
    fn no_matches_returns_empty() {
        let entries = sample_entries();
        let query = Query {
            make: Some("toyota".to_string()),
            ..Default::default()
        };
        assert!(search(&entries, &query).is_empty());
    }

    #[test]
    fn fastsim_version_filters_correctly() {
        let entries = sample_entries();
        let query = Query {
            fastsim_version: Some(3),
            ..Default::default()
        };
        assert_eq!(search(&entries, &query).len(), 4);

        let query = Query {
            fastsim_version: Some(4),
            ..Default::default()
        };
        assert!(search(&entries, &query).is_empty());
    }
}
