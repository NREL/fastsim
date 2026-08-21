use super::IndexEntryV1;
use serde::{Deserialize, Serialize};

/// Structured search constraints over an `IndexEntryV1` slice.
///
/// `None` means "no constraint on this field." `fastsim_version` and
/// `powertrain` are exact-match; `make`, `model`, `year`, and `variant` match
/// on case-insensitive substring (e.g. "f-150" should match "f-150-raptor").
///
/// All fields left as `None` (the `Default`) matches every entry.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryV1 {
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
pub fn search_v1<'a>(entries: &'a [IndexEntryV1], query: &QueryV1) -> Vec<&'a IndexEntryV1> {
    entries.iter().filter(|e| matches(e, query)).collect()
}

/// Case-insensitive substring match.
fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn matches(entry: &IndexEntryV1, query: &QueryV1) -> bool {
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
            .is_none_or(|v| contains_ignore_case(&entry.make, v))
        && query
            .model
            .as_deref()
            .is_none_or(|v| contains_ignore_case(&entry.model, v))
        && query
            .year
            .as_deref()
            .is_none_or(|v| contains_ignore_case(&entry.year, v))
        && query
            .variant
            .as_deref()
            .is_none_or(|v| contains_ignore_case(&entry.variant, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> IndexEntryV1 {
        format!("{id}.yaml").parse::<IndexEntryV1>().unwrap()
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
        let results = search_v1(&entries, &QueryV1::default());
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn filters_by_make_case_insensitively() {
        let entries = sample_entries();
        let query = QueryV1 {
            make: Some("FORD".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|e| e.make == "ford"));
    }

    #[test]
    fn filters_by_powertrain_and_make_together() {
        let entries = sample_entries();
        let query = QueryV1 {
            powertrain: Some("bev".to_string()),
            make: Some("ford".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].model, "f-150-lightning");
    }

    #[test]
    fn model_substring_match_is_case_insensitive() {
        let entries = sample_entries();
        let query = QueryV1 {
            model: Some("F-150".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 2); // f-150 and f-150-lightning
    }

    #[test]
    fn make_substring_match_is_case_insensitive() {
        let entries = sample_entries();
        let query = QueryV1 {
            make: Some("FOR".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|e| e.make == "ford"));
    }

    #[test]
    fn year_substring_match() {
        let entries = sample_entries();
        let query = QueryV1 {
            year: Some("201".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 2); // 2012 and 2018
    }

    #[test]
    fn variant_substring_match_is_case_insensitive() {
        let entries = sample_entries();
        let query = QueryV1 {
            variant: Some("BAS".to_string()),
            ..Default::default()
        };
        let results = search_v1(&entries, &query);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn no_matches_returns_empty() {
        let entries = sample_entries();
        let query = QueryV1 {
            make: Some("toyota".to_string()),
            ..Default::default()
        };
        assert!(search_v1(&entries, &query).is_empty());
    }

    #[test]
    fn fastsim_version_filters_correctly() {
        let entries = sample_entries();
        let query = QueryV1 {
            fastsim_version: Some(3),
            ..Default::default()
        };
        assert_eq!(search_v1(&entries, &query).len(), 4);

        let query = QueryV1 {
            fastsim_version: Some(4),
            ..Default::default()
        };
        assert!(search_v1(&entries, &query).is_empty());
    }
}
