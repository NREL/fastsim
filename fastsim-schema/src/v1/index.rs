//! Index entry structures for vehicle discovery and filtering.
//!
//! Defines `IndexEntryV1` for representing vehicles in the `vehicles.jsonl` index file,
//! with fields for filtering (powertrain, make, model, etc.) and utilities for
//! parsing index files and building download URLs.

use super::{VehicleSchemaV1, VehicleSchemaV1Error};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::str::FromStr;

/// A single entry in the vehicle index (i.e. `vehicles.jsonl`).
///
/// Every field except `path` is redundant with `id` and exists
/// only so consumers can filter without re-parsing `id` themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexEntryV1 {
    pub path: String,
    pub id: String,
    pub fastsim_version: u32,
    pub powertrain: String,
    pub make: String,
    pub model: String,
    pub year: String,
    pub variant: String,
    pub revision: u32,
}

impl IndexEntryV1 {}

impl FromStr for IndexEntryV1 {
    type Err = VehicleSchemaV1Error;

    /// Parse an entry from a repository-relative file path including its extension
    /// (e.g. `"v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml"`).
    ///
    /// The path minus its extension is parsed and canonically validated as a
    /// `VehicleSchemaV1` id — an `IndexEntryV1` can only be constructed from a path
    /// that's already valid, so no separate validation step is needed by
    /// callers building an index from a directory scan.
    ///
    /// # Errors
    /// Returns `VehicleSchemaV1Error` if the path (minus extension) isn't a valid,
    /// canonical `v1` schema path.
    fn from_str(path: &str) -> Result<Self, Self::Err> {
        let id_str = path.rsplit_once('.').map(|(id, _ext)| id).unwrap_or(path);
        let schema = id_str.parse::<VehicleSchemaV1>()?;

        Ok(Self {
            path: path.to_string(),
            id: schema.to_string(),
            fastsim_version: schema.fastsim_version,
            powertrain: schema.powertrain,
            make: schema.make,
            model: schema.model,
            year: schema.year,
            variant: schema.variant,
            revision: schema.revision,
        })
    }
}

/// Serialize a slice of entries into JSON Lines, one entry per line
/// in the given order, written to `writer`. Callers wanting a specific
/// ordering (e.g. sorted by path) should sort `entries` beforehand —
/// this function preserves input order rather than imposing one.
pub fn write_jsonl_v1<W: Write>(
    writer: &mut W,
    entries: &[IndexEntryV1],
) -> Result<(), serde_json::Error> {
    for entry in entries {
        serde_json::to_writer(&mut *writer, entry)?;
        writer.write_all(b"\n").map_err(serde_json::Error::io)?;
    }
    Ok(())
}

/// Parse JSON Lines text (as read from `vehicles.jsonl`, or fetched
/// client-side in the browser widget) into entries.
///
/// Blank lines are skipped; any malformed line surfaces its
/// `serde_json::Error` rather than being silently dropped.
pub fn read_jsonl_v1(text: &str) -> Result<Vec<IndexEntryV1>, serde_json::Error> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect()
}

#[cfg(test)]
mod index_tests {
    use super::*;

    #[test]
    fn new_parses_and_populates_all_fields() {
        let entry = "v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml"
            .parse::<IndexEntryV1>()
            .unwrap();

        assert_eq!(
            entry.path,
            "v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml"
        );
        assert_eq!(entry.id, "v1/fastsim-3/conv/ford/fusion/2012/base/r1");
        assert_eq!(entry.fastsim_version, 3);
        assert_eq!(entry.powertrain, "conv");
        assert_eq!(entry.make, "ford");
        assert_eq!(entry.model, "fusion");
        assert_eq!(entry.year, "2012");
        assert_eq!(entry.variant, "base");
        assert_eq!(entry.revision, 1);
    }

    #[test]
    fn new_rejects_invalid_schema_path() {
        let result = "v1/not-enough-segments.yaml".parse::<IndexEntryV1>();
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_non_canonical_identifiers() {
        // "Ford" is not canonical (must be lowercase) — new should surface
        // the same InvalidIdentifier error SchemaV1::from_str would.
        let result = "v1/fastsim-3/conv/Ford/fusion/2012/base/r1.yaml".parse::<IndexEntryV1>();
        assert!(matches!(
            result,
            Err(VehicleSchemaV1Error::InvalidIdentifier { .. })
        ));
    }

    #[test]
    fn new_round_trips_through_json() {
        let entry = "v1/fastsim-3/bev/tesla/model-3/2020/base/r1.yaml"
            .parse::<IndexEntryV1>()
            .unwrap();
        let json = serde_json::to_string(&entry).unwrap();
        let round_tripped: IndexEntryV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, round_tripped);
    }
}

#[cfg(test)]
mod jsonl_tests {
    use super::*;

    fn sample_entries() -> Vec<IndexEntryV1> {
        vec![
            "v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml"
                .parse::<IndexEntryV1>()
                .unwrap(),
            "v1/fastsim-3/bev/tesla/model-3/2020/base/r1.yaml"
                .parse::<IndexEntryV1>()
                .unwrap(),
        ]
    }

    #[test]
    fn write_then_read_round_trips() {
        let entries = sample_entries();
        let mut buf = Vec::new();
        write_jsonl_v1(&mut buf, &entries).unwrap();
        let jsonl = String::from_utf8(buf).unwrap();
        assert_eq!(jsonl.lines().count(), 2);

        let parsed = read_jsonl_v1(&jsonl).unwrap();
        assert_eq!(parsed, entries);
    }

    #[test]
    fn write_jsonl_of_empty_slice_writes_nothing() {
        let mut buf = Vec::new();
        write_jsonl_v1(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn read_jsonl_of_empty_string_is_empty_vec() {
        let parsed = read_jsonl_v1("").unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn read_jsonl_surfaces_malformed_line_error() {
        let text = "{\"not\": \"a valid IndexEntryV1\"}\n";
        assert!(read_jsonl_v1(text).is_err());
    }
}
