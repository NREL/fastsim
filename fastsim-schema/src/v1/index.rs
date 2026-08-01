//! Index entry structures for vehicle discovery and filtering.
//!
//! Defines `IndexEntryV1` for representing vehicles in the `vehicles.jsonl` index file,
//! with fields for filtering (powertrain, make, model, etc.) and utilities for
//! parsing index files and building download URLs.

use super::{VehicleSchemaV1, VehicleSchemaV1Error};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// A single entry in the vehicle index (i.e. `vehicles.jsonl`).
///
/// Purely path-derived — no vehicle content (mass, fuel type, etc.) lives here.
/// Every field except `path` and `date_added` is redundant with `id` and exists
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
    pub date_added: String,
}

impl IndexEntryV1 {
    /// Construct an entry from a vehicle file's path, relative to the
    /// repository root and including its extension
    /// (e.g. `"v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml"`), and its
    /// `date_added` timestamp.
    ///
    /// The path (minus its extension) is parsed and canonically validated as
    /// a `VehicleSchemaV1` id — an `IndexEntryV1` can only be constructed from a path
    /// that's already valid, so no separate validation step is needed by
    /// callers building an index from a directory scan.
    ///
    /// # Errors
    /// Returns `VehicleSchemaV1Error` if the path (minus extension) isn't a valid,
    /// canonical `v1` schema path.
    pub fn new(
        path: impl Into<String>,
        date_added: impl Into<String>,
    ) -> Result<Self, VehicleSchemaV1Error> {
        let path = path.into();
        let id_str = path.rsplit_once('.').map(|(id, _ext)| id).unwrap_or(&path);
        let schema = id_str.parse::<VehicleSchemaV1>()?;

        Ok(Self {
            path,
            id: schema.to_string(),
            fastsim_version: schema.fastsim_version,
            powertrain: schema.powertrain,
            make: schema.make,
            model: schema.model,
            year: schema.year,
            variant: schema.variant,
            revision: schema.revision,
            date_added: date_added.into(),
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
        let entry = IndexEntryV1::new(
            "v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml",
            "2026-07-31T00:00:00Z",
        )
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
        assert_eq!(entry.date_added, "2026-07-31T00:00:00Z");
    }

    #[test]
    fn new_rejects_invalid_schema_path() {
        let result = IndexEntryV1::new("v1/not-enough-segments.yaml", "2026-01-01T00:00:00Z");
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_non_canonical_identifiers() {
        // "Ford" is not canonical (must be lowercase) — new should surface
        // the same InvalidIdentifier error SchemaV1::from_str would.
        let result = IndexEntryV1::new(
            "v1/fastsim-3/conv/Ford/fusion/2012/base/r1.yaml",
            "2026-01-01T00:00:00Z",
        );
        assert!(matches!(
            result,
            Err(VehicleSchemaV1Error::InvalidIdentifier { .. })
        ));
    }

    #[test]
    fn new_round_trips_through_json() {
        let entry = IndexEntryV1::new(
            "v1/fastsim-3/bev/tesla/model-3/2020/base/r1.yaml",
            "2026-01-01T00:00:00Z",
        )
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
            IndexEntryV1::new(
                "v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml",
                "2026-01-01T00:00:00Z",
            )
            .unwrap(),
            IndexEntryV1::new(
                "v1/fastsim-3/bev/tesla/model-3/2020/base/r1.yaml",
                "2026-02-01T00:00:00Z",
            )
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

        let parsed = read_jsonl(&jsonl).unwrap();
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
        let parsed = read_jsonl("").unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn read_jsonl_surfaces_malformed_line_error() {
        let text = "{\"not\": \"a valid IndexEntryV1\"}\n";
        assert!(read_jsonl(text).is_err());
    }
}
