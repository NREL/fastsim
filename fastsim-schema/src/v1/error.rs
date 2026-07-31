#[derive(Debug, thiserror::Error)]
pub enum VehicleSchemaV1Error {
    #[error("expected 8 path segments, got {actual}: {input:?}")]
    SegmentCount { input: String, actual: usize },

    #[error("expected schema prefix 'v1', got {found:?}")]
    WrongSchemaPrefix { found: String },

    #[error("expected 'fastsim-N' segment, got {segment:?}")]
    MissingFastsimVersionPrefix { segment: String },

    #[error("invalid FASTSim version in {segment:?}")]
    InvalidFastsimVersion {
        segment: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("expected 'rN' revision segment, got {segment:?}")]
    MissingRevisionPrefix { segment: String },

    #[error("invalid revision in {segment:?}")]
    InvalidRevision {
        segment: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error(
        "{field} contains invalid identifier {value:?}, proper identifier would be {suggestion:?}"
    )]
    InvalidIdentifier {
        field: &'static str,
        value: String,
        suggestion: String,
    },
}
