//! Custom error types

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("`Init::init` failed: {0}")]
    InitError(String),
    #[error("`SerdeAPI` failed {0}")]
    SerdeError(String),
    #[error("{0}")]
    SimulationError(String),
    #[error("{0}")]
    NinterpError(String),
    #[error("{0}")]
    Other(String),
}
