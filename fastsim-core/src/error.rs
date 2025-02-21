//! Custom error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FsimError {
    #[error("`Init::init` failed: {0}")]
    InitError(String),
    #[error("`SerdeAPI` failed {0}")]
    SerdeError(String),
    #[error("{0}")]
    SimulationError(String),
    #[error(transparent)]
    NinterpError(#[from] ninterp::Error),
    #[error("{0}")]
    Other(String),
}
