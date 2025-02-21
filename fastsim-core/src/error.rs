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
    #[error(transparent)]
    NinterpError(#[from] ninterp::error::Error),
    #[error("{0}")]
    Other(String),
}

impl Error {
    fn format_dbg(&self) -> Self {
        match &self {
            Self::InitError(err) => Self::InitError(format_dbg!(err)),
            Self::SerdeError(err) => Self::SerdeError(format_dbg!(err)),
            Self::SimulationError(err) => Self::SimulationError(format_dbg!(err)),
            Self::NinterpError(_) => self.clone(), // not possible here
            Self::Other(err) => Self::Other(format_dbg!(err)),
        }
    }
}
