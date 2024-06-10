//! Wrapper for Python-exposed [`Interpolator`] instances
//!
//! PyO3 does not currently support tuple enum variants,
//! which means it is incompatible with the structure of the interpolation module
//!
//! Providing a simple tuple struct that wraps the [`Interpolator`] enum fixes this.
//!
//! When tuple enum variants are supported by PyO3, upgrade, remove this module, and add
//! `#[cfg_attr(feature = "pyo3", pyclass)]`
//! to the [`Interpolator`] enum directly.
//!

use super::*;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InterpolatorWrapper(pub Interpolator);

impl InterpolatorWrapper {
    pub fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        self.0.interpolate(point)
    }
}
