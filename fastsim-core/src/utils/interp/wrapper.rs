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

#[pyo3_api(
    /// Function to get x variable from enum variants
    #[getter("x")]
    pub fn x_py(&self) -> anyhow::Result<Vec<f64>> {
        self.0.x()
    }

    /// Function to set x variable from enum variants
    /// # Arguments
    /// - `new_x`: updated `x` variable to replace the current `x` variable
    #[setter("__set_x")]
    pub fn set_x_py(&mut self, new_x: Vec<f64>) -> PyResult<()> {
        Ok(self.0.set_x(new_x)?)
    }

    /// Function to get f_x variable from enum variants
    #[getter("f_x")]
    pub fn f_x_py(&self) -> anyhow::Result<Vec<f64>> {
        self.0.f_x()
    }

    /// Function to set f_x variable from enum variants
    /// # Arguments
    /// - `new_f_x`: updated `f_x` variable to replace the current `f_x` variable
    #[setter("__set_f_x")]
    pub fn set_f_x_py(&mut self, new_f_x: Vec<f64>) -> anyhow::Result<()> {
        self.0.set_f_x(new_f_x)
    }

    /// Function to get strategy variable from enum variants
    #[getter("strategy")]
    pub fn strategy_py(&self) -> anyhow::Result<Strategy> {
        self.0.strategy()
    }

    /// Function to set strategy variable from enum variants
    /// # Arguments
    /// - `new_strategy`: updated `strategy` variable to replace the current `strategy` variable
    #[setter("__set_strategy")]
    pub fn set_strategy_py(&mut self, new_strategy: Strategy) -> anyhow::Result<()> {
        self.0.set_strategy(new_strategy)
    }

    /// Function to get extrapolate variable from enum variants
    #[getter("extrapolate")]
    pub fn extrapolate_py(&self) -> anyhow::Result<Extrapolate> {
        self.0.extrapolate()
    }

    /// Function to set extrapolate variable from enum variants
    /// # Arguments
    /// - `new_extrapolate`: updated `extrapolate` variable to replace the current `extrapolate` variable
    #[setter("__set_extrapolate")]
    pub fn set_extrapolate_py(&mut self, new_extrapolate: Extrapolate) -> anyhow::Result<()> {
        self.0.set_extrapolate(new_extrapolate)
    }

    /// Function to get y variable from enum variants
    #[getter("y")]
    pub fn y_py(&self) -> anyhow::Result<Vec<f64>> {
        self.0.y()
    }

    /// Function to set y variable from enum variants
    /// # Arguments
    /// - `new_y`: updated `y` variable to replace the current `y` variable
    #[setter("__set_y")]
    pub fn set_y_py(&mut self, new_y: Vec<f64>) -> anyhow::Result<()> {
        self.0.set_y(new_y)
    }

    /// Function to get f_xy variable from enum variants
    #[getter("f_xy")]
    pub fn f_xy_py(&self) -> anyhow::Result<Vec<Vec<f64>>> {
        self.0.f_xy()
    }

    /// Function to set f_xy variable from enum variants
    /// # Arguments
    /// - `new_f_xy`: updated `f_xy` variable to replace the current `f_xy` variable
    #[setter("__set_f_xy")]
    pub fn set_f_xy_py(&mut self, new_f_xy: Vec<Vec<f64>>) -> anyhow::Result<()> {
        self.0.set_f_xy(new_f_xy)
    }

    /// Function to get z variable from enum variants
    #[getter("z")]
    pub fn z_py(&self) -> anyhow::Result<Vec<f64>> {
        self.0.z()
    }

    /// Function to set z variable from enum variants
    /// # Arguments
    /// - `new_z`: updated `z` variable to replace the current `z` variable
    #[setter("__set_z")]
    pub fn set_z_py(&mut self, new_z: Vec<f64>) -> anyhow::Result<()> {
        self.0.set_z(new_z)
    }

    /// Function to get f_xyz variable from enum variants
    #[getter("f_xyz")]
    pub fn f_xyz_py(&self) -> anyhow::Result<Vec<Vec<Vec<f64>>>> {
        self.0.f_xyz()
    }

    /// Function to set f_xyz variable from enum variants
    /// # Arguments
    /// - `new_f_xyz`: updated `f_xyz` variable to replace the current `f_xyz` variable
    #[setter("__set_f_xyz")]
    pub fn set_f_xyz_py(&mut self, new_f_xyz: Vec<Vec<Vec<f64>>>) -> anyhow::Result<()> {
        self.0.set_f_xyz(new_f_xyz)
    }

    /// Function to get grid variable from enum variants
    #[getter("grid")]
    pub fn grid_py(&self) -> anyhow::Result<Vec<Vec<f64>>> {
        self.0.grid()
    }

    /// Function to set grid variable from enum variants
    /// # Arguments
    /// - `new_grid`: updated `grid` variable to replace the current `grid` variable
    #[setter("__set_grid")]
    pub fn set_grid_py(&mut self, new_grid: Vec<Vec<f64>>) -> anyhow::Result<()> {
        self.0.set_grid(new_grid)
    }

    /// Function to get values variable from enum variants
    #[getter("values")]
    pub fn values_py<'py>(&self, py: Python<'py>) -> anyhow::Result<&'py PyArrayDyn<f64>> {
        Ok(self.0.values()?.into_pyarray(py))
    }

    /// Function to set values variable from enum variants
    /// # Arguments
    /// - `new_values`: updated `values` variable to replace the current `values` variable
    #[setter("__set_values")]
    pub fn set_values_py(
        &mut self,
        new_values: &PyArrayDyn<f64>,
    ) -> anyhow::Result<()> {
        self.0.set_values(new_values.to_owned_array())
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct InterpolatorWrapper(pub Interpolator);
// pub struct InterpolatorWrapper {
//     pub interpolator: Interpolator,
// }

impl InterpolatorWrapper {
    pub fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        // self.interpolator.interpolate(point)
        self.0.interpolate(point)
    }
}

impl Init for InterpolatorWrapper {}

impl SerdeAPI for InterpolatorWrapper {}
