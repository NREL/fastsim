use crate::error::Error;
use crate::imports::*;
pub mod serde_api;
pub use serde_api::*;

pub trait Linspace {
    /// Generate linearly spaced vec
    /// # Arguments
    /// - `start` - starting point
    /// - `stop` - stopping point, inclusive
    /// - `n_elements` - number of array elements
    fn linspace(start: f64, stop: f64, n_elements: usize) -> Vec<f64> {
        let n_steps = n_elements - 1;
        let step_size = (stop - start) / n_steps as f64;
        let v_norm: Vec<f64> = (0..=n_steps)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let v = v_norm.iter().map(|x| (x * step_size) + start).collect();
        v
    }
}

impl Linspace for Vec<f64> {}

pub trait Min {
    fn min(&self) -> anyhow::Result<f64>;
}
impl Min for &[f64] {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }
}
impl Min for Vec<f64> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for &[&f64] {
    fn min(&self) -> anyhow::Result<f64> {
        Ok(self.iter().fold(f64::INFINITY, |acc, curr| acc.min(**curr)))
    }
}
impl Min for Vec<&f64> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for &[Vec<f64>] {
    fn min(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.min())
            .try_fold(f64::INFINITY, |acc, x| Ok(acc.min(x?)))
    }
}
impl Min for Vec<Vec<f64>> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for &[Vec<Vec<f64>>] {
    fn min(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.min())
            .try_fold(f64::INFINITY, |acc, x| Ok(acc.min(x?)))
    }
}
impl Min for Vec<Vec<Vec<f64>>> {
    fn min(&self) -> anyhow::Result<f64> {
        self.as_slice().min()
    }
}
impl Min for Interpolator {
    fn min(&self) -> anyhow::Result<f64> {
        match self {
            Interpolator::Interp0D(value) => Ok(*value),
            Interpolator::Interp1D(..) => self.f_x()?.min(),
            Interpolator::Interp2D(..) => self.f_xy()?.min(),
            Interpolator::Interp3D(..) => self.f_xyz()?.min(),
            Interpolator::InterpND(..) => Ok(self
                .values()?
                .iter()
                .fold(f64::INFINITY, |acc, x| acc.min(*x))),
        }
    }
}

pub trait Max {
    fn max(&self) -> anyhow::Result<f64>;
}
impl Max for &[f64] {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }
}
impl Max for Vec<f64> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for &[&f64] {
    fn max(&self) -> anyhow::Result<f64> {
        Ok(self
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(**curr)))
    }
}
impl Max for Vec<&f64> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for &[Vec<f64>] {
    fn max(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.max())
            .try_fold(f64::NEG_INFINITY, |acc, x| Ok(acc.max(x?)))
    }
}
impl Max for Vec<Vec<f64>> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for &[Vec<Vec<f64>>] {
    fn max(&self) -> anyhow::Result<f64> {
        self.iter()
            .map(|v| v.max())
            .try_fold(f64::NEG_INFINITY, |acc, x| Ok(acc.max(x?)))
    }
}
impl Max for Vec<Vec<Vec<f64>>> {
    fn max(&self) -> anyhow::Result<f64> {
        self.as_slice().max()
    }
}
impl Max for Interpolator {
    fn max(&self) -> anyhow::Result<f64> {
        match self {
            Interpolator::Interp0D(value) => Ok(*value),
            Interpolator::Interp1D(..) => self.f_x()?.max(),
            Interpolator::Interp2D(..) => self.f_xy()?.max(),
            Interpolator::Interp3D(..) => self.f_xyz()?.max(),
            Interpolator::InterpND(..) => Ok(self
                .values()?
                .iter()
                .fold(f64::NEG_INFINITY, |acc, x| acc.max(*x))),
        }
    }
}

pub trait Init {
    /// Specialized code to execute upon initialization.  For any struct with fields
    /// that implement `Init`, this should propagate down the hierarchy.
    fn init(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

pub trait Diff<T> {
    /// Returns vec of length `self.len() - 1` where each element in the returned vec at index i is
    /// `self[i + 1] - self[i]`
    fn diff(&self) -> Vec<T>;
}

impl<T: Clone + Sub<T, Output = T> + Default> Diff<T> for Vec<T> {
    fn diff(&self) -> Vec<T> {
        let mut v_diff: Vec<T> = vec![Default::default()];
        v_diff.extend::<Vec<T>>(
            self.windows(2)
                .map(|vs| {
                    let x = &vs[0];
                    let y = &vs[1];
                    y.clone() - x.clone()
                })
                .collect(),
        );
        v_diff
    }
}

/// Super trait to ensure that related traits are implemented together
pub trait StateMethods: SetCumulative + SaveState + Step + CheckAndResetState {}

/// Provides method that saves `self.state` to `self.history` and propagates to any fields with
/// `state`
pub trait SaveState {
    /// Saves `self.state` to `self.history` and propagates to any fields with `state`
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    fn save_state<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()>;
}

/// Trait that provides method for incrementing `i` field of this and all contained structs,
/// recursively
pub trait Step {
    /// Increments `i` field of this and all contained structs, recursively
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    fn step<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()>;
}

/// Trait for setting cumulative values based on rate values
pub trait SetCumulative {
    /// Sets cumulative values based on rate values
    // fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()>;
    fn set_cumulative(&mut self, dt: si::Time) -> anyhow::Result<()>;
}

/// Provides methods for getting and setting the save interval
pub trait HistoryMethods: SaveState {
    /// Recursively sets save interval
    /// # Arguments
    /// - `save_interval`: time step interval at which to save `self.state` to `self.history`
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()>;
    /// Returns save interval for `self` but does not guarantee recursive consistency in nested
    /// objects
    fn save_interval(&self) -> anyhow::Result<Option<usize>>;
    /// Remove all history
    fn clear(&mut self);
}

/// Provides method for checking if struct is default
pub trait EqDefault: std::default::Default + PartialEq {
    /// If `self` is default, returns true
    fn eq_default(&self) -> bool {
        *self == Self::default()
    }
}

impl<T: Default + PartialEq> EqDefault for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace() {
        assert_eq!(Vec::linspace(0., 2., 3), vec![0., 1., 2.]);
    }

    #[test]
    fn test_max_for_vec_f64() {
        assert_eq!(Vec::linspace(-10., 12., 5).max().unwrap(), 12.);
    }
    #[test]
    fn test_min_for_vec_f64() {
        assert_eq!(Vec::linspace(-10., 12., 5).min().unwrap(), -10.);
    }

    #[test]
    fn test_diff() {
        let diff = Vec::linspace(0., 2., 3).diff();
        let ref_diff = vec![0., 1., 1.];
        assert_eq!(diff, ref_diff);
    }
}
