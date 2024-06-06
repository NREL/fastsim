pub mod n;
pub mod one;
pub mod three;
pub mod two;

pub use n::*;
pub use one::*;
pub use three::*;
pub use two::*;

use crate::imports::*;

use std::marker::PhantomData; // used as a private field to disallow direct instantiation

// This method contains code from RouteE Compass, another NREL-developed tool
// https://www.nrel.gov/transportation/route-energy-prediction-model.html
// https://github.com/NREL/routee-compass/
fn find_nearest_index(arr: &[f64], target: f64) -> anyhow::Result<usize> {
    if &target == arr.last().unwrap() {
        return Ok(arr.len() - 2);
    }

    let mut low = 0;
    let mut high = arr.len() - 1;

    while low < high {
        let mid = low + (high - low) / 2;

        if arr[mid] >= target {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    if low > 0 && arr[low] >= target {
        Ok(low - 1)
    } else {
        Ok(low)
    }
}

#[derive(Debug)]
pub enum Interpolator {
    Interp0D(f64),
    Interp1D(Interp1D),
    Interp2D(Interp2D),
    Interp3D(Interp3D),
    InterpND(InterpND),
}

impl Interpolator {
    pub fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        self.validate_inputs(point)?;
        match self {
            Self::Interp0D(value) => Ok(*value),
            Self::Interp1D(interp) => {
                match interp.extrapolate {
                    Extrapolate::Clamp => {
                        let clamped_point =
                            &[point[0].clamp(interp.x[0], *interp.x.last().unwrap())];
                        return interp.interpolate(clamped_point);
                    }
                    Extrapolate::Error => {
                        let x_dim_ok =
                            interp.x[0] <= point[0] && &point[0] <= interp.x.last().unwrap();
                        ensure!(
                            x_dim_ok,
                            "Attempted to interpolate at point beyond grid data: point = {point:?}, grid = {:?}",
                            interp.x,
                        );
                    }
                    _ => {}
                };
                interp.interpolate(point)
            }
            Self::Interp2D(interp) => {
                match interp.extrapolate {
                    Extrapolate::Clamp => {
                        let clamped_point = &[
                            point[0].clamp(interp.x[0], *interp.x.last().unwrap()),
                            point[1].clamp(interp.y[0], *interp.y.last().unwrap()),
                        ];
                        return interp.interpolate(clamped_point);
                    }
                    Extrapolate::Error => {
                        let x_dim_ok =
                            interp.x[0] <= point[0] && &point[0] <= interp.x.last().unwrap();
                        let y_dim_ok =
                            interp.y[0] <= point[1] && &point[1] <= interp.y.last().unwrap();
                        ensure!(
                            x_dim_ok && y_dim_ok,
                            "Attempted to interpolate at point beyond grid data: point = {point:?}, x grid = {:?}, y grid = {:?}",
                            interp.x,
                            interp.y,
                        );
                    }
                    _ => {}
                };
                interp.interpolate(point)
            }
            Self::Interp3D(interp) => {
                match interp.extrapolate {
                    Extrapolate::Clamp => {
                        let clamped_point = &[
                            point[0].clamp(interp.x[0], *interp.x.last().unwrap()),
                            point[1].clamp(interp.y[0], *interp.y.last().unwrap()),
                            point[2].clamp(interp.z[0], *interp.z.last().unwrap()),
                        ];
                        return interp.interpolate(clamped_point);
                    }
                    Extrapolate::Error => {
                        let x_dim_ok =
                            interp.x[0] <= point[0] && &point[0] <= interp.x.last().unwrap();
                        let y_dim_ok =
                            interp.y[0] <= point[1] && &point[1] <= interp.y.last().unwrap();
                        let z_dim_ok =
                            interp.z[0] <= point[2] && &point[2] <= interp.z.last().unwrap();
                        ensure!(x_dim_ok && y_dim_ok && z_dim_ok,
                            "Attempted to interpolate at point beyond grid data: point = {point:?}, x grid = {:?}, y grid = {:?}, z grid = {:?}",
                            interp.x,
                            interp.y,
                            interp.z,
                        );
                    }
                    _ => {}
                };
                interp.interpolate(point)
            }
            Self::InterpND(interp) => {
                match interp.extrapolate {
                    Extrapolate::Clamp => {
                        let clamped_point: Vec<f64> = point
                            .iter()
                            .enumerate()
                            .map(|(dim, pt)|
                                pt.clamp(interp.grid[dim].min().unwrap(), interp.grid[dim].max().unwrap())
                            ).collect();
                        return interp.interpolate(&clamped_point);
                    }
                    Extrapolate::Error => ensure!(
                        point.iter().enumerate().all(|(dim, pt_dim)| &interp.grid[dim][0] <= pt_dim && pt_dim <= interp.grid[dim].last().unwrap()),
                        "Attempted to interpolate at point beyond grid data: point = {point:?}, grid: {:?}",
                        interp.grid,
                    ),
                    _ => {}
                };
                interp.interpolate(&point)
            }
        }
    }

    fn validate_inputs(&self, point: &[f64]) -> anyhow::Result<()> {
        let n = self.ndim();
        // Check supplied point dimensionality
        if n == 0 {
            ensure!(
                point.is_empty(),
                "No point should be provided for 0-D interpolation"
            )
        } else {
            ensure!(
                point.len() == n,
                "Supplied point slice should have length {n} for {n}-D interpolation"
            )
        }
        Ok(())
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Interp0D(_) => 0,
            Self::Interp1D(_) => 1,
            Self::Interp2D(_) => 2,
            Self::Interp3D(_) => 3,
            Self::InterpND(interp) => interp.ndim(),
        }
    }
}

/// Interpolation strategy
#[derive(Debug)]
pub enum Strategy {
    /// Linear interpolation: https://en.wikipedia.org/wiki/Linear_interpolation
    Linear,
    /// Left-nearest (previous value) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    LeftNearest,
    /// Right-nearest (next value) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    RightNearest,
    /// Nearest value (left or right) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    Nearest,
}

/// Extrapolation strategy.
///
/// Controls what happens if supplied interpolant point
/// is outside the bounds of the interpolation grid.
#[derive(Debug)]
pub enum Extrapolate {
    /// If interpolant point is beyond the limits of the interpolation grid,
    /// find result via extrapolation using slope of nearby points.  
    /// Only implemented for 1-D linear interpolation.
    Extrapolate,
    /// Restrict interpolant point to the limits of the interpolation grid, using [`f64::clamp`].
    Clamp,
    /// Return an error when interpolant point is beyond the limits of the interpolation grid.
    Error,
}

pub trait InterpMethods {
    fn validate(&self) -> anyhow::Result<()>;
    fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_0D() {
        let expected = 0.5;
        let interp = Interpolator::Interp0D(expected);
        assert_eq!(interp.interpolate(&[]).unwrap(), expected);
        assert!(interp.interpolate(&[0.]).is_err());
    }
}
