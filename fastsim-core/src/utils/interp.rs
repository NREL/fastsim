use uom::si::length::yard;

use crate::imports::*;

#[derive(Debug)]
pub enum Interpolation {
    Interp0D(f64),
    Interp1D(Interp1D),
    Interp2D(Interp2D),
    Interp3D(Interp3D),
    InterpND(InterpND),
}

impl Interpolation {
    pub fn interpolate(&self, point: &[f64], strategy: &Strategy) -> anyhow::Result<f64> {
        self.validate_inputs(point, strategy)?;
        match self {
            Self::Interp0D(value) => {
                ensure!(
                    matches!(strategy, Strategy::None),
                    "Provided strategy {:?} is not applicable for 0-D, select {:?}",
                    strategy,
                    Strategy::None
                );
                Ok(*value)
            }
            Self::Interp1D(interp) => match strategy {
                Strategy::Linear => interp.linear(point[0]),
                Strategy::LeftNearest => interp.left_nearest(point[0]),
                Strategy::RightNearest => interp.right_nearest(point[0]),
                Strategy::Nearest => interp.nearest(point[0]),
                _ => bail!(
                    "Provided strategy {:?} is not applicable for 1-D interpolation",
                    strategy
                ),
            },
            Self::Interp2D(interp) => match strategy {
                Strategy::Linear => interp.linear(point),
                _ => bail!(
                    "Provided strategy {:?} is not applicable for 2-D interpolation",
                    strategy
                ),
            },
            Self::Interp3D(interp) => match strategy {
                Strategy::Linear => interp.linear(point),
                _ => bail!(
                    "Provided strategy {:?} is not applicable for 3-D interpolation",
                    strategy
                ),
            },
            Self::InterpND(interp) => {
                // ensure!(point.len() == interp.values.ndim(), "Supplied point slice should have length {n} for {n}-D interpolation", n = interp.values.ndim());
                match strategy {
                    Strategy::Linear => interp.linear(point),
                    _ => bail!(
                        "Provided strategy {:?} is not applicable for 3-D interpolation",
                        strategy
                    ),
                }
            }
        }
    }

    fn validate_inputs(&self, point: &[f64], _strategy: &Strategy) -> anyhow::Result<()> {
        let n = self.ndim();
        // Check supplied point dimensionality
        match self {
            Self::Interp0D(_) => ensure!(
                point.is_empty(),
                "No point should be provided for 0-D interpolation"
            ),
            _ => {
                ensure!(
                    point.len() == n,
                    "Supplied point slice should have length {n} for {n}-D interpolation"
                )
            }
        }
        // Check that grid points are monotonically increasing
        match self {
            Self::Interp1D(interp) => {
                anyhow::ensure!(
                    interp.x.windows(2).all(|w| w[0] < w[1]),
                    "Supplied x-values must be sorted and non-repeating"
                );
            }
            Self::Interp2D(interp) => {
                anyhow::ensure!(
                    interp.x.windows(2).all(|w| w[0] < w[1])
                    && interp.y.windows(2).all(|w| w[0] < w[1]),
                    "Supplied values must be sorted and non-repeating"
                );
            }
            Self::Interp3D(interp) => {
                anyhow::ensure!(
                    interp.x.windows(2).all(|w| w[0] < w[1])
                    && interp.y.windows(2).all(|w| w[0] < w[1])
                    && interp.z.windows(2).all(|w| w[0] < w[1]),
                    "Supplied values must be sorted and non-repeating"
                );
            }
            Self::InterpND(interp) => {
                for i in 0..n {
                    anyhow::ensure!(
                    interp.grid[i].windows(2).all(|w| w[0] < w[1]),
                    "Supplied `grid` coordinates must be sorted and non-repeating: dimension {i}, {:?}",
                    interp.grid[i]
                );
                }
            }
            _ => (),
        }
        // Check that grid and values are compatible shapes
        match self {
            Self::Interp1D(interp) => {
                anyhow::ensure!(interp.x.len() == interp.f_x.len(), "Supplied grid and values are not compatible shapes");
            }
            Self::Interp2D(interp) => {
                anyhow::ensure!(
                    interp.x.len() == interp.f_xy.len()
                    && interp.y.len() == interp.f_xy[0].len(),
                    // TODO: protect against different length y-data
                    "Supplied grid and values are not compatible shapes"
                );
            }
            Self::Interp3D(interp) => {
                anyhow::ensure!(
                    interp.x.len() == interp.f_xyz.len()
                    && interp.y.len() == interp.f_xyz[0].len()
                    && interp.z.len() == interp.f_xyz[0][0].len(),
                    // TODO: protect against different length y-data, as well as different length z-data
                    "Supplied grid and values are not compatible shapes"
                );
            }
            Self::InterpND(interp) => {
                for i in 0..n {
                    anyhow::ensure!(
                    interp.grid[i].len() == interp.values.shape()[i],
                    "Supplied grid and values are not compatible shapes: dimension {i}, lengths {} != {}",
                    interp.grid[i].len(),
                    interp.values.shape()[i]
                );
                }
            }
            _ => (),
        }
        // Check that point is within grid in each dimension
        match self {
            Self::Interp1D(interp) => {
                anyhow::ensure!(
                    interp.x[0] <= point[0] && point[0] <= *interp.x.last().unwrap(),
                    "Supplied point must be within grid: point = {point:?}, x = {:?}",
                    interp.x
                );
            }
            Self::Interp2D(interp) => {
                anyhow::ensure!(
                    (interp.x[0] <= point[0] && point[0] <= *interp.x.last().unwrap())
                        && (interp.y[0] <= point[1] && point[1] <= *interp.y.last().unwrap()),
                    "Supplied point must be within grid: point = {point:?}, x = {:?}, y = {:?}",
                    interp.x,
                    interp.y,
                );
            }
            Self::Interp3D(interp) => {
                anyhow::ensure!(
                    (interp.x[0] <= point[0] && point[0] <= *interp.x.last().unwrap())
                    && (interp.y[0] <= point[1] && point[1] <= *interp.y.last().unwrap())
                    && (interp.z[0] <= point[2] && point[2] <= *interp.z.last().unwrap()),
                    "Supplied point must be within grid: point = {point:?}, x = {:?}, y = {:?}, z = {:?}",
                    interp.x,
                    interp.y,
                    interp.z,
                );
            }
            Self::InterpND(interp) => {
                for i in 0..n {
                    anyhow::ensure!(
                    interp.grid[i][0] <= point[i] && point[i] <= *interp.grid[i].last().unwrap(),
                    "Supplied point must be within grid for dimension {i}: point[{i}] = {:?}, grid[{i}] = {:?}",
                    point[i],
                    interp.grid[i],
                );
                }
            }
            _ => (),
        }
        // N-dimensional specific checks
        if let Self::InterpND(interp) = self {
            // Check grid dimensionality
            anyhow::ensure!(
                interp.grid.len() == n,
                "Length of supplied `grid` must be same as `values` dimensionality: {:?} is not {n}-dimensional",
                interp.grid
            );
        }
        Ok(())
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Interp0D(_) => 0,
            Self::Interp1D(_) => 1,
            Self::Interp2D(_) => 2,
            Self::Interp3D(_) => 3,
            Self::InterpND(interp) => interp.values.ndim(),
        }
    }
}

/// Interpolation strategy
#[derive(Debug)]
pub enum Strategy {
    /// N/A (for 0-dimensional interpolation)
    None,
    /// Linear interpolation: https://en.wikipedia.org/wiki/Linear_interpolation
    Linear,
    /// Left-nearest (previous value) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    LeftNearest,
    /// Right-nearest (next value) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    RightNearest,
    /// Nearest value (left or right) interpolation: https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    Nearest,
}

/// 1-dimensional interpolation
#[derive(Debug)]
pub struct Interp1D {
    pub x: Vec<f64>,
    pub f_x: Vec<f64>,
}
impl Interp1D {
    fn linear(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i])
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        let diff = (point - self.x[lower_index]) / (self.x[lower_index + 1] - self.x[lower_index]);
        Ok(self.f_x[lower_index] * (1.0 - diff) + self.f_x[lower_index + 1] * diff)
    }

    fn left_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i])
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        Ok(self.f_x[lower_index])
    }

    fn right_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i])
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        Ok(self.f_x[lower_index + 1])
    }

    fn nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i])
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        let diff = (point - self.x[lower_index]) / (self.x[lower_index + 1] - self.x[lower_index]);
        Ok(if diff < 0.5 {
            self.f_x[lower_index]
        } else {
            self.f_x[lower_index + 1]
        })
    }
}

/// 2-dimensional interpolation
#[derive(Debug)]
pub struct Interp2D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub f_xy: Vec<Vec<f64>>,
}
impl Interp2D {
    fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        todo!()
    }
}

/// 3-dimensional interpolation
#[derive(Debug)]
pub struct Interp3D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub f_xyz: Vec<Vec<Vec<f64>>>,
}
impl Interp3D {
    fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        todo!()
    }
}

/// N-dimensional interpolation
#[derive(Debug)]
pub struct InterpND {
    pub grid: Vec<Vec<f64>>,
    pub values: ArrayD<f64>,
}
impl InterpND {
    fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        todo!() // copy multilinear into this
    }
}

#[cfg(test)]
mod tests_0D {
    use super::*;

    #[test]
    fn test_0d() {
        let strategy = Strategy::None;
        let expected = 0.5;
        let interp = Interpolation::Interp0D(expected);
        assert_eq!(interp.interpolate(&[], &strategy).unwrap(), expected);
        assert!(interp.interpolate(&[0.], &strategy).is_err());
        assert!(interp.interpolate(&[], &strategy).is_err());
    }
}

#[cfg(test)]
mod tests_1D {
    use super::*;

    fn setup_1D() -> Interpolation {
        // f(x) = 0.2x + 0.2
        Interpolation::Interp1D(Interp1D {
            x: vec![0., 1., 2., 3., 4.],
            f_x: vec![0.2, 0.4, 0.6, 0.8, 1.0],
        })
    }

    #[test]
    fn test_invalid_args() {
        let interp = setup_1D();
        assert!(interp.interpolate(&[], &Strategy::Linear).is_err());
        assert!(interp.interpolate(&[1.0], &Strategy::None).is_err());
    }

    #[test]
    fn test_1D_linear() {
        let strategy = Strategy::Linear;
        let interp = setup_1D();
        assert_eq!(interp.interpolate(&[3.00], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75], &strategy).unwrap(), 0.95);
        assert_eq!(interp.interpolate(&[4.00], &strategy).unwrap(), 1.0);
    }

    #[test]
    fn test_1D_left_nearest() {
        let strategy = Strategy::LeftNearest;
        let interp = setup_1D();
        assert_eq!(interp.interpolate(&[3.00], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[4.00], &strategy).unwrap(), 1.0);
    }

    #[test]
    fn test_1D_right_nearest() {
        let strategy = Strategy::RightNearest;
        let interp = setup_1D();
        assert_eq!(interp.interpolate(&[3.00], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25], &strategy).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00], &strategy).unwrap(), 1.0);
    }

    #[test]
    fn test_1D_nearest() {
        let strategy = Strategy::Nearest;
        let interp = setup_1D();
        assert_eq!(interp.interpolate(&[3.00], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25], &strategy).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.50], &strategy).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[3.75], &strategy).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00], &strategy).unwrap(), 1.0);
    }
}

#[cfg(test)]
mod tests_2D {
    use super::*;

    #[test]
    fn test_2D_linear() {
        let strategy = Strategy::Linear;
        let expected = 0.5;
        let interp = Interpolation::Interp2D(Interp2D {
            x: vec![],
            y: vec![],
            f_xy: vec![],
        });
        assert_eq!(interp.interpolate(&[], &strategy).unwrap(), expected);
        assert!(interp.interpolate(&[], &strategy).is_err());
        assert!(interp.interpolate(&[], &strategy).is_err());
    }
}
