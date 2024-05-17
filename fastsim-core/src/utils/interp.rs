use crate::imports::*;

use itertools::Itertools;

#[derive(Debug)]
pub enum Interpolator {
    Interp0D(f64),
    Interp1D(Interp1D),
    Interp2D(Interp2D),
    Interp3D(Interp3D),
    InterpND(InterpND),
}

impl Interpolator {
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
                anyhow::ensure!(
                    interp.x.len() == interp.f_x.len(),
                    "Supplied grid and values are not compatible shapes"
                );
            }
            Self::Interp2D(interp) => {
                anyhow::ensure!(
                    interp.x.len() == interp.f_xy.len() && interp.y.len() == interp.f_xy[0].len(),
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
    pub fn linear(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        let diff = (point - self.x[lower_index]) / (self.x[lower_index + 1] - self.x[lower_index]);
        Ok(self.f_x[lower_index] * (1.0 - diff) + self.f_x[lower_index + 1] * diff)
    }

    pub fn left_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        Ok(self.f_x[lower_index])
    }

    pub fn right_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = self
            .x
            .windows(2)
            .position(|w| w[0] <= point && point < w[1])
            .unwrap();
        Ok(self.f_x[lower_index + 1])
    }

    pub fn nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
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
    pub fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        if let Some(value) = self.try_lower_dim(point)? {
            return Ok(value);
        }

        let x_l = self
            .x
            .windows(2)
            .position(|w| w[0] <= point[0] && point[0] < w[1])
            .unwrap();
        let x_u = x_l + 1;
        let x_diff = (point[0] - self.x[x_l]) / (self.x[x_u] - self.x[x_l]);

        let y_l = self
            .y
            .windows(2)
            .position(|w| w[0] <= point[1] && point[1] < w[1])
            .unwrap();
        let y_u = y_l + 1;
        let y_diff = (point[1] - self.y[y_l]) / (self.y[y_u] - self.y[y_l]);

        // interpolate in the x-direction
        let c0 = self.f_xy[x_l][y_l] * (1.0 - x_diff) + self.f_xy[x_u][y_l] * x_diff;
        let c1 = self.f_xy[x_l][y_u] * (1.0 - x_diff) + self.f_xy[x_u][y_u] * x_diff;

        // interpolate in the y-direction
        Ok(c0 * (1.0 - y_diff) + c1 * y_diff)
    }

    fn try_lower_dim(&self, point: &[f64]) -> anyhow::Result<Option<f64>> {
        // Is point in x grid?
        if let Some(x_idx) = self.x.iter().position(|&x_val| x_val == point[0]) {
            let interp = Interp1D {
                x: self.y.clone(),
                f_x: self.f_xy[x_idx].clone(),
            };
            return Some(interp.linear(point[1])).transpose();
        }
        // Is point in y grid?
        if let Some(y_idx) = self.y.iter().position(|&y_val| y_val == point[1]) {
            let interp = Interp1D {
                x: self.x.clone(),
                f_x: self.f_xy.iter().map(|x| x[y_idx]).collect(),
            };
            return Some(interp.linear(point[0])).transpose();
        }
        Ok(None)
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
    pub fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        if let Some(value) = self.try_lower_dim(point)? {
            return Ok(value);
        }

        let x_l = self
            .x
            .windows(2)
            .position(|w| w[0] <= point[0] && point[0] < w[1])
            .unwrap();
        let x_u = x_l + 1;
        let x_diff = (point[0] - self.x[x_l]) / (self.x[x_u] - self.x[x_l]);

        let y_l = self
            .y
            .windows(2)
            .position(|w| w[0] <= point[1] && point[1] < w[1])
            .unwrap();
        let y_u = y_l + 1;
        let y_diff = (point[1] - self.y[y_l]) / (self.y[y_u] - self.y[y_l]);

        let z_l = self
            .z
            .windows(2)
            .position(|w| w[0] <= point[2] && point[2] < w[1])
            .unwrap();
        let z_u = z_l + 1;
        let z_diff = (point[1] - self.y[z_l]) / (self.z[z_u] - self.z[z_l]);

        // interpolate in the x-direction
        let c00 = self.f_xyz[x_l][y_l][z_l] * (1.0 - x_diff) + self.f_xyz[x_u][y_l][z_l] * x_diff;
        let c01 = self.f_xyz[x_l][y_l][z_u] * (1.0 - x_diff) + self.f_xyz[x_u][y_l][z_u] * x_diff;
        let c10 = self.f_xyz[x_l][y_u][z_l] * (1.0 - x_diff) + self.f_xyz[x_u][y_u][z_l] * x_diff;
        let c11 = self.f_xyz[x_l][y_u][z_u] * (1.0 - x_diff) + self.f_xyz[x_u][y_u][z_u] * x_diff;

        // interpolate in the y-direction
        let c0 = c00 * (1.0 - y_diff) + c10 * y_diff;
        let c1 = c01 * (1.0 - y_diff) + c11 * y_diff;

        // interpolate in the z-direction
        Ok(c0 * (1.0 - z_diff) + c1 * z_diff)
    }

    fn try_lower_dim(&self, point: &[f64]) -> anyhow::Result<Option<f64>> {
        // Is point in x grid?
        if let Some(x_idx) = self.x.iter().position(|&x_val| x_val == point[0]) {
            let interp = Interp2D {
                x: self.y.clone(),
                y: self.z.clone(),
                f_xy: self.f_xyz[x_idx].clone(),
            };
            return Some(interp.linear(&[point[1], point[2]])).transpose();
        }
        // Is point in y grid?
        if let Some(y_idx) = self.y.iter().position(|&y_val| y_val == point[1]) {
            let interp = Interp2D {
                x: self.x.clone(),
                y: self.z.clone(),
                f_xy: self.f_xyz.iter().map(|x| x[y_idx].clone()).collect(),
            };
            return Some(interp.linear(&[point[0], point[2]])).transpose();
        }
        // Is point in z grid?
        if let Some(z_idx) = self.z.iter().position(|&z_val| z_val == point[2]) {
            let interp = Interp2D {
                x: self.x.clone(),
                y: self.y.clone(),
                f_xy: self
                    .f_xyz
                    .iter()
                    .map(|x| x.iter().map(|y| y[z_idx]).collect())
                    .collect(),
            };
            return Some(interp.linear(&[point[0], point[1]])).transpose();
        }
        Ok(None)
    }
}

/// N-dimensional interpolation
#[derive(Debug)]
pub struct InterpND {
    pub grid: Vec<Vec<f64>>,
    pub values: ArrayD<f64>,
}
impl InterpND {
    pub fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        // Dimensionality
        let mut n = self.values.ndim();

        // Point can share up to N values of a grid point, which reduces the problem dimensionality
        // i.e. the point shares one of three values of a 3-D grid point, then the interpolation becomes 2-D at that slice
        // or   if the point shares two of three values of a 3-D grid point, then the interpolation becomes 1-D
        let mut point = point.to_vec();
        let mut grid = self.grid.to_vec();
        let mut values_view = self.values.view();
        for dim in (0..n).rev() {
            // Range is reversed so that removal doesn't affect indexing
            if let Some(pos) = grid[dim]
                .iter()
                .position(|&grid_point| grid_point == point[dim])
            {
                point.remove(dim);
                grid.remove(dim);
                values_view.index_axis_inplace(Axis(dim), pos);
            }
        }
        if values_view.len() == 1 {
            // Supplied point is coincident with a grid point, so just return the value
            return Ok(*values_view.first().unwrap());
        }
        // Simplified dimensionality
        n = values_view.ndim();

        // Extract the lower and upper indices for each dimension,
        // as well as the fraction of how far the supplied point is between the surrounding grid points
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for dim in 0..n {
            let lower_idx = grid[dim]
                .windows(2)
                .position(|w| w[0] < point[dim] && point[dim] < w[1])
                .unwrap();
            let interp_diff = (point[dim] - grid[dim][lower_idx])
                / (grid[dim][lower_idx + 1] - grid[dim][lower_idx]);
            lower_idxs.push(lower_idx);
            interp_diffs.push(interp_diff);
        }
        // `interp_vals` contains all values surrounding the point of interest, starting with shape (2, 2, ...) in N dimensions
        // this gets mutated and reduces in dimension each iteration, filling with the next values to interpolate with
        // this ends up as a 0-dimensional array containing only the final interpolated value
        let mut interp_vals = values_view
            .slice_each_axis(|ax| {
                let lower = lower_idxs[ax.axis.0];
                Slice::from(lower..=lower + 1)
            })
            .to_owned();
        let mut index_permutations = self.get_index_permutations(interp_vals.shape());
        // This loop interpolates in each dimension sequentially
        // each outer loop iteration the dimensionality reduces by 1
        // `interp_vals` ends up as a 0-dimensional array containing only the final interpolated value
        for (dim, diff) in interp_diffs.iter().enumerate() {
            let next_dim = n - 1 - dim;
            let next_shape = vec![2; next_dim];
            // Indeces used for saving results of this dimensions interpolation results
            // assigned to `index_permutations` at end of loop to be used for indexing in next iteration
            let next_idxs = self.get_index_permutations(&next_shape);
            let mut intermediate_arr = Array::default(next_shape);
            for i in 0..next_idxs.len() {
                // `next_idxs` is always half the length of `index_permutations`
                let l = index_permutations[i].as_slice();
                let u = index_permutations[next_idxs.len() + i].as_slice();
                if dim == 0 {
                    anyhow::ensure!(
                            !interp_vals[l].is_nan() && !interp_vals[u].is_nan(),
                            "Surrounding value(s) cannot be NaN:\npoint = {point:?},\ngrid = {grid:?},\nvalues = {:?}",
                            self.values
                        );
                }
                // This calculation happens 2^(n-1) times in the first iteration of the outer loop,
                // 2^(n-2) times in the second iteration, etc.
                intermediate_arr[next_idxs[i].as_slice()] =
                    interp_vals[l] * (1.0 - diff) + interp_vals[u] * diff;
            }
            index_permutations = next_idxs;
            interp_vals = intermediate_arr;
        }

        // return the only value contained within the 0-dimensional array
        Ok(*interp_vals.first().unwrap())
    }

    fn get_index_permutations(&self, shape: &[usize]) -> Vec<Vec<usize>> {
        if shape.is_empty() {
            return vec![vec![]];
        }
        shape
            .iter()
            .map(|&len| 0..len)
            .multi_cartesian_product()
            .collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_0D {
    use super::*;

    #[test]
    fn test_0D() {
        let strategy = Strategy::None;
        let expected = 0.5;
        let interp = Interpolator::Interp0D(expected);
        assert_eq!(interp.interpolate(&[], &strategy).unwrap(), expected);
        assert!(interp.interpolate(&[0.], &strategy).is_err());
        assert!(interp.interpolate(&[], &strategy).is_err());
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_1D {
    use super::*;

    fn setup_1D() -> Interpolator {
        // f(x) = 0.2x + 0.2
        Interpolator::Interp1D(Interp1D {
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_2D {
    use super::*;

    #[test]
    fn test_2D_linear() {
        let strategy = Strategy::Linear;
        let x = vec![0.05, 0.10, 0.15];
        let y = vec![0.10, 0.20, 0.30];
        let f_xy = vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]];
        let interp = Interpolator::Interp2D(Interp2D {
            x: x.clone(),
            y: y.clone(),
            f_xy: f_xy.clone(),
        });
        assert_eq!(interp.interpolate(&[x[2], y[1]], &strategy).unwrap(), 7.);
        assert_eq!(interp.interpolate(&[x[2], y[1]], &strategy).unwrap(), 7.);
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_3D {
    use super::*;

    #[test]
    fn test_3D_linear() {
        let strategy: Strategy = Strategy::Linear;
        let x = vec![0.05, 0.10, 0.15];
        let y = vec![0.10, 0.20, 0.30];
        let z = vec![0.20, 0.40, 0.60];
        let f_xyz = vec![
            vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]],
            vec![vec![9., 10., 11.], vec![12., 13., 14.], vec![15., 16., 17.]],
            vec![
                vec![18., 19., 20.],
                vec![21., 22., 23.],
                vec![24., 25., 26.],
            ],
        ];
        let interp = Interpolator::Interp3D(Interp3D {
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
            f_xyz: f_xyz.clone(),
        });
        // Check that interpolating at grid points just retrieves the value
        for i in 0..x.len() {
            for j in 0..y.len() {
                for k in 0..z.len() {
                    assert_eq!(
                        interp.interpolate(&[x[i], y[j], z[k]], &strategy).unwrap(),
                        f_xyz[i][j][k]
                    );
                }
            }
        }
        assert_eq!(
            interp.interpolate(&[x[0], y[0], 0.3], &strategy).unwrap(),
            0.4999999999999999 // 0.5
        );
        assert_eq!(
            interp.interpolate(&[x[0], 0.15, z[0]], &strategy).unwrap(),
            1.4999999999999996 // 1.5
        );
        assert_eq!(
            interp.interpolate(&[x[0], 0.15, 0.3], &strategy).unwrap(),
            1.9999999999999996 // 2.0
        );
        assert_eq!(
            interp.interpolate(&[0.075, y[0], z[0]], &strategy).unwrap(),
            4.499999999999999 // 4.5
        );
        assert_eq!(
            interp.interpolate(&[0.075, y[0], 0.3], &strategy).unwrap(),
            4.999999999999999 // 5.0
        );
        assert_eq!(
            interp.interpolate(&[0.075, 0.15, z[0]], &strategy).unwrap(),
            5.999999999999998 // 6.0
        );
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests_ND {
    use super::*;

    #[test]
    fn test_ND_linear() {
        let strategy: Strategy = Strategy::Linear;
        let grid = vec![
            vec![0.05, 0.10, 0.15],
            vec![0.10, 0.20, 0.30],
            vec![0.20, 0.40, 0.60],
        ];
        let f_xyz = array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ]
        .into_dyn();
        let interp = Interpolator::InterpND(InterpND {
            grid: grid.clone(),
            values: f_xyz.clone(),
        });
        // Check that interpolating at grid points just retrieves the value
        for i in 0..grid[0].len() {
            for j in 0..grid[1].len() {
                for k in 0..grid[2].len() {
                    assert_eq!(
                        &interp
                            .interpolate(&[grid[0][i], grid[1][j], grid[2][k]], &strategy)
                            .unwrap(),
                        f_xyz.slice(s![i, j, k])
                            .first()
                            .unwrap()
                    );
                }
            }
        }
        assert_eq!(
            interp
                .interpolate(&[grid[0][0], grid[1][0], 0.3], &strategy)
                .unwrap(),
            0.4999999999999999 // 0.5
        );
        assert_eq!(
            interp
                .interpolate(&[grid[0][0], 0.15, grid[2][0]], &strategy)
                .unwrap(),
            1.4999999999999996 // 1.5
        );
        assert_eq!(
            interp
                .interpolate(&[grid[0][0], 0.15, 0.3], &strategy)
                .unwrap(),
            1.9999999999999996 // 2.0
        );
        assert_eq!(
            interp
                .interpolate(&[0.075, grid[1][0], grid[2][0]], &strategy)
                .unwrap(),
            4.499999999999999 // 4.5
        );
        assert_eq!(
            interp
                .interpolate(&[0.075, grid[1][0], 0.3], &strategy)
                .unwrap(),
            4.999999999999999 // 5.0
        );
        assert_eq!(
            interp
                .interpolate(&[0.075, 0.15, grid[2][0]], &strategy)
                .unwrap(),
            5.999999999999998 // 6.0
        );
    }
}
