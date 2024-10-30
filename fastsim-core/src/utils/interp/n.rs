//! N-dimensional interpolation

use super::*;

use itertools::Itertools;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InterpND {
    pub(super) grid: Vec<Vec<f64>>,
    pub(super) values: ArrayD<f64>,
    pub strategy: Strategy,
    #[serde(default)]
    pub extrapolate: Extrapolate,
    /// Phantom private field to prevent direct instantiation in other modules
    #[serde(skip)]
    _phantom: PhantomData<()>,
}

impl InterpND {
    /// Create and validate 2-D interpolator
    pub fn new(
        grid: Vec<Vec<f64>>,
        values: ArrayD<f64>,
        strategy: Strategy,
        extrapolate: Extrapolate,
    ) -> anyhow::Result<Self> {
        let interp = Self {
            grid,
            values,
            strategy,
            extrapolate,
            _phantom: PhantomData,
        };
        interp.validate()?;
        Ok(interp)
    }

    /// Interpolator dimensionality
    pub fn ndim(&self) -> usize {
        if self.values.len() == 1 {
            0
        } else {
            self.values.ndim()
        }
    }

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
            return values_view.first().copied().with_context(|| {
                "Could not extract value (on grid) during multilinear interpolation"
            });
        }
        // Simplified dimensionality
        n = values_view.ndim();

        // Extract the lower and upper indices for each dimension,
        // as well as the fraction of how far the supplied point is between the surrounding grid points
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for dim in 0..n {
            let lower_idx = find_nearest_index(&grid[dim], point[dim])?;
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
                    ensure!(
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
        interp_vals
            .first()
            .copied()
            .with_context(|| "Could not extract value during multilinear interpolation")
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

    /// Function to set grid variable from InterpND
    /// # Arguments
    /// - `new_grid`: updated `grid` variable to replace the current `grid` variable
    pub fn set_grid(&mut self, new_grid: Vec<Vec<f64>>) -> anyhow::Result<()> {
        self.grid = new_grid;
        self.validate()
    }

    /// Function to set grid x variable from InterpND
    /// # Arguments
    /// - `new_x`: updated `grid[0]` to replace the current `grid[0]`
    pub fn set_grid_x(&mut self, new_grid_x: Vec<f64>) -> anyhow::Result<()> {
        self.grid[0] = new_grid_x;
        self.validate()
    }

    /// Function to set grid y variable from InterpND
    /// # Arguments
    /// - `new_y`: updated `grid[1]` to replace the current `grid[1]`
    pub fn set_grid_y(&mut self, new_grid_y: Vec<f64>) -> anyhow::Result<()> {
        self.grid[1] = new_grid_y;
        self.validate()
    }

    /// Function to set grid z variable from InterpND
    /// # Arguments
    /// - `new_z`: updated `grid[2]` to replace the current `grid[2]`
    pub fn set_grid_z(&mut self, new_grid_z: Vec<f64>) -> anyhow::Result<()> {
        self.grid[2] = new_grid_z;
        self.validate()
    }

    /// Function to set values variable from InterpND
    /// # Arguments
    /// - `new_values`: updated `values` variable to replace the current `values` variable
    pub fn set_values(&mut self, new_values: ArrayD<f64>) -> anyhow::Result<()> {
        self.values = new_values;
        self.validate()
    }
}

impl SerdeAPI for InterpND {}
impl Init for InterpND {}

impl InterpMethods for InterpND {
    fn validate(&self) -> anyhow::Result<()> {
        let n = self.ndim();
        ensure!(
            !matches!(self.extrapolate, Extrapolate::Extrapolate),
            "`Extrapolate` is not implemented for N-D, use `Clamp` or `Error` extrapolation strategy instead"
        );

        // Check that each grid dimension has elements
        for i in 0..n {
            // Indexing `grid` directly is okay because `grid == vec![]` is caught at compilation
            ensure!(
                !self.grid[i].is_empty(),
                "Supplied `grid` coordinates cannot be empty: dimension {i}, {:?}",
                self.grid[i]
            );
        }
        // Check that grid points are monotonically increasing
        for i in 0..n {
            ensure!(
                self.grid[i].windows(2).all(|w| w[0] <= w[1]),
                "Supplied `grid` coordinates must be sorted and non-repeating: dimension {i}, {:?}",
                self.grid[i]
            );
        }
        // Check that grid and values are compatible shapes
        for i in 0..n {
            ensure!(
                self.grid[i].len() == self.values.shape()[i],
                "Supplied grid and values are not compatible shapes: dimension {i}, lengths {} != {}",
                self.grid[i].len(),
                self.values.shape()[i]);
        }
        // Check grid dimensionality
        let grid_len = if self.grid[0].is_empty() {
            0
        } else {
            self.grid.len()
        };
        ensure!(
            grid_len == n,
            "Length of supplied `grid` must be same as `values` dimensionality: {:?} is not {n}-dimensional",
            self.grid
        );

        Ok(())
    }

    fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        match self.strategy {
            Strategy::Linear => self.linear(point),
            _ => bail!(
                "Provided strategy {:?} is not applicable for {}-D interpolation",
                self.strategy,
                self.ndim()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
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
        let interp = Interpolator::InterpND(
            InterpND::new(
                grid.clone(),
                f_xyz.clone(),
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        // Check that interpolating at grid points just retrieves the value
        for i in 0..grid[0].len() {
            for j in 0..grid[1].len() {
                for k in 0..grid[2].len() {
                    assert_eq!(
                        &interp
                            .interpolate(&[grid[0][i], grid[1][j], grid[2][k]])
                            .unwrap(),
                        f_xyz.slice(s![i, j, k]).first().unwrap()
                    );
                }
            }
        }
        assert_eq!(
            interp.interpolate(&[grid[0][0], grid[1][0], 0.3]).unwrap(),
            0.4999999999999999 // 0.5
        );
        assert_eq!(
            interp.interpolate(&[grid[0][0], 0.15, grid[2][0]]).unwrap(),
            1.4999999999999996 // 1.5
        );
        assert_eq!(
            interp.interpolate(&[grid[0][0], 0.15, 0.3]).unwrap(),
            1.9999999999999996 // 2.0
        );
        assert_eq!(
            interp
                .interpolate(&[0.075, grid[1][0], grid[2][0]])
                .unwrap(),
            4.499999999999999 // 4.5
        );
        assert_eq!(
            interp.interpolate(&[0.075, grid[1][0], 0.3]).unwrap(),
            4.999999999999999 // 5.0
        );
        assert_eq!(
            interp.interpolate(&[0.075, 0.15, grid[2][0]]).unwrap(),
            5.999999999999998 // 6.0
        );
    }

    #[test]
    fn test_linear_offset() {
        let interp = Interpolator::InterpND(
            InterpND::new(
                vec![vec![0., 1.], vec![0., 1.], vec![0., 1.]],
                array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(
            interp.interpolate(&[0.25, 0.65, 0.9]).unwrap(),
            3.1999999999999997
        ) // 3.2
    }

    #[test]
    fn test_extrapolate_inputs() {
        // Extrapolate::Extrapolate
        assert!(InterpND::new(
            vec![vec![0., 1.], vec![0., 1.], vec![0., 1.]],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
            Strategy::Linear,
            Extrapolate::Extrapolate,
        )
        .is_err());
        // Extrapolate::Error
        let interp = Interpolator::InterpND(
            InterpND::new(
                vec![vec![0., 1.], vec![0., 1.], vec![0., 1.]],
                array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert!(interp.interpolate(&[-1., -1., -1.]).is_err());
        assert!(interp.interpolate(&[2., 2., 2.]).is_err());
    }

    #[test]
    fn test_extrapolate_clamp() {
        let interp = Interpolator::InterpND(
            InterpND::new(
                vec![vec![0., 1.], vec![0., 1.], vec![0., 1.]],
                array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
                Strategy::Linear,
                Extrapolate::Clamp,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[-1., -1., -1.]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[2., 2., 2.]).unwrap(), 7.);
    }
}
