//! 3-dimensional interpolation

use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Interp3D {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) z: Vec<f64>,
    pub(super) f_xyz: Vec<Vec<Vec<f64>>>,
    pub strategy: Strategy,
    #[serde(skip)]
    pub extrapolate: Extrapolate,
    #[serde(skip)]
    // phantom private field to prevent direct instantiation in other modules
    _phantom: PhantomData<()>,
}

impl Interp3D {
    /// Create and validate 3-D interpolator
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        f_xyz: Vec<Vec<Vec<f64>>>,
        strategy: Strategy,
        extrapolate: Extrapolate,
    ) -> anyhow::Result<Self> {
        let interp = Self {
            x,
            y,
            z,
            f_xyz,
            strategy,
            extrapolate,
            _phantom: PhantomData,
        };
        interp.validate()?;
        Ok(interp)
    }

    pub fn linear(&self, point: &[f64]) -> anyhow::Result<f64> {
        let x_l = find_nearest_index(&self.x, point[0])?;
        let x_u = x_l + 1;
        let x_diff = (point[0] - self.x[x_l]) / (self.x[x_u] - self.x[x_l]);

        let y_l = find_nearest_index(&self.y, point[1])?;
        let y_u = y_l + 1;
        let y_diff = (point[1] - self.y[y_l]) / (self.y[y_u] - self.y[y_l]);

        let z_l = find_nearest_index(&self.z, point[2])?;
        let z_u = z_l + 1;
        let z_diff = (point[2] - self.z[z_l]) / (self.z[z_u] - self.z[z_l]);

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

    /// Function to set x variable from Interp3D
    /// # Arguments
    /// - `new_x`: updated `x` variable to replace the current `x` variable
    pub fn set_x(&mut self, new_x: Vec<f64>) -> anyhow::Result<()> {
        self.x = new_x;
        self.validate()
    }

    /// Function to set y variable from Interp3D
    /// # Arguments
    /// - `new_y`: updated `y` variable to replace the current `y` variable
    pub fn set_y(&mut self, new_y: Vec<f64>) -> anyhow::Result<()> {
        self.y = new_y;
        self.validate()
    }

    /// Function to set z variable from Interp3D
    /// # Arguments
    /// - `new_z`: updated `z` variable to replace the current `z` variable
    pub fn set_z(&mut self, new_z: Vec<f64>) -> anyhow::Result<()> {
        self.z = new_z;
        self.validate()
    }

    /// Function to set f_xyz variable from Interp3D
    /// # Arguments
    /// - `new_f_xyz`: updated `f_xyz` variable to replace the current `f_xyz` variable
    pub fn set_f_xyz(&mut self, new_f_xyz: Vec<Vec<Vec<f64>>>) -> anyhow::Result<()> {
        self.f_xyz = new_f_xyz;
        self.validate()
    }
}

impl SerdeAPI for Interp3D {}
impl Init for Interp3D {}

impl InterpMethods for Interp3D {
    fn validate(&self) -> anyhow::Result<()> {
        let x_grid_len = self.x.len();
        let y_grid_len = self.y.len();
        let z_grid_len = self.z.len();

        ensure!(!matches!(self.extrapolate, Extrapolate::Extrapolate), "`Extrapolate` is not implemented for 3-D, use `Clamp` or `Error` extrapolation strategy instead");

        // Check that each grid dimension has elements
        ensure!(
            x_grid_len != 0 || y_grid_len != 0 || z_grid_len != 0,
            "Supplied grid coordinates cannot be empty"
        );
        // Check that grid points are monotonically increasing
        ensure!(
            self.x.windows(2).all(|w| w[0] <= w[1])
                && self.y.windows(2).all(|w| w[0] <= w[1])
                && self.z.windows(2).all(|w| w[0] <= w[1]),
            "Supplied coordinates must be sorted and non-repeating"
        );
        // Check that grid and values are compatible shapes
        let x_dim_ok = x_grid_len == self.f_xyz.len();
        let y_dim_ok = self
            .f_xyz
            .iter()
            .map(|y_vals| y_vals.len())
            .all(|y_val_len| y_val_len == y_grid_len);
        let z_dim_ok = self
            .f_xyz
            .iter()
            .flat_map(|y_vals| y_vals.iter().map(|z_vals| z_vals.len()))
            .all(|z_val_len| z_val_len == z_grid_len);
        ensure!(
            x_dim_ok && y_dim_ok && z_dim_ok,
            "Supplied grid and values are not compatible shapes"
        );

        Ok(())
    }

    fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        match self.strategy {
            Strategy::Linear => self.linear(point),
            _ => bail!(
                "Provided strategy {:?} is not applicable for 3-D interpolation",
                self.strategy
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
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
        let interp = Interpolator::Interp3D(
            Interp3D::new(
                x.clone(),
                y.clone(),
                z.clone(),
                f_xyz.clone(),
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        // Check that interpolating at grid points just retrieves the value
        for (i, x_i) in x.iter().enumerate() {
            for (j, y_j) in y.iter().enumerate() {
                for (k, z_k) in z.iter().enumerate() {
                    assert_eq!(
                        interp.interpolate(&[*x_i, *y_j, *z_k]).unwrap(),
                        f_xyz[i][j][k]
                    );
                }
            }
        }
        assert_eq!(
            interp.interpolate(&[x[0], y[0], 0.3]).unwrap(),
            0.4999999999999999 // 0.5
        );
        assert_eq!(
            interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(),
            1.4999999999999996 // 1.5
        );
        assert_eq!(
            interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(),
            1.9999999999999996 // 2.0
        );
        assert_eq!(
            interp.interpolate(&[0.075, y[0], z[0]]).unwrap(),
            4.499999999999999 // 4.5
        );
        assert_eq!(
            interp.interpolate(&[0.075, y[0], 0.3]).unwrap(),
            4.999999999999999 // 5.0
        );
        assert_eq!(
            interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(),
            5.999999999999998 // 6.0
        );
    }

    #[test]
    fn test_linear_offset() {
        let interp = Interpolator::Interp3D(
            Interp3D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![0., 1.],
                vec![
                    vec![vec![0., 1.], vec![2., 3.]],
                    vec![vec![4., 5.], vec![6., 7.]],
                ],
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
        assert!(Interp3D::new(
            vec![0., 1.],
            vec![0., 1.],
            vec![0., 1.],
            vec![
                vec![vec![0., 1.], vec![2., 3.]],
                vec![vec![4., 5.], vec![6., 7.]],
            ],
            Strategy::Linear,
            Extrapolate::Extrapolate,
        )
        .is_err());
        // Extrapolate::Error
        let interp = Interpolator::Interp3D(
            Interp3D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![0., 1.],
                vec![
                    vec![vec![0., 1.], vec![2., 3.]],
                    vec![vec![4., 5.], vec![6., 7.]],
                ],
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
        let interp = Interpolator::Interp3D(
            Interp3D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![0., 1.],
                vec![
                    vec![vec![0., 1.], vec![2., 3.]],
                    vec![vec![4., 5.], vec![6., 7.]],
                ],
                Strategy::Linear,
                Extrapolate::Clamp,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[-1., -1., -1.]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[2., 2., 2.]).unwrap(), 7.);
    }
}
