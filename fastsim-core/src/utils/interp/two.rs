//! 2-dimensional interpolation

use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Interp2D {
    pub(super) x: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) f_xy: Vec<Vec<f64>>,
    pub strategy: Strategy,
    #[serde(default)]
    pub extrapolate: Extrapolate,
    /// Phantom private field to prevent direct instantiation in other modules
    #[serde(skip)]
    _phantom: PhantomData<()>,
}

impl Interp2D {
    /// Create and validate 2-D interpolator
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        f_xy: Vec<Vec<f64>>,
        strategy: Strategy,
        extrapolate: Extrapolate,
    ) -> anyhow::Result<Self> {
        let interp = Self {
            x,
            y,
            f_xy,
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

        // interpolate in the x-direction
        let c0 = self.f_xy[x_l][y_l] * (1.0 - x_diff) + self.f_xy[x_u][y_l] * x_diff;
        let c1 = self.f_xy[x_l][y_u] * (1.0 - x_diff) + self.f_xy[x_u][y_u] * x_diff;

        // interpolate in the y-direction
        Ok(c0 * (1.0 - y_diff) + c1 * y_diff)
    }

    /// Function to set x variable from Interp2D
    /// # Arguments
    /// - `new_x`: updated `x` variable to replace the current `x` variable
    pub fn set_x(&mut self, new_x: Vec<f64>) -> anyhow::Result<()> {
        self.x = new_x;
        self.validate()
    }

    /// Function to set y variable from Interp2D
    /// # Arguments
    /// - `new_y`: updated `y` variable to replace the current `y` variable
    pub fn set_y(&mut self, new_y: Vec<f64>) -> anyhow::Result<()> {
        self.y = new_y;
        self.validate()
    }

    /// Function to set f_xy variable from Interp2D
    /// # Arguments
    /// - `new_f_xy`: updated `f_xy` variable to replace the current `f_xy` variable
    pub fn set_f_xy(&mut self, new_f_xy: Vec<Vec<f64>>) -> anyhow::Result<()> {
        self.f_xy = new_f_xy;
        self.validate()
    }
}

impl SerdeAPI for Interp2D {}
impl Init for Interp2D {}

impl InterpMethods for Interp2D {
    fn validate(&self) -> anyhow::Result<()> {
        let x_grid_len = self.x.len();
        let y_grid_len = self.y.len();

        ensure!(!matches!(self.extrapolate, Extrapolate::Extrapolate), "`Extrapolate` is not implemented for 2-D, use `Clamp` or `Error` extrapolation strategy instead");

        // Check that each grid dimension has elements
        ensure!(
            x_grid_len != 0 && y_grid_len != 0,
            "Supplied grid coordinates cannot be empty"
        );
        // Check that grid points are monotonically increasing
        ensure!(
            self.x.windows(2).all(|w| w[0] <= w[1]) && self.y.windows(2).all(|w| w[0] <= w[1]),
            "Supplied coordinates must be sorted and non-repeating"
        );
        // Check that grid and values are compatible shapes
        let x_dim_ok = x_grid_len == self.f_xy.len();
        let y_dim_ok = self
            .f_xy
            .iter()
            .map(|y_vals| y_vals.len())
            .all(|y_val_len| y_val_len == y_grid_len);
        ensure!(
            x_dim_ok && y_dim_ok,
            "Supplied grid and values are not compatible shapes"
        );

        Ok(())
    }

    fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        match self.strategy {
            Strategy::Linear => self.linear(point),
            _ => bail!(
                "Provided strategy {:?} is not applicable for 2-D interpolation",
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
        let f_xy = vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]];
        let interp = Interpolator::Interp2D(
            Interp2D::new(
                x.clone(),
                y.clone(),
                f_xy.clone(),
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[x[2], y[1]]).unwrap(), 7.);
        assert_eq!(interp.interpolate(&[x[2], y[1]]).unwrap(), 7.);
    }

    #[test]
    fn test_linear_offset() {
        let interp = Interpolator::Interp2D(
            Interp2D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![vec![0., 1.], vec![2., 3.]],
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        let interp_res = interp.interpolate(&[0.25, 0.65]).unwrap();
        assert_eq!(interp_res, 1.1500000000000001) // 1.15
    }

    #[test]
    fn test_extrapolate_inputs() {
        // Extrapolate::Extrapolate
        assert!(Interp2D::new(
            vec![0., 1.],
            vec![0., 1.],
            vec![vec![0., 1.], vec![2., 3.]],
            Strategy::Linear,
            Extrapolate::Extrapolate,
        )
        .is_err());
        // Extrapolate::Error
        let interp = Interpolator::Interp2D(
            Interp2D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![vec![0., 1.], vec![2., 3.]],
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert!(interp.interpolate(&[-1.]).is_err());
        assert!(interp.interpolate(&[2.]).is_err());
    }

    #[test]
    fn test_extrapolate_clamp() {
        let interp = Interpolator::Interp2D(
            Interp2D::new(
                vec![0., 1.],
                vec![0., 1.],
                vec![vec![0., 1.], vec![2., 3.]],
                Strategy::Linear,
                Extrapolate::Clamp,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[-1., -1.]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[2., 2.]).unwrap(), 3.);
    }
}
