//! 1-dimensional interpolation

use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Interp1D {
    pub(super) x: Vec<f64>,
    pub(super) f_x: Vec<f64>,
    pub strategy: Strategy,
    #[serde(default)]
    pub extrapolate: Extrapolate,
    /// Phantom private field to prevent direct instantiation in other modules
    #[serde(skip)]
    _phantom: PhantomData<()>,
}

impl Interp1D {
    /// Create and validate 1-D interpolator
    pub fn new(
        x: Vec<f64>,
        f_x: Vec<f64>,
        strategy: Strategy,
        extrapolate: Extrapolate,
    ) -> anyhow::Result<Self> {
        let interp = Self {
            x,
            f_x,
            strategy,
            extrapolate,
            _phantom: PhantomData,
        };
        interp.validate()?;
        Ok(interp)
    }

    pub fn linear(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        // Extrapolate, if applicable
        if matches!(self.extrapolate, Extrapolate::Extrapolate) {
            if point < self.x[0] {
                let slope = (self.f_x[1] - self.f_x[0]) / (self.x[1] - self.x[0]);
                return Ok(slope * (point - self.x[0]) + self.f_x[0]);
            } else if &point > self.x.last().unwrap() {
                let slope = (self.f_x.last().unwrap() - self.f_x[self.f_x.len() - 2])
                    / (self.x.last().unwrap() - self.x[self.x.len() - 2]);
                return Ok(slope * (point - self.x.last().unwrap()) + self.f_x.last().unwrap());
            }
        }
        let lower_index = find_nearest_index(&self.x, point)?;
        let diff = (point - self.x[lower_index]) / (self.x[lower_index + 1] - self.x[lower_index]);
        Ok(self.f_x[lower_index] * (1.0 - diff) + self.f_x[lower_index + 1] * diff)
    }

    pub fn left_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = find_nearest_index(&self.x, point)?;
        Ok(self.f_x[lower_index])
    }

    pub fn right_nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = find_nearest_index(&self.x, point)?;
        Ok(self.f_x[lower_index + 1])
    }

    pub fn nearest(&self, point: f64) -> anyhow::Result<f64> {
        if let Some(i) = self.x.iter().position(|&x_val| x_val == point) {
            return Ok(self.f_x[i]);
        }
        let lower_index = find_nearest_index(&self.x, point)?;
        let diff = (point - self.x[lower_index]) / (self.x[lower_index + 1] - self.x[lower_index]);
        Ok(if diff < 0.5 {
            self.f_x[lower_index]
        } else {
            self.f_x[lower_index + 1]
        })
    }

    /// Function to set x variable from Interp1D
    /// # Arguments
    /// - `new_x`: updated `x` variable to replace the current `x` variable
    pub fn set_x(&mut self, new_x: Vec<f64>) -> anyhow::Result<()> {
        self.x = new_x;
        self.validate()
    }

    /// Function to set f_x variable from Interp1D
    /// # Arguments
    /// - `new_f_x`: updated `f_x` variable to replace the current `f_x` variable
    pub fn set_f_x(&mut self, new_f_x: Vec<f64>) -> anyhow::Result<()> {
        self.f_x = new_f_x;
        self.validate()
    }
}

impl SerdeAPI for Interp1D {}
impl Init for Interp1D {}

impl InterpMethods for Interp1D {
    fn validate(&self) -> anyhow::Result<()> {
        let x_grid_len = self.x.len();

        if matches!(self.extrapolate, Extrapolate::Extrapolate) {
            ensure!(
                matches!(self.strategy, Strategy::Linear),
                "`Extrapolate` is only implemented for 1-D linear, use `Clamp` or `Error` extrapolation strategy instead"
            );
            ensure!(
                self.x.len() >= 2,
                "At least 2 data points are required for extrapolation: x = {:?}, f_x = {:?}",
                self.x,
                self.f_x,
            );
        }

        // Check that each grid dimension has elements
        ensure!(x_grid_len != 0, "Supplied x-coordinates cannot be empty");
        // Check that grid points are monotonically increasing
        ensure!(
            self.x.windows(2).all(|w| w[0] <= w[1]),
            "Supplied x-coordinates must be sorted and non-repeating\n{:?}",
            self.x
        );
        // Check that grid and values are compatible shapes
        ensure!(
            x_grid_len == self.f_x.len(),
            "Supplied grid and values are not compatible shapes, {} {}",
            x_grid_len,
            self.f_x.len()
        );

        Ok(())
    }

    fn interpolate(&self, point: &[f64]) -> anyhow::Result<f64> {
        match self.strategy {
            Strategy::Linear => self.linear(point[0]),
            Strategy::LeftNearest => self.left_nearest(point[0]),
            Strategy::RightNearest => self.right_nearest(point[0]),
            Strategy::Nearest => self.nearest(point[0]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_args() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert!(interp.interpolate(&[]).is_err());
        assert_eq!(interp.interpolate(&[1.0]).unwrap(), 0.4);
    }

    #[test]
    fn test_linear() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_left_nearest() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::LeftNearest,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_right_nearest() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::RightNearest,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_nearest() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Nearest,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_extrapolate_inputs() {
        // Incorrect strategy
        assert!(Interp1D::new(
            vec![0., 1., 2., 3., 4.],
            vec![0.2, 0.4, 0.6, 0.8, 1.0],
            Strategy::Nearest,
            Extrapolate::Extrapolate,
        )
        .is_err());
        // Extrapolate::Error
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Linear,
                Extrapolate::Error,
            )
            .unwrap(),
        );
        assert!(interp.interpolate(&[-1.]).is_err());
        assert!(interp.interpolate(&[5.]).is_err());
    }

    #[test]
    fn test_extrapolate_clamp() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Linear,
                Extrapolate::Clamp,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[-1.]).unwrap(), 0.2);
        assert_eq!(interp.interpolate(&[5.]).unwrap(), 1.0);
    }

    #[test]
    fn test_extrapolate() {
        let interp = Interpolator::Interp1D(
            Interp1D::new(
                vec![0., 1., 2., 3., 4.],
                vec![0.2, 0.4, 0.6, 0.8, 1.0],
                Strategy::Linear,
                Extrapolate::Extrapolate,
            )
            .unwrap(),
        );
        assert_eq!(interp.interpolate(&[-1.]).unwrap(), 0.0);
        assert_eq!(interp.interpolate(&[5.]).unwrap(), 1.2);
    }
}
