use crate::imports::*;

/// Methods for proportionally scaling interpolator function data
pub trait InterpolatorMethods {
    fn set_min(&mut self, min: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()>;
    fn set_max(&mut self, max: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()>;
    fn set_range(&mut self, range: f64) -> anyhow::Result<()>;
    fn get_min(&self) -> anyhow::Result<f64>;
    fn get_max(&self) -> anyhow::Result<f64>;
    fn get_range(&self) -> anyhow::Result<f64>;
}

impl InterpolatorMethods for Interpolator {
    #[allow(unused)]
    // scale all values so that the min is the new min
    // (Note: this may change the max, depending on what scaling method is chosen)
    fn set_min(&mut self, min: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()> {
        let scaling = scaling.unwrap_or_default();
        let old_min = self.min()?;
        match scaling {
            utils::interp::ScalingMethods::Proportional => {
                ensure!(
                    old_min != 0.,
                    "Cannot modify min using default calculation when min = 0."
                );
                match self {
                    Interpolator::Interp0D(value) => {
                        *value = min;
                        Ok(())
                    }
                    Interpolator::Interp1D(..) => {
                        Ok(self.set_f_x(self.f_x()?.iter().map(|x| x * min / old_min).collect())?)
                    }
                    Interpolator::Interp2D(..) => Ok(self.set_f_xy(
                        self.f_xy()?
                            .iter()
                            .map(|v| v.iter().map(|x| x * min / old_min).collect())
                            .collect(),
                    )?),
                    Interpolator::Interp3D(..) => Ok(self.set_f_xyz(
                        self.f_xyz()?
                            .iter()
                            .map(|v0| {
                                v0.iter()
                                    .map(|v1| v1.iter().map(|x| x * min / old_min).collect())
                                    .collect()
                            })
                            .collect(),
                    )?),
                    Interpolator::InterpND(..) => {
                        Ok(self.set_values(self.values()?.map(|x| x * min / old_min))?)
                    }
                }
            }
            utils::interp::ScalingMethods::AnchoredProportional => {
                todo!()
            }
            utils::interp::ScalingMethods::Offset => {
                todo!()
            }
        }
    }

    // scale all values so that the max is the new max
    // (Note: may change the min, depending on what scaling method is chosen)
    fn set_max(&mut self, max: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()> {
        let old_max = self.max()?;
        let scaling = scaling.unwrap_or_default();
        match scaling {
            utils::interp::ScalingMethods::Proportional => match self {
                Interpolator::Interp0D(value) => {
                    *value = max;
                    Ok(())
                }
                Interpolator::Interp1D(..) => {
                    Ok(self.set_f_x(self.f_x()?.iter().map(|x| x * max / old_max).collect())?)
                }
                Interpolator::Interp2D(..) => Ok(self.set_f_xy(
                    self.f_xy()?
                        .iter()
                        .map(|v| v.iter().map(|x| x * max / old_max).collect())
                        .collect(),
                )?),
                Interpolator::Interp3D(..) => Ok(self.set_f_xyz(
                    self.f_xyz()?
                        .iter()
                        .map(|v0| {
                            v0.iter()
                                .map(|v1| v1.iter().map(|x| x * max / old_max).collect())
                                .collect()
                        })
                        .collect(),
                )?),
                Interpolator::InterpND(..) => {
                    Ok(self.set_values(self.values()?.map(|x| x * max / old_max))?)
                }
            },
            utils::interp::ScalingMethods::AnchoredProportional => {
                todo!()
            }
            utils::interp::ScalingMethods::Offset => {
                todo!()
            }
        }
    }

    fn set_range(&mut self, range: f64) -> anyhow::Result<()> {
        let old_max = self.max()?;
        let old_range = old_max - self.min()?;
        ensure!(old_range != 0., "Cannot modify range when min == max");
        match self {
            Interpolator::Interp0D(..) => unreachable!("The above `ensure` should trigger"),
            Interpolator::Interp1D(..) => Ok(self.set_f_x(
                self.f_x()?
                    .iter()
                    .map(|x| old_max + (x - old_max) * range / old_range)
                    .collect(),
            )?),
            Interpolator::Interp2D(..) => Ok(self.set_f_xy(
                self.f_xy()?
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|x| old_max + (x - old_max) * range / old_range)
                            .collect()
                    })
                    .collect(),
            )?),
            Interpolator::Interp3D(..) => Ok(self.set_f_xyz(
                self.f_xyz()?
                    .iter()
                    .map(|v0| {
                        v0.iter()
                            .map(|v1| {
                                v1.iter()
                                    .map(|x| old_max + (x - old_max) * range / old_range)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            )?),
            Interpolator::InterpND(..) => Ok(self.set_values(
                self.values()?
                    .map(|x| old_max + (x - old_max) * range / old_range),
            )?),
        }
    }
    fn get_min(&self) -> anyhow::Result<f64> {
        return self.min();
    }
    fn get_max(&self) -> anyhow::Result<f64> {
        return self.max();
    }
    fn get_range(&self) -> anyhow::Result<f64> {
        return Ok(self.max()? - self.min()?);
    }
}

impl Init for Interpolator {
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(self.validate()?)
    }
}
impl SerdeAPI for Interpolator {
    const RESOURCE_PREFIX: &'static str = "interpolators";
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub enum ScalingMethods {
    #[default]
    /// Scales everything by the same factor -- e.g. setting min of [1, 2, 3] to 0.5 yields [0.5, 1, 1.5]
    Proportional,
    /// Scales proportionally to distance from min/max -- e.g. setting min of [1, 2, 3] to 0.5 yields [0.5, 1.5, 3]
    AnchoredProportional,
    /// Scaling by sliding all values up or down by the same offset -- e.g. setting min of [1, 2, 3] to 0.5 yields [0.5, 1.5, 2.5]
    Offset,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_min() {
        let x: Vec<f64> = vec![0.05, 0.10, 0.15];
        let y: Vec<f64> = vec![0.10, 0.20, 0.30];
        let z: Vec<f64> = vec![0.20, 0.40, 0.60];
        let f_xy: Vec<Vec<f64>> = vec![vec![0.1, 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]];
        let f_xyz: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.1, 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]],
            vec![vec![9., 10., 11.], vec![12., 13., 14.], vec![15., 16., 17.]],
            vec![
                vec![18., 19., 20.],
                vec![21., 22., 23.],
                vec![24., 25., 26.],
            ],
        ];
        let mut interp_1d = Interpolator::new_1d(
            vec![85.0, 90.0],
            vec![0.2, 1.0],
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = Interpolator::new_2d(
            x.clone(),
            y.clone(),
            f_xy,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = Interpolator::new_3d(
            x.clone(),
            y.clone(),
            z.clone(),
            f_xyz,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        assert_eq!(interp_1d.get_min().unwrap(), 0.2);
        assert_eq!(interp_2d.get_min().unwrap(), 0.1);
        assert_eq!(interp_3d.get_min().unwrap(), 0.1);
        interp_1d.set_min(0.1, None).unwrap();
        interp_2d.set_min(0.3, None).unwrap();
        interp_3d.set_min(0.3, None).unwrap();
        println!("{:?}", interp_1d.get_min().unwrap());
        println!("{:?}", interp_2d.get_min().unwrap());
        println!("{:?}", interp_3d.get_min().unwrap());
        assert!(almost_eq(interp_1d.get_min().unwrap(), 0.1, Some(1e-3)));
        assert!(almost_eq(interp_2d.get_min().unwrap(), 0.3, Some(1e-3)));
        assert!(almost_eq(interp_3d.get_min().unwrap(), 0.3, Some(1e-3)));
    }

    #[test]
    fn test_max() {
        let x: Vec<f64> = vec![0.05, 0.10, 0.15];
        let y: Vec<f64> = vec![0.10, 0.20, 0.30];
        let z: Vec<f64> = vec![0.20, 0.40, 0.60];
        let f_xy: Vec<Vec<f64>> = vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]];
        let f_xyz: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]],
            vec![vec![9., 10., 11.], vec![12., 13., 14.], vec![15., 16., 17.]],
            vec![
                vec![18., 19., 20.],
                vec![21., 22., 23.],
                vec![24., 25., 26.],
            ],
        ];
        let mut interp_1d = Interpolator::new_1d(
            vec![85.0, 90.0],
            vec![0.2, 1.0],
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = Interpolator::new_2d(
            x.clone(),
            y.clone(),
            f_xy,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = Interpolator::new_3d(
            x.clone(),
            y.clone(),
            z.clone(),
            f_xyz,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        assert_eq!(interp_1d.get_max().unwrap(), 1.0);
        assert_eq!(interp_2d.get_max().unwrap(), 8.);
        assert_eq!(interp_3d.get_max().unwrap(), 26.);
        interp_1d.set_max(2., None).unwrap();
        interp_2d.set_max(7., None).unwrap();
        interp_3d.set_max(5., None).unwrap();
        println!("{:?}", interp_1d.get_max().unwrap());
        println!("{:?}", interp_2d.get_max().unwrap());
        println!("{:?}", interp_3d.get_max().unwrap());
        assert!(almost_eq(interp_1d.get_max().unwrap(), 2., Some(1e-3)));
        assert!(almost_eq(interp_2d.get_max().unwrap(), 7., Some(1e-3)));
        assert!(almost_eq(interp_3d.get_max().unwrap(), 5., Some(1e-3)));
    }

    #[test]
    fn test_range() {
        let x: Vec<f64> = vec![0.05, 0.10, 0.15];
        let y: Vec<f64> = vec![0.10, 0.20, 0.30];
        let z: Vec<f64> = vec![0.20, 0.40, 0.60];
        let f_xy: Vec<Vec<f64>> = vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]];
        let f_xyz: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0., 1., 2.], vec![3., 4., 5.], vec![6., 7., 8.]],
            vec![vec![9., 10., 11.], vec![12., 13., 14.], vec![15., 16., 17.]],
            vec![
                vec![18., 19., 20.],
                vec![21., 22., 23.],
                vec![24., 25., 26.],
            ],
        ];
        let mut interp_1d = Interpolator::new_1d(
            vec![85.0, 90.0],
            vec![0.2, 1.0],
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = Interpolator::new_2d(
            x.clone(),
            y.clone(),
            f_xy,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = Interpolator::new_3d(
            x.clone(),
            y.clone(),
            z.clone(),
            f_xyz,
            Strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        assert_eq!(interp_1d.get_range().unwrap(), 0.8);
        assert_eq!(interp_2d.get_range().unwrap(), 8.);
        assert_eq!(interp_3d.get_range().unwrap(), 26.);
        interp_1d.set_range(2.).unwrap();
        interp_2d.set_range(7.).unwrap();
        interp_3d.set_range(5.).unwrap();
        println!("{:?}", interp_1d.get_range().unwrap());
        println!("{:?}", interp_2d.get_range().unwrap());
        println!("{:?}", interp_3d.get_range().unwrap());
        assert!(almost_eq(interp_1d.get_range().unwrap(), 2., Some(1e-3)));
        assert!(almost_eq(interp_2d.get_range().unwrap(), 7., Some(1e-3)));
        assert!(almost_eq(interp_3d.get_range().unwrap(), 5., Some(1e-3)));
    }
}
