use crate::imports::*;

/// Methods for proportionally scaling interpolator function data
pub trait InterpolatorMethods {
    fn set_min(&mut self, min: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()>;
    fn set_max(&mut self, max: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()>;
    fn set_range(&mut self, range: f64) -> anyhow::Result<()>;
    // fn get_min(&self) -> anyhow::Result<f64>;
    // fn get_max(&self) -> anyhow::Result<f64>;
    fn range(&self) -> anyhow::Result<f64>;
}

impl InterpolatorMethods for InterpolatorEnumOwned<f64> {
    #[allow(unused)]
    // scale all values so that the min is the new min
    // (Note: this may change the max, depending on what scaling method is chosen)
    fn set_min(&mut self, min: f64, scaling: Option<ScalingMethods>) -> anyhow::Result<()> {
        let scaling = scaling.unwrap_or_default();
        let old_min = *self.min()?;
        match scaling {
            utils::interp::ScalingMethods::Proportional => {
                ensure!(
                    old_min != 0.,
                    "Cannot modify min proportionally when old_min == 0."
                );
                match self {
                    Self::Interp0D(Interp0D(v)) => {
                        *v = min;
                        Ok(())
                    }
                    Self::Interp1D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= min / old_min);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::Interp2D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= min / old_min);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::Interp3D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= min / old_min);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::InterpND(interp) => {
                        interp.data.values.map_inplace(|v| *v *= min / old_min);
                        interp.validate()?;
                        Ok(())
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
        let scaling = scaling.unwrap_or_default();
        let old_max = *self.max()?;
        match scaling {
            utils::interp::ScalingMethods::Proportional => {
                ensure!(
                    old_max != 0.,
                    "Cannot modify max proportionally when old_max == 0."
                );
                match self {
                    Self::Interp0D(Interp0D(v)) => {
                        *v = max;
                        Ok(())
                    }
                    Self::Interp1D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= max / old_max);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::Interp2D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= max / old_max);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::Interp3D(interp) => {
                        interp.data.values.map_inplace(|v| *v *= max / old_max);
                        interp.validate()?;
                        Ok(())
                    }
                    Self::InterpND(interp) => {
                        interp.data.values.map_inplace(|v| *v *= max / old_max);
                        interp.validate()?;
                        Ok(())
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

    fn set_range(&mut self, range: f64) -> anyhow::Result<()> {
        let old_max = self.max()?;
        let old_range = old_max - self.min()?;
        ensure!(old_range != 0., "Cannot modify range when min == max");
        // if the new range is 0., chooses the max as the value for all elements of the array
        if range == 0. {
            match self {
                Self::Interp0D(..) => unreachable!("The above `ensure` should trigger"),
                Self::Interp1D(..) => {
                    Ok(self.set_f_x(self.f_x()?.iter().map(|x| old_max).collect())?)
                }
                Self::Interp2D(..) => Ok(self.set_f_xy(
                    self.f_xy()?
                        .iter()
                        .map(|v| v.iter().map(|x| old_max).collect())
                        .collect(),
                )?),
                Self::Interp3D(..) => Ok(self.set_f_xyz(
                    self.f_xyz()?
                        .iter()
                        .map(|v0| {
                            v0.iter()
                                .map(|v1| v1.iter().map(|x| old_max).collect())
                                .collect()
                        })
                        .collect(),
                )?),
                Self::InterpND(..) => Ok(self.set_values(self.values()?.map(|x| old_max))?),
            }
        } else {
            match self {
                Self::Interp0D(..) => unreachable!("The above `ensure` should trigger"),
                Self::Interp1D(..) => Ok(self.set_f_x(
                    self.f_x()?
                        .iter()
                        .map(|x| old_max + (x - old_max) * range / old_range)
                        .collect(),
                )?),
                Self::Interp2D(..) => Ok(self.set_f_xy(
                    self.f_xy()?
                        .iter()
                        .map(|v| {
                            v.iter()
                                .map(|x| old_max + (x - old_max) * range / old_range)
                                .collect()
                        })
                        .collect(),
                )?),
                Self::Interp3D(..) => Ok(self.set_f_xyz(
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
                Self::InterpND(..) => Ok(self.set_values(
                    self.values()?
                        .map(|x| old_max + (x - old_max) * range / old_range),
                )?),
            }
        }
    }
    // fn get_min(&self) -> anyhow::Result<f64> {
    //     return self.min();
    // }
    // fn get_max(&self) -> anyhow::Result<f64> {
    //     return self.max();
    // }
    fn range(&self) -> anyhow::Result<f64> {
        return Ok(self.max()? - self.min()?);
    }
}

impl<D> Init for InterpolatorEnum<D>
where
    D: ndarray::Data + ndarray::RawDataClone + Clone,
    D::Elem: ninterp::num_traits::Num
        + ninterp::num_traits::Euclid
        + PartialOrd
        + Copy
        + std::fmt::Debug,
{
    fn init(&mut self) -> Result<(), Error> {
        Ok(self
            .validate()
            .map_err(ninterp::error::ValidateError::from)?)
    }
}
impl<D> SerdeAPI for InterpolatorEnum<D>
where
    D: ndarray::Data + ndarray::RawDataClone + Clone + ndarray::DataOwned,
    D::Elem: ninterp::num_traits::Num
        + ninterp::num_traits::Euclid
        + PartialOrd
        + Copy
        + std::fmt::Debug
        + Serialize
        + serde::de::DeserializeOwned,
{
    #[cfg(feature = "resources")]
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
        let x = array![0.05, 0.10, 0.15];
        let y = array![0.10, 0.20, 0.30];
        let z = array![0.20, 0.40, 0.60];
        let f_xy = array![[0.1, 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let f_xyz = array![
            [[0.1, 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ];
        let mut interp_1d = InterpolatorEnum::new_1d(
            array![85.0, 90.0],
            array![0.2, 1.0],
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = InterpolatorEnum::new_2d(
            x.view(),
            y.view(),
            f_xy.view(),
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = InterpolatorEnum::new_3d(
            x.view(),
            y.view(),
            z.view(),
            f_xyz.view(),
            strategy::Linear,
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
        let x = array![0.05, 0.10, 0.15];
        let y = array![0.10, 0.20, 0.30];
        let z = array![0.20, 0.40, 0.60];
        let f_xy = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let f_xyz = array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ];
        let mut interp_1d = InterpolatorEnum::new_1d(
            array![85.0, 90.0],
            array![0.2, 1.0],
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = InterpolatorEnum::new_2d(
            x.view(),
            y.view(),
            f_xy.view(),
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = InterpolatorEnum::new_3d(
            x.view(),
            y.view(),
            z.view(),
            f_xyz.view(),
            strategy::Linear,
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
        let x = array![0.05, 0.10, 0.15];
        let y = array![0.10, 0.20, 0.30];
        let z = array![0.20, 0.40, 0.60];
        let f_xy = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let f_xyz = array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ];
        let mut interp_1d = InterpolatorEnum::new_1d(
            array![85.0, 90.0],
            array![0.2, 1.0],
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_2d = InterpolatorEnum::new_2d(
            x.view(),
            y.view(),
            f_xy.view(),
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .unwrap();
        let mut interp_3d = InterpolatorEnum::new_3d(
            x.view(),
            y.view(),
            z.view(),
            f_xyz.view(),
            strategy::Linear,
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
