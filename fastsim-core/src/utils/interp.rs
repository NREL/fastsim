use crate::imports::*;

/// Methods for proportionally scaling interpolator function data
pub trait InterpolatorMethods {
    fn set_min(&mut self, min: f64) -> anyhow::Result<()>;
    fn set_max(&mut self, max: f64) -> anyhow::Result<()>;
    fn set_range(&mut self, range: f64) -> anyhow::Result<()>;
}

impl InterpolatorMethods for Interpolator {
    #[allow(unused)]
    fn set_min(&mut self, min: f64) -> anyhow::Result<()> {
        let old_min = self.min()?;
        match self {
            Interpolator::Interp0D(value) => {
                *value = min;
                Ok(())
            }
            Interpolator::Interp1D(..) => {
                todo!()
            }
            Interpolator::Interp2D(..) => {
                todo!()
            }
            Interpolator::Interp3D(..) => {
                todo!()
            }
            Interpolator::InterpND(..) => {
                todo!()
            }
        }
    }

    fn set_max(&mut self, max: f64) -> anyhow::Result<()> {
        let old_max = self.max()?;
        match self {
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
}

impl Init for Interpolator {
    fn init(&mut self) -> Result<(), FsimError> {
        Ok(self.validate().map_err(ninterp::Error::from)?)
    }
}
impl SerdeAPI for Interpolator {
    const RESOURCE_PREFIX: &'static str = "interpolators";
}
