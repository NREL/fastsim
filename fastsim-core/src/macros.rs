#[macro_export]
macro_rules! eff_test_body {
    ($component:ident, $eff_max:expr, $eff_min:expr, $eff_range:expr) => {
        assert!(almost_eq($component.get_eff_max(), $eff_max, None));
        assert!(almost_eq($component.get_eff_min(), $eff_min, None));
        assert!(almost_eq($component.get_eff_range(), $eff_range, None));

        $component.set_eff_max(0.9).unwrap();
        assert!(almost_eq($component.get_eff_max(), 0.9, None));
        assert!(almost_eq(
            $component.get_eff_min(),
            $eff_min * 0.9 / $eff_max,
            None
        ));
        assert!(almost_eq(
            $component.get_eff_range(),
            $eff_range * 0.9 / $eff_max,
            None
        ));

        $component.set_eff_range(0.2).unwrap();
        assert!(almost_eq($component.get_eff_max(), 0.9, None));
        assert!(almost_eq($component.get_eff_min(), 0.7, None));
        assert!(almost_eq($component.get_eff_range(), 0.2, None));

        $component.set_eff_range(0.98).unwrap();
        assert!(almost_eq($component.get_eff_max(), 0.98, None));
        assert!(almost_eq($component.get_eff_min(), 0.0, None));
        assert!(almost_eq($component.get_eff_range(), 0.98, None));
    };
}

#[macro_export]
macro_rules! make_uom_cmp_fn {
    ($name:ident) => {
        paste! {
            /// # Arguments
            /// - `val1`: LHS
            /// - `val2`: RHS
            /// - `epsilon`: error threshold, defaults to [crate::utils::COMP_EPSILON]
            pub fn [<$name _uom>]<D, U>(
                val1: &uom::si::Quantity<D, U, f64>,
                val2: &uom::si::Quantity<D, U, f64>,
                epsilon: Option<f64>,
            ) -> bool
            where
                D: uom::si::Dimension + ?Sized,
                U: uom::si::Units<f64> + ?Sized,
            {
                $name(val1.value, val2.value, epsilon)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_efficiency_enum {
    ($enum_ty:ty { $($variant:ident),+ $(,)? }) => {
        impl Interpolator<f64> for $enum_ty {
            fn ndim(&self) -> usize {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].ndim(),)+
                    }
                }
            }

            /// Validate efficiency map, ensuring all values are within the range [0, 1] and that the underlying interpolator is valid.
            fn validate(&mut self) -> Result<(), ninterp::error::ValidateError> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].validate(),)+
                    }
                }?;

                <Self as $crate::utils::interp::InterpolatorScanValues>::try_for_each_value::<ninterp::error::ValidateError, _>(self, |value| {
                    if value.is_nan() {
                        return Ok(());
                    }
                    if !(0.0..=1.0).contains(&value) {
                        return Err(ninterp::error::ValidateError::Other(
                            format!(
                                "Efficiency map contains out-of-range value ({value}); values must be in range [0, 1]"
                            )
                            .into(),
                        ));
                    }
                    Ok(())
                })?;

                Ok(())
            }

            /// Interpolate efficiency map at the given point, ensuring the result is within the range (0, 1] and finite.
            fn interpolate(&self, point: &[f64]) -> Result<f64, ninterp::error::InterpolateError> {
                let efficiency = ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].interpolate(point),)+
                    }
                }?;

                if efficiency.is_nan() {
                    return Err(ninterp::error::InterpolateError::Other(
                        format!(
                            "Efficiency interpolation returned NaN at point {:?}; likely missing local map data",
                            point
                        )
                        .into(),
                    ));
                }
                if !efficiency.is_finite() {
                    return Err(ninterp::error::InterpolateError::Other(
                        format!(
                            "Efficiency interpolation returned non-finite value ({efficiency}) at point {:?}",
                            point
                        )
                        .into(),
                    ));
                }
                if efficiency <= 0.0 || efficiency > 1.0 {
                    return Err(ninterp::error::InterpolateError::Other(
                        format!(
                            "Efficiency interpolation returned out-of-range value ({efficiency}) at point {:?}; expected (0, 1]",
                            point
                        )
                        .into(),
                    ));
                }

                Ok(efficiency)
            }

            fn set_extrapolate(
                &mut self,
                extrapolate: Extrapolate<f64>,
            ) -> Result<(), ninterp::error::ValidateError> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].set_extrapolate(extrapolate),)+
                    }
                }
            }
        }

        impl $crate::utils::interp::InterpolatorScanValues for $enum_ty {
            fn try_for_each_value<E, F: FnMut(f64) -> Result<(), E>>(&self, f: F) -> Result<(), E> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].try_for_each_value(f),)+
                    }
                }
            }
        }

        impl Min<f64> for $enum_ty {
            fn min(&self) -> anyhow::Result<&f64> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].min(),)+
                    }
                }
            }
        }

        impl Max<f64> for $enum_ty {
            fn max(&self) -> anyhow::Result<&f64> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].max(),)+
                    }
                }
            }
        }

        impl Range<f64> for $enum_ty {
            fn range(&self) -> anyhow::Result<f64> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].range(),)+
                    }
                }
            }
        }

        impl $crate::utils::interp::InterpolatorMutMethods for $enum_ty {
            fn set_min(&mut self, min: f64, scaling: Option<$crate::utils::interp::ScalingMethods>) -> anyhow::Result<()> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].set_min(min, scaling),)+
                    }
                }
            }

            fn set_max(&mut self, max: f64, scaling: Option<$crate::utils::interp::ScalingMethods>) -> anyhow::Result<()> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].set_max(max, scaling),)+
                    }
                }
            }

            fn set_range(&mut self, range: f64) -> anyhow::Result<()> {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].set_range(range),)+
                    }
                }
            }
        }
    };
}

#[macro_export]
/// Generates a String similar to output of `dbg` but without printing
macro_rules! format_dbg {
    ($dbg_expr:expr) => {
        format!(
            "[{}:{}] {}: {:?}",
            file!(),
            line!(),
            stringify!($dbg_expr),
            $dbg_expr
        )
    };
    () => {
        format!("[{}:{}]", file!(), line!())
    };
}

#[macro_export]
/// Makes it so that optional parameters get set in the `Init::init` call
macro_rules! init_opt_default {
    ($obj:ident, $fieldname:ident, $def_val:expr) => {
        $obj.$fieldname = $obj.$fieldname.or(Some($def_val));
    };
}

#[macro_export]
/// Times the duration whatever gets passed in
macro_rules! timer {
    ($code_block:expr) => {
        #[cfg(feature = "timer")]
        let now_and_then = Instant::now();
        $code_block;
        #[cfg(feature = "timer")]
        println!(
            "{}\nElapsed time: {} μs",
            format_dbg!(),
            now_and_then.elapsed().as_micros()
        );
    };
}
