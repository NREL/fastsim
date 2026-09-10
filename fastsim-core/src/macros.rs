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
        impl From<f64> for $enum_ty {
            fn from(value: f64) -> Self {
                Self::Constant(ninterp::prelude::Interp0D(value))
            }
        }

        impl Default for $enum_ty {
            /// Default to 100% efficiency.
            fn default() -> Self {
                Self::from(1.0)
            }
        }

        impl Interpolator<f64> for $enum_ty {
            fn ndim(&self) -> usize {
                ::paste::paste! {
                    match self {
                        $(Self::$variant([<$variant:snake _interp>]) => [<$variant:snake _interp>].ndim(),)+
                    }
                }
            }

            /// Validate efficiency map, ensuring all values are within the range \[0, 1\] and that the underlying interpolator is valid.
            fn validate(&self) -> Result<(), ninterp::error::ValidateError> {
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
                        $(Self::$variant([<$variant:snake _interp>]) => ninterp::prelude::Interpolator::interpolate([<$variant:snake _interp>], point),)+
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

/// Generate a family of `#[serde(serialize_with)]`-compatible helpers that
/// serialize a UOM SI quantity as a specific (non-base) unit.
///
/// Expands to helper functions covering wrapper combinations used in this
/// codebase:
///
/// | Generated function        | Field type                   |
/// |---------------------------|------------------------------|
/// | `$prefix`                 | `$qty_type`                  |
/// | `tracked_$prefix`         | `TrackedState<$qty_type>`    |
/// | `opt_$prefix`             | `Option<$qty_type>`          |
/// | `vec_$prefix`             | `Vec<$qty_type>`             |
/// | `vec_tracked_$prefix`     | `Vec<TrackedState<$qty_type>>` |
/// | `tracked_opt_$prefix`     | `TrackedState<Option<$qty_type>>` |
/// | `opt_tracked_$prefix`     | `Option<TrackedState<$qty_type>>` |
/// | `vec_tracked_opt_$prefix` | `Vec<TrackedState<Option<$qty_type>>>` |
///
/// # Example
///
/// ```ignore
/// // In serde_helpers.rs:
/// use crate::{si, utils::tracked_state::TrackedState};
/// crate::impl_si_serialize_as!(power_as_kilowatts, si::Power, uom::si::power::kilowatt);
///
/// // In serde_api/serde_utils.rs serialize_with match:
/// "Power" => Some("crate::serde_helpers::power_as_kilowatts"),
///
/// // In the unit_impls match (must match the target unit):
/// "Power" => extract_units!(uom::si::power::kilowatt),
/// ```
#[macro_export]
macro_rules! impl_si_serialize_as {
    ($prefix:ident, $qty_type:ty, $unit:path) => {
        ::paste::paste! {
            /// Serialize as `$unit` (plain field).
            pub fn $prefix<S: ::serde::Serializer>(
                val: &$qty_type,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                s.serialize_f64(val.get::<$unit>())
            }

            /// Serialize `TrackedState<$qty_type>` as `$unit`.
            pub fn [<tracked_ $prefix>]<S: ::serde::Serializer>(
                val: &$crate::utils::tracked_state::TrackedState<$qty_type>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                s.serialize_f64(val.inner().get::<$unit>())
            }

            /// Serialize `Option<$qty_type>` as `$unit` (`None` → `null`).
            pub fn [<opt_ $prefix>]<S: ::serde::Serializer>(
                val: &::std::option::Option<$qty_type>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                match val {
                    Some(v) => s.serialize_some(&v.get::<$unit>()),
                    None => s.serialize_none(),
                }
            }

            /// Serialize `Vec<$qty_type>` as `$unit`.
            pub fn [<vec_ $prefix>]<S: ::serde::Serializer>(
                val: &::std::vec::Vec<$qty_type>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                use ::serde::ser::SerializeSeq;
                let mut seq = s.serialize_seq(Some(val.len()))?;
                for v in val {
                    seq.serialize_element(&v.get::<$unit>())?;
                }
                seq.end()
            }

            /// Serialize `Vec<TrackedState<$qty_type>>` as `$unit` (state history vectors).
            pub fn [<vec_tracked_ $prefix>]<S: ::serde::Serializer>(
                val: &::std::vec::Vec<$crate::utils::tracked_state::TrackedState<$qty_type>>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                use ::serde::ser::SerializeSeq;
                let mut seq = s.serialize_seq(Some(val.len()))?;
                for v in val {
                    seq.serialize_element(&v.inner().get::<$unit>())?;
                }
                seq.end()
            }

            /// Serialize `TrackedState<Option<$qty_type>>` as `$unit` (`None` → `null`).
            pub fn [<tracked_opt_ $prefix>]<S: ::serde::Serializer>(
                val: &$crate::utils::tracked_state::TrackedState<::std::option::Option<$qty_type>>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                match val.inner() {
                    Some(v) => s.serialize_some(&v.get::<$unit>()),
                    None => s.serialize_none(),
                }
            }

            /// Serialize `Option<TrackedState<$qty_type>>` as `$unit` (`None` → `null`).
            pub fn [<opt_tracked_ $prefix>]<S: ::serde::Serializer>(
                val: &::std::option::Option<$crate::utils::tracked_state::TrackedState<$qty_type>>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                match val {
                    Some(v) => s.serialize_some(&v.inner().get::<$unit>()),
                    None => s.serialize_none(),
                }
            }

            /// Serialize `Vec<TrackedState<Option<$qty_type>>>` as `$unit` (`None` → `null`).
            pub fn [<vec_tracked_opt_ $prefix>]<S: ::serde::Serializer>(
                val: &::std::vec::Vec<$crate::utils::tracked_state::TrackedState<::std::option::Option<$qty_type>>>,
                s: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                use ::serde::ser::SerializeSeq;
                let mut seq = s.serialize_seq(Some(val.len()))?;
                for v in val {
                    match v.inner() {
                        Some(inner) => seq.serialize_element(&Some(inner.get::<$unit>()))?,
                        None => seq.serialize_element(&::std::option::Option::<f64>::None)?,
                    }
                }
                seq.end()
            }
        }
    };
}
