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
