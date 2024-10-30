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
