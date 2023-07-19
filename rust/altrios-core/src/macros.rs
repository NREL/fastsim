/// Implements `get_eta_max`, `set_eta_max`, `get_eta_min`, and `set_eta_min` methods
#[macro_export]
macro_rules! impl_get_set_eta_max_min {
    () => {
        /// Returns max value of `eta_interp`
        pub fn get_eta_max(&self) -> f64 {
            // since eta is all f64 between 0 and 1, NEG_INFINITY is safe
            self.eta_interp
                .iter()
                .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr))
        }

        /// Scales eta_interp by ratio of new `eta_max` per current calculated max
        pub fn set_eta_max(&mut self, eta_max: f64) -> Result<(), String> {
            if (0.0..=1.0).contains(&eta_max) {
                let old_max = self.get_eta_max();
                self.eta_interp = self
                    .eta_interp
                    .iter()
                    .map(|x| x * eta_max / old_max)
                    .collect();
                Ok(())
            } else {
                Err(format!(
                    "`eta_max` ({:.3}) must be between 0.0 and 1.0",
                    eta_max,
                ))
            }
        }

        /// Returns min value of `eta_interp`
        pub fn get_eta_min(&self) -> f64 {
            // since eta is all f64 between 0 and 1, NEG_INFINITY is safe
            self.eta_interp
                .iter()
                .fold(f64::INFINITY, |acc, curr| acc.min(*curr))
        }
    };
}

#[macro_export]
macro_rules! impl_get_set_eta_range {
    () => {
        /// Max value of `eta_interp` minus min value of `eta_interp`.
        pub fn get_eta_range(&self) -> f64 {
            self.get_eta_max() - self.get_eta_min()
        }

        /// Scales values of `eta_interp` without changing max such that max - min
        /// is equal to new range.  Will change max if needed to ensure no values are
        /// less than zero.
        pub fn set_eta_range(&mut self, eta_range: f64) -> Result<(), String> {
            let eta_max = self.get_eta_max();
            if eta_range == 0.0 {
                self.eta_interp = vec![eta_max; self.eta_interp.len()];
                Ok(())
            } else if (0.0..=1.0).contains(&eta_range) {
                let old_min = self.get_eta_min();
                let old_range = self.get_eta_max() - old_min;
                if old_range == 0.0 {
                    return Err(format!(
                        "`eta_range` is already zero so it cannot be modified."
                    ));
                }
                self.eta_interp = self
                    .eta_interp
                    .iter()
                    .map(|x| eta_max + (x - eta_max) * eta_range / old_range)
                    .collect();
                if self.get_eta_min() < 0.0 {
                    let x_neg = self.get_eta_min();
                    self.eta_interp = self.eta_interp.iter().map(|x| x - x_neg).collect();
                }
                if self.get_eta_max() > 1.0 {
                    return Err(format!(
                        "`eta_max` ({:.3}) must be no greater than 1.0",
                        self.get_eta_max()
                    ));
                }
                Ok(())
            } else {
                Err(format!(
                    "`eta_range` ({:.3}) must be between 0.0 and 1.0",
                    eta_range,
                ))
            }
        }
    };
}

/// Given pairs of arbitrary keys and values, prints "key: value" to python intepreter.  
/// Given str, prints str.  
/// Using this will break `cargo test` but work with `maturin develop`.  
#[macro_export]
macro_rules! print_to_py {
    ( $( $x:expr, $y:expr ),* ) => {
        {
            pyo3::Python::with_gil(|py| {
                let locals = pyo3::types::PyDict::new(py);
                $(
                    locals.set_item($x, $y).unwrap();
                    py.run(
                        &format!("print(f\"{}: {{{}:.3g}}\")", $x, $x),
                        None,
                        Some(locals),
                    )
                    .expect(&format!("printing `{}` failed", $x));
                )*
            });
        };
    };
    ( $x:expr ) => {
        {
            // use pyo3::py_run;
            pyo3::Python::with_gil(|py| {
                    py.run(
                        &format!("print({})", $x),
                        None,
                        None,
                    )
                    .expect(&format!("printing `{}` failed", $x));
            });
        };
    }
}

#[macro_export]
macro_rules! eta_test_body {
    ($component:ident, $eta_max:expr, $eta_min:expr, $eta_range:expr) => {
        assert!(almost_eq($component.get_eta_max(), $eta_max, None));
        assert!(almost_eq($component.get_eta_min(), $eta_min, None));
        assert!(almost_eq($component.get_eta_range(), $eta_range, None));

        $component.set_eta_max(0.9).unwrap();
        assert!(almost_eq($component.get_eta_max(), 0.9, None));
        assert!(almost_eq(
            $component.get_eta_min(),
            $eta_min * 0.9 / $eta_max,
            None
        ));
        assert!(almost_eq(
            $component.get_eta_range(),
            $eta_range * 0.9 / $eta_max,
            None
        ));

        $component.set_eta_range(0.2).unwrap();
        assert!(almost_eq($component.get_eta_max(), 0.9, None));
        assert!(almost_eq($component.get_eta_min(), 0.7, None));
        assert!(almost_eq($component.get_eta_range(), 0.2, None));

        $component.set_eta_range(0.98).unwrap();
        assert!(almost_eq($component.get_eta_max(), 0.98, None));
        assert!(almost_eq($component.get_eta_min(), 0.0, None));
        assert!(almost_eq($component.get_eta_range(), 0.98, None));
    };
}

#[macro_export]
macro_rules! make_assert_cmp_fn {
    ($name:ident) => {
        paste! {
            pub fn [<assert_ $name>]<D, U>(
                val1: f64,
                val2: f64,
                epsilon: Option<f64>,
            ) {
                assert!($name(val1, val2, epsilon),
                    "
                    assert_{} failed.
                    LHS: {val1:?}
                    RHS: {val2:?}",
                    std::stringify!($name)
                );
            }
        }
    };
}

#[macro_export]
macro_rules! make_uom_cmp_fn {
    ($name:ident) => {
        paste! {
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
macro_rules! make_assert_uom_cmp_fn {
    ($name:ident) => {
        paste! {
            pub fn [<assert_ $name _uom>]<D, U>(
                val1: &uom::si::Quantity<D, U, f64>,
                val2: &uom::si::Quantity<D, U, f64>,
                epsilon: Option<f64>,
            )
            where
                D: uom::si::Dimension + ?Sized,
                U: uom::si::Units<f64> + ?Sized,
            {
                assert!($name(val1.value, val2.value, epsilon),
                    "
                    assert_{}_uom failed.
                    LHS: {val1:?}
                    RHS: {val2:?}",
                    std::stringify!($name)
                );
            }
        }
    };
}

#[macro_export]
macro_rules! make_cmp_fns {
    ($name:ident) => {
        make_assert_cmp_fn!($name);
        make_uom_cmp_fn!($name);
        make_assert_uom_cmp_fn!($name);
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
