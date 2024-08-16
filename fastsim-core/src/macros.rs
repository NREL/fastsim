/// Implements `get_eff_max`, `set_eff_max`, `get_eff_min`, and `set_eff_min` methods
#[macro_export]
macro_rules! impl_get_set_eff_max_min {
    () => {
        /// Returns max value of `eff_interp`
        pub fn get_eff_max(&self) -> anyhow::Result<f64> {
            // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
            Ok(self
                .eff_interp_fwd
                .f_x()?
                .iter()
                .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
        }

        /// Scales eff_interp by ratio of new `eff_max` per current calculated max
        pub fn set_eff_max(&mut self, eff_max: f64) -> anyhow::Result<()> {
            if (0.0..=1.0).contains(&eff_max) {
                let old_max = self.get_eff_max()?;
                match &mut self.eff_interp_fwd {
                    Interpolator::Interp1D(interp1d) => {
                        interp1d.f_x = ;
                    },
                    _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed.")
                }
                Ok(())
            } else {
                Err(anyhow!(
                    "`eff_max` ({:.3}) must be between 0.0 and 1.0",
                    eff_max,
                ))
            }
        }

        /// Returns min value of `eff_interp`
        pub fn get_eff_min(&self) -> anyhow::Result<f64> {
            // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
            Ok(self
                .eff_interp_fwd
                .f_x()
                .with_context(|| "eff_interp_fwd does not have f_x field")?
                .iter()
                .fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
        }
    };
}

#[macro_export]
macro_rules! impl_get_set_eff_range {
    () => {
        /// Max value of `eff_interp` minus min value of `eff_interp`.
        pub fn get_eff_range(&self) -> anyhow::Result<f64> {
            Ok(self.get_eff_max()? - self.get_eff_min()?)
        }

        /// Scales values of `eff_interp` without changing max such that max - min
        /// is equal to new range.  Will change max if needed to ensure no values are
        /// less than zero.
        pub fn set_eff_range(&mut self, eff_range: f64) -> anyhow::Result<()> {
            let eff_max = self.get_eff_max()?;
            if eff_range == 0.0 {
                let f_x = vec![
                    eff_max;
                    self.eff_interp_fwd
                        .f_x()
                        .with_context(|| "eff_interp_fwd does not have f_x field")?
                        .len()
                ];
                let eff_interp_fwd = self.eff_interp_fwd;
                eff_interp_fwd.set_f_x(f_x)?;
                self.eff_interp_fwd = eff_interp_fwd;
                Ok(())
            } else if (0.0..=1.0).contains(&eff_range) {
                let old_min = self.get_eff_min()?;
                let old_range = self.get_eff_max()? - old_min;
                if old_range == 0.0 {
                    return Err(anyhow!(
                        "`eff_range` is already zero so it cannot be modified."
                    ));
                }
                self.eff_interp = self
                    .eff_interp_fwd
                    .f_x()
                    .with_context(|| "eff_interp_fwd does not have f_x field")?
                    .iter()
                    .map(|x| eff_max + (x - eff_max) * eff_range / old_range)
                    .collect();
                if self.get_eff_min() < 0.0 {
                    let x_neg = self.get_eff_min();
                    self.eff_interp = self
                        .eff_interp_fwd
                        .f_x()
                        .with_context(|| "eff_interp_fwd does not have f_x field")?
                        .iter()
                        .map(|x| x - x_neg)
                        .collect();
                }
                if self.get_eff_max() > 1.0 {
                    return anyhow!(format!(
                        "`eff_max` ({:.3}) must be no greater than 1.0",
                        self.get_eff_max()
                    ));
                }
                Ok(())
            } else {
                anyhow!(format!(
                    "`eff_range` ({:.3}) must be between 0.0 and 1.0",
                    eff_range,
                ))
            }
        }
    };
}

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
