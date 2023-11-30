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
macro_rules! check_orphaned_and_set {
    ($struct_self: ident, $field: ident, $value: expr) => {
        // TODO: This seems like it could be used instead, but raises an error
        // ensure!(!$struct_self.orphaned, utils::NESTED_STRUCT_ERR);
        // $struct_self.$field = $value;
        // Ok(())

        if !$struct_self.orphaned {
            $struct_self.$field = $value;
            anyhow::Ok(())
        } else {
            bail!(utils::NESTED_STRUCT_ERR)
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
