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
        if !$struct_self.orphaned {
            $struct_self.$field = $value;
            Ok(())
        } else {
            Err(anyhow!(utils::NESTED_STRUCT_ERR))
        }
    };
}
