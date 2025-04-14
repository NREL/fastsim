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
