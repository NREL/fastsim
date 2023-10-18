macro_rules! impl_vec_get_set {
    ($opts: ident, $fident: ident, $impl_block: ident, $contained_type: ty, $wrapper_type: expr, $has_orphaned: expr) => {
        if !$opts.skip_get {
            let get_name: TokenStream2 = format!("get_{}", $fident).parse().unwrap();
            $impl_block.extend::<TokenStream2>(quote! {
                #[getter]
                pub fn #get_name(&self) -> anyhow::Result<$wrapper_type> {
                    Ok($wrapper_type::new(self.#$fident.clone()))
                }
            });
        }
        if !$opts.skip_set {
            let set_name: TokenStream2 = format!("set_{}", $fident).parse().unwrap();
            match stringify!($wrapper_type) {
                "Pyo3VecF64" => {
                    if $has_orphaned {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> anyhow::Result<()> {
                                if !self.orphaned {
                                    self.#$fident = new_value;
                                    Ok(())
                                } else {
                                    anyhow::bail!(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                                }
                            }
                        })
                    } else {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> anyhow::Result<()> {
                                self.#$fident = new_value;
                                Ok(())
                            }
                        })
                    }
                }
                _ => {
                    if $has_orphaned {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> anyhow::Result<()> {
                                if !self.orphaned {
                                    self.#$fident = Array1::from_vec(new_value);
                                    Ok(())
                                } else {
                                    anyhow::bail!(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                                }
                            }
                        })
                    } else {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> anyhow::Result<()> {
                                self.#$fident = Array1::from_vec(new_value);
                                Ok(())
                            }
                        })
                    }
                }
            }

        }
    };
}

/// Generates pyo3 getter methods
///
/// general match arguments:
/// - type: type of variable (e.g. `f64`)
/// - field: struct field
/// - impl_block: TokenStream2
/// - opts: FieldOptions struct instance
macro_rules! impl_get_body {
    (
        $type: ident, $field: ident, $impl_block: ident, $opts: ident
    ) => {
        if !$opts.skip_get {
            let get_name: TokenStream2 = format!("get_{}", $field).parse().unwrap();
            let get_block = if $opts.field_has_orphaned {
                quote! {
                    #[getter]
                    pub fn #get_name(&mut self) -> anyhow::Result<#$type> {
                        self.#$field.orphaned = true;
                        Ok(self.#$field.clone())
                    }
                }
            } else {
                quote! {
                    #[getter]
                    pub fn #get_name(&self) -> anyhow::Result<#$type> {
                        Ok(self.#$field.clone())
                    }
                }
            };
            $impl_block.extend::<TokenStream2>(get_block)
        }
    };
}

/// Generates pyo3 setter methods
///
/// general match arguments:
/// - type: type of variable (e.g. `f64`)
/// - field: struct field
/// - impl_block: TokenStream2
/// - has_orphaned: bool, true if struct has `orphaned` field
/// - opts: FieldOptions struct instance

macro_rules! impl_set_body {
    ( // for generic
        $type: ident, $field: ident, $impl_block: ident, $has_orphaned: expr, $opts: ident
    ) => {
        if !$opts.skip_set {
            let set_name: TokenStream2 = format!("set_{}", $field).parse().unwrap();
            let orphaned_set_block = if $has_orphaned && $opts.field_has_orphaned {
                quote! {
                    if !self.orphaned {
                        self.#$field = new_value;
                        self.#$field.orphaned = false;
                        Ok(())
                    } else {
                        anyhow::bail!(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                    }
                }
            } else if $has_orphaned {
                quote! {
                    if !self.orphaned {
                        self.#$field = new_value;
                        Ok(())
                    } else {
                        anyhow::bail!(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                    }
                }
            } else {
                quote! {
                        self.#$field = new_value;
                        Ok(())
                }
            };

            $impl_block.extend::<TokenStream2>(quote! {
                #[setter]
                pub fn #set_name(&mut self, new_value: #$type) -> anyhow::Result<()> {
                    #orphaned_set_block
                }
            });
        }
    };
}

#[derive(Debug, Default, Clone)]
pub struct FieldOptions {
    /// if true, getters are not generated for a field
    pub skip_get: bool,
    /// if true, setters are not generated for a field
    pub skip_set: bool,
    /// if true, current field is itself a struct with `orphaned` field
    pub field_has_orphaned: bool,
}
