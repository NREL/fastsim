use crate::imports::*;

// taken from https://github.com/lumol-org/soa-derive/blob/master/soa-derive-internal/src/input.rs
pub(crate) trait TokenStreamIterator {
    fn concat_by(
        self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream;
    fn concat(self) -> proc_macro2::TokenStream;
}

impl<T: Iterator<Item = proc_macro2::TokenStream>> TokenStreamIterator for T {
    fn concat_by(
        mut self,
        f: impl Fn(proc_macro2::TokenStream, proc_macro2::TokenStream) -> proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self.next() {
            Some(first) => self.fold(first, f),
            None => quote! {},
        }
    }

    fn concat(self) -> proc_macro2::TokenStream {
        self.concat_by(|a, b| quote! { #a #b })
    }
}

macro_rules! impl_vec_get_set {
    ($opts: ident, $fident: ident, $impl_block: ident, $contained_type: ty, $wrapper_type: expr, $has_orphaned: expr) => {
        if !$opts.skip_get {
            let get_name: TokenStream2 = format!("get_{}", $fident).parse().unwrap();
            $impl_block.extend::<TokenStream2>(quote! {
                #[getter]
                pub fn #get_name(&self) -> PyResult<$wrapper_type> {
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
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> PyResult<()> {
                                if !self.orphaned {
                                    self.#$fident = new_value;
                                    Ok(())
                                } else {
                                    Err(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                                }
                            }
                        })
                    } else {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> PyResult<()> {
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
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> PyResult<()> {
                                if !self.orphaned {
                                    self.#$fident = Array1::from_vec(new_value);
                                    Ok(())
                                } else {
                                    Err(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                                }
                            }
                        })
                    } else {
                        $impl_block.extend(quote! {
                            #[setter]
                            pub fn #set_name(&mut self, new_value: Vec<$contained_type>) -> PyResult<()> {
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
                    pub fn #get_name(&mut self) -> PyResult<#$type> {
                        self.#$field.orphaned = true;
                        Ok(self.#$field.clone())
                    }
                }
            } else {
                quote! {
                    #[getter]
                    pub fn #get_name(&self) -> PyResult<#$type> {
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
                        Err(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
                    }
                }
            } else if $has_orphaned {
                quote! {
                    if !self.orphaned {
                        self.#$field = new_value;
                        Ok(())
                    } else {
                        Err(PyAttributeError::new_err(crate::utils::NESTED_STRUCT_ERR))
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
                pub fn #set_name(&mut self, new_value: #$type) ->  PyResult<()> {
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

pub fn impl_getters_and_setters(
    type_path: syn::TypePath,
    impl_block: &mut TokenStream2,
    ident: &proc_macro2::Ident,
    opts: FieldOptions,
    has_orphaned: bool,
    ftype: syn::Type,
) {
    let type_str = type_path.into_token_stream().to_string();
    match type_str.as_str() {
        "Array1 < f64 >" => {
            impl_vec_get_set!(opts, ident, impl_block, f64, Pyo3ArrayF64, has_orphaned);
        }
        "Array1 < u32 >" => {
            impl_vec_get_set!(opts, ident, impl_block, u32, Pyo3ArrayU32, has_orphaned);
        }
        "Array1 < i32 >" => {
            impl_vec_get_set!(opts, ident, impl_block, i32, Pyo3ArrayI32, has_orphaned);
        }
        "Array1 < bool >" => {
            impl_vec_get_set!(opts, ident, impl_block, bool, Pyo3ArrayBool, has_orphaned);
        }
        "Vec < f64 >" => {
            impl_vec_get_set!(opts, ident, impl_block, f64, Pyo3VecF64, has_orphaned);
        }
        _ => match ident.to_string().as_str() {
            "orphaned" => {
                impl_block.extend::<TokenStream2>(quote! {
                    #[getter]
                    pub fn get_orphaned(&self) -> PyResult<bool> {
                        Ok(self.orphaned)
                    }
                    /// Reset the orphaned flag to false.
                    pub fn reset_orphaned(&mut self) -> PyResult<()> {
                        self.orphaned = false;
                        Ok(())
                    }
                })
            }
            _ => {
                impl_get_body!(ftype, ident, impl_block, opts);
                impl_set_body!(ftype, ident, impl_block, has_orphaned, opts);
            }
        },
    }
}

const ONLY_FN_MSG: &str = "Only function definitions allowed here.";

/// accepts `attr` TokenStream from attribute-like proc macro and returns
/// TokenStream2 of fn defs that are in `expected_fn_names` and/or not in `forbidden_fn_names`.  
/// If `expected_exlusive` is true, only values in `expected_fn_names` are allowed.  
/// Raises locationally useful errors if mistakes are made in formatting or whatnot.  
pub fn parse_ts_as_fn_defs(
    attr: TokenStream,
    mut expected_fn_names: Vec<String>,
    expected_exclusive: bool,
    forbidden_fn_names: Vec<String>,
) -> TokenStream2 {
    let attr = TokenStream2::from(attr);
    let impl_block = quote! {
        impl Dummy { // this name doesn't really matter as it won't get used
            #attr
        }
    }
    .into();
    // let item_impl = syn::parse_macro_input!(impl_block as syn::ItemImpl);
    let item_impl = syn::parse::<syn::ItemImpl>(impl_block)
        .map_err(|_| abort_call_site!(ONLY_FN_MSG))
        .unwrap();

    let mut fn_from_attr = TokenStream2::new();

    for impl_item in item_impl.items {
        match &impl_item {
            syn::ImplItem::Method(item_meth) => {
                let sig_str = &item_meth.sig.ident.to_token_stream().to_string();
                fn_from_attr.extend(item_meth.clone().to_token_stream());
                // check signature
                if expected_exclusive {
                    if forbidden_fn_names.contains(sig_str) || !expected_fn_names.contains(sig_str)
                    {
                        emit_error!(
                            &item_meth.sig.ident.span(),
                            format!("Function name `{}` is forbidden", sig_str)
                        )
                    }
                }

                let index = expected_fn_names.iter().position(|x| x == sig_str);

                match index {
                    Some(i) => {
                        expected_fn_names.remove(i);
                    }
                    _ => {}
                }
                // remove the matching name from the vec to avoid checking again
                // at the end of iteration, this vec should be empty
            }
            _ => abort_call_site!(ONLY_FN_MSG),
        }
    }

    if !expected_fn_names.is_empty() {
        abort_call_site!(format!(
            "Expected fn def for `{}`",
            expected_fn_names.join(", ")
        ));
    }
    fn_from_attr
}
