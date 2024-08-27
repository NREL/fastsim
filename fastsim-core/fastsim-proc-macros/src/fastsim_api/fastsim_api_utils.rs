use crate::imports::*;

/// Converts multiple uom unit values to a vector of token stream and the plural units name
///
/// - field_units: unit type of value being set (e.g. `uom::si::power::watt`)
macro_rules! extract_units {
    ($($field_units: ty),+) => {{
        let mut unit_impls = vec![];
        $(
            let field_units: TokenStream2 = stringify!($field_units).parse().unwrap();
            let unit_name = <$field_units as uom::si::Unit>::plural().replace(' ', "_");
            unit_impls.push((field_units, unit_name));
        )+
        unit_impls
    }};
}

/// Determine the wrapper type for a specified vector nesting layer
fn vec_layer_type(vec_layers: u8) -> TokenStream2 {
    match vec_layers {
        0 => quote!(f64),
        1 => quote!(Pyo3VecWrapper),
        2 => quote!(Pyo3Vec2Wrapper),
        3 => quote!(Pyo3Vec3Wrapper),
        _ => abort_call_site!("Invalid vector layer {vec_layers}!"),
    }
}

/// Generates pyo3 getter and setter methods for si fields and vector elements
///
/// - impl_block: output TokenStream2
/// - field: struct field name as ident
/// - field_type: token stream of field type (e.g. `si::Power` as a token stream)
/// - field_units: token stream of unit type of value being set (generate using extract_units)
/// - unit_name: plural name of units being used (generate using extract_units)
/// - opts: FieldOptions struct instance
/// - vec_layers: number of nested vector layers
fn impl_get_set_si(
    impl_block: &mut TokenStream2,
    field: &mut syn::Field,
    field_type: &TokenStream2,
    field_units: &TokenStream2,
    unit_name: &str,
    opts: &FieldOptions,
    vec_layers: u8,
) {
    let ident = field.ident.clone().unwrap();
    let field_name: TokenStream2 = match unit_name {
        "" => format!("{ident}").parse().unwrap(),
        _ => {
            if field_has_serde_rename(field) {
                // add the rename attribute for any fields that don't already have it
                let field_name_lit_str = format!("{ident}_{unit_name}");
                field.attrs.push(syn::parse_quote! {
                    #[serde(rename = #field_name_lit_str)]
                })
            }
            format!("{ident}_{unit_name}").parse().unwrap()
        }
    };

    if !opts.skip_get {
        let get_name: TokenStream2 = format!("get_{field_name}").parse().unwrap();
        let get_type = vec_layer_type(vec_layers);
        let unit_func = quote!(get::<#field_units>());
        fn iter_map_collect_vec(inner_func: TokenStream2) -> TokenStream2 {
            quote!(iter().map(|x| x.#inner_func).collect::<Vec<_>>())
        }

        let mut extract_val = unit_func;
        for _ in 0..vec_layers {
            extract_val = iter_map_collect_vec(extract_val);
        }

        let field_val = match vec_layers {
            0 => quote!(self.#ident.#extract_val),
            _ => quote!(#get_type::new(self.#ident.#extract_val)),
        };

        impl_block.extend::<TokenStream2>(quote! {
            #[getter]
            fn #get_name(&self) -> PyResult<#get_type> {
                Ok(#field_val)
            }
        });
    }

    if !opts.skip_set && vec_layers == 0 {
        let set_name: TokenStream2 = format!("set_{field_name}").parse().unwrap();
        let set_err: TokenStream2 = format!("set_{field_name}_err").parse().unwrap();
        let setter_rename: TokenStream2 = format!("__{field_name}").parse().unwrap();

        impl_block.extend::<TokenStream2>(quote! {
            #[setter(#setter_rename)]
            fn #set_err(&mut self, new_val: f64) -> PyResult<()> {
                self.#ident = #field_type::new::<#field_units>(new_val);
                Ok(())
            }
        });

        // Directly setting value raises error to prevent nested struct issues
        impl_block.extend::<TokenStream2>(quote! {
            #[setter]
            fn #set_name(&mut self, new_val: f64) -> PyResult<()> {
                Err(PyAttributeError::new_err(DIRECT_SET_ERR))
            }
        });
    }
}

fn field_has_serde_rename(field: &syn::Field) -> bool {
    !field.attrs.iter().any(|attr| {
        let attr_meta = attr.parse_meta().unwrap();
        if let Meta::List(meta_list) = attr_meta {
            // catch the `serde` in `#[serde(rename = "...")]`
            meta_list.path.is_ident("serde")
                &&
            // catch the `rename` in `#[serde(rename = "...")]`
            meta_list.nested.iter().any(|nm| {
                match nm {
                    NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, ..})) => {
                        path.is_ident("rename")
                    }
                    _ => false
                }
            })
        } else {
            false
        }
    })
}

/// Generates pyo3 getter methods
///
/// - impl_block: TokenStream2
/// - field: struct field
/// - field_type: type of variable (e.g. `f64`)
/// - opts: FieldOptions struct instance
/// - vec_layers: number of nested vector layers
fn impl_get_body(
    impl_block: &mut TokenStream2,
    field: &proc_macro2::Ident,
    field_type: &TokenStream2,
    opts: &FieldOptions,
    vec_layers: u8,
) {
    if !opts.skip_get {
        let get_name: TokenStream2 = format!("get_{field}").parse().unwrap();
        let field_type = match vec_layers {
            0 => field_type.clone(),
            _ => vec_layer_type(vec_layers),
        };

        let field_val = match vec_layers {
            0 => quote!(self.#field.clone()),
            _ => quote!(#field_type::new(self.#field.clone())),
        };

        impl_block.extend::<TokenStream2>(quote! {
            #[getter]
            fn #get_name(&self) -> PyResult<#field_type> {
                Ok(#field_val)
            }
        });
    }
}

/// Generates pyo3 getter methods
///
/// - impl_block: TokenStream2
/// - field: struct field
/// - field_type: type of variable (e.g. `f64`)
/// - opts: FieldOptions struct instance
fn impl_set_body(
    impl_block: &mut TokenStream2,
    field: &proc_macro2::Ident,
    field_type: &TokenStream2,
    opts: &FieldOptions,
) {
    if !opts.skip_set {
        let set_name: TokenStream2 = format!("set_{field}").parse().unwrap();
        let set_err: TokenStream2 = format!("set_{field}_err").parse().unwrap();
        let setter_rename: TokenStream2 = format!("__{field}").parse().unwrap();

        impl_block.extend::<TokenStream2>(quote! {
            #[setter(#setter_rename)]
            fn #set_err(&mut self, new_val: #field_type) -> PyResult<()> {
                self.#field = new_val;
                Ok(())
            }
        });

        // Directly setting value raises error to prevent nested struct issues
        impl_block.extend::<TokenStream2>(quote! {
            #[setter]
            fn #set_name(&mut self, new_val: #field_type) -> PyResult<()> {
                Err(PyAttributeError::new_err(DIRECT_SET_ERR))
            }
        });
    }
}

fn extract_type_path(ty: &syn::Type) -> Option<&syn::Path> {
    match ty {
        syn::Type::Path(type_path) if type_path.qself.is_none() => Some(&type_path.path),
        _ => None,
    }
}

/// adapted from https://stackoverflow.com/questions/55271857/how-can-i-get-the-t-from-an-optiont-when-using-syn
fn extract_type_from_option(ty: &syn::Type) -> Option<&syn::Type> {
    fn extract_option_argument(path: &Path) -> Option<&GenericArgument> {
        let mut ident_path = String::new();
        for segment in &path.segments {
            ident_path.push_str(&segment.ident.to_string());

            // Exit when the inner brackets are found
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(params) => {
                    return match ident_path.as_str() {
                        "Option" | "std::option::Option" | "core::option::Option" => {
                            params.args.first()
                        }
                        _ => None,
                    };
                }
                syn::PathArguments::None => {}
                _ => return None,
            }

            ident_path.push_str("::");
        }
        None
    }

    extract_type_path(ty)
        .and_then(extract_option_argument)
        .and_then(|generic_arg| match *generic_arg {
            GenericArgument::Type(ref ty) => Some(ty),
            _ => None,
        })
}

/// Adapted from https://stackoverflow.com/questions/55271857/how-can-i-get-the-t-from-an-optiont-when-using-syn
/// Extracts contained type from Vec -- i.e. Vec<T> -> T
fn extract_type_from_vec(ty: &syn::Type) -> Option<&syn::Type> {
    fn extract_vec_argument(path: &syn::Path) -> Option<&syn::GenericArgument> {
        let mut ident_path = String::new();
        for segment in &path.segments {
            ident_path.push_str(&segment.ident.to_string());

            // Exit when the inner brackets are found
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(params) => {
                    return match ident_path.as_str() {
                        "Vec" | "std::vec::Vec" => params.args.first(),
                        _ => None,
                    };
                }
                syn::PathArguments::None => {}
                _ => return None,
            }

            ident_path.push_str("::");
        }
        None
    }

    extract_type_path(ty)
        .and_then(extract_vec_argument)
        .and_then(|generic_arg| match generic_arg {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
}

// Extract the quantity name from an absolue uom path or an si path
fn extract_si_quantity(path: &syn::Path) -> Option<String> {
    if path.segments.len() <= 1 {
        return None;
    }
    let mut i = 0;
    if path.segments[i].ident == "uom" {
        i += 1;
        if path.segments.len() <= i + 1 {
            return None;
        }
    }
    if path.segments[i].ident != "si" {
        return None;
    }
    if path.segments[i + 1].ident == "f64" {
        i += 1;
        if path.segments.len() <= i + 1 {
            return None;
        }
    }

    Some(path.segments[i + 1].ident.to_string())
}

pub(crate) fn impl_getters_and_setters(
    impl_block: &mut TokenStream2,
    field: &mut syn::Field,
    opts: &FieldOptions,
) -> Option<()> {
    let ident = field.ident.as_ref().unwrap();
    let ftype = field.ty.clone();
    let mut vec_layers: u8 = 0;
    let mut inner_type = &ftype;

    if let Some(opt_inner_type) = extract_type_from_option(inner_type) {
        inner_type = opt_inner_type;
    }

    // pull out `inner_type` from `Vec<inner_type>`, recursively if there is any nesting
    while let Some(vec_inner_type) = extract_type_from_vec(inner_type) {
        inner_type = vec_inner_type;
        vec_layers += 1;
        if vec_layers >= 4 {
            abort!(ftype.span(), "Too many nested vec layers!");
        }
    }

    let inner_path = extract_type_path(inner_type)?;
    let inner_type = &inner_path.to_token_stream();
    let field_type = extract_type_path(&ftype)?.to_token_stream();
    if let Some(quantity) = extract_si_quantity(inner_path) {
        // Make sure to use absolute paths here to avoid issues with si.rs in the main fastsim-core!
        let unit_impls = match quantity.as_str() {
            "Acceleration" => extract_units!(uom::si::acceleration::meter_per_second_squared),
            "Angle" => extract_units!(uom::si::angle::radian),
            "Area" => extract_units!(uom::si::area::square_meter),
            "SpecificEnergy" => extract_units!(
                uom::si::available_energy::joule_per_kilogram,
                uom::si::available_energy::kilojoule_per_kilogram,
                uom::si::available_energy::megajoule_per_kilogram
            ),
            "Energy" => extract_units!(uom::si::energy::joule),
            "Force" => extract_units!(uom::si::force::newton),
            "InverseVelocity" => extract_units!(uom::si::inverse_velocity::second_per_meter),
            "Length" => extract_units!(uom::si::length::meter, uom::si::length::mile),
            "Mass" => extract_units!(uom::si::mass::kilogram),
            "MomentOfInertia" => extract_units!(uom::si::moment_of_inertia::kilogram_square_meter),
            "Power" => extract_units!(uom::si::power::watt),
            "SpecificPower" => extract_units!(uom::si::specific_power::watt_per_kilogram),
            "PowerRate" => extract_units!(uom::si::power_rate::watt_per_second),
            "Pressure" => extract_units!(uom::si::pressure::kilopascal, uom::si::pressure::bar),
            "Ratio" => extract_units!(uom::si::ratio::ratio),
            "Time" => extract_units!(uom::si::time::second, uom::si::time::hour),
            "Velocity" => extract_units!(
                uom::si::velocity::meter_per_second,
                uom::si::velocity::mile_per_hour
            ),
            "Volume" => extract_units!(uom::si::volume::cubic_meter, uom::si::volume::liter),
            "MassDensity" => extract_units!(uom::si::mass_density::kilogram_per_cubic_meter),
            _ => abort!(
                inner_path.span(),
                "Unknown si quantity! Make sure it's implemented in `impl_getters_and_setters`"
            ),
        };
        for (field_units, unit_name) in &unit_impls {
            impl_get_set_si(
                impl_block,
                field,
                inner_type,
                field_units,
                unit_name,
                opts,
                vec_layers,
            );
        }
    } else if inner_type.to_string().as_str() == "f64" {
        impl_get_body(impl_block, ident, inner_type, opts, vec_layers);
        impl_set_body(impl_block, ident, &field_type, opts);
    } else {
        impl_get_body(impl_block, ident, &field_type, opts, 0);
        if ident != "history" {
            impl_set_body(impl_block, ident, &field_type, opts);
        }
    }

    Some(())
}

#[derive(Debug, Default, Clone)]
pub(crate) struct FieldOptions {
    /// if true, getters are not generated for a field
    pub skip_get: bool,
    /// if true, setters are not generated for a field
    pub skip_set: bool,
    // TODO: uncomment and clean up, and then create equivalent `set_<fieldname>_from_pydict`
    // and `<fieldname>_to_pydict` methods via `setattr`
    // /// if true, writes methods to get and set enum via json
    // pub enum_as_json: bool,
}
