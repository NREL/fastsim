use crate::imports::*;
use std::collections::HashSet;

/// Converts multiple uom unit values to a vector of token stream and the plural units name
///
/// - field_units: unit type of value being set (e.g. `uom::si::power::watt`)
macro_rules! extract_units {
    ($($field_units: ty),+) => {{
        let mut unit_impls = vec![];
        $(
            let field_units: TokenStream2 = stringify!($field_units).parse().expect("failed to parse `field_units`");
            let mut unit_name = <$field_units as uom::si::Unit>::plural().to_lowercase().replace(' ', "_");

            // UOM has a bug where ratio.plural() returns an empty string
            // Default to "ratio" for the ratio unit when plural() is empty
            if unit_name.is_empty() {
                let debug_name = stringify!($field_units);
                if debug_name.contains("ratio") {
                    unit_name = "ratio".to_string();
                }
            }

            // fix UOM pluralization edge cases
            let unit_name = unit_name.replace("kelvins", "kelvin");
            let unit_name = unit_name.replace("ratios", "ratio");
            unit_impls.push((field_units, unit_name));
        )+
        unit_impls
    }};
}

/// Generates pyo3 getter and setter methods for si fields and vector elements
///
/// - field: struct field name as ident
/// - unit_name: plural name of units being used (generate using extract_units)
fn serde_attrs_for_si_field(field: &mut syn::Field, unit_name: &str) {
    let ident = field.ident.clone().unwrap();
    match unit_name {
        "" => {}
        _ => {
            if !field_has_serde_rename(field) {
                // add the rename attribute for any fields that don't already have it
                let field_name_lit_str = format!("{ident}_{unit_name}");
                field.attrs.push(syn::parse_quote! {
                    #[serde(rename = #field_name_lit_str)]
                });
            }
        }
    }
}

fn field_has_serde_rename(field: &syn::Field) -> bool {
    field.attrs.iter().any(|attr| {
        if let Meta::List(ml) = &attr.meta {
            // catch the `serde` in `#[serde(rename = "...")]`
            ml.path.is_ident("serde")
                &&
            // catch the `rename` in `#[serde(rename = "...")]`
            ml.tokens.to_string().contains("rename")
        } else {
            false
        }
    })
}

fn extract_type_path(ty: &syn::Type) -> Option<&syn::Path> {
    match ty {
        syn::Type::Path(type_path) if type_path.qself.is_none() => Some(&type_path.path),
        _ => None,
    }
}

fn extract_type_from_container(ty: &syn::Type) -> Option<&syn::Type> {
    fn extract_container_arg(path: &Path) -> Option<&GenericArgument> {
        let mut ident_path = String::new();
        for segment in &path.segments {
            ident_path.push_str(&segment.ident.to_string());

            // Exit when the inner brackets are found
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(params) => return params.args.first(),
                syn::PathArguments::None => {}
                _ => return None,
            }

            ident_path.push_str("::");
        }
        None
    }

    extract_type_path(ty)
        .and_then(extract_container_arg)
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

pub(crate) fn serde_attrs_for_si_fields(field: &mut syn::Field) -> Option<()> {
    let ftype = field.ty.clone();
    let mut vec_layers: u8 = 0;
    let mut inner_type = &ftype;

    while let Some(opt_inner_type) = extract_type_from_container(inner_type) {
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
            "HeatTransferCoeff" => extract_units!(
                uom::si::heat_transfer::watt_per_square_meter_kelvin,
                uom::si::heat_transfer::watt_per_square_meter_degree_celsius
            ),
            "Curvature" => extract_units!(
                uom::si::curvature::radian_per_meter,
                uom::si::curvature::degree_per_meter
            ),
            "HeatCapacity" => {
                extract_units!(
                    uom::si::heat_capacity::joule_per_kelvin,
                    uom::si::heat_capacity::joule_per_degree_celsius
                )
            }
            "TemperatureInterval" => extract_units!(uom::si::temperature_interval::kelvin),
            "Temperature" => {
                extract_units!(uom::si::thermodynamic_temperature::kelvin)
            }
            "ThermalConductance" => {
                extract_units!(uom::si::thermal_conductance::watt_per_kelvin)
            }
            "ThermalConductivity" => {
                extract_units!(
                    uom::si::thermal_conductivity::watt_per_meter_kelvin,
                    uom::si::thermal_conductivity::watt_per_meter_degree_celsius
                )
            }
            "DynamicViscosity" => {
                extract_units!(uom::si::dynamic_viscosity::pascal_second)
            }
            "Velocity" => extract_units!(
                uom::si::velocity::meter_per_second,
                uom::si::velocity::mile_per_hour
            ),
            "Volume" => extract_units!(uom::si::volume::cubic_meter, uom::si::volume::liter),
            "EnergyDensity" => vec![(
                quote! {EnergyDensity},
                String::from("joule_per_cubic_meter"),
            )],
            "MassDensity" => extract_units!(uom::si::mass_density::kilogram_per_cubic_meter),
            _ => abort!(
                inner_path.span(),
                "Unknown si quantity! Make sure it's implemented in `impl_getters_and_setters`"
            ),
        };
        for (_, unit_name) in &unit_impls {
            serde_attrs_for_si_field(field, unit_name);
        }
    }
    Some(())
}

#[derive(Clone, Debug)]
pub struct SIFieldData {
    pub field_ident: syn::Ident,
    pub quantity: String,
    pub units: Vec<(TokenStream2, String)>, // (unit_type, plural_name)
}

/// Collect SI field information for helper struct generation
pub fn collect_si_field_data(field: &syn::Field) -> Option<SIFieldData> {
    let field_ident = field.ident.as_ref()?.clone();
    let mut inner_type = &field.ty;

    // Unwrap Option<T>
    if let Some(opt_inner) = extract_type_from_container(inner_type) {
        inner_type = opt_inner;
    }

    // Unwrap Vec<T>
    if let Some(vec_inner) = extract_type_from_vec(inner_type) {
        inner_type = vec_inner;
    }

    let inner_path = extract_type_path(inner_type)?;
    let quantity = extract_si_quantity(inner_path)?;

    // Get units for this quantity
    let units = get_units_for_quantity(&quantity);

    Some(SIFieldData {
        field_ident,
        quantity,
        units,
    })
}

fn get_units_for_quantity(quantity: &str) -> Vec<(TokenStream2, String)> {
    match quantity {
        "Mass" => extract_units!(uom::si::mass::kilogram),
        "Power" => extract_units!(uom::si::power::watt, uom::si::power::kilowatt),
        "Time" => extract_units!(uom::si::time::second, uom::si::time::hour),
        "Temperature" => extract_units!(
            uom::si::thermodynamic_temperature::kelvin,
            uom::si::thermodynamic_temperature::degree_celsius,
            uom::si::thermodynamic_temperature::degree_fahrenheit
        ),
        "Velocity" => extract_units!(
            uom::si::velocity::meter_per_second,
            uom::si::velocity::kilometer_per_hour,
            uom::si::velocity::mile_per_hour
        ),
        "Energy" => extract_units!(uom::si::energy::joule, uom::si::energy::kilowatt_hour),
        "Ratio" => extract_units!(uom::si::ratio::ratio, uom::si::ratio::percent),
        "Area" => extract_units!(uom::si::area::square_meter),
        _ => vec![],
    }
}

pub fn generate_helper_struct(
    helper_name: &syn::Ident,
    struct_ast: &syn::ItemStruct,
    si_fields: &[SIFieldData],
) -> TokenStream2 {
    // Build a map of SI field idents for quick lookup
    let si_field_idents: std::collections::HashSet<_> = si_fields
        .iter()
        .map(|f| f.field_ident.to_string())
        .collect();

    let mut helper_fields = vec![];

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &struct_ast.fields {
        for field in named {
            let field_ident = field.ident.as_ref().unwrap();
            let field_ident_str = field_ident.to_string();

            if si_field_idents.contains(&field_ident_str) {
                // This is an SI field - generate helper fields for each unit variant
                if let Some(si_field) = si_fields
                    .iter()
                    .find(|f| f.field_ident.to_string() == field_ident_str)
                {
                    let bare_name = field_ident.to_string();
                    for (idx, (_unit_type, unit_name)) in si_field.units.iter().enumerate() {
                        let helper_field_name = syn::Ident::new(
                            &format!("{}_{}_{}", field_ident, unit_name, "macrogenerated"),
                            field_ident.span(),
                        );

                        let field_ty = &field.ty;
                        let wrapper_type = detect_outer_wrapper(field_ty);

                        let field_type = match wrapper_type {
                            WrapperType::Vec => quote! { Option<Vec<f64>> },
                            _ => quote! { Option<f64> },
                        };

                        // The JSON key is "field_unit" (without _macrogenerated)
                        let json_key = format!("{}_{}", field_ident, unit_name);

                        // The primary unit (first in list) also accepts the bare field name as alias
                        let serde_attr = if idx == 0 {
                            quote! { #[serde(default, rename = #json_key, alias = #bare_name)] }
                        } else {
                            quote! { #[serde(default, rename = #json_key)] }
                        };

                        helper_fields.push(quote! {
                            #serde_attr
                            pub #helper_field_name: #field_type
                        });
                    }
                }
            } else {
                // Non-SI field
                let field_ty = &field.ty;
                let has_skip = has_serde_skip(field);
                let contains_self = type_contains_self(field_ty);
                let has_default = has_serde_default(field);

                if has_skip {
                    // Skip: omit from helper struct entirely; From impl will use Default::default()
                } else if contains_self && !has_default {
                    // Self-referential container without skip/default: emit a compile error
                    let field_name = field_ident.to_string();
                    helper_fields.push(quote! {
                        compile_error!(concat!(
                            "Field `", #field_name,
                            "` contains `Self` and is used with #[serde_api]. ",
                            "Add #[serde(skip)] or #[serde(default)] to this field."
                        ));
                    });
                } else {
                    let struct_name = &struct_ast.ident;
                    let final_ty = replace_self_in_type(field_ty, struct_name);
                    // Preserve only serde attributes (not custom derive helpers like #[has_state])
                    let serde_attrs: Vec<_> = field
                        .attrs
                        .iter()
                        .filter(|a| a.path().is_ident("serde"))
                        .collect();
                    helper_fields.push(quote! {
                        #(#serde_attrs)*
                        pub #field_ident: #final_ty
                    });
                }
            }
        }
    }

    quote! {
        #[derive(::serde::Deserialize)]
        #[serde(crate = "::serde")]
        struct #helper_name {
            #(#helper_fields),*
        }
    }
}

pub fn generate_from_impl(
    original_name: &syn::Ident,
    helper_name: &syn::Ident,
    struct_ast: &syn::ItemStruct,
    si_fields: &[SIFieldData],
) -> TokenStream2 {
    // Build a map of SI field idents for quick lookup
    let si_field_idents: std::collections::HashSet<_> = si_fields
        .iter()
        .map(|f| f.field_ident.to_string())
        .collect();

    let mut field_conversions = vec![];

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &struct_ast.fields {
        for field in named {
            let field_ident = field.ident.as_ref().unwrap();
            let field_ident_str = field_ident.to_string();

            if si_field_idents.contains(&field_ident_str) {
                // This is an SI field - convert from helper unit variants
                if let Some(si_field) = si_fields
                    .iter()
                    .find(|f| f.field_ident.to_string() == field_ident_str)
                {
                    let quantity_str = &si_field.quantity;
                    let quantity_type = match quantity_str.as_str() {
                        "Mass" => quote! { uom::si::f64::Mass },
                        "Power" => quote! { uom::si::f64::Power },
                        "Time" => quote! { uom::si::f64::Time },
                        "Temperature" => quote! { uom::si::f64::ThermodynamicTemperature },
                        "Velocity" => quote! { uom::si::f64::Velocity },
                        "Energy" => quote! { uom::si::f64::Energy },
                        "Ratio" => quote! { uom::si::f64::Ratio },
                        "Area" => quote! { uom::si::f64::Area },
                        _ => continue,
                    };

                    // Check if field is wrapped in TrackedState, Option, or Vec
                    let field_ty = &field.ty;
                    let wrapper_type = detect_outer_wrapper(field_ty);

                    let mut match_arms = vec![];

                    if wrapper_type == WrapperType::Vec {
                        // For Vec fields, convert each element in the vector
                        for (unit_type, unit_name) in &si_field.units {
                            let helper_field_name = syn::Ident::new(
                                &format!("{}_{}_{}", field_ident, unit_name, "macrogenerated"),
                                field_ident.span(),
                            );

                            match_arms.push(quote! {
                                helper.#helper_field_name.map(|vals| vals.into_iter().map(|v| #quantity_type::new::<#unit_type>(v)).collect::<Vec<_>>())
                            });
                        }
                    } else {
                        // For non-Vec fields, convert the single value
                        for (unit_type, unit_name) in &si_field.units {
                            let helper_field_name = syn::Ident::new(
                                &format!("{}_{}_{}", field_ident, unit_name, "macrogenerated"),
                                field_ident.span(),
                            );

                            match_arms.push(quote! {
                                helper.#helper_field_name.map(|val| #quantity_type::new::<#unit_type>(val))
                            });
                        }
                    }

                    if match_arms.is_empty() {
                        continue;
                    }

                    // Combine all match arms with .or_else()
                    let conversion = if match_arms.len() == 1 {
                        match_arms.into_iter().next().unwrap()
                    } else {
                        let mut result = match_arms[0].clone();
                        for arm in &match_arms[1..] {
                            result = quote! {
                                #result.or_else(|| #arm)
                            };
                        }
                        result
                    };

                    let wrapped_conversion = match wrapper_type {
                        WrapperType::TrackedState => {
                            let ts_path = outer_type_path_tokens(field_ty).unwrap_or_else(
                                || quote! { crate::utils::tracked_state::TrackedState },
                            );
                            quote! {
                                (#conversion).map(|val| #ts_path::new(val))
                                    .expect(&format!("Missing field {} in any unit variant", stringify!(#field_ident)))
                            }
                        }
                        WrapperType::Option => {
                            quote! {
                                (#conversion)
                            }
                        }
                        WrapperType::Vec => {
                            quote! {
                                (#conversion).unwrap_or_default()
                            }
                        }
                        WrapperType::None => {
                            quote! {
                                (#conversion).expect(&format!("Missing field {} in any unit variant", stringify!(#field_ident)))
                            }
                        }
                    };

                    field_conversions.push(quote! {
                        #field_ident: #wrapped_conversion
                    });
                }
            } else {
                // Non-SI field
                if has_serde_skip(field) {
                    field_conversions.push(quote! {
                        #field_ident: Default::default()
                    });
                } else {
                    field_conversions.push(quote! {
                        #field_ident: helper.#field_ident
                    });
                }
            }
        }
    }

    quote! {
        impl From<#helper_name> for #original_name {
            fn from(helper: #helper_name) -> Self {
                Self {
                    #(#field_conversions),*
                }
            }
        }
    }
}

/// Returns true if the field has `#[serde(skip)]`
fn has_serde_skip(field: &syn::Field) -> bool {
    has_serde_flag(field, "skip")
}

/// Returns true if the field has `#[serde(default)]`
fn has_serde_default(field: &syn::Field) -> bool {
    has_serde_flag(field, "default")
}

fn has_serde_flag(field: &syn::Field, flag: &str) -> bool {
    field.attrs.iter().any(|attr| {
        if !attr.path().is_ident("serde") {
            return false;
        }
        let mut found = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(flag) {
                found = true;
            }
            // consume "= value" if present so parse_nested_meta doesn't error
            if meta.input.peek(syn::Token![=]) {
                let _: syn::Token![=] = meta.input.parse().unwrap();
                let _: proc_macro2::TokenTree = meta.input.parse().unwrap();
            }
            Ok(())
        });
        found
    })
}

/// Replace bare `Self` path segments with `replacement` in a type.
fn replace_self_in_type(ty: &syn::Type, replacement: &syn::Ident) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            let mut new_path = type_path.clone();
            for segment in &mut new_path.path.segments {
                if segment.ident == "Self" {
                    segment.ident = replacement.clone();
                }
                if let syn::PathArguments::AngleBracketed(ref mut args) = segment.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            *inner_ty = replace_self_in_type(inner_ty, replacement);
                        }
                    }
                }
            }
            syn::Type::Path(new_path)
        }
        _ => ty.clone(),
    }
}

/// Returns true if the type tree contains a bare `Self` path segment.
fn type_contains_self(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => type_path.path.segments.iter().any(|seg| {
            seg.ident == "Self"
                || if let syn::PathArguments::AngleBracketed(ref args) = seg.arguments {
                    args.args.iter().any(|arg| {
                        if let syn::GenericArgument::Type(inner) = arg {
                            type_contains_self(inner)
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
        }),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WrapperType {
    TrackedState,
    Option,
    Vec,
    None,
}

fn detect_outer_wrapper(ty: &syn::Type) -> WrapperType {
    if let Some(path) = extract_type_path(ty) {
        if let Some(segment) = path.segments.last() {
            if segment.ident == "TrackedState" {
                return WrapperType::TrackedState;
            } else if segment.ident == "Option" {
                return WrapperType::Option;
            } else if segment.ident == "Vec" {
                return WrapperType::Vec;
            }
        }
    }
    WrapperType::None
}

/// Extract the outer type path without generic args as a token stream.
/// e.g. `crate::utils::TrackedState<si::Power>` → `crate::utils::TrackedState`
/// Used to call `::new()` on the outer wrapper type with the correct path.
fn outer_type_path_tokens(ty: &syn::Type) -> Option<TokenStream2> {
    if let syn::Type::Path(type_path) = ty {
        let mut path = type_path.path.clone();
        // Strip generic args from the last segment
        if let Some(last) = path.segments.last_mut() {
            last.arguments = syn::PathArguments::None;
        }
        Some(quote! { #path })
    } else {
        None
    }
}

fn is_tracked_state_wrapper(ty: &syn::Type) -> bool {
    detect_outer_wrapper(ty) == WrapperType::TrackedState
}
