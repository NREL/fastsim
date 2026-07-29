use crate::imports::*;

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

/// Generates serde attributes for si fields
///
/// - field: struct field name as ident
/// - unit_name: plural name of units being used (generate using extract_units)
fn serde_attrs_for_si_field(field: &mut syn::Field, unit_name: &str, serialize_with: Option<&str>) {
    let ident = field.ident.clone().unwrap();
    match unit_name {
        // Empty unit name or "ratio" → canonical is the bare field name; no rename needed.
        // TODO: remove ratio exception once all efficiencies use an efficiency enum
        "" | "ratio" => {}
        _ => {
            if !field_has_serde_rename(field) {
                // add the rename attribute for any fields that don't already have it
                let field_name_lit_str = format!("{ident}_{unit_name}");
                if let Some(sw_base) = serialize_with {
                    // The serialize_with path names the plain-field helper (e.g.
                    // "crate::utils::serde_helpers::power_as_kilowatts").
                    // Adjust it for wrapper types by inserting the appropriate prefix
                    // before the function name: vec_ / opt_ / tracked_.
                    let sw_path = adjust_serialize_with_for_wrapper(sw_base, &field.ty);
                    field.attrs.push(syn::parse_quote! {
                        #[serde(rename = #field_name_lit_str, serialize_with = #sw_path)]
                    });
                } else {
                    field.attrs.push(syn::parse_quote! {
                        #[serde(rename = #field_name_lit_str)]
                    });
                }
            }
        }
    }
}

/// Given a base serialize_with path (for plain `T`) and a field type, insert the
/// wrapper-appropriate prefix before the function name:
///   Vec<TrackedState<T>>  → vec_tracked_fn
///   Vec<T>                → vec_fn
///   Option<T>             → opt_fn
///   TrackedState<T>       → tracked_fn
///   T                     → fn (unchanged)
fn adjust_serialize_with_for_wrapper(base_path: &str, field_ty: &syn::Type) -> String {
    let prefix = match detect_outer_wrapper(field_ty) {
        WrapperType::Vec => {
            // Check if the inner type is TrackedState (Vec<TrackedState<T>>)
            if let Some(inner) = extract_type_from_vec(field_ty) {
                if detect_outer_wrapper(inner) == WrapperType::TrackedState {
                    "vec_tracked_"
                } else {
                    "vec_"
                }
            } else {
                "vec_"
            }
        }
        WrapperType::Option => "opt_",
        WrapperType::TrackedState => "tracked_",
        WrapperType::None => return base_path.to_string(),
    };
    // Insert prefix before the last path segment: "a::b::fn_name" → "a::b::prefix_fn_name"
    if let Some(sep) = base_path.rfind("::") {
        format!("{}::{}{}", &base_path[..sep], prefix, &base_path[sep + 2..])
    } else {
        format!("{}{}", prefix, base_path)
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
        // Each arm returns (unit_impls, serialize_with).
        // unit_impls:    sets the rename key added to the struct field for **serialization**.
        // serialize_with: optional path to a fn in crate::utils::serde_helpers that converts
        //                 the UOM base-unit value to the target unit before writing.
        //                 Must match the unit in unit_impls. Leave None to serialize in base units.
        //
        // Example — to serialize Power in kilowatts instead of watts, change the "Power" arm to:
        //   "Power" => (
        //       extract_units!(uom::si::power::kilowatt),
        //       Some("fastsim_core::utils::serde_helpers::power_as_kilowatts"),
        //   ),
        // and uncomment the matching line in fastsim_core::utils::serde_helpers.
        let (unit_impls, serialize_with): (Vec<(TokenStream2, String)>, Option<&str>) =
            match quantity.as_str() {
                "Acceleration" => (
                    extract_units!(uom::si::acceleration::meter_per_second_squared),
                    None,
                ),
                "Angle" => (extract_units!(uom::si::angle::radian), None),
                "Area" => (extract_units!(uom::si::area::square_meter), None),
                "SpecificEnergy" => (
                    extract_units!(uom::si::available_energy::joule_per_kilogram),
                    None,
                ),
                "Energy" => (extract_units!(uom::si::energy::joule), None),
                "Force" => (extract_units!(uom::si::force::newton), None),
                "InverseVelocity" => (
                    extract_units!(uom::si::inverse_velocity::second_per_meter),
                    None,
                ),
                "Length" => (extract_units!(uom::si::length::meter), None),
                "Mass" => (extract_units!(uom::si::mass::kilogram), None),
                "MomentOfInertia" => (
                    extract_units!(uom::si::moment_of_inertia::kilogram_square_meter),
                    None,
                ),
                "Power" => (extract_units!(uom::si::power::watt), None),
                // "Power" => (
                //     extract_units!(uom::si::power::kilowatt),
                //     Some("fastsim_core::utils::serde_helpers::power_as_kilowatts"),
                // ),
                "SpecificPower" => (
                    extract_units!(uom::si::specific_power::watt_per_kilogram),
                    None,
                ),
                "PowerRate" => (extract_units!(uom::si::power_rate::watt_per_second), None),
                "Pressure" => (extract_units!(uom::si::pressure::kilopascal), None),
                "Ratio" => (extract_units!(uom::si::ratio::ratio), None),
                "Time" => (extract_units!(uom::si::time::second), None),
                "HeatTransferCoeff" => (
                    extract_units!(uom::si::heat_transfer::watt_per_square_meter_kelvin),
                    None,
                ),
                "Curvature" => (extract_units!(uom::si::curvature::radian_per_meter), None),
                "HeatCapacity" => (
                    extract_units!(uom::si::heat_capacity::joule_per_kelvin),
                    None,
                ),
                "TemperatureInterval" => {
                    (extract_units!(uom::si::temperature_interval::kelvin), None)
                }
                "Temperature" => (
                    extract_units!(uom::si::thermodynamic_temperature::kelvin),
                    None,
                ),
                "ThermalConductance" => (
                    extract_units!(uom::si::thermal_conductance::watt_per_kelvin),
                    None,
                ),
                "ThermalConductivity" => (
                    extract_units!(uom::si::thermal_conductivity::watt_per_meter_kelvin),
                    None,
                ),
                "DynamicViscosity" => (
                    extract_units!(uom::si::dynamic_viscosity::pascal_second),
                    None,
                ),
                "Velocity" => (extract_units!(uom::si::velocity::meter_per_second), None),
                "Volume" => (extract_units!(uom::si::volume::cubic_meter), None),
                "EnergyDensity" => (
                    vec![(
                        quote! {EnergyDensity},
                        String::from("joule_per_cubic_meter"),
                    )],
                    None,
                ),
                "MassDensity" => (
                    extract_units!(uom::si::mass_density::kilogram_per_cubic_meter),
                    None,
                ),
                _ => abort!(
                    inner_path.span(),
                    "Unknown si quantity! Make sure it's implemented in `impl_getters_and_setters`"
                ),
            };
        for (_, unit_name) in &unit_impls {
            serde_attrs_for_si_field(field, unit_name, serialize_with);
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

    // Fields with #[serde(skip)] are excluded from the SI helper path.
    if has_serde_skip(field) {
        return None;
    }

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

/// NOTE: this is where each available unit is defined for each quantity.
/// To support a new unit in **deserialization**, add it to the appropriate quantity arm here.
/// The first entry in each arm is the **primary (canonical) unit** used for round-trip:
/// it is the one selected when neither a unit-suffixed key nor a bare key is present (for structs
/// with `#[serde(default)]`), and its suffix is the only one guaranteed to appear in serialized output.
fn get_units_for_quantity(quantity: &str) -> Vec<(TokenStream2, String)> {
    match quantity {
        "Mass" => extract_units!(uom::si::mass::kilogram),
        "Power" => extract_units!(
            uom::si::power::watt,
            uom::si::power::kilowatt,
            uom::si::power::horsepower
        ),
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
                    // Collect any extra name aliases (alternative prefixes) from the field.
                    // Each alias is treated as an alternative field-name prefix: the macro
                    // expands it into unit-suffixed variants for every unit, mirroring how the
                    // canonical name is expanded.  The bare alias also routes to the base unit.
                    //
                    // Example: `budget_power: si::Power` with `#[serde(alias = "budget")]`
                    //   primary helper  → rename "budget_power_watts",  alias "budget_power",
                    //                     alias "budget" (bare), alias "budget_watts"
                    //   alternate helper → rename "budget_power_kilowatts", alias "budget_kilowatts"
                    let extra_aliases = extract_serde_aliases(field);

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

                        // The serde key is "field_unit"
                        let serde_key = format!("{}_{}", field_ident, unit_name);

                        // For Ratio quantities, the canonical serialized name is the bare field
                        // name (e.g. `grade`, not `grade_ratio`) for backward compatibility.
                        // The unit-suffixed name becomes an alias instead.
                        // For all other quantities, field_unit is canonical and bare is alias.
                        let use_bare_as_canonical =
                            si_field.quantity == "Ratio" && unit_name == "ratio";
                        let (canonical_name, unit_alias) = if use_bare_as_canonical {
                            (bare_name.clone(), serde_key.clone())
                        } else {
                            (serde_key.clone(), bare_name.clone())
                        };

                        // The primary unit (first in list) also accepts the bare field name as
                        // alias.  Extra aliases are expanded: each alias A contributes
                        // alias "A_{unit}" to this unit's helper, plus bare "A" on primary.
                        let expanded_aliases: Vec<String> = if idx == 0 {
                            extra_aliases
                                .iter()
                                .flat_map(|a| [a.clone(), format!("{}_{}", a, unit_name)])
                                .collect()
                        } else {
                            extra_aliases
                                .iter()
                                .map(|a| format!("{}_{}", a, unit_name))
                                .collect()
                        };

                        let serde_attr = if idx == 0 {
                            let extra = std::iter::once(unit_alias.as_str())
                                .chain(expanded_aliases.iter().map(String::as_str))
                                .map(|a| quote! { , alias = #a });
                            quote! { #[serde(default, rename = #canonical_name #(#extra)*)] }
                        } else if expanded_aliases.is_empty() {
                            quote! { #[serde(default, rename = #serde_key)] }
                        } else {
                            let extra = expanded_aliases.iter().map(|a| quote! { , alias = #a });
                            quote! { #[serde(default, rename = #serde_key #(#extra)*)] }
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

    // Propagate struct-level serde attributes to the helper.
    let deny_unknown = if has_serde_deny_unknown_fields(struct_ast) {
        quote! { #[serde(deny_unknown_fields)] }
    } else {
        quote! {}
    };
    let (struct_default_attr, default_derive) = if has_serde_struct_default(struct_ast) {
        (quote! { #[serde(default)] }, quote! { Default, })
    } else {
        (quote! {}, quote! {})
    };

    quote! {
        #[derive(::serde::Deserialize, #default_derive)]
        #[serde(crate = "::serde")]
        #deny_unknown
        #struct_default_attr
        struct #helper_name {
            #(#helper_fields),*
        }
    }
}

/// Generate `impl TryFrom<Helper> for Struct` that converts the helper's optional
/// unit-variant fields into the original struct's SI field values.
///
/// Returns `Err(String)` (surfaced by serde as a proper deserialization error with
/// field context) when a required SI field has no populated unit variant in the input.
pub fn generate_try_from_impl(
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
                            if has_serde_struct_default(struct_ast) {
                                quote! {
                                    (#conversion).map(|val| #ts_path::new(val))
                                        .unwrap_or_default()
                                }
                            } else {
                                quote! {
                                    (#conversion).map(|val| #ts_path::new(val))
                                        .ok_or_else(|| format!("Missing field {} in any unit variant", stringify!(#field_ident)))?
                                }
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
                            if has_serde_struct_default(struct_ast) {
                                quote! {
                                    (#conversion).unwrap_or_default()
                                }
                            } else {
                                quote! {
                                    (#conversion).ok_or_else(|| format!("Missing field {} in any unit variant", stringify!(#field_ident)))?
                                }
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
        impl ::std::convert::TryFrom<#helper_name> for #original_name {
            type Error = ::std::string::String;

            fn try_from(helper: #helper_name) -> ::std::result::Result<Self, ::std::string::String> {
                ::std::result::Result::Ok(Self {
                    #(#field_conversions),*
                })
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

/// Extract all `alias = "..."` string values from a field's `#[serde(...)]` attributes.
fn extract_serde_aliases(field: &syn::Field) -> Vec<String> {
    let mut aliases = vec![];
    for attr in &field.attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("alias") && meta.input.peek(syn::Token![=]) {
                let _: syn::Token![=] = meta.input.parse().unwrap();
                let lit: syn::LitStr = meta.input.parse().unwrap();
                aliases.push(lit.value());
            } else if meta.input.peek(syn::Token![=]) {
                // consume other "= value" tokens so parse_nested_meta doesn't error
                let _: syn::Token![=] = meta.input.parse().unwrap();
                let _: proc_macro2::TokenTree = meta.input.parse().unwrap();
            }
            Ok(())
        });
    }
    aliases
}

/// Returns true if the struct has `#[serde(deny_unknown_fields)]`.
fn has_serde_deny_unknown_fields(struct_ast: &syn::ItemStruct) -> bool {
    has_serde_struct_flag(struct_ast, "deny_unknown_fields")
}

/// Returns true if the struct has `#[serde(default)]` at the struct level.
fn has_serde_struct_default(struct_ast: &syn::ItemStruct) -> bool {
    has_serde_struct_flag(struct_ast, "default")
}

fn has_serde_struct_flag(struct_ast: &syn::ItemStruct, flag: &str) -> bool {
    struct_ast.attrs.iter().any(|attr| {
        if !attr.path().is_ident("serde") {
            return false;
        }
        let mut found = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(flag) {
                found = true;
            }
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
