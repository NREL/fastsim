use crate::imports::*;

/// Converts multiple uom unit values to a vector of token stream and the plural units name
///
/// - field_units: unit type of value being set (e.g. `uom::si::power::watt`)
macro_rules! extract_units {
    ($($field_units: ty),+) => {{
        let mut unit_impls = vec![];
        $(
            let field_units: TokenStream2 = stringify!($field_units).parse().expect("failed to parse `field_units`");
            let unit_name = <$field_units as uom::si::Unit>::plural().replace(' ', "_");
            // fix the UOM Kelvin atrocity
            let unit_name = unit_name.replace("kelvins", "kelvin");
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
