use crate::imports::*;
use std::collections::HashMap;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, SerdeAPI)]
#[altrios_api]
pub struct RailVehicle {
    /// Unique user-defined identifier for the car type
    #[serde(rename = "Car Type")]
    pub car_type: String,

    /// Railcar length (between pulling-faces)
    #[serde(rename = "Length (m)")]
    pub length: si::Length,
    /// Railcar axle count (typically 4)
    #[serde(rename = "Axle Count")]
    pub axle_count: u8,
    /// Brake valve count (typically 1)
    #[serde(rename = "Brake Count")]
    pub brake_count: u8,

    /// Railcar empty mass (excluding freight)
    #[serde(rename = "Mass Static Empty (kg)")]
    pub mass_static_empty: si::Mass,
    /// Railcar loaded mass (including freight)
    #[serde(rename = "Mass Static Loaded (kg)")]
    pub mass_static_loaded: si::Mass,
    /// Railcar speed limit when empty
    #[serde(rename = "Speed Max Empty (m/s)")]
    pub speed_max_empty: si::Velocity,
    /// Railcar speed limit when loaded
    #[serde(rename = "Speed Max Loaded (m/s)")]
    pub speed_max_loaded: si::Velocity,
    /// Braking ratio at empty mass
    #[serde(rename = "Braking Ratio Empty")]
    pub braking_ratio_empty: si::Ratio,
    /// Braking ratio at loaded mass
    #[serde(rename = "Braking Ratio Loaded")]
    pub braking_ratio_loaded: si::Ratio,

    /// Additional mass value to adjust for rotating mass in wheels and axles (typically 1,500 lbs)
    #[serde(rename = "Mass Extra per Axle (kg)")]
    pub mass_extra_per_axle: si::Mass,
    /// Bearing resistance as force
    #[serde(rename = "Bearing Res per Axle (N)")]
    pub bearing_res_per_axle: si::Force,
    /// Rolling resistance ratio (lb/ton is customary, lb/lb internal to code)
    #[serde(rename = "Rolling Ratio")]
    pub rolling_ratio: si::Ratio,
    /// Davis B coefficient (typically very close to zero)
    #[serde(rename = "Davis B (s/m)")]
    pub davis_b: si::InverseVelocity,
    /// Drag area (Cd*A) when empty, where Cd is drag coefficient and A is front cross-sectional area
    #[serde(rename = "Drag Area Cd*A Empty (m^2)")]
    pub drag_area_empty: si::Area,
    /// Drag area (Cd*A) when loaded, where Cd is drag coefficient and A is front cross-sectional area
    #[serde(rename = "Drag Area Cd*A Loaded (m^2)")]
    pub drag_area_loaded: si::Area,
    /// Curve coefficient 0
    #[serde(rename = "Curve Coefficient 0")]
    pub curve_coeff_0: si::Ratio,
    /// Curve coefficient 1
    #[serde(rename = "Curve Coefficient 1")]
    pub curve_coeff_1: si::Ratio,
    /// Curve coefficient 2
    #[serde(rename = "Curve Coefficient 2")]
    pub curve_coeff_2: si::Ratio,
}

pub type RailVehicleMap = HashMap<String, RailVehicle>;

#[cfg_attr(feature = "pyo3", pyfunction(name = "import_rail_vehicles"))]
pub fn import_rail_vehicles_py(filename: String) -> anyhow::Result<RailVehicleMap> {
    import_rail_vehicles(&PathBuf::from(filename))
}

pub fn import_rail_vehicles(filename: &Path) -> anyhow::Result<RailVehicleMap> {
    let file_read = File::open(filename)?;
    let mut reader = csv::Reader::from_reader(file_read);
    let mut rail_vehicle_map = RailVehicleMap::default();
    for result in reader.deserialize() {
        let rail_vehicle: RailVehicle = result?;
        rail_vehicle_map.insert(rail_vehicle.car_type.clone(), rail_vehicle);
    }
    Ok(rail_vehicle_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicles_import() {
        import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap();
    }
}
