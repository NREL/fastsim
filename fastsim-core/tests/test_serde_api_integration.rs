//! Integration tests for #[serde_api] macro demonstrating multi-unit deserialization
//!
//! This test suite validates the wrapper-based approach to multi-unit JSON deserialization.
//!
//! The macro applies a wrapper struct pattern to enable multi-unit JSON support for SI quantities:
//! - Each SI field generates deserializer options for all supported unit variants
//! - JSON can use ANY unit variant (primary or alternate) and values auto-convert to SI base units
//! - Serialization always outputs primary units for consistency
//!
//! **Note on Optional/Vector Fields:**
//! Optional<SI> and Vec<SI> field support has been implemented and these tests validate it.

use fastsim_core::si;
use fastsim_core::utils::tracked_state::TrackedState;
use fastsim_proc_macros::serde_api;
use serde::{Deserialize, Serialize};

// ============================================================================
// TEST STRUCT: Pure SI Quantities with Optional and Vec fields
// ============================================================================

/// Struct for testing multi-unit deserialization with required, optional, and vector SI fields
#[serde_api]
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
struct TestDevice {
    /// Mass in kilograms - supports kilogram (primary)
    mass: si::Mass,

    /// Power in watts - supports watt and kilowatt
    power: si::Power,

    /// Velocity in m/s - supports meter_per_second, kilometer_per_hour, and mile_per_hour
    speed: si::Velocity,

    /// Time in seconds - supports second and hour
    duration: si::Time,

    /// Area in square meters
    area: si::Area,

    /// Temperature in Kelvin - supports kelvin, degree_celsius, and degree_fahrenheit
    temperature: si::Temperature,

    /// Energy in joules - supports joule and kilowatt_hour
    energy: si::Energy,

    /// Optional secondary temperature - supports kelvin, degree_celsius, and degree_fahrenheit
    secondary_temperature: Option<si::Temperature>,

    /// Vector of temperatures for multiple readings - supports kelvin, degree_celsius, and degree_fahrenheit
    temperature_readings: Vec<si::Temperature>,

    /// Efficiency ratio - supports ratio and percent
    efficiency: si::Ratio,

    /// Tracked power state, wrapped in TrackedState - accepts watts or kilowatts
    tracked_power: TrackedState<si::Power>,

    /// Tracked efficiency state, wrapped in TrackedState - accepts ratio or percent
    tracked_efficiency: TrackedState<si::Ratio>,

    /// Renamed from "old_power" - accepts old_power, old_power_watts, old_power_kilowatts, old_power_horsepower,
    /// as well as renamed_power, renamed_power_watts, renamed_power_kilowatts, renamed_power_horsepower
    #[serde(alias = "old_power")]
    renamed_power: si::Power,
}

// ============================================================================
// TESTS: Pure SI Struct with Wrapper (8 tests - all passing)
// ============================================================================

#[test]
fn test_deserialize_all_primary_units() {
    let json = r#"{
        "mass_kilograms": 2000.0,
        "power_watts": 150.0,
        "speed_meters_per_second": 30.0,
        "duration_seconds": 120.0,
        "area_square_meters": 5.0,
        "temperature_kelvin": 300.0,
        "energy_joules": 1000.0,
        "efficiency_ratio": 0.85,
        "temperature_readings_kelvin": [290.0, 300.0, 310.0],
        "tracked_power_watts": 100.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize with primary units");

    assert_eq!(device.mass.get::<si::kilogram>(), 2000.0);
    assert_eq!(device.power.get::<si::watt>(), 150.0);
    assert_eq!(device.speed.get::<si::meter_per_second>(), 30.0);
    assert_eq!(device.duration.get::<si::second>(), 120.0);
    assert_eq!(device.area.get::<si::square_meter>(), 5.0);
    assert_eq!(device.temperature.get::<si::kelvin_abs>(), 300.0);
    assert_eq!(device.energy.get::<si::joule>(), 1000.0);
    assert_eq!(device.efficiency.get::<si::ratio>(), 0.85);
    assert_eq!(device.secondary_temperature, None);
    assert_eq!(device.temperature_readings.len(), 3);
    assert_eq!(
        device
            .tracked_power
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::watt>(),
        100.0
    );
    assert_eq!(
        device
            .tracked_efficiency
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::ratio>(),
        0.90
    );
}

#[test]
fn test_deserialize_fails_with_missing_required_fields() {
    let json = r#"{
        "mass_kilograms": 2000.0,
        "power_watts": 150.0,
        "speed_meters_per_second": 30.0,
        "duration_seconds": 120.0,
        "area_square_meters": 5.0,
        "energy_joules": 1000.0,
        "efficiency_ratio": 0.85,
        "tracked_power_watts": 100.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 50.0
    }"#;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: TestDevice = serde_json::from_str(json).unwrap();
    }));

    assert!(
        result.is_err(),
        "Should panic when required temperature field is missing"
    );
}

#[test]
fn test_deserialize_with_alternate_units() {
    // **KEY TEST**: Multi-unit JSON deserialization works!
    // JSON uses alternate unit field names and values automatically convert to SI base units
    // Also demonstrates Option and Vec fields working with alternate units
    let json = r#"{
        "mass_kilograms": 2000.0,
        "power_kilowatts": 0.15,
        "speed_kilometers_per_hour": 108.0,
        "duration_hours": 2.0,
        "area_square_meters": 5.0,
        "temperature_kelvin": 300.15,
        "energy_kilowatt_hours": 1.0,
        "efficiency_percent": 85.0,
        "secondary_temperature_degrees_celsius": 25.0,
        "temperature_readings_degrees_fahrenheit": [32.0, 68.0, 104.0],
        "tracked_power_kilowatts": 0.5,
        "tracked_efficiency_percent": 80.0,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize with alternate units");

    // Verify unit conversions
    assert_eq!(device.power.get::<si::watt>(), 150.0); // 0.15 kW
    let speed = device.speed.get::<si::meter_per_second>();
    assert!((speed - 30.0).abs() < 0.1); // 108 km/h ≈ 30 m/s
    assert_eq!(device.energy.get::<si::joule>(), 3_600_000.0); // 1 kWh
    assert_eq!(device.duration.get::<si::second>(), 7_200.0); // 2 hours
    assert_eq!(device.efficiency.get::<si::ratio>(), 0.85); // 85%
    assert!(
        (device
            .tracked_power
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::watt>()
            - 500.0)
            .abs()
            < 1e-9
    ); // 0.5 kW
    assert!(
        (device
            .tracked_efficiency
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::ratio>()
            - 0.80)
            .abs()
            < 1e-9
    ); // 80%

    // Verify Optional field with alternate unit (celsius -> kelvin conversion)
    assert!(device.secondary_temperature.is_some());
    let secondary_temp = device.secondary_temperature.unwrap();
    assert!((secondary_temp.get::<si::kelvin_abs>() - 298.15).abs() < 0.01); // 25°C ≈ 298.15K

    // Verify Vec field with alternate units (fahrenheit -> kelvin)
    assert_eq!(device.temperature_readings.len(), 3);
    // 32°F = 273.15K, 68°F ≈ 293.15K, 104°F ≈ 313.15K
    assert!((device.temperature_readings[0].get::<si::kelvin_abs>() - 273.15).abs() < 0.5);
    assert!((device.temperature_readings[1].get::<si::kelvin_abs>() - 293.15).abs() < 0.5);
    assert!((device.temperature_readings[2].get::<si::kelvin_abs>() - 313.15).abs() < 0.5);
}

#[test]
fn test_deserialize_time_in_hours() {
    let json = r#"{
        "mass_kilograms": 2000.0,
        "power_watts": 150.0,
        "speed_meters_per_second": 30.0,
        "duration_hours": 2.0,
        "area_square_meters": 5.0,
        "temperature_kelvin": 300.0,
        "energy_joules": 1000.0,
        "efficiency_ratio": 0.85,
        "tracked_power_watts": 100.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize with primary units");

    assert_eq!(device.duration.get::<si::second>(), 7_200.0); // 2 hours
}

#[test]
fn test_deserialize_multiple_si_quantities() {
    let json = r#"{
        "mass_kilograms": 1500.0,
        "power_watts": 200.0,
        "speed_meters_per_second": 25.0,
        "duration_seconds": 300.0,
        "area_square_meters": 10.0,
        "temperature_kelvin": 298.15,
        "energy_joules": 5000.0,
        "efficiency_ratio": 0.90,
        "tracked_power_watts": 50.0,
        "tracked_efficiency_ratio": 0.88,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize multiple SI quantities");

    assert_eq!(device.mass.get::<si::kilogram>(), 1500.0);
    assert_eq!(device.power.get::<si::watt>(), 200.0);
    assert_eq!(device.speed.get::<si::meter_per_second>(), 25.0);
    assert_eq!(device.duration.get::<si::second>(), 300.0);
    assert_eq!(device.area.get::<si::square_meter>(), 10.0);
    assert_eq!(device.temperature.get::<si::kelvin_abs>(), 298.15);
    assert_eq!(device.energy.get::<si::joule>(), 5000.0);
    assert_eq!(device.efficiency.get::<si::ratio>(), 0.90);
}

#[test]
fn test_serialize_uses_primary_units() {
    let json = r#"{
        "mass_kilograms": 2000.0,
        "power_watts": 150.0,
        "speed_meters_per_second": 30.0,
        "duration_seconds": 120.0,
        "area_square_meters": 5.0,
        "temperature_kelvin": 300.0,
        "energy_joules": 1000.0,
        "efficiency_ratio": 0.85,
        "temperature_readings_kelvin": [290.0, 300.0, 310.0],
        "tracked_power_watts": 100.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice = serde_json::from_str(json).expect("Failed to deserialize");

    let serialized = serde_json::to_string(&device).expect("Failed to serialize");

    let value: serde_json::Value =
        serde_json::from_str(&serialized).expect("Failed to parse serialized JSON");

    // Serialization always uses primary units (kg, W, m/s, s, m², kelvin, J, ratio, [kelvin])
    // Ratio fields serialize as bare name (no _ratio suffix) for backward compatibility
    assert!(value.get("mass_kilograms").is_some());
    assert!(value.get("power_watts").is_some());
    assert!(value.get("speed_meters_per_second").is_some());
    assert!(value.get("duration_seconds").is_some());
    assert!(value.get("area_square_meters").is_some());
    assert!(value.get("temperature_kelvin").is_some());
    assert!(value.get("energy_joules").is_some());
    assert!(value.get("efficiency").is_some()); // Ratio: bare name
    assert!(value.get("tracked_power_watts").is_some());
    assert!(value.get("tracked_efficiency").is_some()); // Ratio: bare name
    assert!(value.get("renamed_power_watts").is_some());
}

#[test]
fn test_round_trip_conversion_preserves_values() {
    let json_input = r#"{
        "mass_kilograms": 2000.0,
        "power_watts": 150.0,
        "speed_meters_per_second": 30.0,
        "duration_seconds": 3600.0,
        "area_square_meters": 5.0,
        "temperature_kelvin": 300.0,
        "energy_joules": 1000.0,
        "efficiency_ratio": 0.85,
        "temperature_readings_kelvin": [290.0, 300.0, 310.0],
        "tracked_power_watts": 100.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice = serde_json::from_str(json_input).expect("Failed to deserialize");

    let serialized = serde_json::to_string(&device).expect("Failed to serialize");

    let value: serde_json::Value =
        serde_json::from_str(&serialized).expect("Failed to parse serialized JSON");

    // Values are preserved through round-trip (in primary units)
    // Ratio fields round-trip using bare name
    assert_eq!(value["mass_kilograms"], 2000.0);
    assert_eq!(value["power_watts"], 150.0);
    assert_eq!(value["speed_meters_per_second"], 30.0);
    assert_eq!(value["duration_seconds"], 3600.0);
    assert_eq!(value["area_square_meters"], 5.0);
    assert_eq!(value["temperature_kelvin"], 300.0);
    assert_eq!(value["energy_joules"], 1000.0);
    assert_eq!(value["efficiency"], 0.85); // Ratio: bare name
}

#[test]
fn test_deserialize_with_different_values() {
    let json = r#"{
        "mass_kilograms": 5000.0,
        "power_watts": 500.0,
        "speed_meters_per_second": 50.0,
        "duration_seconds": 600.0,
        "area_square_meters": 20.0,
        "temperature_kelvin": 400.0,
        "energy_joules": 10000.0,
        "efficiency_ratio": 0.95,
        "temperature_readings_kelvin": [380.0, 400.0, 420.0],
        "tracked_power_watts": 200.0,
        "tracked_efficiency_ratio": 0.75,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize with different values");

    assert_eq!(device.mass.get::<si::kilogram>(), 5000.0);
    assert_eq!(device.power.get::<si::watt>(), 500.0);
    assert_eq!(device.speed.get::<si::meter_per_second>(), 50.0);
    assert_eq!(device.duration.get::<si::second>(), 600.0);
    assert_eq!(device.area.get::<si::square_meter>(), 20.0);
    assert_eq!(device.temperature.get::<si::kelvin_abs>(), 400.0);
    assert_eq!(device.energy.get::<si::joule>(), 10000.0);
}

// ============================================================================
// TESTS: TrackedState fields (using TestDevice)
// ============================================================================

#[test]
fn test_tracked_state_deserialize_primary_units() {
    let json = r#"{
        "mass_kilograms": 1000.0,
        "power_watts": 100.0,
        "speed_meters_per_second": 10.0,
        "duration_seconds": 60.0,
        "area_square_meters": 2.0,
        "temperature_kelvin": 293.0,
        "energy_joules": 500.0,
        "efficiency_ratio": 0.80,
        "tracked_power_watts": 250.0,
        "tracked_efficiency_ratio": 0.92,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize TestDevice primary units");

    assert_eq!(
        device
            .tracked_power
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::watt>(),
        250.0
    );
    assert_eq!(
        device
            .tracked_efficiency
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::ratio>(),
        0.92
    );
}

#[test]
fn test_tracked_state_deserialize_alternate_units() {
    let json = r#"{
        "mass_kilograms": 1000.0,
        "power_watts": 100.0,
        "speed_meters_per_second": 10.0,
        "duration_seconds": 60.0,
        "area_square_meters": 2.0,
        "temperature_kelvin": 293.0,
        "energy_joules": 500.0,
        "efficiency_ratio": 0.80,
        "tracked_power_kilowatts": 0.5,
        "tracked_efficiency_percent": 85.0,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize TestDevice alternate units");

    assert!(
        (device
            .tracked_power
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::watt>()
            - 500.0)
            .abs()
            < 1e-9
    );
    assert!(
        (device
            .tracked_efficiency
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::ratio>()
            - 0.85)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_tracked_state_deserialize_bare_name() {
    // Bare field names (without unit suffix) are accepted as aliases for the primary unit
    let json = r#"{
        "mass_kilograms": 1000.0,
        "power_watts": 100.0,
        "speed_meters_per_second": 10.0,
        "duration_seconds": 60.0,
        "area_square_meters": 2.0,
        "temperature_kelvin": 293.0,
        "energy_joules": 500.0,
        "efficiency_ratio": 0.80,
        "tracked_power": 300.0,
        "tracked_efficiency": 0.78,
        "renamed_power_watts": 50.0
    }"#;

    let device: TestDevice =
        serde_json::from_str(json).expect("Failed to deserialize TestDevice with bare names");

    assert_eq!(
        device
            .tracked_power
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::watt>(),
        300.0
    );
    assert_eq!(
        device
            .tracked_efficiency
            .get_fresh(|| "".into())
            .unwrap()
            .get::<si::ratio>(),
        0.78
    );
}

// ============================================================================
// TEST: Drive cycle CSV loading with bare "grade" column
// ============================================================================

#[test]
#[cfg(feature = "csv")]
fn test_cycle_csv_loads_bare_grade_column() {
    use fastsim_core::drive_cycle::CycleElement;

    // CSV with bare "grade" column (no unit suffix) - the real-world format used by
    // fastsim CSV files like udds.csv and hwfet.csv
    let csv = "time_seconds,speed_meters_per_second,grade\n\
               0,0,0\n\
               1,2.5,0.01\n\
               2,5.0,-0.005\n";

    let mut rdr = csv::Reader::from_reader(csv.as_bytes());
    let elements: Vec<CycleElement> = rdr
        .deserialize()
        .map(|r| r.expect("Failed to deserialize CycleElement row"))
        .collect();

    assert_eq!(elements.len(), 3);
    assert_eq!(elements[0].time.get::<si::second>(), 0.0);
    assert_eq!(elements[1].speed.get::<si::meter_per_second>(), 2.5);
    // grade in ratio units (0.01 = 1% grade)
    assert!((elements[1].grade.unwrap().get::<si::ratio>() - 0.01).abs() < 1e-10);
    assert!((elements[2].grade.unwrap().get::<si::ratio>() - (-0.005)).abs() < 1e-10);
}

#[test]
#[cfg(feature = "csv")]
fn test_cycle_csv_loads_grade_ratio_column() {
    use fastsim_core::drive_cycle::CycleElement;

    // CSV with explicit "grade_ratio" column (new unit-suffixed format)
    let csv = "time_seconds,speed_meters_per_second,grade_ratio\n\
               0,0,0\n\
               1,10.0,0.02\n";

    let mut rdr = csv::Reader::from_reader(csv.as_bytes());
    let elements: Vec<CycleElement> = rdr
        .deserialize()
        .map(|r| r.expect("Failed to deserialize CycleElement row"))
        .collect();

    assert_eq!(elements.len(), 2);
    assert!((elements[1].grade.unwrap().get::<si::ratio>() - 0.02).abs() < 1e-10);
}

#[test]
#[cfg(feature = "csv")]
fn test_cycle_csv_legacy_aliases() {
    use fastsim_core::drive_cycle::CycleElement;

    // CSV with legacy fastsim2 column names: cycSecs, cycMps, cycGrade
    // These were serde aliases on the original fields and must still be forwarded
    // through the helper struct so old CSV files keep working.
    let csv = "cycSecs,cycMps,cycGrade\n\
               0,0,0\n\
               1,5.0,0.05\n\
               2,10.0,-0.02\n";

    let mut rdr = csv::Reader::from_reader(csv.as_bytes());
    let elements: Vec<CycleElement> = rdr
        .deserialize()
        .map(|r| r.expect("Failed to deserialize CycleElement row with legacy aliases"))
        .collect();

    assert_eq!(elements.len(), 3);
    assert_eq!(elements[0].time.get::<si::second>(), 0.0);
    assert_eq!(elements[1].speed.get::<si::meter_per_second>(), 5.0);
    assert!((elements[1].grade.unwrap().get::<si::ratio>() - 0.05).abs() < 1e-10);
    assert!((elements[2].grade.unwrap().get::<si::ratio>() - (-0.02)).abs() < 1e-10);
}

// ============================================================================
// TEST: alias = name prefix expands to all unit variants (backward compat rename)
// ============================================================================

#[test]
fn test_alias_prefix_expands_to_all_unit_variants() {
    let base_json = r#"{
        "mass_kilograms": 1000.0,
        "power_watts": 100.0,
        "speed_meters_per_second": 10.0,
        "duration_seconds": 60.0,
        "area_square_meters": 2.0,
        "temperature_kelvin": 293.0,
        "energy_joules": 500.0,
        "efficiency_ratio": 0.80,
        "tracked_power_watts": 50.0,
        "tracked_efficiency_ratio": 0.90,
        "renamed_power_watts": 0.0
    }"#;

    // Old bare name routes to base unit (watts)
    let json = base_json.replace(r#""renamed_power_watts": 0.0"#, r#""old_power": 200.0"#);
    let d: TestDevice = serde_json::from_str(&json).expect("old_power (bare) should deserialize");
    assert_eq!(d.renamed_power.get::<si::watt>(), 200.0);

    // Old name + _watts suffix
    let json = base_json.replace(
        r#""renamed_power_watts": 0.0"#,
        r#""old_power_watts": 300.0"#,
    );
    let d: TestDevice = serde_json::from_str(&json).expect("old_power_watts should deserialize");
    assert_eq!(d.renamed_power.get::<si::watt>(), 300.0);

    // Old name + _kilowatts suffix — converts correctly
    let json = base_json.replace(
        r#""renamed_power_watts": 0.0"#,
        r#""old_power_kilowatts": 0.5"#,
    );
    let d: TestDevice =
        serde_json::from_str(&json).expect("old_power_kilowatts should deserialize");
    assert!((d.renamed_power.get::<si::watt>() - 500.0).abs() < 1e-9); // 0.5 kW = 500 W

    // Old name + _horsepower suffix — converts correctly
    let json = base_json.replace(
        r#""renamed_power_watts": 0.0"#,
        r#""old_power_horsepower": 1.0"#,
    );
    let d: TestDevice =
        serde_json::from_str(&json).expect("old_power_horsepower should deserialize");
    assert!((d.renamed_power.get::<si::watt>() - 745.7).abs() < 0.1); // 1 hp ≈ 745.7 W

    // New canonical name still works
    let json = base_json.replace(
        r#""renamed_power_watts": 0.0"#,
        r#""renamed_power_kilowatts": 0.15"#,
    );
    let d: TestDevice =
        serde_json::from_str(&json).expect("renamed_power_kilowatts should deserialize");
    assert!((d.renamed_power.get::<si::watt>() - 150.0).abs() < 1e-9);
}
