use std::path::Path;

use fastsim_core::traits::*;
use fastsim_core::*;

const REFERENCE_VEHICLE: &str = include_str!("assets/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml");

#[test]
#[cfg(feature = "default")]
fn test_from_cache() {
    let test_path = "1110_2022_Tesla_Model_Y_RWD_opt45017_from_cache.yaml";
    let comparison_vehicle =
        crate::vehicle::RustVehicle::from_yaml(REFERENCE_VEHICLE, false).unwrap();
    crate::vehicle::RustVehicle::to_cache(&comparison_vehicle, test_path).unwrap();
    let vehicle = crate::vehicle::RustVehicle::from_cache(test_path, false).unwrap();
    assert_eq!(comparison_vehicle, vehicle);
    let full_file_path = Path::new("vehicles").join(test_path);
    let path_including_directory = utils::path_to_cache().unwrap().join(full_file_path);
    std::fs::remove_file(path_including_directory).unwrap();
}

#[test]
#[cfg(feature = "default")]
fn test_to_cache() {
    let comparison_vehicle =
        crate::vehicle::RustVehicle::from_yaml(REFERENCE_VEHICLE, false).unwrap();
    crate::vehicle::RustVehicle::to_cache(
        &comparison_vehicle,
        "1110_2022_Tesla_Model_Y_RWD_opt45017.yaml",
    )
    .unwrap();
    let data_subdirectory = utils::create_project_subdir("vehicles").unwrap();
    let file_path = data_subdirectory.join("1110_2022_Tesla_Model_Y_RWD_opt45017.yaml");
    println!("{}", file_path.to_string_lossy());
    println!("{}", crate::vehicle::RustVehicle::CACHE_FOLDER);
    let vehicle_b = crate::vehicle::RustVehicle::from_file(&file_path, false).unwrap();
    assert_eq!(comparison_vehicle, vehicle_b);
    std::fs::remove_file(file_path).unwrap();
}

#[test]
#[cfg(all(feature = "directories", feature = "url"))]
fn test_url_to_cache() {
    utils::url_to_cache("https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml", "vehicles").unwrap();
    let data_subdirectory = utils::create_project_subdir("vehicles").unwrap();
    let file_path = data_subdirectory.join("1110_2022_Tesla_Model_Y_RWD_opt45017.yaml");
    println!("{}", file_path.to_string_lossy());
    let vehicle = crate::vehicle::RustVehicle::from_file(&file_path, false).unwrap();
    let comparison_vehicle =
        crate::vehicle::RustVehicle::from_yaml(REFERENCE_VEHICLE, false).unwrap();
    assert_eq!(vehicle, comparison_vehicle);
    std::fs::remove_file(file_path).unwrap();
}

#[test]
fn test_from_github_or_url() {
    let mut comparison_vehicle = vehicle::RustVehicle::from_yaml(REFERENCE_VEHICLE, false).unwrap();
    comparison_vehicle.doc = Some("Vehicle from https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml".to_owned());
    // test no url provided
    let vehicle = vehicle::RustVehicle::from_github_or_url(
        "public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml",
        None,
    )
    .unwrap();
    assert_eq!(vehicle, comparison_vehicle);
    // test url provided
    let vehicle_1 = vehicle::RustVehicle::from_github_or_url(
        "1110_2022_Tesla_Model_Y_RWD_opt45017.yaml",
        Some("https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/"),
    )
    .unwrap();
    assert_eq!(vehicle_1, comparison_vehicle);
    let vehicle_2 = vehicle::RustVehicle::from_github_or_url(
        "public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml",
        Some("https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/"),
    )
    .unwrap();
    assert_eq!(vehicle_2, comparison_vehicle);
}

#[test]
fn test_from_url() {
    let vehicle = vehicle::RustVehicle::from_url("https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml", false).unwrap();
    let mut comparison_vehicle = vehicle::RustVehicle::from_yaml(REFERENCE_VEHICLE, false).unwrap();
    comparison_vehicle.doc = Some("Vehicle from https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml".to_owned());
    assert_eq!(vehicle, comparison_vehicle);
}
