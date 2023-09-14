//! Module for utility functions that support the vehicle struct.

use argmin::core::{CostFunction, Error, Executor, OptimizationResult, State};
use argmin::solver::neldermead::NelderMead;
use curl::easy::Easy;
use directories::ProjectDirs;
use ndarray::{array, Array1};
use polynomial::Polynomial;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::Write;
use std::io::Read;
use std::iter::FromIterator;
use std::num::ParseIntError;
use std::option::Option;
use std::path::PathBuf;
use zip::ZipArchive;

use crate::air::*;
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::add_pyo3_api;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::simdrive::RustSimDrive;
use crate::vehicle::RustVehicle;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing list of makes for a year from fueleconomy.gov
struct VehicleMakesFE {
    #[serde(rename = "menuItem")]
    /// List of vehicle makes
    makes: Vec<MakeFE>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing make information for a year fueleconomy.gov
struct MakeFE {
    #[serde(rename = "text")]
    /// Transmission of vehicle
    make_name: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing list of models for a year and make from fueleconomy.gov
struct VehicleModelsFE {
    #[serde(rename = "menuItem")]
    /// List of vehicle models
    models: Vec<ModelFE>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing model information for a year and make from fueleconomy.gov
struct ModelFE {
    #[serde(rename = "text")]
    /// Transmission of vehicle
    model_name: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing list of transmission options for vehicle from fueleconomy.gov
struct VehicleOptionsFE {
    #[serde(rename = "menuItem")]
    /// List of vehicle options (transmission and id)
    options: Vec<OptionFE>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[add_pyo3_api]
/// Struct containing transmission and id of a vehicle option from fueleconomy.gov
pub struct OptionFE {
    #[serde(rename = "text")]
    /// Transmission of vehicle
    pub transmission: String,
    #[serde(rename = "value")]
    /// ID of vehicle on fueleconomy.gov
    pub id: String,
}

impl SerdeAPI for OptionFE {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[add_pyo3_api]
/// Struct containing vehicle data from fueleconomy.gov
pub struct VehicleDataFE {
    /// Vehicle ID
    pub id: i32,
    #[serde(default, rename = "atvType")]
    /// Type of alternative fuel vehicle (Hybrid, Plug-in Hybrid, EV)
    pub alt_veh_type: String,
    #[serde(rename = "city08")]
    /// City MPG for fuel 1
    pub city_mpg_fuel1: i32,
    #[serde(rename = "cityA08")]
    /// City MPG for fuel 2
    pub city_mpg_fuel2: i32,
    #[serde(rename = "co2")]
    /// Tailpipe CO2 emissions in grams/mile
    pub co2_g_per_mi: i32,
    #[serde(rename = "comb08")]
    /// Combined MPG for fuel 1
    pub comb_mpg_fuel1: i32,
    #[serde(rename = "combA08")]
    /// Combined MPG for fuel 2
    pub comb_mpg_fuel2: i32,
    #[serde(default)]
    /// Number of engine cylinders
    pub cylinders: String,
    #[serde(default)]
    /// Engine displacement in liters
    pub displ: String,
    /// Drive axle type (FWD, RWD, AWD, 4WD)
    pub drive: String,
    #[serde(rename = "emissionsList")]
    /// List of emissions tests
    pub emissions_list: EmissionsListFE,
    #[serde(default)]
    /// Description of engine
    pub eng_dscr: String,
    #[serde(default, rename = "evMotor")]
    /// Electric motor power (kW)
    pub ev_motor_kw: String,
    #[serde(rename = "feScore")]
    /// EPA fuel economy score
    pub fe_score: i32,
    #[serde(rename = "fuelType")]
    /// Combined vehicle fuel type (fuel 1 and fuel 2)
    pub fuel_type: String,
    #[serde(rename = "fuelType1")]
    /// Fuel type 1
    pub fuel1: String,
    #[serde(default, rename = "fuelType2")]
    /// Fuel type 2
    pub fuel2: String,
    #[serde(rename = "ghgScore")]
    /// EPA GHG Score
    pub ghg_score: i32,
    #[serde(rename = "highway08")]
    /// Highway MPG for fuel 1
    pub highway_mpg_fuel1: i32,
    #[serde(rename = "highwayA08")]
    /// Highway MPG for fuel 2
    pub highway_mpg_fuel2: i32,
    /// Manufacturer
    pub make: String,
    #[serde(rename = "mfrCode")]
    /// Manufacturer code
    pub mfr_code: String,
    /// Model name
    pub model: String,
    #[serde(rename = "phevBlended")]
    /// Vehicle operates on blend of gasoline and electricity
    pub phev_blended: bool,
    #[serde(rename = "phevCity")]
    /// EPA composite gasoline-electricity city MPGe
    pub phev_city_mpge: i32,
    #[serde(rename = "phevComb")]
    /// EPA composite gasoline-electricity combined MPGe
    pub phev_comb_mpge: i32,
    #[serde(rename = "phevHwy")]
    /// EPA composite gasoline-electricity highway MPGe
    pub phev_hwy_mpge: i32,
    #[serde(rename = "range")]
    /// Range for EV
    pub range_ev: i32,
    #[serde(rename = "startStop")]
    /// Stop-start technology
    pub start_stop: String,
    /// transmission
    pub trany: String,
    #[serde(rename = "VClass")]
    /// EPA vehicle size class
    pub veh_class: String,
    /// Model year
    pub year: u32,
    #[serde(default, rename = "sCharger")]
    /// Vehicle is supercharged
    pub super_charge: String,
    #[serde(default, rename = "tCharger")]
    /// Vehicle is turbocharged
    pub turbo_charge: String,
}

impl SerdeAPI for VehicleDataFE {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "camelCase")]
#[add_pyo3_api]
/// Struct containing list of emissions tests from fueleconomy.gov
pub struct EmissionsListFE {
    ///
    pub emissions_info: Vec<EmissionsInfoFE>,
}

impl SerdeAPI for EmissionsListFE {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "camelCase")]
#[add_pyo3_api]
/// Struct containing emissions test results from fueleconomy.gov
pub struct EmissionsInfoFE {
    /// Engine family id / EPA test group
    pub efid: String,
    /// EPA smog rating
    pub score: f64,
    /// SmartWay score
    pub smartway_score: i32,
    /// Vehicle emission standard code
    pub standard: String,
    /// Vehicle emission standard
    pub std_text: String,
}

impl SerdeAPI for EmissionsInfoFE {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[derive(Default, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[add_pyo3_api]
/// Struct containing vehicle data from EPA database
pub struct VehicleDataEPA {
    #[serde(rename = "Model Year")]
    /// Model year
    pub year: u32,
    #[serde(rename = "Veh Mfr Code")]
    /// Vehicle manufacturer code
    pub mfr_code: String,
    #[serde(rename = "Represented Test Veh Make")]
    /// Vehicle make
    pub make: String,
    #[serde(rename = "Represented Test Veh Model")]
    /// Vehicle model
    pub model: String,
    #[serde(rename = "Actual Tested Testgroup")]
    /// Vehicle test group
    pub test_id: String,
    #[serde(rename = "Test Veh Displacement (L)")]
    /// Engine displacement
    pub displ: f64,
    #[serde(rename = "Rated Horsepower")]
    /// Engine power in hp
    pub eng_pwr_hp: u32,
    #[serde(rename = "# of Cylinders and Rotors")]
    /// Number of cylinders
    pub cylinders: String,
    #[serde(rename = "Tested Transmission Type Code")]
    /// Transmission type code
    pub trany_code: String,
    #[serde(rename = "Tested Transmission Type")]
    /// Transmission type
    pub trany_type: String,
    #[serde(rename = "# of Gears")]
    /// Number of gears
    pub gears: u32,
    #[serde(rename = "Drive System Code")]
    /// Drive system code
    pub drive_code: String,
    #[serde(rename = "Drive System Description")]
    /// Drive system type
    pub drive: String,
    #[serde(rename = "Equivalent Test Weight (lbs.)")]
    /// Test weight in lbs
    pub test_weight_lbs: f64,
    #[serde(rename = "Test Fuel Type Description")]
    /// Fuel type used for EPA test
    pub test_fuel_type: String,
    #[serde(rename = "Target Coef A (lbf)")]
    /// Dyno coefficient a in lbf
    pub a_lbf: f64,
    #[serde(rename = "Target Coef B (lbf/mph)")]
    /// Dyno coefficient b in lbf/mph
    pub b_lbf_per_mph: f64,
    #[serde(rename = "Target Coef C (lbf/mph**2)")]
    /// Dyno coefficient c in lbf/mph^2
    pub c_lbf_per_mph2: f64,
}

impl SerdeAPI for VehicleDataEPA {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Gets options from fueleconomy.gov for the given vehicle year, make, and model
///
/// Arguments:
/// ----------
/// year: Vehicle year
/// make: Vehicle make
/// model: Vehicle model (must match model on fueleconomy.gov)
///
/// Returns:
/// --------
/// Vec<OptionFE>: Data for the available options for that vehicle year/make/model from fueleconomy.gov
pub fn get_options_for_year_make_model(
    year: &str,
    make: &str,
    model: &str,
    cache_url: Option<String>,
    data_dir: Option<String>,
) -> Result<Vec<VehicleDataFE>, Error> {
    // prep the cache for year
    let y: u32 = year.trim().parse()?;
    let ys: HashSet<u32> = {
        let mut h = HashSet::new();
        h.insert(y);
        h
    };
    let ddpath = if let Some(dd) = &data_dir {
        let ddpath = Path::new(dd);
        ddpath.to_path_buf()
    } else if let Some(ddpath) = get_fastsim_data_dir() {
        ddpath
    } else {
        return Ok(vec![]);
    };
    let cache_url = if let Some(url) = &cache_url {
        url.clone()
    } else {
        get_default_cache_url()
    };
    let has_data = populate_cache_for_given_years_if_needed(ddpath.as_path(), &ys, &cache_url)?;
    if !has_data {
        return Err(anyhow!(
            "Unable to load or download cache data from {cache_url}"
        ));
    }
    let emissions_data = load_emissions_data_for_given_years(ddpath.as_path(), &ys)?;
    let fegov_data_by_year =
        load_fegov_data_for_given_years(ddpath.as_path(), &emissions_data, &ys)?;
    if let Some(fegov_db) = fegov_data_by_year.get(&y) {
        let mut hits: Vec<VehicleDataFE> = Vec::new();
        for item in fegov_db.iter() {
            if item.make == make && item.model == model {
                hits.push(item.clone());
            }
        }
        Ok(hits)
    } else {
        Ok(vec![])
    }
}

/// Match EPA Test Data with FuelEconomy.gov data and return best match
fn match_epatest_with_fegov(
    fegov: &VehicleDataFE,
    epatest_data: &[VehicleDataEPA],
) -> Option<VehicleDataEPA> {
    // Keep track of best match to fueleconomy.gov model name for all vehicles and vehicles with matching efid/test id
    let mut veh_list_overall: HashMap<String, Vec<VehicleDataEPA>> = HashMap::new();
    let mut veh_list_efid: HashMap<String, Vec<VehicleDataEPA>> = HashMap::new();
    let mut best_match_percent_efid: f64 = 0.0;
    let mut best_match_model_efid: String = String::new();
    let mut best_match_percent_overall: f64 = 0.0;
    let mut best_match_model_overall: String = String::new();

    let fe_model_upper: String = fegov.model.to_uppercase().replace("4WD", "AWD");
    let fe_model_words: Vec<&str> = fe_model_upper.split(' ').collect();
    let num_fe_model_words = fe_model_words.len();
    if fegov.emissions_list.emissions_info.is_empty() {
        return None;
    }
    let efid: &String = &fegov.emissions_list.emissions_info[0].efid;

    for veh_epa in epatest_data {
        // Find matches between EPA vehicle model name and fe.gov vehicle model name
        let mut match_count: i64 = 0;
        let epa_model_upper = veh_epa.model.to_uppercase().replace("4WD", "AWD");
        let epa_model_words: Vec<&str> = epa_model_upper.split(' ').collect();
        let num_epa_model_words = epa_model_words.len();
        for word in &epa_model_words {
            match_count += fe_model_words.contains(word) as i64;
        }
        // Calculate composite match percentage
        let match_percent: f64 = (match_count as f64 * match_count as f64)
            / (num_epa_model_words as f64 * num_fe_model_words as f64);

        // Update overall hashmap with new entry
        if veh_list_overall.contains_key(&veh_epa.model) {
            if let Some(x) = veh_list_overall.get_mut(&veh_epa.model) {
                (*x).push(veh_epa.clone());
            }
        } else {
            veh_list_overall.insert(veh_epa.model.clone(), vec![veh_epa.clone()]);

            if match_percent > best_match_percent_overall {
                best_match_percent_overall = match_percent;
                best_match_model_overall = veh_epa.model.clone();
            }
        }

        // Update efid hashmap if fe.gov efid matches EPA test id
        // (for some reason first character in id is almost always different)
        if veh_epa.test_id.ends_with(&efid[1..efid.len()]) {
            if veh_list_efid.contains_key(&veh_epa.model) {
                if let Some(x) = veh_list_efid.get_mut(&veh_epa.model) {
                    (*x).push(veh_epa.clone());
                }
            } else {
                veh_list_efid.insert(veh_epa.model.clone(), vec![veh_epa.clone()]);
                if match_percent > best_match_percent_efid {
                    best_match_percent_efid = match_percent;
                    best_match_model_efid = veh_epa.model.clone();
                }
            }
        }
    }

    // Get EPA vehicle model that is best match to fe.gov vehicle
    let veh_list: Vec<VehicleDataEPA> = if best_match_model_efid == best_match_model_overall {
        let x = veh_list_efid.get(&best_match_model_efid);
        x?;
        x.unwrap().to_vec()
    } else {
        veh_list_overall
            .get(&best_match_model_overall)
            .unwrap()
            .to_vec()
    };

    // Get number of gears and convert fe.gov transmission description to EPA transmission description
    let num_gears_fe_gov: u32;
    let transmission_fe_gov: String;
    // Based on reference: https://www.fueleconomy.gov/feg/findacarhelp.shtml#engine
    if fegov.trany.contains("Manual") {
        transmission_fe_gov = String::from('M');
        num_gears_fe_gov = fegov.trany.as_str()
            [fegov.trany.find("-spd").unwrap() - 1..fegov.trany.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else if fegov.trany.contains("variable gear ratios") {
        transmission_fe_gov = String::from("CVT");
        num_gears_fe_gov = 1;
    } else if fegov.trany.contains("AV-S") {
        transmission_fe_gov = String::from("SCV");
        num_gears_fe_gov = fegov.trany.as_str()
            [fegov.trany.find('S').unwrap() + 1..fegov.trany.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.trany.contains("AM-S") {
        transmission_fe_gov = String::from("AMS");
        num_gears_fe_gov = fegov.trany.as_str()
            [fegov.trany.find('S').unwrap() + 1..fegov.trany.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.trany.contains('S') {
        transmission_fe_gov = String::from("SA");
        num_gears_fe_gov = fegov.trany.as_str()
            [fegov.trany.find('S').unwrap() + 1..fegov.trany.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.trany.contains("-spd") {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = fegov.trany.as_str()
            [fegov.trany.find("-spd").unwrap() - 1..fegov.trany.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = {
            let res: Result<u32, ParseIntError> = fegov.trany.as_str()
                [fegov.trany.find("(A").unwrap() + 2..fegov.trany.find(')').unwrap()]
                .parse();
            if let Ok(n) = res {
                n
            } else {
                1
            }
        }
    }

    // Find EPA vehicle entry that matches fe.gov vehicle data
    // If same vehicle model has multiple configurations, get most common configuration
    let mut most_common_veh: VehicleDataEPA = VehicleDataEPA::default();
    let mut most_common_count: i32 = 0;
    let mut current_veh: VehicleDataEPA = VehicleDataEPA::default();
    let mut current_count: i32 = 0;
    for mut veh_epa in veh_list {
        if veh_epa.model.contains("4WD")
            || veh_epa.model.contains("AWD")
            || veh_epa.drive.contains("4-Wheel Drive")
        {
            veh_epa.drive_code = String::from('A');
            veh_epa.drive = String::from("All Wheel Drive");
        }
        if !veh_epa.test_fuel_type.contains("Cold CO")
            && (veh_epa.trany_code == transmission_fe_gov
                || fegov.trany.starts_with(veh_epa.trany_type.as_str()))
            && veh_epa.gears == num_gears_fe_gov
            && veh_epa.drive_code == fegov.drive[0..1]
            && ((fegov.alt_veh_type == *"EV"
                && veh_epa.displ.round() == 0.0
                && veh_epa.cylinders == String::new())
                || ((veh_epa.displ * 10.0).round() / 10.0
                    == (fegov.displ.parse::<f64>().unwrap_or_default())
                    && veh_epa.cylinders == fegov.cylinders))
        {
            if veh_epa == current_veh {
                current_count += 1;
            } else {
                if current_count > most_common_count {
                    most_common_veh = current_veh.clone();
                    most_common_count = current_count;
                }
                current_veh = veh_epa.clone();
                current_count = 1;
            }
        }
    }
    if current_count > most_common_count {
        Some(current_veh)
    } else {
        Some(most_common_veh)
    }
}

#[derive(Default, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[add_pyo3_api(
    #[new]
    pub fn __new__(
        vehicle_width_in: f64,
        vehicle_height_in: f64,
        fuel_tank_gal: f64,
        ess_max_kwh: f64,
        mc_max_kw: f64,
        ess_max_kw: f64,
        fc_max_kw: Option<f64>
    ) -> Self {
        OtherVehicleInputs {
            vehicle_width_in,
            vehicle_height_in,
            fuel_tank_gal,
            ess_max_kwh,
            mc_max_kw,
            ess_max_kw,
            fc_max_kw
        }
    }
)]
pub struct OtherVehicleInputs {
    pub vehicle_width_in: f64,
    pub vehicle_height_in: f64,
    pub fuel_tank_gal: f64,
    pub ess_max_kwh: f64,
    pub mc_max_kw: f64,
    pub ess_max_kw: f64,
    pub fc_max_kw: Option<f64>,
}

impl SerdeAPI for OtherVehicleInputs {
    fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        // check if the extension is csv, and if it is, then call Self::from_csv_file
        let pathbuf = PathBuf::from(filename);
        let file = File::open(filename)?;
        let extension = pathbuf.extension().unwrap().to_str().unwrap();
        match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        }
    }
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Creates RustVehicle for the given vehicle using data from fueleconomy.gov and EPA databases
/// The created RustVehicle is also written as a yaml file
///
/// Arguments:
/// ----------
/// vehicle_id: i32, Identifier at fueleconomy.gov for the desired vehicle
/// year: u32, the year of the vehicle
/// other_inputs: Other vehicle inputs required to create the vehicle
///
/// Returns:
/// --------
/// veh: RustVehicle for specificed vehicle
pub fn vehicle_import_by_id_and_year(
    vehicle_id: i32,
    year: u32,
    other_inputs: &OtherVehicleInputs,
    cache_url: Option<String>,
    data_dir: Option<String>,
) -> Result<RustVehicle, Error> {
    let mut maybe_veh: Option<RustVehicle> = None;
    let data_dir_path = if let Some(data_dir) = &data_dir {
        PathBuf::from(data_dir)
    } else if let Some(data_dir_path) = get_fastsim_data_dir() {
        data_dir_path
    } else {
        return Err(anyhow!("Could not determine the data directory"));
    };
    let data_dir_path = data_dir_path.as_path();
    let model_years = {
        let mut h: HashSet<u32> = HashSet::new();
        h.insert(year);
        h
    };
    let cache_url = if let Some(cache_url) = &cache_url {
        cache_url.clone()
    } else {
        get_default_cache_url()
    };
    let has_data =
        populate_cache_for_given_years_if_needed(data_dir_path, &model_years, &cache_url)?;
    if !has_data {
        return Err(anyhow!(
            "Unable to load or download cache data from {cache_url}"
        ));
    }
    let emissions_data = load_emissions_data_for_given_years(data_dir_path, &model_years)?;
    let fegov_data_by_year =
        load_fegov_data_for_given_years(data_dir_path, &emissions_data, &model_years)?;
    let epatest_db = read_epa_test_data_for_given_years(data_dir_path, &model_years)?;
    if let Some(fe_gov_data) = fegov_data_by_year.get(&year) {
        if let Some(epa_data) = epatest_db.get(&year) {
            let fe_gov_data = {
                let mut maybe_data = None;
                for item in fe_gov_data {
                    if item.id == vehicle_id {
                        maybe_data = Some(item.clone());
                        break;
                    }
                }
                maybe_data
            };
            if let Some(fe_gov_data) = fe_gov_data {
                if let Some(epa_data) = match_epatest_with_fegov(&fe_gov_data, epa_data) {
                    maybe_veh = try_make_single_vehicle(&fe_gov_data, &epa_data, other_inputs);
                }
            }
        }
    }
    match maybe_veh {
        Some(veh) => Ok(veh),
        None => Err(anyhow!("Unable to find/match vehicle in DB")),
    }
}

fn get_fuel_economy_gov_data_for_input_record(
    vir: &VehicleInputRecord,
    fegov_data: &[VehicleDataFE],
) -> Vec<VehicleDataFE> {
    let mut output: Vec<VehicleDataFE> = Vec::new();
    let vir_make = String::from(vir.make.to_lowercase().trim());
    let vir_model = String::from(vir.model.to_lowercase().trim());
    for fedat in fegov_data {
        let fe_make = String::from(fedat.make.to_lowercase().trim());
        let fe_model = String::from(fedat.model.to_lowercase().trim());
        if fedat.year == vir.year && fe_make.eq(&vir_make) && fe_model.eq(&vir_model) {
            output.push(fedat.clone());
        }
    }
    output
}

/// Try to make a single vehicle using the provided data sets.
fn try_make_single_vehicle(
    fe_gov_data: &VehicleDataFE,
    epa_data: &VehicleDataEPA,
    other_inputs: &OtherVehicleInputs,
) -> Option<RustVehicle> {
    if epa_data == &VehicleDataEPA::default() {
        return None;
    }
    let veh_pt_type: &str = match fe_gov_data.alt_veh_type.as_str() {
        "Hybrid" => crate::vehicle::HEV,
        "Plug-in Hybrid" => crate::vehicle::PHEV,
        "EV" => crate::vehicle::BEV,
        _ => crate::vehicle::CONV,
    };

    let fs_max_kw: f64;
    let fc_max_kw: f64;
    let fc_eff_type: String;
    let fc_eff_map: Array1<f64>;
    let mc_max_kw: f64;
    let min_soc: f64;
    let max_soc: f64;
    let ess_dischg_to_fc_max_eff_perc: f64;
    let mph_fc_on: f64;
    let kw_demand_fc_on: f64;
    let aux_kw: f64;
    let trans_eff: f64;
    let val_range_miles: f64;
    let ess_max_kw: f64;
    let ess_max_kwh: f64;
    let fs_kwh: f64;

    let ref_veh: RustVehicle = Default::default();

    if veh_pt_type == crate::vehicle::CONV {
        fs_max_kw = 2000.0;
        fs_kwh = other_inputs.fuel_tank_gal * ref_veh.props.kwh_per_gge;
        fc_max_kw = epa_data.eng_pwr_hp as f64 / HP_PER_KW;
        fc_eff_type = String::from(crate::vehicle::SI);
        fc_eff_map = Array::from_vec(vec![
            0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
        ]);
        mc_max_kw = 0.0;
        min_soc = 0.1;
        max_soc = 0.95;
        ess_dischg_to_fc_max_eff_perc = 0.0;
        mph_fc_on = 55.0;
        kw_demand_fc_on = 100.0;
        aux_kw = 0.7;
        trans_eff = 0.92;
        val_range_miles = 0.0;
        ess_max_kw = 0.0;
        ess_max_kwh = 0.0;
    } else if veh_pt_type == crate::vehicle::HEV {
        fs_max_kw = 2000.0;
        fs_kwh = other_inputs.fuel_tank_gal * ref_veh.props.kwh_per_gge;
        fc_max_kw = other_inputs
            .fc_max_kw
            .unwrap_or(epa_data.eng_pwr_hp as f64 / HP_PER_KW);
        fc_eff_type = String::from(crate::vehicle::ATKINSON);
        fc_eff_map = Array::from_vec(vec![
            0.1, 0.12, 0.28, 0.35, 0.38, 0.39, 0.4, 0.4, 0.38, 0.37, 0.36, 0.35,
        ]);
        min_soc = 0.4;
        max_soc = 0.8;
        ess_dischg_to_fc_max_eff_perc = 0.0;
        mph_fc_on = 1.0;
        kw_demand_fc_on = 100.0;
        aux_kw = 0.5;
        trans_eff = 0.95;
        val_range_miles = 0.0;
        ess_max_kw = other_inputs.ess_max_kw;
        ess_max_kwh = other_inputs.ess_max_kwh;
        mc_max_kw = other_inputs.mc_max_kw;
    } else if veh_pt_type == crate::vehicle::PHEV {
        fs_max_kw = 2000.0;
        fs_kwh = other_inputs.fuel_tank_gal * ref_veh.props.kwh_per_gge;
        fc_max_kw = other_inputs
            .fc_max_kw
            .unwrap_or(epa_data.eng_pwr_hp as f64 / HP_PER_KW);
        fc_eff_type = String::from(crate::vehicle::ATKINSON);
        fc_eff_map = Array::from_vec(vec![
            0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
        ]);
        min_soc = 0.15;
        max_soc = 0.9;
        ess_dischg_to_fc_max_eff_perc = 1.0;
        mph_fc_on = 85.0;
        kw_demand_fc_on = 120.0;
        aux_kw = 0.3;
        trans_eff = 0.98;
        val_range_miles = 0.0;
        ess_max_kw = other_inputs.ess_max_kw;
        ess_max_kwh = other_inputs.ess_max_kwh;
        mc_max_kw = other_inputs.mc_max_kw;
    } else if veh_pt_type == crate::vehicle::BEV {
        fs_max_kw = 0.0;
        fs_kwh = 0.0;
        fc_max_kw = 0.0;
        fc_eff_type = String::from(crate::vehicle::SI);
        fc_eff_map = Array::from_vec(vec![
            0.1, 0.12, 0.28, 0.35, 0.38, 0.39, 0.4, 0.4, 0.38, 0.37, 0.36, 0.35,
        ]);
        mc_max_kw = epa_data.eng_pwr_hp as f64 / HP_PER_KW;
        min_soc = 0.05;
        max_soc = 0.98;
        ess_max_kw = 1.05 * mc_max_kw;
        ess_max_kwh = other_inputs.ess_max_kwh;
        mph_fc_on = 1.0;
        kw_demand_fc_on = 100.0;
        aux_kw = 0.25;
        trans_eff = 0.98;
        val_range_miles = fe_gov_data.range_ev as f64;
        ess_dischg_to_fc_max_eff_perc = 0.0;
    } else {
        println!("Unhandled vehicle powertrain type: {veh_pt_type}");
        return None;
    }

    let glider_kg = (epa_data.test_weight_lbs / LBS_PER_KG)
        - ref_veh.cargo_kg
        - ref_veh.trans_kg
        - ref_veh.comp_mass_multiplier
            * ((fs_max_kw / ref_veh.fs_kwh_per_kg)
                + (ref_veh.fc_base_kg + fc_max_kw / ref_veh.fc_kw_per_kg)
                + (ref_veh.mc_pe_base_kg + mc_max_kw * ref_veh.mc_pe_kg_per_kw)
                + (ref_veh.ess_base_kg + ess_max_kwh * ref_veh.ess_kg_per_kwh));
    let mut veh = RustVehicle {
        veh_cg_m: match fe_gov_data.drive.as_str() {
            "Front-Wheel Drive" => 0.53,
            _ => -0.53,
        },
        glider_kg,
        scenario_name: format!(
            "{} {} {}",
            fe_gov_data.year, fe_gov_data.make, fe_gov_data.model
        ),
        max_roadway_chg_kw: Default::default(),
        selection: 0,
        veh_year: fe_gov_data.year,
        veh_pt_type: String::from(veh_pt_type),
        drag_coef: 0.0,
        frontal_area_m2: (other_inputs.vehicle_width_in * other_inputs.vehicle_height_in)
            / (IN_PER_M * IN_PER_M),
        fs_kwh,
        idle_fc_kw: fc_max_kw / 100.0, // TODO: Figure out if idle_fc_kw is needed
        mc_eff_map: Array1::from(vec![
            0.41, 0.45, 0.48, 0.54, 0.58, 0.62, 0.83, 0.93, 0.94, 0.93, 0.92,
        ]),
        wheel_rr_coef: 0.0,
        stop_start: fe_gov_data.start_stop == "Y",
        force_aux_on_fc: false,
        val_udds_mpgge: fe_gov_data.city_mpg_fuel1 as f64,
        val_hwy_mpgge: fe_gov_data.highway_mpg_fuel1 as f64,
        val_comb_mpgge: fe_gov_data.comb_mpg_fuel1 as f64,
        fc_peak_eff_override: None,
        mc_peak_eff_override: Some(0.95),
        fs_max_kw,
        fc_max_kw,
        fc_eff_type,
        fc_eff_map,
        mc_max_kw,
        min_soc,
        max_soc,
        ess_dischg_to_fc_max_eff_perc,
        mph_fc_on,
        kw_demand_fc_on,
        aux_kw,
        trans_eff,
        val_range_miles,
        ess_max_kwh,
        ess_max_kw,
        ..Default::default()
    };
    veh.set_derived().unwrap();

    abc_to_drag_coeffs(
        &mut veh,
        epa_data.a_lbf,
        epa_data.b_lbf_per_mph,
        epa_data.c_lbf_per_mph2,
        Some(false),
        None,
        None,
        Some(true),
        Some(false),
    );
    Some(veh)
}

fn try_import_vehicles(
    vir: &VehicleInputRecord,
    fegov_data: &[VehicleDataFE],
    epatest_data: &[VehicleDataEPA],
) -> Vec<RustVehicle> {
    let other_inputs = vir_to_other_inputs(vir);
    // TODO: Aaron wanted custom scenario name option
    let mut outputs: Vec<RustVehicle> = Vec::new();
    let fegov_hits: Vec<VehicleDataFE> =
        get_fuel_economy_gov_data_for_input_record(vir, fegov_data);
    for hit in fegov_hits {
        if let Some(epa_data) = match_epatest_with_fegov(&hit, epatest_data) {
            if let Some(v) = try_make_single_vehicle(&hit, &epa_data, &other_inputs) {
                let mut v = v.clone();
                if hit.alt_veh_type == *"EV" {
                    v.scenario_name = format!("{} (EV)", v.scenario_name);
                } else {
                    let alt_type = if hit.alt_veh_type.is_empty() {
                        String::from("")
                    } else {
                        format!("{}, ", hit.alt_veh_type)
                    };
                    v.scenario_name = format!(
                        "{} ( {} {} cylinders, {} L, {} )",
                        v.scenario_name, alt_type, hit.cylinders, hit.displ, hit.trany
                    );
                }
                outputs.push(v);
            } else {
                println!(
                    "Unable to create vehicle for {}-{}-{}",
                    vir.year, vir.make, vir.model
                );
            }
        } else {
            println!(
                "Did not match any EPA data for {}-{}-{}...",
                vir.year, vir.make, vir.model
            );
        }
    }
    outputs
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Export the given RustVehicle to file
///
/// veh: The RustVehicle to export
/// file_path: the path to export to
///
/// NOTE: the file extension is used to determine the export format.
/// Supported file types include yaml and JSON
///
/// RETURN:
/// ()
pub fn export_vehicle_to_file(veh: &RustVehicle, file_path: String) -> Result<(), anyhow::Error> {
    let processed_path = PathBuf::from(file_path);
    let path_str = processed_path.to_str().unwrap_or("");
    veh.to_file(path_str)?;
    Ok(())
}

#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "pyo3", pyfunction)]
#[allow(clippy::too_many_arguments)]
pub fn abc_to_drag_coeffs(
    veh: &mut RustVehicle,
    a_lbf: f64,
    b_lbf__mph: f64,
    c_lbf__mph2: f64,
    custom_rho: Option<bool>,
    custom_rho_temp_degC: Option<f64>,
    custom_rho_elevation_m: Option<f64>,
    simdrive_optimize: Option<bool>,
    _show_plots: Option<bool>,
) -> (f64, f64) {
    // For a given vehicle and target A, B, and C coefficients;
    // calculate and return drag and rolling resistance coefficients.
    //
    // Arguments:
    // ----------
    // veh: vehicle.RustVehicle with all parameters correct except for drag and rolling resistance coefficients
    // a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    // custom_rho: if True, use `air::get_rho()` to calculate the current ambient density
    // custom_rho_temp_degC: ambient temperature [degree C] for `get_rho()`;
    //     will only be used when `custom_rho` is True
    // custom_rho_elevation_m: location elevation [degree C] for `get_rho()`;
    //     will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
    // simdrive_optimize: if True, use `SimDrive` to optimize the drag and rolling resistance;
    //     otherwise, directly use target A, B, C to calculate the results
    // show_plots: if True, plots are shown

    let air_props: AirProperties = AirProperties::default();
    let props: RustPhysicalProperties = RustPhysicalProperties::default();
    let cur_ambient_air_density_kg__m3: f64 = if custom_rho.unwrap_or(false) {
        air_props.get_rho(custom_rho_temp_degC.unwrap_or(20.0), custom_rho_elevation_m)
    } else {
        props.air_density_kg_per_m3
    };

    let vmax_mph: f64 = 70.0;
    let a_newton: f64 = a_lbf * super::params::N_PER_LBF;
    let _b_newton__mps: f64 = b_lbf__mph * super::params::N_PER_LBF * super::params::MPH_PER_MPS;
    let c_newton__mps2: f64 = c_lbf__mph2
        * super::params::N_PER_LBF
        * super::params::MPH_PER_MPS
        * super::params::MPH_PER_MPS;

    let cd_len: usize = 300;

    let cyc: RustCycle = RustCycle::new(
        (0..cd_len as i32).map(f64::from).collect(),
        Array::linspace(vmax_mph / super::params::MPH_PER_MPS, 0.0, cd_len).to_vec(),
        vec![0.0; cd_len],
        vec![0.0; cd_len],
        String::from("cycle"),
    );

    // polynomial function for pounds vs speed
    let dyno_func_lb: Polynomial<f64> = Polynomial::new(vec![a_lbf, b_lbf__mph, c_lbf__mph2]);

    let drag_coef: f64;
    let wheel_rr_coef: f64;

    if simdrive_optimize.unwrap_or(true) {
        let cost: GetError = GetError {
            cycle: &cyc,
            vehicle: veh,
            dyno_func_lb: &dyno_func_lb,
        };
        let solver: NelderMead<Array1<f64>, f64> =
            NelderMead::new(vec![array![0.0, 0.0], array![0.5, 0.0], array![0.5, 0.1]]);
        let res: OptimizationResult<_, _, _> = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();
        let best_param: &Array1<f64> = res.state().get_best_param().unwrap();
        drag_coef = best_param[0];
        wheel_rr_coef = best_param[1];
    } else {
        drag_coef = c_newton__mps2 / (0.5 * veh.frontal_area_m2 * cur_ambient_air_density_kg__m3);
        wheel_rr_coef = a_newton / veh.veh_kg / props.a_grav_mps2;
    }

    veh.drag_coef = drag_coef;
    veh.wheel_rr_coef = wheel_rr_coef;

    (drag_coef, wheel_rr_coef)
}

pub fn get_error_val(model: Array1<f64>, test: Array1<f64>, time_steps: Array1<f64>) -> f64 {
    // Returns time-averaged error for model and test signal.
    // Arguments:
    // ----------
    // model: array of values for signal from model
    // test: array of values for signal from test data
    // time_steps: array (or scalar for constant) of values for model time steps [s]
    // test: array of values for signal from test

    // Output:
    // -------
    // err: integral of absolute value of difference between model and
    // test per time

    assert!(
        model.len() == test.len() && test.len() == time_steps.len(),
        "{}, {}, {}",
        model.len(),
        test.len(),
        time_steps.len()
    );

    let mut err: f64 = 0.0;
    let y: Array1<f64> = (model - test).mapv(f64::abs);

    for index in 0..time_steps.len() - 1 {
        err += 0.5 * (time_steps[index + 1] - time_steps[index]) * (y[index] + y[index + 1]);
    }

    return err / (time_steps.last().unwrap() - time_steps[0]);
}

struct GetError<'a> {
    cycle: &'a RustCycle,
    vehicle: &'a RustVehicle,
    dyno_func_lb: &'a Polynomial<f64>,
}

impl CostFunction for GetError<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let mut veh: RustVehicle = self.vehicle.clone();
        let cyc: RustCycle = self.cycle.clone();
        let dyno_func_lb: Polynomial<f64> = self.dyno_func_lb.clone();

        veh.drag_coef = x[0];
        veh.wheel_rr_coef = x[1];

        let mut sd_coast: RustSimDrive = RustSimDrive::new(self.cycle.clone(), veh);
        sd_coast.impose_coast = Array::from_vec(vec![true; sd_coast.impose_coast.len()]);
        let _sim_drive_result: Result<_, _> = sd_coast.sim_drive(None, None);

        let cutoff_vec: Vec<usize> = sd_coast
            .mps_ach
            .indexed_iter()
            .filter_map(|(index, &item)| (item < 0.1).then_some(index))
            .collect();
        let cutoff: usize = if cutoff_vec.is_empty() {
            sd_coast.mps_ach.len()
        } else {
            cutoff_vec[0]
        };

        Ok(get_error_val(
            (Array::from_vec(vec![1000.0; sd_coast.mps_ach.len()])
                * (sd_coast.drag_kw + sd_coast.rr_kw)
                / sd_coast.mps_ach)
                .slice_move(s![0..cutoff]),
            (sd_coast.mph_ach.map(|x| dyno_func_lb.eval(*x))
                * Array::from_vec(vec![super::params::N_PER_LBF; sd_coast.mph_ach.len()]))
            .slice_move(s![0..cutoff]),
            cyc.time_s.slice_move(s![0..cutoff]),
        ))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VehicleInputRecord {
    pub make: String,
    pub model: String,
    pub year: u32,
    pub output_file_name: String,
    pub vehicle_width_in: f64,
    pub vehicle_height_in: f64,
    pub fuel_tank_gal: f64,
    pub ess_max_kwh: f64,
    pub mc_max_kw: f64,
    pub ess_max_kw: f64,
    pub fc_max_kw: Option<f64>,
}

/// Transltate a VehicleInputRecord to OtherVehicleInputs
fn vir_to_other_inputs(vir: &VehicleInputRecord) -> OtherVehicleInputs {
    OtherVehicleInputs {
        vehicle_width_in: vir.vehicle_width_in,
        vehicle_height_in: vir.vehicle_height_in,
        fuel_tank_gal: vir.fuel_tank_gal,
        ess_max_kwh: vir.ess_max_kwh,
        mc_max_kw: vir.mc_max_kw,
        ess_max_kw: vir.ess_max_kw,
        fc_max_kw: vir.fc_max_kw,
    }
}

fn read_vehicle_input_records_from_file(
    filepath: &Path,
) -> Result<Vec<VehicleInputRecord>, anyhow::Error> {
    let f = File::open(filepath)?;
    read_records_from_file(f)
}

fn read_records_from_file<T: DeserializeOwned>(
    rdr: impl std::io::Read + std::io::Seek,
) -> Result<Vec<T>, anyhow::Error> {
    let mut output: Vec<T> = Vec::new();
    let mut reader = csv::Reader::from_reader(rdr);
    for result in reader.deserialize() {
        let record: T = result?;
        output.push(record);
    }
    Ok(output)
}

fn read_fuelecon_gov_emissions_to_hashmap(
    rdr: impl std::io::Read + std::io::Seek,
) -> HashMap<u32, Vec<EmissionsInfoFE>> {
    let mut output: HashMap<u32, Vec<EmissionsInfoFE>> = HashMap::new();
    let mut reader = csv::Reader::from_reader(rdr);
    for result in reader.deserialize() {
        if result.is_ok() {
            let ok_result: Option<HashMap<String, String>> = result.ok();
            if let Some(item) = ok_result {
                if let Some(id_str) = item.get("id") {
                    if let Ok(id) = str::parse::<u32>(id_str) {
                        output.entry(id).or_insert_with(Vec::new);
                        if let Some(ers) = output.get_mut(&id) {
                            let emiss = EmissionsInfoFE {
                                efid: item.get("efid").unwrap().clone(),
                                score: item.get("score").unwrap().parse().unwrap(),
                                smartway_score: item.get("smartwayScore").unwrap().parse().unwrap(),
                                standard: item.get("standard").unwrap().clone(),
                                std_text: item.get("stdText").unwrap().clone(),
                            };
                            ers.push(emiss);
                        }
                    }
                }
            }
        }
    }
    output
}

fn read_fuelecon_gov_data_from_file(
    rdr: impl std::io::Read + std::io::Seek,
    emissions: &HashMap<u32, Vec<EmissionsInfoFE>>,
) -> Result<Vec<VehicleDataFE>, anyhow::Error> {
    let mut output: Vec<VehicleDataFE> = Vec::new();
    let mut reader = csv::Reader::from_reader(rdr);
    for result in reader.deserialize() {
        let item: HashMap<String, String> = result?;
        let id: u32 = item.get("id").unwrap().parse::<u32>().unwrap();
        let emissions_list: EmissionsListFE = if emissions.contains_key(&id) {
            EmissionsListFE {
                emissions_info: emissions.get(&id).unwrap().to_vec(),
            }
        } else {
            EmissionsListFE::default()
        };
        let vd = VehicleDataFE {
            id: item.get("id").unwrap().trim().parse().unwrap(),
            // #[serde(default, rename = "atvType")]
            // /// Type of alternative fuel vehicle (Hybrid, Plug-in Hybrid, EV)
            // pub alt_veh_type: String,
            alt_veh_type: item.get("atvType").unwrap().clone(),
            // #[serde(rename = "city08")]
            // /// City MPG for fuel 1
            // pub city_mpg_fuel1: i32,
            city_mpg_fuel1: item.get("city08").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "cityA08")]
            // /// City MPG for fuel 2
            // pub city_mpg_fuel2: i32,
            city_mpg_fuel2: item.get("cityA08").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "co2")]
            // /// Tailpipe CO2 emissions in grams/mile
            // pub co2_g_per_mi: i32,
            co2_g_per_mi: item.get("co2").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "comb08")]
            // /// Combined MPG for fuel 1
            // pub comb_mpg_fuel1: i32,
            comb_mpg_fuel1: item.get("comb08").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "combA08")]
            // /// Combined MPG for fuel 2
            // pub comb_mpg_fuel2: i32,
            comb_mpg_fuel2: item.get("combA08").unwrap().parse::<i32>().unwrap(),
            // #[serde(default)]
            // /// Number of engine cylinders
            // pub cylinders: String,
            cylinders: item.get("cylinders").unwrap().clone(),
            // #[serde(default)]
            // /// Engine displacement in liters
            // pub displ: String,
            displ: item.get("displ").unwrap().clone(),
            // /// Drive axle type (FWD, RWD, AWD, 4WD)
            // pub drive: String,
            drive: item.get("drive").unwrap().clone(),
            // #[serde(rename = "emissionsList")]
            // /// List of emissions tests
            // pub emissions_list: EmissionsListFE,
            emissions_list,
            // #[serde(default)]
            // /// Description of engine
            // pub eng_dscr: String,
            eng_dscr: item.get("eng_dscr").unwrap().clone(),
            // #[serde(default, rename = "evMotor")]
            // /// Electric motor power (kW)
            // pub ev_motor_kw: String,
            ev_motor_kw: item.get("evMotor").unwrap().clone(),
            // #[serde(rename = "feScore")]
            // /// EPA fuel economy score
            // pub fe_score: i32,
            fe_score: item.get("feScore").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "fuelType")]
            // /// Combined vehicle fuel type (fuel 1 and fuel 2)
            // pub fuel_type: String,
            fuel_type: item.get("fuelType").unwrap().clone(),
            // #[serde(rename = "fuelType1")]
            // /// Fuel type 1
            // pub fuel1: String,
            fuel1: item.get("fuelType1").unwrap().clone(),
            // #[serde(default, rename = "fuelType2")]
            // /// Fuel type 2
            // pub fuel2: String,
            fuel2: item.get("fuelType2").unwrap().clone(),
            // #[serde(rename = "ghgScore")]
            // /// EPA GHG Score
            // pub ghg_score: i32,
            ghg_score: item.get("ghgScore").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "highway08")]
            // /// Highway MPG for fuel 1
            // pub highway_mpg_fuel1: i32,
            highway_mpg_fuel1: item.get("highway08").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "highwayA08")]
            // /// Highway MPG for fuel 2
            // pub highway_mpg_fuel2: i32,
            highway_mpg_fuel2: item.get("highwayA08").unwrap().parse::<i32>().unwrap(),
            // /// Manufacturer
            // pub make: String,
            make: item.get("make").unwrap().clone(),
            // #[serde(rename = "mfrCode")]
            // /// Manufacturer code
            // pub mfr_code: String,
            mfr_code: item.get("mfrCode").unwrap().clone(),
            // /// Model name
            // pub model: String,
            model: item.get("model").unwrap().clone(),
            // #[serde(rename = "phevBlended")]
            // /// Vehicle operates on blend of gasoline and electricity
            // pub phev_blended: bool,
            phev_blended: item
                .get("phevBlended")
                .unwrap()
                .trim()
                .to_lowercase()
                .parse::<bool>()
                .unwrap(),
            // #[serde(rename = "phevCity")]
            // /// EPA composite gasoline-electricity city MPGe
            // pub phev_city_mpge: i32,
            phev_city_mpge: item.get("phevCity").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "phevComb")]
            // /// EPA composite gasoline-electricity combined MPGe
            // pub phev_comb_mpge: i32,
            phev_comb_mpge: item.get("phevComb").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "phevHwy")]
            // /// EPA composite gasoline-electricity highway MPGe
            // pub phev_hwy_mpge: i32,
            phev_hwy_mpge: item.get("phevHwy").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "range")]
            // /// Range for EV
            // pub range_ev: i32,
            range_ev: item.get("range").unwrap().parse::<i32>().unwrap(),
            // #[serde(rename = "startStop")]
            // /// Stop-start technology
            // pub start_stop: String,
            start_stop: item.get("startStop").unwrap().clone(),
            // /// transmission
            // pub trany: String,
            trany: item.get("trany").unwrap().clone(),
            // #[serde(rename = "VClass")]
            // /// EPA vehicle size class
            // pub veh_class: String,
            veh_class: item.get("VClass").unwrap().clone(),
            // /// Model year
            // pub year: u32,
            year: item.get("year").unwrap().parse::<u32>().unwrap(),
            // #[serde(default, rename = "sCharger")]
            // /// Vehicle is supercharged
            // pub super_charge: String,
            super_charge: item.get("sCharger").unwrap().clone(),
            // #[serde(default, rename = "tCharger")]
            // /// Vehicle is turbocharged
            // pub turbo_charge: String,
            turbo_charge: item.get("tCharger").unwrap().clone(),
        };
        output.push(vd);
    }
    Ok(output)
}

/// Given the path to a zip archive, print out the names of the files within that archive
pub fn list_zip_contents(filepath: &Path) -> Result<(), anyhow::Error> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    for i in 0..zip.len() {
        let file = zip.by_index(i)?;
        println!("Filename: {}", file.name());
    }
    Ok(())
}

/// Creates/gets an OS-specific data directory and returns the path.
pub fn get_fastsim_data_dir() -> Option<PathBuf> {
    if let Some(proj_dirs) = ProjectDirs::from("gov", "NREL", "fastsim") {
        let mut path = PathBuf::from(proj_dirs.config_dir());
        path.push(Path::new("data"));
        if !path.exists() {
            let result = std::fs::create_dir_all(path.as_path());
            if result.is_err() {
                None
            } else {
                Some(path)
            }
        } else {
            Some(path)
        }
    } else {
        None
    }
}

/// Extract zip archive at filepath to destination directory at dest_dir
pub fn extract_zip(filepath: &Path, dest_dir: &Path) -> Result<(), anyhow::Error> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    zip.extract(&dest_dir)?;
    Ok(())
}

/// Assumes the parent directory exists. Assumes file doesn't exist (i.e., newly created) or that it will be truncated if it does.
pub fn download_file_from_url(url: &str, file_path: &Path) -> Result<(), anyhow::Error> {
    let mut handle = Easy::new();
    handle.follow_location(true)?;
    handle.url(url)?;
    let mut buffer = Vec::new();
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|data| {
            buffer.extend_from_slice(data);
            Ok(data.len())
        })?;
        let result = transfer.perform();
        if result.is_err() {
            return Err(anyhow!("Could not download from {}", url));
        }
    }
    println!("Downloaded data from {} of length {}", url, buffer.len());
    if buffer.is_empty() {
        return Err(anyhow!("No data available from {url}"));
    }
    {
        let mut file = match File::create(&file_path) {
            Err(why) => {
                return Err(anyhow!(
                    "couldn't open {}: {}",
                    file_path.to_str().unwrap(),
                    why
                ))
            }
            Ok(file) => file,
        };
        file.write_all(buffer.as_slice())?;
    }
    Ok(())
}

fn read_epa_test_data_for_given_years(
    data_dir_path: &Path,
    years: &HashSet<u32>,
) -> Result<HashMap<u32, Vec<VehicleDataEPA>>, anyhow::Error> {
    let mut epatest_db: HashMap<u32, Vec<VehicleDataEPA>> = HashMap::new();
    for year in years {
        let file_name = format!("{year}-testcar.csv");
        let p = data_dir_path.join(Path::new(&file_name));
        let f = File::open(p)?;
        let records = read_records_from_file(f)?;
        epatest_db.insert(*year, records);
    }
    Ok(epatest_db)
}

fn determine_model_years_of_interest(virs: &[VehicleInputRecord]) -> HashSet<u32> {
    HashSet::from_iter(virs.iter().map(|vir| vir.year))
}

fn load_emissions_data_for_given_years(
    data_dir_path: &Path,
    years: &HashSet<u32>,
) -> Result<HashMap<u32, HashMap<u32, Vec<EmissionsInfoFE>>>, anyhow::Error> {
    let mut data = HashMap::<u32, HashMap<u32, Vec<EmissionsInfoFE>>>::new();
    for year in years {
        let file_name = format!("{year}-emissions.csv");
        let emissions_path = data_dir_path.join(Path::new(&file_name));
        if !emissions_path.exists() {
            // download from URL and cache
            println!(
                "DATA DOES NOT EXIST AT {}",
                emissions_path.to_string_lossy()
            );
        }
        let emissions_db: HashMap<u32, Vec<EmissionsInfoFE>> = {
            let emissions_file = File::open(emissions_path)?;
            read_fuelecon_gov_emissions_to_hashmap(emissions_file)
        };
        data.insert(*year, emissions_db);
    }
    Ok(data)
}

fn load_fegov_data_for_given_years(
    data_dir_path: &Path,
    emissions_by_year_and_by_id: &HashMap<u32, HashMap<u32, Vec<EmissionsInfoFE>>>,
    years: &HashSet<u32>,
) -> Result<HashMap<u32, Vec<VehicleDataFE>>, anyhow::Error> {
    let mut data = HashMap::<u32, Vec<VehicleDataFE>>::new();
    for year in years {
        if let Some(emissions_by_id) = emissions_by_year_and_by_id.get(year) {
            let file_name = format!("{year}-vehicles.csv");
            let fegov_path = data_dir_path.join(Path::new(&file_name));
            let fegov_db: Vec<VehicleDataFE> = {
                let fegov_file = File::open(fegov_path.as_path())?;
                read_fuelecon_gov_data_from_file(fegov_file, emissions_by_id)?
            };
            data.insert(*year, fegov_db);
        } else {
            println!("No fe.gov emissions data available for {year}");
        }
    }
    Ok(data)
}

pub fn get_default_cache_url() -> String {
    String::from("https://github.com/NREL/fastsim-data/raw/main/")
}

fn get_cache_url_for_year(cache_url: &str, year: &u32) -> Result<Option<String>, anyhow::Error> {
    let maybe_slash = if cache_url.ends_with('/') { "" } else { "/" };
    let target_url = format!("{cache_url}{maybe_slash}{year}.zip");
    Ok(Some(target_url))
}

fn extract_file_from_zip(
    zip_file_path: &Path,
    name_of_file_to_extract: &str,
    path_to_save_to: &Path,
) -> Result<(), anyhow::Error> {
    let zipfile = File::open(zip_file_path)?;
    let mut archive = ZipArchive::new(zipfile)?;
    let mut file = archive.by_name(name_of_file_to_extract)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    std::fs::write(path_to_save_to, contents)?;
    Ok(())
}

/// Checks the cache directory to see if data files have been downloaded
/// If so, moves on without any further action.
/// If not, downloads data by year from remote site if it exists
fn populate_cache_for_given_years_if_needed(
    data_dir_path: &Path,
    years: &HashSet<u32>,
    cache_url: &str,
) -> Result<bool, anyhow::Error> {
    let mut all_data_available = true;
    for year in years {
        println!("Checking {year}...");
        let veh_file_exists = {
            let name = format!("{year}-vehicles.csv");
            let path = data_dir_path.join(Path::new(&name));
            path.exists()
        };
        let emissions_file_exists = {
            let name = format!("{year}-emissions.csv");
            let path = data_dir_path.join(Path::new(&name));
            path.exists()
        };
        let epa_file_exists = {
            let name = format!("{year}-testcar.csv");
            let path = data_dir_path.join(Path::new(&name));
            path.exists()
        };
        if !veh_file_exists || !emissions_file_exists || !epa_file_exists {
            all_data_available = false;
            let zip_file_name = format!("{year}.zip");
            let zip_file_path = data_dir_path.join(Path::new(&zip_file_name));
            if let Some(url) = get_cache_url_for_year(cache_url, year)? {
                println!("Downloading data for {year}: {url}");
                download_file_from_url(&url, &zip_file_path)?;
                println!("... downloading data for {year}");
                let emissions_name = format!("{year}-emissions.csv");
                extract_file_from_zip(
                    zip_file_path.as_path(),
                    &emissions_name,
                    data_dir_path.join(Path::new(&emissions_name)).as_path(),
                )?;
                println!("... extracted {}", emissions_name);
                let vehicles_name = format!("{year}-vehicles.csv");
                extract_file_from_zip(
                    zip_file_path.as_path(),
                    &vehicles_name,
                    data_dir_path.join(Path::new(&vehicles_name)).as_path(),
                )?;
                println!("... extracted {}", vehicles_name);
                let epatests_name = format!("{year}-testcar.csv");
                extract_file_from_zip(
                    zip_file_path.as_path(),
                    &epatests_name,
                    data_dir_path.join(Path::new(&epatests_name)).as_path(),
                )?;
                println!("... extracted {}", epatests_name);
                all_data_available = true;
            }
        }
    }
    Ok(all_data_available)
}

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Import All Vehicles for the given Year, Make, and Model and supplied other inputs
pub fn import_all_vehicles(
    year: u32,
    make: &str,
    model: &str,
    other_inputs: &OtherVehicleInputs,
    cache_url: Option<String>,
    data_dir: Option<String>,
) -> Result<Vec<RustVehicle>, anyhow::Error> {
    let vir = VehicleInputRecord {
        year,
        make: make.to_string(),
        model: model.to_string(),
        output_file_name: String::from(""),
        vehicle_width_in: other_inputs.vehicle_width_in,
        vehicle_height_in: other_inputs.vehicle_height_in,
        fuel_tank_gal: other_inputs.fuel_tank_gal,
        ess_max_kwh: other_inputs.ess_max_kwh,
        mc_max_kw: other_inputs.mc_max_kw,
        ess_max_kw: other_inputs.ess_max_kw,
        fc_max_kw: other_inputs.fc_max_kw,
    };
    let inputs = vec![vir];
    let model_years = {
        let mut h: HashSet<u32> = HashSet::new();
        h.insert(year);
        h
    };
    let data_dir_path = if let Some(dd_path) = &data_dir {
        PathBuf::from(dd_path.clone())
    } else if let Some(dd_path) = get_fastsim_data_dir() {
        dd_path
    } else {
        return Err(anyhow!("Unable to determine data directory"));
    };
    let data_dir_path = data_dir_path.as_path();
    let cache_url = if let Some(cache_url) = &cache_url {
        cache_url.clone()
    } else {
        get_default_cache_url()
    };
    let has_data =
        populate_cache_for_given_years_if_needed(data_dir_path, &model_years, &cache_url)?;
    if !has_data {
        return Err(anyhow!(
            "Unable to load or download cache data from {cache_url}"
        ));
    }
    let emissions_data = load_emissions_data_for_given_years(data_dir_path, &model_years)?;
    let fegov_data_by_year =
        load_fegov_data_for_given_years(data_dir_path, &emissions_data, &model_years)?;
    let epatest_db = read_epa_test_data_for_given_years(data_dir_path, &model_years)?;
    let vehs = import_all_vehicles_from_record(&inputs, &fegov_data_by_year, &epatest_db)
        .into_iter()
        .map(|x| -> RustVehicle { x.1 })
        .collect();
    Ok(vehs)
}

/// Import and Save All Vehicles Specified via Input File
pub fn import_and_save_all_vehicles_from_file(
    input_path: &Path,
    data_dir_path: &Path,
    output_dir_path: &Path,
    cache_url: Option<String>,
) -> Result<(), anyhow::Error> {
    let cache_url = if let Some(url) = &cache_url {
        url.clone()
    } else {
        get_default_cache_url()
    };
    let inputs: Vec<VehicleInputRecord> = read_vehicle_input_records_from_file(input_path)?;
    println!("Found {} vehicle input records", inputs.len());
    let model_years = determine_model_years_of_interest(&inputs);
    let has_data =
        populate_cache_for_given_years_if_needed(data_dir_path, &model_years, &cache_url)?;
    if !has_data {
        return Err(anyhow!(
            "Unable to load or download cache data from {cache_url}"
        ));
    }
    let emissions_data = load_emissions_data_for_given_years(data_dir_path, &model_years)?;
    let fegov_data_by_year =
        load_fegov_data_for_given_years(data_dir_path, &emissions_data, &model_years)?;
    let epatest_db = read_epa_test_data_for_given_years(data_dir_path, &model_years)?;
    println!("Read {} files of epa test vehicle data", epatest_db.len());
    import_and_save_all_vehicles(&inputs, &fegov_data_by_year, &epatest_db, output_dir_path)
}

pub fn import_all_vehicles_from_record(
    inputs: &[VehicleInputRecord],
    fegov_data_by_year: &HashMap<u32, Vec<VehicleDataFE>>,
    epatest_data_by_year: &HashMap<u32, Vec<VehicleDataEPA>>,
) -> Vec<(VehicleInputRecord, RustVehicle)> {
    let mut vehs: Vec<(VehicleInputRecord, RustVehicle)> = Vec::new();
    for vir in inputs {
        if let Some(fegov_data) = fegov_data_by_year.get(&vir.year) {
            if let Some(epatest_data) = epatest_data_by_year.get(&vir.year) {
                let vs = try_import_vehicles(vir, fegov_data, epatest_data);
                for v in vs.iter() {
                    vehs.push((vir.clone(), v.clone()));
                }
            } else {
                println!("No EPA test data available for year {}", vir.year);
            }
        } else {
            println!("No FE.gov data available for year {}", vir.year);
        }
    }
    vehs
}

pub fn import_and_save_all_vehicles(
    inputs: &[VehicleInputRecord],
    fegov_data_by_year: &HashMap<u32, Vec<VehicleDataFE>>,
    epatest_data_by_year: &HashMap<u32, Vec<VehicleDataEPA>>,
    output_dir_path: &Path,
) -> Result<(), anyhow::Error> {
    for (idx, (vir, veh)) in
        import_all_vehicles_from_record(inputs, fegov_data_by_year, epatest_data_by_year)
            .iter()
            .enumerate()
    {
        let mut outfile: PathBuf = PathBuf::new();
        outfile.push(output_dir_path);
        if idx > 0 {
            let path = Path::new(&vir.output_file_name);
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let ext = path.extension().unwrap().to_str().unwrap();
            let output_file_name = format!("{stem}-{idx}.{ext}");
            println!("Multiple configurations found: output_file_name = {output_file_name}");
            outfile.push(Path::new(&output_file_name));
        } else {
            outfile.push(Path::new(&vir.output_file_name));
        }
        if let Some(full_outfile) = outfile.to_str() {
            veh.to_file(full_outfile)?;
        } else {
            println!("Could not determine output file path");
        }
    }
    Ok(())
}

#[cfg(test)]
mod vehicle_utils_tests {
    use super::*;

    #[test]
    fn test_get_error_val() {
        let time_steps: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let model: Array1<f64> = array![1.1, 4.6, 2.5, 3.7, 5.0];
        let test: Array1<f64> = array![2.1, 4.5, 3.4, 4.8, 6.3];

        let error_val: f64 = get_error_val(model, test, time_steps);
        println!("Error Value: {}", error_val);

        assert!(error_val.approx_eq(&0.8124999999999998, 1e-10));
    }

    #[test]
    fn test_abc_to_drag_coeffs() {
        let mut veh: RustVehicle = RustVehicle::mock_vehicle();
        let a: f64 = 25.91;
        let b: f64 = 0.1943;
        let c: f64 = 0.01796;

        let (drag_coef, wheel_rr_coef): (f64, f64) = abc_to_drag_coeffs(
            &mut veh,
            a,
            b,
            c,
            Some(false),
            None,
            None,
            Some(true),
            Some(false),
        );
        println!("Drag Coef: {}", drag_coef);
        println!("Wheel RR Coef: {}", wheel_rr_coef);

        assert!(drag_coef.approx_eq(&0.24676817210529464, 1e-5));
        assert!(wheel_rr_coef.approx_eq(&0.0068603812443132645, 1e-6));
        assert_eq!(drag_coef, veh.drag_coef);
        assert_eq!(wheel_rr_coef, veh.wheel_rr_coef);
    }

    #[test]
    fn test_create_new_vehicle_from_input_data() {
        let veh_record = VehicleInputRecord {
            make: String::from("Toyota"),
            model: String::from("Camry"),
            year: 2020,
            output_file_name: String::from("2020-toyota-camry.yaml"),
            vehicle_width_in: 72.4,
            vehicle_height_in: 56.9,
            fuel_tank_gal: 15.8,
            ess_max_kwh: 0.0,
            mc_max_kw: 0.0,
            ess_max_kw: 0.0,
            fc_max_kw: None,
        };
        let emiss_info = vec![
            EmissionsInfoFE {
                efid: String::from("LTYXV03.5M5B"),
                score: 5.0,
                smartway_score: -1,
                standard: String::from("L3ULEV70"),
                std_text: String::from("California LEV-III ULEV70"),
            },
            EmissionsInfoFE {
                efid: String::from("LTYXV03.5M5B"),
                score: 5.0,
                smartway_score: -1,
                standard: String::from("T3B70"),
                std_text: String::from("Federal Tier 3 Bin 70"),
            },
        ];
        let emiss_list = EmissionsListFE {
            emissions_info: emiss_info,
        };
        let fegov_data = VehicleDataFE {
            id: 32204,
            alt_veh_type: String::from(""),
            city_mpg_fuel1: 22,
            city_mpg_fuel2: 0,
            co2_g_per_mi: 338,
            comb_mpg_fuel1: 26,
            comb_mpg_fuel2: 0,
            cylinders: String::from("6"),
            displ: String::from("3.5"),
            drive: String::from("Front-Wheel Drive"),
            emissions_list: emiss_list,
            eng_dscr: String::from("SIDI & PFI"),
            ev_motor_kw: String::from(""),
            fe_score: 5,
            fuel_type: String::from("Regular"),
            fuel1: String::from("Regular Gasoline"),
            fuel2: String::from(""),
            ghg_score: 5,
            highway_mpg_fuel1: 33,
            highway_mpg_fuel2: 0,
            make: String::from("Toyota"),
            mfr_code: String::from("TYX"),
            model: String::from("Camry"),
            phev_blended: false,
            phev_city_mpge: 0,
            phev_comb_mpge: 0,
            phev_hwy_mpge: 0,
            range_ev: 0,
            start_stop: String::from("N"),
            trany: String::from("Automatic (S8)"),
            veh_class: String::from("Midsize Cars"),
            year: 2020,
            super_charge: String::from(""),
            turbo_charge: String::from(""),
        };
        let epatest_data = VehicleDataEPA {
            year: 2020,
            mfr_code: String::from("TXY"),
            make: String::from("TOYOTA"),
            model: String::from("CAMRY"),
            test_id: String::from("JTYXV03.5M5B"),
            displ: 3.456,
            eng_pwr_hp: 301,
            cylinders: String::from("6"),
            trany_code: String::from("A"),
            trany_type: String::from("Automatic"),
            gears: 8,
            drive_code: String::from("F"),
            drive: String::from("2-Wheel Drive, Front"),
            test_weight_lbs: 3875.0,
            test_fuel_type: String::from("61"),
            a_lbf: 24.843,
            b_lbf_per_mph: 0.40298,
            c_lbf_per_mph2: 0.015068,
        };
        let other_inputs = vir_to_other_inputs(&veh_record);
        let v = try_make_single_vehicle(&fegov_data, &epatest_data, &other_inputs);
        assert!(v.is_some());
        if let Some(vs) = v {
            assert_eq!(vs.scenario_name, String::from("2020 Toyota Camry"));
            assert_eq!(vs.val_comb_mpgge, 26.0);
        }
    }

    #[test]
    fn test_get_options_for_year_make_model() {
        let year = String::from("2020");
        let make = String::from("Toyota");
        let model = String::from("Corolla");
        let res = get_options_for_year_make_model(&year, &make, &model, None, None);
        if let Err(err) = &res {
            panic!("{:?}", err);
        } else if let Ok(vs) = &res {
            assert!(!vs.is_empty());
        }
    }

    #[test]
    fn test_import_robustness() {
        // Ensure 2019 data is cached
        let ddpath = get_fastsim_data_dir();
        assert!(ddpath.is_some());
        let ddpath = ddpath.unwrap();
        let model_year: u32 = 2019;
        let years = {
            let mut s = HashSet::new();
            s.insert(model_year);
            s
        };
        let cache_url = get_default_cache_url();
        let res = populate_cache_for_given_years_if_needed(ddpath.as_path(), &years, &cache_url);
        assert!(res.is_ok());
        // Load all year/make/models for 2019
        let vehicles_path = ddpath.join(Path::new("2019-vehicles.csv"));
        let veh_records = {
            let file = File::open(vehicles_path);
            if let Ok(f) = file {
                let data_result: Result<Vec<HashMap<String, String>>, Error> =
                    read_records_from_file(f);
                if let Ok(data) = data_result {
                    data
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        };
        let num_records = veh_records.len();
        let mut num_success: usize = 0;
        let other_inputs = OtherVehicleInputs {
            vehicle_height_in: 72.4,
            vehicle_width_in: 56.9,
            fuel_tank_gal: 15.8,
            ess_max_kwh: 0.0,
            mc_max_kw: 0.0,
            ess_max_kw: 0.0,
            fc_max_kw: None,
        };
        for vr in veh_records {
            let make = vr.get("make");
            let model = vr.get("model");
            if let (Some(make), Some(model)) = (make, model) {
                let result =
                    import_all_vehicles(model_year, make, model, &other_inputs, None, None);
                if let Ok(vehs) = &result {
                    if !vehs.is_empty() {
                        num_success += 1;
                    }
                }
            }
        }
        let success_frac: f64 = (num_success as f64) / (num_records as f64);
        assert!(success_frac > 0.5);
    }

    // Enable the test below after github.com/NREL/temp-data (or equivalent) get created
    // #[test]
    // fn test_get_options_for_year_make_model_for_specified_cacheurl_and_data_dir() {
    //     let year = String::from("2020");
    //     let make = String::from("Toyota");
    //     let model = String::from("Corolla");
    //     let data_dir = PathBuf::from("./temp");
    //     let cacheurl = get_default_cache_url();
    //     let r1 = std::fs::create_dir_all(data_dir.as_path());
    //     assert!(r1.is_ok());
    //     let r2 = get_options_for_year_make_model(
    //         &year,
    //         &make,
    //         &model,
    //         Some(cacheurl),
    //         Some(data_dir.to_str().unwrap_or("").to_string()),
    //     );
    //     assert!(r2.is_ok());
    //     if let Ok(vs) = r2 {
    //         assert!(!vs.is_empty());
    //     }
    //     // TODO: delete the data_dir and all contents
    // }
}
