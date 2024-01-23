//! Module for utility functions that support the vehicle struct.

use argmin::core::{CostFunction, Error, Executor, OptimizationResult, State};
use argmin::solver::neldermead::NelderMead;
use curl::easy::Easy;
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
use std::{result::Result, thread, time::Duration};
use ureq::{Error as OtherError, Error::Status, Response};
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

impl SerdeAPI for OptionFE {}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[add_pyo3_api]
/// Struct containing vehicle data from fueleconomy.gov
pub struct VehicleDataFE {
    /// Vehicle ID
    pub id: i32,

    /// Model year
    pub year: u32,
    /// Vehicle make
    pub make: String,
    /// Vehicle model
    pub model: String,

    /// EPA vehicle size class
    #[serde(rename = "VClass")]
    pub veh_class: String,

    /// Drive axle type (FWD, RWD, AWD, 4WD)
    pub drive: String,
    /// Type of alternative fuel vehicle (Hybrid, Plug-in Hybrid, EV)
    #[serde(default, rename = "atvType")]
    pub alt_veh_type: String,

    /// Combined vehicle fuel type (fuel 1 and fuel 2)
    #[serde(rename = "fuelType")]
    pub fuel_type: String,
    /// Fuel type 1
    #[serde(rename = "fuelType1")]
    pub fuel1: String,
    /// Fuel type 2
    #[serde(default, rename = "fuelType2")]
    pub fuel2: String,

    /// Description of engine
    #[serde(default)]
    pub eng_dscr: String,
    /// Number of engine cylinders
    #[serde(default)]
    pub cylinders: String,
    /// Engine displacement in liters
    #[serde(default)]
    pub displ: String,
    /// transmission
    #[serde(rename = "trany")]
    pub transmission: String,

    /// "S" if vehicle has supercharger
    #[serde(default, rename = "sCharger")]
    pub super_charger: String,
    /// "T" if vehicle has turbocharger
    #[serde(default, rename = "tCharger")]
    pub turbo_charger: String,

    /// Stop-start technology
    #[serde(rename = "startStop")]
    pub start_stop: String,

    /// Vehicle operates on blend of gasoline and electricity
    #[serde(rename = "phevBlended")]
    pub phev_blended: bool,
    /// EPA composite gasoline-electricity city MPGe
    #[serde(rename = "phevCity")]
    pub phev_city_mpge: i32,
    /// EPA composite gasoline-electricity combined MPGe
    #[serde(rename = "phevComb")]
    pub phev_comb_mpge: i32,
    /// EPA composite gasoline-electricity highway MPGe
    #[serde(rename = "phevHwy")]
    pub phev_hwy_mpge: i32,

    /// Electric motor power (kW), not very consistent as an input
    #[serde(default, rename = "evMotor")]
    pub ev_motor_kw: String,
    /// EV range
    #[serde(rename = "range")]
    pub range_ev: i32,

    /// City MPG for fuel 1
    #[serde(rename = "city08U")]
    pub city_mpg_fuel1: f64,
    /// City MPG for fuel 2
    #[serde(rename = "cityA08U")]
    pub city_mpg_fuel2: f64,
    /// Unadjusted unroaded city MPG for fuel 1
    #[serde(rename = "UCity")]
    pub unadj_city_mpg_fuel1: f64,
    /// Unadjusted unroaded city MPG for fuel 2
    #[serde(rename = "UCityA")]
    pub unadj_city_mpg_fuel2: f64,
    /// City electricity consumption in kWh/100 mi
    #[serde(rename = "cityE")]
    pub city_kwh_per_100mi: f64,

    /// Adjusted unrounded highway MPG for fuel 1
    #[serde(rename = "highway08U")]
    pub highway_mpg_fuel1: f64,
    /// Adjusted unrounded highway MPG for fuel 2
    #[serde(rename = "highwayA08U")]
    pub highway_mpg_fuel2: f64,
    /// Unadjusted unrounded highway MPG for fuel 1
    #[serde(default, rename = "UHighway")]
    pub unadj_highway_mpg_fuel1: f64,
    /// Unadjusted unrounded highway MPG for fuel 2
    #[serde(default, rename = "UHighwayA")]
    pub unadj_highway_mpg_fuel2: f64,
    /// Highway electricity consumption in kWh/100 mi
    #[serde(default, rename = "highwayE")]
    pub highway_kwh_per_100mi: f64,

    /// Combined MPG for fuel 1
    #[serde(rename = "comb08U")]
    pub comb_mpg_fuel1: f64,
    /// Combined MPG for fuel 2
    #[serde(rename = "combA08U")]
    pub comb_mpg_fuel2: f64,
    /// Combined electricity consumption in kWh/100 mi
    #[serde(default, rename = "combE")]
    pub comb_kwh_per_100mi: f64,

    /// List of emissions tests
    #[serde(rename = "emissionsList")]
    pub emissions_list: EmissionsListFE,
}

impl SerdeAPI for VehicleDataFE {}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "camelCase")]
#[add_pyo3_api]
/// Struct containing list of emissions tests from fueleconomy.gov
pub struct EmissionsListFE {
    ///
    pub emissions_info: Vec<EmissionsInfoFE>,
}

impl SerdeAPI for EmissionsListFE {}

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

impl SerdeAPI for EmissionsInfoFE {}

#[derive(Default, PartialEq, Clone, Debug, Deserialize, Serialize)]
#[add_pyo3_api]
/// Struct containing vehicle data from EPA database
pub struct VehicleDataEPA {
    /// Model year
    #[serde(rename = "Model Year")]
    pub year: u32,
    /// Vehicle make
    #[serde(rename = "Represented Test Veh Make")]
    pub make: String,
    /// Vehicle model
    #[serde(rename = "Represented Test Veh Model")]
    pub model: String,
    /// Vehicle test group
    #[serde(rename = "Actual Tested Testgroup")]
    pub test_id: String,
    /// Engine displacement
    #[serde(rename = "Test Veh Displacement (L)")]
    pub displ: f64,
    /// Engine power in hp
    #[serde(rename = "Rated Horsepower")]
    pub eng_pwr_hp: u32,
    /// Number of cylinders
    #[serde(rename = "# of Cylinders and Rotors")]
    pub cylinders: String,
    /// Transmission type code
    #[serde(rename = "Tested Transmission Type Code")]
    pub transmission_code: String,
    /// Transmission type
    #[serde(rename = "Tested Transmission Type")]
    pub transmission_type: String,
    /// Number of gears
    #[serde(rename = "# of Gears")]
    pub gears: u32,
    /// Drive system code
    #[serde(rename = "Drive System Code")]
    pub drive_code: String,
    /// Drive system type
    #[serde(rename = "Drive System Description")]
    pub drive: String,
    /// Test weight in lbs
    #[serde(rename = "Equivalent Test Weight (lbs.)")]
    pub test_weight_lbs: f64,
    /// Fuel type used for EPA test
    #[serde(rename = "Test Fuel Type Description")]
    pub test_fuel_type: String,
    /// Dyno coefficient a in lbf
    #[serde(rename = "Target Coef A (lbf)")]
    pub a_lbf: f64,
    /// Dyno coefficient b in lbf/mph
    #[serde(rename = "Target Coef B (lbf/mph)")]
    pub b_lbf_per_mph: f64,
    /// Dyno coefficient c in lbf/mph^2
    #[serde(rename = "Target Coef C (lbf/mph**2)")]
    pub c_lbf_per_mph2: f64,
}

impl SerdeAPI for VehicleDataEPA {}

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
) -> anyhow::Result<Vec<VehicleDataFE>> {
    // prep the cache for year
    let y: u32 = year.trim().parse()?;
    let ys: HashSet<u32> = {
        let mut h = HashSet::new();
        h.insert(y);
        h
    };
    let ddpath = if let Some(dd) = data_dir {
        PathBuf::from(dd)
    } else {
        create_project_subdir("fe_label_data")?
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

fn derive_transmission_specs(fegov: &VehicleDataFE) -> (u32, String) {
    let num_gears_fe_gov: u32;
    let transmission_fe_gov: String;
    // Based on reference: https://www.fueleconomy.gov/feg/findacarhelp.shtml#engine
    if fegov.transmission.contains("Manual") {
        transmission_fe_gov = String::from('M');
        num_gears_fe_gov = fegov.transmission.as_str()[fegov.transmission.find("-spd").unwrap() - 1
            ..fegov.transmission.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("variable gear ratios") {
        transmission_fe_gov = String::from("CVT");
        num_gears_fe_gov = 1;
    } else if fegov.transmission.contains("AV-S") {
        transmission_fe_gov = String::from("SCV");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("AM-S") {
        transmission_fe_gov = String::from("AMS");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains('S') {
        transmission_fe_gov = String::from("SA");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("-spd") {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = fegov.transmission.as_str()[fegov.transmission.find("-spd").unwrap() - 1
            ..fegov.transmission.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = {
            let res: Result<u32, ParseIntError> = fegov.transmission.as_str()
                [fegov.transmission.find("(A").unwrap() + 2..fegov.transmission.find(')').unwrap()]
                .parse();
            if let Ok(n) = res {
                n
            } else {
                1
            }
        }
    }
    (num_gears_fe_gov, transmission_fe_gov)
}

/// Match EPA Test Data with FuelEconomy.gov data and return best match
/// The matching algorithm tries to find the best match in the EPA Test data for the given FuelEconomy.gov data
/// The algorithm works as follows:
/// - only EPA Test Data matching the year and make of the FuelEconomy.gov data will be considered
/// - we try to match on both the efid/test id and also the model name
/// - next, for each match, we calculate a score based on matching various powertrain aspects based on:
///     - transmission type
///     - number of gears in the transmission
///     - drive type (all-wheel drive / 4-wheel drive, etc.)
///     - (for non-EVs)
///         - engine displacement
///         - number of cylinders
/// RETURNS: the EPA Test data with the best match on make and/or efid/test id. When multiple vehicles match
///          the same make name/ efid/test-id, we return the one with the highest score
fn match_epatest_with_fegov_v2(
    fegov: &VehicleDataFE,
    epatest_data: &[VehicleDataEPA],
) -> Option<VehicleDataEPA> {
    let fe_model_upper: String = fegov.model.to_uppercase().replace("4WD", "AWD");
    let fe_model_words: Vec<&str> = fe_model_upper.split_ascii_whitespace().collect();
    let num_fe_model_words = fe_model_words.len();
    let fegov_disp = fegov.displ.parse::<f64>().unwrap_or_default();
    let efid = if !fegov.emissions_list.emissions_info.is_empty() {
        fegov.emissions_list.emissions_info[0].efid.clone()
    } else {
        String::new()
    };
    let fegov_drive = {
        let mut s = String::new();
        if !fegov.drive.is_empty() {
            let maybe_char = fegov.drive.chars().next();
            if let Some(c) = maybe_char {
                s.push(c);
            }
            s
        } else {
            s
        }
    };
    let (num_gears_fe_gov, transmission_fe_gov) = derive_transmission_specs(fegov);
    let epa_candidates = {
        let mut xs: Vec<(f64, f64, VehicleDataEPA)> = Vec::new();
        for x in epatest_data {
            if x.year == fegov.year && x.make.eq_ignore_ascii_case(&fegov.make) {
                let mut score = 0.0;

                // Things we Don't Want to Match
                if x.test_fuel_type.contains("Cold CO") {
                    continue;
                }
                let matching_test_id = if !x.test_id.is_empty() && !efid.is_empty() {
                    x.test_id.ends_with(&efid[1..efid.len()])
                } else {
                    false
                };
                // ID match
                let name_match = if matching_test_id || x.model.eq_ignore_ascii_case(&fegov.model) {
                    1.0
                } else {
                    let epa_model_upper = x.model.to_uppercase().replace("4WD", "AWD");
                    let epa_model_words: Vec<&str> =
                        epa_model_upper.split_ascii_whitespace().collect();
                    let num_epa_model_words = epa_model_words.len();
                    let mut match_count = 0;
                    for word in &epa_model_words {
                        match_count += fe_model_words.contains(word) as i64;
                    }
                    let match_frac: f64 = (match_count as f64 * match_count as f64)
                        / (num_epa_model_words as f64 * num_fe_model_words as f64);
                    match_frac
                };
                if name_match == 0.0 {
                    continue;
                }
                // By PT Type
                if fegov.alt_veh_type == *"EV" {
                    if x.cylinders.is_empty() && x.displ.round() == 0.0 {
                        score += 1.0;
                    }
                } else {
                    let epa_disp = (x.displ * 10.0).round() / 10.0;
                    if x.cylinders == fegov.cylinders && epa_disp == fegov_disp {
                        score += 1.0;
                    }
                }
                // Drive Code
                let drive_code = if x.model.contains("4WD")
                    || x.model.contains("AWD")
                    || x.drive.contains("4-Wheel Drive")
                {
                    String::from('A')
                } else {
                    x.drive.clone()
                };
                if drive_code == fegov_drive {
                    score += 1.0;
                }
                // Transmission Type and Num Gears
                if x.transmission_code == transmission_fe_gov {
                    score += 0.5;
                } else if transmission_fe_gov.starts_with(x.transmission_type.as_str()) {
                    score += 0.25;
                }
                if x.gears == num_gears_fe_gov {
                    score += 0.5;
                }
                xs.push((name_match, score, x.clone()));
            }
        }
        xs
    };
    if epa_candidates.is_empty() {
        None
    } else {
        let mut largest_id_match_value = 0.0;
        let mut largest_score_value = 0.0;
        let mut best_idx: usize = 0;
        for (idx, item) in epa_candidates.iter().enumerate() {
            if item.0 > largest_id_match_value
                || (item.0 == largest_id_match_value && item.1 > largest_score_value)
            {
                largest_id_match_value = item.0;
                largest_score_value = item.1;
                best_idx = idx;
            }
        }
        if largest_id_match_value == 0.0 {
            None
        } else {
            Some(epa_candidates[best_idx].2.clone())
        }
    }
}

/// Match EPA Test Data with FuelEconomy.gov data and return best match
#[allow(dead_code)]
fn match_epatest_with_fegov(
    fegov: &VehicleDataFE,
    epatest_data: &[VehicleDataEPA],
) -> Option<VehicleDataEPA> {
    if fegov.emissions_list.emissions_info.is_empty() {
        return None;
    }
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
    if fegov.transmission.contains("Manual") {
        transmission_fe_gov = String::from('M');
        num_gears_fe_gov = fegov.transmission.as_str()[fegov.transmission.find("-spd").unwrap() - 1
            ..fegov.transmission.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("variable gear ratios") {
        transmission_fe_gov = String::from("CVT");
        num_gears_fe_gov = 1;
    } else if fegov.transmission.contains("AV-S") {
        transmission_fe_gov = String::from("SCV");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("AM-S") {
        transmission_fe_gov = String::from("AMS");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains('S') {
        transmission_fe_gov = String::from("SA");
        num_gears_fe_gov = fegov.transmission.as_str()
            [fegov.transmission.find('S').unwrap() + 1..fegov.transmission.find(')').unwrap()]
            .parse()
            .unwrap();
    } else if fegov.transmission.contains("-spd") {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = fegov.transmission.as_str()[fegov.transmission.find("-spd").unwrap() - 1
            ..fegov.transmission.find("-spd").unwrap()]
            .parse()
            .unwrap();
    } else {
        transmission_fe_gov = String::from('A');
        num_gears_fe_gov = {
            let res: Result<u32, ParseIntError> = fegov.transmission.as_str()
                [fegov.transmission.find("(A").unwrap() + 2..fegov.transmission.find(')').unwrap()]
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
            && (veh_epa.transmission_code == transmission_fe_gov
                || fegov
                    .transmission
                    .starts_with(veh_epa.transmission_type.as_str()))
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

impl SerdeAPI for OtherVehicleInputs {}

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
    let data_dir_path = if let Some(data_dir) = data_dir {
        PathBuf::from(data_dir)
    } else {
        create_project_subdir("fe_label_data")?
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
                if let Some(epa_data) = match_epatest_with_fegov_v2(&fe_gov_data, epa_data) {
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
        min_soc = 0.0;
        max_soc = 1.0;
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
            0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35,
        ]);
        min_soc = 0.0;
        max_soc = 1.0;
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
            0.10, 0.12, 0.28, 0.35, 0.375, 0.39, 0.40, 0.40, 0.38, 0.37, 0.36, 0.35,
        ]);
        min_soc = 0.0;
        max_soc = 1.0;
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
            0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30,
        ]);
        mc_max_kw = epa_data.eng_pwr_hp as f64 / HP_PER_KW;
        min_soc = 0.0;
        max_soc = 1.0;
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

    // TODO: fix glider_kg calculation
    // https://github.com/NREL/fastsim/pull/30#issuecomment-1841413126
    //
    // let glider_kg = (epa_data.test_weight_lbs / LBS_PER_KG)
    //     - ref_veh.cargo_kg
    //     - ref_veh.trans_kg
    //     - ref_veh.comp_mass_multiplier
    //         * ((fs_max_kw / ref_veh.fs_kwh_per_kg)
    //             + (ref_veh.fc_base_kg + fc_max_kw / ref_veh.fc_kw_per_kg)
    //             + (ref_veh.mc_pe_base_kg + mc_max_kw * ref_veh.mc_pe_kg_per_kw)
    //             + (ref_veh.ess_base_kg + ess_max_kwh * ref_veh.ess_kg_per_kwh));
    let mut veh = RustVehicle {
        veh_override_kg: Some(epa_data.test_weight_lbs / LBS_PER_KG),
        veh_cg_m: match fe_gov_data.drive.as_str() {
            "Front-Wheel Drive" => 0.53,
            _ => -0.53,
        },
        // glider_kg,
        scenario_name: format!(
            "{} {} {}",
            fe_gov_data.year, fe_gov_data.make, fe_gov_data.model
        ),
        max_roadway_chg_kw: Default::default(),
        selection: 0,
        veh_year: fe_gov_data.year,
        veh_pt_type: String::from(veh_pt_type),
        drag_coef: 0.0, // overridden
        frontal_area_m2: 0.85 * (other_inputs.vehicle_width_in * other_inputs.vehicle_height_in)
            / (IN_PER_M * IN_PER_M),
        fs_kwh,
        idle_fc_kw: 0.0,
        mc_eff_map: Array1::<f64>::zeros(LARGE_BASELINE_EFF.len()),
        wheel_rr_coef: 0.0, // overridden
        stop_start: fe_gov_data.start_stop == "Y",
        force_aux_on_fc: false,
        val_udds_mpgge: fe_gov_data.city_mpg_fuel1,
        val_hwy_mpgge: fe_gov_data.highway_mpg_fuel1,
        val_comb_mpgge: fe_gov_data.comb_mpg_fuel1,
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
        if let Some(epa_data) = match_epatest_with_fegov_v2(&hit, epatest_data) {
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
                        v.scenario_name, alt_type, hit.cylinders, hit.displ, hit.transmission
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
pub fn export_vehicle_to_file(veh: &RustVehicle, file_path: String) -> anyhow::Result<()> {
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

    let cyc: RustCycle = RustCycle {
        time_s: (0..cd_len as i32).map(f64::from).collect(),
        mps: Array::linspace(vmax_mph / super::params::MPH_PER_MPS, 0.0, cd_len),
        grade: Array::zeros(cd_len),
        road_type: Array::zeros(cd_len),
        name: String::from("cycle"),
        orphaned: false,
    };

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
) -> anyhow::Result<Vec<VehicleInputRecord>> {
    let f = File::open(filepath)?;
    read_records_from_file(f)
}

fn read_records_from_file<T: DeserializeOwned>(
    rdr: impl std::io::Read + std::io::Seek,
) -> anyhow::Result<Vec<T>> {
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
                        output.entry(id).or_default();
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
) -> anyhow::Result<Vec<VehicleDataFE>> {
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

            year: item.get("year").unwrap().parse::<u32>().unwrap(),
            make: item.get("make").unwrap().clone(),
            model: item.get("model").unwrap().clone(),

            veh_class: item.get("VClass").unwrap().clone(),

            drive: item.get("drive").unwrap().clone(),
            alt_veh_type: item.get("atvType").unwrap().clone(),

            fuel_type: item.get("fuelType").unwrap().clone(),
            fuel1: item.get("fuelType1").unwrap().clone(),
            fuel2: item.get("fuelType2").unwrap().clone(),

            eng_dscr: item.get("eng_dscr").unwrap().clone(),
            cylinders: item.get("cylinders").unwrap().clone(),
            displ: item.get("displ").unwrap().clone(),
            transmission: item.get("trany").unwrap().clone(),

            super_charger: item.get("sCharger").unwrap().clone(),
            turbo_charger: item.get("tCharger").unwrap().clone(),

            start_stop: item.get("startStop").unwrap().clone(),

            phev_blended: item
                .get("phevBlended")
                .unwrap()
                .trim()
                .to_lowercase()
                .parse::<bool>()
                .unwrap(),
            phev_city_mpge: item.get("phevCity").unwrap().parse::<i32>().unwrap(),
            phev_comb_mpge: item.get("phevComb").unwrap().parse::<i32>().unwrap(),
            phev_hwy_mpge: item.get("phevHwy").unwrap().parse::<i32>().unwrap(),

            ev_motor_kw: item.get("evMotor").unwrap().clone(),
            range_ev: item.get("range").unwrap().parse::<i32>().unwrap(),

            city_mpg_fuel1: item.get("city08U").unwrap().parse::<f64>().unwrap(),
            city_mpg_fuel2: item.get("cityA08U").unwrap().parse::<f64>().unwrap(),
            unadj_city_mpg_fuel1: item.get("UCity").unwrap().parse::<f64>().unwrap(),
            unadj_city_mpg_fuel2: item.get("UCityA").unwrap().parse::<f64>().unwrap(),
            city_kwh_per_100mi: item.get("cityE").unwrap().parse::<f64>().unwrap(),

            highway_mpg_fuel1: item.get("highway08U").unwrap().parse::<f64>().unwrap(),
            highway_mpg_fuel2: item.get("highwayA08U").unwrap().parse::<f64>().unwrap(),
            unadj_highway_mpg_fuel1: item.get("UHighway").unwrap().parse::<f64>().unwrap(),
            unadj_highway_mpg_fuel2: item.get("UHighwayA").unwrap().parse::<f64>().unwrap(),
            highway_kwh_per_100mi: item.get("highwayE").unwrap().parse::<f64>().unwrap(),

            comb_mpg_fuel1: item.get("comb08U").unwrap().parse::<f64>().unwrap(),
            comb_mpg_fuel2: item.get("combA08U").unwrap().parse::<f64>().unwrap(),
            comb_kwh_per_100mi: item.get("combE").unwrap().parse::<f64>().unwrap(),

            emissions_list,
        };
        output.push(vd);
    }
    Ok(output)
}

/// Given the path to a zip archive, print out the names of the files within that archive
pub fn list_zip_contents(filepath: &Path) -> anyhow::Result<()> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    for i in 0..zip.len() {
        let file = zip.by_index(i)?;
        println!("Filename: {}", file.name());
    }
    Ok(())
}

/// Extract zip archive at filepath to destination directory at dest_dir
pub fn extract_zip(filepath: &Path, dest_dir: &Path) -> anyhow::Result<()> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    zip.extract(dest_dir)?;
    Ok(())
}

/// Assumes the parent directory exists. Assumes file doesn't exist (i.e., newly created) or that it will be truncated if it does.
pub fn download_file_from_url(url: &str, file_path: &Path) -> anyhow::Result<()> {
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
    println!("Downloaded data from {}; bytes: {}", url, buffer.len());
    if buffer.is_empty() {
        return Err(anyhow!("No data available from {url}"));
    }
    {
        let mut file = match File::create(file_path) {
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
) -> anyhow::Result<HashMap<u32, Vec<VehicleDataEPA>>> {
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
) -> anyhow::Result<HashMap<u32, HashMap<u32, Vec<EmissionsInfoFE>>>> {
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
) -> anyhow::Result<HashMap<u32, Vec<VehicleDataFE>>> {
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
    String::from("https://github.com/NREL/vehicle-data/raw/main/")
}

fn get_cache_url_for_year(cache_url: &str, year: &u32) -> anyhow::Result<Option<String>> {
    let maybe_slash = if cache_url.ends_with('/') { "" } else { "/" };
    let target_url = format!("{cache_url}{maybe_slash}{year}.zip");
    Ok(Some(target_url))
}

fn extract_file_from_zip(
    zip_file_path: &Path,
    name_of_file_to_extract: &str,
    path_to_save_to: &Path,
) -> anyhow::Result<()> {
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
) -> anyhow::Result<bool> {
    let mut all_data_available = true;
    for year in years {
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
) -> anyhow::Result<Vec<RustVehicle>> {
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
    let data_dir_path = if let Some(dd_path) = data_dir {
        PathBuf::from(dd_path.clone())
    } else {
        create_project_subdir("fe_label_data")?
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
) -> anyhow::Result<()> {
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
) -> anyhow::Result<()> {
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

#[derive(Deserialize)]
pub struct ObjectLinks {
    #[serde(rename = "self")]
    pub self_url: Option<String>,
    pub git: Option<String>,
    pub html: Option<String>,
}

#[derive(Deserialize)]
pub struct GitObjectInfo {
    pub name: String,
    pub path: String,
    pub sha: Option<String>,
    pub size: Option<i64>,
    pub url: String,
    pub html_url: Option<String>,
    pub git_url: Option<String>,
    pub download_url: Option<String>,
    #[serde(rename = "type")]
    pub url_type: String,
    #[serde(rename = "_links")]
    pub links: Option<ObjectLinks>,
}

const VEHICLE_REPO_LIST_URL: &'static str =
    &"https://api.github.com/repos/NREL/fastsim-vehicles/contents/public";

/// Function that takes a url and calls the url. If a 503 or 429 error is
/// thrown, it tries again after a pause, up to four times. It returns either a
/// result or an error.  
/// # Arguments  
/// - url: url to be called
/// Source: https://docs.rs/ureq/latest/ureq/enum.Error.html
fn get_response<S: AsRef<str>>(url: S) -> Result<Response, OtherError> {
    for _ in 1..4 {
        match ureq::get(url.as_ref()).call() {
            Err(Status(503, r)) | Err(Status(429, r)) | Err(Status(403, r)) => {
                let retry: Option<u64> = r.header("retry-after").and_then(|h| h.parse().ok());
                let retry = retry.unwrap_or(5);
                eprintln!("{} for {}, retry in {}", r.status(), r.get_url(), retry);
                thread::sleep(Duration::from_secs(retry));
            }
            result => return result,
        };
    }
    // Ran out of retries; try one last time and return whatever result we get.
    ureq::get(url.as_ref()).call()
}

/// Returns a list of vehicle file names in the Fastsim Vehicle Repo, or,
/// optionally, a different GitHub repo, in which case the url provided needs to
/// be the url for the file tree within GitHub for the root folder the Rust
/// objects, for example
/// "https://api.github.com/repos/NREL/fastsim-vehicles/contents/public"  
/// Note: for each file, the output will list the vehicle file name, including
/// the path from the root of the repository  
/// # Arguments  
/// - repo_url: url to the GitHub repository, Option, if None, defaults to the
///   FASTSim Vehicle Repo
pub fn fetch_github_list(repo_url: Option<String>) -> anyhow::Result<Vec<String>> {
    let repo_url = repo_url.unwrap_or(VEHICLE_REPO_LIST_URL.to_string());
    let response = get_response(repo_url)?.into_reader();
    let github_list: Vec<GitObjectInfo> =
        serde_json::from_reader(response).with_context(|| "Cannot parse github vehicle list.")?;
    let mut vehicle_name_list: Vec<String> = Vec::new();
    for object in github_list.iter() {
        if object.url_type == "dir" {
            let url = &object.url;
            let vehicle_name_sublist = fetch_github_list(Some(url.to_owned()))?;
            for name in vehicle_name_sublist.iter() {
                vehicle_name_list.push(name.to_owned());
            }
        } else if object.url_type == "file" {
            let url = url::Url::parse(&object.url)?;
            let path = &object.path;
            let format = url
                .path_segments()
                .and_then(|segments| segments.last())
                .and_then(|filename| Path::new(filename).extension())
                .and_then(OsStr::to_str)
                .with_context(|| "Could not parse file format from URL: {url:?}")?;
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => vehicle_name_list.push(path.to_owned()),
                "json" => vehicle_name_list.push(path.to_owned()),
                _ => continue,
            }
        } else {
            continue;
        }
    }
    Ok(vehicle_name_list)
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

            year: 2020,
            make: String::from("Toyota"),
            model: String::from("Camry"),

            veh_class: String::from("Midsize Cars"),

            drive: String::from("Front-Wheel Drive"),
            alt_veh_type: String::from(""),

            fuel_type: String::from("Regular"),
            fuel1: String::from("Regular Gasoline"),
            fuel2: String::from(""),

            eng_dscr: String::from("SIDI & PFI"),
            cylinders: String::from("6"),
            displ: String::from("3.5"),
            transmission: String::from("Automatic (S8)"),

            super_charger: String::from(""),
            turbo_charger: String::from(""),

            start_stop: String::from("N"),

            phev_blended: false,
            phev_city_mpge: 0,
            phev_comb_mpge: 0,
            phev_hwy_mpge: 0,

            ev_motor_kw: String::from(""),
            range_ev: 0,

            city_mpg_fuel1: 16.4596,
            city_mpg_fuel2: 0.0,
            unadj_city_mpg_fuel1: 20.2988,
            unadj_city_mpg_fuel2: 0.0,
            city_kwh_per_100mi: 0.0,

            highway_mpg_fuel1: 22.5568,
            highway_mpg_fuel2: 0.0,
            unadj_highway_mpg_fuel1: 30.1798,
            unadj_highway_mpg_fuel2: 0.0,
            highway_kwh_per_100mi: 0.0,

            comb_mpg_fuel1: 18.7389,
            comb_mpg_fuel2: 0.0,
            comb_kwh_per_100mi: 0.0,

            emissions_list: emiss_list,
        };
        let epatest_data = VehicleDataEPA {
            year: 2020,
            make: String::from("TOYOTA"),
            model: String::from("CAMRY"),
            test_id: String::from("JTYXV03.5M5B"),
            displ: 3.456,
            eng_pwr_hp: 301,
            cylinders: String::from("6"),
            transmission_code: String::from("A"),
            transmission_type: String::from("Automatic"),
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
            assert_eq!(vs.val_comb_mpgge, 18.7389);
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
        let ddpath = create_project_subdir("fe_label_data").unwrap();
        let model_year: u32 = 2019;
        let years = {
            let mut s = HashSet::new();
            s.insert(model_year);
            s
        };
        let cache_url = get_default_cache_url();
        populate_cache_for_given_years_if_needed(ddpath.as_path(), &years, &cache_url).unwrap();
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
        let mut num_records = 0;
        let max_iter = veh_records.len();
        // NOTE: below, we can use fewer records in the interest of time as this is a long test with all records
        // We skip because the vehicles at the beginning of the file tend to be more exotic and to not have
        // EPA test entries. Thus, they are a bad representation of the whole.
        let skip_idx: usize = 200;
        for (num_iter, vr) in veh_records.iter().enumerate() {
            if num_iter % skip_idx != 0 {
                continue;
            }
            if num_iter >= max_iter {
                break;
            }
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
            } else {
                panic!("Unable to find make and model in vehicle record");
            }
            num_records += 1;
        }
        let success_frac: f64 = (num_success as f64) / (num_records as f64);
        assert!(success_frac > 0.90, "success_frac = {}", success_frac);
    }

    #[test]
    fn test_get_options_for_year_make_model_for_specified_cacheurl_and_data_dir() {
        let year = String::from("2020");
        let make = String::from("Toyota");
        let model = String::from("Corolla");
        let temp_dir = tempfile::tempdir().unwrap();
        let data_dir = temp_dir.path();
        let cacheurl = get_default_cache_url();
        assert!(!get_options_for_year_make_model(
            &year,
            &make,
            &model,
            Some(cacheurl),
            Some(data_dir.to_str().unwrap().to_string()),
        )
        .unwrap()
        .is_empty());
    }

    // NOTE: this test does not seem to reliably pass. Sometimes the function
    // will give a 403 error and sometimes it will succeed -- I don't think
    // there's any way to ensure the function succeeds 100% of the time.
    #[test]
    fn test_fetch_github_list() {
        let list = fetch_github_list(Some(
            "https://api.github.com/repos/NREL/fastsim-vehicles/contents".to_owned(),
        ))
        .unwrap();
        let other_list = fetch_github_list(None).unwrap();
        println!("{:?}", list);
        println!("{:?}", other_list);
    }
}
