//! Module for utility functions that support the vehicle struct.

use argmin::core::{CostFunction, Error, Executor, OptimizationResult, State};
use argmin::solver::neldermead::NelderMead;
use curl::easy::Easy;
use ndarray::{array, Array1};
use polynomial::Polynomial;
use serde_xml_rs::from_str;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use crate::air::*;
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::simdrive::RustSimDrive;
use crate::vehicle::RustVehicle;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing list of transmission options for vehicle from fueleconomy.gov
struct VehicleOptionsFE {
    #[serde(rename = "menuItem")]
    /// List of vehicle options (transmission and id)
    options: Vec<OptionFE>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing transmission and id of a vehicle option from fueleconomy.gov
struct OptionFE {
    #[serde(rename = "text")]
    /// Transmission of vehicle
    transmission: String,
    #[serde(rename = "value")]
    /// ID of vehicle on fueleconomy.gov
    id: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
/// Struct containing vehicle data from fueleconomy.gov
pub struct VehicleDataFE {
    #[serde(default, rename = "atvType")]
    /// Type of alternative fuel vehicle (Hybrid, Plug-in Hybrid, EV)
    alt_veh_type: String,
    #[serde(rename = "city08")]
    /// City MPG for fuel 1
    city_mpg_fuel1: i32,
    #[serde(rename = "cityA08")]
    /// City MPG for fuel 2
    city_mpg_fuel2: i32,
    #[serde(rename = "co2")]
    /// Tailpipe CO2 emissions in grams/mile
    co2_g_per_mi: i32,
    #[serde(rename = "comb08")]
    /// Combined MPG for fuel 1
    comb_mpg_fuel1: i32,
    #[serde(rename = "combA08")]
    /// Combined MPG for fuel 2
    comb_mpg_fuel2: i32,
    #[serde(default)]
    /// Number of engine cylinders
    cylinders: String,
    #[serde(default)]
    /// Engine displacement in liters
    displ: String,
    /// Drive axle type (FWD, RWD, AWD, 4WD)
    drive: String,
    #[serde(rename = "emissionsList")]
    /// List of emissions tests
    emissions_list: EmissionsListFE,
    #[serde(default)]
    /// Description of engine
    eng_dscr: String,
    #[serde(default, rename = "evMotor")]
    /// Electric motor power (kW)
    ev_motor_kw: String,
    #[serde(rename = "feScore")]
    /// EPA fuel economy score
    fe_score: i32,
    #[serde(rename = "fuelType")]
    /// Combined vehicle fuel type (fuel 1 and fuel 2)
    fuel_type: String,
    #[serde(rename = "fuelType1")]
    /// Fuel type 1
    fuel1: String,
    #[serde(rename = "fuelType2")]
    /// Fuel type 2
    fuel2: String,
    #[serde(rename = "ghgScore")]
    /// EPA GHG Score
    ghg_score: i32,
    #[serde(rename = "highway08")]
    /// Highway MPG for fuel 1
    highway_mpg_fuel1: i32,
    #[serde(rename = "highwayA08")]
    /// Highway MPG for fuel 2
    highway_mpg_fuel2: i32,
    /// Manufacturer
    make: String,
    #[serde(rename = "mfrCode")]
    /// Manufacturer code
    mfr_code: String,
    /// Model name
    model: String,
    #[serde(rename = "phevBlended")]
    /// Vehicle operates on blend of gasoline and electricity
    phev_blended: bool,
    #[serde(rename = "phevCity")]
    /// EPA composite gasoline-electricity city MPGe
    phev_city_mpge: i32,
    #[serde(rename = "phevComb")]
    /// EPA composite gasoline-electricity combined MPGe
    phev_comb_mpge: i32,
    #[serde(rename = "phevHwy")]
    /// EPA composite gasoline-electricity highway MPGe
    phev_hwy_mpge: i32,
    #[serde(rename = "startStop")]
    /// Stop-start technology
    start_stop: String,
    /// transmission
    trany: String,
    #[serde(rename = "VClass")]
    /// EPA vehicle size class
    veh_class: String,
    /// Model year
    year: u32,
    #[serde(default, rename = "sCharger")]
    /// Vehicle is supercharged
    super_charge: String,
    #[serde(default, rename = "tCharger")]
    /// Vehicle is turbocharged
    turbo_charge: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
/// Struct containing list of emissions tests from fueleconomy.gov
struct EmissionsListFE {
    ///
    emissions_info: Vec<EmissionsInfoFE>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
/// Struct containing emissions test results from fueleconomy.gov
struct EmissionsInfoFE {
    /// Engine family id / EPA test group
    efid: String,
    /// EPA smog rating
    score: f64,
    /// SmartWay score
    smartway_score: i32,
    /// Vehicle emission standard code
    standard: String,
    /// Vehicle emission standard
    std_text: String,
}

#[derive(Default, PartialEq, Clone, Debug, Deserialize, Serialize)]
/// Struct containing vehicle data from EPA database
struct VehicleDataEPA {
    #[serde(rename = "Model Year")]
    /// Model year
    year: u32,
    #[serde(rename = "Veh Mfr Code")]
    /// Vehicle manufacturer code
    mfr_code: String,
    #[serde(rename = "Represented Test Veh Make")]
    /// Vehicle make
    make: String,
    #[serde(rename = "Represented Test Veh Model")]
    /// Vehicle model
    model: String,
    #[serde(rename = "Actual Tested Testgroup")]
    /// Vehicle test group
    test_id: String,
    #[serde(rename = "Test Veh Displacement (L)")]
    /// Engine displacement
    displ: f64,
    #[serde(rename = "Rated Horsepower")]
    /// Engine power in hp
    eng_pwr_hp: u32,
    #[serde(rename = "# of Cylinders and Rotors")]
    /// Number of cylinders
    cylinders: String,
    #[serde(rename = "Tested Transmission Type Code")]
    /// Transmission type code
    trany_code: String,
    #[serde(rename = "Tested Transmission Type")]
    /// Transmission type
    trany_type: String,
    #[serde(rename = "# of Gears")]
    /// Number of gears
    gears: u32,
    #[serde(rename = "Drive System Code")]
    /// Drive system code
    drive_code: String,
    #[serde(rename = "Drive System Description")]
    /// Drive system type
    drive: String,
    #[serde(rename = "Equivalent Test Weight (lbs.)")]
    /// Test weight in lbs
    test_weight_lbs: f64,
    #[serde(rename = "Test Fuel Type Description")]
    /// Fuel type used for EPA test
    test_fuel_type: String,
    #[serde(rename = "Target Coef A (lbf)")]
    /// Dyno coefficient a in lbf
    a_lbf: f64,
    #[serde(rename = "Target Coef B (lbf/mph)")]
    /// Dyno coefficient b in lbf/mph
    b_lbf_per_mph: f64,
    #[serde(rename = "Target Coef C (lbf/mph**2)")]
    /// Dyno coefficient c in lbf/mph^2
    c_lbf_per_mph2: f64,
}

fn get_fuel_economy_gov_data(year: &str, make: &str, model: &str) -> Result<VehicleDataFE, Error> {
    // Gets data from fueleconomy.gov for the given vehicle
    //
    // Arguments:
    // ----------
    // year: Vehicle year
    // make: Vehicle make
    // model: Vehicle model
    //
    // Returns:
    // --------
    // vehicle_data_fe: Data for the given vehicle from fueleconomy.gov
    let mut handle: Easy = Easy::new();
    let url: String = format!("https://www.fueleconomy.gov/ws/rest/vehicle/menu/options?year={year}&make={make}&model={model}").replace(' ', "%20");
    handle.url(&url)?;
    let mut buf: String = String::new();
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|data| {
            buf.push_str(std::str::from_utf8(data).unwrap());
            Ok(data.len())
        })?;
        transfer.perform()?;
    }

    let vehicle_options: VehicleOptionsFE = from_str(&buf)?;
    let mut index: usize = 0;
    if vehicle_options.options.len() > 1 {
        println!(
            "Multiple engine configurations found. Please enter the index of the correct one."
        );
        for i in 0..vehicle_options.options.len() {
            println!("{i}: {}", vehicle_options.options[i].transmission);
        }
        let mut input: String = String::new();
        let _num_bytes: usize = std::io::stdin().read_line(&mut input)?;
        index = input.trim().parse()?;
    }

    handle.url(&format!(
        "https://www.fueleconomy.gov/ws/rest/vehicle/{}",
        vehicle_options.options[index].id
    ))?;
    let mut veh_buf: String = String::new();
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|data| {
            veh_buf.push_str(std::str::from_utf8(data).unwrap());
            Ok(data.len())
        })?;
        transfer.perform()?;
    }

    let mut vehicle_data_fe: VehicleDataFE = from_str(&veh_buf)?;
    if vehicle_data_fe.drive.contains("4-Wheel") {
        vehicle_data_fe.drive = String::from("All-Wheel Drive");
    }
    return Ok(vehicle_data_fe);
}

fn get_epa_data(fe_gov_vehicle_data: &VehicleDataFE) -> Result<VehicleDataEPA, Error> {
    // Gets data from EPA vehicle database for the given vehicle
    //
    // Arguments:
    // ----------
    // fe_gov_vehicle_data: Vehicle data from fueleconomy.gov
    //
    // Returns:
    // --------
    // vehicle_data_epa: Data for the given vehicle from EPA vehicle database
    let pathbuf: PathBuf =
        PathBuf::from("../../fastsim/resources/epa_vehdb/22-tstcar-2022-04-15.csv");
    let file: File = File::open(&pathbuf).unwrap();
    let _name: String = String::from(pathbuf.file_stem().unwrap().to_str().unwrap());
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut veh_list_overall: HashMap<String, Vec<VehicleDataEPA>> = HashMap::new();
    let mut veh_list_efid: HashMap<String, Vec<VehicleDataEPA>> = HashMap::new();
    let mut best_match_percent_efid: f64 = 0.0;
    let mut best_match_model_efid: String = String::new();
    let mut best_match_percent_overall: f64 = 0.0;
    let mut best_match_model_overall: String = String::new();

    let fe_model_upper: String = fe_gov_vehicle_data
        .model
        .to_uppercase()
        .replace("4WD", "AWD");
    let fe_model_words: Vec<&str> = fe_model_upper.split(' ').collect();
    let efid: &String = &fe_gov_vehicle_data.emissions_list.emissions_info[0].efid;

    for result in rdr.deserialize() {
        let veh_epa: VehicleDataEPA = result?;

        let mut match_count: i64 = 0;
        let epa_model_upper = veh_epa.model.to_uppercase().replace("4WD", "AWD");
        let epa_model_words: Vec<&str> = epa_model_upper.split(' ').collect();
        for word in &epa_model_words {
            // let match_list: Vec<&str> = model.matches(word).collect();
            // match_count += match_list.len();
            match_count += fe_model_words.contains(word) as i64;
        }
        let match_percent: f64 = (match_count as f64 * match_count as f64)
            / (epa_model_words.len() as f64 * fe_model_words.len() as f64);

        if veh_list_overall.contains_key(&veh_epa.model) {
            if let Some(x) = veh_list_overall.get_mut(&veh_epa.model) {
                (*x).push(veh_epa.clone());
            }
        } else {
            veh_list_overall.insert(veh_epa.model.clone(), vec![veh_epa.clone()]);

            // let mut match_count: i64 = 0;
            // let epa_model_upper = veh_epa.model.to_uppercase().replace("4WD", "AWD");
            // let epa_model_words: Vec<&str> = epa_model_upper.split(' ').collect();
            // for word in &epa_model_words {
            //     // let match_list: Vec<&str> = model.matches(word).collect();
            //     // match_count += match_list.len();
            //     match_count += fe_model_words.contains(word) as i64;
            // }
            // let match_percent: f64 = (match_count as f64 * match_count as f64) / (epa_model_words.len() as f64 * fe_model_words.len() as f64);
            // println!("Matches: {} {} {} {match_percent}", veh_epa.year, veh_epa.make, veh_epa.model);
            if match_percent > best_match_percent_overall {
                best_match_percent_overall = match_percent;
                best_match_model_overall = veh_epa.model.clone();
            }
            // if veh_epa.test_id.ends_with(&efid[1..efid.len()]) {
            //     veh_list_efid.insert(veh_epa.model.clone(), vec![veh_epa.clone()]);
            //     if match_percent > best_match_percent_efid { //&& (model.to_uppercase().contains(&veh_epa.model.to_uppercase()) || veh_epa.model.to_uppercase().contains(&model.to_uppercase())) {
            //     // println!("{} {} {}", veh_epa.year, veh_epa.make, veh_epa.model);
            //     // println!("{}", model.to_uppercase().ends_with(&veh_epa.model.to_uppercase()));
            //     // println!("{}", veh_epa.model.to_uppercase().ends_with(&model.to_uppercase()));
            //     // veh_list.push(veh_epa);
            //         best_match_percent_efid = match_percent;
            //         best_match_model_efid = veh_epa.model.clone();
            //     }
            // }
        }

        if veh_epa.test_id.ends_with(&efid[1..efid.len()]) {
            if veh_list_efid.contains_key(&veh_epa.model) {
                if let Some(x) = veh_list_efid.get_mut(&veh_epa.model) {
                    (*x).push(veh_epa.clone());
                }
            } else {
                veh_list_efid.insert(veh_epa.model.clone(), vec![veh_epa.clone()]);
                if match_percent > best_match_percent_efid {
                    //&& (model.to_uppercase().contains(&veh_epa.model.to_uppercase()) || veh_epa.model.to_uppercase().contains(&model.to_uppercase())) {
                    // println!("{} {} {}", veh_epa.year, veh_epa.make, veh_epa.model);
                    // println!("{}", model.to_uppercase().ends_with(&veh_epa.model.to_uppercase()));
                    // println!("{}", veh_epa.model.to_uppercase().ends_with(&model.to_uppercase()));
                    // veh_list.push(veh_epa);
                    best_match_percent_efid = match_percent;
                    best_match_model_efid = veh_epa.model.clone();
                }
            }
        }
    }
    println!("Best Match Overall: {best_match_model_overall} {best_match_percent_overall}");
    println!("Best Match efid: {best_match_model_efid} {best_match_percent_efid}");
    let veh_list: Vec<VehicleDataEPA> = if best_match_model_efid == best_match_model_overall {
        println!(
            "EPA: {} {} {} {}",
            veh_list_efid.get(&best_match_model_efid).unwrap()[0].year,
            veh_list_efid.get(&best_match_model_efid).unwrap()[0].make,
            veh_list_efid.get(&best_match_model_efid).unwrap()[0].model,
            veh_list_efid.get(&best_match_model_efid).unwrap()[0].test_id
        );
        veh_list_efid.get(&best_match_model_efid).unwrap().to_vec()
    } else {
        println!(
            "EPA: {} {} {} {}",
            veh_list_overall.get(&best_match_model_overall).unwrap()[0].year,
            veh_list_overall.get(&best_match_model_overall).unwrap()[0].make,
            veh_list_overall.get(&best_match_model_overall).unwrap()[0].model,
            veh_list_overall.get(&best_match_model_overall).unwrap()[0].test_id
        );
        veh_list_overall
            .get(&best_match_model_overall)
            .unwrap()
            .to_vec()
    };

    let num_gears_fe_gov: u32;
    let transmission_fe_gov: String;
    if fe_gov_vehicle_data.trany.contains("Manual") {
        transmission_fe_gov = String::from('M');
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("-spd").unwrap() - 1
                ..fe_gov_vehicle_data.trany.find("-spd").unwrap()]
                .parse()
                .unwrap();
    } else if fe_gov_vehicle_data.trany.contains("variable gear ratios") {
        transmission_fe_gov = String::from("CVT");
        num_gears_fe_gov = 1;
    } else if fe_gov_vehicle_data.trany.contains("AV-S") {
        transmission_fe_gov = String::from("SCV");
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("S").unwrap() + 1
                ..fe_gov_vehicle_data.trany.find(")").unwrap()]
                .parse()
                .unwrap();
    } else if fe_gov_vehicle_data.trany.contains("AM-S") {
        transmission_fe_gov = String::from("AMS");
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("S").unwrap() + 1
                ..fe_gov_vehicle_data.trany.find(")").unwrap()]
                .parse()
                .unwrap();
    } else if fe_gov_vehicle_data.trany.contains("S") {
        transmission_fe_gov = String::from("SA");
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("S").unwrap() + 1
                ..fe_gov_vehicle_data.trany.find(")").unwrap()]
                .parse()
                .unwrap();
    } else if fe_gov_vehicle_data.trany.contains("-spd") {
        transmission_fe_gov = String::from("A");
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("-spd").unwrap() - 1
                ..fe_gov_vehicle_data.trany.find("-spd").unwrap()]
                .parse()
                .unwrap();
    } else {
        transmission_fe_gov = String::from("A");
        num_gears_fe_gov =
            fe_gov_vehicle_data.trany.as_str()[fe_gov_vehicle_data.trany.find("(A").unwrap() + 2
                ..fe_gov_vehicle_data.trany.find(")").unwrap()]
                .parse()
                .unwrap();
    }

    let mut most_common_veh: VehicleDataEPA = veh_list[0].clone();
    let mut most_common_count: i32 = 0;
    let mut current_veh: VehicleDataEPA = veh_list[0].clone();
    let mut current_count: i32 = 0;
    for mut veh_epa in veh_list {
        if veh_epa.model.contains("4WD") || veh_epa.model.contains("AWD") {
            veh_epa.drive_code = String::from("A");
            veh_epa.drive = String::from("All Wheel Drive");
        }
        if fe_gov_vehicle_data.alt_veh_type == String::from("EV") {
            if (veh_epa.displ.round() == 0.0)
                && (veh_epa.cylinders == String::new())
                && (veh_epa.trany_code == transmission_fe_gov)
                && (veh_epa.gears == num_gears_fe_gov)
                && (veh_epa.drive_code == fe_gov_vehicle_data.drive[0..1])
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
        } else {
            if (veh_epa.displ.round() == (fe_gov_vehicle_data.displ.parse::<f64>().unwrap()))
                && (veh_epa.cylinders == fe_gov_vehicle_data.cylinders)
                && (veh_epa.trany_code == transmission_fe_gov)
                && (veh_epa.gears == num_gears_fe_gov)
                && (veh_epa.drive_code == fe_gov_vehicle_data.drive[0..1])
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
    }

    return Ok(most_common_veh);
}

#[allow(non_snake_case)]
#[cfg_attr(feature = "pyo3", pyfunction)]
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

    return (drag_coef, wheel_rr_coef);
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

    return err / (time_steps[time_steps.len() - 1] - time_steps[0]);
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
            .filter_map(|(index, &item)| (item < 0.1).then(|| index))
            .collect();
        let cutoff: usize = if cutoff_vec.is_empty() {
            sd_coast.mps_ach.len()
        } else {
            cutoff_vec[0]
        };

        return Ok(get_error_val(
            (Array::from_vec(vec![1000.0; sd_coast.mps_ach.len()])
                * (sd_coast.drag_kw + sd_coast.rr_kw)
                / sd_coast.mps_ach)
                .slice_move(s![0..cutoff]),
            (sd_coast.mph_ach.map(|x| dyno_func_lb.eval(*x))
                * Array::from_vec(vec![super::params::N_PER_LBF; sd_coast.mph_ach.len()]))
            .slice_move(s![0..cutoff]),
            cyc.time_s.slice_move(s![0..cutoff]),
        ));
    }
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

        assert_eq!(error_val, 0.8124999999999998);
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

        assert!((0.24676817210529464 - drag_coef).abs() < 1e-5);
        assert!((0.0068603812443132645 - wheel_rr_coef).abs() < 1e-6);
        assert_eq!(drag_coef, veh.drag_coef);
        assert_eq!(wheel_rr_coef, veh.wheel_rr_coef);
    }

    #[test]
    fn test_get_fuel_economy_gov_data() {
        let year = "2022";
        let make = "Toyota";
        let model = "Prius Prime";
        let prius_prime_fe_gov_data: VehicleDataFE =
            get_fuel_economy_gov_data(year, make, model).unwrap();
        println!(
            "FuelEconomy.gov: {} {} {}",
            prius_prime_fe_gov_data.year,
            prius_prime_fe_gov_data.make,
            prius_prime_fe_gov_data.model
        );

        let emissions_info1: EmissionsInfoFE = EmissionsInfoFE {
            efid: String::from("NTYXV01.8P35"),
            score: 7.0,
            smartway_score: 1,
            standard: String::from("L3SULEV30"),
            std_text: String::from("California LEV-III SULEV30"),
        };
        let emissions_info2: EmissionsInfoFE = EmissionsInfoFE {
            efid: String::from("NTYXV01.8P35"),
            score: 7.0,
            smartway_score: 1,
            standard: String::from("T3B30"),
            std_text: String::from("Federal Tier 3 Bin 30"),
        };
        let prius_prime_fe_truth: VehicleDataFE = VehicleDataFE {
            alt_veh_type: String::from("Plug-in Hybrid"),
            city_mpg_fuel1: 55,
            city_mpg_fuel2: 145,
            co2_g_per_mi: 78,
            comb_mpg_fuel1: 54,
            comb_mpg_fuel2: 133,
            cylinders: String::from("4"),
            displ: String::from("1.8"),
            drive: String::from("Front-Wheel Drive"),
            emissions_list: EmissionsListFE {
                emissions_info: vec![emissions_info1, emissions_info2],
            },
            eng_dscr: String::from("PHEV"),
            ev_motor_kw: String::from("22 and 53 kW AC Induction"),
            fe_score: 10,
            fuel_type: String::from("Regular Gas and Electricity"),
            fuel1: String::from("Regular Gasoline"),
            fuel2: String::from("Electricity"),
            ghg_score: 10,
            highway_mpg_fuel1: 53,
            highway_mpg_fuel2: 121,
            make: String::from("Toyota"),
            mfr_code: String::from("TYX"),
            model: String::from("Prius Prime"),
            phev_blended: true,
            phev_city_mpge: 83,
            phev_comb_mpge: 78,
            phev_hwy_mpge: 72,
            start_stop: String::from("Y"),
            trany: String::from("Automatic (variable gear ratios)"),
            veh_class: String::from("Midsize Cars"),
            year: 2022,
            super_charge: String::new(),
            turbo_charge: String::new(),
        };

        assert_eq!(prius_prime_fe_gov_data, prius_prime_fe_truth);
    }

    #[test]
    fn test_get_epa_data() {
        let emissions_info: EmissionsInfoFE = EmissionsInfoFE {
            efid: String::from("NVVXJ02.0U73"),
            score: 5.0,
            smartway_score: -1,
            standard: String::from("T3B70"),
            std_text: String::from("Federal Tier 3 Bin 70"),
        };
        let volvo_s60_b5_awd_fe_truth: VehicleDataFE = VehicleDataFE {
            alt_veh_type: String::new(),
            city_mpg_fuel1: 25,
            city_mpg_fuel2: 0,
            co2_g_per_mi: 316,
            comb_mpg_fuel1: 28,
            comb_mpg_fuel2: 0,
            cylinders: String::from("4"),
            displ: String::from("2.0"),
            drive: String::from("All-Wheel Drive"),
            emissions_list: EmissionsListFE {
                emissions_info: vec![emissions_info],
            },
            eng_dscr: String::from("SIDI"),
            ev_motor_kw: String::new(),
            fe_score: 6,
            fuel_type: String::from("Premium"),
            fuel1: String::from("Premium Gasoline"),
            fuel2: String::new(),
            ghg_score: 6,
            highway_mpg_fuel1: 33,
            highway_mpg_fuel2: 0,
            make: String::from("Volvo"),
            mfr_code: String::from("VVX"),
            model: String::from("S60 B5 AWD"),
            phev_blended: false,
            phev_city_mpge: 0,
            phev_comb_mpge: 0,
            phev_hwy_mpge: 0,
            start_stop: String::from("Y"),
            trany: String::from("Automatic (S8)"),
            veh_class: String::from("Compact Cars"),
            year: 2022,
            super_charge: String::new(),
            turbo_charge: String::from("T"),
        };

        let volvo_s60_b5_awd_epa_data = get_epa_data(&volvo_s60_b5_awd_fe_truth).unwrap();
        println!(
            "Output: {} {} {} {}",
            volvo_s60_b5_awd_epa_data.year,
            volvo_s60_b5_awd_epa_data.make,
            volvo_s60_b5_awd_epa_data.model,
            volvo_s60_b5_awd_epa_data.test_id
        )
        // match value {
        //     Ok(v) => println!("Output: {} {} {} {}", v.year, v.make, v.model, v.test_id),
        //     Err(e) => println!("error parsing header: {e:?}"),
        // };
    }
}
