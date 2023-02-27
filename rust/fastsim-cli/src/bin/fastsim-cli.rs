use clap::{ArgGroup, Parser};
use serde::{Deserialize, Serialize};

use std::fs;

extern crate fastsim_core;
use fastsim_core::{
    cycle::RustCycle, simdrive::RustSimDrive, simdrivelabel::get_label_fe, vehicle::RustVehicle,
    vehicle_utils::abc_to_drag_coeffs,
};

/// Wrapper for fastsim.
/// After running `cargo build --release`, run with
/// ```bash
/// ./target/release/fastsim-cli --veh-file ~/Documents/GitHub/fastsim/fastsim/resources/vehdb/2012_Ford_Fusion.yaml --cyc-file ~/Documents/GitHub/fastsim/fastsim/resources/cycles/udds.csv
/// ```.
/// For calculation of drag and wheel rr coefficients from coastdown test, run with
/// ```bash
/// ./target/release/fastsim-cli --veh-file ~/Documents/GitHub/fastsim/fastsim/resources/vehdb/2012_Ford_Fusion.yaml --cyc-file coastdown --a 25.91 --b 0.1943 --c 0.01796
/// ```
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(group(
    ArgGroup::new("cycle")
    .required(true)
    .args(&["cyc", "cyc-file", "adopt", "adopt-hd"])
))]
#[clap(group(
    ArgGroup::new("vehicle")
    .required(true)
    .args(&["veh", "veh-file"])
))]
#[clap(group(
    ArgGroup::new("coastdown")
    .multiple(true)
    .args(&["a", "b", "c"])
))]
// #[clap(author, version, about, long_about = None)]
// struct Args {
//     #[clap(long, short, action)]
//     it_works: bool,
// }
struct FastSimApi {
    /// Cycle as json string
    #[clap(long, value_parser)]
    cyc: Option<String>,
    #[clap(long, value_parser)]
    /// Path to cycle file (csv or yaml) or "coastdown" for coefficient calculation from coastdown test
    cyc_file: Option<String>,
    #[clap(value_parser, long)]
    //adopt flag
    adopt: Option<bool>,
    #[clap(value_parser, long)]
    //adopt HD flag
    adopt_hd: Option<bool>,
    /// Vehicle as json string
    #[clap(value_parser, long)]
    veh: Option<String>,
    #[clap(long, value_parser)]
    /// Path to vehicle file (yaml)
    veh_file: Option<String>,
    #[clap(long, value_parser)]
    /// How to return results: `adopt_json`, `mpgge`, ... TBD
    res_fmt: Option<String>,
    #[clap(long, value_parser)]
    /// coastdown coefficients for road load vs speed (lbf)
    a: Option<f64>,
    #[clap(long, value_parser)]
    /// coastdown coefficients for road load vs speed (lbf/mph)
    b: Option<f64>,
    #[clap(long, value_parser)]
    /// coastdown coefficients for road load vs speed (lbf/mph^2)
    c: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[allow(non_snake_case)]
struct AdoptResults {
    adjCombMpgge: f64,
    rangeMiles: f64,
    UF: f64,
    adjCombKwhPerMile: f64,
    accel: f64,
    // add more results here
}

impl AdoptResults {
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

pub fn main() {
    let fastsim_api = FastSimApi::parse();
    let mut veh = RustVehicle::mock_vehicle();

    if let Some(_cyc_json_str) = fastsim_api.cyc {
        panic!("Need to implement: let cyc = RustCycle::from_json(cyc_json_str)");
    }

    let cyc = if let Some(cyc_file_path) = fastsim_api.cyc_file {
        if cyc_file_path == *"coastdown" {
            if fastsim_api.a.is_some() && fastsim_api.b.is_some() && fastsim_api.c.is_some() {
                let (drag_coeff, wheel_rr_coeff) = abc_to_drag_coeffs(
                    &mut veh,
                    fastsim_api.a.unwrap(),
                    fastsim_api.b.unwrap(),
                    fastsim_api.c.unwrap(),
                    Some(false),
                    None,
                    None,
                    Some(true),
                    Some(false),
                );
                println!("Drag Coefficient: {}", drag_coeff);
                println!("Wheel RR Coefficient: {}", wheel_rr_coeff);
                return;
            } else {
                panic!("Need to provide coastdown test coefficients for drag and wheel rr coefficient calculation");
            }
        } else {
            RustCycle::from_file(&cyc_file_path)
        }
    } else {
        //TODO? use pathbuff to string, for robustness
        Ok(RustCycle::new(vec![0.0],vec![0.0],vec![0.0],vec![0.0],"".to_string()))
    }
    .unwrap();

    // let veh = RustVehicle::mock_vehicle();

    let veh: RustVehicle = if let Some(veh_string) = fastsim_api.veh {
        if fastsim_api.adopt != None {
            let veh_string = json_regex(veh_string);
            RustVehicle::from_str(&veh_string)
        } else {
            RustVehicle::from_str(&veh_string)
        }
    } else if let Some(veh_file_path) = fastsim_api.veh_file {
        if fastsim_api.adopt != None {
            let vehstring = fs::read_to_string(veh_file_path).unwrap();
            let vehstring = json_regex(vehstring);
            RustVehicle::from_str(&vehstring)
        } else {
            RustVehicle::from_file(&veh_file_path)
        }
    } else {
        Ok(RustVehicle::mock_vehicle())
    }
    .unwrap();

    #[cfg(not(windows))]
    macro_rules! main_separator {
        () => {
            "/"
        };
    }

    #[cfg(windows)]
    macro_rules! main_separator {
        () => {
            r#"\"#
        };
    }

    if fastsim_api.adopt != None {
        let sdl = get_label_fe(&veh, Some(false), Some(false)).unwrap();
        let res = AdoptResults {
            adjCombMpgge: sdl.0.adj_comb_mpgge,
            rangeMiles: sdl.0.net_range_miles,
            UF: sdl.0.uf,
            adjCombKwhPerMile: sdl.0.adj_comb_kwh_per_mi,
            accel: sdl.0.net_accel,
        };
        println!("{}", res.to_json());
    } else if fastsim_api.adopt_hd != None {
        let hd_cyc_filestring = include_str!(concat!(
            "..",
            main_separator!(),
            "..",
            main_separator!(),
            "..",
            main_separator!(),
            "..",
            main_separator!(),
            "fastsim",
            main_separator!(),
            "resources",
            main_separator!(),
            "cycles",
            main_separator!(),
            "HHDDTCruiseSmooth.csv"
        ));
        let cyc =
            RustCycle::from_csv_string(hd_cyc_filestring, "HHDDTCruiseSmooth".to_string()).unwrap();
        let mut sim_drive = RustSimDrive::new(cyc, veh);
        // // this does nothing if it has already been called for the constructed `sim_drive`
        sim_drive.sim_drive(None, None).unwrap();
        println!("{}", sim_drive.mpgge);
    } else {
        let mut sim_drive = RustSimDrive::new(cyc, veh);
        // // this does nothing if it has already been called for the constructed `sim_drive`
        sim_drive.sim_drive(None, None).unwrap();
        println!("{}", sim_drive.mpgge);
    }
    // else {
    //     println!("Invalid option `{}` for `--res-fmt`", res_fmt);
    // }
}

#[allow(non_snake_case)]
fn translateVehPtType(x: &str) -> &str {
    if x.eq("1") {
        r#""Conv""#
    } else if x.eq("2") {
        r#""HEV""#
    } else if x.eq("3") {
        r#""PHEV""#
    } else if x.eq("4") {
        r#""BEV""#
    } else {
        "other"
    }
}

#[allow(non_snake_case)]
fn translatefcEffType(x: &str) -> &str {
    if x.eq("1") {
        r#""SI""#
    } else if x.eq("2") {
        r#""ATKINSON""#
    } else if x.eq("3") {
        r#""DIESEL""#
    } else if x.eq("4") {
        r#""H2FC""#
    } else if x.eq("5") {
        r#""HD_DIESEL""#
    } else {
        "other"
    }
}

#[allow(non_snake_case)]
fn translateforceAuxOnFC(x: &str) -> bool {
    if x.eq("0") {
        false
    } else {
        true
    }
}

#[allow(non_snake_case)]
fn countCommas(x: &str) -> usize {
    x.matches(",").count() + 1
}

#[allow(non_snake_case)]
fn arrToVec(x: &str) -> String {
    format!("{{\"v\":1,\"dim\":[{}],\"data\":{}}}", countCommas(x), x)
}

fn json_regex(x: String) -> String {
    use regex::Regex;
    let adoptstring = x;

    let re = Regex::new(r#""vehPtType":(?P<a>\d)"#).unwrap();
    let adoptstring = re.replace_all(&adoptstring, |caps: &regex::Captures| {
        format!("\"vehPtType\":{}", translateVehPtType(&caps["a"]))
    });

    let re = Regex::new(r#""fcEffType":(?P<a>\d)"#).unwrap();
    let adoptstring = re.replace_all(&adoptstring, |caps: &regex::Captures| {
        format!("\"fcEffType\":{}", translatefcEffType(&caps["a"]))
    });

    let re = Regex::new(r#""forceAuxOnFC":(?P<a>\d)"#).unwrap();
    let adoptstring = re.replace_all(&adoptstring, |caps: &regex::Captures| {
        format!("\"forceAuxOnFC\":{}", translateforceAuxOnFC(&caps["a"]))
    });

    let re = Regex::new(r#""fcPwrOutPerc":(?P<a>\[.*?\])"#).unwrap();
    let arr1 = format!(
        "\"fcPwrOutPerc\":{}",
        arrToVec(&re.captures(&adoptstring).unwrap()["a"])
    );

    let re = Regex::new(r#""fcEffArray":(?P<a>\[.*?\])"#).unwrap();
    let arr2 = format!(
        "\"fcEffArray\":{}",
        &re.captures(&adoptstring).unwrap()["a"]
    );

    let re = Regex::new(r#""mcEffArray":(?P<a>\[.*?\])"#).unwrap();
    let arr3 = format!(
        "\"mcEffArray\":{}",
        arrToVec(&re.captures(&adoptstring).unwrap()["a"])
    );

    let re = Regex::new(r#""mcPwrOutPerc":(?P<a>\[.*?\])"#).unwrap();
    let arr4 = format!(
        "\"mcPwrOutPerc\":{}",
        arrToVec(&re.captures(&adoptstring).unwrap()["a"])
    );

    let re = Regex::new(r#"(?P<a>"idleFcKw":.*?)[,}]"#).unwrap();
    let cap1 = &re.captures(&adoptstring).unwrap()["a"];

    let re = Regex::new(r#"(?P<a>"mcMaxElecInKw":.*?)[,}]"#).unwrap();
    let cap2 = &re.captures(&adoptstring).unwrap()["a"];

    let s_s = "\"stop_start\": false";

    return format!(
        "{},{},{},{},{},{},{},{}}}",
        &adoptstring[0..adoptstring.len() - 1],
        cap1,
        cap2,
        arr1,
        arr2,
        arr3,
        arr4,
        s_s
    );
}
