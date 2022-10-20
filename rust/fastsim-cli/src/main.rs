use clap::{ArgGroup, Parser};
use serde::{Deserialize, Serialize};

extern crate fastsim_core;
use fastsim_core::{
    cycle::RustCycle, simdrive::RustSimDrive, vehicle::RustVehicle,
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
    .args(&["cyc", "cyc-file"])
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
struct FastSimApi {
    /// Cycle as json string
    #[clap(long, value_parser)]
    cyc: Option<String>,
    #[clap(long, value_parser)]
    /// Path to cycle file (csv or yaml) or "coastdown" for coefficient calculation from coastdown test
    cyc_file: Option<String>,
    #[clap(value_parser, long)]
    /// Vehicle as json string
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
struct AdoptResults {
    mpgge: f64,
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
        RustCycle::from_file("../fastsim/resources/cycles/udds.csv")
    }
    .unwrap();

    let mut sim_drive = RustSimDrive::new(cyc, veh);
    // this does nothing if it has already been called for the constructed `sim_drive`
    sim_drive.sim_drive(None, None).unwrap();

    let res_fmt = fastsim_api.res_fmt.unwrap_or_else(|| String::from("mpgge"));

    if res_fmt == "adopt_json" {
        let res = AdoptResults {
            mpgge: sim_drive.mpgge,
        };
        println!("{}", res.to_json());
    } else if res_fmt == "mpgge" {
        println!("{}", sim_drive.mpgge);
    } else {
        println!("Invalid option `{}` for `--res-fmt`", res_fmt);
    }
}
