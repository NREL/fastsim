extern crate ndarray;
use clap::{IntoApp, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::process::Command as ProcCommand;

#[macro_use]
pub mod macros;
extern crate proc_macros;

pub mod cycle;
pub mod params;
pub mod utils;
use cycle::RustCycle;
pub mod vehicle;
use vehicle::RustVehicle;
pub mod simdrive;
use simdrive::RustSimDrive;
pub mod simdrive_impl;

/// Wrapper for fastsim.
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct FastSimApi {
    // `conflicts_with` tells rust one or the other, not both, can be provided
    /// Cycle as json string
    #[clap(short, long, value_parser, conflicts_with = "cyc_file")]
    cyc: Option<String>,
    #[clap(short, long, value_parser)]
    /// Path to cycle file
    cyc_file: Option<String>,
    #[clap(value_parser, short, long)]
    /// vehicle as json string
    veh: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct SimDriveResults {
    mpgge: f64,
}

impl SimDriveResults {
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

pub fn main() {
    let fastsim_api = FastSimApi::parse();

    if let Some(_cyc_json_str) = fastsim_api.cyc {
        panic!("Need to implement: let cyc = RustCycle::from_json(cyc_json_str)");
    }

    let cyc = if let Some(cyc_file_path) = fastsim_api.cyc_file {
        RustCycle::from_file(&cyc_file_path)
    } else {
        RustCycle::from_file("../fastsim/resources/cycles/udds.csv")
    };

    let veh = RustVehicle::mock_vehicle();
    let mut sim_drive = RustSimDrive::new(cyc, veh);
    sim_drive.sim_drive(None, None).unwrap();
    let res = SimDriveResults {
        mpgge: sim_drive.mpgge,
    };
    println!("{}", res.to_json());
}
