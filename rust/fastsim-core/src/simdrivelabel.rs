//! Module containing classes and methods for calculating label fuel economy.
// For example usage, see ../README.md

use ndarray::Array;
use std::collections::HashMap;

// crate local
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;
use crate::simdrive::{RustSimDrive, RustSimDriveParams};
use crate::utils::*;
use crate::vehicle::*;

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq)]
/// Label fuel economy values
pub struct LabelFe {
    veh: RustVehicle,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq)]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    regen_soc_buffer: f64,
    udds: PHEVCycleCalc,
    hwy: PHEVCycleCalc,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq)]
/// Label fuel economy calculations for a specific cycle of a PHEV vehicle
pub struct PHEVCycleCalc {}

pub fn get_label_fe(
    veh: &RustVehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> Result<(LabelFe, Option<HashMap<String, RustSimDrive>>), anyhow::Error> {
    // Generates label fuel economy (FE) values for a provided vehicle.
    //
    // Arguments:
    // ----------
    // veh : vehicle.Vehicle()
    // full_detail : boolean, default False
    //     If True, sim_drive objects for each cycle are also returned.
    // verbose : boolean, default false
    //     If true, print out key results
    //
    // Returns label fuel economy values as a dict and (optionally)
    // simdrive.SimDriveClassic objects.

    let sim_params: RustSimDriveParams = RustSimDriveParams::default();
    let props: RustPhysicalProperties = RustPhysicalProperties::default();

    let mut cyc: HashMap<&str, RustCycle> = HashMap::new();
    let mut sd: HashMap<&str, RustSimDrive> = HashMap::new();
    let mut out: LabelFe = LabelFe::default();

    out.veh = veh.clone();

    // load the cycles and intstantiate simdrive objects
    let accel_cyc_secs = Array::range(0., 300., 0.1);
    let mut accel_cyc_mps = Array::ones(accel_cyc_secs.len()) * 90.0 / MPH_PER_MPS;
    accel_cyc_mps[0] = 0.0;

    cyc.insert(
        "accel",
        RustCycle::new(
            accel_cyc_secs.to_vec(),
            accel_cyc_mps.to_vec(),
            Array::ones(accel_cyc_secs.len()).to_vec(),
            Array::ones(accel_cyc_secs.len()).to_vec(),
            String::from("accel"),
        ),
    );
    cyc.insert("udds", RustCycle::from_file("udds")?);
    cyc.insert("hwy", RustCycle::from_file("hwfet")?);

    // run simdrive for non-phev powertrains
    sd.insert("udds", RustSimDrive::new(cyc["udds"].clone(), veh.clone()));
    sd.insert("hwy", RustSimDrive::new(cyc["hwy"].clone(), veh.clone()));

    for (_, val) in sd.iter_mut() {
        val.sim_drive(None, None);
    }

    return Ok((out, None));
}

pub fn get_label_fe_phev(
    veh: &RustVehicle,
    sd: &HashMap<String, RustSimDrive>,
) -> Result<LabelFePHEV, anyhow::Error> {
    let mut phev_calc: LabelFePHEV = LabelFePHEV::default();
    return Ok(phev_calc);
}
