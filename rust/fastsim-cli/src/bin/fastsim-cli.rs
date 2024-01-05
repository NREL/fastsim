use clap::{ArgGroup, Parser};
use ndarray::array;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use std::fs;

use fastsim_core::{
    cycle::RustCycle, params::MPH_PER_MPS, simdrive::RustSimDrive, simdrivelabel::get_label_fe,
    simdrivelabel::get_net_accel, simdrivelabel::make_accel_trace, traits::SerdeAPI,
    utils::interpolate_vectors as interp, vehicle::RustVehicle, vehicle_utils::abc_to_drag_coeffs,
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
    adopt_hd: Option<String>,
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

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[allow(non_snake_case)]
struct AdoptResults {
    adjCombMpgge: f64,
    rangeMiles: f64,
    UF: f64,
    adjCombKwhPerMile: f64,
    accel: f64,
    traceMissInMph: f64,
    h2AndDiesel: Option<H2AndDieselResults>,
}

impl SerdeAPI for AdoptResults {}

#[derive(Debug, Deserialize, Serialize)]
#[allow(non_snake_case)]
struct AdoptHDResults {
    adjCombMpgge: f64,
    rangeMiles: f64,
    UF: f64,
    adjCombKwhPerMile: f64,
    accel: f64,
    // add more results here
}

impl SerdeAPI for H2AndDieselResults {}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct H2AndDieselResults {
    pub h2_kwh: f64,
    pub h2_gge: f64,
    pub h2_mpgge: f64,
    pub diesel_kwh: f64,
    pub diesel_gals: f64,
    pub diesel_gge: f64,
    pub diesel_mpg: f64,
}

pub fn calculate_mpgge_for_h2_diesel_ice(
    dist_mi: f64,
    max_fc_power_kw: f64,
    kwh_per_gge: f64,
    fc_kw_out_ach: &Vec<f64>,
    fs_kwh_out_ach: &Vec<f64>,
    fc_pwr_out_perc: &Vec<f64>,
    h2share: &Vec<f64>,
) -> anyhow::Result<H2AndDieselResults> {
    anyhow::ensure!(fc_kw_out_ach.len() == fs_kwh_out_ach.len());
    anyhow::ensure!(fc_pwr_out_perc.len() == h2share.len());
    let kwh_per_gallon_diesel = 37.95;
    let gge_per_kwh = 1.0 / kwh_per_gge;
    let mut total_diesel_kwh = 0.0;
    let mut total_diesel_gals = 0.0;
    let mut total_diesel_gge = 0.0;
    let mut total_h2_kwh = 0.0;
    let mut total_h2_gge = 0.0;
    for idx in 0..fs_kwh_out_ach.len() {
        let fc_kw_out = fc_kw_out_ach[idx];
        let fs_kwh_out = fs_kwh_out_ach[idx];
        let fc_perc_pwr = fc_kw_out / max_fc_power_kw;
        let h2_perc = interp(&fc_perc_pwr, fc_pwr_out_perc, h2share, false);
        let h2_kwh = h2_perc * fs_kwh_out;
        let h2_gge = gge_per_kwh * h2_kwh;
        total_h2_kwh += h2_kwh;
        total_h2_gge += h2_gge;

        let diesel_perc = 1.0 - h2_perc;
        let diesel_kwh = diesel_perc * fs_kwh_out;
        let diesel_gals = diesel_kwh / kwh_per_gallon_diesel;
        let diesel_gge = diesel_kwh * gge_per_kwh;
        total_diesel_kwh += diesel_kwh;
        total_diesel_gals += diesel_gals;
        total_diesel_gge += diesel_gge;
    }
    Ok(H2AndDieselResults {
        h2_kwh: total_h2_kwh,
        h2_gge: total_h2_gge,
        h2_mpgge: if total_h2_gge > 0.0 {
            dist_mi / total_h2_gge
        } else {
            0.0
        },
        diesel_kwh: total_diesel_kwh,
        diesel_gals: total_diesel_gals,
        diesel_gge: total_diesel_gge,
        diesel_mpg: if total_diesel_gals > 0.0 {
            dist_mi / total_diesel_gals
        } else {
            0.0
        },
    })
}

pub fn integrate_power_to_kwh(dts_s: &Vec<f64>, ps_kw: &Vec<f64>) -> anyhow::Result<Vec<f64>> {
    anyhow::ensure!(dts_s.len() == ps_kw.len());
    let mut energy_kwh = Vec::<f64>::with_capacity(dts_s.len());
    for idx in 0..dts_s.len() {
        let dt_s = dts_s[idx];
        let p_kw = ps_kw[idx];
        energy_kwh.push(p_kw * dt_s / 3600.0);
    }
    Ok(energy_kwh)
}

pub fn main() -> anyhow::Result<()> {
    let fastsim_api = FastSimApi::parse();

    if let Some(_cyc_json_str) = fastsim_api.cyc {
        // TODO: this probably could be filled out...
        anyhow::bail!("Need to implement: let cyc = RustCycle::from_json(cyc_json_str)");
    }
    let (is_adopt_hd, adopt_hd_string, adopt_hd_has_cycle) =
        if let Some(adopt_hd_string) = &fastsim_api.adopt_hd {
            // NOTE: specifying the --adopt-hd flag implies TRUE. Thus specifying --adopt-hd false or --adopt-hd true just
            // sets the driving cycle to the default HHDDT cycle
            let adopt_hd_str_lc = adopt_hd_string.to_lowercase();
            let true_string = String::from("true");
            let false_string = String::from("false");
            let adopt_hd_has_cycle = !adopt_hd_str_lc.is_empty()
                && adopt_hd_str_lc != true_string
                && adopt_hd_str_lc != false_string;
            (true, adopt_hd_string.clone(), adopt_hd_has_cycle)
        } else {
            (false, String::default(), false)
        };
    let cyc = if let Some(cyc_file_path) = fastsim_api.cyc_file {
        if cyc_file_path == *"coastdown" {
            if fastsim_api.a.is_some() && fastsim_api.b.is_some() && fastsim_api.c.is_some() {
                let mut veh = RustVehicle::mock_vehicle();
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
                return Ok(());
            } else {
                anyhow::bail!("Need to provide coastdown test coefficients for drag and wheel rr coefficient calculation");
            }
        } else {
            RustCycle::from_file(&cyc_file_path)
        }
    } else if is_adopt_hd && adopt_hd_has_cycle {
        RustCycle::from_file(adopt_hd_string)
    } else {
        //TODO? use pathbuff to string, for robustness
        Ok(RustCycle {
            time_s: array![0.0],
            mps: array![0.0],
            grade: array![0.0],
            road_type: array![0.0],
            name: String::default(),
            orphaned: false,
        })
    }?;

    // TODO: put in logic here for loading vehicle for adopt-hd
    // with same file format as regular adopt and same outputs retured
    let is_adopt: bool = fastsim_api.adopt.is_some() && fastsim_api.adopt.unwrap();
    let mut fc_pwr_out_perc: Option<Vec<f64>> = None;
    let mut hd_h2_diesel_ice_h2share: Option<Vec<f64>> = None;
    let veh = if let Some(veh_string) = fastsim_api.veh {
        if is_adopt || is_adopt_hd {
            let (veh_string, pwr_out_perc, h2share) = json_rewrite(veh_string)?;
            hd_h2_diesel_ice_h2share = h2share;
            fc_pwr_out_perc = pwr_out_perc;
            let mut veh = RustVehicle::from_json(&veh_string)?;
            veh.set_derived()?;
            Ok(veh)
        } else {
            let mut veh = RustVehicle::from_json(&veh_string)?;
            veh.set_derived()?;
            Ok(veh)
        }
    } else if let Some(veh_file_path) = fastsim_api.veh_file {
        if is_adopt || is_adopt_hd {
            let veh_string = fs::read_to_string(veh_file_path)?;
            let (veh_string, pwr_out_perc, h2share) = json_rewrite(veh_string)?;
            hd_h2_diesel_ice_h2share = h2share;
            fc_pwr_out_perc = pwr_out_perc;
            let mut veh = RustVehicle::from_json(&veh_string)?;
            veh.set_derived()?;
            Ok(veh)
        } else {
            RustVehicle::from_file(&veh_file_path)
        }
    } else {
        Ok(RustVehicle::mock_vehicle())
    }?;

    if is_adopt {
        let sdl = get_label_fe(&veh, Some(false), Some(false))?;
        let res = AdoptResults {
            adjCombMpgge: sdl.0.adj_comb_mpgge,
            rangeMiles: sdl.0.net_range_miles,
            UF: sdl.0.uf,
            adjCombKwhPerMile: sdl.0.adj_comb_kwh_per_mi,
            accel: sdl.0.net_accel,
            traceMissInMph: sdl.0.trace_miss_speed_mph,
            h2AndDiesel: None,
        };
        println!("{}", res.to_json()?);
    } else if is_adopt_hd {
        let cyc = if adopt_hd_has_cycle {
            cyc
        } else {
            RustCycle::from_resource("cycles/HHDDTCruiseSmooth.csv")?
        };
        let mut sim_drive = RustSimDrive::new(cyc, veh.clone());
        sim_drive.sim_drive(None, None)?;
        let mut sim_drive_accel = RustSimDrive::new(make_accel_trace(), veh.clone());
        let net_accel = get_net_accel(&mut sim_drive_accel, &veh.scenario_name)?;
        let mut mpgge = sim_drive.mpgge;
        let h2_diesel_results = if let Some(hd_h2_diesel_ice_h2share) = hd_h2_diesel_ice_h2share {
            if let Some(fc_pwr_out_perc) = fc_pwr_out_perc {
                let dist_mi = sim_drive.dist_mi.sum();
                let r = calculate_mpgge_for_h2_diesel_ice(
                    dist_mi,
                    sim_drive.veh.fc_max_kw,
                    sim_drive.props.kwh_per_gge,
                    &sim_drive.fc_kw_out_ach.to_vec(),
                    &sim_drive.fs_kwh_out_ach.to_vec(),
                    &fc_pwr_out_perc,
                    &hd_h2_diesel_ice_h2share,
                )?;
                mpgge = dist_mi / (r.diesel_gge + r.h2_gge);
                Some(r)
            } else {
                None
            }
        } else {
            None
        };

        let res = AdoptResults {
            adjCombMpgge: mpgge,
            rangeMiles: if mpgge > 0.0 {
                (veh.fs_kwh / sim_drive.props.kwh_per_gge) * mpgge
            } else if sim_drive.battery_kwh_per_mi > 0.0 {
                veh.ess_max_kwh / sim_drive.battery_kwh_per_mi
            } else {
                0.0
            },
            UF: 0.0,
            adjCombKwhPerMile: sim_drive.battery_kwh_per_mi,
            accel: net_accel,
            traceMissInMph: sim_drive.trace_miss_speed_mps * MPH_PER_MPS,
            h2AndDiesel: h2_diesel_results,
        };
        println!("{}", res.to_json()?);
    } else {
        let mut sim_drive = RustSimDrive::new(cyc, veh);
        // // this does nothing if it has already been called for the constructed `sim_drive`
        sim_drive.sim_drive(None, None)?;
        println!("{}", sim_drive.mpgge);
    }
    // else {
    //     println!("Invalid option `{}` for `--res-fmt`", res_fmt);
    // }
    Ok(())
}

fn translate_veh_pt_type(x: i64) -> String {
    if x == 1 {
        String::from("Conv")
    } else if x == 2 {
        String::from("HEV")
    } else if x == 3 {
        String::from("PHEV")
    } else if x == 4 {
        String::from("BEV")
    } else {
        x.to_string()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
struct ArrayObject {
    pub v: i64,
    pub dim: Vec<usize>,
    pub data: Vec<f64>,
}

/// Takes a vector of floats and transforms it into an object representation
/// used by the ndarray library.
fn array_to_object_representation(xs: &Vec<f64>) -> ArrayObject {
    ArrayObject {
        v: 1,
        dim: vec![xs.len()],
        data: xs.clone(),
    }
}

fn transform_array_of_value_to_vec_of_f64(array_of_values: &[Value]) -> Vec<f64> {
    array_of_values.iter().fold(
        Vec::<f64>::with_capacity(array_of_values.len()),
        |mut acc, x| {
            if x.is_number() {
                acc.push(x.as_f64().unwrap());
            }
            acc
        },
    )
}

fn transform_array_of_value_to_ndarray_representation(array_of_values: &[Value]) -> ArrayObject {
    array_to_object_representation(&transform_array_of_value_to_vec_of_f64(array_of_values))
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
struct ParsedValue(Value);

impl SerdeAPI for ParsedValue {}

/// Rewrites the ADOPT JSON string to be in compliance with what FASTSim expects for JSON input.
#[allow(clippy::type_complexity)]
fn json_rewrite(x: String) -> anyhow::Result<(String, Option<Vec<f64>>, Option<Vec<f64>>)> {
    let adoptstring = x;

    let mut fc_pwr_out_perc: Option<Vec<f64>> = None;
    let mut hd_h2_diesel_ice_h2share: Option<Vec<f64>> = None;

    let mut parsed_data: Value = serde_json::from_str(&adoptstring)?;

    let veh_pt_type_raw = &parsed_data["vehPtType"];
    if veh_pt_type_raw.is_i64() {
        let veh_pt_type_value = veh_pt_type_raw.as_i64().unwrap();
        let new_veh_pt_type_value = translate_veh_pt_type(veh_pt_type_value);
        parsed_data["vehPtType"] = json!(new_veh_pt_type_value);
    }

    let fc_eff_type_raw = &parsed_data["fuelConverter"]["fcEffType"];
    if fc_eff_type_raw.is_string() {
        let fc_eff_type_value = fc_eff_type_raw.as_str().unwrap();
        let fc_eff_type = String::from(fc_eff_type_value);
        parsed_data["fcEffType"] = Value::String(fc_eff_type.clone());
        if fc_eff_type == *"HDH2DieselIce" {
            let fc_pwr_out_perc_raw = &parsed_data["fuelConverter"]["fcPwrOutPerc"];
            if fc_pwr_out_perc_raw.is_array() {
                fc_pwr_out_perc = Some(transform_array_of_value_to_vec_of_f64(
                    fc_pwr_out_perc_raw.as_array().unwrap(),
                ));
            }
            let h2share_raw = &parsed_data["fuelConverter"]["HDH2DieselIceH2Share"];
            if h2share_raw.is_array() {
                hd_h2_diesel_ice_h2share = Some(transform_array_of_value_to_vec_of_f64(
                    h2share_raw.as_array().unwrap(),
                ));
            }
        }
    }

    let force_aux_on_fc_raw = &parsed_data["forceAuxOnFC"];
    if force_aux_on_fc_raw.is_i64() {
        let force_aux_on_fc_value = force_aux_on_fc_raw.as_i64().unwrap();
        parsed_data["forceAuxOnFC"] = json!(force_aux_on_fc_value != 0)
    }

    let mut is_rear_wheel_drive: bool = false;
    let fwd1rwd2awd3_raw = &parsed_data["fwd1rwd2awd3"];
    if fwd1rwd2awd3_raw.is_i64() {
        let fwd1rwd2awd3_value = fwd1rwd2awd3_raw.as_i64().unwrap();
        is_rear_wheel_drive = fwd1rwd2awd3_value == 2 || fwd1rwd2awd3_value == 3;
    }
    let veh_cg_raw = &parsed_data["vehCgM"];
    if veh_cg_raw.is_number() {
        let veh_cg_value = veh_cg_raw.as_f64().unwrap();
        if is_rear_wheel_drive && veh_cg_value > 0.0 {
            parsed_data["vehCgM"] = json!(-1.0 * veh_cg_value);
        }
    }

    let fc_pwr_out_perc_raw = &parsed_data["fuelConverter"]["fcPwrOutPerc"];
    if fc_pwr_out_perc_raw.is_array() {
        parsed_data["fcPwrOutPerc"] = json!(transform_array_of_value_to_ndarray_representation(
            fc_pwr_out_perc_raw.as_array().unwrap()
        ));
    }

    let fc_eff_array_raw = &parsed_data["fuelConverter"]["fcEffArray"];
    if fc_eff_array_raw.is_array() {
        parsed_data["fcEffArray"] = json!(transform_array_of_value_to_vec_of_f64(
            fc_eff_array_raw.as_array().unwrap()
        ));
    }

    let mc_eff_array_raw = &parsed_data["motor"]["mcEffArray"];
    if mc_eff_array_raw.is_array() {
        parsed_data["mcEffArray"] = json!(transform_array_of_value_to_ndarray_representation(
            mc_eff_array_raw.as_array().unwrap()
        ));
    }

    let mc_pwr_out_perc_raw = &parsed_data["motor"]["mcPwrOutPerc"];
    if mc_pwr_out_perc_raw.is_array() {
        parsed_data["mcPwrOutPerc"] = json!(transform_array_of_value_to_ndarray_representation(
            mc_pwr_out_perc_raw.as_array().unwrap()
        ));
    }

    let idle_fc_kw_raw = &parsed_data["fuelConverter"]["idleFcKw"];
    if idle_fc_kw_raw.is_number() {
        let idle_fc_kw_value = idle_fc_kw_raw.as_f64().unwrap();
        parsed_data["idleFcKw"] = json!(idle_fc_kw_value);
    }

    let mc_max_elec_in_kw_raw = &parsed_data["motor"]["mcMaxElecInKw"];
    if mc_max_elec_in_kw_raw.is_number() {
        let mc_max_elec_in_kw_value = mc_max_elec_in_kw_raw.as_f64().unwrap();
        parsed_data["mcMaxElecInKw"] = json!(mc_max_elec_in_kw_value);
    }

    parsed_data["stop_start"] = json!(false);

    let adoptstring = ParsedValue(parsed_data).to_json()?;

    Ok((adoptstring, fc_pwr_out_perc, hd_h2_diesel_ice_h2share))
}
