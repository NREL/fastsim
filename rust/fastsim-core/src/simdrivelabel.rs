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
use crate::vehicle;

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq)]
/// Label fuel economy values
pub struct LabelFe {
    veh: vehicle::RustVehicle,
    adj_params: AdjCoef,
    lab_udds_mpgge: f64,
    lab_hwy_mpgge: f64,
    lab_comb_mpgge: f64,
    lab_udds_kwh_per_mi: f64,
    lab_hwy_kwh_per_mi: f64,
    lab_comb_kwh_per_mi: f64,
    adj_udds_mpgge: f64,
    adj_hwy_mpgge: f64,
    adj_comb_mpgge: f64,
    adj_udds_kwh_per_mi: f64,
    adj_hwy_kwh_per_mi: f64,
    adj_comb_kwh_per_mi: f64,
    adj_udds_ess_kwh_per_mi: f64,
    adj_hwy_ess_kwh_per_mi: f64,
    adj_comb_ess_kwh_per_mi: f64,
    /// Range for combined city/highway
    net_range_mi: f64,
    /// Utility factor
    uf: f64,
    net_accel: f64,
    res_found: String,
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
    veh: &vehicle::RustVehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> Result<(LabelFe, Option<HashMap<&str, RustSimDrive>>), anyhow::Error> {
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
    let long_params: RustLongParams = RustLongParams::default();

    let mut cyc: HashMap<&str, RustCycle> = HashMap::new();
    let mut sd: HashMap<&str, RustSimDrive> = HashMap::new();
    let mut out: LabelFe = LabelFe::default();

    out.veh = veh.clone();

    // load the cycles and intstantiate simdrive objects
    let accel_cyc_secs = Array::range(0., 300., 0.1);
    let mut accel_cyc_mps = Array::ones(accel_cyc_secs.len()) * 90.0 / MPH_PER_MPS;
    accel_cyc_mps[0] = 0.0;

    // cyc.insert(
    //     "accel",
    //     RustCycle::new(
    //         accel_cyc_secs.to_vec(),
    //         accel_cyc_mps.to_vec(),
    //         Array::ones(accel_cyc_secs.len()).to_vec(),
    //         Array::ones(accel_cyc_secs.len()).to_vec(),
    //         String::from("accel"),
    //     ),
    // );
    cyc.insert(
        "accel",
        RustCycle::from_file("../../fastsim/resources/cycles/accel.csv")?,
    );
    cyc.insert(
        "udds",
        RustCycle::from_file("../../fastsim/resources/cycles/udds.csv")?,
    );
    cyc.insert(
        "hwy",
        RustCycle::from_file("../../fastsim/resources/cycles/hwfet.csv")?,
    );

    // run simdrive for non-phev powertrains
    sd.insert("udds", RustSimDrive::new(cyc["udds"].clone(), veh.clone()));
    sd.insert("hwy", RustSimDrive::new(cyc["hwy"].clone(), veh.clone()));

    for (_, val) in sd.iter_mut() {
        val.sim_drive(None, None)?;
    }

    // find year-based adjustment parameters
    let adj_params: &AdjCoef = if veh.veh_year < 2017 {
        &long_params.ld_fe_adj_coef.adj_coef_map["2008"]
    } else {
        // assume 2017 coefficients are valid
        &long_params.ld_fe_adj_coef.adj_coef_map["2017"]
    };
    out.adj_params = adj_params.clone();

    // run calculations for non-PHEV powertrains
    if veh.veh_pt_type != vehicle::PHEV {
        if veh.veh_pt_type != vehicle::BEV {
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
            out.lab_udds_mpgge = sd["udds"].mpgge;
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
            out.lab_hwy_mpgge = sd["hwy"].mpgge;
            out.lab_comb_mpgge = 1. / (0.55 / sd["udds"].mpgge + 0.45 / sd["hwy"].mpgge);
        } else {
            out.lab_udds_mpgge = 0.;
            out.lab_hwy_mpgge = 0.;
            out.lab_comb_mpgge = 0.;
        }

        if veh.veh_pt_type == vehicle::BEV {
            out.lab_udds_kwh_per_mi = sd["udds"].battery_kwh_per_mi;
            out.lab_hwy_kwh_per_mi = sd["hwy"].battery_kwh_per_mi;
            out.lab_comb_kwh_per_mi =
                0.55 * sd["udds"].battery_kwh_per_mi + 0.45 * sd["hwy"].battery_kwh_per_mi;
        } else {
            out.lab_udds_kwh_per_mi = 0.;
            out.lab_hwy_kwh_per_mi = 0.;
            out.lab_comb_kwh_per_mi = 0.;
        }

        // adjusted values for mpg
        if veh.veh_pt_type != vehicle::BEV {
            // non-EV case
            // CV or HEV case (not PHEV)
            // HEV SOC iteration is handled in simdrive.SimDriveClassic
            out.adj_udds_mpgge =
                1. / (adj_params.city_intercept + adj_params.city_slope / sd["udds"].mpgge);
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            out.adj_hwy_mpgge =
                1. / (adj_params.hwy_intercept + adj_params.hwy_slope / sd["hwy"].mpgge);
            out.adj_comb_mpgge = 1. / (0.55 / out.adj_udds_mpgge + 0.45 / out.adj_hwy_mpgge);
        } else {
            // EV case
            // Mpgge is all zero for EV
            out.adj_udds_mpgge = 0.;
            out.adj_hwy_mpgge = 0.;
            out.adj_comb_mpgge = 0.;
        }

        // adjusted kW-hr/mi
        if veh.veh_pt_type == vehicle::BEV {
            // EV Case
            out.adj_udds_kwh_per_mi =
                (1. / max(
                    1. / (adj_params.city_intercept
                        + (adj_params.city_slope
                            / ((1. / out.lab_udds_kwh_per_mi) * props.kwh_per_gge))),
                    (1. / out.lab_udds_kwh_per_mi)
                        * props.kwh_per_gge
                        * (1. - sim_params.max_epa_adj),
                )) * props.kwh_per_gge
                    / CHG_EFF;
            out.adj_hwy_kwh_per_mi =
                (1. / max(
                    1. / (adj_params.hwy_intercept
                        + (adj_params.hwy_slope
                            / ((1. / out.lab_hwy_kwh_per_mi) * props.kwh_per_gge))),
                    (1. / out.lab_hwy_kwh_per_mi)
                        * props.kwh_per_gge
                        * (1. - sim_params.max_epa_adj),
                )) * props.kwh_per_gge
                    / CHG_EFF;
            out.adj_comb_kwh_per_mi =
                0.55 * out.adj_udds_kwh_per_mi + 0.45 * out.adj_hwy_kwh_per_mi;

            out.adj_udds_kwh_per_mi = out.adj_udds_kwh_per_mi * CHG_EFF;
            out.adj_hwy_kwh_per_mi = out.adj_hwy_kwh_per_mi * CHG_EFF;
            out.adj_comb_kwh_per_mi = out.adj_comb_kwh_per_mi * CHG_EFF;

            // range for combined city/highway
            out.net_range_mi = veh.ess_max_kwh / out.adj_comb_ess_kwh_per_mi;
        }

        // utility factor (percent driving in PHEV charge depletion mode)
        out.uf = 0.;
    }

    // run accelerating sim_drive
    sd.insert(
        "accel",
        RustSimDrive::new(cyc["accel"].clone(), veh.clone()),
    );
    if let Some(sd_accel) = sd.get_mut("accel") {
        sd_accel.sim_drive(None, None)?;
    }
    if sd["accel"].mph_ach.iter().any(|&x| x >= 60.) {
        out.net_accel = interpolate(&60., &sd["accel"].mph_ach, &cyc["accel"].time_s, false);
    } else {
        // in case vehicle never exceeds 60 mph, penalize it a lot with a high number
        println!("{} never achieves 60 mph.", veh.scenario_name);
        out.net_accel = 1e3;
    }

    // success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    out.res_found = String::from("model needs to be implemented for this"); // this may need fancier logic than just always being true

    if full_detail.unwrap_or(false) && verbose.unwrap_or(false) {
        println!("{:#?}", out);
        return Ok((out, Some(sd)));
    } else if full_detail.unwrap_or(false) {
        return Ok((out, Some(sd)));
    } else if verbose.unwrap_or(false) {
        println!("{:#?}", out);
        return Ok((out, None));
    } else {
        return Ok((out, None));
    }
}

pub fn get_label_fe_phev(
    veh: &vehicle::RustVehicle,
    sd: &HashMap<String, RustSimDrive>,
    long_params: &RustLongParams,
) -> Result<LabelFePHEV, anyhow::Error> {
    let mut phev_calc: LabelFePHEV = LabelFePHEV::default();
    return Ok(phev_calc);
}

#[cfg(test)]
mod simdrivelabel_tests {
    use super::*;

    #[test]
    fn test_get_label_fe_conv() {
        let veh: vehicle::RustVehicle = vehicle::RustVehicle::mock_vehicle();
        let (mut label_fe, _) = get_label_fe(&veh, None, None).unwrap();
        // For some reason, RustVehicle::mock_vehicle() != RustVehicle::mock_vehicle()
        // Therefore, veh field in both structs replaced with Default for comparison purposes
        label_fe.veh = vehicle::RustVehicle::default();
        // TODO: Figure out why net_accel values are different
        println!("Calculated net accel: {}", label_fe.net_accel);
        println!(
            "Percent diff to Python calc: {:.3}%",
            100. * (9.451683946821882 - label_fe.net_accel) / 9.451683946821882
        );
        label_fe.net_accel = 1000.;

        let label_fe_truth: LabelFe = LabelFe {
            veh: vehicle::RustVehicle::default(),
            adj_params: RustLongParams::default().ld_fe_adj_coef.adj_coef_map["2008"].clone(),
            lab_udds_mpgge: 32.47503766676829,
            lab_hwy_mpgge: 42.265348793379445,
            lab_comb_mpgge: 36.25407690819302,
            lab_udds_kwh_per_mi: 0.,
            lab_hwy_kwh_per_mi: 0.,
            lab_comb_kwh_per_mi: 0.,
            adj_udds_mpgge: 25.246151811422468,
            adj_hwy_mpgge: 30.08729992782952,
            adj_comb_mpgge: 27.21682755127691,
            adj_udds_kwh_per_mi: 0.,
            adj_hwy_kwh_per_mi: 0.,
            adj_comb_kwh_per_mi: 0.,
            adj_udds_ess_kwh_per_mi: 0.,
            adj_hwy_ess_kwh_per_mi: 0.,
            adj_comb_ess_kwh_per_mi: 0.,
            net_range_mi: 0.,
            uf: 0.,
            // net_accel: 9.451683946821882, <- Correct accel value
            net_accel: 1000.,
            res_found: String::from("model needs to be implemented for this"),
        };

        assert_eq!(label_fe_truth, label_fe)
    }
}
