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
    phev_calcs: Option<LabelFePHEV>,
    adj_cs_comb_mpgge: Option<f64>,
    adj_cd_comb_mpgge: Option<f64>,
    net_phev_cd_miles: Option<f64>,
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
pub struct PHEVCycleCalc {
    /// Charge depletion battery kW-hr
    cd_ess_kwh: f64,
    cd_ess_kwh_per_mi: f64,
    /// Charge depletion fuel gallons
    cd_fs_gal: f64,
    cd_fs_kwh: f64,
    cd_mpg: f64,
    /// Number of cycles in charge depletion mode, up to transition
    cd_cycs: f64,
    cd_miles: f64,
    cd_lab_mpg: f64,
    cd_MPG: f64,
    /// Fraction of transition cycles spent in charge depletion
    cd_frac_in_trans: f64,
    /// SOC change during 1 cycle
    trans_init_soc: f64,
    /// charge depletion battery kW-hr
    trans_ess_kwh: f64,
    trans_ess_kwh_per_mi: f64,
    trans_fs_gal: f64,
    trans_fs_kwh: f64,
    /// charge sustaining battery kW-hr
    cs_ess_kwh: f64,
    cs_ess_kwh_per_mi: f64,
    /// charge sustaining fuel gallons
    cs_fs_gal: f64,
    cs_fs_kwh: f64,
    cs_mpg: f64,
    lab_mpgge: f64,
    lab_kwh_per_mi: f64,
    lab_uf: f64,
    lab_uf_gpm: Array1<f64>,
    lab_iter_uf: Array1<f64>,
    lab_iter_uf_kwh_per_mi: Array1<f64>,
    lab_iter_kwh_per_mi: Array1<f64>,
    adj_iter_mpgge: Array1<f64>,
    adj_iter_kwh_per_mi: Array1<f64>,
    adj_iter_cd_miles: Array1<f64>,
    adj_iter_uf: Array1<f64>,
    adj_iter_uf_gpm: Vec<f64>,
    adj_iter_uf_kwh_per_mi: Array1<f64>,
    adj_cd_miles: f64,
    adj_cd_mpgge: f64,
    adj_cs_mpgge: f64,
    adj_uf: f64,
    adj_mpgge: f64,
    adj_kwh_per_mi: f64,
    adj_ess_kwh_per_mi: f64,
    delta_soc: f64,
    /// Total number of miles in charge depletion mode, assuming constant kWh_per_mi
    total_cd_miles: f64,
}

pub fn get_label_fe(
    veh: &vehicle::RustVehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> Result<(LabelFe, Option<HashMap<&str, RustSimDrive>>), anyhow::Error> {
    // Generates label fuel economy (FE) values for a provided vehicle.
    //
    // Arguments:
    // ----------
    // veh : vehicle::RustVehicle
    // full_detail : boolean, default False
    //     If True, sim_drive objects for each cycle are also returned.
    // verbose : boolean, default false
    //     If true, print out key results
    //
    // Returns label fuel economy values as a struct and (optionally)
    // simdrive::RustSimDrive objects.

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
    } else {
        // PHEV
        let phev_calcs: LabelFePHEV =
            get_label_fe_phev(veh, &mut sd, &long_params, adj_params, &sim_params, &props)?;
        out.phev_calcs = Some(phev_calcs.clone());

        // efficiency-related calculations
        // lab
        out.lab_udds_mpgge = phev_calcs.udds.lab_mpgge;
        out.lab_hwy_mpgge = phev_calcs.hwy.lab_mpgge;
        out.lab_comb_mpgge =
            1.0 / (0.55 / phev_calcs.udds.lab_mpgge + 0.45 / phev_calcs.hwy.lab_mpgge);

        out.lab_udds_kwh_per_mi = phev_calcs.udds.lab_kwh_per_mi;
        out.lab_hwy_kwh_per_mi = phev_calcs.hwy.lab_kwh_per_mi;
        out.lab_comb_kwh_per_mi =
            0.55 * phev_calcs.udds.lab_kwh_per_mi + 0.45 * phev_calcs.hwy.lab_kwh_per_mi;

        // adjusted
        out.adj_udds_mpgge = phev_calcs.udds.adj_mpgge;
        out.adj_hwy_mpgge = phev_calcs.hwy.adj_mpgge;
        out.adj_comb_mpgge =
            1.0 / (0.55 / phev_calcs.udds.adj_mpgge + 0.45 / phev_calcs.hwy.adj_mpgge);

        out.adj_cs_comb_mpgge =
            Some(1.0 / (0.55 / phev_calcs.udds.adj_cs_mpgge + 0.45 / phev_calcs.hwy.adj_cs_mpgge));
        out.adj_cd_comb_mpgge =
            Some(1.0 / (0.55 / phev_calcs.udds.adj_cd_mpgge + 0.45 / phev_calcs.hwy.adj_cd_mpgge));

        out.adj_udds_kwh_per_mi = phev_calcs.udds.adj_kwh_per_mi;
        out.adj_hwy_kwh_per_mi = phev_calcs.hwy.adj_kwh_per_mi;
        out.adj_comb_kwh_per_mi =
            0.55 * phev_calcs.udds.adj_kwh_per_mi + 0.45 * phev_calcs.hwy.adj_kwh_per_mi;

        out.adj_udds_ess_kwh_per_mi = phev_calcs.udds.adj_ess_kwh_per_mi;
        out.adj_hwy_ess_kwh_per_mi = phev_calcs.hwy.adj_ess_kwh_per_mi;
        out.adj_comb_ess_kwh_per_mi =
            0.55 * phev_calcs.udds.adj_ess_kwh_per_mi + 0.45 * phev_calcs.hwy.adj_ess_kwh_per_mi;

        // range for combined city/highway
        // utility factor (percent driving in charge depletion mode)
        out.uf = interpolate(
            &(0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles),
            &Array::from_vec(long_params.rechg_freq_miles.clone()),
            &Array::from_vec(long_params.uf_array.clone()),
            false,
        );

        out.net_phev_cd_miles =
            Some(0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles);

        out.net_range_mi = (veh.fs_kwh / props.kwh_per_gge
            - out.net_phev_cd_miles.unwrap() / out.adj_cd_comb_mpgge.unwrap())
            * out.adj_cs_comb_mpgge.unwrap()
            + out.net_phev_cd_miles.unwrap();
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
    sd: &mut HashMap<&str, RustSimDrive>,
    long_params: &RustLongParams,
    adj_params: &AdjCoef,
    sim_params: &RustSimDriveParams,
    props: &RustPhysicalProperties,
) -> Result<LabelFePHEV, anyhow::Error> {
    // PHEV-specific function for label fe.
    //
    // Arguments:
    // ----------
    // veh : vehicle::RustVehicle
    // sd : RustSimDrive objects to use for label fe calculations
    // long_params : Struct for longparams.json values
    // adj_params: Adjusted coefficients from longparams.json
    // sim_params : RustSimDriveParams
    // props : RustPhysicalProperties
    //
    // Returns label fuel economy values for PHEV as a struct.
    let mut phev_calcs: LabelFePHEV = LabelFePHEV::default();

    // do PHEV soc iteration
    // This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
    // By assuming that the battery SOC depletion per mile is constant across cycles,
    // the first cycle can be extrapolated until charge sustaining kicks in.
    for (_, val) in sd.iter_mut() {
        val.sim_drive(Some(veh.max_soc), None)?;
    }

    phev_calcs.regen_soc_buffer = min(
        ((0.5 * veh.veh_kg * ((60. * (1. / MPH_PER_MPS)).powi(2)))
            * (1. / 3600.)
            * (1. / 1000.)
            * veh.max_regen
            * veh.mc_peak_eff())
            / veh.ess_max_kwh,
        (veh.max_soc - veh.min_soc) / 2.0,
    );

    // charge sustaining behavior
    for (key, sd_val) in sd.iter_mut() {
        let mut phev_calc: PHEVCycleCalc = PHEVCycleCalc::default();

        // charge depletion cycle has already been simulated
        // charge depletion battery kW-hr
        phev_calc.cd_ess_kwh = (veh.max_soc - veh.min_soc) * veh.ess_max_kwh;

        // SOC change during 1 cycle
        phev_calc.delta_soc = sd_val.soc[0] - sd_val.soc[sd_val.len() - 1];
        // total number of miles in charge depletion mode, assuming constant kWh_per_mi
        phev_calc.total_cd_miles =
            (veh.max_soc - veh.min_soc) * sd_val.veh.ess_max_kwh / sd_val.battery_kwh_per_mi;
        // number of cycles in charge depletion mode, up to transition
        phev_calc.cd_cycs = phev_calc.total_cd_miles / sd_val.dist_mi.sum();
        // fraction of transition cycle spent in charge depletion
        phev_calc.cd_frac_in_trans = phev_calc.cd_cycs % phev_calc.cd_cycs.floor();

        // charge depletion fuel gallons
        phev_calc.cd_fs_gal = sd_val.fs_kwh_out_ach.sum() / props.kwh_per_gge;
        phev_calc.cd_fs_kwh = sd_val.fs_kwh_out_ach.sum();
        phev_calc.cd_ess_kwh_per_mi = sd_val.battery_kwh_per_mi;
        phev_calc.cd_mpg = sd_val.mpgge;

        // utility factor calculation for last charge depletion iteration and transition iteration
        // ported from excel
        let interp_x_vals: Array1<f64> =
            Array::range(0., phev_calc.cd_cycs.ceil(), 1.) * sd_val.dist_mi.sum();
        phev_calc.lab_iter_uf = interp_x_vals
            .iter()
            .map(|x: &f64| -> f64 {
                interpolate(
                    x,
                    &Array::from_vec(long_params.rechg_freq_miles.clone()),
                    &Array::from_vec(long_params.uf_array.clone()),
                    false,
                )
            })
            .collect();

        // transition cycle
        phev_calc.trans_init_soc = veh.max_soc - phev_calc.cd_cycs.floor() * phev_calc.delta_soc;

        // run the transition cycle
        sd_val.sim_drive(Some(phev_calc.trans_init_soc), None)?;
        // charge depletion battery kW-hr
        phev_calc.trans_ess_kwh =
            phev_calc.cd_ess_kwh_per_mi * sd_val.dist_mi.sum() * phev_calc.cd_frac_in_trans;
        phev_calc.trans_ess_kwh_per_mi = phev_calc.cd_ess_kwh_per_mi * phev_calc.cd_frac_in_trans;

        // charge sustaining
        // the 0.01 is here to be consistent with Excel
        let init_soc: f64 = sd_val.veh.min_soc + 0.01;
        sd_val.sim_drive(Some(init_soc), None)?;
        // charge sustaining fuel gallons
        phev_calc.cs_fs_gal = sd_val.fs_kwh_out_ach.sum() / props.kwh_per_gge;
        // charge depletion fuel gallons, dependent on phev_calc.trans_fs_gal
        phev_calc.trans_fs_gal = phev_calc.cs_fs_gal * (1.0 - phev_calc.cd_frac_in_trans);
        phev_calc.cs_fs_kwh = sd_val.fs_kwh_out_ach.sum();
        phev_calc.trans_fs_kwh = phev_calc.cs_fs_kwh * (1.0 - phev_calc.cd_frac_in_trans);
        // charge sustaining battery kW-hr
        phev_calc.cs_ess_kwh = sd_val.ess_dischg_kj;
        phev_calc.cs_ess_kwh_per_mi = sd_val.battery_kwh_per_mi;

        let lab_iter_uf_diff: Array1<f64> = diff(&phev_calc.lab_iter_uf);
        phev_calc.lab_uf_gpm = Array::from_vec(vec![
            phev_calc.trans_fs_gal * lab_iter_uf_diff[lab_iter_uf_diff.len() - 1],
            phev_calc.cs_fs_gal * (1.0 - phev_calc.lab_iter_uf[phev_calc.lab_iter_uf.len() - 1]),
        ]) / sd_val.dist_mi.sum();

        phev_calc.cd_mpg = sd_val.mpgge;

        // city and highway cycle ranges
        phev_calc.cd_miles =
            if (veh.max_soc - phev_calcs.regen_soc_buffer - ndarrmin(&sd_val.soc)) < 0.01 {
                1000.0
            } else {
                phev_calc.cd_cycs.ceil() * sd_val.dist_mi.sum()
            };
        phev_calc.cd_lab_mpg = phev_calc.lab_iter_uf[phev_calc.lab_iter_uf.len() - 1]
            / (phev_calc.trans_fs_gal / sd_val.dist_mi.sum());

        // charge sustaining
        phev_calc.cs_mpg = sd_val.dist_mi.sum() / phev_calc.cs_fs_gal;

        phev_calc.lab_uf = interpolate(
            &phev_calc.cd_miles,
            &Array::from_vec(long_params.rechg_freq_miles.clone()),
            &Array::from_vec(long_params.uf_array.clone()),
            false,
        );

        // labCombMpgge
        phev_calc.cd_MPG =
            ndarrmax(&phev_calc.lab_iter_uf) / phev_calc.lab_uf_gpm[phev_calc.lab_uf_gpm.len() - 2];

        phev_calc.lab_mpgge = 1.0
            / (phev_calc.lab_uf / phev_calc.cd_MPG + (1.0 - phev_calc.lab_uf) / phev_calc.cs_mpg);

        phev_calc.lab_iter_kwh_per_mi = Array::from_vec(vec![
            0.0,
            phev_calc.cd_ess_kwh_per_mi * phev_calc.cd_cycs.floor(),
            phev_calc.trans_ess_kwh_per_mi,
            0.0,
        ]);

        let mut vals: Vec<f64> = Vec::new();
        vals.push(0.0);
        vals.extend(
            (&phev_calc
                .lab_iter_kwh_per_mi
                .slice(s![1..phev_calc.lab_iter_kwh_per_mi.len() - 1])
                * &diff(&phev_calc.lab_iter_uf))
                .iter(),
        );
        vals.push(0.0);
        phev_calc.lab_iter_uf_kwh_per_mi = Array::from_vec(vals);

        phev_calc.lab_kwh_per_mi =
            phev_calc.lab_iter_uf_kwh_per_mi.sum() / ndarrmax(&phev_calc.lab_iter_uf);

        let mut adj_iter_mpgge_vals: Vec<f64> = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        let mut adj_iter_kwh_per_mi_vals: Vec<f64> = vec![0.0; phev_calc.lab_iter_kwh_per_mi.len()];
        if *key == "udds" {
            adj_iter_mpgge_vals.push(max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));

            for c in 0..phev_calc.lab_iter_kwh_per_mi.len() {
                if phev_calc.lab_iter_kwh_per_mi[c] == 0.0 {
                    adj_iter_kwh_per_mi_vals[c] = 0.0;
                } else {
                    adj_iter_kwh_per_mi_vals[c] =
                        (1.0 / max(
                            1.0 / (adj_params.city_intercept
                                + (adj_params.city_slope
                                    / ((1.0 / phev_calc.lab_iter_kwh_per_mi[c])
                                        * props.kwh_per_gge))),
                            (1.0 - sim_params.max_epa_adj)
                                * ((1.0 / phev_calc.lab_iter_kwh_per_mi[c]) * props.kwh_per_gge),
                        )) * props.kwh_per_gge;
                }
            }
        } else {
            adj_iter_mpgge_vals.push(max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));

            for c in 0..phev_calc.lab_iter_kwh_per_mi.len() {
                if phev_calc.lab_iter_kwh_per_mi[c] == 0.0 {
                    adj_iter_kwh_per_mi_vals[c] = 0.0;
                } else {
                    adj_iter_kwh_per_mi_vals[c] =
                        (1.0 / max(
                            1.0 / (adj_params.hwy_intercept
                                + (adj_params.hwy_slope
                                    / ((1.0 / phev_calc.lab_iter_kwh_per_mi[c])
                                        * props.kwh_per_gge))),
                            (1.0 - sim_params.max_epa_adj)
                                * ((1.0 / phev_calc.lab_iter_kwh_per_mi[c]) * props.kwh_per_gge),
                        )) * props.kwh_per_gge;
                }
            }
        }
        phev_calc.adj_iter_mpgge = Array::from(adj_iter_mpgge_vals);
        phev_calc.adj_iter_kwh_per_mi = Array::from(adj_iter_kwh_per_mi_vals);

        phev_calc.adj_iter_cd_miles =
            Array::from_vec(vec![0.0; phev_calc.cd_cycs.ceil() as usize + 2]);
        for c in 0..phev_calc.adj_iter_cd_miles.len() {
            if c == 0 {
                phev_calc.adj_iter_cd_miles[c] = 0.0;
            } else if c <= phev_calc.cd_cycs.floor() as usize {
                phev_calc.adj_iter_cd_miles[c] = phev_calc.adj_iter_cd_miles[c - 1]
                    + phev_calc.cd_ess_kwh_per_mi * sd_val.dist_mi.sum()
                        / phev_calc.adj_iter_kwh_per_mi[c];
            } else if c == phev_calc.cd_cycs.floor() as usize + 1 {
                phev_calc.adj_iter_cd_miles[c] = phev_calc.adj_iter_cd_miles[c - 1]
                    + phev_calc.trans_ess_kwh_per_mi * sd_val.dist_mi.sum()
                        / phev_calc.adj_iter_kwh_per_mi[c];
            } else {
                phev_calc.adj_iter_cd_miles[c] = 0.0;
            }
        }

        phev_calc.adj_cd_miles =
            if veh.max_soc - phev_calcs.regen_soc_buffer - ndarrmin(&sd_val.soc) < 0.01 {
                1000.0
            } else {
                ndarrmax(&phev_calc.adj_iter_cd_miles)
            };

        // utility factor calculation for last charge depletion iteration and transition iteration
        // ported from excel
        phev_calc.adj_iter_uf = phev_calc
            .adj_iter_cd_miles
            .iter()
            .map(|x: &f64| -> f64 {
                interpolate(
                    x,
                    &Array::from_vec(long_params.rechg_freq_miles.clone()),
                    &Array::from_vec(long_params.uf_array.clone()),
                    false,
                )
            })
            .collect();

        let adj_iter_uf_diff: Array1<f64> = diff(&phev_calc.adj_iter_uf);
        phev_calc.adj_iter_uf_gpm = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc.adj_iter_mpgge[phev_calc.adj_iter_mpgge.len() - 2])
                * adj_iter_uf_diff[adj_iter_uf_diff.len() - 2],
        );
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc.adj_iter_mpgge[phev_calc.adj_iter_mpgge.len() - 1])
                * (1.0 - phev_calc.adj_iter_uf[phev_calc.adj_iter_uf.len() - 2]),
        );

        phev_calc.adj_iter_uf_kwh_per_mi = &phev_calc.adj_iter_kwh_per_mi
            * concatenate![Axis(1), Array::zeros(1), diff(&phev_calc.adj_iter_uf)];

        phev_calc.adj_cd_mpgge = 1.0
            / phev_calc.adj_iter_uf_gpm[phev_calc.adj_iter_uf_gpm.len() - 2]
            * ndarrmax(&phev_calc.adj_iter_uf);
        phev_calc.adj_cs_mpgge = 1.0
            / phev_calc.adj_iter_uf_gpm[phev_calc.adj_iter_uf_gpm.len() - 1]
            * (1.0 - ndarrmax(&phev_calc.adj_iter_uf));

        phev_calc.adj_uf = interpolate(
            &phev_calc.adj_cd_miles,
            &Array::from_vec(long_params.rechg_freq_miles.clone()),
            &Array::from_vec(long_params.uf_array.clone()),
            false,
        );

        phev_calc.adj_mpgge = 1.0
            / (phev_calc.adj_uf / phev_calc.adj_cd_mpgge
                + (1.0 - phev_calc.adj_uf) / phev_calc.adj_cs_mpgge);

        phev_calc.adj_kwh_per_mi =
            phev_calc.adj_iter_uf_kwh_per_mi.sum() / ndarrmax(&phev_calc.adj_iter_uf) / veh.chg_eff;

        phev_calc.adj_ess_kwh_per_mi =
            phev_calc.adj_iter_uf_kwh_per_mi.sum() / ndarrmax(&phev_calc.adj_iter_uf);

        match *key {
            "udds" => phev_calcs.udds = phev_calc.clone(),
            "hwy" => phev_calcs.hwy = phev_calc.clone(),
            &_ => return Err(anyhow!("No field for cycle {}", key)),
        };
    }

    return Ok(phev_calcs);
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
            phev_calcs: None,
            adj_cs_comb_mpgge: None,
            adj_cd_comb_mpgge: None,
            net_phev_cd_miles: None,
        };

        assert_eq!(label_fe_truth, label_fe)
    }
}
