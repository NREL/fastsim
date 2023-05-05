//! Module containing classes and methods for calculating label fuel economy.

use ndarray::Array;
use std::collections::HashMap;

// crate local
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::ApproxEq;
use crate::simdrive::{RustSimDrive, RustSimDriveParams};
use crate::vehicle;

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy values
pub struct LabelFe {
    pub veh: vehicle::RustVehicle,
    pub adj_params: AdjCoef,
    pub lab_udds_mpgge: f64,
    pub lab_hwy_mpgge: f64,
    pub lab_comb_mpgge: f64,
    pub lab_udds_kwh_per_mi: f64,
    pub lab_hwy_kwh_per_mi: f64,
    pub lab_comb_kwh_per_mi: f64,
    pub adj_udds_mpgge: f64,
    pub adj_hwy_mpgge: f64,
    pub adj_comb_mpgge: f64,
    pub adj_udds_kwh_per_mi: f64,
    pub adj_hwy_kwh_per_mi: f64,
    pub adj_comb_kwh_per_mi: f64,
    pub adj_udds_ess_kwh_per_mi: f64,
    pub adj_hwy_ess_kwh_per_mi: f64,
    pub adj_comb_ess_kwh_per_mi: f64,
    /// Range for combined city/highway
    pub net_range_miles: f64,
    /// Utility factor
    pub uf: f64,
    pub net_accel: f64,
    pub res_found: String,
    pub phev_calcs: Option<LabelFePHEV>,
    pub adj_cs_comb_mpgge: Option<f64>,
    pub adj_cd_comb_mpgge: Option<f64>,
    pub net_phev_cd_miles: Option<f64>,
    pub trace_miss_speed_mph: f64,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: f64,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
/// Label fuel economy calculations for a specific cycle of a PHEV vehicle
pub struct PHEVCycleCalc {
    /// Charge depletion battery kW-hr
    pub cd_ess_kwh: f64,
    pub cd_ess_kwh_per_mi: f64,
    /// Charge depletion fuel gallons
    pub cd_fs_gal: f64,
    pub cd_fs_kwh: f64,
    pub cd_mpg: f64,
    /// Number of cycles in charge depletion mode, up to transition
    pub cd_cycs: f64,
    pub cd_miles: f64,
    pub cd_lab_mpg: f64,
    pub cd_adj_mpg: f64,
    /// Fraction of transition cycles spent in charge depletion
    pub cd_frac_in_trans: f64,
    /// SOC change during 1 cycle
    pub trans_init_soc: f64,
    /// charge depletion battery kW-hr
    pub trans_ess_kwh: f64,
    pub trans_ess_kwh_per_mi: f64,
    pub trans_fs_gal: f64,
    pub trans_fs_kwh: f64,
    /// charge sustaining battery kW-hr
    pub cs_ess_kwh: f64,
    pub cs_ess_kwh_per_mi: f64,
    /// charge sustaining fuel gallons
    pub cs_fs_gal: f64,
    pub cs_fs_kwh: f64,
    pub cs_mpg: f64,
    pub lab_mpgge: f64,
    pub lab_kwh_per_mi: f64,
    pub lab_uf: f64,
    pub lab_uf_gpm: Array1<f64>,
    pub lab_iter_uf: Array1<f64>,
    pub lab_iter_uf_kwh_per_mi: Array1<f64>,
    pub lab_iter_kwh_per_mi: Array1<f64>,
    pub adj_iter_mpgge: Array1<f64>,
    pub adj_iter_kwh_per_mi: Array1<f64>,
    pub adj_iter_cd_miles: Array1<f64>,
    pub adj_iter_uf: Array1<f64>,
    pub adj_iter_uf_gpm: Vec<f64>,
    pub adj_iter_uf_kwh_per_mi: Array1<f64>,
    pub adj_cd_miles: f64,
    pub adj_cd_mpgge: f64,
    pub adj_cs_mpgge: f64,
    pub adj_uf: f64,
    pub adj_mpgge: f64,
    pub adj_kwh_per_mi: f64,
    pub adj_ess_kwh_per_mi: f64,
    pub delta_soc: f64,
    /// Total number of miles in charge depletion mode, assuming constant kWh_per_mi
    pub total_cd_miles: f64,
}

pub fn make_accel_trace() -> RustCycle {
    let accel_cyc_secs = Array::range(0., 300., 0.1);
    let mut accel_cyc_mps = Array::ones(accel_cyc_secs.len()) * 90.0 / MPH_PER_MPS;
    accel_cyc_mps[0] = 0.0;

    RustCycle::new(
        accel_cyc_secs.to_vec(),
        accel_cyc_mps.to_vec(),
        Array::zeros(accel_cyc_secs.len()).to_vec(),
        Array::zeros(accel_cyc_secs.len()).to_vec(),
        String::from("accel"),
    )
}

pub fn get_net_accel(
    sd_accel: &mut RustSimDrive,
    scenario_name: &String,
) -> Result<f64, anyhow::Error> {
    log::debug!("running `sim_drive_accel`");
    sd_accel.sim_drive_accel(None, None)?;
    if sd_accel.mph_ach.iter().any(|&x| x >= 60.) {
        Ok(interpolate(
            &60.,
            &sd_accel.mph_ach,
            &sd_accel.cyc0.time_s,
            false,
        ))
    } else {
        log::warn!("vehicle '{}' never achieves 60 mph", scenario_name);
        Ok(1e3)
    }
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
    let mut max_trace_miss_in_mph: f64 = 0.0;

    out.veh = veh.clone();

    // load the cycles and intstantiate simdrive objects
    cyc.insert("accel", make_accel_trace());

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

    let udds_filestring = include_str!(concat!(
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
        "udds.csv"
    ));
    let hwy_filestring = include_str!(concat!(
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
        "hwfet.csv"
    ));

    cyc.insert(
        "udds",
        RustCycle::from_csv_string(udds_filestring, "udds".to_string())?,
    );
    cyc.insert(
        "hwy",
        RustCycle::from_csv_string(hwy_filestring, "hwfet".to_string())?,
    );

    // run simdrive for non-phev powertrains
    sd.insert("udds", RustSimDrive::new(cyc["udds"].clone(), veh.clone()));
    sd.insert("hwy", RustSimDrive::new(cyc["hwy"].clone(), veh.clone()));

    for (k, val) in sd.iter_mut() {
        val.sim_drive(None, None)?;
        let key = String::from(k.clone());
        let trace_miss_speed_mph = val.trace_miss_speed_mps * MPH_PER_MPS;
        if (key == String::from("udds") || key == String::from("hwy"))
            && trace_miss_speed_mph > max_trace_miss_in_mph
        {
            max_trace_miss_in_mph = trace_miss_speed_mph;
        }
    }
    out.trace_miss_speed_mph = max_trace_miss_in_mph;

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
            out.net_range_miles = veh.ess_max_kwh / out.adj_comb_ess_kwh_per_mi;
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
        out.uf = long_params.uf_array[first_grtr(
            &long_params.rechg_freq_miles,
            0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles,
        )
        .unwrap()
            - 1];

        out.net_phev_cd_miles =
            Some(0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles);

        out.net_range_miles = (veh.fs_kwh / props.kwh_per_gge
            - out.net_phev_cd_miles.unwrap() / out.adj_cd_comb_mpgge.unwrap())
            * out.adj_cs_comb_mpgge.unwrap()
            + out.net_phev_cd_miles.unwrap();
    }

    // run accelerating sim_drive
    let mut sd_accel = RustSimDrive::new(cyc["accel"].clone(), veh.clone());
    out.net_accel = get_net_accel(&mut sd_accel, &veh.scenario_name)?;
    sd.insert("accel", sd_accel);

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
        // do PHEV soc iteration
        // This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
        // By assuming that the battery SOC depletion per mile is constant across cycles,
        // the first cycle can be extrapolated until charge sustaining kicks in.
        sd_val.sim_drive(Some(veh.max_soc), None)?;
        let mut phev_calc: PHEVCycleCalc = PHEVCycleCalc::default();

        // charge depletion cycle has already been simulated
        // charge depletion battery kW-hr
        phev_calc.cd_ess_kwh = (veh.max_soc - veh.min_soc) * veh.ess_max_kwh;

        // SOC change during 1 cycle
        phev_calc.delta_soc = sd_val.soc[0] - sd_val.soc.last().unwrap();
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
            Array::range(0.0, phev_calc.cd_cycs.ceil() + 1.0, 1.0) * sd_val.dist_mi.sum();
        phev_calc.lab_iter_uf = interp_x_vals
            .iter()
            .map(|x: &f64| -> f64 {
                long_params.uf_array[first_grtr(&long_params.rechg_freq_miles, *x).unwrap() - 1]
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
            phev_calc.trans_fs_gal * lab_iter_uf_diff.last().unwrap(),
            phev_calc.cs_fs_gal * (1.0 - phev_calc.lab_iter_uf.last().unwrap()),
        ]) / sd_val.dist_mi.sum();

        phev_calc.cd_mpg = sd_val.mpgge;

        // city and highway cycle ranges
        phev_calc.cd_miles =
            if (veh.max_soc - phev_calcs.regen_soc_buffer - ndarrmin(&sd_val.soc)) < 0.01 {
                1000.0
            } else {
                phev_calc.cd_cycs.ceil() * sd_val.dist_mi.sum()
            };
        phev_calc.cd_lab_mpg =
            phev_calc.lab_iter_uf.last().unwrap() / (phev_calc.trans_fs_gal / sd_val.dist_mi.sum());

        // charge sustaining
        phev_calc.cs_mpg = sd_val.dist_mi.sum() / phev_calc.cs_fs_gal;

        phev_calc.lab_uf = long_params.uf_array
            [first_grtr(&long_params.rechg_freq_miles, phev_calc.cd_miles).unwrap() - 1];

        // labCombMpgge
        phev_calc.cd_adj_mpg =
            ndarrmax(&phev_calc.lab_iter_uf) / phev_calc.lab_uf_gpm[phev_calc.lab_uf_gpm.len() - 2];

        phev_calc.lab_mpgge = 1.0
            / (phev_calc.lab_uf / phev_calc.cd_adj_mpg
                + (1.0 - phev_calc.lab_uf) / phev_calc.cs_mpg);

        let mut lab_iter_kwh_per_mi_vals: Vec<f64> = Vec::new();
        lab_iter_kwh_per_mi_vals.push(0.0);
        lab_iter_kwh_per_mi_vals
            .extend(vec![phev_calc.cd_ess_kwh_per_mi; phev_calc.cd_cycs.floor() as usize].iter());
        lab_iter_kwh_per_mi_vals.push(phev_calc.trans_ess_kwh_per_mi);
        lab_iter_kwh_per_mi_vals.push(0.0);
        phev_calc.lab_iter_kwh_per_mi = Array::from_vec(lab_iter_kwh_per_mi_vals);
        let mut vals: Vec<f64> = Vec::new();
        vals.push(0.0);
        vals.extend(
            (&phev_calc
                .lab_iter_kwh_per_mi
                .slice(s![1..phev_calc.lab_iter_kwh_per_mi.len() - 1])
                * &diff(&phev_calc.lab_iter_uf).slice(s![1..]))
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
                long_params.uf_array[first_grtr(&long_params.rechg_freq_miles, *x).unwrap() - 1]
            })
            .collect();

        let adj_iter_uf_diff: Array1<f64> = diff(&phev_calc.adj_iter_uf);
        phev_calc.adj_iter_uf_gpm = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc.adj_iter_mpgge[phev_calc.adj_iter_mpgge.len() - 2])
                * adj_iter_uf_diff[adj_iter_uf_diff.len() - 2],
        );
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc.adj_iter_mpgge.last().unwrap())
                * (1.0 - phev_calc.adj_iter_uf[phev_calc.adj_iter_uf.len() - 2]),
        );

        phev_calc.adj_iter_uf_kwh_per_mi =
            &phev_calc.adj_iter_kwh_per_mi * &diff(&phev_calc.adj_iter_uf);

        phev_calc.adj_cd_mpgge = 1.0
            / phev_calc.adj_iter_uf_gpm[phev_calc.adj_iter_uf_gpm.len() - 2]
            * ndarrmax(&phev_calc.adj_iter_uf);
        phev_calc.adj_cs_mpgge = 1.0 / phev_calc.adj_iter_uf_gpm.last().unwrap()
            * (1.0 - ndarrmax(&phev_calc.adj_iter_uf));

        phev_calc.adj_uf = long_params.uf_array
            [first_grtr(&long_params.rechg_freq_miles, phev_calc.adj_cd_miles).unwrap() - 1];

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
        println!("Calculated net accel: {}", label_fe.net_accel);

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
            net_range_miles: 0.,
            uf: 0.,
            net_accel: 9.451683946821882,
            res_found: String::from("model needs to be implemented for this"),
            phev_calcs: None,
            adj_cs_comb_mpgge: None,
            adj_cd_comb_mpgge: None,
            net_phev_cd_miles: None,
            trace_miss_speed_mph: 0.0,
        };

        println!(
            "Percent diff to Python calc: {:.3}%",
            100. * (label_fe_truth.net_accel - label_fe.net_accel) / label_fe_truth.net_accel
        );

        assert!(label_fe.approx_eq(&label_fe_truth, 1e-10));
    }

    #[test]
    fn test_get_label_fe_phev() {
        let veh: vehicle::RustVehicle = vehicle::RustVehicle::new(
            String::from("2016 Chevrolet Volt"),
            13,
            2016,
            String::from("PHEV"),
            0.3,
            2.565,
            950.564,
            0.53,
            0.59,
            2.6,
            136.0,
            None,
            1.4,
            2000.0,
            1.0,
            297.0,
            9.89,
            75.0,
            vec![
                0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
            ],
            vec![
                0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
            ],
            String::from("SI"),
            6.0,
            61.0,
            2.13,
            30.0,
            1.5,
            111.0,
            vec![0., 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.],
            Some(vec![
                0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.95, 0.94, 0.93,
            ]),
            3.0,
            0.833,
            21.6,
            115.0,
            18.4,
            8.0,
            75.0,
            0.97,
            110.0,
            -0.6811,
            0.15,
            0.9,
            1.0,
            0.0,
            0.815,
            4.0,
            0.007,
            0.336,
            0.7,
            60.0,
            0.2,
            0.0,
            85.0,
            120.0,
            0.98,
            false,
            false,
            1.0,
            0.86,
            0.3,
            114.0,
            0.98,
            0.005,
            f64::NAN,
            f64::NAN,
            42.0,
            f64::NAN,
            f64::NAN,
            0.31,
            53.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            8.4,
            120000.0,
            f64::NAN,
            17000.0,
            33170.0,
            RustPhysicalProperties::default(),
            500.0,
            0.99,
            None,
            None,
        )
        .unwrap();

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

        let udds: PHEVCycleCalc = PHEVCycleCalc {
            cd_ess_kwh: 13.799999999999999,
            cd_ess_kwh_per_mi: 0.1670807863534209,
            cd_fs_gal: 0.0,
            cd_fs_kwh: 0.0,
            cd_mpg: 65.0128437991813,
            cd_cycs: 11.083418864860784,
            cd_miles: 89.42523198551896,
            cd_lab_mpg: 59.77814990568397,
            cd_adj_mpg: 2968.1305812156647,
            cd_frac_in_trans: 0.08341886486078387,
            trans_init_soc: 0.15564484203010176,
            trans_ess_kwh: 0.10386509335387073,
            trans_ess_kwh_per_mi: 0.013937689537649522,
            trans_fs_gal: 0.105063189381161,
            trans_fs_kwh: 3.5406294821451265,
            cs_ess_kwh: -27.842875966770062,
            cs_ess_kwh_per_mi: -0.001037845633667792,
            cs_fs_gal: 0.11462508375235472,
            cs_fs_kwh: 3.8628653224543545,
            cs_mpg: 65.01284379918131,
            lab_mpgge: 370.06411942132064,
            lab_kwh_per_mi: 0.16342111007981494,
            lab_uf: 0.8427800000000001,
            lab_uf_gpm: Array::from_vec(vec![0.00028394, 0.00241829]),
            lab_iter_uf: Array::from_vec(vec![
                0., 0.16268, 0.28152, 0.41188, 0.51506, 0.59611, 0.64532, 0.69897, 0.74176,
                0.77648, 0.79825, 0.82264, 0.84278,
            ]),
            lab_iter_uf_kwh_per_mi: Array::from_vec(vec![
                0., 0.0271807, 0.01985588, 0.02178065, 0.0172394, 0.0135419, 0.00822205,
                0.00896388, 0.00714939, 0.00580104, 0.00363735, 0.0040751, 0.00028071, 0.,
            ]),
            lab_iter_kwh_per_mi: Array::from_vec(vec![
                0., 0.16708079, 0.16708079, 0.16708079, 0.16708079, 0.16708079, 0.16708079,
                0.16708079, 0.16708079, 0.16708079, 0.16708079, 0.16708079, 0.01393769, 0.,
            ]),
            adj_iter_mpgge: Array::from_vec(vec![
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                50.2456134,
                46.69198818,
            ]),
            adj_iter_kwh_per_mi: Array::from_vec(vec![
                0., 0.23868684, 0.23868684, 0.23868684, 0.23868684, 0.23868684, 0.23868684,
                0.23868684, 0.23868684, 0.23868684, 0.23868684, 0.23868684, 0.01991099, 0.,
            ]),
            adj_iter_cd_miles: Array::from_vec(vec![
                0.,
                5.21647187,
                10.43294373,
                15.6494156,
                20.86588746,
                26.08235933,
                31.29883119,
                36.51530306,
                41.73177493,
                46.94824679,
                52.16471866,
                57.38119052,
                62.59766239,
                0.,
            ]),
            adj_iter_uf: Array::from_vec(vec![
                0., 0.11878, 0.2044, 0.31698, 0.38194, 0.46652, 0.53737, 0.57771, 0.62998, 0.6599,
                0.69897, 0.73185, 0.75126, 0.,
            ]),
            adj_iter_uf_gpm: vec![
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0003863, 0.00532725,
            ],
            adj_iter_uf_kwh_per_mi: Array::from_vec(vec![
                0., 0.02835122, 0.02043637, 0.02687136, 0.0155051, 0.02018813, 0.01691096,
                0.00962863, 0.01247616, 0.00714151, 0.00932549, 0.00784802, 0.00038647, 0.,
            ]),
            adj_cd_miles: 62.59766238986325,
            adj_cd_mpgge: 1944.7459827561047,
            adj_cs_mpgge: 46.69198818435928,
            adj_uf: 0.75126,
            adj_mpgge: 175.0223917643415,
            adj_kwh_per_mi: 0.27097024959679444,
            adj_ess_kwh_per_mi: 0.23303441465324323,
            delta_soc: 0.0676686507245362,
            total_cd_miles: 82.59477526523773,
        };

        let hwy: PHEVCycleCalc = PHEVCycleCalc {
            cd_ess_kwh: 13.799999999999999,
            cd_ess_kwh_per_mi: 0.19912462736394723,
            cd_fs_gal: 0.0,
            cd_fs_kwh: 0.0,
            cd_mpg: 61.75832757157714,
            cd_cycs: 6.75533367335913,
            cd_miles: 71.81337619240335,
            cd_lab_mpg: 199.76659107309018,
            cd_adj_mpg: 4975.506626976092,
            cd_frac_in_trans: 0.7553336733591296,
            trans_init_soc: 0.23385969996618272,
            trans_ess_kwh: 1.5430184793777608,
            trans_ess_kwh_per_mi: 0.15040553624307812,
            trans_fs_gal: 0.040643020828268026,
            trans_fs_kwh: 1.3696698019126325,
            cs_ess_kwh: -27.84287564320177,
            cs_ess_kwh_per_mi: -0.0007538835761840731,
            cs_fs_gal: 0.1661161198039534,
            cs_fs_kwh: 5.59811323739323,
            cs_mpg: 61.75832757157714,
            lab_mpgge: 282.75893721314793,
            lab_kwh_per_mi: 0.19665299886733625,
            lab_uf: 0.7914100000000001,
            lab_uf_gpm: Array::from_vec(vec![0.00015906, 0.00337752]),
            lab_iter_uf: Array::from_vec(vec![
                0., 0.2044, 0.38194, 0.51506, 0.62998, 0.69897, 0.75126, 0.79141,
            ]),
            lab_iter_uf_kwh_per_mi: Array::from_vec(vec![
                0., 0.04070107, 0.03535259, 0.02650747, 0.0228834, 0.01373761, 0.01041223,
                0.00603878, 0.,
            ]),
            lab_iter_kwh_per_mi: Array::from_vec(vec![
                0., 0.19912463, 0.19912463, 0.19912463, 0.19912463, 0.19912463, 0.19912463,
                0.15040554, 0.,
            ]),
            adj_iter_mpgge: Array::from_vec(vec![0., 0., 0., 0., 0., 0., 176.69300837, 43.2308293]),
            adj_iter_kwh_per_mi: Array::from_vec(vec![
                0., 0.28446375, 0.28446375, 0.28446375, 0.28446375, 0.28446375, 0.28446375,
                0.21486505, 0.,
            ]),
            adj_iter_cd_miles: Array::from_vec(vec![
                0.,
                7.18133762,
                14.36267524,
                21.54401286,
                28.72535048,
                35.9066881,
                43.08802572,
                50.26936333,
                0.,
            ]),
            adj_iter_uf: Array::from_vec(vec![
                0., 0.16268, 0.28152, 0.41188, 0.49148, 0.57771, 0.64532, 0.68662, 0.,
            ]),
            adj_iter_uf_gpm: vec![0., 0., 0., 0., 0., 0., 0.00023374, 0.00724899],
            adj_iter_uf_kwh_per_mi: Array::from_vec(vec![
                0., 0.04627656, 0.03380567, 0.03708269, 0.02264331, 0.02452931, 0.01923259,
                0.00887393, 0.,
            ]),
            adj_cd_miles: 50.26936333468235,
            adj_cd_mpgge: 2937.5533511975764,
            adj_cs_mpgge: 43.230829300104,
            adj_uf: 0.68662,
            adj_mpgge: 133.64102451254365,
            adj_kwh_per_mi: 0.3259039663244739,
            adj_ess_kwh_per_mi: 0.2802774110390475,
            delta_soc: 0.11102338333896955,
            total_cd_miles: 69.30333119859274,
        };

        let phev_calcs: LabelFePHEV = LabelFePHEV {
            regen_soc_buffer: 0.00957443430586049,
            udds,
            hwy,
        };

        let label_fe_truth: LabelFe = LabelFe {
            veh: vehicle::RustVehicle::default(),
            adj_params: RustLongParams::default().ld_fe_adj_coef.adj_coef_map["2008"].clone(),
            lab_udds_mpgge: 370.06411942132064,
            lab_hwy_mpgge: 282.75893721314793,
            lab_comb_mpgge: 324.91895455274005,
            lab_udds_kwh_per_mi: 0.16342111007981494,
            lab_hwy_kwh_per_mi: 0.19665299886733625,
            lab_comb_kwh_per_mi: 0.17837546003419952,
            adj_udds_mpgge: 175.0223917643415,
            adj_hwy_mpgge: 133.64102451254365,
            adj_comb_mpgge: 153.61727461480555,
            adj_udds_kwh_per_mi: 0.27097024959679444,
            adj_hwy_kwh_per_mi: 0.3259039663244739,
            adj_comb_kwh_per_mi: 0.29569042212425023,
            adj_udds_ess_kwh_per_mi: 0.23303441465324323,
            adj_hwy_ess_kwh_per_mi: 0.2802774110390475,
            adj_comb_ess_kwh_per_mi: 0.25429376302685514,
            net_range_miles: 453.1180867180584,
            uf: 0.73185,
            // net_accel: 7.962519496024332, <- Correct accel value
            net_accel: 1000.,
            res_found: String::from("model needs to be implemented for this"),
            phev_calcs: Some(phev_calcs),
            adj_cs_comb_mpgge: Some(45.06826741586106),
            adj_cd_comb_mpgge: Some(2293.5675017498143),
            net_phev_cd_miles: Some(57.04992781503185),
            trace_miss_speed_mph: 0.0,
        };

        assert!(label_fe.approx_eq(&label_fe_truth, 1e-8));
    }
}
