//! Module containing classes and methods for calculating label fuel economy.
#![cfg(feature = "simdrivelabel")]

use ndarray::Array1;
use std::collections::HashMap;

// crate local
use crate::drive_cycle::Cycle;
use crate::imports::*;
use crate::si::{self, Ratio};
use crate::simdrive::{SimDrive, SimParams};
use crate::utils::interp::{first_grtr, interpolate};
use crate::vehicle::{PowertrainType, Vehicle};

const MPH_PER_MPS: f64 = 2.2369362921;
const CHG_EFF: f64 = 0.86;

#[fastsim_proc_macros::serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LabelFe {
    pub veh: Vehicle,
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
    pub net_range_miles: f64,
    pub uf: f64,
    pub net_accel: f64,
    pub res_found: String,
    pub phev_calcs: Option<LabelFePHEV>,
    pub adj_cs_comb_mpgge: Option<f64>,
    pub adj_cd_comb_mpgge: Option<f64>,
    pub net_phev_cd_miles: Option<f64>,
    pub trace_miss_speed_mph: f64,
}

#[fastsim_proc_macros::serde_api]
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: f64,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[fastsim_proc_macros::serde_api]
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
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

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct AdjCoef {
    pub city_intercept: f64,
    pub city_slope: f64,
    pub hwy_intercept: f64,
    pub hwy_slope: f64,
}

impl Default for AdjCoef {
    fn default() -> Self {
        Self {
            city_intercept: 0.003259,
            city_slope: 1.1805,
            hwy_intercept: 0.001376,
            hwy_slope: 1.3466,
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct RustLongParams {
    pub ld_fe_adj_coef: AdjCoefMap,
    pub rechg_freq_miles: Vec<f64>,
    pub uf_array: Vec<f64>,
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct AdjCoefMap {
    pub adj_coef_map: HashMap<String, AdjCoef>,
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct RustPhysicalProperties {
    pub air_density_kg_per_m3: f64,
    pub a_grav_mps2: f64,
    pub kwh_per_gge: f64,
    pub fuel_rho_kg__L: f64,
    pub fuel_afr_stoich: f64,
    pub orphaned: bool,
}

impl Default for LabelFe {
    fn default() -> Self {
        Self {
            veh: Vehicle::default(),
            adj_params: AdjCoef::default(),
            lab_udds_mpgge: 0.0,
            lab_hwy_mpgge: 0.0,
            lab_comb_mpgge: 0.0,
            lab_udds_kwh_per_mi: 0.0,
            lab_hwy_kwh_per_mi: 0.0,
            lab_comb_kwh_per_mi: 0.0,
            adj_udds_mpgge: 0.0,
            adj_hwy_mpgge: 0.0,
            adj_comb_mpgge: 0.0,
            adj_udds_kwh_per_mi: 0.0,
            adj_hwy_kwh_per_mi: 0.0,
            adj_comb_kwh_per_mi: 0.0,
            adj_udds_ess_kwh_per_mi: 0.0,
            adj_hwy_ess_kwh_per_mi: 0.0,
            adj_comb_ess_kwh_per_mi: 0.0,
            net_range_miles: 0.0,
            uf: 0.0,
            net_accel: 0.0,
            res_found: String::new(),
            phev_calcs: None,
            adj_cs_comb_mpgge: None,
            adj_cd_comb_mpgge: None,
            net_phev_cd_miles: None,
            trace_miss_speed_mph: 0.0,
        }
    }
}

fn make_accel_trace() -> Cycle {
    let accel_cyc_secs = Array1::range(0., 300., 0.1);
    let cyc_len = accel_cyc_secs.len();
    let mut accel_cyc_mps = Array1::ones(cyc_len) * 90.0 / MPH_PER_MPS;
    accel_cyc_mps[0] = 0.0;

    Cycle {
        time_s: accel_cyc_secs,
        mps: accel_cyc_mps,
        grade: Array1::zeros(cyc_len),
        road_type: Array1::zeros(cyc_len),
        name: String::from("accel"),
        orphaned: false,
    }
}

#[cfg(feature = "pyo3")]
#[imports::pyfunction(name = "make_accel_trace")]
/// pyo3 version of [make_accel_trace]
pub fn make_accel_trace_py() -> Cycle {
    make_accel_trace()
}

pub fn get_net_accel(sd_accel: &mut SimDrive, scenario_name: &String) -> anyhow::Result<f64> {
    #[cfg(feature = "logging")]
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
        #[cfg(feature = "logging")]
        log::warn!("vehicle '{}' never achieves 60 mph", scenario_name);
        Ok(1e3)
    }
}

#[cfg(feature = "pyo3")]
#[imports::pyfunction(name = "get_net_accel")]
/// pyo3 version of [get_net_accel]
pub fn get_net_accel_py(sd_accel: &mut SimDrive, scenario_name: &str) -> anyhow::Result<f64> {
    let result = get_net_accel(sd_accel, &scenario_name.to_string())?;
    Ok(result)
}

pub fn get_label_fe(
    veh: &Vehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> anyhow::Result<(LabelFe, Option<HashMap<&str, SimDrive>>)> {
    // Generates label fuel economy (FE) values for a provided vehicle.
    //
    // Arguments:
    // ----------
    // veh : vehicle::Vehicle
    // full_detail : boolean, default False
    //     If True, sim_drive objects for each cycle are also returned.
    // verbose : boolean, default false
    //     If true, print out key results
    //
    // Returns label fuel economy values as a struct and (optionally)
    // simdrive::SimDrive objects.

    let sim_params = SimParams::default();
    let props = RustPhysicalProperties::default();
    let long_params = RustLongParams::default();

    let mut cyc = HashMap::new();
    let mut sd = HashMap::new();
    let mut out = LabelFe::default();
    let mut max_trace_miss_in_mph = 0.0;

    out.veh = veh.clone();

    // load the cycles and instantiate simdrive objects
    cyc.insert("accel", make_accel_trace());

    // In fastsim-3, we'd need to implement cycle loading from resources
    // This is a placeholder - actual implementation would depend on how cycles are stored
    cyc.insert("udds", Cycle::from_resource("udds.csv", false)?);
    cyc.insert("hwy", Cycle::from_resource("hwfet.csv", false)?);

    // run simdrive for non-phev powertrains
    sd.insert("udds", SimDrive::new(cyc["udds"].clone(), veh.clone()));
    sd.insert("hwy", SimDrive::new(cyc["hwy"].clone(), veh.clone()));

    for (k, val) in sd.iter_mut() {
        val.sim_drive(None, None)?;
        let key = k;
        let trace_miss_speed_mph = val.trace_miss_speed_mps * MPH_PER_MPS;
        if (key == &"udds" || key == &"hwy") && trace_miss_speed_mph > max_trace_miss_in_mph {
            max_trace_miss_in_mph = trace_miss_speed_mph;
        }
    }
    out.trace_miss_speed_mph = max_trace_miss_in_mph;

    // find year-based adjustment parameters
    let adj_params = if veh.veh_year < 2017 {
        &long_params.ld_fe_adj_coef.adj_coef_map["2008"]
    } else {
        // assume 2017 coefficients are valid
        &long_params.ld_fe_adj_coef.adj_coef_map["2017"]
    };
    out.adj_params = adj_params.clone();

    // Check powertrain type
    // In fastsim-3, we use the PowertrainType enum
    let is_phev = false; // TODO: Set this properly once PHEV is implemented
    let is_bev = matches!(veh.powertrain, PowertrainType::BatteryElectricVehicle(_));
    
    // run calculations for non-PHEV powertrains
    if !is_phev {
        if !is_bev {
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

        if is_bev {
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
        if !is_bev {
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
        if is_bev {
            // EV Case
            out.adj_udds_kwh_per_mi =
                (1. / f64::max(
                    1. / (adj_params.city_intercept
                        + (adj_params.city_slope
                            / ((1. / out.lab_udds_kwh_per_mi) * props.kwh_per_gge))),
                    (1. / out.lab_udds_kwh_per_mi)
                        * props.kwh_per_gge
                        * (1. - sim_params.max_epa_adj),
                )) * props.kwh_per_gge
                    / CHG_EFF;
            out.adj_hwy_kwh_per_mi =
                (1. / f64::max(
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

            out.adj_udds_ess_kwh_per_mi = out.adj_udds_kwh_per_mi * CHG_EFF;
            out.adj_hwy_ess_kwh_per_mi = out.adj_hwy_kwh_per_mi * CHG_EFF;
            out.adj_comb_ess_kwh_per_mi = out.adj_comb_kwh_per_mi * CHG_EFF;

            // range for combined city/highway
            // Get energy capacity from the proper powertrain 
            if let PowertrainType::BatteryElectricVehicle(bev) = &veh.powertrain {
                out.net_range_miles = bev.res.energy_capacity().unwrap_or(0.0) / out.adj_comb_ess_kwh_per_mi;
            }
        }

        // utility factor (percent driving in PHEV charge depletion mode)
        out.uf = 0.;
    } else {
        // PHEV not yet implemented in fastsim-3
        bail!("PHEV not yet implemented in fastsim-3");
    }

    // run accelerating sim_drive
    let mut sd_accel = SimDrive::new(cyc["accel"].clone(), veh.clone());
    out.net_accel = get_net_accel(&mut sd_accel, &veh.name)?;
    sd.insert("accel", sd_accel);

    // success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    out.res_found = String::from("model needs to be implemented for this");

    if full_detail.unwrap_or(false) && verbose.unwrap_or(false) {
        println!("{:#?}", out);
        Ok((out, Some(sd)))
    } else if full_detail.unwrap_or(false) {
        Ok((out, Some(sd)))
    } else if verbose.unwrap_or(false) {
        println!("{:#?}", out);
        Ok((out, None))
    } else {
        Ok((out, None))
    }
}

#[cfg(feature = "pyo3")]
#[imports::pyfunction(name = "get_label_fe")]
#[cfg_attr(feature = "pyo3", pyo3(signature = (veh, full_detail=None, verbose=None)))]
/// pyo3 version of [get_label_fe]
pub fn get_label_fe_py(
    veh: &Vehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> anyhow::Result<(LabelFe, Option<HashMap<&str, SimDrive>>)> {
    let result = get_label_fe(veh, full_detail, verbose)?;
    Ok(result)
}

/// Utility function to calculate difference between adjacent elements in an array
fn diff(arr: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(arr.len() - 1);
    for i in 0..result.len() {
        result[i] = arr[i + 1] - arr[i];
    }
    result
}