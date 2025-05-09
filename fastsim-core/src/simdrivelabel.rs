//! Module containing classes and methods for calculating label fuel economy.
#![cfg(feature = "simdrivelabel")]

use std::collections::HashMap;

// crate local
use crate::drive_cycle::CYC_ACCEL;
use crate::imports::*;
use crate::prelude::*;
use crate::simdrive::{SimDrive, SimParams};
use crate::vehicle::{PowertrainType, Vehicle};

const MPH_PER_MPS: f64 = 2.2369362921;
const CHG_EFF: f64 = 0.86;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FuelProperties {
    /// kWh per gallon of fuel
    pub energy_density: si::,
    /// fuel density
    pub density: si::MassDensity,
}

impl Default for FuelProperties {
    /// Default values for gasoline
    fn default() -> Self {
        Self {
            energy_density: 33.7,
            density: 0.75 * uc::KG / uc::L,
        }
    }
}

#[serde_api]
#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq)]
pub struct LabelFe {
    pub veh: Option<Vehicle>,
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
}

#[serde_api]
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: f64,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[serde_api]
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
    pub lab_uf_gpm: Vec<f64>,
    pub lab_iter_uf: Vec<f64>,
    pub lab_iter_uf_kwh_per_mi: Vec<f64>,
    pub lab_iter_kwh_per_mi: Vec<f64>,
    pub adj_iter_mpgge: Vec<f64>,
    pub adj_iter_kwh_per_mi: Vec<f64>,
    pub adj_iter_cd_miles: Vec<f64>,
    pub adj_iter_uf: Vec<f64>,
    pub adj_iter_uf_gpm: Vec<f64>,
    pub adj_iter_uf_kwh_per_mi: Vec<f64>,
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
    /// Frequency of recharge events
    pub rechg_freq_miles: Vec<f64>,
    /// Array of utility factor
    pub uf_array: Vec<f64>,
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct AdjCoefMap {
    pub adj_coef_map: HashMap<String, AdjCoef>,
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

    let long_params = RustLongParams::default();

    let mut cyc: HashMap<&str, Cycle> = HashMap::new();
    let mut sd = HashMap::new();
    let mut label_fe = LabelFe::default();

    label_fe.veh = Some(veh.clone());

    // load the cycles and instantiate simdrive objects
    cyc.insert("accel", *CYC_ACCEL);
    cyc.insert("udds", Cycle::from_resource("udds.csv", false)?);
    cyc.insert("hwy", Cycle::from_resource("hwfet.csv", false)?);

    // run simdrive for non-phev powertrains
    sd.insert(
        "udds",
        SimDrive::new(veh.clone(), cyc["udds"].clone(), None),
    );
    sd.insert(
        "accel",
        SimDrive::new(veh.clone(), cyc["accel"].clone(), None),
    );
    sd.insert("hwy", SimDrive::new(veh.clone(), cyc["hwy"].clone(), None));

    for (k, val) in sd.iter_mut() {
        val.walk()?;
    }

    // find year-based adjustment parameters
    let adj_params = if veh.year < 2017 {
        &long_params.ld_fe_adj_coef.adj_coef_map["2008"]
    } else {
        // assume 2017 coefficients are valid
        &long_params.ld_fe_adj_coef.adj_coef_map["2017"]
    };
    label_fe.adj_params = adj_params.clone();

    // Check powertrain type
    // In fastsim-3, we use the PowertrainType enum
    let is_phev = matches!(veh.pt_type, PowertrainType::PlugInHybridElectricVehicle(_));
    let is_bev = matches!(veh.pt_type, PowertrainType::BatteryElectricVehicle(_));

    // run calculations for non-PHEV powertrains
    if !is_phev {
        if !is_bev {
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
            label_fe.lab_udds_mpgge = sd["udds"].mpgge;
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
            label_fe.lab_hwy_mpgge = sd["hwy"].mpgge;
            label_fe.lab_comb_mpgge = 1. / (0.55 / sd["udds"].mpgge + 0.45 / sd["hwy"].mpgge);
        } else {
            label_fe.lab_udds_mpgge = 0.;
            label_fe.lab_hwy_mpgge = 0.;
            label_fe.lab_comb_mpgge = 0.;
        }

        if is_bev {
            label_fe.lab_udds_kwh_per_mi = sd["udds"].battery_kwh_per_mi;
            label_fe.lab_hwy_kwh_per_mi = sd["hwy"].battery_kwh_per_mi;
            label_fe.lab_comb_kwh_per_mi =
                0.55 * sd["udds"].battery_kwh_per_mi + 0.45 * sd["hwy"].battery_kwh_per_mi;
        } else {
            label_fe.lab_udds_kwh_per_mi = 0.;
            label_fe.lab_hwy_kwh_per_mi = 0.;
            label_fe.lab_comb_kwh_per_mi = 0.;
        }

        // adjusted values for mpg
        if !is_bev {
            // non-EV case
            // CV or HEV case (not PHEV)
            // HEV SOC iteration is handled in simdrive.SimDriveClassic
            label_fe.adj_udds_mpgge =
                1. / (adj_params.city_intercept + adj_params.city_slope / sd["udds"].mpgge);
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            label_fe.adj_hwy_mpgge =
                1. / (adj_params.hwy_intercept + adj_params.hwy_slope / sd["hwy"].mpgge);
            label_fe.adj_comb_mpgge =
                1. / (0.55 / label_fe.adj_udds_mpgge + 0.45 / label_fe.adj_hwy_mpgge);
        } else {
            // EV case
            // Mpgge is all zero for EV
            label_fe.adj_udds_mpgge = 0.;
            label_fe.adj_hwy_mpgge = 0.;
            label_fe.adj_comb_mpgge = 0.;
        }

        // adjusted kW-hr/mi
        if is_bev {
            // EV Case
            label_fe.adj_udds_kwh_per_mi =
                (1. / f64::max(
                    1. / (adj_params.city_intercept
                        + (adj_params.city_slope
                            / ((1. / label_fe.lab_udds_kwh_per_mi) * props.kwh_per_gge))),
                    (1. / label_fe.lab_udds_kwh_per_mi)
                        * props.kwh_per_gge
                        * (1. - sim_params.max_epa_adj),
                )) * props.kwh_per_gge
                    / CHG_EFF;
            label_fe.adj_hwy_kwh_per_mi =
                (1. / f64::max(
                    1. / (adj_params.hwy_intercept
                        + (adj_params.hwy_slope
                            / ((1. / label_fe.lab_hwy_kwh_per_mi) * props.kwh_per_gge))),
                    (1. / label_fe.lab_hwy_kwh_per_mi)
                        * props.kwh_per_gge
                        * (1. - sim_params.max_epa_adj),
                )) * props.kwh_per_gge
                    / CHG_EFF;
            label_fe.adj_comb_kwh_per_mi =
                0.55 * label_fe.adj_udds_kwh_per_mi + 0.45 * label_fe.adj_hwy_kwh_per_mi;

            label_fe.adj_udds_ess_kwh_per_mi = label_fe.adj_udds_kwh_per_mi * CHG_EFF;
            label_fe.adj_hwy_ess_kwh_per_mi = label_fe.adj_hwy_kwh_per_mi * CHG_EFF;
            label_fe.adj_comb_ess_kwh_per_mi = label_fe.adj_comb_kwh_per_mi * CHG_EFF;

            // range for combined city/highway
            // Get energy capacity from the proper powertrain
            if let PowertrainType::BatteryElectricVehicle(bev) = &veh.powertrain {
                label_fe.net_range_miles =
                    bev.res.energy_capacity().unwrap_or(0.0) / label_fe.adj_comb_ess_kwh_per_mi;
            }
        }

        // utility factor (percent driving in PHEV charge depletion mode)
        label_fe.uf = 0.;
    } else {
        // PHEV
        let phev_calcs =
            get_label_fe_phev(veh, &mut sd, &long_params, adj_params, &sim_params, &props)?;
        label_fe.phev_calcs = Some(phev_calcs.clone());

        // efficiency-related calculations
        // lab
        label_fe.lab_udds_mpgge = phev_calcs.udds.lab_mpgge;
        label_fe.lab_hwy_mpgge = phev_calcs.hwy.lab_mpgge;
        label_fe.lab_comb_mpgge =
            1.0 / (0.55 / phev_calcs.udds.lab_mpgge + 0.45 / phev_calcs.hwy.lab_mpgge);

        label_fe.lab_udds_kwh_per_mi = phev_calcs.udds.lab_kwh_per_mi;
        label_fe.lab_hwy_kwh_per_mi = phev_calcs.hwy.lab_kwh_per_mi;
        label_fe.lab_comb_kwh_per_mi =
            0.55 * phev_calcs.udds.lab_kwh_per_mi + 0.45 * phev_calcs.hwy.lab_kwh_per_mi;

        // adjusted
        label_fe.adj_udds_mpgge = phev_calcs.udds.adj_mpgge;
        label_fe.adj_hwy_mpgge = phev_calcs.hwy.adj_mpgge;
        label_fe.adj_comb_mpgge =
            1.0 / (0.55 / phev_calcs.udds.adj_mpgge + 0.45 / phev_calcs.hwy.adj_mpgge);

        label_fe.adj_cs_comb_mpgge =
            Some(1.0 / (0.55 / phev_calcs.udds.adj_cs_mpgge + 0.45 / phev_calcs.hwy.adj_cs_mpgge));
        label_fe.adj_cd_comb_mpgge =
            Some(1.0 / (0.55 / phev_calcs.udds.adj_cd_mpgge + 0.45 / phev_calcs.hwy.adj_cd_mpgge));

        label_fe.adj_udds_kwh_per_mi = phev_calcs.udds.adj_kwh_per_mi;
        label_fe.adj_hwy_kwh_per_mi = phev_calcs.hwy.adj_kwh_per_mi;
        label_fe.adj_comb_kwh_per_mi =
            0.55 * phev_calcs.udds.adj_kwh_per_mi + 0.45 * phev_calcs.hwy.adj_kwh_per_mi;

        label_fe.adj_udds_ess_kwh_per_mi = phev_calcs.udds.adj_ess_kwh_per_mi;
        label_fe.adj_hwy_ess_kwh_per_mi = phev_calcs.hwy.adj_ess_kwh_per_mi;
        label_fe.adj_comb_ess_kwh_per_mi =
            0.55 * phev_calcs.udds.adj_ess_kwh_per_mi + 0.45 * phev_calcs.hwy.adj_ess_kwh_per_mi;

        // range for combined city/highway
        // utility factor (percent driving in charge depletion mode)
        label_fe.uf = long_params.uf_array[first_grtr(
            &long_params.rechg_freq_miles,
            0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles,
        )
        .unwrap()
            - 1];

        label_fe.net_phev_cd_miles =
            Some(0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles);

        // For PHEVs, calculate net range as the sum of CD range and CS range
        if let PowertrainType::PlugInHybridElectricVehicle(phev) = &veh.powertrain {
            // Get CS range by determining how much fuel energy remains after depleting the battery
            let fuel_energy_kwh = phev.fs.energy_capacity().unwrap_or(0.0);
            let fuel_energy_gge = fuel_energy_kwh / props.kwh_per_gge;

            label_fe.net_range_miles = (fuel_energy_gge
                - label_fe.net_phev_cd_miles.unwrap() / label_fe.adj_cd_comb_mpgge.unwrap())
                * label_fe.adj_cs_comb_mpgge.unwrap()
                + label_fe.net_phev_cd_miles.unwrap();
        }
    }

    // run accelerating sim_drive
    let mut sd_accel = SimDrive::new(cyc["accel"].clone(), veh.clone());
    label_fe.net_accel = get_net_accel(&mut sd_accel, &veh.name)?;
    sd.insert("accel", sd_accel);

    // success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    label_fe.res_found = String::from("model needs to be implemented for this");

    if full_detail.unwrap_or(false) && verbose.unwrap_or(false) {
        println!("{:#?}", label_fe);
        Ok((label_fe, Some(sd)))
    } else if full_detail.unwrap_or(false) {
        Ok((label_fe, Some(sd)))
    } else if verbose.unwrap_or(false) {
        println!("{:#?}", label_fe);
        Ok((label_fe, None))
    } else {
        Ok((label_fe, None))
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "get_label_fe")]
#[cfg_attr(feature = "pyo3", pyo3(signature = (veh, full_detail=None, verbose=None)))]
/// pyo3 version of [get_label_fe]
pub fn get_label_fe_py(
    veh: &Vehicle,
    full_detail: Option<bool>,
    verbose: Option<bool>,
) -> anyhow::Result<(LabelFe, Option<HashMap<&str, SimDrive>>)> {
    let label_fe = get_label_fe(veh, full_detail, verbose)?;
    Ok(label_fe)
}

/// PHEV-specific function for label fe.
///
/// # Arguments
/// - `veh` : vehicle::Vehicle
/// - `sd` : SimDrive objects to use for label fe calculations
/// - `long_params` : Struct for longparams.json values
/// - `adj_params`: Adjusted coefficients from longparams.json
/// - `sim_params` : SimParams
/// - `props` : RustPhysicalProperties
///
/// # Returns
/// label fuel economy values for PHEV as a struct.
pub fn get_label_fe_phev(
    veh: &Vehicle,
    sd: &mut HashMap<&str, SimDrive>,
    long_params: &RustLongParams,
    adj_params: &AdjCoef,
    sim_params: &SimParams,
    props: &RustPhysicalProperties,
) -> anyhow::Result<LabelFePHEV> {
    // Get access to the PHEV powertrain
    let phev_max_soc;
    let phev_min_soc;
    let phev_max_regen;
    let phev_veh_kg;
    let phev_mc_peak_eff;
    let phev_ess_max_kwh;
    let phev_chg_eff;

    if let PowertrainType::PlugInHybridElectricVehicle(phev) = &veh.powertrain {
        phev_max_soc = phev.res.max_soc;
        phev_min_soc = phev.res.min_soc;
        phev_max_regen = veh
            .state
            .max_regen
            .get_fresh(|| format_dbg!())?
            .unwrap_or(0.0);
        phev_veh_kg = veh.mass().unwrap_or(0.0).get::<si::kilogram>();
        phev_mc_peak_eff = phev.em.get_peak_eff();
        phev_ess_max_kwh = phev.res.energy_capacity().unwrap_or(0.0);
        phev_chg_eff = phev
            .state
            .chg_eff
            .get_fresh(|| format_dbg!())?
            .unwrap_or(CHG_EFF);
    } else {
        bail!("Vehicle is not a PHEV");
    }

    let mut phev_calcs = LabelFePHEV {
        regen_soc_buffer: f64::min(
            ((0.5 * phev_veh_kg * ((60. * (1. / MPH_PER_MPS)).powi(2)))
                * (1. / 3600.)
                * (1. / 1000.)
                * phev_max_regen
                * phev_mc_peak_eff)
                / phev_ess_max_kwh,
            (phev_max_soc - phev_min_soc) / 2.0,
        ),
        ..Default::default()
    };

    // charge sustaining behavior
    for (key, sd_val) in sd.iter_mut() {
        // do PHEV soc iteration
        // This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
        // By assuming that the battery SOC depletion per mile is constant across cycles,
        // the first cycle can be extrapolated until charge sustaining kicks in.
        sd_val.sim_drive(Some(phev_max_soc), None)?;
        let mut phev_calc = PHEVCycleCalc::default();

        // charge depletion cycle has already been simulated
        // charge depletion battery kW-hr
        phev_calc.cd_ess_kwh = (phev_max_soc - phev_min_soc) * phev_ess_max_kwh;

        // SOC change during 1 cycle
        phev_calc.delta_soc = sd_val.soc[0] - sd_val.soc.last().unwrap();
        // total number of miles in charge depletion mode, assuming constant kWh_per_mi
        phev_calc.total_cd_miles =
            (phev_max_soc - phev_min_soc) * phev_ess_max_kwh / sd_val.battery_kwh_per_mi;
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
        let interp_x_vals =
            Array1::range(0.0, phev_calc.cd_cycs.ceil() + 1.0, 1.0) * sd_val.dist_mi.sum();
        phev_calc.lab_iter_uf = interp_x_vals
            .iter()
            .map(|x: &f64| -> f64 {
                long_params.uf_array[first_grtr(&long_params.rechg_freq_miles, *x).unwrap() - 1]
            })
            .collect();

        // transition cycle
        phev_calc.trans_init_soc = phev_max_soc - phev_calc.cd_cycs.floor() * phev_calc.delta_soc;

        // run the transition cycle
        sd_val.sim_drive(Some(phev_calc.trans_init_soc), None)?;
        // charge depletion battery kW-hr
        phev_calc.trans_ess_kwh =
            phev_calc.cd_ess_kwh_per_mi * sd_val.dist_mi.sum() * phev_calc.cd_frac_in_trans;
        phev_calc.trans_ess_kwh_per_mi = phev_calc.cd_ess_kwh_per_mi * phev_calc.cd_frac_in_trans;

        // charge sustaining
        // the 0.01 is here to be consistent with Excel
        let init_soc = sd_val.veh.min_soc + 0.01;
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

        let lab_iter_uf_diff = diff(&phev_calc.lab_iter_uf);
        phev_calc.lab_uf_gpm = Array1::from_vec(vec![
            phev_calc.trans_fs_gal * lab_iter_uf_diff.last().unwrap(),
            phev_calc.cs_fs_gal * (1.0 - phev_calc.lab_iter_uf.last().unwrap()),
        ]) / sd_val.dist_mi.sum();

        phev_calc.cd_mpg = sd_val.mpgge;

        // city and highway cycle ranges
        phev_calc.cd_miles =
            if (phev_max_soc - phev_calcs.regen_soc_buffer - sd_val.soc.min()?) < 0.01 {
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
            phev_calc.lab_iter_uf.max()? / phev_calc.lab_uf_gpm[phev_calc.lab_uf_gpm.len() - 2];

        phev_calc.lab_mpgge = 1.0
            / (phev_calc.lab_uf / phev_calc.cd_adj_mpg
                + (1.0 - phev_calc.lab_uf) / phev_calc.cs_mpg);

        let mut lab_iter_kwh_per_mi_vals = Vec::new();
        lab_iter_kwh_per_mi_vals.push(0.0);
        lab_iter_kwh_per_mi_vals
            .extend(vec![phev_calc.cd_ess_kwh_per_mi; phev_calc.cd_cycs.floor() as usize].iter());
        lab_iter_kwh_per_mi_vals.push(phev_calc.trans_ess_kwh_per_mi);
        lab_iter_kwh_per_mi_vals.push(0.0);
        phev_calc.lab_iter_kwh_per_mi = Array1::from_vec(lab_iter_kwh_per_mi_vals);
        let mut vals = Vec::new();
        vals.push(0.0);
        vals.extend(
            (&phev_calc
                .lab_iter_kwh_per_mi
                .slice(s![1..phev_calc.lab_iter_kwh_per_mi.len() - 1])
                * &diff(&phev_calc.lab_iter_uf).slice(s![1..]))
                .iter(),
        );
        vals.push(0.0);
        phev_calc.lab_iter_uf_kwh_per_mi = Array1::from_vec(vals);

        phev_calc.lab_kwh_per_mi =
            phev_calc.lab_iter_uf_kwh_per_mi.sum() / phev_calc.lab_iter_uf.max()?;

        let mut adj_iter_mpgge_vals = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        let mut adj_iter_kwh_per_mi_vals = vec![0.0; phev_calc.lab_iter_kwh_per_mi.len()];
        if *key == "udds" {
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));

            for (c, _) in phev_calc.lab_iter_kwh_per_mi.iter().enumerate() {
                if phev_calc.lab_iter_kwh_per_mi[c] == 0.0 {
                    adj_iter_kwh_per_mi_vals[c] = 0.0;
                } else {
                    adj_iter_kwh_per_mi_vals[c] =
                        (1.0 / f64::max(
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
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.trans_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)))),
                sd_val.dist_mi.sum() / (phev_calc.cs_fs_kwh / props.kwh_per_gge)
                    * (1.0 - sim_params.max_epa_adj),
            ));

            for (c, _) in phev_calc.lab_iter_kwh_per_mi.iter().enumerate() {
                if phev_calc.lab_iter_kwh_per_mi[c] == 0.0 {
                    adj_iter_kwh_per_mi_vals[c] = 0.0;
                } else {
                    adj_iter_kwh_per_mi_vals[c] =
                        (1.0 / f64::max(
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
        phev_calc.adj_iter_mpgge = Array1::from(adj_iter_mpgge_vals);
        phev_calc.adj_iter_kwh_per_mi = Array1::from(adj_iter_kwh_per_mi_vals);

        phev_calc.adj_iter_cd_miles =
            Array1::from_vec(vec![0.0; phev_calc.cd_cycs.ceil() as usize + 2]);
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
            if phev_max_soc - phev_calcs.regen_soc_buffer - sd_val.soc.min()? < 0.01 {
                1000.0
            } else {
                *phev_calc.adj_iter_cd_miles.max()?
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

        let adj_iter_uf_diff = diff(&phev_calc.adj_iter_uf);
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
            * phev_calc.adj_iter_uf.max()?;
        phev_calc.adj_cs_mpgge = 1.0 / phev_calc.adj_iter_uf_gpm.last().unwrap()
            * (1.0 - phev_calc.adj_iter_uf.max()?);

        phev_calc.adj_uf = long_params.uf_array
            [first_grtr(&long_params.rechg_freq_miles, phev_calc.adj_cd_miles).unwrap() - 1];

        phev_calc.adj_mpgge = 1.0
            / (phev_calc.adj_uf / phev_calc.adj_cd_mpgge
                + (1.0 - phev_calc.adj_uf) / phev_calc.adj_cs_mpgge);

        phev_calc.adj_kwh_per_mi =
            phev_calc.adj_iter_uf_kwh_per_mi.sum() / phev_calc.adj_iter_uf.max()? / phev_chg_eff;

        phev_calc.adj_ess_kwh_per_mi =
            phev_calc.adj_iter_uf_kwh_per_mi.sum() / phev_calc.adj_iter_uf.max()?;

        match *key {
            "udds" => phev_calcs.udds = phev_calc.clone(),
            "hwy" => phev_calcs.hwy = phev_calc.clone(),
            &_ => bail!("No field for cycle {}", key),
        };
    }

    Ok(phev_calcs)
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "get_label_fe_phev")]
/// pyo3 version of [get_label_fe_phev]
pub fn get_label_fe_phev_py(
    veh: &Vehicle,
    sd_dict: pyo3::Bound<'_, pyo3::types::PyDict>,
    adj_params: AdjCoef,
    long_params: RustLongParams,
    sim_params: &SimParams,
    props: RustPhysicalProperties,
) -> anyhow::Result<LabelFePHEV> {
    let mut sd_mut: HashMap<String, SimDrive> = HashMap::new();

    for (key, value) in sd_dict.iter() {
        let key_extracted = key
            .extract::<String>()
            .with_context(|| format!("{}\nFailed to extract key", format_dbg!()))?;
        let value_extracted = value
            .extract()
            .with_context(|| format!("{}\nFailed to extract value", format_dbg!()))?;
        sd_mut.insert(key_extracted, value_extracted);
    }

    // type conversion on keys to satisfy function arg below
    let mut sd_mut =
        HashMap::from_iter(sd_mut.iter().map(|item| (item.0.as_str(), item.1.clone())));

    get_label_fe_phev(
        veh,
        &mut sd_mut,
        &long_params,
        &adj_params,
        sim_params,
        &props,
    )
}
