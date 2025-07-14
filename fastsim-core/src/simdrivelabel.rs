//! Module containing classes and methods for calculating label fuel economy.

use std::collections::HashMap;

// crate local
use crate::drive_cycle::{Cycle, CYC_ACCEL};
use crate::imports::*;
use crate::simdrive::SimDrive;
use crate::vehicle::{PowertrainType, Vehicle};

/// Return first index of `arr` greater than `cut`
fn first_grtr(arr: &[f64], cut: f64) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x > cut).unwrap_or(len - 1)) // unwrap_or allows for default if not found
}

/// Returns time [s] for 0-60 mph acceleration at max power
pub fn get_0_to_60_time(sd_accel: &mut SimDrive) -> anyhow::Result<f64> {
    sd_accel.sim_params.trace_miss_opts = TraceMissOptions::Allow;
    sd_accel.walk().with_context(|| format_dbg!())?;

    // Extract speed values in mph
    let mut speed_mph: Vec<f64> = vec![];
    for s in sd_accel.veh.history.speed_ach.clone() {
        speed_mph.push(s.get_fresh(|| format_dbg!())?.get::<si::mile_per_hour>())
    }

    // Extract time values in seconds
    let time_s: Vec<f64> = sd_accel
        .cyc
        .time
        .iter()
        .map(|t| t.get::<si::second>())
        .collect();

    // Check if vehicle reaches 60 mph
    if speed_mph.iter().any(|&x| x >= 60.0) {
        // Create interpolator from speed to time
        let interp: InterpolatorEnumOwned<f64> = InterpolatorEnum::new_1d(
            speed_mph.clone().into(),
            time_s.clone().into(),
            strategy::Linear,
            Extrapolate::Clamp,
        )
        .with_context(|| format_dbg!())?;

        // Interpolate time at 60 mph
        let accel_time = interp.interpolate(&[60.0])?;
        Ok(accel_time)
    } else {
        // Vehicle doesn't reach 60 mph
        println!(
            "Warning: Vehicle '{}' doesn't reach 60 mph in the acceleration test",
            sd_accel.veh.name
        );
        Ok(f64::NAN)
    }
}

// const MPH_PER_MPS: f64 = 2.2369362921;
const DEFAULT_CHG_EFF: f64 = 0.86;

#[serde_api]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct FuelProperties {
    // TODO: make a way to serialize/deserialize with "J/m^3"
    // fuel energy density
    /// fuel energy density (i.e. energy per unit mass, which has the same base
    /// units as pressure)
    pub energy_density: si::Pressure,
    /// fuel density
    pub density: si::MassDensity,
}

impl Init for FuelProperties {}
impl SerdeAPI for FuelProperties {}

#[pyo3_api]
impl FuelProperties {}

impl Default for FuelProperties {
    /// Default values for gasoline
    fn default() -> Self {
        Self {
            energy_density: 33.7 * uc::KWH / uc::GALLON,
            density: 0.75 * uc::KG / uc::L,
        }
    }
}

const J_PER_KWH: f64 = 3_600.0;
lazy_static! {
    static ref CUBIC_METER_PER_GAL: f64 = 3.79e-3;
}

impl FuelProperties {
    fn kwh_per_gge(&self) -> f64 {
        self.energy_density.get::<si::joule_per_cubic_meter>() / J_PER_KWH * *CUBIC_METER_PER_GAL
    }
}

trait VehicleEfficiency {
    fn mpg(&self, energy_density: si::Pressure) -> anyhow::Result<f64>;

    fn kwh_per_mi(&self) -> anyhow::Result<f64>;
}

impl VehicleEfficiency for Vehicle {
    fn mpg(&self, energy_density: si::Pressure) -> anyhow::Result<f64> {
        if let Some(fc) = self.fc() {
            Ok(self
                .state
                .dist
                .get_fresh(|| format_dbg!())?
                .get::<si::mile>()
                / (*fc.state.energy_fuel.get_fresh(|| format_dbg!())? / energy_density)
                    .get::<si::gallon>())
        } else {
            Ok(f64::NAN)
        }
    }

    fn kwh_per_mi(&self) -> anyhow::Result<f64> {
        if let Some(res) = self.res() {
            Ok(res
                .state
                .energy_out_chemical
                .get_fresh(|| format_dbg!())?
                .get::<si::kilowatt_hour>()
                / self
                    .state
                    .dist
                    .get_fresh(|| format_dbg!())?
                    .get::<si::mile>())
        } else {
            Ok(f64::NAN)
        }
    }
}

#[serde_api]
#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
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

#[pyo3_api]
impl LabelFe {}

impl Init for LabelFe {}
impl SerdeAPI for LabelFe {}

#[serde_api]
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Label fuel economy values for a PHEV vehicle
pub struct LabelFePHEV {
    pub regen_soc_buffer: si::Ratio,
    pub udds: PHEVCycleCalc,
    pub hwy: PHEVCycleCalc,
}

#[pyo3_api]
impl LabelFePHEV {}

impl Init for LabelFePHEV {}
impl SerdeAPI for LabelFePHEV {}

#[serde_api]
#[derive(Default, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
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
    pub trans_init_soc: si::Ratio,
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
    pub delta_soc: si::Ratio,
    /// Total number of miles in charge depletion mode, assuming constant kWh_per_mi
    pub total_cd_miles: f64,
}

impl Init for PHEVCycleCalc {}
impl SerdeAPI for PHEVCycleCalc {}

#[pyo3_api]
impl PHEVCycleCalc {}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct AdjCoef {
    pub city_intercept: f64,
    pub city_slope: f64,
    pub hwy_intercept: f64,
    pub hwy_slope: f64,
}

#[pyo3_api]
impl AdjCoef {}

impl Init for AdjCoef {}
impl SerdeAPI for AdjCoef {}

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

#[serde_api]
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
pub struct PhevUtilizationParams {
    pub adj_coef_map: HashMap<String, AdjCoef>,
    /// Frequency of recharge events
    pub rechg_freq_miles: Vec<f64>,
    /// Array of utility factor
    pub uf_array: Vec<f64>,
}

impl Init for PhevUtilizationParams {}
impl SerdeAPI for PhevUtilizationParams {}

impl Default for PhevUtilizationParams {
    fn default() -> Self {
        Self::from_json(&*PHEV_UTIL_PARAMS, false).unwrap()
    }
}

lazy_static! {
    static ref PHEV_UTIL_PARAMS: String =
        include_str!("./simdrivelabel/longparams.json").to_string();
}

/// Generates label fuel economy (FE) values for a provided vehicle.
///
/// # Arguments
///
/// - `veh`: vehicle::Vehicle
/// - `full_detail`: boolean, default False
///   If True, sim_drive objects for each cycle are also returned.
/// - `verbose`: boolean, default false
///   If true, print out key results
///
/// Returns label fuel economy values as a struct and (optionally)
/// simdrive::SimDrive objects.
pub fn get_label_fe(
    veh: &mut Vehicle,
    max_epa_adj: Option<f64>,
    full_detail: bool,
    fuel_props: Option<FuelProperties>,
    phev_utilization_params: Option<PhevUtilizationParams>,
    verbose: bool,
) -> anyhow::Result<(LabelFe, Option<HashMap<&str, SimDrive>>)> {
    let max_epa_adj = max_epa_adj.unwrap_or(0.3);
    let phev_utilization_params = &phev_utilization_params.unwrap_or_default();
    let fuel_props = fuel_props.unwrap_or_default();

    let mut cyc: HashMap<&str, Cycle> = HashMap::new();
    let mut sd = HashMap::new();
    let mut label_fe = LabelFe::default();

    label_fe.veh = Some(veh.clone());

    // load the cycles and instantiate simdrive objects
    cyc.insert("accel", CYC_ACCEL.clone());
    cyc.insert("udds", Cycle::from_resource("udds.csv", false)?);
    cyc.insert("hwy", Cycle::from_resource("hwfet.csv", false)?);

    if veh.pt_type.is_plug_in_hybrid_electric_vehicle() {
        let rm = veh.res_mut().unwrap();
        rm.state.soc.check_and_reset(|| format_dbg!()).unwrap();
        rm.state.soc.update(rm.max_soc, || format_dbg!()).unwrap();
    }

    // run simdrive for non-phev powertrains
    sd.insert(
        "udds",
        SimDrive::new(veh.clone(), cyc["udds"].clone(), None),
    );
    sd.insert("hwy", SimDrive::new(veh.clone(), cyc["hwy"].clone(), None));

    for (k, val) in sd.iter_mut() {
        val.walk().with_context(|| format_dbg!(k))?;
    }

    // find year-based adjustment parameters
    let adj_params = if veh.year < 2017 {
        &phev_utilization_params.adj_coef_map["2008"]
    } else {
        // assume 2017 coefficients are valid
        &phev_utilization_params.adj_coef_map["2017"]
    };
    label_fe.adj_params = adj_params.clone();

    // Check powertrain type
    let is_phev = matches!(veh.pt_type, PowertrainType::PlugInHybridElectricVehicle(_));
    let is_bev = matches!(veh.pt_type, PowertrainType::BatteryElectricVehicle(_));

    // run calculations for non-PHEV powertrains
    if !is_phev {
        if !is_bev {
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labUddsMpgge
            label_fe.lab_udds_mpgge = sd["udds"].veh.mpg(fuel_props.energy_density)?;
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!labHwyMpgge
            label_fe.lab_hwy_mpgge = sd["hwy"].veh.mpg(fuel_props.energy_density)?;
            label_fe.lab_comb_mpgge = 1.
                / (0.55 / sd["udds"].veh.mpg(fuel_props.energy_density)?
                    + 0.45 / sd["hwy"].veh.mpg(fuel_props.energy_density)?);
        } else {
            label_fe.lab_udds_mpgge = 0.;
            label_fe.lab_hwy_mpgge = 0.;
            label_fe.lab_comb_mpgge = 0.;
        }

        if is_bev {
            label_fe.lab_udds_kwh_per_mi = sd["udds"].veh.kwh_per_mi()?;
            label_fe.lab_hwy_kwh_per_mi = sd["hwy"].veh.kwh_per_mi()?;
            label_fe.lab_comb_kwh_per_mi =
                0.55 * sd["udds"].veh.kwh_per_mi()? + 0.45 * sd["hwy"].veh.kwh_per_mi()?;
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
            label_fe.adj_udds_mpgge = 1.
                / (adj_params.city_intercept
                    + adj_params.city_slope / sd["udds"].veh.mpg(fuel_props.energy_density)?);
            // compare to Excel 'VehicleIO'!C203 or 'VehicleIO'!adjHwyMpgge
            label_fe.adj_hwy_mpgge = 1.
                / (adj_params.hwy_intercept
                    + adj_params.hwy_slope / sd["hwy"].veh.mpg(fuel_props.energy_density)?);
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
                            / ((1. / label_fe.lab_udds_kwh_per_mi) * fuel_props.kwh_per_gge()))),
                    (1. / label_fe.lab_udds_kwh_per_mi)
                        * fuel_props.kwh_per_gge()
                        * (1. - max_epa_adj),
                )) * fuel_props.kwh_per_gge()
                    / DEFAULT_CHG_EFF;
            label_fe.adj_hwy_kwh_per_mi =
                (1. / f64::max(
                    1. / (adj_params.hwy_intercept
                        + (adj_params.hwy_slope
                            / ((1. / label_fe.lab_hwy_kwh_per_mi) * fuel_props.kwh_per_gge()))),
                    (1. / label_fe.lab_hwy_kwh_per_mi)
                        * fuel_props.kwh_per_gge()
                        * (1. - max_epa_adj),
                )) * fuel_props.kwh_per_gge()
                    / DEFAULT_CHG_EFF;
            label_fe.adj_comb_kwh_per_mi =
                0.55 * label_fe.adj_udds_kwh_per_mi + 0.45 * label_fe.adj_hwy_kwh_per_mi;

            label_fe.adj_udds_ess_kwh_per_mi = label_fe.adj_udds_kwh_per_mi * DEFAULT_CHG_EFF;
            label_fe.adj_hwy_ess_kwh_per_mi = label_fe.adj_hwy_kwh_per_mi * DEFAULT_CHG_EFF;
            label_fe.adj_comb_ess_kwh_per_mi = label_fe.adj_comb_kwh_per_mi * DEFAULT_CHG_EFF;

            // range for combined city/highway
            // Get energy capacity from the proper powertrain
            if let PowertrainType::BatteryElectricVehicle(bev) = &veh.pt_type {
                label_fe.net_range_miles = bev.res.energy_capacity.get::<si::kilowatt_hour>()
                    / label_fe.adj_comb_ess_kwh_per_mi;
            }
        }

        // utility factor (percent driving in PHEV charge depletion mode)
        label_fe.uf = 0.;
    } else {
        // PHEV
        let phev_calcs = get_label_fe_phev(
            veh,
            phev_utilization_params,
            adj_params,
            max_epa_adj,
            &fuel_props,
        )?;
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
        label_fe.uf = phev_utilization_params.uf_array[first_grtr(
            &phev_utilization_params.rechg_freq_miles,
            0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles,
        )
        .with_context(|| format_dbg!())?
            - 1];

        label_fe.net_phev_cd_miles =
            Some(0.55 * phev_calcs.udds.adj_cd_miles + 0.45 * phev_calcs.hwy.adj_cd_miles);

        // For PHEVs, calculate net range as the sum of CD range and CS range
        if let PowertrainType::PlugInHybridElectricVehicle(phev) = &veh.pt_type {
            // Get CS range by determining how much fuel energy remains after depleting the battery
            let fuel_energy_kwh = phev.fs.energy_capacity.get::<si::kilowatt_hour>();
            let fuel_energy_gge = fuel_energy_kwh / fuel_props.kwh_per_gge();

            label_fe.net_range_miles = (fuel_energy_gge
                - label_fe.net_phev_cd_miles.with_context(|| format_dbg!())?
                    / label_fe.adj_cd_comb_mpgge.with_context(|| format_dbg!())?)
                * label_fe.adj_cs_comb_mpgge.with_context(|| format_dbg!())?
                + label_fe.net_phev_cd_miles.with_context(|| format_dbg!())?;
        }
    }

    // run accelerating sim_drive
    let mut sd_accel = SimDrive::new(veh.clone(), cyc["accel"].clone(), None);
    label_fe.net_accel = get_0_to_60_time(&mut sd_accel)
        .with_context(|| format!("`get_0_to_60_time`: {}", format_dbg!()))?;
    sd.insert("accel", sd_accel);

    // success Boolean -- did all of the tests work(e.g. met trace within ~2 mph)?
    label_fe.res_found = String::from("model needs to be implemented for this");

    if full_detail && verbose {
        println!("{label_fe:#?}");
        Ok((label_fe, Some(sd)))
    } else if full_detail {
        Ok((label_fe, Some(sd)))
    } else if verbose {
        println!("{label_fe:#?}");
        Ok((label_fe, None))
    } else {
        Ok((label_fe, None))
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "get_label_fe")]
#[cfg_attr(
    feature = "pyo3",
    pyo3(signature = (
        veh, max_epa_adj=None, full_detail=None, fuel_props=None, phev_utilization_params=None, verbose=None))
)]
/// pyo3 version of [get_label_fe]
pub fn get_label_fe_py(
    veh: &mut Vehicle,
    max_epa_adj: Option<f64>,
    full_detail: Option<bool>,
    fuel_props: Option<FuelProperties>,
    phev_utilization_params: Option<PhevUtilizationParams>,
    verbose: Option<bool>,
) -> anyhow::Result<LabelFe> {
    let (label_fe, _) = get_label_fe(
        veh,
        max_epa_adj,
        full_detail.unwrap_or_default(),
        fuel_props,
        phev_utilization_params,
        verbose.unwrap_or_default(),
    )?;
    Ok(label_fe)
}

/// PHEV-specific function for label fe.
///
/// # Arguments
/// - max_epa_adj: maximum EPA adjustment factor
///
/// # Returns
/// label fuel economy values for PHEV as a struct.
pub fn get_label_fe_phev(
    veh: &Vehicle,
    phev_utilization_params: &PhevUtilizationParams,
    adj_params: &AdjCoef,
    max_epa_adj: f64,
    fuel_props: &FuelProperties,
) -> anyhow::Result<LabelFePHEV> {
    // Get access to the PHEV powertrain
    let max_soc: si::Ratio;
    let min_soc: si::Ratio;
    let phev_max_regen: si::Ratio;
    let veh_mass: si::Mass;
    // equivalent to fastsim-2 `mc_peak_eff`
    let em_peak_eff: si::Ratio;
    // battery total energy capacity from soc of 1.0 to 0.0
    let energy_capacity: si::Energy;
    let chg_eff: f64;

    if let PowertrainType::PlugInHybridElectricVehicle(phev) = &veh.pt_type {
        max_soc = phev.res.max_soc;
        min_soc = phev.res.min_soc;
        phev_max_regen = 0.98 * uc::R;
        veh_mass = *veh.state.mass.get_fresh(|| format_dbg!())?;
        em_peak_eff = *phev
            .em
            .eff_interp_achieved
            .max()
            .with_context(|| format_dbg!())?
            * uc::R;
        energy_capacity = phev.res.energy_capacity;
        chg_eff = DEFAULT_CHG_EFF; // Use default charging efficiency
    } else {
        bail!("Vehicle is not a PHEV");
    }

    let mut label_fe_phev = LabelFePHEV {
        regen_soc_buffer: ((0.5 * veh_mass * ((60. * uc::MPH).powi(P2::new())))
            * phev_max_regen
            * em_peak_eff
            / energy_capacity)
            .min((max_soc - min_soc) / 2.0),
        ..Default::default()
    };

    // Create SimDrive objects for PHEV calculations
    let mut sd: HashMap<&str, SimDrive> = HashMap::new();
    sd.insert(
        "udds",
        SimDrive::new(veh.clone(), Cycle::from_resource("udds.csv", false)?, None),
    );
    sd.insert(
        "hwy",
        SimDrive::new(veh.clone(), Cycle::from_resource("hwfet.csv", false)?, None),
    );

    // charge sustaining behavior
    for (key, sd) in sd.iter_mut() {
        // do PHEV soc iteration
        // This runs 1 cycle starting at max SOC then runs 1 cycle starting at min SOC.
        // By assuming that the battery SOC depletion per mile is constant across cycles,
        // the first cycle can be extrapolated until charge sustaining kicks in.
        sd.walk()?;
        let mut phev_calc = PHEVCycleCalc::default();

        // charge depletion cycle has already been simulated
        // charge depletion battery kW-hr
        phev_calc.cd_ess_kwh = ((max_soc - min_soc) * energy_capacity).get::<si::kilowatt_hour>();

        // Get SOC and distance values
        let res = sd.veh.res().with_context(|| format_dbg!())?;
        let soc_start = *res
            .history
            .soc
            .first()
            .with_context(|| format_dbg!())?
            .get_fresh(|| format_dbg!())?;
        let soc_end = *res
            .history
            .soc
            .last()
            .with_context(|| format_dbg!())?
            .get_fresh(|| format_dbg!())?;
        let dist_mi = sd
            .veh
            .state
            .dist
            .get_fresh(|| format_dbg!())?
            .get::<si::mile>();

        // SOC change during 1 cycle
        phev_calc.delta_soc = soc_start - soc_end;
        // total number of miles in charge depletion mode, assuming constant kWh_per_mi
        phev_calc.total_cd_miles = ((max_soc - min_soc) * energy_capacity)
            .get::<si::kilowatt_hour>()
            / sd.veh.kwh_per_mi()?;
        // number of cycles in charge depletion mode, up to transition
        phev_calc.cd_cycs = phev_calc.total_cd_miles / dist_mi;
        // fraction of transition cycle spent in charge depletion
        phev_calc.cd_frac_in_trans = phev_calc.cd_cycs % phev_calc.cd_cycs.floor();

        // charge depletion fuel gallons - get from fuel converter
        let fuel_energy_kwh = if let Some(fc) = sd.veh.fc() {
            fc.state
                .energy_fuel
                .get_fresh(|| format_dbg!())?
                .get::<si::kilowatt_hour>()
        } else {
            0.0
        };
        phev_calc.cd_fs_gal = fuel_energy_kwh / fuel_props.kwh_per_gge();
        phev_calc.cd_fs_kwh = fuel_energy_kwh;
        phev_calc.cd_ess_kwh_per_mi = sd.veh.kwh_per_mi()?;
        phev_calc.cd_mpg = sd.veh.mpg(fuel_props.energy_density)?;

        // utility factor calculation for last charge depletion iteration and transition iteration
        // ported from excel
        let interp_x_vals: Vec<f64> = (0..((phev_calc.cd_cycs.ceil() + 1.0) as usize))
            .map(|i| i as f64 * dist_mi)
            .collect();

        phev_calc.lab_iter_uf = vec![];
        for x in interp_x_vals {
            phev_calc.lab_iter_uf.push(
                phev_utilization_params.uf_array[first_grtr(
                    &phev_utilization_params.rechg_freq_miles,
                    x,
                )
                .with_context(|| format_dbg!())?
                    - 1],
            );
        }

        // transition cycle
        phev_calc.trans_init_soc = max_soc - phev_calc.cd_cycs.floor() * phev_calc.delta_soc;

        // run the transition cycle by setting initial SOC
        let res_mut = sd.veh.res_mut().with_context(|| format_dbg!())?;
        res_mut.state.soc.mark_stale();
        res_mut
            .state
            .soc
            .update(phev_calc.trans_init_soc, || format_dbg!())?;
        sd.walk()?;

        // charge depletion battery kW-hr
        phev_calc.trans_ess_kwh =
            phev_calc.cd_ess_kwh_per_mi * dist_mi * phev_calc.cd_frac_in_trans;
        phev_calc.trans_ess_kwh_per_mi = phev_calc.cd_ess_kwh_per_mi * phev_calc.cd_frac_in_trans;

        // charge sustaining
        // the 0.01 is here to be consistent with Excel
        let init_soc = min_soc + 0.01 * uc::R;
        let res_mut = sd.veh.res_mut().with_context(|| format_dbg!())?;
        res_mut.state.soc.mark_stale();
        res_mut.state.soc.update(init_soc, || format_dbg!())?;
        sd.walk()?;

        // charge sustaining fuel gallons
        let cs_fuel_energy_kwh = if let Some(fc) = sd.veh.fc() {
            fc.state
                .energy_fuel
                .get_fresh(|| format_dbg!())?
                .get::<si::kilowatt_hour>()
        } else {
            0.0
        };
        phev_calc.cs_fs_gal = cs_fuel_energy_kwh / fuel_props.kwh_per_gge();
        // charge depletion fuel gallons, dependent on phev_calc.trans_fs_gal
        phev_calc.trans_fs_gal = phev_calc.cs_fs_gal * (1.0 - phev_calc.cd_frac_in_trans);
        phev_calc.cs_fs_kwh = cs_fuel_energy_kwh;
        phev_calc.trans_fs_kwh = phev_calc.cs_fs_kwh * (1.0 - phev_calc.cd_frac_in_trans);
        // charge sustaining battery kW-hr
        let cs_ess_energy_kwh = if let Some(res) = sd.veh.res() {
            res.state
                .energy_out_chemical
                .get_fresh(|| format_dbg!())?
                .get::<si::kilowatt_hour>()
        } else {
            0.0
        };
        phev_calc.cs_ess_kwh = cs_ess_energy_kwh;
        phev_calc.cs_ess_kwh_per_mi = sd.veh.kwh_per_mi()?;

        let lab_iter_uf_diff = phev_calc.lab_iter_uf.diff();
        phev_calc.lab_uf_gpm = [
            phev_calc.trans_fs_gal * lab_iter_uf_diff.last().with_context(|| format_dbg!())?,
            phev_calc.cs_fs_gal
                * (1.0
                    - phev_calc
                        .lab_iter_uf
                        .last()
                        .with_context(|| format_dbg!())?),
        ]
        .iter()
        .map(|x| *x / dist_mi)
        .collect();

        phev_calc.cd_mpg = sd.veh.mpg(fuel_props.energy_density)?;

        // city and highway cycle ranges
        let min_soc_in_cycle = phev_calc.delta_soc.abs(); // Use delta_soc as proxy for min SOC change
        phev_calc.cd_miles =
            if (max_soc - label_fe_phev.regen_soc_buffer - min_soc_in_cycle) < 0.01 * uc::R {
                1000.0
            } else {
                phev_calc.cd_cycs.ceil() * dist_mi
            };
        phev_calc.cd_lab_mpg = phev_calc
            .lab_iter_uf
            .last()
            .with_context(|| format_dbg!())?
            / (phev_calc.trans_fs_gal / dist_mi);

        // charge sustaining
        phev_calc.cs_mpg = dist_mi / phev_calc.cs_fs_gal;

        phev_calc.lab_uf = phev_utilization_params.uf_array[first_grtr(
            &phev_utilization_params.rechg_freq_miles,
            phev_calc.cd_miles,
        )
        .with_context(|| format_dbg!())?
            - 1];

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
        phev_calc.lab_iter_kwh_per_mi = lab_iter_kwh_per_mi_vals;

        let uf_diff = phev_calc.lab_iter_uf.diff();
        let mut vals = Vec::new();
        vals.push(0.0);
        for i in 1..phev_calc.lab_iter_kwh_per_mi.len() - 1 {
            if i - 1 < uf_diff.len() {
                vals.push(phev_calc.lab_iter_kwh_per_mi[i] * uf_diff[i - 1]);
            }
        }
        vals.push(0.0);
        phev_calc.lab_iter_uf_kwh_per_mi = vals;

        phev_calc.lab_kwh_per_mi = phev_calc
            .lab_iter_uf_kwh_per_mi
            .iter()
            .fold(0.0, |acc, x| acc + x)
            / phev_calc
                .lab_iter_uf
                .iter()
                .fold(0.0f64, |acc, x| acc.max(*x));

        let mut adj_iter_mpgge_vals = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        let mut adj_iter_kwh_per_mi_vals = vec![0.0; phev_calc.lab_iter_kwh_per_mi.len()];
        if *key == "udds" {
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd
                            .veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                            / (phev_calc.trans_fs_kwh / fuel_props.kwh_per_gge())))),
                sd.veh
                    .state
                    .dist
                    .get_fresh(|| format_dbg!())?
                    .get::<si::mile>()
                    / (phev_calc.trans_fs_kwh / fuel_props.kwh_per_gge())
                    * (1.0 - max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.city_intercept
                    + (adj_params.city_slope
                        / (sd
                            .veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                            / (phev_calc.cs_fs_kwh / fuel_props.kwh_per_gge())))),
                sd.veh
                    .state
                    .dist
                    .get_fresh(|| format_dbg!())?
                    .get::<si::mile>()
                    / (phev_calc.cs_fs_kwh / fuel_props.kwh_per_gge())
                    * (1.0 - max_epa_adj),
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
                                        * fuel_props.kwh_per_gge()))),
                            (1.0 - max_epa_adj)
                                * ((1.0 / phev_calc.lab_iter_kwh_per_mi[c])
                                    * fuel_props.kwh_per_gge()),
                        )) * fuel_props.kwh_per_gge();
                }
            }
        } else {
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd
                            .veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                            / (phev_calc.trans_fs_kwh / fuel_props.kwh_per_gge())))),
                sd.veh
                    .state
                    .dist
                    .get_fresh(|| format_dbg!())?
                    .get::<si::mile>()
                    / (phev_calc.trans_fs_kwh / fuel_props.kwh_per_gge())
                    * (1.0 - max_epa_adj),
            ));
            adj_iter_mpgge_vals.push(f64::max(
                1.0 / (adj_params.hwy_intercept
                    + (adj_params.hwy_slope
                        / (sd
                            .veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                            / (phev_calc.cs_fs_kwh / fuel_props.kwh_per_gge())))),
                sd.veh
                    .state
                    .dist
                    .get_fresh(|| format_dbg!())?
                    .get::<si::mile>()
                    / (phev_calc.cs_fs_kwh / fuel_props.kwh_per_gge())
                    * (1.0 - max_epa_adj),
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
                                        * fuel_props.kwh_per_gge()))),
                            (1.0 - max_epa_adj)
                                * ((1.0 / phev_calc.lab_iter_kwh_per_mi[c])
                                    * fuel_props.kwh_per_gge()),
                        )) * fuel_props.kwh_per_gge();
                }
            }
        }
        phev_calc.adj_iter_mpgge = adj_iter_mpgge_vals;
        phev_calc.adj_iter_kwh_per_mi = adj_iter_kwh_per_mi_vals;

        phev_calc.adj_iter_cd_miles = vec![0.0; phev_calc.cd_cycs.ceil() as usize + 2];
        for c in 0..phev_calc.adj_iter_cd_miles.len() {
            if c == 0 {
                phev_calc.adj_iter_cd_miles[c] = 0.0;
            } else if c <= phev_calc.cd_cycs.floor() as usize {
                phev_calc.adj_iter_cd_miles[c] = phev_calc.adj_iter_cd_miles[c - 1]
                    + phev_calc.cd_ess_kwh_per_mi
                        * sd.veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                        / phev_calc.adj_iter_kwh_per_mi[c];
            } else if c == phev_calc.cd_cycs.floor() as usize + 1 {
                phev_calc.adj_iter_cd_miles[c] = phev_calc.adj_iter_cd_miles[c - 1]
                    + phev_calc.trans_ess_kwh_per_mi
                        * sd.veh
                            .state
                            .dist
                            .get_fresh(|| format_dbg!())?
                            .get::<si::mile>()
                        / phev_calc.adj_iter_kwh_per_mi[c];
            } else {
                phev_calc.adj_iter_cd_miles[c] = 0.0;
            }
        }

        let mut soc_hist: Vec<f64> = vec![];
        for soc in sd
            .veh
            .res()
            .with_context(|| format_dbg!())?
            .history
            .soc
            .clone()
        {
            soc_hist.push(soc.get_fresh(|| format_dbg!())?.get::<si::ratio>());
        }

        phev_calc.adj_cd_miles =
            if max_soc - label_fe_phev.regen_soc_buffer - (*soc_hist.min()? * uc::R) < 0.01 * uc::R
            {
                1000.0
            } else {
                *phev_calc.adj_iter_cd_miles.max()?
            };

        // utility factor calculation for last charge depletion iteration and transition iteration
        // ported from excel

        phev_calc.adj_iter_uf = vec![];
        for x in phev_calc.adj_iter_cd_miles.clone() {
            phev_calc.adj_iter_uf.push(
                phev_utilization_params.uf_array[first_grtr(
                    &phev_utilization_params.rechg_freq_miles,
                    x,
                )
                .with_context(|| format_dbg!())?
                    - 1],
            )
        }

        let adj_iter_uf_diff = phev_calc.adj_iter_uf.diff();
        phev_calc.adj_iter_uf_gpm = vec![0.0; phev_calc.cd_cycs.floor() as usize];
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc.adj_iter_mpgge[phev_calc.adj_iter_mpgge.len() - 2])
                * adj_iter_uf_diff[adj_iter_uf_diff.len() - 2],
        );
        phev_calc.adj_iter_uf_gpm.push(
            (1.0 / phev_calc
                .adj_iter_mpgge
                .last()
                .with_context(|| format_dbg!())?)
                * (1.0 - phev_calc.adj_iter_uf[phev_calc.adj_iter_uf.len() - 2]),
        );

        let adj_uf_diff = phev_calc.adj_iter_uf.diff();
        phev_calc.adj_iter_uf_kwh_per_mi = phev_calc
            .adj_iter_kwh_per_mi
            .iter()
            .zip(adj_uf_diff.iter())
            .map(|(kwh, uf)| kwh * uf)
            .collect();

        let max_uf: f64 = phev_calc
            .adj_iter_uf
            .iter()
            .fold(0.0f64, |acc, x| acc.max(*x));
        phev_calc.adj_cd_mpgge =
            1.0 / phev_calc.adj_iter_uf_gpm[phev_calc.adj_iter_uf_gpm.len() - 2] * max_uf;
        phev_calc.adj_cs_mpgge = 1.0
            / phev_calc
                .adj_iter_uf_gpm
                .last()
                .with_context(|| format_dbg!())?
            * (1.0 - max_uf);

        phev_calc.adj_uf = phev_utilization_params.uf_array[first_grtr(
            &phev_utilization_params.rechg_freq_miles,
            phev_calc.adj_cd_miles,
        )
        .with_context(|| format_dbg!())?
            - 1];

        phev_calc.adj_mpgge = 1.0
            / (phev_calc.adj_uf / phev_calc.adj_cd_mpgge
                + (1.0 - phev_calc.adj_uf) / phev_calc.adj_cs_mpgge);

        let uf_kwh_sum: f64 = phev_calc
            .adj_iter_uf_kwh_per_mi
            .iter()
            .fold(0.0, |acc, x| acc + x);
        phev_calc.adj_kwh_per_mi = uf_kwh_sum / max_uf / chg_eff;

        phev_calc.adj_ess_kwh_per_mi = uf_kwh_sum / max_uf;

        match *key {
            "udds" => label_fe_phev.udds = phev_calc.clone(),
            "hwy" => label_fe_phev.hwy = phev_calc.clone(),
            &_ => bail!("No field for cycle {}", key),
        };
    }

    Ok(label_fe_phev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::vehicle_model::f3veh_with_f2_eff;

    // /// Test that label FE calculations for conventional vehicles match FASTSim-2 results
    // #[test]
    // #[cfg(all(feature = "resources", feature = "yaml"))]
    // fn test_label_fe_conv_vs_fastsim2() {
    //     let file_contents = include_str!("vehicle/fastsim-2_2012_Ford_Fusion.yaml");
    //     use fastsim_2::traits::SerdeAPI;
    //     let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
    //     let mut veh = Vehicle::try_from(f2veh.clone()).unwrap();
    //     f3veh_with_f2_eff(&f2veh, &mut veh);

    //     // Get FASTSim-3 label FE results
    //     let (label_fe_f3, _) = get_label_fe(&mut veh, None, false, None, None, false)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     // Get FASTSim-2 label FE results
    //     let (label_fe_f2, _) = fastsim_2::simdrivelabel::get_label_fe(&f2veh.clone(), None, None)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     // Compare key results (allowing for small numerical differences)
    //     let tolerance = 0.01; // 1% tolerance

    //     // Check MPGe values
    //     assert!(
    //         (label_fe_f3.lab_udds_mpgge - label_fe_f2.lab_udds_mpgge).abs()
    //             / label_fe_f2.lab_udds_mpgge
    //             < tolerance,
    //         "UDDS MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_udds_mpgge,
    //         label_fe_f2.lab_udds_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.lab_hwy_mpgge - label_fe_f2.lab_hwy_mpgge).abs()
    //             / label_fe_f2.lab_hwy_mpgge
    //             < tolerance,
    //         "Highway MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_hwy_mpgge,
    //         label_fe_f2.lab_hwy_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.lab_comb_mpgge - label_fe_f2.lab_comb_mpgge).abs()
    //             / label_fe_f2.lab_comb_mpgge
    //             < tolerance,
    //         "Combined MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_comb_mpgge,
    //         label_fe_f2.lab_comb_mpgge
    //     );

    //     // Check adjusted values
    //     assert!(
    //         (label_fe_f3.adj_udds_mpgge - label_fe_f2.adj_udds_mpgge).abs()
    //             / label_fe_f2.adj_udds_mpgge
    //             < tolerance,
    //         "Adjusted UDDS MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.adj_udds_mpgge,
    //         label_fe_f2.adj_udds_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.adj_comb_mpgge - label_fe_f2.adj_comb_mpgge).abs()
    //             / label_fe_f2.adj_comb_mpgge
    //             < tolerance,
    //         "Adjusted combined MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.adj_comb_mpgge,
    //         label_fe_f2.adj_comb_mpgge
    //     );

    //     println!("Conventional vehicle label FE test passed!");
    //     println!(
    //         "F3 Combined MPGe: {:.3}, F2: {:.3}",
    //         label_fe_f3.lab_comb_mpgge, label_fe_f2.lab_comb_mpgge
    //     );
    // }

    // /// Test that label FE calculations for BEV vehicles match FASTSim-2 results
    // #[test]
    // #[cfg(all(feature = "resources", feature = "yaml"))]
    // fn test_label_fe_bev_vs_fastsim2() {
    //     let file_contents = include_str!("vehicle/fastsim-2_2022_Renault_Zoe_ZE50_R135.yaml");
    //     use fastsim_2::traits::SerdeAPI;
    //     let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
    //     let mut veh = Vehicle::try_from(f2veh.clone()).unwrap();
    //     f3veh_with_f2_eff(&f2veh, &mut veh);

    //     // Get FASTSim-3 label FE results
    //     let (label_fe_f3, _) = get_label_fe(&mut veh, None, false, None, None, false)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     let (label_fe_f2, _) = fastsim_2::simdrivelabel::get_label_fe(&f2veh.clone(), None, None)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     let tolerance = 0.011; // 1.1% tolerance

    //     // For BEV, check kWh/mi values instead of MPGe
    //     assert!(
    //         (label_fe_f3.lab_udds_kwh_per_mi - label_fe_f2.lab_udds_kwh_per_mi).abs()
    //             / label_fe_f2.lab_udds_kwh_per_mi
    //             < tolerance,
    //         "UDDS kWh/mi mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_udds_kwh_per_mi,
    //         label_fe_f2.lab_udds_kwh_per_mi
    //     );

    //     assert!(
    //         (label_fe_f3.lab_comb_kwh_per_mi - label_fe_f2.lab_comb_kwh_per_mi).abs()
    //             / label_fe_f2.lab_comb_kwh_per_mi
    //             < tolerance,
    //         "Combined kWh/mi mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_comb_kwh_per_mi,
    //         label_fe_f2.lab_comb_kwh_per_mi
    //     );

    //     println!("BEV label FE test passed!");
    //     println!(
    //         "F3 Combined kWh/mi: {:.3}, F2: {:.3}",
    //         label_fe_f3.lab_comb_kwh_per_mi, label_fe_f2.lab_comb_kwh_per_mi
    //     );
    // }

    // /// Test that label FE calculations for HEV vehicles match FASTSim-2 results
    // #[test]
    // #[cfg(all(feature = "resources", feature = "yaml"))]
    // fn test_label_fe_hev_vs_fastsim2() {
    //     let file_contents = include_str!("vehicle/fastsim-2_2016_TOYOTA_Prius_Two.yaml");
    //     use fastsim_2::traits::SerdeAPI;
    //     let f2veh = fastsim_2::vehicle::RustVehicle::from_yaml(file_contents, false).unwrap();
    //     let mut veh = Vehicle::try_from(f2veh.clone()).unwrap();
    //     f3veh_with_f2_eff(&f2veh, &mut veh);

    //     // Get FASTSim-3 label FE results
    //     let (label_fe_f3, _) = get_label_fe(&mut veh, None, false, None, None, false)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     let (label_fe_f2, _) = fastsim_2::simdrivelabel::get_label_fe(&f2veh, None, None)
    //         .with_context(|| format_dbg!())
    //         .unwrap();

    //     let tolerance = 0.1; // Temporarily increased tolerance to 10% for debugging

    //     // Check MPGe values for HEV
    //     assert!(
    //         (label_fe_f3.lab_udds_mpgge - label_fe_f2.lab_udds_mpgge).abs()
    //             / label_fe_f2.lab_udds_mpgge
    //             < tolerance,
    //         "UDDS MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_udds_mpgge,
    //         label_fe_f2.lab_udds_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.lab_comb_mpgge - label_fe_f2.lab_comb_mpgge).abs()
    //             / label_fe_f2.lab_comb_mpgge
    //             < tolerance,
    //         "Combined MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_comb_mpgge,
    //         label_fe_f2.lab_comb_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.lab_hwy_mpgge - label_fe_f2.lab_hwy_mpgge).abs()
    //             / label_fe_f2.lab_hwy_mpgge
    //             < tolerance,
    //         "Hwy MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.lab_hwy_mpgge,
    //         label_fe_f2.lab_hwy_mpgge
    //     );

    //     assert!(
    //         (label_fe_f3.net_accel - label_fe_f2.net_accel).abs() / label_fe_f2.net_accel
    //             < tolerance,
    //         "Hwy MPGe mismatch: F3={:.3}, F2={:.3}",
    //         label_fe_f3.net_accel,
    //         label_fe_f2.net_accel
    //     );
    // }

    /// Test that creates a mock PHEV vehicle from FASTSim-2 data and compares label FE calculations
    #[test]
    #[cfg(all(feature = "resources", feature = "yaml"))]
    fn test_label_fe_phev_vs_fastsim2() {
        // Load a PHEV vehicle from the calibration directory (FASTSim-2 format)
        let f2_veh_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .with_context(|| format_dbg!())
            .unwrap()
            .join("cal_and_val/f2-vehicles/2016 CHEVROLET Volt.yaml");

        if !f2_veh_path.exists() {
            println!("PHEV vehicle file not found, skipping test");
            return;
        }

        let veh_contents = std::fs::read_to_string(&f2_veh_path)
            .with_context(|| format_dbg!())
            .unwrap();

        // Load FASTSim-2 vehicle and convert to FASTSim-3
        let f2_veh: fastsim_2::vehicle::RustVehicle =
            fastsim_2::traits::SerdeAPI::from_yaml(&veh_contents, false)
                .with_context(|| format_dbg!())
                .unwrap();
        assert!(f2_veh.veh_pt_type == fastsim_2::vehicle::PHEV);
        let mut veh = Vehicle::try_from(f2_veh.clone())
            .with_context(|| format_dbg!())
            .unwrap();
        assert!(
            veh.pt_type.is_plug_in_hybrid_electric_vehicle(),
            "`veh.pt_type.variant_as_str()`: {}\n`f2_veh.veh_pt_type`: {}",
            veh.pt_type.variant_as_str(),
            f2_veh.veh_pt_type
        );

        // Get FASTSim-3 label FE results (if PHEV functionality is implemented)
        let label_fe_f3 = get_label_fe(&mut veh, None, false, None, None, false)
            .unwrap()
            .0;

        // Get FASTSim-2 label FE results
        let label_fe_f2 = fastsim_2::simdrivelabel::get_label_fe(&f2_veh, None, None)
            .unwrap()
            .0;

        let tolerance = 0.05; // 5% tolerance for PHEV (more complex calculations)

        // Check MPGe values for HEV
        assert!(
            (label_fe_f3.lab_udds_mpgge - label_fe_f2.lab_udds_mpgge).abs()
                / label_fe_f2.lab_udds_mpgge
                < tolerance,
            "UDDS MPGe mismatch: F3={:.3}, F2={:.3}",
            label_fe_f3.lab_udds_mpgge,
            label_fe_f2.lab_udds_mpgge
        );

        assert!(
            (label_fe_f3.lab_comb_mpgge - label_fe_f2.lab_comb_mpgge).abs()
                / label_fe_f2.lab_comb_mpgge
                < tolerance,
            "Combined MPGe mismatch: F3={:.3}, F2={:.3}",
            label_fe_f3.lab_comb_mpgge,
            label_fe_f2.lab_comb_mpgge
        );

        assert!(
            (label_fe_f3.lab_hwy_mpgge - label_fe_f2.lab_hwy_mpgge).abs()
                / label_fe_f2.lab_hwy_mpgge
                < tolerance,
            "Hwy MPGe mismatch: F3={:.3}, F2={:.3}",
            label_fe_f3.lab_hwy_mpgge,
            label_fe_f2.lab_hwy_mpgge
        );

        assert!(
            (label_fe_f3.net_accel - label_fe_f2.net_accel).abs() / label_fe_f2.net_accel
                < tolerance,
            "Hwy MPGe mismatch: F3={:.3}, F2={:.3}",
            label_fe_f3.net_accel,
            label_fe_f2.net_accel
        );
    }
}
