//! Module containing vehicle struct and related functions.

// local
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::{add_pyo3_api, ApproxEq};
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;

pub const CONV: &str = "Conv";
pub const HEV: &str = "HEV";
pub const PHEV: &str = "PHEV";
pub const BEV: &str = "BEV";
pub const VEH_PT_TYPES: [&str; 4] = [CONV, HEV, PHEV, BEV];

pub const SI: &str = "SI";
pub const ATKINSON: &str = "Atkinson";
pub const DIESEL: &str = "Diesel";
pub const H2FC: &str = "H2FC";
pub const HD_DIESEL: &str = "HD_Diesel";

pub const FC_EFF_TYPES: [&str; 5] = [SI, ATKINSON, DIESEL, H2FC, HD_DIESEL];

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq)]
#[add_pyo3_api(
    #[pyo3(name = "set_veh_mass")]
    pub fn set_veh_mass_py(&mut self) {
        self.set_veh_mass()
    }

    pub fn get_max_regen_kwh(&self) -> f64 {
        self.max_regen_kwh()
    }

    #[getter]
    pub fn get_mc_peak_eff(&self) -> f64 {
        self.mc_peak_eff()
    }

    // TODO: refactor this to have a non-py and `_py` version
    #[setter]
    pub fn set_mc_peak_eff(&mut self, new_peak: f64) {
        let mc_max_eff = ndarrmax(&self.mc_eff_array);
        self.mc_eff_array *= new_peak / mc_max_eff;
        let mc_max_full_eff = arrmax(&self.mc_full_eff_array);
        self.mc_full_eff_array = self
            .mc_full_eff_array
            .iter()
            .map(|e: &f64| -> f64 { e * (new_peak / mc_max_full_eff) })
            .collect();
    }

    #[getter]
    pub fn get_max_fc_eff_kw(&self) -> f64 {
        self.max_fc_eff_kw()
    }

    #[getter]
    pub fn get_fc_peak_eff(&self) -> f64 {
        self.fc_peak_eff()
    }

    #[setter]
    pub fn set_fc_peak_eff(&mut self, new_peak: f64) {
        let old_fc_peak_eff = self.fc_peak_eff();
        let multiplier = new_peak / old_fc_peak_eff;
        self.fc_eff_array = self
            .fc_eff_array
            .iter()
            .map(|eff: &f64| -> f64 { eff * multiplier })
            .collect();
        let new_fc_peak_eff = self.fc_peak_eff();
        let eff_map_multiplier = new_peak / new_fc_peak_eff;
        self.fc_eff_map = self
            .fc_eff_map
            .map(|eff| -> f64 { eff * eff_map_multiplier });
    }

    #[pyo3(name = "set_derived")]
    pub fn set_derived_py(&mut self) {
        self.set_derived()
    }

    /// An identify function to allow RustVehicle to be used as a python vehicle and respond to this method
    /// Returns a clone of the current object
    pub fn to_rust(&self) -> PyResult<Self> {
        Ok(self.clone())
    }
)]
/// Struct containing vehicle attributes
/// # Python Examples
/// ```python
/// import fastsim
///
/// ## Load drive cycle by name
/// cyc_py = fastsim.cycle.Cycle.from_file("udds")
/// cyc_rust = cyc_py.to_rust()
/// ```
pub struct RustVehicle {
    #[serde(skip)]
    #[api(has_orphaned)]
    /// Physical properties, see [RustPhysicalProperties](RustPhysicalProperties)
    pub props: RustPhysicalProperties,
    /// Vehicle name
    #[serde(alias = "name")]
    pub scenario_name: String,
    /// Vehicle database ID
    #[serde(skip)]
    pub selection: u32,
    /// Vehicle year
    #[serde(alias = "vehModelYear")]
    pub veh_year: u32,
    /// Vehicle powertrain type, one of \[[CONV](CONV), [HEV](HEV), [PHEV](PHEV), [BEV](BEV)\]
    #[serde(alias = "vehPtType")]
    pub veh_pt_type: String,
    /// Aerodynamic drag coefficient
    #[serde(alias = "dragCoef")]
    pub drag_coef: f64,
    /// Frontal area, $m^2$
    #[serde(alias = "frontalAreaM2")]
    pub frontal_area_m2: f64,
    /// Vehicle mass excluding cargo, passengers, and powertrain components, $kg$
    #[serde(alias = "gliderKg")]
    pub glider_kg: f64,
    /// Vehicle center of mass height, $m$  
    /// **NOTE:** positive for FWD, negative for RWD, AWD, 4WD
    #[serde(alias = "vehCgM")]
    pub veh_cg_m: f64,
    /// Fraction of weight on the drive axle while stopped
    #[serde(alias = "driveAxleWeightFrac")]
    pub drive_axle_weight_frac: f64,
    /// Wheelbase, $m$
    #[serde(alias = "wheelBaseM")]
    pub wheel_base_m: f64,
    /// Cargo mass including passengers, $kg$
    #[serde(alias = "cargoKg")]
    pub cargo_kg: f64,
    /// Total vehicle mass, overrides mass calculation, $kg$
    #[serde(alias = "vehOverrideKg")]
    pub veh_override_kg: Option<f64>,
    /// Component mass multiplier for vehicle mass calculation
    #[serde(alias = "compMassMultiplier")]
    pub comp_mass_multiplier: f64,
    /// Fuel storage max power output, $kW$
    #[serde(alias = "maxFuelStorKw")]
    pub fs_max_kw: f64,
    /// Fuel storage time to peak power, $s$
    #[serde(alias = "fuelStorSecsToPeakPwr")]
    pub fs_secs_to_peak_pwr: f64,
    /// Fuel storage energy capacity, $kWh$
    #[serde(alias = "fuelStorKwh")]
    pub fs_kwh: f64,
    /// Fuel specific energy, $\frac{kWh}{kg}$
    #[serde(alias = "fuelStorKwhPerKg")]
    pub fs_kwh_per_kg: f64,
    /// Fuel converter peak continuous power, $kW$
    #[serde(alias = "maxFuelConvKw")]
    pub fc_max_kw: f64,
    /// Fuel converter output power percentage map, x-values of [fc_eff_map](RustVehicle::fc_eff_map)
    #[serde(alias = "fcPwrOutPerc")]
    pub fc_pwr_out_perc: Array1<f64>,
    /// Fuel converter efficiency map
    #[serde(default)]
    pub fc_eff_map: Array1<f64>,
    /// Fuel converter efficiency type, one of \[[SI](SI), [ATKINSON](ATKINSON), [DIESEL](DIESEL), [H2FC](H2FC), [HD_DIESEL](HD_DIESEL)\]  
    /// Used for calculating [fc_eff_map](RustVehicle::fc_eff_map), and other calculations if H2FC
    #[serde(alias = "fcEffType")]
    pub fc_eff_type: String,
    /// Fuel converter time to peak power, $s$
    #[serde(alias = "fuelConvSecsToPeakPwr")]
    pub fc_sec_to_peak_pwr: f64,
    /// Fuel converter base mass, $kg$
    #[serde(alias = "fuelConvBaseKg")]
    pub fc_base_kg: f64,
    /// Fuel converter specific power (power-to-weight ratio), $\frac{kW}{kg}$
    #[serde(alias = "fuelConvKwPerKg")]
    pub fc_kw_per_kg: f64,
    /// Minimum time fuel converter must be on before shutoff (for HEV, PHEV)
    #[serde(alias = "minFcTimeOn")]
    pub min_fc_time_on: f64,
    /// Fuel converter idle power, $kW$
    #[serde(alias = "idleFcKw")]
    pub idle_fc_kw: f64,
    /// Peak continuous electric motor power, $kW$
    #[serde(alias = "mcMaxElecInKw")]
    pub mc_max_kw: f64,
    /// Electric motor output power percentage map, x-values of [mc_eff_map](RustVehicle::mc_eff_map)
    #[serde(alias = "mcPwrOutPerc")]
    pub mc_pwr_out_perc: Array1<f64>,
    /// Electric motor efficiency map
    #[serde(alias = "mcEffArray")]
    pub mc_eff_map: Array1<f64>,
    /// Electric motor time to peak power, $s$
    #[serde(alias = "motorSecsToPeakPwr")]
    pub mc_sec_to_peak_pwr: f64,
    /// Motor power electronics mass per power output, $\frac{kg}{kW}$
    #[serde(alias = "mcPeKgPerKw")]
    pub mc_pe_kg_per_kw: f64,
    /// Motor power electronics base mass, $kg$
    #[serde(alias = "mcPeBaseKg")]
    pub mc_pe_base_kg: f64,
    /// Traction battery maximum power output, $kW$
    #[serde(alias = "maxEssKw")]
    pub ess_max_kw: f64,
    /// Traction battery energy capacity, $kWh$
    #[serde(alias = "maxEssKwh")]
    pub ess_max_kwh: f64,
    /// Traction battery mass per energy, $\frac{kg}{kWh}$
    #[serde(alias = "essKgPerKwh")]
    pub ess_kg_per_kwh: f64,
    /// Traction battery base mass, $kg$
    #[serde(alias = "essBaseKg")]
    pub ess_base_kg: f64,
    /// Traction battery round-trip efficiency
    #[serde(alias = "essRoundTripEff")]
    pub ess_round_trip_eff: f64,
    /// Traction battery cycle life coefficient A, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
    #[serde(alias = "essLifeCoefA")]
    pub ess_life_coef_a: f64,
    /// Traction battery cycle life coefficient B, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
    #[serde(alias = "essLifeCoefB")]
    pub ess_life_coef_b: f64,
    /// Traction battery minimum state of charge
    #[serde(alias = "minSoc")]
    pub min_soc: f64,
    /// Traction battery maximum state of charge
    #[serde(alias = "maxSoc")]
    pub max_soc: f64,
    /// ESS discharge effort toward max FC efficiency
    #[serde(alias = "essDischgToFcMaxEffPerc")]
    pub ess_dischg_to_fc_max_eff_perc: f64,
    /// ESS charge effort toward max FC efficiency
    #[serde(alias = "essChgToFcMaxEffPerc")]
    pub ess_chg_to_fc_max_eff_perc: f64,
    /// Mass moment of inertia per wheel, $kg \cdot m^2$
    #[serde(alias = "wheelInertiaKgM2")]
    pub wheel_inertia_kg_m2: f64,
    /// Number of wheels
    #[serde(alias = "numWheels")]
    pub num_wheels: f64,
    /// Rolling resistance coefficient
    #[serde(alias = "wheelRrCoef")]
    pub wheel_rr_coef: f64,
    /// Wheel radius, $m$
    #[serde(alias = "wheelRadiusM")]
    pub wheel_radius_m: f64,
    /// Wheel coefficient of friction
    #[serde(alias = "wheelCoefOfFric")]
    pub wheel_coef_of_fric: f64,
    /// Speed where the battery reserved for accelerating is zero
    #[serde(alias = "maxAccelBufferMph")]
    pub max_accel_buffer_mph: f64,
    /// Percent of usable battery energy reserved to help accelerate
    #[serde(alias = "maxAccelBufferPercOfUseableSoc")]
    pub max_accel_buffer_perc_of_useable_soc: f64,
    /// Percent SOC buffer for high accessory loads during cycles with long idle time
    #[serde(alias = "percHighAccBuf")]
    pub perc_high_acc_buf: f64,
    /// Speed at which the fuel converter must turn on, $mph$
    #[serde(alias = "mphFcOn")]
    pub mph_fc_on: f64,
    /// Power demand above which to require fuel converter on, $kW$
    #[serde(alias = "kwDemandFcOn")]
    pub kw_demand_fc_on: f64,
    /// Maximum brake regeneration efficiency
    #[serde(alias = "maxRegen")]
    pub max_regen: f64,
    /// Stop/start micro-HEV flag
    pub stop_start: bool,
    /// Force auxiliary power load to come from fuel converter
    #[serde(alias = "forceAuxOnFC")]
    pub force_aux_on_fc: bool,
    /// Alternator efficiency
    #[serde(alias = "altEff")]
    pub alt_eff: f64,
    /// Charger efficiency
    #[serde(alias = "chgEff")]
    pub chg_eff: f64,
    /// Auxiliary load power, $kW$
    #[serde(alias = "auxKw")]
    pub aux_kw: f64,
    /// Transmission mass, $kg$
    #[serde(alias = "transKg")]
    pub trans_kg: f64,
    /// Transmission efficiency
    #[serde(alias = "transEff")]
    pub trans_eff: f64,
    /// Maximum acceptable overall change in ESS energy relative to energy from fuel (HEV SOC balancing only), $\frac{\Delta E_{ESS}}{\Delta E_{fuel}}$
    #[serde(alias = "essToFuelOkError")]
    pub ess_to_fuel_ok_error: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub small_motor_power_kw: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub large_motor_power_kw: f64,
    // this and other fixed-size arrays can probably be vectors
    // without any performance penalty with the current implementation
    // of the functions in utils.rs
    #[doc(hidden)]
    #[serde(skip)]
    pub fc_perc_out_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub regen_a: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub regen_b: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub charging_on: bool,
    #[doc(hidden)]
    #[serde(skip)]
    pub no_elec_sys: bool,
    #[doc(hidden)]
    // all of the parameters that are set in `set_derived` should be skipped by serde
    #[serde(skip)]
    pub no_elec_aux: bool,
    #[doc(hidden)]
    #[serde(skip)]
    pub max_roadway_chg_kw: Array1<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub input_kw_out_array: Array1<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub fc_kw_out_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(default)]
    #[serde(alias = "fcEffArray")]
    pub fc_eff_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub modern_max: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_eff_array: Array1<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_kw_in_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_kw_out_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_max_elec_in_kw: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_full_eff_array: Vec<f64>,
    #[doc(hidden)]
    #[serde(alias = "vehKg")]
    pub veh_kg: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub max_trac_mps2: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub ess_mass_kg: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_mass_kg: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub fc_mass_kg: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub fs_mass_kg: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub mc_perc_out_array: Vec<f64>,
    // these probably don't need to be in rust
    #[doc(hidden)]
    #[serde(skip)]
    pub val_udds_mpgge: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_hwy_mpgge: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_comb_mpgge: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_udds_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_hwy_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_comb_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_cd_range_mi: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_const65_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_const60_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_const55_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_const45_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_unadj_udds_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_unadj_hwy_kwh_per_mile: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val0_to60_mph: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_ess_life_miles: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_range_miles: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_veh_base_cost: f64,
    #[doc(hidden)]
    #[serde(skip)]
    pub val_msrp: f64,
    /// Fuel converter efficiency peak override, scales entire curve
    #[serde(skip)]
    pub fc_peak_eff_override: Option<f64>,
    /// Motor efficiency peak override, scales entire curve
    #[serde(skip)]
    pub mc_peak_eff_override: Option<f64>,
    #[serde(skip)]
    #[doc(hidden)]
    pub orphaned: bool,
}

/// RustVehicle rust methods
impl RustVehicle {
    /// Sets the following parameters:
    /// - `ess_mass_kg`
    /// - `mc_mass_kg`
    /// - `fc_mass_kg`
    /// - `fs_mass_kg`
    /// - `veh_kg`
    /// - `max_trac_mps2`
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn set_veh_mass(&mut self) {
        if self.veh_override_kg.is_none() {
            self.ess_mass_kg = if self.ess_max_kwh == 0.0 || self.ess_max_kw == 0.0 {
                0.0
            } else {
                ((self.ess_max_kwh * self.ess_kg_per_kwh) + self.ess_base_kg)
                    * self.comp_mass_multiplier
            };
            self.mc_mass_kg = if self.mc_max_kw == 0.0 {
                0.0
            } else {
                (self.mc_pe_base_kg + (self.mc_pe_kg_per_kw * self.mc_max_kw))
                    * self.comp_mass_multiplier
            };
            self.fc_mass_kg = if self.fc_max_kw == 0.0 {
                0.0
            } else {
                (1.0 / self.fc_kw_per_kg * self.fc_max_kw + self.fc_base_kg)
                    * self.comp_mass_multiplier
            };
            self.fs_mass_kg = if self.fs_max_kw == 0.0 {
                0.0
            } else {
                ((1.0 / self.fs_kwh_per_kg) * self.fs_kwh) * self.comp_mass_multiplier
            };
            self.veh_kg = self.cargo_kg
                + self.glider_kg
                + self.trans_kg * self.comp_mass_multiplier
                + self.ess_mass_kg
                + self.mc_mass_kg
                + self.fc_mass_kg
                + self.fs_mass_kg;
        } else {
            // if positive real number is specified for veh_override_kg, use that
            self.veh_kg = self.veh_override_kg.unwrap();
        }

        self.max_trac_mps2 = (self.wheel_coef_of_fric
            * self.drive_axle_weight_frac
            * self.veh_kg
            * self.props.a_grav_mps2
            / (1.0 + self.veh_cg_m * self.wheel_coef_of_fric / self.wheel_base_m))
            / (self.veh_kg * self.props.a_grav_mps2)
            * self.props.a_grav_mps2;
    }

    pub fn max_regen_kwh(&self) -> f64 {
        0.5 * self.veh_kg * (27.0 * 27.0) / (3_600.0 * 1_000.0)
    }

    pub fn mc_peak_eff(&self) -> f64 {
        arrmax(&self.mc_full_eff_array)
    }

    pub fn max_fc_eff_kw(&self) -> f64 {
        let fc_eff_arr_max_i =
            first_eq(&self.fc_eff_array, arrmax(&self.fc_eff_array)).unwrap_or(0);
        self.fc_kw_out_array[fc_eff_arr_max_i]
    }

    pub fn fc_peak_eff(&self) -> f64 {
        arrmax(&self.fc_eff_array)
    }

    /// Sets derived parameters:
    /// - `no_elec_sys`
    /// - `no_elec_aux`
    /// - `fc_perc_out_array`
    /// - `input_kw_out_array`
    /// - `fc_kw_out_array`
    /// - `fc_eff_array`
    /// - `modern_diff`
    /// - `large_baseline_eff_adj`
    /// - `mc_kw_adj_perc`
    /// - `mc_eff_map`
    /// - `mc_eff_array`
    /// - `mc_perc_out_array`
    /// - `mc_kw_out_array`
    /// - `mc_full_eff_array`
    /// - `mc_kw_in_array`
    /// - `mc_max_elec_in_kw`
    /// - `set_fc_peak_eff()`
    /// - `set_mc_peak_eff()`
    /// - `set_veh_mass()`
    ///     - `ess_mass_kg`
    ///     - `mc_mass_kg`
    ///     - `fc_mass_kg`
    ///     - `fs_mass_kg`
    ///     - `veh_kg`
    ///     - `max_trac_mps2`
    pub fn set_derived(&mut self) {
        if self.scenario_name != "Template Vehicle for setting up data types" {
            if self.veh_pt_type == BEV {
                assert!(
                    self.fs_max_kw == 0.0,
                    "max_fuel_stor_kw must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.fs_kwh == 0.0,
                    "fuel_stor_kwh must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.fc_max_kw == 0.0,
                    "max_fuel_conv_kw must be zero for provided BEV powertrain type in {}",
                    self.scenario_name
                );
            } else if (self.veh_pt_type == CONV) && !self.stop_start {
                assert!(
                    self.mc_max_kw == 0.0,
                    "max_mc_kw must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.ess_max_kw == 0.0,
                    "max_ess_kw must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
                assert!(
                    self.ess_max_kwh == 0.0,
                    "max_ess_kwh must be zero for provided Conv powertrain type in {}",
                    self.scenario_name
                );
            }
        }
        // ### Build roadway power lookup table
        // self.max_roadway_chg_kw = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // self.charging_on = false;

        // # Checking if a vehicle has any hybrid components
        if (self.ess_max_kwh == 0.0) || (self.ess_max_kw == 0.0) || (self.mc_max_kw == 0.0) {
            self.no_elec_sys = true;
        } else {
            self.no_elec_sys = false;
        }

        // # Checking if aux loads go through an alternator
        if self.no_elec_sys || (self.mc_max_kw <= self.aux_kw) || self.force_aux_on_fc {
            self.no_elec_aux = true;
        } else {
            self.no_elec_aux = false;
        }

        self.fc_perc_out_array = FC_PERC_OUT_ARRAY.clone().to_vec();

        // # discrete array of possible engine power outputs
        self.input_kw_out_array = self.fc_pwr_out_perc.clone() * self.fc_max_kw;
        // # Relatively continuous array of possible engine power outputs
        self.fc_kw_out_array = self
            .fc_perc_out_array
            .iter()
            .map(|n| n * self.fc_max_kw)
            .collect();
        // # Creates relatively continuous array for fc_eff
        if self.fc_eff_array.is_empty() {
            self.fc_eff_array = self
                .fc_perc_out_array
                .iter()
                .map(|x: &f64| -> f64 {
                    interpolate(
                        x,
                        &Array1::from(self.fc_pwr_out_perc.to_vec()),
                        &self.fc_eff_map,
                        false,
                    )
                })
                .collect();
        }
        //self.modern_max = MODERN_MAX;

        // NOTE: unused because the first part of if/else commented below is unused
        let modern_diff = self.modern_max - arrmax(&LARGE_BASELINE_EFF);
        let large_baseline_eff_adj: Vec<f64> =
            LARGE_BASELINE_EFF.iter().map(|x| x + modern_diff).collect();
        // Should the above lines be moved to another file? Or maybe have the outputs hardcoded?
        let mc_kw_adj_perc = max(
            0.0,
            min(
                (self.mc_max_kw - self.small_motor_power_kw)
                    / (self.large_motor_power_kw - self.small_motor_power_kw),
                1.0,
            ),
        );

        // NOTE: it should not be possible to have `None in self.mc_eff_map` in Rust (although NaN is possible...).
        //       if we want to express that mc_eff_map should be calculated in some cases, but not others,
        //       we may need some sort of option type ?
        //if None in self.mc_eff_map:
        //    self.mc_eff_array = mc_kw_adj_perc * large_baseline_eff_adj + \
        //            (1 - mc_kw_adj_perc) * self.small_baseline_eff
        //    self.mc_eff_map = self.mc_eff_array
        //else:
        //    self.mc_eff_array = self.mc_eff_map
        if self.mc_eff_map == Array1::<f64>::zeros(LARGE_BASELINE_EFF.len()) {
            self.mc_eff_map = large_baseline_eff_adj
                .iter()
                .zip(SMALL_BASELINE_EFF.iter())
                .map(|(&x, &y)| mc_kw_adj_perc * x + (1.0 - mc_kw_adj_perc) * y)
                .collect();
        }
        self.mc_eff_array = self.mc_eff_map.clone();
        // println!("{:?}",self.mc_eff_map);
        // self.mc_eff_array = mc_kw_adj_perc * large_baseline_eff_adj
        //     + (1.0 - mc_kw_adj_perc) * self.small_baseline_eff.clone();
        // self.mc_eff_map = self.mc_eff_array.clone();

        self.mc_perc_out_array = MC_PERC_OUT_ARRAY.clone().to_vec();

        self.mc_kw_out_array =
            (Array::linspace(0.0, 1.0, self.mc_perc_out_array.len()) * self.mc_max_kw).to_vec();

        self.mc_full_eff_array = self
            .mc_perc_out_array
            .iter()
            .enumerate()
            .map(|(idx, &x): (usize, &f64)| -> f64 {
                if idx == 0 {
                    0.0
                } else {
                    interpolate(&x, &self.mc_pwr_out_perc, &self.mc_eff_array, false)
                }
            })
            .collect();

        self.mc_kw_in_array = [0.0; 101]
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if idx == 0 {
                    0.0
                } else {
                    self.mc_kw_out_array[idx] / self.mc_full_eff_array[idx]
                }
            })
            .collect();

        self.mc_max_elec_in_kw = arrmax(&self.mc_kw_in_array);

        #[cfg(feature = "pyo3")]
        if let Some(new_fc_peak) = self.fc_peak_eff_override {
            self.set_fc_peak_eff(new_fc_peak);
            self.fc_peak_eff_override = None;
        }
        #[cfg(feature = "pyo3")]
        if let Some(new_mc_peak) = self.mc_peak_eff_override {
            self.set_mc_peak_eff(new_mc_peak);
            self.mc_peak_eff_override = None;
        }

        // check that efficiencies are not violating the first law of thermo
        assert!(
            arrmin(&self.fc_eff_array) >= 0.0,
            "min MC eff < 0 is not allowed"
        );
        assert!(self.fc_peak_eff() < 1.0, "fcPeakEff >= 1 is not allowed.");
        assert!(
            arrmin(&self.mc_full_eff_array) >= 0.0,
            "min MC eff < 0 is not allowed"
        );
        assert!(self.mc_peak_eff() < 1.0, "mcPeakEff >= 1 is not allowed.");

        self.set_veh_mass();
    }

    pub fn mock_vehicle() -> Self {
        let mut v = Self {
            scenario_name: String::from("2016 FORD Escape 4cyl 2WD"),
            selection: 5,
            veh_year: 2016,
            veh_pt_type: String::from("Conv"),
            drag_coef: 0.355,
            frontal_area_m2: 3.066,
            glider_kg: 1359.166,
            veh_cg_m: 0.53,
            drive_axle_weight_frac: 0.59,
            wheel_base_m: 2.6,
            cargo_kg: 136.0,
            veh_override_kg: None,
            comp_mass_multiplier: 1.4,
            fs_max_kw: 2000.0,
            fs_secs_to_peak_pwr: 1.0,
            fs_kwh: 504.0,
            fs_kwh_per_kg: 9.89,
            fc_max_kw: 125.0,
            fc_pwr_out_perc: array![
                0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
            ],
            fc_eff_map: array![
                0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
            ],
            fc_peak_eff_override: Default::default(),
            fc_eff_type: String::from("SI"),
            fc_sec_to_peak_pwr: 6.0,
            fc_base_kg: 61.0,
            fc_kw_per_kg: 2.13,
            min_fc_time_on: 30.0,
            idle_fc_kw: 2.5,
            mc_max_kw: 0.0,
            mc_peak_eff_override: Default::default(),
            mc_pwr_out_perc: array![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            mc_eff_map: array![0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,],
            mc_sec_to_peak_pwr: 4.0,
            mc_pe_kg_per_kw: 0.833,
            mc_pe_base_kg: 21.6,
            small_motor_power_kw: 7.5,
            large_motor_power_kw: 75.0,
            modern_max: MODERN_MAX,
            charging_on: false,
            max_roadway_chg_kw: Array1::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ess_max_kw: 0.0,
            ess_max_kwh: 0.0,
            ess_kg_per_kwh: 8.0,
            ess_base_kg: 75.0,
            ess_round_trip_eff: 0.97,
            ess_life_coef_a: 110.0,
            ess_life_coef_b: -0.6811,
            min_soc: 0.4,
            max_soc: 0.8,
            ess_dischg_to_fc_max_eff_perc: 0.0,
            ess_chg_to_fc_max_eff_perc: 0.0,
            wheel_inertia_kg_m2: 0.815,
            num_wheels: 4.0,
            wheel_rr_coef: 0.006,
            wheel_radius_m: 0.336,
            wheel_coef_of_fric: 0.7,
            max_accel_buffer_mph: 60.0,
            max_accel_buffer_perc_of_useable_soc: 0.2,
            perc_high_acc_buf: 0.0,
            mph_fc_on: 30.0,
            kw_demand_fc_on: 100.0,
            max_regen: 0.98,
            stop_start: false,
            force_aux_on_fc: true,
            alt_eff: 1.0,
            chg_eff: 0.86,
            aux_kw: 0.7,
            trans_kg: 114.0,
            trans_eff: 0.92,
            ess_to_fuel_ok_error: 0.005,
            val_udds_mpgge: 23.0,
            val_hwy_mpgge: 32.0,
            val_comb_mpgge: 26.0,
            val_udds_kwh_per_mile: f64::NAN,
            val_hwy_kwh_per_mile: f64::NAN,
            val_comb_kwh_per_mile: f64::NAN,
            val_cd_range_mi: f64::NAN,
            val_const65_mph_kwh_per_mile: f64::NAN,
            val_const60_mph_kwh_per_mile: f64::NAN,
            val_const55_mph_kwh_per_mile: f64::NAN,
            val_const45_mph_kwh_per_mile: f64::NAN,
            val_unadj_udds_kwh_per_mile: f64::NAN,
            val_unadj_hwy_kwh_per_mile: f64::NAN,
            val0_to60_mph: 9.9,
            val_ess_life_miles: f64::NAN,
            val_range_miles: f64::NAN,
            val_veh_base_cost: f64::NAN,
            val_msrp: f64::NAN,
            props: RustPhysicalProperties::default(),
            regen_a: 500.0,
            regen_b: 0.99,
            orphaned: Default::default(),
            // fields that get overriden by `set_derived`
            no_elec_sys: Default::default(),
            no_elec_aux: Default::default(),
            fc_perc_out_array: Default::default(),
            input_kw_out_array: Default::default(),
            fc_kw_out_array: Default::default(),
            fc_eff_array: Default::default(),
            mc_eff_array: Default::default(),
            mc_perc_out_array: Default::default(),
            mc_kw_out_array: Default::default(),
            mc_full_eff_array: Default::default(),
            mc_kw_in_array: Default::default(),
            mc_max_elec_in_kw: Default::default(),
            ess_mass_kg: Default::default(),
            mc_mass_kg: Default::default(),
            fc_mass_kg: Default::default(),
            fs_mass_kg: Default::default(),
            veh_kg: Default::default(),
            max_trac_mps2: Default::default(),
        };
        v.set_derived();
        v
    }

    pub fn from_str(filename: &str) -> Result<Self, anyhow::Error> {
        let mut veh_res: Result<RustVehicle, anyhow::Error> = Ok(serde_json::from_str(filename)?);
        veh_res.as_mut().unwrap().set_derived();
        veh_res
    }
}

impl SerdeAPI for RustVehicle {
    fn init(&mut self) {
        self.set_derived();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_derived_via_new() {
        let veh = RustVehicle::mock_vehicle();
        assert!(veh.veh_kg > 0.0);
    }

    #[test]
    fn test_veh_kg_override() {
        let mut veh_file = resources_path();
        veh_file.push("vehdb/test_overrides.yaml");
        let veh = RustVehicle::from_file(veh_file.as_os_str().to_str().unwrap()).unwrap();
        assert!(veh.veh_kg == veh.veh_override_kg.unwrap());
    }
}
