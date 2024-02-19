//! Module containing vehicle struct and related functions.

// local
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::{add_pyo3_api, doc_field, ApproxEq};
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;

use lazy_static::lazy_static;
use regex::Regex;
use validator::Validate;

// veh_pt_type options
pub const CONV: &str = "Conv";
pub const HEV: &str = "HEV";
pub const PHEV: &str = "PHEV";
pub const BEV: &str = "BEV";
pub const VEH_PT_TYPES: [&str; 4] = [CONV, HEV, PHEV, BEV];
lazy_static! {
    static ref VEH_PT_TYPE_OPTIONS_REGEX: Regex = Regex::new("Conv|HEV|PHEV|BEV").unwrap();
}

// fc_eff_type options
pub const SI: &str = "SI";
pub const ATKINSON: &str = "Atkinson";
pub const DIESEL: &str = "Diesel";
pub const H2FC: &str = "H2FC";
pub const HD_DIESEL: &str = "HD_Diesel";
pub const FC_EFF_TYPES: [&str; 5] = [SI, ATKINSON, DIESEL, H2FC, HD_DIESEL];
lazy_static! {
    static ref FC_EFF_TYPE_OPTIONS_REGEX: Regex =
        Regex::new("SI|Atkinson|Diesel|H2FC|HD_Diesel").unwrap();
}

#[doc_field]
#[add_pyo3_api(
    #[pyo3(name = "set_veh_mass")]
    pub fn set_veh_mass_py(&mut self) {
        // TODO: not urgent, but I think it'd better for all instances
        // of `set_veh_mass` to be `update_veh_mass`
        self.set_veh_mass()
    }

    #[getter]
    pub fn get_mc_peak_eff(&self) -> f64 {
        self.mc_peak_eff()
    }

    #[setter("mc_peak_eff")]
    pub fn set_mc_peak_eff_py(&mut self, new_peak: f64) {
        self.set_mc_peak_eff(new_peak);
    }

    #[getter]
    pub fn get_max_fc_eff_kw(&self) -> f64 {
        self.max_fc_eff_kw()
    }

    #[setter("fc_peak_eff")]
    pub fn set_fc_peak_eff_py(&mut self, new_peak: f64) {
        self.set_fc_peak_eff(new_peak);
    }

    #[getter]
    pub fn get_fc_peak_eff(&self) -> f64 {
        self.fc_peak_eff()
    }

    #[pyo3(name = "set_derived")]
    pub fn set_derived_py(&mut self) {
        self.set_derived().unwrap()
    }

    /// An identify function to allow RustVehicle to be used as a python vehicle and respond to this method
    /// Returns a clone of the current object
    pub fn to_rust(&self) -> Self {
        self.clone()
    }

    #[staticmethod]
    #[pyo3(name = "mock_vehicle")]
    fn mock_vehicle_py() -> Self {
        Self::mock_vehicle()
    }
)]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate)]
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
    #[doc_field(skip_doc)]
    pub props: RustPhysicalProperties,
    /// Vehicle name
    #[serde(alias = "name")]
    #[doc_field(skip_doc)]
    pub scenario_name: String,
    /// Vehicle database ID
    #[serde(skip)]
    #[doc_field(skip_doc)]
    pub selection: u32,
    /// Vehicle year
    #[serde(alias = "vehModelYear")]
    #[doc_field(skip_doc)]
    pub veh_year: u32,
    /// Vehicle powertrain type, one of \[[CONV](CONV), [HEV](HEV), [PHEV](PHEV), [BEV](BEV)\]
    #[serde(alias = "vehPtType")]
    #[validate(regex(
        path = "VEH_PT_TYPE_OPTIONS_REGEX",
        message = "must be one of [\"Conv\", \"HEV\", \"PHEV\", \"BEV\"]"
    ))]
    #[doc_field(skip_doc)]
    pub veh_pt_type: String,
    /// Aerodynamic drag coefficient
    #[serde(alias = "dragCoef")]
    #[validate(range(min = 0))]
    pub drag_coef: f64,
    /// Frontal area, $m^2$
    #[serde(alias = "frontalAreaM2")]
    #[validate(range(min = 0))]
    pub frontal_area_m2: f64,
    /// Vehicle mass excluding cargo, passengers, and powertrain components, $kg$
    #[serde(alias = "gliderKg")]
    #[validate(range(min = 0))]
    pub glider_kg: f64,
    /// Vehicle center of mass height, $m$
    /// **NOTE:** positive for FWD, negative for RWD, AWD, 4WD
    #[serde(alias = "vehCgM")]
    pub veh_cg_m: f64,
    /// Fraction of weight on the drive axle while stopped
    #[serde(alias = "driveAxleWeightFrac")]
    #[validate(range(min = 0, max = 1))]
    pub drive_axle_weight_frac: f64,
    /// Wheelbase, $m$
    #[serde(alias = "wheelBaseM")]
    #[validate(range(min = 0))]
    pub wheel_base_m: f64,
    /// Cargo mass including passengers, $kg$
    #[serde(alias = "cargoKg")]
    #[validate(range(min = 0))]
    pub cargo_kg: f64,
    /// Total vehicle mass, overrides mass calculation, $kg$
    #[serde(alias = "vehOverrideKg")]
    #[validate(range(min = 0))]
    pub veh_override_kg: Option<f64>,
    /// Component mass multiplier for vehicle mass calculation
    #[serde(alias = "compMassMultiplier")]
    #[validate(range(min = 0))]
    pub comp_mass_multiplier: f64,
    /// Fuel storage max power output, $kW$
    #[serde(alias = "maxFuelStorKw")]
    #[validate(range(min = 0))]
    pub fs_max_kw: f64,
    /// Fuel storage time to peak power, $s$
    #[serde(alias = "fuelStorSecsToPeakPwr")]
    #[validate(range(min = 0))]
    pub fs_secs_to_peak_pwr: f64,
    /// Fuel storage energy capacity, $kWh$
    #[serde(alias = "fuelStorKwh")]
    #[validate(range(min = 0))]
    pub fs_kwh: f64,
    /// Fuel specific energy, $\frac{kWh}{kg}$
    #[serde(alias = "fuelStorKwhPerKg")]
    #[validate(range(min = 0))]
    pub fs_kwh_per_kg: f64,
    /// Fuel converter peak continuous power, $kW$
    #[serde(alias = "maxFuelConvKw")]
    #[validate(range(min = 0))]
    pub fc_max_kw: f64,
    /// Fuel converter output power percentage map, x values of [fc_eff_map](RustVehicle::fc_eff_map)
    #[serde(alias = "fcPwrOutPerc")]
    pub fc_pwr_out_perc: Array1<f64>,
    /// Fuel converter efficiency map
    #[serde(default)]
    pub fc_eff_map: Array1<f64>,
    /// Fuel converter efficiency type, one of \[[SI](SI), [ATKINSON](ATKINSON), [DIESEL](DIESEL), [H2FC](H2FC), [HD_DIESEL](HD_DIESEL)\]
    /// Used for calculating [fc_eff_map](RustVehicle::fc_eff_map), and other calculations if H2FC
    #[serde(alias = "fcEffType")]
    #[validate(regex(
        path = "FC_EFF_TYPE_OPTIONS_REGEX",
        message = "must be one of [\"SI\", \"Atkinson\", \"Diesel\", \"H2FC\", \"HD_Diesel\"]"
    ))]
    pub fc_eff_type: String,
    /// Fuel converter time to peak power, $s$
    #[serde(alias = "fuelConvSecsToPeakPwr")]
    #[validate(range(min = 0))]
    pub fc_sec_to_peak_pwr: f64,
    /// Fuel converter base mass, $kg$
    #[serde(alias = "fuelConvBaseKg")]
    #[validate(range(min = 0))]
    pub fc_base_kg: f64,
    /// Fuel converter specific power (power-to-weight ratio), $\frac{kW}{kg}$
    #[serde(alias = "fuelConvKwPerKg")]
    #[validate(range(min = 0))]
    pub fc_kw_per_kg: f64,
    /// Minimum time fuel converter must be on before shutoff (for HEV, PHEV)
    #[serde(alias = "minFcTimeOn")]
    #[validate(range(min = 0))]
    pub min_fc_time_on: f64,
    /// Fuel converter idle power, $kW$
    #[serde(alias = "idleFcKw")]
    #[validate(range(min = 0))]
    pub idle_fc_kw: f64,
    /// Peak continuous electric motor power, $kW$
    #[serde(alias = "mcMaxElecInKw")]
    #[validate(range(min = 0))]
    pub mc_max_kw: f64,
    /// Electric motor output power percentage map, x values of [mc_eff_map](RustVehicle::mc_eff_map)
    #[serde(alias = "mcPwrOutPerc")]
    pub mc_pwr_out_perc: Array1<f64>,
    /// Electric motor efficiency map
    #[serde(alias = "mcEffArray")]
    pub mc_eff_map: Array1<f64>,
    /// Electric motor time to peak power, $s$
    #[serde(alias = "motorSecsToPeakPwr")]
    #[validate(range(min = 0))]
    pub mc_sec_to_peak_pwr: f64,
    /// Motor power electronics mass per power output, $\frac{kg}{kW}$
    #[serde(alias = "mcPeKgPerKw")]
    #[validate(range(min = 0))]
    pub mc_pe_kg_per_kw: f64,
    /// Motor power electronics base mass, $kg$
    #[serde(alias = "mcPeBaseKg")]
    #[validate(range(min = 0))]
    pub mc_pe_base_kg: f64,
    /// Traction battery maximum power output, $kW$
    #[serde(alias = "maxEssKw")]
    #[validate(range(min = 0))]
    pub ess_max_kw: f64,
    /// Traction battery energy capacity, $kWh$
    #[serde(alias = "maxEssKwh")]
    #[validate(range(min = 0))]
    pub ess_max_kwh: f64,
    /// Traction battery mass per energy, $\frac{kg}{kWh}$
    #[serde(alias = "essKgPerKwh")]
    #[validate(range(min = 0))]
    pub ess_kg_per_kwh: f64,
    /// Traction battery base mass, $kg$
    #[serde(alias = "essBaseKg")]
    #[validate(range(min = 0))]
    pub ess_base_kg: f64,
    /// Traction battery round-trip efficiency
    #[serde(alias = "essRoundTripEff")]
    #[validate(range(min = 0, max = 1))]
    pub ess_round_trip_eff: f64,
    /// Traction battery cycle life coefficient A, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
    #[serde(alias = "essLifeCoefA")]
    pub ess_life_coef_a: f64,
    /// Traction battery cycle life coefficient B, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
    #[serde(alias = "essLifeCoefB")]
    pub ess_life_coef_b: f64,
    /// Traction battery minimum state of charge
    #[serde(alias = "minSoc")]
    #[validate(range(min = 0, max = 1))]
    pub min_soc: f64,
    /// Traction battery maximum state of charge
    #[serde(alias = "maxSoc")]
    #[validate(range(min = 0, max = 1))]
    pub max_soc: f64,
    /// ESS discharge effort toward max FC efficiency
    #[serde(alias = "essDischgToFcMaxEffPerc")]
    #[validate(range(min = 0, max = 1))]
    pub ess_dischg_to_fc_max_eff_perc: f64,
    /// ESS charge effort toward max FC efficiency
    #[serde(alias = "essChgToFcMaxEffPerc")]
    #[validate(range(min = 0, max = 1))]
    pub ess_chg_to_fc_max_eff_perc: f64,
    /// Mass moment of inertia per wheel, $kg \cdot m^2$
    #[serde(alias = "wheelInertiaKgM2")]
    #[validate(range(min = 0))]
    pub wheel_inertia_kg_m2: f64,
    /// Number of wheels
    #[serde(alias = "numWheels")]
    #[validate(range(min = 0))]
    pub num_wheels: f64, // TODO: Shouldn't this just be a unsigned integer? u8 would work fine.
    /// Rolling resistance coefficient
    #[serde(alias = "wheelRrCoef")]
    #[validate(range(min = 0))]
    pub wheel_rr_coef: f64,
    /// Wheel radius, $m$
    #[serde(alias = "wheelRadiusM")]
    #[validate(range(min = 0))]
    pub wheel_radius_m: f64,
    /// Wheel coefficient of friction
    #[serde(alias = "wheelCoefOfFric")]
    #[validate(range(min = 0))]
    pub wheel_coef_of_fric: f64,
    /// Speed where the battery reserved for accelerating is zero
    #[serde(alias = "maxAccelBufferMph")]
    #[validate(range(min = 0))]
    pub max_accel_buffer_mph: f64,
    /// Percent of usable battery energy reserved to help accelerate
    #[serde(alias = "maxAccelBufferPercOfUseableSoc")]
    #[validate(range(min = 0, max = 1))]
    pub max_accel_buffer_perc_of_useable_soc: f64,
    /// Percent SOC buffer for high accessory loads during cycles with long idle time
    #[serde(alias = "percHighAccBuf")]
    #[validate(range(min = 0))]
    pub perc_high_acc_buf: f64,
    /// Speed at which the fuel converter must turn on, $mph$
    #[serde(alias = "mphFcOn")]
    #[validate(range(min = 0))]
    pub mph_fc_on: f64,
    /// Power demand above which to require fuel converter on, $kW$
    #[serde(alias = "kwDemandFcOn")]
    #[validate(range(min = 0))]
    pub kw_demand_fc_on: f64,
    /// Maximum brake regeneration efficiency
    #[serde(alias = "maxRegen")]
    #[validate(range(min = 0, max = 1))]
    pub max_regen: f64,
    /// Stop/start micro-HEV flag
    pub stop_start: bool,
    /// Force auxiliary power load to come from fuel converter
    #[serde(alias = "forceAuxOnFC")]
    pub force_aux_on_fc: bool,
    /// Alternator efficiency
    #[serde(alias = "altEff")]
    #[validate(range(min = 0, max = 1))]
    pub alt_eff: f64,
    /// Charger efficiency
    #[serde(alias = "chgEff")]
    #[validate(range(min = 0, max = 1))]
    pub chg_eff: f64,
    /// Auxiliary load power, $kW$
    #[serde(alias = "auxKw")]
    #[validate(range(min = 0))]
    pub aux_kw: f64,
    /// Transmission mass, $kg$
    #[serde(alias = "transKg")]
    #[validate(range(min = 0))]
    pub trans_kg: f64,
    /// Transmission efficiency
    #[serde(alias = "transEff")]
    #[validate(range(min = 0, max = 1))]
    pub trans_eff: f64,
    /// Maximum acceptable ratio of change in ESS energy to expended fuel energy (used in hybrid SOC balancing), $\frac{\Delta E_{ESS}}{\Delta E_{fuel}}$
    #[serde(alias = "essToFuelOkError")]
    #[validate(range(min = 0))]
    pub ess_to_fuel_ok_error: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub small_motor_power_kw: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub large_motor_power_kw: f64,
    // this and other fixed-size arrays can probably be vectors
    // without any performance penalty with the current implementation
    // of the functions in utils.rs
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub fc_perc_out_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(default = "RustVehicle::default_regen_a")]
    pub regen_a: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(default = "RustVehicle::default_regen_b")]
    pub regen_b: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub charging_on: bool,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub no_elec_sys: bool,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    // all of the parameters that are set in `set_derived` should be skipped by serde
    #[serde(skip)]
    pub no_elec_aux: bool,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub max_roadway_chg_kw: Array1<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub input_kw_out_array: Array1<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub fc_kw_out_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(default)]
    #[serde(alias = "fcEffArray", skip_serializing)]
    pub fc_eff_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub modern_max: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_eff_array: Array1<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_kw_in_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_kw_out_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_max_elec_in_kw: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_full_eff_array: Vec<f64>,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub veh_kg: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub max_trac_mps2: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub ess_mass_kg: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_mass_kg: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub fc_mass_kg: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub fs_mass_kg: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub mc_perc_out_array: Vec<f64>,
    // these probably don't need to be in rust
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_udds_mpgge: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_hwy_mpgge: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_comb_mpgge: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_udds_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_hwy_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_comb_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_cd_range_mi: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_const65_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_const60_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_const55_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_const45_mph_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_unadj_udds_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_unadj_hwy_kwh_per_mile: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val0_to60_mph: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_ess_life_miles: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_range_miles: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_veh_base_cost: f64,
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    #[serde(skip)]
    pub val_msrp: f64,
    /// Fuel converter efficiency peak override, scales entire curve
    #[serde(skip)]
    #[validate(range(min = 0, max = 1))]
    pub fc_peak_eff_override: Option<f64>,
    /// Motor efficiency peak override, scales entire curve
    #[serde(skip)]
    #[validate(range(min = 0, max = 1))]
    pub mc_peak_eff_override: Option<f64>,
    #[serde(skip)]
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    pub orphaned: bool,
    #[serde(skip)]
    #[doc(hidden)]
    #[doc_field(skip_doc)]
    pub max_regen_kwh: f64,
}

/// RustVehicle rust methods
impl RustVehicle {
    const VEHICLE_DIRECTORY_URL: &'static str =
        &"https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/";
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

    const fn default_regen_a() -> f64 {
        500.0
    }
    const fn default_regen_b() -> f64 {
        0.99
    }

    pub fn mc_peak_eff(&self) -> f64 {
        arrmax(&self.mc_full_eff_array)
    }

    /// Returns _first_ FC output power at which peak efficiency occurs
    pub fn max_fc_eff_kw(&self) -> f64 {
        let fc_eff_arr_max_i =
            first_eq(&self.fc_eff_array, arrmax(&self.fc_eff_array)).unwrap_or(0);
        self.fc_kw_out_array[fc_eff_arr_max_i]
    }

    pub fn fc_peak_eff(&self) -> f64 {
        arrmax(&self.fc_eff_array)
    }

    pub fn set_mc_peak_eff(&mut self, new_peak: f64) {
        let mc_max_eff = self.mc_eff_array.max().unwrap();
        self.mc_eff_array *= new_peak / mc_max_eff;
        let mc_max_full_eff = arrmax(&self.mc_full_eff_array);
        self.mc_full_eff_array = self
            .mc_full_eff_array
            .iter()
            .map(|e: &f64| -> f64 { e * (new_peak / mc_max_full_eff) })
            .collect();
    }

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
    pub fn set_derived(&mut self) -> anyhow::Result<()> {
        // Vehicle input validation
        self.validate()?;

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

        // Checking if a vehicle has any hybrid components
        self.no_elec_sys =
            (self.ess_max_kwh == 0.0) || (self.ess_max_kw == 0.0) || (self.mc_max_kw == 0.0);

        // Checking if aux loads go through an alternator
        self.no_elec_aux =
            self.no_elec_sys || (self.mc_max_kw <= self.aux_kw) || self.force_aux_on_fc;

        // TODO: this probably shouldnt be set if already provided
        self.fc_perc_out_array = FC_PERC_OUT_ARRAY.clone().to_vec();

        // discrete array of possible engine power outputs
        self.input_kw_out_array = &self.fc_pwr_out_perc * self.fc_max_kw;
        // Relatively continuous array of possible engine power outputs
        self.fc_kw_out_array = self
            .fc_perc_out_array
            .iter()
            .map(|n| n * self.fc_max_kw)
            .collect();
        // Creates relatively continuous array for fc_eff
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
        if self.modern_max == 0.0 {
            self.modern_max = MODERN_MAX;
        }

        let modern_diff = self.modern_max - arrmax(&LARGE_BASELINE_EFF);
        let large_baseline_eff_adj: Vec<f64> =
            LARGE_BASELINE_EFF.iter().map(|x| x + modern_diff).collect();
        let mc_kw_adj_perc = max(
            0.0,
            min(
                (self.mc_max_kw - self.small_motor_power_kw)
                    / (self.large_motor_power_kw - self.small_motor_power_kw),
                1.0,
            ),
        );

        if self.mc_eff_map == Array1::<f64>::zeros(LARGE_BASELINE_EFF.len()) {
            self.mc_eff_map = large_baseline_eff_adj
                .iter()
                .zip(SMALL_BASELINE_EFF.iter())
                .map(|(&x, &y)| mc_kw_adj_perc * x + (1.0 - mc_kw_adj_perc) * y)
                .collect();
        }
        self.mc_eff_array = self.mc_eff_map.clone();

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
        // TODO: this could perhaps be done in the input validators
        ensure!(
            arrmin(&self.fc_eff_array) >= 0.0,
            "minimum FC efficiency < 0 is not allowed"
        );
        ensure!(self.fc_peak_eff() < 1.0, "fc_peak_eff >= 1 is not allowed");
        if !self.no_elec_sys {
            ensure!(
                arrmin(&self.mc_full_eff_array) >= 0.0,
                "minimum MC efficiency < 0 is not allowed"
            );
            ensure!(self.mc_peak_eff() < 1.0, "mc_peak_eff >= 1 is not allowed");
        }

        self.set_veh_mass();

        self.max_regen_kwh = 0.5 * self.veh_kg * (27.0 * 27.0) / (3_600.0 * 1_000.0);

        Ok(())
    }

    pub fn mock_vehicle() -> Self {
        // NOTE: Default::default() uses mock_vehicle so this method can't "splat" defaults in
        // (i.e., = Self { ..Default::default() } will cause a stack overflow)
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
            max_regen_kwh: Default::default(),
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
            doc: Default::default(),
            drag_coef_doc: Default::default(),
            frontal_area_m2_doc: Default::default(),
            glider_kg_doc: Default::default(),
            veh_cg_m_doc: Default::default(),
            drive_axle_weight_frac_doc: Default::default(),
            wheel_base_m_doc: Default::default(),
            cargo_kg_doc: Default::default(),
            veh_override_kg_doc: Default::default(),
            comp_mass_multiplier_doc: Default::default(),
            fs_max_kw_doc: Default::default(),
            fs_secs_to_peak_pwr_doc: Default::default(),
            fs_kwh_doc: Default::default(),
            fs_kwh_per_kg_doc: Default::default(),
            fc_max_kw_doc: Default::default(),
            fc_pwr_out_perc_doc: Default::default(),
            fc_eff_map_doc: Default::default(),
            fc_eff_type_doc: Default::default(),
            fc_sec_to_peak_pwr_doc: Default::default(),
            fc_base_kg_doc: Default::default(),
            fc_kw_per_kg_doc: Default::default(),
            min_fc_time_on_doc: Default::default(),
            idle_fc_kw_doc: Default::default(),
            mc_max_kw_doc: Default::default(),
            mc_pwr_out_perc_doc: Default::default(),
            mc_eff_map_doc: Default::default(),
            mc_sec_to_peak_pwr_doc: Default::default(),
            mc_pe_kg_per_kw_doc: Default::default(),
            mc_pe_base_kg_doc: Default::default(),
            ess_max_kw_doc: Default::default(),
            ess_max_kwh_doc: Default::default(),
            ess_kg_per_kwh_doc: Default::default(),
            ess_base_kg_doc: Default::default(),
            ess_round_trip_eff_doc: Default::default(),
            ess_life_coef_a_doc: Default::default(),
            ess_life_coef_b_doc: Default::default(),
            min_soc_doc: Default::default(),
            max_soc_doc: Default::default(),
            ess_dischg_to_fc_max_eff_perc_doc: Default::default(),
            ess_chg_to_fc_max_eff_perc_doc: Default::default(),
            wheel_inertia_kg_m2_doc: Default::default(),
            num_wheels_doc: Default::default(),
            wheel_rr_coef_doc: Default::default(),
            wheel_radius_m_doc: Default::default(),
            wheel_coef_of_fric_doc: Default::default(),
            max_accel_buffer_mph_doc: Default::default(),
            max_accel_buffer_perc_of_useable_soc_doc: Default::default(),
            perc_high_acc_buf_doc: Default::default(),
            mph_fc_on_doc: Default::default(),
            kw_demand_fc_on_doc: Default::default(),
            max_regen_doc: Default::default(),
            stop_start_doc: Default::default(),
            force_aux_on_fc_doc: Default::default(),
            alt_eff_doc: Default::default(),
            chg_eff_doc: Default::default(),
            aux_kw_doc: Default::default(),
            trans_kg_doc: Default::default(),
            trans_eff_doc: Default::default(),
            ess_to_fuel_ok_error_doc: Default::default(),
            fc_peak_eff_override_doc: Default::default(),
            mc_peak_eff_override_doc: Default::default(),
        };
        v.set_derived().unwrap();
        v
    }

    /// Downloads specified vehicle from FASTSim vehicle repo or url and
    /// instantiates it into a RustVehicle. Notes in vehicle.doc the origin of
    /// the vehicle. Returns vehicle.  
    /// # Arguments  
    /// - vehicle_file_name: file name for vehicle to be downloaded, including
    ///   path from url directory or FASTSim repository (if applicable)  
    /// - url: url for vehicle repository where vehicle will be downloaded from,
    ///   if None, assumed to be downloaded from vehicle FASTSim repo  
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL, split up so that the "url" argument is the path to the
    /// directory, and the "vehicle_file_name" is the path within the directory
    /// to the file.  
    /// Note: If downloading from the FASTSim Vehicle Repo, the
    /// vehicle_file_name should include the path to the file from the root of
    /// the Repo, as listed in the output of the
    /// vehicle_utils::fetch_github_list() function.  
    /// Note: the url should not include the file name, only the path to the
    /// file or a root directory of the file.
    pub fn from_github_or_url<S: AsRef<str>>(
        vehicle_file_name: S,
        url: Option<S>,
    ) -> anyhow::Result<Self> {
        let url_internal = match url {
            Some(s) => {
                s.as_ref().trim_end_matches('/').to_owned()
                    + "/"
                    + &vehicle_file_name.as_ref().trim_start_matches('/')
            }
            None => Self::VEHICLE_DIRECTORY_URL.to_string() + vehicle_file_name.as_ref(),
        };
        let mut vehicle =
            Self::from_url(&url_internal).with_context(|| "Could not parse vehicle from url")?;
        let vehicle_origin = "Vehicle from ".to_owned() + url_internal.as_str();
        vehicle.doc = Some(vehicle_origin);
        Ok(vehicle)
    }
}

impl Default for RustVehicle {
    fn default() -> Self {
        let mut veh = RustVehicle::mock_vehicle();
        veh.scenario_name = Default::default();
        veh.selection = Default::default();
        veh.veh_year = Default::default();
        veh.val_udds_kwh_per_mile = Default::default();
        veh.val_hwy_kwh_per_mile = Default::default();
        veh.val_comb_kwh_per_mile = Default::default();
        veh.val_cd_range_mi = Default::default();
        veh.val_const65_mph_kwh_per_mile = Default::default();
        veh.val_const60_mph_kwh_per_mile = Default::default();
        veh.val_const55_mph_kwh_per_mile = Default::default();
        veh.val_const45_mph_kwh_per_mile = Default::default();
        veh.val_unadj_udds_kwh_per_mile = Default::default();
        veh.val_unadj_hwy_kwh_per_mile = Default::default();
        veh.val0_to60_mph = Default::default();
        veh.val_ess_life_miles = Default::default();
        veh.val_range_miles = Default::default();
        veh.val_veh_base_cost = Default::default();
        veh.val_msrp = Default::default();
        veh
    }
}

impl SerdeAPI for RustVehicle {
    const CACHE_FOLDER: &'static str = &"vehicles";

    fn init(&mut self) -> anyhow::Result<()> {
        self.set_derived()
    }

    /// instantiates a vehicle from a url, and notes in vehicle.doc the origin
    /// of the vehicle.  
    /// accepts yaml and json file types  
    /// # Arguments  
    /// - url: URL (either as a string or url type) to object  
    /// Note: The URL needs to be a URL pointing directly to a file, for example
    /// a raw github URL.
    fn from_url<S: AsRef<str>>(url: S) -> anyhow::Result<Self> {
        let url = url::Url::parse(url.as_ref())?;
        let format = url
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|filename| Path::new(filename).extension())
            .and_then(OsStr::to_str)
            .with_context(|| "Could not parse file format from URL: {url:?}")?;
        let response = ureq::get(url.as_ref()).call()?.into_reader();
        let mut vehicle = Self::from_reader(response, format)?;
        let vehicle_origin = "Vehicle from ".to_owned() + url.as_ref();
        vehicle.doc = Some(vehicle_origin);
        Ok(vehicle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use validator::ValidationErrors;

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
        // test input validation by providing bad inputs, then checking
        // the produced error for the offending field names
    }

    #[test]
    fn test_input_validation() {
        // set up vehicle input parameters
        let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
        let selection: u32 = 5;
        let veh_year: u32 = 2016;
        let veh_pt_type = String::from("whoops"); // bad input
        let drag_coef: f64 = 0.355;
        let frontal_area_m2: f64 = 3.066;
        let glider_kg: f64 = -50.0; // bad input
        let veh_cg_m: f64 = 0.53;
        let drive_axle_weight_frac: f64 = 0.59;
        let wheel_base_m: f64 = 2.6;
        let cargo_kg: f64 = 136.0;
        let veh_override_kg: Option<f64> = None;
        let comp_mass_multiplier: f64 = 1.4;
        let fs_max_kw: f64 = 2000.0;
        let fs_secs_to_peak_pwr: f64 = 1.0;
        let fs_kwh: f64 = 504.0;
        let fs_kwh_per_kg: f64 = 9.89;
        let fc_max_kw: f64 = -60.0; // bad input
        let fc_pwr_out_perc: Vec<f64> = vec![
            0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
        ];
        let fc_eff_type: String = String::from("SI");
        let fc_sec_to_peak_pwr: f64 = 6.0;
        let fc_base_kg: f64 = 61.0;
        let fc_kw_per_kg: f64 = 2.13;
        let min_fc_time_on: f64 = 30.0;
        let idle_fc_kw: f64 = 2.5;
        let mc_max_kw: f64 = 0.0;
        let mc_sec_to_peak_pwr: f64 = 4.0;
        let mc_pe_kg_per_kw: f64 = 0.833;
        let mc_pe_base_kg: f64 = 21.6;
        let ess_max_kw: f64 = 0.0;
        let ess_max_kwh: f64 = 0.0;
        let ess_kg_per_kwh: f64 = 8.0;
        let ess_base_kg: f64 = 75.0;
        let ess_round_trip_eff: f64 = 0.97;
        let ess_life_coef_a: f64 = 110.0;
        let ess_life_coef_b: f64 = -0.6811;
        let min_soc: f64 = -0.5; // bad input
        let max_soc: f64 = 1.5; // bad input
        let ess_dischg_to_fc_max_eff_perc: f64 = 0.0;
        let ess_chg_to_fc_max_eff_perc: f64 = 0.0;
        let wheel_inertia_kg_m2: f64 = 0.815;
        let num_wheels: f64 = 4.0;
        let wheel_rr_coef: f64 = 0.006;
        let wheel_radius_m: f64 = 0.336;
        let wheel_coef_of_fric: f64 = 0.7;
        let max_accel_buffer_mph: f64 = 60.0;
        let max_accel_buffer_perc_of_useable_soc: f64 = 0.2;
        let perc_high_acc_buf: f64 = 0.0;
        let mph_fc_on: f64 = 30.0;
        let kw_demand_fc_on: f64 = 100.0;
        let max_regen: f64 = 0.98;
        let stop_start: bool = false;
        let force_aux_on_fc: bool = true;
        let alt_eff: f64 = 1.0;
        let chg_eff: f64 = 0.86;
        let aux_kw: f64 = 0.7;
        let trans_kg: f64 = 114.0;
        let trans_eff: f64 = 0.92;
        let ess_to_fuel_ok_error: f64 = 0.005;
        let val_udds_mpgge: f64 = 23.0;
        let val_hwy_mpgge: f64 = 32.0;
        let val_comb_mpgge: f64 = 26.0;
        let val_udds_kwh_per_mile: f64 = f64::NAN;
        let val_hwy_kwh_per_mile: f64 = f64::NAN;
        let val_comb_kwh_per_mile: f64 = f64::NAN;
        let val_cd_range_mi: f64 = f64::NAN;
        let val_const65_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const60_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const55_mph_kwh_per_mile: f64 = f64::NAN;
        let val_const45_mph_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_udds_kwh_per_mile: f64 = f64::NAN;
        let val_unadj_hwy_kwh_per_mile: f64 = f64::NAN;
        let val0_to60_mph: f64 = 9.9;
        let val_ess_life_miles: f64 = f64::NAN;
        let val_range_miles: f64 = f64::NAN;
        let val_veh_base_cost: f64 = f64::NAN;
        let val_msrp: f64 = f64::NAN;
        let props = RustPhysicalProperties::default();
        let regen_a: f64 = 500.0;
        let regen_b: f64 = 0.99;
        let fc_peak_eff_override: Option<f64> = None;
        let mc_peak_eff_override: Option<f64> = Some(-0.50); // bad input
        let small_motor_power_kw = 7.5;
        let large_motor_power_kw = 75.0;
        let fc_perc_out_array = FC_PERC_OUT_ARRAY.clone().to_vec();
        let mc_eff_map = Array1::<f64>::zeros(LARGE_BASELINE_EFF.len());
        let mc_kw_out_array =
            (Array::linspace(0.0, 1.0, MC_PERC_OUT_ARRAY.len()) * mc_max_kw).to_vec();
        let mc_perc_out_array = MC_PERC_OUT_ARRAY.clone().to_vec();
        let mc_pwr_out_perc = array![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mc_full_eff_array: Vec<f64> = mc_perc_out_array
            .iter()
            .enumerate()
            .map(|(idx, &x): (usize, &f64)| -> f64 {
                if idx == 0 {
                    0.0
                } else {
                    interpolate(&x, &mc_pwr_out_perc, &mc_eff_map, false)
                }
            })
            .collect();
        let mc_kw_in_array: Vec<f64> = [0.0; 101]
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if idx == 0 {
                    0.0
                } else {
                    mc_kw_out_array[idx] / mc_full_eff_array[idx]
                }
            })
            .collect();
        let mc_max_elec_in_kw = arrmax(&mc_kw_in_array);

        // instantiate vehicle result
        let mut veh = RustVehicle {
            small_motor_power_kw,
            large_motor_power_kw,
            fc_perc_out_array: FC_PERC_OUT_ARRAY.clone().to_vec(),
            charging_on: Default::default(),
            no_elec_sys: Default::default(),
            no_elec_aux: Default::default(),
            max_roadway_chg_kw: Default::default(),
            input_kw_out_array: Array1::from_vec(fc_pwr_out_perc.clone()) * fc_max_kw,
            fc_kw_out_array: fc_perc_out_array.iter().map(|n| n * fc_max_kw).collect(),
            fc_eff_array: fc_perc_out_array
                .iter()
                .map(|x: &f64| -> f64 {
                    interpolate(
                        x,
                        &Array1::from(fc_pwr_out_perc.to_vec()),
                        &array![
                            0.10, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.30
                        ],
                        false,
                    )
                })
                .collect(),
            modern_max: MODERN_MAX,
            mc_eff_array: mc_eff_map,
            mc_kw_in_array: [0.0; 101]
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    if idx == 0 {
                        0.0
                    } else {
                        mc_kw_out_array[idx] / mc_full_eff_array[idx]
                    }
                })
                .collect(),
            mc_kw_out_array,
            mc_max_elec_in_kw,
            mc_full_eff_array,
            // these get calculated in `se
            veh_kg: Default::default(),
            max_trac_mps2: Default::default(),
            ess_mass_kg: Default::default(),
            mc_mass_kg: Default::default(),
            fc_mass_kg: Default::default(),
            fs_mass_kg: Default::default(),
            mc_perc_out_array,
            orphaned: Default::default(),
            scenario_name,
            selection,
            veh_year,
            veh_pt_type, // bad input
            drag_coef,
            frontal_area_m2,
            glider_kg, // bad input
            veh_cg_m,
            drive_axle_weight_frac,
            wheel_base_m,
            cargo_kg,
            veh_override_kg,
            comp_mass_multiplier,
            fs_max_kw,
            fs_secs_to_peak_pwr,
            fs_kwh,
            fs_kwh_per_kg,
            fc_max_kw, // bad input
            fc_pwr_out_perc: array![
                0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
            ],
            fc_eff_map: array![
                0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
            ],
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            mc_eff_map: array![0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92],
            mc_sec_to_peak_pwr,
            mc_pe_kg_per_kw,
            mc_pe_base_kg,
            ess_max_kw,
            ess_max_kwh,
            ess_kg_per_kwh,
            ess_base_kg,
            ess_round_trip_eff,
            ess_life_coef_a,
            ess_life_coef_b,
            min_soc, // bad input
            max_soc, // bad input
            ess_dischg_to_fc_max_eff_perc,
            ess_chg_to_fc_max_eff_perc,
            wheel_inertia_kg_m2,
            num_wheels,
            wheel_rr_coef,
            wheel_radius_m,
            wheel_coef_of_fric,
            max_accel_buffer_mph,
            max_accel_buffer_perc_of_useable_soc,
            perc_high_acc_buf,
            mph_fc_on,
            kw_demand_fc_on,
            max_regen,
            stop_start,
            force_aux_on_fc,
            alt_eff,
            chg_eff,
            aux_kw,
            trans_kg,
            trans_eff,
            ess_to_fuel_ok_error,
            val_udds_mpgge,
            val_hwy_mpgge,
            val_comb_mpgge,
            val_udds_kwh_per_mile,
            val_hwy_kwh_per_mile,
            val_comb_kwh_per_mile,
            val_cd_range_mi,
            val_const65_mph_kwh_per_mile,
            val_const60_mph_kwh_per_mile,
            val_const55_mph_kwh_per_mile,
            val_const45_mph_kwh_per_mile,
            val_unadj_udds_kwh_per_mile,
            val_unadj_hwy_kwh_per_mile,
            val0_to60_mph,
            val_ess_life_miles,
            val_range_miles,
            val_veh_base_cost,
            val_msrp,
            props,
            regen_a,
            regen_b,
            fc_peak_eff_override,
            mc_peak_eff_override, // bad input
            ..Default::default()
        };

        let validation_result = veh.set_derived();

        // hard-coded fields where bad inputs were provided above
        let bad_fields = [
            "veh_pt_type",
            "glider_kg",
            "fc_max_kw",
            "min_soc",
            "max_soc",
            "mc_peak_eff_override",
        ];
        // downcast anyhow::error back into validator::ValidationErrors
        // this test will fail on the unwrap() if the error is not downcastable to ValidationErrors
        // e.g. if the error was not from input validation
        let validation_errs = validation_result
            .unwrap_err()
            .downcast::<ValidationErrors>()
            .unwrap();
        let validation_errs_hashmap = validation_errs.errors();
        // assert that specified bad fields were caught
        assert!(validation_errs_hashmap
            .keys()
            .all(|key| bad_fields.contains(key)));
        assert!(validation_errs_hashmap.len() == bad_fields.len());
    }
}
