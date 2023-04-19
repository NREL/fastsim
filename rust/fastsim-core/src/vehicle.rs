//! Module containing vehicle struct and related functions.

// local
use crate::imports::*;
use crate::params::*;
use crate::proc_macros::{add_pyo3_api, ApproxEq};
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

#[derive(Default, Serialize, Deserialize, Clone, Debug, PartialEq, ApproxEq, Validate)]
#[add_pyo3_api(
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn __new__(
        scenario_name: String,
        selection: u32,
        veh_year: u32,
        veh_pt_type: String,
        drag_coef: f64,
        frontal_area_m2: f64,
        glider_kg: f64,
        veh_cg_m: f64,
        drive_axle_weight_frac: f64,
        wheel_base_m: f64,
        cargo_kg: f64,
        veh_override_kg: Option<f64>,
        comp_mass_multiplier: f64,
        fs_max_kw: f64,
        fs_secs_to_peak_pwr: f64,
        fs_kwh: f64,
        fs_kwh_per_kg: f64,
        fc_max_kw: f64,
        fc_pwr_out_perc: Vec<f64>,
        fc_eff_map: Vec<f64>,
        fc_eff_type: String,
        fc_sec_to_peak_pwr: f64,
        fc_base_kg: f64,
        fc_kw_per_kg: f64,
        min_fc_time_on: f64,
        idle_fc_kw: f64,
        mc_max_kw: f64,
        mc_pwr_out_perc: Vec<f64>,
        // todo: check how this behaves w.r.t. to being a keyword argument with positional arguments after it
        mc_eff_map: Option<Vec<f64>>,
        mc_sec_to_peak_pwr: f64,
        mc_pe_kg_per_kw: f64,
        mc_pe_base_kg: f64,
        ess_max_kw: f64,
        ess_max_kwh: f64,
        ess_kg_per_kwh: f64,
        ess_base_kg: f64,
        ess_round_trip_eff: f64,
        ess_life_coef_a: f64,
        ess_life_coef_b: f64,
        min_soc: f64,
        max_soc: f64,
        ess_dischg_to_fc_max_eff_perc: f64,
        ess_chg_to_fc_max_eff_perc: f64,
        wheel_inertia_kg_m2: f64,
        num_wheels: f64,
        wheel_rr_coef: f64,
        wheel_radius_m: f64,
        wheel_coef_of_fric: f64,
        max_accel_buffer_mph: f64,
        max_accel_buffer_perc_of_useable_soc: f64,
        perc_high_acc_buf: f64,
        mph_fc_on: f64,
        kw_demand_fc_on: f64,
        max_regen: f64,
        stop_start: bool,
        force_aux_on_fc: bool,
        alt_eff: f64,
        chg_eff: f64,
        aux_kw: f64,
        trans_kg: f64,
        trans_eff: f64,
        ess_to_fuel_ok_error: f64,
        val_udds_mpgge: f64,
        val_hwy_mpgge: f64,
        val_comb_mpgge: f64,
        val_udds_kwh_per_mile: f64,
        val_hwy_kwh_per_mile: f64,
        val_comb_kwh_per_mile: f64,
        val_cd_range_mi: f64,
        val_const65_mph_kwh_per_mile: f64,
        val_const60_mph_kwh_per_mile: f64,
        val_const55_mph_kwh_per_mile: f64,
        val_const45_mph_kwh_per_mile: f64,
        val_unadj_udds_kwh_per_mile: f64,
        val_unadj_hwy_kwh_per_mile: f64,
        val0_to60_mph: f64,
        val_ess_life_miles: f64,
        val_range_miles: f64,
        val_veh_base_cost: f64,
        val_msrp: f64,
        props: RustPhysicalProperties,
        //small_motor_power_kw: f64,
        //large_motor_power_kw: f64,
        //fc_perc_out_array: Option<Vec<f64>>,
        //charging_on: bool,
        //no_elec_sys: bool,
        //no_elec_aux: bool,
        //max_roadway_chg_kw: Vec<f64>,
        //input_kw_out_array: Option<Vec<f64>>,
        //fc_kw_out_array: Vec<f64>,
        //fc_eff_array: Vec<f64>,
        //modern_max: f64,
        //mc_eff_array: Vec<f64>,
        //mc_kw_in_array: Vec<f64>,
        //mc_kw_out_array: Vec<f64>,
        //mc_max_elec_in_kw: f64,
        //mc_full_eff_array: Option<Vec<f64>>,
        regen_a: f64,
        regen_b: f64,
        //veh_kg: f64,
        //max_trac_mps2: f64,
        //ess_mass_kg: f64,
        //mc_mass_kg: f64,
        //fc_mass_kg: f64,
        //fs_mass_kg: f64,
        //mc_perc_out_array: Option<Vec<f64>>,
        fc_peak_eff_override: Option<f64>,
        mc_peak_eff_override: Option<f64>,
    ) -> Result<Self, anyhow::Error> {
        Self::new(
            scenario_name,
            selection,
            veh_year,
            veh_pt_type,
            drag_coef,
            frontal_area_m2,
            glider_kg,
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
            fc_max_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            mc_eff_map,
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
            min_soc,
            max_soc,
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
            //small_motor_power_kw: f64,
            //large_motor_power_kw: f64,
            //fc_perc_out_array: Option<Vec<f64>>,
            //charging_on: bool,
            //no_elec_sys: bool,
            //no_elec_aux: bool,
            //max_roadway_chg_kw: Vec<f64>,
            //input_kw_out_array: Option<Vec<f64>>,
            //fc_kw_out_array: Vec<f64>,
            //fc_eff_array: Vec<f64>,
            //modern_max: f64,
            //mc_eff_array: Vec<f64>,
            //mc_kw_in_array: Vec<f64>,
            //mc_kw_out_array: Vec<f64>,
            //mc_max_elec_in_kw: f64,
            //mc_full_eff_array: Option<Vec<f64>>,
            regen_a,
            regen_b,
            //veh_kg: f64,
            //max_trac_mps2: f64,
            //ess_mass_kg: f64,
            //mc_mass_kg: f64,
            //fc_mass_kg: f64,
            //fs_mass_kg: f64,
            //mc_perc_out_array: Option<Vec<f64>>,
            fc_peak_eff_override,
            mc_peak_eff_override,
        )
    }

    pub fn __getnewargs__(&self) {
        todo!();
    }

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
    #[validate(regex(
        path = "VEH_PT_TYPE_OPTIONS_REGEX",
        message = "must be one of [\"Conv\", \"HEV\", \"PHEV\", \"BEV\"]"
    ))]
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
    pub veh_override_kg: f64,
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
    /// Fuel converter output power percentage map, x-values of [fc_eff_map](RustVehicle::fc_eff_map)
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
    /// Electric motor output power percentage map, x-values of [mc_eff_map](RustVehicle::mc_eff_map)
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
    #[validate(range(min = 0, max = 1))]
    pub fc_peak_eff_override: Option<f64>,
    /// Motor efficiency peak override, scales entire curve
    #[serde(skip)]
    #[validate(range(min = 0, max = 1))]
    pub mc_peak_eff_override: Option<f64>,
    #[serde(skip)]
    #[doc(hidden)]
    pub orphaned: bool,
}

/// RustVehicle rust methods
impl RustVehicle {
    #[allow(clippy::too_many_arguments)]
    /// Create new vehicle instance
    pub fn new(
        scenario_name: String,
        selection: u32,
        veh_year: u32,
        veh_pt_type: String,
        drag_coef: f64,
        frontal_area_m2: f64,
        glider_kg: f64,
        veh_cg_m: f64,
        drive_axle_weight_frac: f64,
        wheel_base_m: f64,
        cargo_kg: f64,
        veh_override_kg: Option<f64>,
        comp_mass_multiplier: f64,
        fs_max_kw: f64,
        fs_secs_to_peak_pwr: f64,
        fs_kwh: f64,
        fs_kwh_per_kg: f64,
        fc_max_kw: f64,
        fc_pwr_out_perc: Vec<f64>,
        fc_eff_map: Vec<f64>,
        fc_eff_type: String,
        fc_sec_to_peak_pwr: f64,
        fc_base_kg: f64,
        fc_kw_per_kg: f64,
        min_fc_time_on: f64,
        idle_fc_kw: f64,
        mc_max_kw: f64,
        mc_pwr_out_perc: Vec<f64>,
        mc_eff_map: Option<Vec<f64>>,
        mc_sec_to_peak_pwr: f64,
        mc_pe_kg_per_kw: f64,
        mc_pe_base_kg: f64,
        ess_max_kw: f64,
        ess_max_kwh: f64,
        ess_kg_per_kwh: f64,
        ess_base_kg: f64,
        ess_round_trip_eff: f64,
        ess_life_coef_a: f64,
        ess_life_coef_b: f64,
        min_soc: f64,
        max_soc: f64,
        ess_dischg_to_fc_max_eff_perc: f64,
        ess_chg_to_fc_max_eff_perc: f64,
        wheel_inertia_kg_m2: f64,
        num_wheels: f64,
        wheel_rr_coef: f64,
        wheel_radius_m: f64,
        wheel_coef_of_fric: f64,
        max_accel_buffer_mph: f64,
        max_accel_buffer_perc_of_useable_soc: f64,
        perc_high_acc_buf: f64,
        mph_fc_on: f64,
        kw_demand_fc_on: f64,
        max_regen: f64,
        stop_start: bool,
        force_aux_on_fc: bool,
        alt_eff: f64,
        chg_eff: f64,
        aux_kw: f64,
        trans_kg: f64,
        trans_eff: f64,
        ess_to_fuel_ok_error: f64,
        val_udds_mpgge: f64,
        val_hwy_mpgge: f64,
        val_comb_mpgge: f64,
        val_udds_kwh_per_mile: f64,
        val_hwy_kwh_per_mile: f64,
        val_comb_kwh_per_mile: f64,
        val_cd_range_mi: f64,
        val_const65_mph_kwh_per_mile: f64,
        val_const60_mph_kwh_per_mile: f64,
        val_const55_mph_kwh_per_mile: f64,
        val_const45_mph_kwh_per_mile: f64,
        val_unadj_udds_kwh_per_mile: f64,
        val_unadj_hwy_kwh_per_mile: f64,
        val0_to60_mph: f64,
        val_ess_life_miles: f64,
        val_range_miles: f64,
        val_veh_base_cost: f64,
        val_msrp: f64,
        props: RustPhysicalProperties,
        regen_a: f64,
        regen_b: f64,
        fc_peak_eff_override: Option<f64>,
        mc_peak_eff_override: Option<f64>,
    ) -> Result<Self, anyhow::Error> {
        let fc_pwr_out_perc = Array::from_vec(fc_pwr_out_perc);
        let fc_eff_map = Array::from_vec(fc_eff_map);
        let mc_pwr_out_perc = Array::from_vec(mc_pwr_out_perc);
        let mc_eff_map: Array1<f64> =
            Array::from_vec(mc_eff_map.unwrap_or_else(|| vec![0.0; LARGE_BASELINE_EFF.len()]));
        let veh_override_kg: f64 = veh_override_kg.unwrap_or(0.0);

        let mut veh = Self {
            scenario_name,
            selection,
            veh_year,
            veh_pt_type,
            drag_coef,
            frontal_area_m2,
            glider_kg,
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
            fc_max_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            mc_eff_map,
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
            min_soc,
            max_soc,
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
            small_motor_power_kw: 7.5,
            large_motor_power_kw: 75.0,
            fc_perc_out_array: Default::default(),
            regen_a,
            regen_b,
            charging_on: false,
            no_elec_sys: false,
            no_elec_aux: false,
            max_roadway_chg_kw: Array1::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            input_kw_out_array: Array1::<f64>::zeros(1),
            fc_kw_out_array: Vec::new(),
            fc_eff_array: Vec::new(),
            modern_max: MODERN_MAX,
            mc_eff_array: Array1::<f64>::zeros(1),
            mc_kw_in_array: Vec::new(),
            mc_kw_out_array: Vec::new(),
            mc_max_elec_in_kw: 0.0,
            mc_full_eff_array: Vec::new(),
            veh_kg: 0.0,
            max_trac_mps2: 0.0,
            ess_mass_kg: 0.0,
            mc_mass_kg: 0.0,
            fc_mass_kg: 0.0,
            fs_mass_kg: 0.0,
            mc_perc_out_array: Default::default(),
            fc_peak_eff_override,
            mc_peak_eff_override,
            orphaned: false,
        };
        match veh.validate() {
            Ok(_) => (),
            Err(e) => bail!(e),
        };
        veh.set_derived();
        Ok(veh)
    }

    /// Calculate total vehicle mass. Sum up component masses if
    /// positive real number is not specified for self.veh_override_kg
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn set_veh_mass(&mut self) {
        let mut ess_mass_kg = 0.0;
        let mut mc_mass_kg = 0.0;
        let mut fc_mass_kg = 0.0;
        let mut fs_mass_kg = 0.0;

        if !(self.veh_override_kg > 0.0) {
            ess_mass_kg = if self.ess_max_kwh == 0.0 || self.ess_max_kw == 0.0 {
                0.0
            } else {
                ((self.ess_max_kwh * self.ess_kg_per_kwh) + self.ess_base_kg)
                    * self.comp_mass_multiplier
            };
            mc_mass_kg = if self.mc_max_kw == 0.0 {
                0.0
            } else {
                (self.mc_pe_base_kg + (self.mc_pe_kg_per_kw * self.mc_max_kw))
                    * self.comp_mass_multiplier
            };
            fc_mass_kg = if self.fc_max_kw == 0.0 {
                0.0
            } else {
                (1.0 / self.fc_kw_per_kg * self.fc_max_kw + self.fc_base_kg)
                    * self.comp_mass_multiplier
            };
            fs_mass_kg = if self.fs_max_kw == 0.0 {
                0.0
            } else {
                ((1.0 / self.fs_kwh_per_kg) * self.fs_kwh) * self.comp_mass_multiplier
            };
            self.veh_kg = self.cargo_kg
                + self.glider_kg
                + self.trans_kg * self.comp_mass_multiplier
                + ess_mass_kg
                + mc_mass_kg
                + fc_mass_kg
                + fs_mass_kg;
        } else {
            // if positive real number is specified for veh_override_kg, use that
            self.veh_kg = self.veh_override_kg;
        }

        self.max_trac_mps2 = (self.wheel_coef_of_fric
            * self.drive_axle_weight_frac
            * self.veh_kg
            * self.props.a_grav_mps2
            / (1.0 + self.veh_cg_m * self.wheel_coef_of_fric / self.wheel_base_m))
            / (self.veh_kg * self.props.a_grav_mps2)
            * self.props.a_grav_mps2;

        // copying to instance attributes
        self.ess_mass_kg = ess_mass_kg;
        self.mc_mass_kg = mc_mass_kg;
        self.fc_mass_kg = fc_mass_kg;
        self.fs_mass_kg = fs_mass_kg;
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

    /// Sets derived parameters.  
    /// Arguments:  
    /// ----------  
    /// mc_peak_eff_override: float (0, 1), if provided, overrides motor peak efficiency  
    ///     with proportional scaling.  Default of -1 has no effect.  
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

        let mc_kw_out_array: Vec<f64> =
            (Array::linspace(0.0, 1.0, self.mc_perc_out_array.len()) * self.mc_max_kw).to_vec();

        let mc_full_eff_array: Vec<f64> = self
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

        self.mc_kw_in_array = mc_kw_in_array;
        self.mc_kw_out_array = mc_kw_out_array;
        // self.mc_max_elec_in_kw = arrmax(&mc_kw_in_array);
        self.mc_full_eff_array = mc_full_eff_array;

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
        let scenario_name = String::from("2016 FORD Escape 4cyl 2WD");
        let selection: u32 = 5;
        let veh_year: u32 = 2016;
        let veh_pt_type = String::from("Conv");
        let drag_coef: f64 = 0.355;
        let frontal_area_m2: f64 = 3.066;
        let glider_kg: f64 = 1359.166;
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
        let fc_max_kw: f64 = 125.0;
        let fc_pwr_out_perc: Vec<f64> = vec![
            0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
        ];
        let fc_eff_map: Vec<f64> = vec![
            0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
        ];
        let fc_eff_type: String = String::from("SI");
        let fc_sec_to_peak_pwr: f64 = 6.0;
        let fc_base_kg: f64 = 61.0;
        let fc_kw_per_kg: f64 = 2.13;
        let min_fc_time_on: f64 = 30.0;
        let idle_fc_kw: f64 = 2.5;
        let mc_max_kw: f64 = 0.0;
        let mc_pwr_out_perc: Vec<f64> =
            vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mc_eff_map: Vec<f64> = vec![
            0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,
        ];
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
        let min_soc: f64 = 0.4;
        let max_soc: f64 = 0.8;
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
        //let small_motor_power_kw: f64 = 7.5;
        //let large_motor_power_kw: f64 = 75.0;
        // TODO: make this look more like:
        // fc_perc_out_array = np.r_[np.arange(0, 3.0, 0.1), np.arange(
        //     3.0, 7.0, 0.5), np.arange(7.0, 60.0, 1.0), np.arange(60.0, 105.0, 5.0)] / 100  # hardcoded ***
        ///////////// (Not used by below)//let fc_perc_out_array: Vec<f64> = FC_PERC_OUT_ARRAY.to_vec();
        //let max_roadway_chg_kw: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        //let charging_on: bool = false;
        //let no_elec_sys: bool = true;
        //let no_elec_aux: bool = true;
        //let modern_max: f64 = 0.95;
        let regen_a: f64 = 500.0;
        let regen_b: f64 = 0.99;
        //let mc_max_elec_in_kw: f64 = 100.0;
        //let ess_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        //let mc_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        //let fc_mass_kg: f64 = 0.0;
        // TODO: implement proper derivation for ess_mass_kg; see Vehicle.set_veh_mass(...)
        //let fs_mass_kg: f64 = 0.0;
        // DERIVED
        //let input_kw_out_array = fc_pwr_out_perc.iter().map(|&x| x * fc_max_kw).collect();
        //let fc_kw_out_array = fc_perc_out_array.iter().map(|&x| x * fc_max_kw).collect();
        //let fc_eff_array = fc_perc_out_array
        //   .iter()
        //    .map(|&x| {
        //        interpolate(
        //            &x,
        //            &Array::from(fc_pwr_out_perc.clone()),
        //            &Array::from(fc_eff_map.clone()),
        //            false,
        //        )
        //    })
        //    .collect::<Vec<_>>();
        ///////////// (Not used by below)//let mc_perc_out_array = MC_PERC_OUT_ARRAY.to_vec();
        //let mc_kw_out_array = (Array::linspace(0.0, 1.0, mc_perc_out_array.len()) * mc_max_kw).to_vec();
        //let mc_eff_array: Vec<f64> = LARGE_BASELINE_EFF
        //    .iter()
        //    .map(|&x| {
        //        interpolate(
        //            &x,
        //            &Array::from(mc_pwr_out_perc.clone()),
        //            &Array::from(mc_eff_map.clone()),
        //            false,
        //        )
        //    })
        //    .collect();
        //let mc_kw_in_array = Array::ones(mc_kw_out_array.len()).to_vec();
        //let veh_kg: f64 = 0.0;
        /*
        cargo_kg + glider_kg + trans_kg * comp_mass_multiplier
            + ess_mass_kg + mc_mass_kg + fc_mass_kg + fs_mass_kg;
        */
        //let max_trac_mps2: f64 =
        //    (wheel_coef_of_fric * drive_axle_weight_frac * veh_kg * props.a_grav_mps2
        //        / (1.0 + veh_cg_m * wheel_coef_of_fric / wheel_base_m))
        //        / (veh_kg * props.a_grav_mps2)
        //        * props.a_grav_mps2;

        Self::new(
            scenario_name,
            selection,
            veh_year,
            veh_pt_type,
            drag_coef,
            frontal_area_m2,
            glider_kg,
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
            fc_max_kw,
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            Some(mc_eff_map),
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
            min_soc,
            max_soc,
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
            //small_motor_power_kw,
            //large_motor_power_kw,
            //None,
            //charging_on,
            //no_elec_sys,
            //no_elec_aux,
            //max_roadway_chg_kw,
            //input_kw_out_array,
            //fc_kw_out_array,
            //fc_eff_array,
            //modern_max,
            //mc_eff_array,
            //mc_kw_in_array,
            //mc_kw_out_array,
            //mc_max_elec_in_kw,
            //None,
            regen_a,
            regen_b,
            //veh_kg,
            //max_trac_mps2,
            //ess_mass_kg,
            //mc_mass_kg,
            //fc_mass_kg,
            //fs_mass_kg,
            //None,
            None,
            None,
        )
        .unwrap()
    }

    pub fn from_file(filename: &str) -> Result<Self, anyhow::Error> {
        let extension = Path::new(filename)
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("");

        let file = File::open(filename)?;
        let mut veh_res: Result<RustVehicle, anyhow::Error> = match extension {
            "yaml" => Ok(serde_yaml::from_reader(file)?),
            "json" => Ok(serde_json::from_reader(file)?),
            _ => Err(anyhow!("Unsupported file extension {}", extension)),
        };
        veh_res.as_mut().unwrap().set_derived();
        veh_res
    }
    pub fn from_str(filename: &str) -> Result<Self, anyhow::Error> {
        let mut veh_res: Result<RustVehicle, anyhow::Error> = Ok(serde_json::from_str(filename)?);
        veh_res.as_mut().unwrap().set_derived();
        veh_res
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

    // test input validation by providing bad inputs, then checking
    // the produced error for the offending field names
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
        let fc_eff_map: Vec<f64> = vec![
            0.1, 0.12, 0.16, 0.22, 0.28, 0.33, 0.35, 0.36, 0.35, 0.34, 0.32, 0.3,
        ];
        let fc_eff_type: String = String::from("SI");
        let fc_sec_to_peak_pwr: f64 = 6.0;
        let fc_base_kg: f64 = 61.0;
        let fc_kw_per_kg: f64 = 2.13;
        let min_fc_time_on: f64 = 30.0;
        let idle_fc_kw: f64 = 2.5;
        let mc_max_kw: f64 = 0.0;
        let mc_pwr_out_perc: Vec<f64> =
            vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mc_eff_map: Vec<f64> = vec![
            0.12, 0.16, 0.21, 0.29, 0.35, 0.42, 0.75, 0.92, 0.93, 0.93, 0.92,
        ];
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

        // instantiate vehicle result
        let veh_result = RustVehicle::new(
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
            fc_pwr_out_perc,
            fc_eff_map,
            fc_eff_type,
            fc_sec_to_peak_pwr,
            fc_base_kg,
            fc_kw_per_kg,
            min_fc_time_on,
            idle_fc_kw,
            mc_max_kw,
            mc_pwr_out_perc,
            Some(mc_eff_map),
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
        );

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
        let validation_errs = veh_result
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
