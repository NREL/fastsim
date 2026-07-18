pub use fastsim_2::simdrivelabel::get_label_fe; // do not replicate - use cached results instead
pub use fastsim_2::simdrivelabel::LabelFe;
pub use fastsim_2::vehicle::RustVehicle;

use super::*;

// #[derive(Serialize, Deserialize)]
// pub struct LabelFe {
//     pub veh: vehicle::RustVehicle,
//     pub adj_params: AdjCoef,
//     pub lab_udds_mpgge: f64,
//     pub lab_hwy_mpgge: f64,
//     pub lab_comb_mpgge: f64,
//     pub lab_udds_kwh_per_mi: f64,
//     pub lab_hwy_kwh_per_mi: f64,
//     pub lab_comb_kwh_per_mi: f64,
//     pub adj_udds_mpgge: f64,
//     pub adj_hwy_mpgge: f64,
//     pub adj_comb_mpgge: f64,
//     pub adj_udds_kwh_per_mi: f64,
//     pub adj_hwy_kwh_per_mi: f64,
//     pub adj_comb_kwh_per_mi: f64,
//     pub adj_udds_ess_kwh_per_mi: f64,
//     pub adj_hwy_ess_kwh_per_mi: f64,
//     pub adj_comb_ess_kwh_per_mi: f64,
//     pub net_range_miles: f64,
//     pub uf: f64,
//     pub net_accel: f64,
//     pub res_found: String,
//     pub phev_calcs: Option<LabelFePHEV>,
//     pub adj_cs_comb_mpgge: Option<f64>,
//     pub adj_cd_comb_mpgge: Option<f64>,
//     pub net_phev_cd_miles: Option<f64>,
//     pub trace_miss_speed_mph: f64,
// }

// impl SerdeAPI for LabelFe {}

// #[derive(Serialize, Deserialize)]
// /// Label fuel economy values for a PHEV vehicle
// pub struct LabelFePHEV {
//     pub regen_soc_buffer: f64,
//     pub udds: PHEVCycleCalc,
//     pub hwy: PHEVCycleCalc,
// }

// impl SerdeAPI for LabelFePHEV {}

// #[derive(Serialize, Deserialize)]
// /// Struct containing vehicle attributes
// /// # Python Examples
// /// ```python
// /// import fastsim
// ///
// /// ## Load drive cycle by name
// /// cyc_py = fastsim.cycle.Cycle.from_file("udds")
// /// cyc_rust = cyc_py.to_rust()
// /// ```
// pub struct RustVehicle {
//     /// Vehicle name
//     #[serde(alias = "name")]
//     pub scenario_name: String,
//     /// Vehicle database ID
//     #[serde(skip)]
//     pub selection: u32,
//     /// Vehicle year
//     #[serde(alias = "vehModelYear")]
//     pub veh_year: u32,
//     /// Vehicle powertrain type, one of \[[CONV](CONV), [HEV](HEV), [PHEV](PHEV), [BEV](BEV)\]
//     #[serde(alias = "vehPtType")]
//     pub veh_pt_type: String,
//     /// Aerodynamic drag coefficient
//     #[serde(alias = "dragCoef")]
//     pub drag_coef: f64,
//     /// Frontal area, $m^2$
//     #[serde(alias = "frontalAreaM2")]
//     pub frontal_area_m2: f64,
//     /// Vehicle mass excluding cargo, passengers, and powertrain components, $kg$
//     #[serde(alias = "gliderKg")]
//     pub glider_kg: f64,
//     /// Vehicle center of mass height, $m$
//     /// **NOTE:** positive for FWD, negative for RWD, AWD, 4WD
//     #[serde(alias = "vehCgM")]
//     pub veh_cg_m: f64,
//     /// Fraction of weight on the drive axle while stopped
//     #[serde(alias = "driveAxleWeightFrac")]
//     pub drive_axle_weight_frac: f64,
//     /// Wheelbase, $m$
//     #[serde(alias = "wheelBaseM")]
//     pub wheel_base_m: f64,
//     /// Cargo mass including passengers, $kg$
//     #[serde(alias = "cargoKg")]
//     pub cargo_kg: f64,
//     /// Total vehicle mass, overrides mass calculation, $kg$
//     #[serde(alias = "vehOverrideKg")]
//     pub veh_override_kg: Option<f64>,
//     /// Component mass multiplier for vehicle mass calculation
//     #[serde(alias = "compMassMultiplier")]
//     pub comp_mass_multiplier: f64,
//     /// Fuel storage max power output, $kW$
//     #[serde(alias = "maxFuelStorKw")]
//     pub fs_max_kw: f64,
//     /// Fuel storage time to peak power, $s$
//     #[serde(alias = "fuelStorSecsToPeakPwr")]
//     pub fs_secs_to_peak_pwr: f64,
//     /// Fuel storage energy capacity, $kWh$
//     #[serde(alias = "fuelStorKwh")]
//     pub fs_kwh: f64,
//     /// Fuel specific energy, $\frac{kWh}{kg}$
//     #[serde(alias = "fuelStorKwhPerKg")]
//     pub fs_kwh_per_kg: f64,
//     /// Fuel converter peak continuous power, $kW$
//     #[serde(alias = "maxFuelConvKw")]
//     pub fc_max_kw: f64,
//     /// Fuel converter output power percentage map, x values of [fc_eff_map](RustVehicle::fc_eff_map)
//     #[serde(alias = "fcPwrOutPerc")]
//     pub fc_pwr_out_perc: Array1<f64>,
//     /// Fuel converter efficiency map
//     #[serde(default)]
//     pub fc_eff_map: Array1<f64>,
//     /// Fuel converter efficiency type, one of \[[SI](SI), [ATKINSON](ATKINSON), [DIESEL](DIESEL), [H2FC](H2FC), [HD_DIESEL](HD_DIESEL)\]
//     /// Used for calculating [fc_eff_map](RustVehicle::fc_eff_map), and other calculations if H2FC
//     #[serde(alias = "fcEffType")]
//     pub fc_eff_type: String,
//     /// Fuel converter time to peak power, $s$
//     #[serde(alias = "fuelConvSecsToPeakPwr")]
//     pub fc_sec_to_peak_pwr: f64,
//     /// Fuel converter base mass, $kg$
//     #[serde(alias = "fuelConvBaseKg")]
//     pub fc_base_kg: f64,
//     /// Fuel converter specific power (power-to-weight ratio), $\frac{kW}{kg}$
//     #[serde(alias = "fuelConvKwPerKg")]
//     pub fc_kw_per_kg: f64,
//     /// Minimum time fuel converter must be on before shutoff (for HEV, PHEV)
//     #[serde(alias = "minFcTimeOn")]
//     pub min_fc_time_on: f64,
//     /// Fuel converter idle power, $kW$
//     #[serde(alias = "idleFcKw")]
//     pub idle_fc_kw: f64,
//     /// Peak continuous electric motor power, $kW$
//     #[serde(alias = "mcMaxElecInKw")]
//     pub mc_max_kw: f64,
//     /// Electric motor output power percentage map, x values of [mc_eff_map](RustVehicle::mc_eff_map)
//     #[serde(alias = "mcPwrOutPerc")]
//     pub mc_pwr_out_perc: Array1<f64>,
//     /// Electric motor efficiency map
//     #[serde(alias = "mcEffArray")]
//     pub mc_eff_map: Array1<f64>,
//     /// Electric motor time to peak power, $s$
//     #[serde(alias = "motorSecsToPeakPwr")]
//     pub mc_sec_to_peak_pwr: f64,
//     /// Motor power electronics mass per power output, $\frac{kg}{kW}$
//     #[serde(alias = "mcPeKgPerKw")]
//     pub mc_pe_kg_per_kw: f64,
//     /// Motor power electronics base mass, $kg$
//     #[serde(alias = "mcPeBaseKg")]
//     pub mc_pe_base_kg: f64,
//     /// Traction battery maximum power output, $kW$
//     #[serde(alias = "maxEssKw")]
//     pub ess_max_kw: f64,
//     /// Traction battery energy capacity, $kWh$
//     #[serde(alias = "maxEssKwh")]
//     pub ess_max_kwh: f64,
//     /// Traction battery mass per energy, $\frac{kg}{kWh}$
//     #[serde(alias = "essKgPerKwh")]
//     pub ess_kg_per_kwh: f64,
//     /// Traction battery base mass, $kg$
//     #[serde(alias = "essBaseKg")]
//     pub ess_base_kg: f64,
//     /// Traction battery round-trip efficiency
//     #[serde(alias = "essRoundTripEff")]
//     pub ess_round_trip_eff: f64,
//     /// Traction battery cycle life coefficient A, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
//     #[serde(alias = "essLifeCoefA")]
//     pub ess_life_coef_a: f64,
//     /// Traction battery cycle life coefficient B, see [reference](https://web.archive.org/web/20090529194442/http://www.ocean.udel.edu/cms/wkempton/Kempton-V2G-pdfFiles/PDF%20format/Duvall-V2G-batteries-June05.pdf)
//     #[serde(alias = "essLifeCoefB")]
//     pub ess_life_coef_b: f64,
//     /// Traction battery minimum state of charge
//     #[serde(alias = "minSoc")]
//     pub min_soc: f64,
//     /// Traction battery maximum state of charge
//     #[serde(alias = "maxSoc")]
//     pub max_soc: f64,
//     /// ESS discharge effort toward max FC efficiency
//     #[serde(alias = "essDischgToFcMaxEffPerc")]
//     pub ess_dischg_to_fc_max_eff_perc: f64,
//     /// ESS charge effort toward max FC efficiency
//     #[serde(alias = "essChgToFcMaxEffPerc")]
//     pub ess_chg_to_fc_max_eff_perc: f64,
//     /// Mass moment of inertia per wheel, $kg \cdot m^2$
//     #[serde(alias = "wheelInertiaKgM2")]
//     pub wheel_inertia_kg_m2: f64,
//     /// Number of wheels
//     #[serde(alias = "numWheels")]
//     pub num_wheels: f64, // TODO: Shouldn't this just be a unsigned integer? u8 would work fine.
//     /// Rolling resistance coefficient
//     #[serde(alias = "wheelRrCoef")]
//     pub wheel_rr_coef: f64,
//     /// Wheel radius, $m$
//     #[serde(alias = "wheelRadiusM")]
//     pub wheel_radius_m: f64,
//     /// Wheel coefficient of friction
//     #[serde(alias = "wheelCoefOfFric")]
//     pub wheel_coef_of_fric: f64,
//     /// Speed where the battery reserved for accelerating is zero
//     #[serde(alias = "maxAccelBufferMph")]
//     pub max_accel_buffer_mph: f64,
//     /// Percent of usable battery energy reserved to help accelerate
//     #[serde(alias = "maxAccelBufferPercOfUseableSoc")]
//     pub max_accel_buffer_perc_of_useable_soc: f64,
//     /// Percent SOC buffer for high accessory loads during cycles with long idle time
//     #[serde(alias = "percHighAccBuf")]
//     pub perc_high_acc_buf: f64,
//     /// Speed at which the fuel converter must turn on, $mph$
//     #[serde(alias = "mphFcOn")]
//     pub mph_fc_on: f64,
//     /// Power demand above which to require fuel converter on, $kW$
//     #[serde(alias = "kwDemandFcOn")]
//     pub kw_demand_fc_on: f64,
//     /// Maximum brake regeneration efficiency
//     #[serde(alias = "maxRegen")]
//     pub max_regen: f64,
//     /// Stop/start micro-HEV flag
//     pub stop_start: bool,
//     /// Force auxiliary power load to come from fuel converter
//     #[serde(alias = "forceAuxOnFC")]
//     pub force_aux_on_fc: bool,
//     /// Alternator efficiency
//     #[serde(alias = "altEff")]
//     pub alt_eff: f64,
//     /// Charger efficiency
//     #[serde(alias = "chgEff")]
//     pub chg_eff: f64,
//     /// Auxiliary load power, $kW$
//     #[serde(alias = "auxKw")]
//     pub aux_kw: f64,
//     /// Transmission mass, $kg$
//     #[serde(alias = "transKg")]
//     pub trans_kg: f64,
//     /// Transmission efficiency
//     #[serde(alias = "transEff")]
//     pub trans_eff: f64,
//     /// Maximum acceptable ratio of change in ESS energy to expended fuel energy (used in hybrid SOC balancing), $\frac{\Delta E_{ESS}}{\Delta E_{fuel}}$
//     #[serde(alias = "essToFuelOkError")]
//     pub ess_to_fuel_ok_error: f64,
//     #[serde(default = "default_regen_a")]
//     pub regen_a: f64,
//     #[serde(default = "default_regen_b")]
//     pub regen_b: f64,
//     #[serde(default)]
//     #[serde(alias = "fcEffArray", skip_serializing)]
//     pub fc_eff_array: Vec<f64>,
// }

const fn default_regen_a() -> f64 {
    500.0
}
const fn default_regen_b() -> f64 {
    0.99
}

use include_dir::{include_dir, Dir};
pub(crate) const ASSETS_DIR: &'static Dir<'_> =
    &include_dir!("$CARGO_MANIFEST_DIR/src/compat/fastsim2/assets");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_vehicles_deserialize_and_convert() {
        // Vehicles
        ASSETS_DIR
            .get_dir("vehicles")
            .unwrap()
            .files()
            .for_each(|file| {
                let f2veh = RustVehicle::from_reader(file.contents(), "yaml", false).unwrap();
                let f3veh = Vehicle::try_from(f2veh).unwrap();
            });
    }

    // #[test]
    // fn test_all_labelfe_deserialize() {
    //     // Stored LabelFe results
    //     ASSETS_DIR
    //         .get_dir("labelfe")
    //         .unwrap()
    //         .files()
    //         .iter()
    //         .for_each(|file| {
    //             LabelFe::from_file(file.contents(), false).unwrap();
    //         });
    // }
}

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    // const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "toml", "bin"];
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml"];

    /// Specialized code to execute upon initialization
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Read (deserialize) an object from a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    ///
    /// # Arguments:
    ///
    /// * `filepath`: The filepath from which to read the object
    ///
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = File::open(filepath).with_context(|| {
            if !filepath.exists() {
                format!("File not found: {filepath:?}")
            } else {
                format!("Could not open file: {filepath:?}")
            }
        })?;
        Self::from_reader(file, extension, skip_init)
    }

    /// Deserialize an object from anything that implements [`std::io::Read`]
    ///
    /// # Arguments:
    ///
    /// * `rdr` - The reader from which to read object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn from_reader<R: std::io::Read>(
        rdr: R,
        format: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let mut deserialized: Self = match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            // "json" => serde_json::from_reader(rdr)?,
            // "toml" => {
            //     let mut buf = String::new();
            //     rdr.read_to_string(&mut buf)?;
            //     Self::from_toml(buf, skip_init)?
            // }
            // #[cfg(feature = "bincode")]
            // "bin" => bincode::deserialize_from(rdr)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        };
        if !skip_init {
            deserialized.init()?;
        }
        Ok(deserialized)
    }
}

impl SerdeAPI for RustVehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        self.set_derived()
    }
}

impl TryFrom<RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: RustVehicle) -> anyhow::Result<Self> {
        let mut f2veh = f2veh.clone();
        f2veh
            .set_derived()
            .with_context(|| anyhow!(format_dbg!()))?;
        let save_interval = Some(1);
        let pt_type = PowertrainType::try_from(&f2veh).with_context(|| anyhow!(format_dbg!()))?;

        let mut f3veh = Self {
            name: f2veh.scenario_name.clone(),
            year: f2veh.veh_year,
            doc: f2veh.doc.clone(),
            pt_type,
            chassis: Chassis::try_from(&f2veh).with_context(|| format_dbg!())?,
            cabin: Default::default(),
            hvac: Default::default(),
            pwr_aux_base: f2veh.aux_kw * uc::KW,
            state: Default::default(),
            save_interval,
            history: Default::default(),
            mass: Some(f2veh.veh_kg * uc::KG),
        };
        f3veh.expunge_mass_fields();
        f3veh.init().with_context(|| anyhow!(format_dbg!()))?;

        Ok(f3veh)
    }
}

impl TryFrom<&RustVehicle> for PowertrainType {
    type Error = anyhow::Error;
    /// Returns fastsim-3 vehicle given fastsim-2 vehicle
    ///
    /// # Arguments
    /// * `f2veh` - fastsim-2 vehicle
    fn try_from(f2veh: &RustVehicle) -> anyhow::Result<PowertrainType> {
        // TODO: implement the `_doc` fields in fastsim-3 and make sure they get carried over from fastsim-2
        // see https://github.com/NREL/fastsim/blob/fastsim-2/rust/fastsim-core/fastsim-proc-macros/src/doc_field.rs and do something similar
        match f2veh.veh_pt_type.as_str() {
            CONV => {
                let conv = ConventionalVehicle::try_from(f2veh)?;
                Ok(PowertrainType::ConventionalVehicle(Box::new(conv)))
            }
            HEV => {
                let hev = HybridElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::HybridElectricVehicle(Box::new(hev)))
            }
            PHEV => {
                let phev = HybridElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::PlugInHybridElectricVehicle(Box::new(phev)))
            }
            BEV => {
                let bev = BatteryElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::BatteryElectricVehicle(Box::new(bev)))
            }
            _ => {
                bail!(
                    "Invalid powertrain type: {}.
Expected one of {}",
                    f2veh.veh_pt_type,
                    [CONV, HEV, PHEV, BEV].join(", "),
                )
            }
        }
    }
}

impl Vehicle {
    #[allow(dead_code)]
    pub fn from_f2_file(file: PathBuf) -> anyhow::Result<Self> {
        let f2veh = RustVehicle::from_file(file, false).with_context(|| format_dbg!())?;
        Self::try_from(f2veh)
    }
}

pub(crate) const MC_PERC_OUT_ARRAY: [f64; 101] = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31,
    0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
    0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63,
    0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
    0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99, 1.,
];

// TODO: remove pub(crate) when able
pub(crate) const FUEL_LHV_MJ_PER_KG: f64 = 43.2;
pub(crate) const CONV: &str = "Conv";
pub(crate) const HEV: &str = "HEV";
pub(crate) const PHEV: &str = "PHEV";
pub(crate) const BEV: &str = "BEV";

impl TryFrom<&RustVehicle> for ConventionalVehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: &RustVehicle) -> anyhow::Result<ConventionalVehicle> {
        let conv = ConventionalVehicle {
            fs: {
                let fs = FuelStorage {
                    pwr_out_max: f2veh.fs_max_kw * uc::KW,
                    pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
                    energy_capacity: f2veh.fs_kwh * uc::KWH,
                    specific_energy: Some(FUEL_LHV_MJ_PER_KG * uc::MJ / uc::KG),
                    mass: None,
                };
                fs
            },
            fc: FuelConverter::try_from(f2veh.clone())?,
            transmission: Transmission::try_from(f2veh.clone())?,
            pt_cntrl: ConvPowertrainControls::Normal,
            dfco_cntrl: DfcoControls::default(),
            mass: None,
            alt_eff: f2veh.alt_eff * uc::R,
        };
        Ok(conv)
    }
}

impl TryFrom<RustVehicle> for Transmission {
    type Error = anyhow::Error;
    fn try_from(f2veh: RustVehicle) -> anyhow::Result<Transmission> {
        let transmission = Transmission {
            mass: None,
            eff_interp: InterpolatorEnum::new_0d(f2veh.trans_eff),
            state: Default::default(),
            history: Default::default(),
            save_interval: Some(1),
        };
        Ok(transmission)
    }
}

impl TryFrom<&RustVehicle> for BatteryElectricVehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: &RustVehicle) -> anyhow::Result<BatteryElectricVehicle> {
        let bev = BatteryElectricVehicle {
            res: ReversibleEnergyStorage::try_from(f2veh.clone()).with_context(|| format_dbg!())?,
            em: ElectricMachine {
                state: Default::default(),
                eff_interp_achieved: InterpolatorEnum::new_1d(
                    f2veh.mc_pwr_out_perc.clone(),
                    f2veh.mc_eff_array.clone(),
                    strategy::Linear,
                    Extrapolate::Error,
                )?,
                eff_interp_at_max_input: Some(InterpolatorEnum::new_1d(
                    // before adding the interpolator, pwr_in_frac_interp was set as Default::default(), can this
                    // be transferred over as done here, or does a new defualt need to be defined?
                    f2veh
                        .mc_pwr_out_perc
                        .iter()
                        .zip(f2veh.mc_eff_array.iter())
                        .map(|(x, y)| x / y)
                        .collect(),
                    f2veh.mc_eff_array.clone(),
                    strategy::Linear,
                    Extrapolate::Error,
                )?),
                pwr_out_max: f2veh.mc_max_kw * uc::KW,
                specific_pwr: None,
                mass: None,
                save_interval: Some(1),
                history: Default::default(),
            },
            transmission: Transmission::try_from(f2veh.clone())?,
            mass: None,
        };
        Ok(bev)
    }
}

impl TryFrom<&RustVehicle> for Chassis {
    type Error = anyhow::Error;
    fn try_from(f2veh: &RustVehicle) -> anyhow::Result<Self> {
        let drive_type = if f2veh.drive_axle_weight_frac > 0.9 {
            chassis::DriveTypes::AWD
        } else if f2veh.veh_cg_m < 0. {
            chassis::DriveTypes::RWD
        } else {
            chassis::DriveTypes::FWD
        };

        Ok(Self {
            drag_coef: f2veh.drag_coef * uc::R,
            frontal_area: f2veh.frontal_area_m2 * uc::M2,
            cg_height: f2veh.veh_cg_m.abs() * uc::M,
            wheel_fric_coef: f2veh.wheel_coef_of_fric * uc::R,
            drive_type,
            drive_axle_weight_frac: f2veh.drive_axle_weight_frac * uc::R,
            wheel_base: f2veh.wheel_base_m * uc::M,
            wheel_inertia: f2veh.wheel_inertia_kg_m2 * uc::KGM2,
            wheel_rr_coef: f2veh.wheel_rr_coef * uc::R,
            num_wheels: f2veh.num_wheels as u8,
            wheel_radius: Some(f2veh.wheel_radius_m * uc::M),
            tire_code: None,
            mass: None,
            glider_mass: Some(f2veh.glider_kg * uc::KG),
            cargo_mass: Some(f2veh.cargo_kg * uc::KG),
        })
    }
}

impl TryFrom<&RustVehicle> for HybridElectricVehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: &RustVehicle) -> anyhow::Result<HybridElectricVehicle> {
        let pt_cntrl = HEVPowertrainControls::RGWDB(Box::new(hev::RESGreedyWithDynamicBuffers {
            speed_soc_fc_on_buffer: None,
            speed_soc_fc_on_buffer_coeff: None,
            speed_soc_disch_buffer: None,
            speed_soc_disch_buffer_coeff: None,
            speed_soc_regen_buffer: None,
            speed_soc_regen_buffer_coeff: None,
            // note that this exists in `fastsim-2` but has no apparent effect!
            fc_min_time_on: None,
            speed_fc_forced_on: Some(f2veh.mph_fc_on * uc::MPH),
            frac_pwr_demand_fc_forced_on: Some(
                f2veh.kw_demand_fc_on / (f2veh.fc_max_kw + f2veh.ess_max_kw.min(f2veh.mc_max_kw))
                    * uc::R,
            ),
            frac_of_most_eff_pwr_to_run_fc: None,
            temp_fc_forced_on: None,
            temp_fc_allowed_off: None,
            save_interval: Some(1),
            state: Default::default(),
            history: Default::default(),
        }));
        let mut hev = HybridElectricVehicle {
            fs: FuelStorage {
                pwr_out_max: f2veh.fs_max_kw * uc::KW,
                pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
                energy_capacity: f2veh.fs_kwh * 3.6 * uc::MJ,
                specific_energy: None,
                mass: None,
            },
            fc: FuelConverter::try_from(f2veh.clone())?,
            res: ReversibleEnergyStorage::try_from(f2veh.clone()).with_context(|| format_dbg!())?,
            em: ElectricMachine::try_from(f2veh.clone())?,
            transmission: Transmission::try_from(f2veh.clone())?,
            pt_cntrl,
            mass: None,
            sim_params: Default::default(),
            aux_cntrl: Default::default(),
            soc_bal_iter_history: Default::default(),
            soc_bal_iters: Default::default(),
        };
        hev.init()?;
        Ok(hev)
    }
}

impl TryFrom<RustVehicle> for ElectricMachine {
    type Error = anyhow::Error;
    fn try_from(f2veh: RustVehicle) -> Result<ElectricMachine, anyhow::Error> {
        Ok(powertrain::electric_machine::EMBuilder {
            eff_interp_achieved: {
                // fastsim-2's hard-coded short vector of percent of peak power
                let short_perc_out_vec =
                    vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
                // `InterpolatorEnum` for fastsim-3
                InterpolatorEnum::new_1d(
                    short_perc_out_vec.clone().into(),
                    {
                        // convert 101 element f2 array to shorter f2 array and use
                        // linear rather than left-nearest interpolation
                        let mc_full_eff = Array1::from_vec(f2veh.mc_full_eff_array.clone());
                        ensure!(mc_full_eff.len() == 101);
                        let shortener = Interp1D::new(
                            MC_PERC_OUT_ARRAY.to_vec().into(),
                            mc_full_eff,
                            strategy::Linear,
                            Extrapolate::Error,
                        )
                        .with_context(|| format_dbg!())?;
                        let mut short_eff: Vec<f64> = short_perc_out_vec
                            .iter()
                            .map(|x| shortener.interpolate(&[*x]).unwrap())
                            .collect();
                        short_eff[0] = short_eff[1];
                        short_eff.into()
                    },
                    strategy::Linear,
                    Extrapolate::Error,
                )
            }
            .with_context(|| {
                format!(
                    "{}\n{}",
                    format_dbg!(f2veh.mc_full_eff_array.len()),
                    format_dbg!(f2veh.mc_perc_out_array.len())
                )
            })?,
            pwr_out_max: f2veh.mc_max_kw * uc::KW,
        }
        .try_into()
        .with_context(|| format_dbg!())?)
    }
}

impl TryFrom<RustVehicle> for FuelConverter {
    type Error = anyhow::Error;
    fn try_from(f2veh: RustVehicle) -> Result<FuelConverter, anyhow::Error> {
        let mut fc: FuelConverter = powertrain::fuel_converter::FCBuilder {
            pwr_out_max: f2veh.fc_max_kw * uc::KW,
            pwr_ramp_lag: f2veh.fc_sec_to_peak_pwr * uc::S,
            eff_interp_from_pwr_out: InterpolatorEnum::new_1d(
                // hard-coded vec from fastsim-2
                vec![
                    0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0,
                ]
                .into(),
                f2veh.fc_eff_map.clone().into(),
                strategy::Linear,
                Extrapolate::Error,
            )
            .with_context(|| format_dbg!())?,
            pwr_for_peak_eff: uc::KW * f64::NAN, // this gets updated in `init`
            // this means that aux power must include idle fuel
            pwr_idle_fuel: si::Power::ZERO,
            save_interval: Some(1),
        }
        .try_into()
        .with_context(|| format_dbg!())?;
        fc.init()?;
        Ok(fc)
    }
}

impl TryFrom<RustVehicle> for ReversibleEnergyStorage {
    type Error = anyhow::Error;
    fn try_from(f2veh: RustVehicle) -> anyhow::Result<ReversibleEnergyStorage> {
        let f3_res = ReversibleEnergyStorage {
            thrml: Default::default(),
            state: Default::default(),
            mass: None,
            specific_energy: None,
            pwr_out_max: f2veh.ess_max_kw * uc::KW,
            energy_capacity: f2veh.ess_max_kwh * uc::KWH,
            eff_interp: powertrain::reversible_energy_storage::EffInterp::Constant(Interp0D::new(
                f2veh.ess_round_trip_eff.sqrt(),
            )),
            min_soc: f2veh.min_soc * uc::R,
            max_soc: f2veh.max_soc * uc::R,
            save_interval: Some(1),
            history: Default::default(),
        };
        Ok(f3_res)
    }
}
