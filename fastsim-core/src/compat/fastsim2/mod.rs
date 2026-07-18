use fastsim_core::traits::SerdeAPI;

use super::*;
pub use fastsim_2 as fastsim_core;

use include_dir::{include_dir, Dir};
pub const ASSETS_DIR: &'static Dir<'_> =
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
                let f2veh =
                    fastsim_core::vehicle::RustVehicle::from_reader(file.contents(), "yaml", false)
                        .unwrap();
                let f3veh = Vehicle::try_from(f2veh).unwrap();
            });
    }
}

impl TryFrom<fastsim_core::vehicle::RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_core::vehicle::RustVehicle) -> anyhow::Result<Self> {
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

impl TryFrom<&fastsim_core::vehicle::RustVehicle> for PowertrainType {
    type Error = anyhow::Error;
    /// Returns fastsim-3 vehicle given fastsim-2 vehicle
    ///
    /// # Arguments
    /// * `f2veh` - fastsim-2 vehicle
    fn try_from(f2veh: &fastsim_core::vehicle::RustVehicle) -> anyhow::Result<PowertrainType> {
        // TODO: implement the `_doc` fields in fastsim-3 and make sure they get carried over from fastsim-2
        // see https://github.com/NREL/fastsim/blob/fastsim-2/rust/fastsim-core/fastsim-proc-macros/src/doc_field.rs and do something similar
        match f2veh.veh_pt_type.as_str() {
            fastsim_core::vehicle::CONV => {
                let conv = ConventionalVehicle::try_from(f2veh)?;
                Ok(PowertrainType::ConventionalVehicle(Box::new(conv)))
            }
            fastsim_core::vehicle::HEV => {
                let hev = HybridElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::HybridElectricVehicle(Box::new(hev)))
            }
            fastsim_core::vehicle::PHEV => {
                let phev = HybridElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::PlugInHybridElectricVehicle(Box::new(phev)))
            }
            fastsim_core::vehicle::BEV => {
                let bev = BatteryElectricVehicle::try_from(f2veh)?;
                Ok(PowertrainType::BatteryElectricVehicle(Box::new(bev)))
            }
            _ => {
                bail!(
                    "Invalid powertrain type: {}.
Expected one of {}",
                    f2veh.veh_pt_type,
                    [
                        fastsim_core::vehicle::CONV,
                        fastsim_core::vehicle::HEV,
                        fastsim_core::vehicle::PHEV,
                        fastsim_core::vehicle::BEV
                    ]
                    .join(", "),
                )
            }
        }
    }
}

impl Vehicle {
    #[allow(dead_code)]
    pub fn from_f2_file(file: PathBuf) -> anyhow::Result<Self> {
        let f2veh = fastsim_core::vehicle::RustVehicle::from_file(file, false)
            .with_context(|| format_dbg!())?;
        Self::try_from(f2veh)
    }
}

pub const FUEL_LHV_MJ_PER_KG: f64 = 43.2;

impl TryFrom<&fastsim_core::vehicle::RustVehicle> for ConventionalVehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: &fastsim_core::vehicle::RustVehicle) -> anyhow::Result<ConventionalVehicle> {
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

impl TryFrom<fastsim_core::vehicle::RustVehicle> for Transmission {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_core::vehicle::RustVehicle) -> anyhow::Result<Transmission> {
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

impl TryFrom<&fastsim_core::vehicle::RustVehicle> for BatteryElectricVehicle {
    type Error = anyhow::Error;
    fn try_from(
        f2veh: &fastsim_core::vehicle::RustVehicle,
    ) -> anyhow::Result<BatteryElectricVehicle> {
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

impl TryFrom<&fastsim_core::vehicle::RustVehicle> for Chassis {
    type Error = anyhow::Error;
    fn try_from(f2veh: &fastsim_core::vehicle::RustVehicle) -> anyhow::Result<Self> {
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

impl TryFrom<&fastsim_core::vehicle::RustVehicle> for HybridElectricVehicle {
    type Error = anyhow::Error;
    fn try_from(
        f2veh: &fastsim_core::vehicle::RustVehicle,
    ) -> anyhow::Result<HybridElectricVehicle> {
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

impl TryFrom<fastsim_core::vehicle::RustVehicle> for ElectricMachine {
    type Error = anyhow::Error;
    fn try_from(
        f2veh: fastsim_core::vehicle::RustVehicle,
    ) -> Result<ElectricMachine, anyhow::Error> {
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
                            fastsim_core::params::MC_PERC_OUT_ARRAY.to_vec().into(),
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

impl TryFrom<fastsim_core::vehicle::RustVehicle> for FuelConverter {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_core::vehicle::RustVehicle) -> Result<FuelConverter, anyhow::Error> {
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

impl TryFrom<fastsim_core::vehicle::RustVehicle> for ReversibleEnergyStorage {
    type Error = anyhow::Error;
    fn try_from(
        f2veh: fastsim_core::vehicle::RustVehicle,
    ) -> anyhow::Result<ReversibleEnergyStorage> {
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
