use super::*;
use crate::vehicle::powertrain::reversible_energy_storage::EffInterp as ResEffInterp;

impl TryFrom<fastsim_2::vehicle::RustVehicle> for Vehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: fastsim_2::vehicle::RustVehicle) -> anyhow::Result<Self> {
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

impl TryFrom<&fastsim_2::vehicle::RustVehicle> for PowertrainType {
    type Error = anyhow::Error;
    /// Returns fastsim-3 vehicle given fastsim-2 vehicle
    ///
    /// # Arguments
    /// * `f2veh` - fastsim-2 vehicle
    fn try_from(f2veh: &fastsim_2::vehicle::RustVehicle) -> anyhow::Result<PowertrainType> {
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
    /// Function to convert back to fastsim-2 format.  Note that this is
    /// probably not 100% reliable.
    pub fn to_fastsim2(&self) -> anyhow::Result<fastsim_2::vehicle::RustVehicle> {
        let mut veh = fastsim_2::vehicle::RustVehicle {
            alt_eff: match &self.pt_type {
                PowertrainType::ConventionalVehicle(conv) => conv.alt_eff.get::<si::ratio>(),
                _ => 1.0,
            },
            alt_eff_doc: None,
            aux_kw: self.pwr_aux_base.get::<si::kilowatt>(),
            aux_kw_doc: None,
            cargo_kg: self
                .chassis
                .cargo_mass
                .unwrap_or_default()
                .get::<si::kilogram>(),
            cargo_kg_doc: None,
            charging_on: false,
            chg_eff: 0.86, // TODO: revisit?
            chg_eff_doc: None,
            comp_mass_multiplier: 1.4,
            comp_mass_multiplier_doc: None,
            // TODO: replace with `doc` field once implemented in fastsim-3
            doc: None,
            drag_coef: self.chassis.drag_coef.get::<si::ratio>(),
            drag_coef_doc: None,
            drive_axle_weight_frac: self.chassis.drive_axle_weight_frac.get::<si::ratio>(),
            drive_axle_weight_frac_doc: None,
            ess_base_kg: 75.0, // NOTE: this hardcoded value could cause trouble
            ess_base_kg_doc: None,
            ess_chg_to_fc_max_eff_perc: 0.0, // TODO: ??? update later
            ess_chg_to_fc_max_eff_perc_doc: None,
            ess_dischg_to_fc_max_eff_perc: 0.0, // TODO: ??? update later
            ess_dischg_to_fc_max_eff_perc_doc: None,
            ess_kg_per_kwh: 8.0, // TODO: revisit
            ess_kg_per_kwh_doc: None,
            ess_life_coef_a: 110.,
            ess_life_coef_a_doc: None,
            ess_life_coef_b: -0.6811,
            ess_life_coef_b_doc: None,
            ess_mass_kg: self.res().map_or(anyhow::Ok(0.), |res| {
                Ok(res.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            ess_max_kw: self
                .res()
                .map(|res| res.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            ess_max_kw_doc: None,
            ess_max_kwh: self
                .res()
                .map(|res| res.energy_capacity.get::<si::kilowatt_hour>())
                .unwrap_or_default(),
            ess_max_kwh_doc: None,
            ess_round_trip_eff: self
                .res()
                .map(|res| {
                    if let ResEffInterp::Constant(Interp0D(eff)) = res.eff_interp {
                        Ok(eff.powi(2))
                    } else {
                        bail!("`to_fastsim2` is not implemented for non-0D `res.eff_interp`")
                    }
                })
                .transpose()?
                .unwrap_or(f64::NAN),
            ess_round_trip_eff_doc: None,
            ess_to_fuel_ok_error: 0.005, // TODO: update when hybrid logic is implemented
            ess_to_fuel_ok_error_doc: None,
            fc_base_kg: 61.0, // TODO: revisit
            fc_base_kg_doc: None,
            fc_eff_array: Default::default(),
            fc_eff_map: self
                .fc()
                .map(|fc| match &fc.eff_interp_from_pwr_out {
                    InterpolatorEnum::Interp1D(interp) => Ok(interp.data.values.clone()),
                    _ => bail!(format_dbg!(
                        "Only 1-D interpolators can be converted to FASTSim 2"
                    )),
                })
                .transpose()?
                .unwrap_or_else(|| array![0., 0.]),
            fc_eff_map_doc: None,
            fc_eff_type: match &self.pt_type {
                PowertrainType::ConventionalVehicle(_) => "SI".into(),
                PowertrainType::HybridElectricVehicle(_) => "Atkinson".into(),
                PowertrainType::PlugInHybridElectricVehicle(_) => "Atkinson".into(),
                PowertrainType::BatteryElectricVehicle(_) => "SI".into(),
            },
            fc_eff_type_doc: None,
            fc_kw_out_array: Default::default(),
            fc_kw_per_kg: 2.13, // TODO: revisit
            fc_kw_per_kg_doc: None,
            fc_mass_kg: self.fc().map_or(anyhow::Ok(0.), |fc| {
                Ok(fc.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            fc_max_kw: self
                .fc()
                .map(|fc| fc.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            fc_max_kw_doc: None,
            fc_peak_eff_override: None,
            fc_peak_eff_override_doc: None,
            fc_perc_out_array: Default::default(),
            fc_pwr_out_perc: self
                .fc()
                .map(|fc| match &fc.eff_interp_from_pwr_out {
                    InterpolatorEnum::Interp1D(interp) => Ok(interp.data.grid[0].clone()),
                    _ => bail!(format_dbg!(
                        "Only 1-D interpolators can be converted to FASTSim 2"
                    )),
                })
                .transpose()?
                .unwrap_or_else(|| array![0., 1.]),
            fc_pwr_out_perc_doc: None,
            fc_sec_to_peak_pwr: self
                .fc()
                .map(|fc| fc.pwr_ramp_lag.get::<si::second>())
                .unwrap_or_default(),
            fc_sec_to_peak_pwr_doc: None,
            force_aux_on_fc: matches!(self.pt_type, PowertrainType::ConventionalVehicle(_)),
            force_aux_on_fc_doc: None,
            frontal_area_m2: self.chassis.frontal_area.get::<si::square_meter>(),
            frontal_area_m2_doc: None,
            fs_kwh: self
                .fs()
                .map(|fs| fs.energy_capacity.get::<si::kilowatt_hour>())
                .unwrap_or_default(),
            fs_kwh_doc: None,
            fs_kwh_per_kg: self
                .fs()
                .and_then(|fs| fs.specific_energy)
                .map(|specific_energy| specific_energy.get::<si::kilojoule_per_kilogram>() / 3600.)
                .unwrap_or_default(),
            fs_kwh_per_kg_doc: None,
            fs_mass_kg: self.fs().map_or(anyhow::Ok(0.), |fs| {
                Ok(fs.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            fs_max_kw: self
                .fs()
                .map(|fs| fs.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            fs_max_kw_doc: None,
            fs_secs_to_peak_pwr: self
                .fs()
                .map(|fs| fs.pwr_ramp_lag.get::<si::second>())
                .unwrap_or_default(),
            fs_secs_to_peak_pwr_doc: None,
            glider_kg: self
                .chassis
                .glider_mass
                .unwrap_or_default()
                .get::<si::kilogram>(),
            glider_kg_doc: None,
            // 4.3 is for 2016 Toyota Prius Two, not sure this matters, though
            idle_fc_kw: 4.3,
            idle_fc_kw_doc: None,
            input_kw_out_array: Default::default(), // calculated in `set_derived()`
            kw_demand_fc_on: match &self.pt_type {
                PowertrainType::HybridElectricVehicle(hev) => match &hev.pt_cntrl {
                    HEVPowertrainControls::RGWDB(rgwb) => (rgwb
                        .frac_pwr_demand_fc_forced_on
                        .with_context(|| format_dbg!("Expected `Some`."))?
                        * (hev.fc.pwr_out_max + hev.res.pwr_out_max.min(hev.em.pwr_out_max)))
                    .get::<si::kilowatt>(),
                    HEVPowertrainControls::StopStart(_) => 0.0,
                },
                _ => 0.0,
            },
            kw_demand_fc_on_doc: None,
            large_motor_power_kw: 75.0,
            max_accel_buffer_mph: 60.0, // TODO: placeholder, revisit
            max_accel_buffer_mph_doc: None,
            max_accel_buffer_perc_of_useable_soc: 0.2, // TODO: placeholder, revisit
            max_accel_buffer_perc_of_useable_soc_doc: None,
            max_regen: 0.98, // TODO: placeholder, revisit
            max_regen_doc: None,
            max_regen_kwh: Default::default(),
            max_roadway_chg_kw: Default::default(),
            max_soc: self
                .res()
                .map(|res| res.max_soc.get::<si::ratio>())
                .unwrap_or_else(|| 1.0),
            max_soc_doc: None,
            max_trac_mps2: Default::default(),
            mc_eff_array: Default::default(), // calculated in `set_derived`
            mc_eff_map: self
                .em()
                .map(|em| match &em.eff_interp_achieved {
                    InterpolatorEnum::Interp1D(interp) => Ok(interp.data.values.clone()),
                    _ => bail!(format_dbg!(
                        "Only 1-D interpolators can be converted to FASTSim 2"
                    )),
                })
                .transpose()?
                .map(|f_x| f_x.to_vec())
                .unwrap_or_else(|| vec![0., 1.])
                .into(),
            mc_eff_map_doc: None,
            mc_full_eff_array: Default::default(), // calculated in `set_derived`
            mc_kw_in_array: Default::default(),    // calculated in `set_derived`
            mc_kw_out_array: Default::default(),   // calculated in `set_derived`
            mc_mass_kg: self.em().map_or(anyhow::Ok(0.), |em| {
                Ok(em.mass()?.unwrap_or_default().get::<si::kilogram>())
            })?,
            mc_max_elec_in_kw: Default::default(), // calculated in `set_derived`
            mc_max_kw: self
                .em()
                .map(|em| em.pwr_out_max.get::<si::kilowatt>())
                .unwrap_or_default(),
            mc_max_kw_doc: None,
            mc_pe_base_kg: 0.0, // placeholder, TODO: review when implementing xEVs
            mc_pe_base_kg_doc: None,
            mc_pe_kg_per_kw: 0.833, // placeholder, TODO: review when implementing xEVs
            mc_pe_kg_per_kw_doc: None,
            mc_peak_eff_override: Default::default(),
            mc_peak_eff_override_doc: None,
            mc_perc_out_array: Default::default(),
            // short array that can use xEV when implented.  TODO: fix this when implementing xEV
            mc_pwr_out_perc: self
                .em()
                .map(|em| match &em.eff_interp_achieved {
                    InterpolatorEnum::Interp1D(interp) => Ok(interp.data.grid[0].clone()),
                    _ => bail!(format_dbg!(
                        "Only 1-D interpolators can be converted to FASTSim 2"
                    )),
                })
                .transpose()?
                .map(|x| x.to_vec())
                .unwrap_or_else(|| vec![0., 1.])
                .into(),
            mc_pwr_out_perc_doc: None,
            // 4.8 is hardcoded for 2016 Toyota Prius Two
            mc_sec_to_peak_pwr: 4.8,
            mc_sec_to_peak_pwr_doc: None,
            // NOTE: this seems to have no effect in fastsim-2 so it can be anything
            min_fc_time_on: 60.0,
            min_fc_time_on_doc: None,
            min_soc: self
                .res()
                .map(|res| res.min_soc.get::<si::ratio>())
                .unwrap_or_default(),
            min_soc_doc: None,
            modern_max: 0.95,
            mph_fc_on: match &self.pt_type {
                PowertrainType::HybridElectricVehicle(hev) => match &hev.pt_cntrl {
                    HEVPowertrainControls::RGWDB(rgwb) => rgwb
                        .speed_fc_forced_on
                        .with_context(|| format_dbg!("Expected Some"))?
                        .get::<si::mile_per_hour>(),
                    HEVPowertrainControls::StopStart(_) => 0.0,
                },
                _ => 0.0,
            },
            mph_fc_on_doc: None,
            no_elec_aux: false, // TODO: revisit when implemementing HEV
            no_elec_sys: false, // TODO: revisit when implemementing HEV
            num_wheels: self.chassis.num_wheels as f64,
            num_wheels_doc: None,
            orphaned: false,
            perc_high_acc_buf: Default::default(), // TODO: revisit when implemementing HEV
            perc_high_acc_buf_doc: None,
            props: fastsim_2::params::RustPhysicalProperties::default(),
            regen_a: 500.0, //TODO: placeholder
            regen_b: 0.99,  //TODO: placeholder
            scenario_name: self.name.clone(),
            selection: 0, // there is no equivalent in fastsim-3
            small_motor_power_kw: 7.5,
            stop_start: false, // TODO: revisit when implemementing mild hybrids and stop/start vehicles
            stop_start_doc: None,
            trans_eff: {
                match self
                    .trans()
                    .cloned()
                    .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))?
                    .eff_interp
                {
                    InterpolatorEnum::Interp0D(eff) => eff.0,
                    _ => todo!(),
                }
            },
            trans_eff_doc: None,
            trans_kg: 114.0, // TODO: replace with actual transmission mass
            trans_kg_doc: None,
            val0_to60_mph: f64::NAN,
            val_cd_range_mi: f64::NAN,
            val_comb_kwh_per_mile: f64::NAN,
            val_comb_mpgge: f64::NAN,
            val_const45_mph_kwh_per_mile: f64::NAN,
            val_const55_mph_kwh_per_mile: f64::NAN,
            val_const60_mph_kwh_per_mile: f64::NAN,
            val_const65_mph_kwh_per_mile: f64::NAN,
            val_ess_life_miles: f64::NAN,
            val_hwy_kwh_per_mile: f64::NAN,
            val_hwy_mpgge: f64::NAN,
            val_msrp: f64::NAN,
            val_range_miles: f64::NAN,
            val_udds_kwh_per_mile: f64::NAN,
            val_udds_mpgge: f64::NAN,
            val_unadj_hwy_kwh_per_mile: f64::NAN,
            val_unadj_udds_kwh_per_mile: f64::NAN,
            val_veh_base_cost: f64::NAN,
            veh_cg_m: self.chassis.cg_height.get::<si::meter>()
                * match self.chassis.drive_type {
                    chassis::DriveTypes::FWD => 1.0,
                    chassis::DriveTypes::RWD
                    | chassis::DriveTypes::AWD
                    | chassis::DriveTypes::FourWD => -1.0,
                },
            veh_cg_m_doc: None,
            veh_kg: self
                .mass()?
                .context("Vehicle mass is `None`")?
                .get::<si::kilogram>(),
            veh_override_kg: self.mass()?.map(|m| m.get::<si::kilogram>()),
            veh_override_kg_doc: None,
            veh_pt_type: match &self.pt_type {
                PowertrainType::ConventionalVehicle(_) => "Conv".into(),
                PowertrainType::HybridElectricVehicle(_) => "HEV".into(),
                PowertrainType::PlugInHybridElectricVehicle(_) => "PHEV".into(),
                PowertrainType::BatteryElectricVehicle(_) => "BEV".into(),
            },
            veh_year: self.year,
            wheel_base_m: self.chassis.wheel_base.get::<si::meter>(),
            wheel_base_m_doc: None,
            wheel_coef_of_fric: self.chassis.wheel_fric_coef.get::<si::ratio>(),
            wheel_coef_of_fric_doc: None,
            wheel_inertia_kg_m2: self
                .chassis
                .wheel_inertia
                .get::<si::kilogram_square_meter>(),
            wheel_inertia_kg_m2_doc: None,
            wheel_radius_m: self.chassis.wheel_radius.unwrap().get::<si::meter>(),
            wheel_radius_m_doc: None,
            wheel_rr_coef: self.chassis.wheel_rr_coef.get::<si::ratio>(),
            wheel_rr_coef_doc: None,
        };
        veh.set_derived().with_context(|| anyhow!(format_dbg!()))?;
        Ok(veh)
    }
}
