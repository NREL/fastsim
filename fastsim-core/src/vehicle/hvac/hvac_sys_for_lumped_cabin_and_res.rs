use super::*;

#[fastsim_api(
    #[staticmethod]
    #[pyo3(name = "default")]
    fn default_py() -> Self {
        Default::default()
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods)]
#[serde(deny_unknown_fields)]
/// HVAC system for `LumpedCabin` and `ReversibleEnergyStorage::thrml`
pub struct HVACSystemForLumpedCabinAndRES {
    /// set point temperature
    pub te_set_cab: Option<si::Temperature>,
    /// Deadband half range.  Any cabin temperature within `te_deadband_cab` of
    /// `te_set_cab` results in no HVAC power draw
    pub te_deadband_cab: si::TemperatureInterval,
    /// HVAC proportional gain for cabin
    pub p_cabin: si::ThermalConductance,
    /// HVAC integral gain [W / K / s] for cabin, resets at zero crossing events  
    /// NOTE: `uom` crate does not have this unit, but it may be possible to make a custom unit for this
    pub i_cabin: f64,
    /// value at which state.i stops accumulating for cabin
    pub pwr_i_max_cabin: si::Power,
    /// HVAC derivative gain [W / K * s] for cabin
    /// NOTE: `uom` crate does not have this unit, but it may be possible to make a custom unit for this
    pub d_cabin: f64,
    /// set point temperature
    pub te_set_res: Option<si::Temperature>,
    /// Deadband half range.  Any res temperature within `te_deadband_res` of
    /// `te_set_res` results in no HVAC power draw
    pub te_deadband_res: si::TemperatureInterval,
    /// max HVAC thermal power
    /// HVAC proportional gain for [ReversibleEnergyStorage]
    pub p_res: si::ThermalConductance,
    /// HVAC integral gain [W / K / s] for [ReversibleEnergyStorage], resets at zero crossing events  
    /// NOTE: `uom` crate does not have this unit, but it may be possible to make a custom unit for this
    pub i_res: f64,
    /// value at which state.i stops accumulating for [ReversibleEnergyStorage]
    pub pwr_i_max_res: si::Power,
    /// HVAC derivative gain [W / K * s] for [ReversibleEnergyStorage]
    /// NOTE: `uom` crate does not have this unit, but it may be possible to make a custom unit for this
    pub d_res: f64,
    /// max HVAC thermal power
    pub pwr_thrml_max: si::Power,
    /// coefficient between 0 and 1 to calculate HVAC efficiency by multiplying by
    /// coefficient of performance (COP)
    pub frac_of_ideal_cop: f64,
    /// cabin heat source
    pub cabin_heat_source: CabinHeatSource,
    /// res heat source
    pub res_heat_source: RESHeatSource,
    /// res cooling source
    pub res_cooling_source: RESCoolingSource,
    /// max allowed aux load for cabin thermal management
    pub pwr_aux_for_hvac_cab_max: si::Power,
    /// max allowed aux load for res thermal management
    pub pwr_aux_for_hvac_res_max: si::Power,
    /// coefficient of performance of vapor compression cycle
    #[serde(default)]
    pub state: HVACSystemForLumpedCabinAndRESState,
    #[serde(
        default,
        skip_serializing_if = "HVACSystemForLumpedCabinAndRESStateHistoryVec::is_empty"
    )]
    pub history: HVACSystemForLumpedCabinAndRESStateHistoryVec,
    pub save_interval: Option<usize>,
}
impl Default for HVACSystemForLumpedCabinAndRES {
    fn default() -> Self {
        Self {
            te_set_cab: Some(*TE_STD_AIR),
            te_deadband_cab: 1.5 * uc::KELVIN_INT,
            p_cabin: Default::default(),
            i_cabin: Default::default(),
            d_cabin: Default::default(),
            pwr_i_max_cabin: 5. * uc::KW,
            te_set_res: Some(*TE_STD_AIR + 8.0 * uc::KELVIN_INT),
            te_deadband_res: 8.0 * uc::KELVIN_INT,
            p_res: Default::default(),
            i_res: Default::default(),
            d_res: Default::default(),
            pwr_i_max_res: 5. * uc::KW,
            pwr_thrml_max: 10. * uc::KW,
            frac_of_ideal_cop: 0.15,
            cabin_heat_source: CabinHeatSource::ResistanceHeater,
            res_heat_source: RESHeatSource::ResistanceHeater,
            res_cooling_source: RESCoolingSource::HVAC,
            pwr_aux_for_hvac_cab_max: uc::KW * 5.,
            pwr_aux_for_hvac_res_max: uc::KW * 5.,
            state: Default::default(),
            history: Default::default(),
            save_interval: Some(1),
        }
    }
}
impl SetCumulative for HVACSystemForLumpedCabinAndRES {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}
impl Init for HVACSystemForLumpedCabinAndRES {}
impl SerdeAPI for HVACSystemForLumpedCabinAndRES {}
impl HistoryMethods for HVACSystemForLumpedCabinAndRES {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear();
    }
}

impl HVACSystemForLumpedCabinAndRES {
    /// # Arguments
    /// - `te_amb_air`: ambient air temperature
    /// - `te_fc`: [FuelConverter] temperature, if equipped
    /// - `cab_state`: [LumpedCabinState]
    /// - `cab_heat_cap`: [LumpedCabinState] heat capacity
    /// - `res_temp`: [ReversibleEnergyStorage] temperatures at current and previous time step
    /// - `dt`: simulation time step size
    ///
    /// # Returns
    /// - `pwr_thrml_hvac_to_cabin`: thermal power flowing from [Vehicle::hvac] system to cabin  
    /// - `pwr_thrml_fc_to_cabin`: thermal power flowing from [FuelConverter] to cabin  
    /// - `pwr_thrml_hvac_to_res`: thermal power flowing from [Vehicle::hvac] system to
    ///     [ReversibleEnergyStorage] `thrml` system  
    ///
    /// # Assumptions and Caveats
    /// - Cabin cooling never occurs concurrently with battery heating
    /// - Cabin heating never occurs concurrently with battery cooling
    /// - For real vehicles, control parameters for battery heating and cooling
    ///   are generally different during charging, and we do not currently account for
    ///   that
    #[allow(clippy::too_many_arguments)] // the order is reasonably protected by typing
    pub fn solve(
        &mut self,
        te_amb_air: si::Temperature,
        te_fc: Option<si::Temperature>,
        cab_state: LumpedCabinState,
        cab_heat_cap: si::HeatCapacity,
        res_thrml_state: RESLumpedThermalState,
        dt: si::Time,
    ) -> anyhow::Result<(si::Power, si::Power, si::Power)> {
        let (res_temp, res_temp_prev) = (res_thrml_state.temperature, res_thrml_state.temp_prev);
        ensure!(!res_temp.is_nan(), format_dbg!(res_temp));
        ensure!(!res_temp_prev.is_nan(), format_dbg!(res_temp_prev));

        let (te_cab_delta_vs_set, te_cab_delta_vs_amb, te_res_delta_vs_set, te_res_delta_vs_amb) =
            self.get_te_deltas(te_amb_air, cab_state, res_thrml_state);

        ensure!(
            te_res_delta_vs_set.is_none() || !(self.res_cooling_source.is_none() && self.res_heat_source.is_none()),
            format!(
                "{}\n{}",
                format_dbg!(),
                "If RES set temperature is provided, either `res_cooling_source` and/or `res_heat_source` must not be None"
            )
        );

        let (te_ref, te_ref_delta_vs_set, te_ref_delta_vs_amb) =
            self.state.mode.set_mode_and_get_te_for_cop(
                cab_state,
                res_thrml_state,
                (
                    te_cab_delta_vs_set,
                    te_cab_delta_vs_amb,
                    te_res_delta_vs_set,
                    te_res_delta_vs_amb,
                ),
                (self.cabin_heat_source, self.te_deadband_cab),
                (
                    self.res_heat_source,
                    self.res_cooling_source,
                    self.te_deadband_res,
                ),
            )?;

        self.state.te_ref = te_ref;
        self.state.cop = self
            .get_cop_ideal_vcs(te_ref, te_ref_delta_vs_set, te_ref_delta_vs_amb)?
            .map(|cop_ideal| cop_ideal * self.frac_of_ideal_cop);

        self.solve_for_cabin(te_fc, cab_state, cab_heat_cap, dt)
            .with_context(|| format_dbg!())?;
        self.solve_for_res(res_temp, res_temp_prev, dt)
            .with_context(|| format_dbg!())?;

        // self.state.pwr_thrml_hvac_to_cabin = pwr_thrml_hvac_to_cab;
        // self.state.pwr_thrml_fc_to_cabin = pwr_thrml_fc_to_cabin;
        // self.state.pwr_thrml_hvac_to_res = pwr_thrml_hvac_to_res;

        Ok((
            self.state.pwr_thrml_hvac_to_cabin,
            self.state.pwr_thrml_fc_to_cabin,
            self.state.pwr_thrml_hvac_to_res,
        ))
    }

    fn get_te_deltas(
        &mut self,
        te_amb_air: si::Temperature,
        cab_state: LumpedCabinState,
        res_thrml_state: RESLumpedThermalState,
    ) -> (
        Option<si::TemperatureInterval>,
        si::TemperatureInterval,
        Option<si::TemperatureInterval>,
        si::TemperatureInterval,
    ) {
        let te_cab_delta_vs_set: Option<si::TemperatureInterval> =
            self.te_set_cab.map(|te_set_cab| {
                (cab_state.temperature.get::<si::degree_celsius>()
                    - te_set_cab.get::<si::degree_celsius>())
                    * uc::KELVIN_INT
            });

        let te_cab_delta_vs_amb: si::TemperatureInterval =
            (cab_state.temperature.get::<si::degree_celsius>()
                - te_amb_air.get::<si::degree_celsius>())
                * uc::KELVIN_INT;

        let te_res_delta_vs_set: Option<si::TemperatureInterval> =
            self.te_set_res.map(|te_set_res| {
                (res_thrml_state.temperature.get::<si::degree_celsius>()
                    - te_set_res.get::<si::degree_celsius>())
                    * uc::KELVIN_INT
            });

        let te_res_delta_vs_amb: si::TemperatureInterval =
            (res_thrml_state.temperature.get::<si::degree_celsius>()
                - te_amb_air.get::<si::degree_celsius>())
                * uc::KELVIN_INT;
        (
            te_cab_delta_vs_set,
            te_cab_delta_vs_amb,
            te_res_delta_vs_set,
            te_res_delta_vs_amb,
        )
    }

    /// Returns ideal coefficient of performance (COP) for vapor compression system (VCS)
    fn get_cop_ideal_vcs(
        &mut self,
        te_ref: Option<si::Temperature>,
        te_ref_delta_vs_set: Option<si::TemperatureInterval>,
        te_ref_delta_vs_amb: Option<si::TemperatureInterval>,
    ) -> anyhow::Result<Option<si::Ratio>> {
        // ideal COP if vapor compression sytem is active
        let cop_ideal_vcs =
            if let (Some(te_ref), Some(te_ref_delta_vs_set), Some(te_ref_delta_vs_amb)) =
                (te_ref, te_ref_delta_vs_set, te_ref_delta_vs_amb)
            {
                ensure!(
                    te_ref > si::Temperature::ZERO,
                    format!(
                        "{}\n`te_ref`: {} K",
                        format_dbg!(),
                        te_ref.get::<si::kelvin_abs>()
                    )
                );
                if te_ref_delta_vs_set > si::TemperatureInterval::ZERO {
                    // COOLING MODE; reference component is hotter than set point

                    // https://en.wikipedia.org/wiki/Coefficient_of_performance#Theoretical_performance_limits
                    // cop_ideal is t_h / (t_h - t_c) for heating
                    // cop_ideal is t_c / (t_h - t_c) for cooling

                    // divide-by-zero protection and realistic limit on COP
                    let cop_ideal = if -te_ref_delta_vs_amb < 5.0 * uc::KELVIN_INT {
                        // cabin is cooler than ambient + threshold
                        // TODO: make this `5.0` not hardcoded
                        te_ref / (5.0 * uc::KELVIN)
                    } else {
                        te_ref / te_ref_delta_vs_amb.abs()
                    };
                    ensure!(cop_ideal > 0.0 * uc::R, format_dbg!(cop_ideal));
                    Some(cop_ideal)
                } else {
                    // HEATING MODE; cabin is colder than set point

                    // https://en.wikipedia.org/wiki/Coefficient_of_performance#Theoretical_performance_limits
                    // cop_ideal is t_h / (t_h - t_c) for heating
                    // cop_ideal is t_c / (t_h - t_c) for cooling

                    // divide-by-zero protection and realistic limit on COP
                    let cop_ideal = if te_ref_delta_vs_amb < 5.0 * uc::KELVIN_INT {
                        // cabin is cooler than ambient + threshold
                        // TODO: make this `5.0` not hardcoded
                        te_ref / (5.0 * uc::KELVIN)
                    } else {
                        te_ref / te_ref_delta_vs_amb.abs()
                    };
                    ensure!(cop_ideal > 0.0 * uc::R, format_dbg!(cop_ideal));
                    Some(cop_ideal)
                }
            } else {
                None
            };
        Ok(cop_ideal_vcs)
    }

    /// Solve for thermal power for [LumpedCabin]
    fn solve_for_cabin(
        &mut self,
        te_fc: Option<si::Temperature>,
        cab_state: LumpedCabinState,
        cab_heat_cap: si::HeatCapacity,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // DO NOT uncomment the following line because this is a cumulative state variable!
        // self.state.pwr_i_cab = uc::W * f64::NAN;
        self.state.pwr_p_cab = uc::W * f64::NAN;
        self.state.pwr_d_cab = uc::W * f64::NAN;
        self.state.pwr_aux_for_cab_hvac_req = uc::W * f64::NAN;
        self.state.pwr_aux_for_cab_hvac = uc::W * f64::NAN;
        self.state.pwr_thrml_hvac_to_cabin = uc::W * f64::NAN;

        match self.te_set_cab {
            Some(te_set_cab) => {
                match self.state.mode.cabin_mode {
                    HvacMode::InsideDeadband => {
                        self.state.pwr_i_cab = si::Power::ZERO; // reset to 0.0
                        self.state.pwr_p_cab = si::Power::ZERO;
                        self.state.pwr_d_cab = si::Power::ZERO;
                        self.state.pwr_aux_for_cab_hvac_req = si::Power::ZERO;
                        self.state.pwr_aux_for_cab_hvac = si::Power::ZERO;
                        self.state.pwr_thrml_hvac_to_cabin = si::Power::ZERO;
                    }
                    HvacMode::Cooling => {
                        self.set_cab_cntrl_state(cab_state, dt, te_set_cab);
                        self.state.pwr_thrml_to_cab_req = {
                            if self.state.pwr_i_cab > si::Power::ZERO {
                                // If `pwr_i` is greater than zero, reset to switch from heating to cooling
                                self.state.pwr_i_cab = si::Power::ZERO;
                            }
                            let pwr_thrml_hvac_to_cab_req: si::Power = (self.state.pwr_p_cab
                                + self.state.pwr_i_cab
                                + self.state.pwr_d_cab)
                                .max(-self.pwr_thrml_max);
                            ensure!(
                                pwr_thrml_hvac_to_cab_req < si::Power::ZERO,
                                "{}\nHVAC should be cooling cabin",
                                format_dbg!(pwr_thrml_hvac_to_cab_req)
                            );
                            ensure!(
                                pwr_thrml_hvac_to_cab_req < si::Power::ZERO,
                                "HVAC should be cooling cabin\n{}\n{}\n{}",
                                format_dbg!(pwr_thrml_hvac_to_cab_req),
                                format_dbg!(self.state.pwr_aux_for_cab_hvac),
                                format_dbg!(self.state.cop)
                            );
                            pwr_thrml_hvac_to_cab_req
                        };
                        self.state.pwr_aux_for_cab_hvac_req = -self.state.pwr_thrml_to_cab_req
                            / self.state.cop.with_context(|| {
                                format!(
                                    "{}\nExpected `self.state.cop` to be Some.",
                                    format_dbg!(self.state.cop)
                                )
                            })?;

                        // Correct aux power components to account for any limit violations
                        if self.state.pwr_aux_for_cab_hvac_req > self.pwr_aux_for_hvac_cab_max {
                            self.state.pwr_aux_for_cab_hvac = self.pwr_aux_for_hvac_cab_max;
                            self.state.pwr_thrml_hvac_to_cabin = -self.state.pwr_aux_for_cab_hvac
                                * self.state.cop.with_context(|| {
                                    format!(
                                        "{}\nExpected `self.state.cop` to be Some.",
                                        format_dbg!(self.state.cop)
                                    )
                                })?;
                        } else {
                            self.state.pwr_aux_for_cab_hvac = self.state.pwr_aux_for_cab_hvac_req;
                            self.state.pwr_thrml_hvac_to_cabin = self.state.pwr_thrml_to_cab_req;
                        }
                    }
                    HvacMode::Heating => {
                        self.set_cab_cntrl_state(cab_state, dt, te_set_cab);
                        if self.state.pwr_i_cab < si::Power::ZERO {
                            // If `pwr_i` is less than zero reset to switch from cooling to heating
                            self.state.pwr_i_cab = si::Power::ZERO;
                        }
                        self.state.pwr_thrml_to_cab_req =
                            (self.state.pwr_p_cab + self.state.pwr_i_cab + self.state.pwr_d_cab)
                                .min(self.pwr_thrml_max);
                        ensure!(
                            self.state.pwr_thrml_to_cab_req > si::Power::ZERO,
                            "{}\nHVAC should be heating cabin",
                            format_dbg!(self.state.pwr_thrml_to_cab_req)
                        );

                        // Assumes blower has negligible impact on aux load, may want to revise later
                        if self.cabin_heat_source.is_fuel_converter() {
                            ensure!(
                                        te_fc.is_some(),
                                        "{}\nExpected vehicle with [FuelConverter] with thermal plant model.",
                                        format_dbg!()
                                    );
                            // limit heat transfer to be substantially less than what is physically possible
                            // i.e. the engine can't drop below cabin temperature to heat the cabin
                            self.state.pwr_thrml_to_cab_req = self.state.pwr_thrml_to_cab_req
                                            .min(
                                                cab_heat_cap *
                                            (te_fc.unwrap().get::<si::degree_celsius>() - cab_state.temperature.get::<si::degree_celsius>()) * uc::KELVIN_INT
                                                * 0.1 // so that it's substantially less
                                                / dt,
                                            )
                                            .max(si::Power::ZERO);
                        };
                        ensure!(
                            self.state.pwr_thrml_to_cab_req >= si::Power::ZERO,
                            "{}\nHVAC should be heating cabin",
                            format_dbg!(self.state.pwr_thrml_to_cab_req)
                        );

                        self.state.pwr_aux_for_cab_hvac_req =
                                // Heating
                                match self.cabin_heat_source {
                                    CabinHeatSource::FuelConverter => {
                                        // NOTE: should make this scale with power demand because it does require blower
                                        si::Power::ZERO
                                    }
                                    CabinHeatSource::ResistanceHeater => self.state.pwr_thrml_to_cab_req,
                                    CabinHeatSource::HeatPump => {
                                        self.state.pwr_thrml_to_cab_req
                                            / self.state.cop.with_context(|| {
                                                format!(
                                                    "{}\nExpected `self.state.cop` to be Some.",
                                                    format_dbg!(self.state.cop)
                                                )
                                            })?
                                    }
                                };
                        // Correct aux power components to account for any limit violations
                        if self.state.pwr_aux_for_cab_hvac_req > self.pwr_aux_for_hvac_cab_max {
                            self.state.pwr_aux_for_cab_hvac = self.pwr_aux_for_hvac_cab_max;
                            self.state.pwr_thrml_hvac_to_cabin = match self.cabin_heat_source {
                                CabinHeatSource::FuelConverter => {
                                    bail!("{}\nThis should be unreachable", format_dbg!());
                                }
                                CabinHeatSource::ResistanceHeater => {
                                    self.state.pwr_aux_for_cab_hvac
                                }
                                CabinHeatSource::HeatPump => {
                                    self.state.pwr_aux_for_cab_hvac
                                        * self.state.cop.with_context(|| {
                                            format!(
                                                "{}\nExpected `self.state.cop` to be Some.",
                                                format_dbg!(self.state.cop)
                                            )
                                        })?
                                }
                            }
                        } else {
                            self.state.pwr_aux_for_cab_hvac = self.state.pwr_aux_for_cab_hvac_req;
                            self.state.pwr_thrml_hvac_to_cabin = self.state.pwr_thrml_to_cab_req;
                        }
                    }
                    HvacMode::Inactive => {
                        self.state.pwr_i_cab = si::Power::ZERO;
                        self.state.pwr_p_cab = si::Power::ZERO;
                        self.state.pwr_d_cab = si::Power::ZERO;
                        self.state.pwr_aux_for_cab_hvac_req = si::Power::ZERO;
                        self.state.pwr_aux_for_cab_hvac = si::Power::ZERO;
                        self.state.pwr_thrml_hvac_to_cabin = si::Power::ZERO;
                    }
                    HvacMode::Invalid => {
                        unreachable!();
                    }
                };
            }
            None => {
                self.state.pwr_i_cab = si::Power::ZERO;
                self.state.pwr_p_cab = si::Power::ZERO;
                self.state.pwr_d_cab = si::Power::ZERO;
                self.state.pwr_aux_for_cab_hvac_req = si::Power::ZERO;
                self.state.pwr_aux_for_cab_hvac = si::Power::ZERO;
                self.state.pwr_thrml_hvac_to_cabin = si::Power::ZERO;
            }
        }
        // The following ensures are not likely to occur for users, only for developers
        ensure!(!self.state.pwr_i_cab.is_nan(), format_dbg!());
        ensure!(!self.state.pwr_p_cab.is_nan(), format_dbg!());
        ensure!(!self.state.pwr_d_cab.is_nan(), format_dbg!());
        ensure!(!self.state.pwr_aux_for_cab_hvac_req.is_nan(), format_dbg!());
        ensure!(!self.state.pwr_aux_for_cab_hvac.is_nan(), format_dbg!());
        ensure!(!self.state.pwr_thrml_hvac_to_cabin.is_nan(), format_dbg!());
        Ok(())
    }

    fn set_cab_cntrl_state(
        &mut self,
        cab_state: LumpedCabinState,
        dt: si::Time,
        te_set_cab: si::Temperature,
    ) {
        let te_delta_vs_set_cab = (cab_state.temperature.get::<si::degree_celsius>()
            - te_set_cab.get::<si::degree_celsius>())
            * uc::KELVIN_INT;

        self.state.pwr_p_cab = -self.p_cabin * te_delta_vs_set_cab;
        self.state.pwr_i_cab -=
            self.i_cabin * uc::W / uc::KELVIN / uc::S * te_delta_vs_set_cab * dt;
        self.state.pwr_i_cab = self
            .state
            .pwr_i_cab
            .max(-self.pwr_i_max_cabin)
            .min(self.pwr_i_max_cabin);
        self.state.pwr_d_cab = -self.d_cabin * uc::J / uc::KELVIN
            * ((cab_state.temperature.get::<si::degree_celsius>()
                - cab_state.temp_prev.get::<si::degree_celsius>())
                * uc::KELVIN_INT
                / dt);
    }

    /// Solve for thermal power for [RESLumpedThermal]
    fn solve_for_res(
        &mut self,
        // reversible energy storage temp
        res_temp: si::Temperature,
        // reversible energy storage temp at previous time step
        res_temp_prev: si::Temperature,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // DO NOT uncomment the following line because this is a cumulative state variable!
        // self.state.pwr_i_res = uc::W * f64::NAN;
        self.state.pwr_p_res = uc::W * f64::NAN;
        self.state.pwr_d_res = uc::W * f64::NAN;
        self.state.pwr_aux_for_res_hvac_req = uc::W * f64::NAN;
        self.state.pwr_aux_for_res_hvac = uc::W * f64::NAN;
        self.state.pwr_thrml_hvac_to_res = uc::W * f64::NAN;

        match self.te_set_res {
            Some(te_set_res) => {
                match self.state.mode.res_mode {
                    HvacMode::InsideDeadband => {
                        // inside deadband; no hvac power is needed
                        self.state.pwr_i_res = si::Power::ZERO;
                        self.state.pwr_p_res = si::Power::ZERO;
                        self.state.pwr_d_res = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac_req = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac = si::Power::ZERO;
                        self.state.pwr_thrml_hvac_to_res = si::Power::ZERO;
                    }
                    HvacMode::Cooling => {
                        self.set_res_cntrl_state(res_temp, res_temp_prev, dt, te_set_res);

                        if self.state.pwr_i_res > si::Power::ZERO {
                            // If `pwr_i_res` is greater than zero, reset to switch from heating to cooling
                            self.state.pwr_i_res = si::Power::ZERO;
                        }
                        self.state.pwr_thrml_to_res_req =
                            (self.state.pwr_p_res + self.state.pwr_i_res + self.state.pwr_d_res)
                                .max(-self.pwr_thrml_max);
                        ensure!(
                            self.state.pwr_thrml_to_res_req < si::Power::ZERO,
                            "{}\nHVAC should be cooling RES",
                            format_dbg!(self.state.pwr_thrml_to_res_req)
                        );
                        ensure!(
                            self.state.pwr_thrml_to_res_req < si::Power::ZERO,
                            "HVAC should be cooling RES\n{}\n{}\n{}",
                            format_dbg!(self.state.pwr_thrml_to_res_req),
                            format_dbg!(self.state.pwr_aux_for_res_hvac),
                            format_dbg!(self.state.cop)
                        );
                        self.state.pwr_aux_for_res_hvac_req = match self.res_cooling_source {
                            RESCoolingSource::HVAC => {
                                -self.state.pwr_thrml_to_res_req
                                    / self.state.cop.with_context(|| {
                                        format!(
                                            "{}\nExpected `self.state.cop` to be Some.",
                                            format_dbg!(self.state.cop)
                                        )
                                    })?
                            }
                            RESCoolingSource::None => si::Power::ZERO,
                        };

                        // Correct aux power components to account for any limit violations
                        if self.state.pwr_aux_for_res_hvac_req > self.pwr_aux_for_hvac_res_max {
                            self.state.pwr_aux_for_res_hvac = self.pwr_aux_for_hvac_res_max;
                            self.state.pwr_thrml_hvac_to_res = match self.res_cooling_source {
                                RESCoolingSource::HVAC => {
                                    -self.state.pwr_aux_for_res_hvac
                                        * self.state.cop.with_context(|| {
                                            format!(
                                                "{}\nExpected `self.state.cop` to be Some.",
                                                format_dbg!(self.state.cop)
                                            )
                                        })?
                                }
                                RESCoolingSource::None => {
                                    bail!("{}\nThis should be unreachable", format_dbg!());
                                }
                            }
                        } else {
                            self.state.pwr_aux_for_res_hvac = self.state.pwr_aux_for_res_hvac_req;
                            self.state.pwr_thrml_hvac_to_res = self.state.pwr_thrml_to_res_req;
                        };
                    }
                    HvacMode::Heating => {
                        self.set_res_cntrl_state(res_temp, res_temp_prev, dt, te_set_res);

                        self.state.pwr_thrml_to_res_req = {
                            // HEATING MODE; Reversible Energy Storage is colder than set point

                            if self.state.pwr_i_res < si::Power::ZERO {
                                // If `pwr_i_res` is less than zero reset to switch from cooling to heating
                                self.state.pwr_i_res = si::Power::ZERO;
                            }
                            (self.state.pwr_p_res + self.state.pwr_i_res + self.state.pwr_d_res)
                                .min(self.pwr_thrml_max)
                        };
                        ensure!(
                            self.state.pwr_thrml_to_res_req > si::Power::ZERO,
                            "{}\nHVAC should be heating RES",
                            format_dbg!(self.state.pwr_thrml_to_res_req)
                        );
                        ensure!(
                            self.state.pwr_thrml_to_res_req > si::Power::ZERO,
                            "HVAC should be heating RES\n{}\n{}\n{}",
                            format_dbg!(self.state.pwr_thrml_to_res_req),
                            format_dbg!(self.state.pwr_aux_for_res_hvac),
                            format_dbg!(self.state.cop)
                        );
                        self.state.pwr_aux_for_res_hvac_req = match self.res_heat_source {
                            RESHeatSource::ResistanceHeater => self.state.pwr_thrml_to_res_req,
                            RESHeatSource::HeatPump => {
                                self.state.pwr_thrml_to_res_req
                                    / self.state.cop.with_context(|| {
                                        format!(
                                            "{}\nExpected `self.state.cop` to be Some.",
                                            format_dbg!(self.state.cop)
                                        )
                                    })?
                            }
                            RESHeatSource::None => si::Power::ZERO,
                        };

                        // Correct aux power components to account for any limit violations
                        if self.state.pwr_aux_for_res_hvac_req > self.pwr_aux_for_hvac_res_max {
                            self.state.pwr_aux_for_res_hvac = self.pwr_aux_for_hvac_res_max;
                            self.state.pwr_thrml_hvac_to_res = match self.res_heat_source {
                                RESHeatSource::ResistanceHeater => self.state.pwr_aux_for_res_hvac,
                                RESHeatSource::HeatPump => {
                                    self.state.pwr_aux_for_res_hvac
                                        * self.state.cop.with_context(|| {
                                            format!(
                                                "{}\nExpected `self.state.cop` to be Some.",
                                                format_dbg!(self.state.cop)
                                            )
                                        })?
                                }
                                RESHeatSource::None => {
                                    bail!("{}\nThis should be unreachable", format_dbg!());
                                }
                            }
                        } else {
                            self.state.pwr_aux_for_res_hvac = self.state.pwr_aux_for_res_hvac_req;
                            self.state.pwr_thrml_hvac_to_res = self.state.pwr_thrml_to_res_req;
                        };
                    }
                    HvacMode::Inactive => {
                        self.state.pwr_i_res = si::Power::ZERO;
                        self.state.pwr_p_res = si::Power::ZERO;
                        self.state.pwr_d_res = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac_req = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac = si::Power::ZERO;
                        self.state.pwr_thrml_hvac_to_res = si::Power::ZERO;
                    }
                    HvacMode::Invalid => {
                        self.state.pwr_i_res = si::Power::ZERO;
                        self.state.pwr_p_res = si::Power::ZERO;
                        self.state.pwr_d_res = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac_req = si::Power::ZERO;
                        self.state.pwr_aux_for_res_hvac = si::Power::ZERO;
                        self.state.pwr_thrml_hvac_to_res = si::Power::ZERO;
                    }
                }
            }
            None => {
                self.state.pwr_i_res = si::Power::ZERO;
                self.state.pwr_p_res = si::Power::ZERO;
                self.state.pwr_d_res = si::Power::ZERO;
                self.state.pwr_aux_for_res_hvac_req = si::Power::ZERO;
                self.state.pwr_aux_for_res_hvac = si::Power::ZERO;
                self.state.pwr_thrml_hvac_to_res = si::Power::ZERO;
            }
        }

        Ok(())
    }

    fn set_res_cntrl_state(
        &mut self,
        res_temp: si::Temperature,
        res_temp_prev: si::Temperature,
        dt: si::Time,
        te_set_res: si::Temperature,
    ) {
        let te_delta_vs_set = (res_temp.get::<si::degree_celsius>()
            - te_set_res.get::<si::degree_celsius>())
            * uc::KELVIN_INT;
        self.state.pwr_p_res = -self.p_res * te_delta_vs_set;
        self.state.pwr_i_res -= self.i_res * uc::W / uc::KELVIN / uc::S * te_delta_vs_set * dt;
        self.state.pwr_i_res = self
            .state
            .pwr_i_res
            .max(-self.pwr_i_max_res)
            .min(self.pwr_i_max_res);
        self.state.pwr_d_res = -self.d_res * uc::J / uc::KELVIN
            * ((res_temp.get::<si::degree_celsius>() - res_temp_prev.get::<si::degree_celsius>())
                * uc::KELVIN_INT
                / dt);
    }
}

#[fastsim_api]
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative,
)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct HVACSystemForLumpedCabinAndRESState {
    /// time step counter
    pub i: TrackedStateWithMemory<usize>,
    /// portion of total HVAC cooling/heating (negative/positive) power due to
    /// proportional gain
    pub pwr_p_cab: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// due to proportional gain
    pub energy_p_cab: TrackedStateWithMemory<si::Energy>,
    /// portion of total HVAC cooling/heating (negative/positive) power due to
    /// integral gain
    pub pwr_i_cab: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// due to integral gain
    pub energy_i_cab: TrackedStateWithMemory<si::Energy>,
    /// portion of total HVAC cooling/heating (negative/positive) power due to
    /// derivative gain
    pub pwr_d_cab: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// due to derivative gain
    pub energy_d_cab: TrackedStateWithMemory<si::Energy>,
    /// portion of total HVAC cooling/heating (negative/positive) power to
    /// [ReversibleEnergyStorage::thrml] due to proportional gain
    pub pwr_p_res: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// to [ReversibleEnergyStorage::thrml] due to proportional gain
    pub energy_p_res: TrackedStateWithMemory<si::Energy>,
    /// portion of total HVAC cooling/heating (negative/positive) power to
    /// [ReversibleEnergyStorage::thrml] due to integral gain
    pub pwr_i_res: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// to [ReversibleEnergyStorage::thrml] due to integral gain
    pub energy_i_res: TrackedStateWithMemory<si::Energy>,
    /// portion of total HVAC cooling/heating (negative/positive) power to
    /// [ReversibleEnergyStorage::thrml] due to derivative gain
    pub pwr_d_res: TrackedState<si::Power>,
    /// portion of total HVAC cooling/heating (negative/positive) cumulative energy
    /// to [ReversibleEnergyStorage::thrml] due to derivative gain
    pub energy_d_res: TrackedStateWithMemory<si::Energy>,
    /// coefficient of performance (i.e. efficiency) of vapor compression cycle
    pub cop: TrackedState<Option<si::Ratio>>,
    /// Reference temperature used to calculate coefficient of performance (i.e.
    /// efficiency) of vapor compression cycle
    pub te_ref: Option<si::Temperature>,
    /// Requested aux power demand from [Vehicle::hvac] system for cabin thermal
    /// managemement.
    pub pwr_aux_for_cab_hvac_req: TrackedState<si::Power>,
    /// Requested thermal power demand from [Vehicle::hvac] system for cabin thermal
    /// managemement.
    pub pwr_thrml_to_cab_req: TrackedState<si::Power>,
    /// Requested aux power demand from [Vehicle::hvac] system for res thermal
    /// managemement.
    pub pwr_aux_for_res_hvac_req: TrackedState<si::Power>,
    /// Requested thermal power demand from [Vehicle::hvac] system for res thermal
    /// managemement.
    pub pwr_thrml_to_res_req: TrackedState<si::Power>,
    /// Arbitrated power demand from [Vehicle::hvac] system for cabin thermal
    /// managemement
    ///
    /// NOTE: If battery is hot and cabin is cold or vice versa, this is an overestimate.
    pub pwr_aux_for_cab_hvac: TrackedState<si::Power>,
    /// Arbitrated cumulative energy demand from [Vehicle::hvac] system for cabin
    /// thermal managemement
    ///
    /// NOTE: If battery is hot and cabin is cold or vice versa, this is an overestimate.
    pub energy_aux_for_cab_hvac: TrackedStateWithMemory<si::Energy>,
    /// Arbitrated power demand from [Vehicle::hvac] system for res thermal
    /// managemement
    ///
    /// NOTE: If battery is hot and cabin is cold or vice versa, this is an overestimate.
    pub pwr_aux_for_res_hvac: TrackedState<si::Power>,
    /// Arbitrated cumulative energy demand from [Vehicle::hvac] system for res
    /// thermal managemement
    ///
    /// NOTE: If battery is hot and cabin is cold or vice versa, this is an overestimate.
    pub energy_aux_for_res_hvac: TrackedStateWithMemory<si::Energy>,
    /// Thermal power from HVAC system to cabin, positive is heating the cabin
    pub pwr_thrml_hvac_to_cabin: TrackedState<si::Power>,
    /// Cumulative thermal energy from HVAC system to cabin, positive is heating
    /// the cabin
    pub energy_thrml_hvac_to_cabin: TrackedStateWithMemory<si::Energy>,
    /// Thermal power from [FuelConverter] to [Cabin]
    pub pwr_thrml_fc_to_cabin: TrackedState<si::Power>,
    /// Cumulative thermal energy from [FuelConverter] to [Cabin]
    pub energy_thrml_fc_to_cabin: TrackedStateWithMemory<si::Energy>,
    /// Thermal power from HVAC to [ReversibleEnergyStorage]
    pub pwr_thrml_hvac_to_res: TrackedState<si::Power>,
    /// Cumulative thermal energy from HVAC to [ReversibleEnergyStorage]
    pub energy_thrml_hvac_to_res: TrackedStateWithMemory<si::Energy>,
    pub mode: HvacModeForLumpedCabinAndRes,
}
impl Init for HVACSystemForLumpedCabinAndRESState {}
impl SerdeAPI for HVACSystemForLumpedCabinAndRESState {}

#[fastsim_api]
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative,
)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct HvacModeForLumpedCabinAndRes {
    /// Component corresponding to [te_ref]
    pub te_ref_component: TeRefComp,
    /// Current mode of cabin hvac
    pub cabin_mode: HvacMode,
    /// Current mode of [ReversibleEnergyStorage] hvac
    pub res_mode: HvacMode,
}
impl Init for HvacModeForLumpedCabinAndRes {}
impl SerdeAPI for HvacModeForLumpedCabinAndRes {}
impl HvacModeForLumpedCabinAndRes {
    fn set_mode_and_get_te_for_cop(
        &mut self,
        cab_state: LumpedCabinState,
        res_thrml_state: RESLumpedThermalState,
        te_deltas: (
            Option<si::TemperatureInterval>,
            si::TemperatureInterval,
            Option<si::TemperatureInterval>,
            si::TemperatureInterval,
        ),
        cab_params: (CabinHeatSource, si::TemperatureInterval),
        res_params: (RESHeatSource, RESCoolingSource, si::TemperatureInterval),
    ) -> anyhow::Result<(
        Option<si::Temperature>,
        Option<si::TemperatureInterval>,
        Option<si::TemperatureInterval>,
    )> {
        let (te_cab_delta_vs_set, te_cab_delta_vs_amb, te_res_delta_vs_set, te_res_delta_vs_amb) =
            te_deltas;

        let (cabin_heat_source, te_deadband_cab) = cab_params;
        let (res_heat_source, res_cooling_source, te_deadband_res) = res_params;

        self.cabin_mode = HvacMode::Invalid;
        self.res_mode = HvacMode::Invalid;
        self.te_ref_component = TeRefComp::Invalid;

        // NOTE: if `_` is used, order of match statements has an effect on
        // which mode gets set because the `_` are greedy!
        match (te_res_delta_vs_set, te_cab_delta_vs_set) {
            (Some(te_res_delta_vs_set), Some(te_cab_delta_vs_set)) => {
                match (
                    te_res_delta_vs_set > si::TemperatureInterval::ZERO,
                    te_res_delta_vs_set.abs() > te_deadband_res,
                    te_cab_delta_vs_set > si::TemperatureInterval::ZERO,
                    te_cab_delta_vs_set.abs() > te_deadband_cab,
                ) {
                    (true, true, true, true) => {
                        // battery is hot and outside deadband
                        // cabin is hot and outside the deadband
                        self.cabin_mode = HvacMode::Cooling;
                        (self.res_mode, self.te_ref_component) = match res_cooling_source {
                            RESCoolingSource::HVAC => (
                                HvacMode::Cooling,
                                if te_res_delta_vs_amb.abs() > te_cab_delta_vs_amb.abs() {
                                    TeRefComp::RES
                                } else {
                                    TeRefComp::Cabin
                                },
                            ),
                            RESCoolingSource::None => (HvacMode::Inactive, TeRefComp::Cabin),
                        }
                    }
                    (true, true, true, false) => {
                        // battery is hot and outside deadband
                        // cabin is hot and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        (self.res_mode, self.te_ref_component) = match res_cooling_source {
                            RESCoolingSource::HVAC => (HvacMode::Cooling, TeRefComp::RES),
                            RESCoolingSource::None => (HvacMode::Inactive, TeRefComp::None),
                        };
                    }
                    (true, true, false, true) => {
                        // battery is hot and outside deadband
                        // cabin is cold and outside the deadband
                        self.cabin_mode = HvacMode::Heating;
                        (self.res_mode, self.te_ref_component) =
                            match (cabin_heat_source, res_cooling_source) {
                                (CabinHeatSource::HeatPump, RESCoolingSource::HVAC) => (
                                    HvacMode::Cooling,
                                    if te_res_delta_vs_amb.abs() > te_cab_delta_vs_amb.abs() {
                                        TeRefComp::RES
                                    } else {
                                        TeRefComp::Cabin
                                    },
                                ),
                                (CabinHeatSource::HeatPump, RESCoolingSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::Cabin)
                                }
                                (CabinHeatSource::ResistanceHeater, RESCoolingSource::HVAC) => {
                                    (HvacMode::Cooling, TeRefComp::RES)
                                }
                                (CabinHeatSource::ResistanceHeater, RESCoolingSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::None)
                                }
                                (CabinHeatSource::FuelConverter, RESCoolingSource::HVAC) => {
                                    (HvacMode::Cooling, TeRefComp::RES)
                                }
                                (CabinHeatSource::FuelConverter, RESCoolingSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::None)
                                }
                            };
                    }
                    (true, true, false, false) => {
                        // battery is hot and outside deadband
                        // cabin is cold and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        (self.res_mode, self.te_ref_component) = match res_cooling_source {
                            RESCoolingSource::HVAC => (HvacMode::Cooling, TeRefComp::RES),
                            RESCoolingSource::None => (HvacMode::Inactive, TeRefComp::None),
                        }
                    }
                    (true, false, true, true) => {
                        // battery is hot and within the deadband
                        // cabin is hot and outside the deadband
                        self.res_mode = match res_cooling_source {
                            RESCoolingSource::HVAC => HvacMode::InsideDeadband,
                            RESCoolingSource::None => HvacMode::Inactive,
                        };
                        self.cabin_mode = HvacMode::Cooling;
                        self.te_ref_component = TeRefComp::Cabin;
                    }
                    (true, false, false, true) => {
                        // battery is hot and within deadband
                        // cabin is cold and outside the deadband
                        self.res_mode = match res_cooling_source {
                            RESCoolingSource::HVAC => HvacMode::InsideDeadband,
                            RESCoolingSource::None => HvacMode::Inactive,
                        };
                        self.cabin_mode = HvacMode::Heating;
                        self.te_ref_component = match cabin_heat_source {
                            CabinHeatSource::HeatPump => TeRefComp::Cabin,
                            _ => TeRefComp::None,
                        }
                    }
                    (true, false, true, false) => {
                        // battery is hot and within the deadband
                        // cabin is hot  and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        self.res_mode = match res_cooling_source {
                            RESCoolingSource::HVAC => HvacMode::InsideDeadband,
                            RESCoolingSource::None => HvacMode::Inactive,
                        };
                        self.te_ref_component = TeRefComp::None;
                    }
                    (true, false, false, false) => {
                        // battery is hot and within the deadband
                        // cabin is cold and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        self.res_mode = match res_cooling_source {
                            RESCoolingSource::HVAC => HvacMode::InsideDeadband,
                            RESCoolingSource::None => HvacMode::Inactive,
                        };
                        self.te_ref_component = TeRefComp::None;
                    }
                    (false, true, true, true) => {
                        // battery is cold and outside deadband
                        // cabin is hot and outside the deadband
                        self.cabin_mode = HvacMode::Cooling;
                        (self.res_mode, self.te_ref_component) = match res_heat_source {
                            RESHeatSource::HeatPump => (
                                HvacMode::Heating,
                                if te_res_delta_vs_amb.abs() > te_cab_delta_vs_amb.abs() {
                                    TeRefComp::RES
                                } else {
                                    TeRefComp::Cabin
                                },
                            ),
                            RESHeatSource::ResistanceHeater => {
                                (HvacMode::Heating, TeRefComp::Cabin)
                            }
                            RESHeatSource::None => (HvacMode::Inactive, TeRefComp::Cabin),
                        }
                    }
                    (false, true, true, false) => {
                        // battery is cold and outside the deadband
                        // cabin is hot and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        (self.res_mode, self.te_ref_component) = match res_heat_source {
                            RESHeatSource::HeatPump => (HvacMode::Heating, TeRefComp::RES),
                            RESHeatSource::ResistanceHeater => (HvacMode::Heating, TeRefComp::None),
                            RESHeatSource::None => (HvacMode::Inactive, TeRefComp::None),
                        };
                    }
                    (false, true, false, true) => {
                        // battery is cold and outside deadband
                        // cabin is cold and outside the deadband
                        self.cabin_mode = HvacMode::Heating;
                        (self.res_mode, self.te_ref_component) =
                            match (cabin_heat_source, res_heat_source) {
                                (CabinHeatSource::HeatPump, RESHeatSource::HeatPump) => (
                                    HvacMode::Heating,
                                    if te_res_delta_vs_amb.abs() > te_cab_delta_vs_amb.abs() {
                                        TeRefComp::RES
                                    } else {
                                        TeRefComp::Cabin
                                    },
                                ),
                                (CabinHeatSource::HeatPump, RESHeatSource::ResistanceHeater) => {
                                    (HvacMode::Heating, TeRefComp::Cabin)
                                }
                                (CabinHeatSource::HeatPump, RESHeatSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::Cabin)
                                }
                                (CabinHeatSource::FuelConverter, RESHeatSource::HeatPump) => {
                                    (HvacMode::Heating, TeRefComp::RES)
                                }
                                (
                                    CabinHeatSource::FuelConverter,
                                    RESHeatSource::ResistanceHeater,
                                ) => (HvacMode::Heating, TeRefComp::None),
                                (CabinHeatSource::FuelConverter, RESHeatSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::None)
                                }
                                (CabinHeatSource::ResistanceHeater, RESHeatSource::HeatPump) => {
                                    (HvacMode::Heating, TeRefComp::RES)
                                }
                                (
                                    CabinHeatSource::ResistanceHeater,
                                    RESHeatSource::ResistanceHeater,
                                ) => (HvacMode::Heating, TeRefComp::None),
                                (CabinHeatSource::ResistanceHeater, RESHeatSource::None) => {
                                    (HvacMode::Inactive, TeRefComp::None)
                                }
                            }
                    }
                    (false, true, false, false) => {
                        // battery is cold and outside deadband
                        // cabin is cold and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        (self.res_mode, self.te_ref_component) = match res_heat_source {
                            RESHeatSource::HeatPump => (
                                HvacMode::Heating,
                                if te_res_delta_vs_amb.abs() > te_cab_delta_vs_amb.abs() {
                                    TeRefComp::RES
                                } else {
                                    TeRefComp::Cabin
                                },
                            ),
                            RESHeatSource::ResistanceHeater => {
                                (HvacMode::Heating, TeRefComp::Cabin)
                            }
                            RESHeatSource::None => (HvacMode::Inactive, TeRefComp::Cabin),
                        }
                    }
                    (false, false, true, true) => {
                        // battery is cold and within deadband
                        // cabin is hot and outside the deadband
                        self.cabin_mode = HvacMode::Cooling;
                        self.res_mode = match res_heat_source {
                            RESHeatSource::HeatPump => HvacMode::InsideDeadband,
                            RESHeatSource::ResistanceHeater => HvacMode::InsideDeadband,
                            RESHeatSource::None => HvacMode::Inactive,
                        };
                        self.te_ref_component = TeRefComp::Cabin;
                    }
                    (false, false, true, false) => {
                        // battery is cold and within deadband
                        // cabin is hot and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        self.res_mode = match res_heat_source {
                            RESHeatSource::HeatPump => HvacMode::InsideDeadband,
                            RESHeatSource::ResistanceHeater => HvacMode::InsideDeadband,
                            RESHeatSource::None => HvacMode::Inactive,
                        };
                        self.te_ref_component = TeRefComp::None;
                    }
                    (false, false, false, true) => {
                        // battery is cold and within deadband
                        // cabin is cold and outside the deadband
                        self.res_mode = match res_heat_source {
                            RESHeatSource::HeatPump => HvacMode::InsideDeadband,
                            RESHeatSource::ResistanceHeater => HvacMode::InsideDeadband,
                            RESHeatSource::None => HvacMode::Inactive,
                        };
                        (self.cabin_mode, self.te_ref_component) = match cabin_heat_source {
                            CabinHeatSource::HeatPump => (HvacMode::Heating, TeRefComp::Cabin),
                            _ => (HvacMode::Heating, TeRefComp::None),
                        }
                    }
                    (false, false, false, false) => {
                        // battery is cold and within deadband
                        // cabin is cold and within the deadband
                        self.cabin_mode = HvacMode::InsideDeadband;
                        self.res_mode = match res_heat_source {
                            RESHeatSource::HeatPump => HvacMode::InsideDeadband,
                            RESHeatSource::ResistanceHeater => HvacMode::InsideDeadband,
                            RESHeatSource::None => HvacMode::Inactive,
                        };
                        self.te_ref_component = TeRefComp::None;
                    }
                }
            }
            (Some(te_res_delta_vs_set), None) => {
                // positive and outside the deadband
                self.cabin_mode = HvacMode::Inactive;
                (self.res_mode, self.te_ref_component) = if te_res_delta_vs_set > te_deadband_res {
                    // positive  - i.e. cooling mode
                    match res_cooling_source {
                        RESCoolingSource::HVAC => (HvacMode::Cooling, TeRefComp::RES),
                        RESCoolingSource::None => (HvacMode::Inactive, TeRefComp::None),
                    }
                } else if te_res_delta_vs_set < -te_deadband_res {
                    // negative -- i.e. heating mode
                    match res_heat_source {
                        RESHeatSource::HeatPump => (HvacMode::Heating, TeRefComp::RES),
                        RESHeatSource::ResistanceHeater => (HvacMode::Heating, TeRefComp::None),
                        RESHeatSource::None => (HvacMode::Inactive, TeRefComp::None),
                    }
                } else {
                    (
                        // actual temperature is within deadband range of setpoint temperature
                        HvacMode::InsideDeadband,
                        TeRefComp::None,
                    )
                }
            }
            (None, Some(te_cab_delta_vs_set)) => {
                self.res_mode = HvacMode::Inactive;
                // positive and outside the deadband
                (self.cabin_mode, self.te_ref_component) = if te_cab_delta_vs_set > te_deadband_cab
                {
                    (HvacMode::Cooling, TeRefComp::Cabin)
                } else if te_cab_delta_vs_set < -te_deadband_cab {
                    (
                        HvacMode::Heating,
                        // negative -- i.e. heating mode
                        match cabin_heat_source {
                            CabinHeatSource::HeatPump => TeRefComp::Cabin,
                            _ => TeRefComp::None,
                        },
                    )
                } else {
                    (
                        // actual temperature is within deadband range of setpoint temperature
                        HvacMode::InsideDeadband,
                        TeRefComp::None,
                    )
                }
            }
            (None, None) => {
                // thermal management is totally inactive
                self.cabin_mode = HvacMode::Inactive;
                self.res_mode = HvacMode::Inactive;
                self.te_ref_component = TeRefComp::None;
            }
        }
        ensure!(
            !self.te_ref_component.is_invalid(),
            format!("{}\n`te_ref_component` has not been updated", format_dbg!())
        );
        ensure!(
            !self.cabin_mode.is_invalid(),
            format!("{}\n`cabin_mode` has not been updated", format_dbg!())
        );
        ensure!(
            !self.res_mode.is_invalid(),
            format!("{}\n`res_mode` has not been updated", format_dbg!())
        );

        let (te_ref, te_ref_delta_vs_set, te_ref_delta_vs_amb): (
            Option<si::Temperature>,
            Option<si::TemperatureInterval>,
            Option<si::TemperatureInterval>,
        ) = match self.te_ref_component {
            TeRefComp::Cabin => (
                Some(cab_state.temperature),
                te_cab_delta_vs_set,
                Some(te_cab_delta_vs_amb),
            ),
            TeRefComp::RES => (
                Some(res_thrml_state.temperature),
                te_res_delta_vs_set,
                Some(te_res_delta_vs_amb),
            ),
            TeRefComp::None => (None, None, None),
            TeRefComp::Invalid => unreachable!(),
        };

        Ok((te_ref, te_ref_delta_vs_set, te_ref_delta_vs_amb))
    }
}

#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
/// Heat source for [RESLumpedThermal]
pub enum RESHeatSource {
    /// Resistance heater provides heat for HVAC system
    ResistanceHeater,
    /// Heat pump provides heat for HVAC system
    HeatPump,
    /// The battery is not actively heated
    None,
}
impl Init for RESHeatSource {}
impl SerdeAPI for RESHeatSource {}

#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, IsVariant, derive_more::From, TryInto,
)]
/// Cooling source for [RESLumpedThermal]
pub enum RESCoolingSource {
    /// Vapor compression system used for cabin HVAC also cools [RESLumpedThermal]
    HVAC,
    /// [RESLumpedThermal] is not actively cooled
    None,
}
impl Init for RESCoolingSource {}
impl SerdeAPI for RESCoolingSource {}

#[derive(
    Clone,
    Copy,
    Default,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    IsVariant,
    derive_more::From,
    TryInto,
)]
/// Component used for reference temperature in calulating [HVACSystemForLumpedCabinAndRESState::cop]
pub enum TeRefComp {
    /// Cabin
    Cabin,
    /// ReversibleEnergyStorage
    RES,
    /// Vapor compression system is inactive
    None,
    #[default]
    /// Used for debugging
    Invalid,
}
impl Init for TeRefComp {}
impl SerdeAPI for TeRefComp {}
