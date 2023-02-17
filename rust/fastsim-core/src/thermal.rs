//! Module for simulating thermal behavior of powertrains

use proc_macros::{add_pyo3_api, HistoryVec};

use crate::air::AirProperties;
use crate::cycle;
use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::simdrive;
use crate::vehicle;
use crate::vehicle_thermal::*;

#[add_pyo3_api(
    /// method for instantiating SimDriveHot
    #[new]
    pub fn __new__(
        cyc: cycle::RustCycle,
        veh: vehicle::RustVehicle,
        vehthrm: VehicleThermal,
        init_state: Option<ThermalState>,
        amb_te_deg_c: Option<Vec<f64>>,
     ) -> Self {
        Self::new(cyc, veh, vehthrm, init_state, amb_te_deg_c.map(Array1::from))
    }

    #[pyo3(name = "gap_to_lead_vehicle_m")]
    /// Provides the gap-with lead vehicle from start to finish
    pub fn gap_to_lead_vehicle_m_py(&self) -> PyResult<Vec<f64>> {
        Ok(self.gap_to_lead_vehicle_m().to_vec())
    }
     #[pyo3(name = "sim_drive")]
    /// Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
    /// Arguments
    /// ------------
    /// init_soc: initial SOC for electrified vehicles.
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///     Default of None causes veh.aux_kw to be used.
    pub fn sim_drive_py(
        &mut self,
        init_soc: Option<f64>,
        aux_in_kw_override: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        Ok(self.sim_drive(init_soc, aux_in_kw_override)?)
    }

    /// Receives second-by-second cycle information, vehicle properties,
    /// and an initial state of charge and runs sim_drive_step to perform a
    /// backward facing powertrain simulation. Method 'sim_drive' runs this
    /// iteratively to achieve correct SOC initial and final conditions, as
    /// needed.
    ///
    /// Arguments
    /// ------------
    /// init_soc (optional): initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///         None causes veh.aux_kw to be used.
    pub fn sim_drive_walk(
        &mut self,
        init_soc: f64,
        aux_in_kw_override: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        self.walk(init_soc, aux_in_kw_override);
        Ok(())
    }

    #[pyo3(name = "init_for_step")]
    /// This is a specialty method which should be called prior to using
    /// sim_drive_step in a loop.
    /// Arguments
    /// ------------
    /// init_soc: initial battery state-of-charge (SOC) for electrified vehicles
    /// aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.
    ///         Default of None causes veh.aux_kw to be used.
    pub fn init_for_step_py(
        &mut self,
        init_soc:f64,
        aux_in_kw_override: Option<Vec<f64>>
    ) -> PyResult<()> {
        let aux_in_kw_override = aux_in_kw_override.map(Array1::from);
        self.init_for_step(init_soc, aux_in_kw_override);
        Ok(())
    }

    /// Step through 1 time step.
    pub fn sim_drive_step(&mut self) -> PyResult<()> {
        Ok(self.step()?)
    }
    #[pyo3(name = "solve_step")]
    /// Perform all the calculations to solve 1 time step.
    pub fn solve_step_py(&mut self, i: usize) -> PyResult<()> {
        self.solve_step(i);
        Ok(())
    }

    #[pyo3(name = "set_misc_calcs")]
    /// Sets misc. calculations at time step 'i'
    /// Arguments:
    /// ----------
    /// i: index of time step
    pub fn set_misc_calcs_py(&mut self, i: usize) -> PyResult<()> {
        self.set_misc_calcs(i);
        Ok(())
    }

    #[pyo3(name = "set_comp_lims")]
    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_comp_lims_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_comp_lims(i)?)
    }

    #[pyo3(name = "set_power_calcs")]
    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_power_calcs(i)?)
    }

    #[pyo3(name = "set_ach_speed")]
    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_ach_speed(i)?)
    }

    #[pyo3(name = "set_hybrid_cont_calcs")]
    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_hybrid_cont_calcs(i)?)
    }

    #[pyo3(name = "set_fc_forced_state")]
    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    /// `_py` extension is needed to avoid name collision with getter/setter methods
    pub fn set_fc_forced_state_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_fc_forced_state_rust(i)?)
    }

    #[pyo3(name = "set_hybrid_cont_decisions")]
    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_hybrid_cont_decisions(i)?)
    }

    #[pyo3(name = "set_fc_power")]
    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_power_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_fc_power(i)?)
    }

    #[pyo3(name = "set_time_dilation")]
    /// Sets the time dilation for the current step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_time_dilation_py(&mut self, i: usize) -> PyResult<()> {
        Ok(self.set_time_dilation(i)?)
    }

    #[pyo3(name = "set_post_scalars")]
    /// Sets scalar variables that can be calculated after a cycle is run.
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars_py(&mut self) -> PyResult<()> {
        Ok(self.set_post_scalars()?)
    }
)]
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SimDriveHot {
    #[api(has_orphaned)]
    pub sd: simdrive::RustSimDrive,
    #[api(has_orphaned)]
    pub vehthrm: VehicleThermal,
    #[api(skip_get, skip_set)]
    #[serde(skip)]
    air: AirProperties,
    #[api(has_orphaned)]
    pub state: ThermalState,
    pub history: ThermalStateHistoryVec,
    pub hvac_model_history: HVACModelHistoryVec,
    #[api(skip_get, skip_set)]
    amb_te_deg_c: Option<Array1<f64>>,
}

impl SimDriveHot {
    pub fn new(
        cyc: cycle::RustCycle,
        veh: vehicle::RustVehicle,
        vehthrm: VehicleThermal,
        init_state: Option<ThermalState>,
        amb_te_deg_c: Option<Array1<f64>>,
    ) -> Self {
        let sd = simdrive::RustSimDrive::new(cyc, veh);
        let air = AirProperties::default();
        let history = ThermalStateHistoryVec::default();

        let (amb_te_deg_c_arr, state) = match amb_te_deg_c {
            Some(amb_te_deg_c_arr) => match init_state {
                Some(state) => {
                    assert_eq!(state.amb_te_deg_c, amb_te_deg_c_arr[0]);
                    (Some(amb_te_deg_c_arr), state)
                }
                None => {
                    let state = ThermalState {
                        amb_te_deg_c: amb_te_deg_c_arr[0],
                        ..ThermalState::default()
                    };
                    (Some(amb_te_deg_c_arr), state)
                }
            },
            None => (
                None, // 1st return element
                match init_state {
                    Some(state) => state, // 2nd return element
                    None => ThermalState::default(),
                },
            ),
        };

        Self {
            sd,
            vehthrm,
            air,
            state,
            history,
            hvac_model_history: HVACModelHistoryVec::default(),
            amb_te_deg_c: amb_te_deg_c_arr,
        }
    }

    pub fn gap_to_lead_vehicle_m(&self) -> Array1<f64> {
        self.sd.gap_to_lead_vehicle_m()
    }

    pub fn sim_drive(
        &mut self,
        init_soc: Option<f64>,
        aux_in_kw_override: Option<Array1<f64>>,
    ) -> Result<(), anyhow::Error> {
        self.sd.hev_sim_count = 0;

        let init_soc = match init_soc {
            Some(x) => x,
            None => {
                if self.sd.veh.veh_pt_type == vehicle::CONV {
                    // If no EV / Hybrid components, no SOC considerations.
                    (self.sd.veh.max_soc + self.sd.veh.min_soc) / 2.0
                } else if self.sd.veh.veh_pt_type == vehicle::HEV {
                    // ####################################
                    // ### Charge Balancing Vehicle SOC ###
                    // ####################################
                    // Charge balancing SOC for HEV vehicle types. Iterating init_soc and comparing to final SOC.
                    // Iterating until tolerance met or 30 attempts made.
                    let mut init_soc = (self.sd.veh.max_soc + self.sd.veh.min_soc) / 2.0;
                    let mut ess_2fuel_kwh = 1.0;
                    while ess_2fuel_kwh > self.sd.veh.ess_to_fuel_ok_error
                        && self.sd.hev_sim_count < self.sd.sim_params.sim_count_max
                    {
                        self.sd.hev_sim_count += 1;
                        self.walk(init_soc, aux_in_kw_override.clone());
                        let fuel_kj = (&self.sd.fs_kw_out_ach * self.sd.cyc.dt_s()).sum();
                        let roadway_chg_kj =
                            (&self.sd.roadway_chg_kw_out_ach * self.sd.cyc.dt_s()).sum();
                        if (fuel_kj + roadway_chg_kj) > 0.0 {
                            ess_2fuel_kwh = ((self.sd.soc[0] - self.sd.soc.last().unwrap())
                                * self.sd.veh.ess_max_kwh
                                * 3.6e3
                                / (fuel_kj + roadway_chg_kj))
                                .abs();
                        } else {
                            ess_2fuel_kwh = 0.0;
                        }
                        init_soc = min(1.0, max(0.0, *self.sd.soc.last().unwrap()));
                    }
                    init_soc
                } else if self.sd.veh.veh_pt_type == vehicle::PHEV
                    || self.sd.veh.veh_pt_type == vehicle::BEV
                {
                    // If EV, initializing initial SOC to maximum SOC.
                    self.sd.veh.max_soc
                } else {
                    panic!("Failed to properly initialize SOC.");
                }
            }
        };

        self.walk(init_soc, aux_in_kw_override);

        self.set_post_scalars()?;
        Ok(())
    }

    pub fn walk(&mut self, init_soc: f64, aux_in_kw_override: Option<Array1<f64>>) {
        self.init_for_step(init_soc, aux_in_kw_override);
        while self.sd.i < self.sd.cyc.time_s.len() {
            self.step().unwrap();
        }
    }

    pub fn init_for_step(&mut self, init_soc: f64, aux_in_kw_override: Option<Array1<f64>>) {
        self.history.push(self.state.clone()); // TODO: eventually make this dependent on `save_interval` usize per ALTRIOS
        match &self.vehthrm.cabin_hvac_model {
            CabinHvacModelTypes::Internal(hvac_mod) => {
                self.hvac_model_history.push(hvac_mod.clone())
            }
            CabinHvacModelTypes::External => {}
        }
        self.sd.init_for_step(init_soc, aux_in_kw_override).unwrap();
    }

    pub fn set_speed_for_target_gap_using_idm(&mut self, i: usize) {
        self.sd.set_speed_for_target_gap_using_idm(i);
    }

    pub fn set_speed_for_target_gap(&mut self, i: usize) {
        self.sd.set_speed_for_target_gap(i);
    }

    pub fn step(&mut self) -> Result<(), anyhow::Error> {
        self.set_thermal_calcs(self.sd.i);
        self.set_misc_calcs(self.sd.i);
        self.set_comp_lims(self.sd.i)?;
        self.set_power_calcs(self.sd.i)?;
        self.set_ach_speed(self.sd.i)?;
        self.set_hybrid_cont_calcs(self.sd.i)?;
        self.set_fc_forced_state_rust(self.sd.i)?;
        self.set_hybrid_cont_decisions(self.sd.i)?;
        self.set_fc_power(self.sd.i)?;

        self.sd.i += 1; // increment time step counter
        self.history.push(self.state.clone());
        match &self.vehthrm.cabin_hvac_model {
            CabinHvacModelTypes::Internal(hvac_mod) => {
                self.hvac_model_history.push(hvac_mod.clone());
            }
            CabinHvacModelTypes::External => {}
        }
        Ok(())
    }

    pub fn solve_step(&mut self, i: usize) {
        self.sd.solve_step(i).unwrap();
    }

    pub fn set_thermal_calcs(&mut self, i: usize) {
        // most of the thermal equations are at [i-1] because the various thermally
        // sensitive component efficiencies dependent on the [i] temperatures, but
        // these are in turn dependent on [i-1] heat transfer processes
        // verify that valid option is specified

        if let Some(amb_te_deg_c) = &self.amb_te_deg_c {
            self.state.amb_te_deg_c = amb_te_deg_c[i];
        }

        if let FcModelTypes::Internal(..) = &self.vehthrm.fc_model {
            self.set_fc_thermal_calcs(i);
        }

        if let CabinHvacModelTypes::Internal(_) = &self.vehthrm.cabin_hvac_model {
            self.set_cab_thermal_calcs(i);
        }

        if self.vehthrm.exhport_model == ComponentModelTypes::Internal {
            self.set_exhport_thermal_calcs(i)
        }

        if self.vehthrm.cat_model == ComponentModelTypes::Internal {
            self.set_cat_thermal_calcs(i)
        }

        if self.vehthrm.fc_model != FcModelTypes::External {
            // Energy balance for fuel converter
            self.state.fc_te_deg_c += (self.state.fc_qdot_kw
                - self.state.fc_qdot_to_amb_kw
                - self.state.fc_qdot_to_htr_kw)
                / self.vehthrm.fc_c_kj__k
                * self.sd.cyc.dt_s_at_i(i)
        }
    }

    /// Solve fuel converter thermal behavior assuming convection parameters of sphere.
    pub fn set_fc_thermal_calcs(&mut self, i: usize) {
        // Constitutive equations for fuel converter
        // calculation of adiabatic flame temperature
        self.state.fc_te_adiabatic_deg_c = self.air.get_te_from_h(
            ((1.0 + self.state.fc_lambda * self.sd.props.fuel_afr_stoich)
                * self.air.get_h(self.state.amb_te_deg_c)
                + self.sd.props.get_fuel_lhv_kj_per_kg() * 1e3 * self.state.fc_lambda.min(1.0))
                / (1.0 + self.state.fc_lambda * self.sd.props.fuel_afr_stoich),
        );

        // limited between 0 and 1, but should really not get near 1
        self.state.fc_qdot_per_net_heat = (self.vehthrm.fc_coeff_from_comb
            * (self.state.fc_te_adiabatic_deg_c - self.state.fc_te_deg_c))
            .min(1.0)
            .max(0.0);

        // heat generation
        self.state.fc_qdot_kw = self.state.fc_qdot_per_net_heat
            * (self.sd.fc_kw_in_ach[i - 1] - self.sd.fc_kw_out_ach[i - 1]);

        // film temperature for external convection calculations
        let fc_air_film_te_deg_c = 0.5 * (self.state.fc_te_deg_c + self.state.amb_te_deg_c);

        // density * speed * diameter / dynamic viscosity
        let fc_air_film_re = self.air.get_rho(fc_air_film_te_deg_c, None)
            * self.sd.mps_ach[i - 1]
            * self.vehthrm.fc_l
            / self.air.get_mu(fc_air_film_te_deg_c);

        // calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
        if self.sd.mps_ach[i - 1] < 1.0 {
            // if stopped, scale based on thermostat opening and constant convection
            self.state.fc_htc_to_amb = interpolate(
                &self.state.fc_te_deg_c,
                &Array1::from_vec(vec![
                    self.vehthrm.tstat_te_sto_deg_c,
                    self.vehthrm.tstat_te_fo_deg_c(),
                ]),
                &Array1::from_vec(vec![
                    self.vehthrm.fc_htc_to_amb_stop,
                    self.vehthrm.fc_htc_to_amb_stop * self.vehthrm.rad_eps,
                ]),
                false,
            )
        } else {
            // Calculate heat transfer coefficient for sphere,
            // from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            let fc_sphere_conv_params = get_sphere_conv_params(fc_air_film_re);
            let fc_htc_to_amb_sphere = (fc_sphere_conv_params.0
                * fc_air_film_re.powf(fc_sphere_conv_params.1))
                * self.air.get_pr(fc_air_film_te_deg_c).powf(1.0 / 3.0)
                * self.air.get_k(fc_air_film_te_deg_c)
                / self.vehthrm.fc_l;
            self.state.fc_htc_to_amb = interpolate(
                &self.state.fc_te_deg_c,
                &Array1::from_vec(vec![
                    self.vehthrm.tstat_te_sto_deg_c,
                    self.vehthrm.tstat_te_fo_deg_c(),
                ]),
                &Array1::from_vec(vec![
                    fc_htc_to_amb_sphere,
                    fc_htc_to_amb_sphere * self.vehthrm.rad_eps,
                ]),
                false,
            )
        }

        self.state.fc_qdot_to_amb_kw = self.state.fc_htc_to_amb
            * 1e-3
            * self.vehthrm.fc_area_ext()
            * (self.state.fc_te_deg_c - self.state.amb_te_deg_c)
    }

    /// Solve cabin thermal behavior.
    pub fn set_cab_thermal_calcs(&mut self, i: usize) {
        if let CabinHvacModelTypes::Internal(hvac_model) = &mut self.vehthrm.cabin_hvac_model {
            // flat plate model for isothermal, mixed-flow from Incropera and deWitt, Fundamentals of Heat and Mass
            // Transfer, 7th Edition
            let cab_te_film_ext_deg_c: f64 =
                0.5 * (self.state.cab_te_deg_c + self.state.amb_te_deg_c);
            let re_l: f64 = self.air.get_rho(cab_te_film_ext_deg_c, None)
                * self.sd.mps_ach[i - 1]
                * self.vehthrm.cab_l_length
                / self.air.get_mu(cab_te_film_ext_deg_c);
            let re_l_crit: f64 = 5.0e5; // critical Re for transition to turbulence

            let nu_l_bar = if re_l < re_l_crit {
                // equation 7.30
                0.664 * re_l.powf(0.5) * self.air.get_pr(cab_te_film_ext_deg_c).powf(1.0 / 3.0)
            } else {
                // equation 7.38
                let a = 871.0; // equation 7.39
                (0.037 * re_l.powf(0.8) - a) * self.air.get_pr(cab_te_film_ext_deg_c)
            };

            if self.sd.mph_ach[i - 1] > 2.0 {
                self.state.cab_qdot_to_amb_kw = 1e-3
                    * (self.vehthrm.cab_l_length * self.vehthrm.cab_l_width)
                    / (1.0
                        / (nu_l_bar * self.air.get_k(cab_te_film_ext_deg_c)
                            / self.vehthrm.cab_l_length)
                        + self.vehthrm.cab_r_to_amb)
                    * (self.state.cab_te_deg_c - self.state.amb_te_deg_c);
            } else {
                self.state.cab_qdot_to_amb_kw = 1e-3
                    * (self.vehthrm.cab_l_length * self.vehthrm.cab_l_width)
                    / (1.0 / self.vehthrm.cab_htc_to_amb_stop + self.vehthrm.cab_r_to_amb)
                    * (self.state.cab_te_deg_c - self.state.amb_te_deg_c);
            }

            let te_delta_vs_set_deg_c = self.state.cab_te_deg_c - hvac_model.te_set_deg_c;
            let te_delta_vs_amb_deg_c = self.state.cab_te_deg_c - self.state.amb_te_deg_c;

            if self.state.cab_te_deg_c <= hvac_model.te_set_deg_c + hvac_model.te_deadband_deg_c
                && self.state.cab_te_deg_c >= hvac_model.te_set_deg_c - hvac_model.te_deadband_deg_c
            {
                // inside deadband; no hvac power is needed

                self.state.cab_qdot_from_hvac_kw = 0.0;
                hvac_model.i_cntrl_kw = 0.0; // reset to 0.0
            } else {
                hvac_model.p_cntrl_kw = hvac_model.p_cntrl_kw_per_deg_c * te_delta_vs_set_deg_c;
                // integral control effort increases in magnitude by
                // 1 time step worth of error
                hvac_model.i_cntrl_kw += hvac_model.i_cntrl_kw_per_deg_c_scnds
                    * te_delta_vs_set_deg_c
                    * self.sd.cyc.dt_s_at_i(i);

                hvac_model.d_cntrl_kw = hvac_model.d_cntrl_kj_per_deg_c
                    * ((self.state.cab_te_deg_c - self.state.cab_prev_te_deg_c)
                        / self.sd.cyc.dt_s_at_i(i));

                // https://en.wikipedia.org/wiki/Coefficient_of_performance#Theoretical_performance_limits
                // cop_ideal is t_h / (t_h - t_c) for heating
                // cop_ideal is t_c / (t_h - t_c) for cooling

                // divide-by-zero protection and realistic limit on COP
                let cop_ideal = if te_delta_vs_amb_deg_c.abs() < 5.0 {
                    // cabin is cooler than ambient + threshold
                    (self.state.cab_te_deg_c + 273.15) / 5.0
                } else {
                    (self.state.cab_te_deg_c + 273.15) / te_delta_vs_amb_deg_c.abs()
                };
                hvac_model.cop = cop_ideal * hvac_model.frac_of_ideal_cop;
                assert!(hvac_model.cop > 0.0);

                if self.state.cab_te_deg_c > hvac_model.te_set_deg_c + hvac_model.te_deadband_deg_c
                {
                    // COOLING MODE; cabin is hotter than set point

                    if hvac_model.i_cntrl_kw < 0.0 {
                        // reset to switch from heating to cooling
                        hvac_model.i_cntrl_kw = 0.0;
                    }
                    hvac_model.i_cntrl_kw = hvac_model.i_cntrl_kw.min(hvac_model.cntrl_max_kw);
                    self.state.cab_qdot_from_hvac_kw =
                        (-hvac_model.p_cntrl_kw - hvac_model.i_cntrl_kw - hvac_model.d_cntrl_kw)
                            .max(-hvac_model.cntrl_max_kw);

                    self.state.cab_hvac_pwr_aux_kw = (-self.state.cab_qdot_from_hvac_kw
                        / hvac_model.cop)
                        .min(hvac_model.pwr_max_aux_load_for_cooling_kw)
                        .max(0.0);
                    // correct if limit is exceeded
                    self.state.cab_qdot_from_hvac_kw =
                        -self.state.cab_hvac_pwr_aux_kw * hvac_model.cop;
                } else {
                    // HEATING MODE; cabin is colder than set point

                    if hvac_model.i_cntrl_kw > 0.0 {
                        // reset to switch from cooling to heating
                        hvac_model.i_cntrl_kw = 0.0;
                    }
                    hvac_model.i_cntrl_kw = hvac_model.i_cntrl_kw.max(-hvac_model.cntrl_max_kw);

                    self.state.cab_qdot_from_hvac_kw =
                        (-hvac_model.p_cntrl_kw - hvac_model.i_cntrl_kw - hvac_model.d_cntrl_kw)
                            .min(hvac_model.cntrl_max_kw);

                    if hvac_model.use_fc_waste_heat {
                        // limit heat transfer to be substantially less than what is physically possible
                        // i.e. the engine can't drop below cabin temperature to heat the cabin
                        self.state.cab_qdot_from_hvac_kw = self
                            .state
                            .cab_qdot_from_hvac_kw
                            .min(
                                (self.state.fc_te_deg_c - self.state.cab_te_deg_c)
                                * 0.1 // so that it's substantially less
                                * self.vehthrm.cab_c_kj__k
                                    / self.sd.cyc.dt_s_at_i(i),
                            )
                            .max(0.0);
                        self.state.fc_qdot_to_htr_kw = self.state.cab_qdot_from_hvac_kw;
                        // TODO: think about what to do for PHEV, which needs careful consideration here
                        // HEV probably also needs careful consideration
                        // There needs to be an engine temperature (e.g. 60°C) below which the engine is forced on
                        assert!(self.sd.veh.veh_pt_type != "BEV");
                        // assume blower has negligible impact on aux load, may want to revise later
                    } else {
                        self.state.cab_hvac_pwr_aux_kw = (self.state.cab_qdot_from_hvac_kw
                            / hvac_model.cop)
                            .min(hvac_model.pwr_max_aux_load_for_cooling_kw)
                            .max(0.0);
                        self.state.cab_qdot_from_hvac_kw =
                            self.state.cab_hvac_pwr_aux_kw * hvac_model.cop;
                    }
                }
            }

            self.state.cab_prev_te_deg_c = self.state.cab_te_deg_c;
            self.state.cab_te_deg_c += (self.state.cab_qdot_from_hvac_kw
                - self.state.cab_qdot_to_amb_kw)
                / self.vehthrm.cab_c_kj__k
                * self.sd.cyc.dt_s_at_i(i);
        }
    }

    /// Solve exhport thermal behavior.
    pub fn set_exhport_thermal_calcs(&mut self, i: usize) {
        // lambda index may need adjustment, depending on how this ends up being modeled.
        self.state.exh_mdot = self.sd.fs_kw_out_ach[i - 1] / self.sd.props.get_fuel_lhv_kj_per_kg()
            * (1.0 + self.sd.props.fuel_afr_stoich * self.state.fc_lambda);
        self.state.exh_hdot_kw = (1.0 - self.state.fc_qdot_per_net_heat)
            * (self.sd.fc_kw_in_ach[i - 1] - self.sd.fc_kw_out_ach[i - 1]);

        if self.state.exh_mdot > 5e-4 {
            self.state.exhport_exh_te_in_deg_c = min(
                self.air
                    .get_te_from_h(self.state.exh_hdot_kw * 1e3 / self.state.exh_mdot),
                self.state.fc_te_adiabatic_deg_c,
            );
            // when flow is small, assume inlet temperature is temporally constant
            // so previous value is not overwritten
        }

        // calculate heat transfer coeff. from exhaust port to ambient [W / (m ** 2 * K)]
        if (self.state.exhport_te_deg_c - self.state.fc_te_deg_c) > 0.0 {
            // if exhaust port is hotter than ambient, make sure heat transfer cannot violate the second law
            self.state.exhport_qdot_to_amb = min(
                // nominal heat transfer to amb
                self.vehthrm.exhport_ha_to_amb
                    * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c),
                // max possible heat transfer to amb
                self.vehthrm.exhport_c_kj__k
                    * 1e3
                    * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        } else {
            // exhaust port cooler than the ambient
            self.state.exhport_qdot_to_amb = max(
                // nominal heat transfer to amb
                self.vehthrm.exhport_ha_to_amb
                    * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c),
                // max possible heat transfer to amb
                self.vehthrm.exhport_c_kj__k
                    * 1e3
                    * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        }

        if (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c) > 0.0 {
            // exhaust hotter than exhaust port
            self.state.exhport_qdot_from_exh = arrmin(&[
                // nominal heat transfer to exhaust port
                self.vehthrm.exhport_ha_int
                    * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c),
                // max possible heat transfer from exhaust
                self.state.exh_mdot
                    * (self.air.get_h(self.state.exhport_exh_te_in_deg_c)
                        - self.air.get_h(self.state.exhport_te_deg_c)),
                // max possible heat transfer to exhaust port
                self.vehthrm.exhport_c_kj__k
                    * 1e3
                    * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            ]);
        } else {
            // exhaust cooler than exhaust port
            self.state.exhport_qdot_from_exh = arrmax(&[
                // nominal heat transfer to exhaust port
                self.vehthrm.exhport_ha_int
                    * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c),
                // max possible heat transfer from exhaust
                self.state.exh_mdot
                    * (self.air.get_h(self.state.exhport_exh_te_in_deg_c)
                        - self.air.get_h(self.state.exhport_te_deg_c)),
                // max possible heat transfer to exhaust port
                self.vehthrm.exhport_c_kj__k
                    * 1e3
                    * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            ]);
        }

        self.state.exhport_qdot_net =
            self.state.exhport_qdot_from_exh - self.state.exhport_qdot_to_amb;
        self.state.exhport_te_deg_c += self.state.exhport_qdot_net
            / (self.vehthrm.exhport_c_kj__k * 1e3)
            * self.sd.cyc.dt_s_at_i(i);
    }

    pub fn thermal_soak_walk(&mut self) {
        self.sd.i = 1;
        while self.sd.i < self.sd.cyc.time_s.len() {
            self.set_thermal_calcs(self.sd.i);
            self.sd.i += 1;
        }
    }

    /// Solve catalyst thermal behavior.
    pub fn set_cat_thermal_calcs(&mut self, i: usize) {
        // external or internal model handling catalyst thermal behavior

        // Constitutive equations for catalyst
        // catalyst film temperature for property calculation
        let cat_te_ext_film_deg_c: f64 = 0.5 * (self.state.cat_te_deg_c + self.state.amb_te_deg_c);
        // density * speed * diameter / dynamic viscosity
        self.state.cat_re_ext = self.air.get_rho(cat_te_ext_film_deg_c, None)
            * self.sd.mps_ach[i - 1]
            * self.vehthrm.cat_l
            / self.air.get_mu(cat_te_ext_film_deg_c);

        // calculate heat transfer coeff. from cat to ambient [W / (m ** 2 * K)]
        if self.sd.mps_ach[i - 1] < 1.0 {
            // if stopped, scale based on constant convection
            self.state.cat_htc_to_amb = self.vehthrm.cat_htc_to_amb_stop;
        } else {
            // if moving, scale based on speed dependent convection and thermostat opening
            // Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            let cat_sphere_conv_params = get_sphere_conv_params(self.state.cat_re_ext);
            self.state.fc_htc_to_amb = (cat_sphere_conv_params.0
                * self.state.cat_re_ext.powf(cat_sphere_conv_params.1))
                * self.air.get_pr(cat_te_ext_film_deg_c).powf(1.0 / 3.0)
                * self.air.get_k(cat_te_ext_film_deg_c)
                / self.vehthrm.cat_l;
        }

        if (self.state.cat_te_deg_c - self.state.amb_te_deg_c) > 0.0 {
            // cat hotter than ambient
            self.state.cat_qdot_to_amb = min(
                // nominal heat transfer to ambient
                self.state.cat_htc_to_amb
                    * self.vehthrm.cat_area_ext()
                    * (self.state.cat_te_deg_c - self.state.amb_te_deg_c),
                // max possible heat transfer to ambient
                self.vehthrm.cat_c_kj__K
                    * 1e3
                    * (self.state.cat_te_deg_c - self.state.amb_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        } else {
            // ambient hotter than cat (less common)
            self.state.cat_qdot_to_amb = max(
                // nominal heat transfer to ambient
                self.state.cat_htc_to_amb
                    * self.vehthrm.cat_area_ext()
                    * (self.state.cat_te_deg_c - self.state.amb_te_deg_c),
                // max possible heat transfer to ambient
                self.vehthrm.cat_c_kj__K
                    * 1e3
                    * (self.state.cat_te_deg_c - self.state.amb_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        }

        if self.state.exh_mdot > 5e-4 {
            self.state.cat_exh_te_in_deg_c = min(
                self.air.get_te_from_h(
                    (self.state.exh_hdot_kw * 1e3 - self.state.exhport_qdot_from_exh)
                        / self.state.exh_mdot,
                ),
                self.state.fc_te_adiabatic_deg_c,
            );
            // when flow is small, assume inlet temperature is temporally constant
            // so previous value is not overwritten
        }

        if (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c) > 0.0 {
            // exhaust hotter than cat
            self.state.cat_qdot_from_exh = min(
                // limited by exhaust heat capacitance flow
                self.state.exh_mdot
                    * (self.air.get_h(self.state.cat_exh_te_in_deg_c)
                        - self.air.get_h(self.state.cat_te_deg_c)),
                // limited by catalyst thermal mass temperature change
                self.vehthrm.cab_c_kj__k
                    * 1e3
                    * (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        } else {
            // cat hotter than exhaust (less common)
            self.state.cat_qdot_from_exh = max(
                // limited by exhaust heat capacitance flow
                self.state.exh_mdot
                    * (self.air.get_h(self.state.cat_exh_te_in_deg_c)
                        - self.air.get_h(self.state.cat_te_deg_c)),
                // limited by catalyst thermal mass temperature change
                self.vehthrm.cat_c_kj__K
                    * 1e3
                    * (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c)
                    / self.sd.cyc.dt_s_at_i(i),
            );
        }

        // catalyst heat generation
        self.state.cat_qdot = 0.0; // TODO: put something substantive here eventually

        // net heat generetion/transfer in cat
        self.state.cat_qdot_net =
            self.state.cat_qdot + self.state.cat_qdot_from_exh - self.state.cat_qdot_to_amb;

        self.state.cat_te_deg_c +=
            self.state.cat_qdot_net * 1e-3 / self.vehthrm.cat_c_kj__K * self.sd.cyc.dt_s_at_i(i);
    }

    pub fn set_misc_calcs(&mut self, i: usize) {
        // if cycle iteration is used, auxInKw must be re-zeroed to trigger the below if statement
        // TODO: this is probably computationally expensive and was probably a workaround for numba
        // figure out a way to not need this
        if self.sd.aux_in_kw.slice(s![i..]).iter().all(|&x| x == 0.0) {
            // if all elements after i-1 are zero, trigger default behavior; otherwise, use override value
            if self.sd.veh.no_elec_aux {
                self.sd.aux_in_kw[i] = self.sd.veh.aux_kw / self.sd.veh.alt_eff;
            } else {
                self.sd.aux_in_kw[i] = self.sd.veh.aux_kw;
            }
        }
        self.sd.aux_in_kw[i] += self.state.cab_hvac_pwr_aux_kw;
        // Is SOC below min threshold?
        if self.sd.soc[i - 1] < (self.sd.veh.min_soc + self.sd.veh.perc_high_acc_buf) {
            self.sd.reached_buff[i] = false;
        } else {
            self.sd.reached_buff[i] = true;
        }

        // Does the engine need to be on for low SOC or high acceleration
        if self.sd.soc[i - 1] < self.sd.veh.min_soc
            || (self.sd.high_acc_fc_on_tag[i - 1] && !(self.sd.reached_buff[i]))
        {
            self.sd.high_acc_fc_on_tag[i] = true
        } else {
            self.sd.high_acc_fc_on_tag[i] = false
        }
        self.sd.max_trac_mps[i] =
            self.sd.mps_ach[i - 1] + (self.sd.veh.max_trac_mps2 * self.sd.cyc.dt_s_at_i(i));
    }

    pub fn set_comp_lims(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_comp_lims(i)
    }

    pub fn set_power_calcs(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_power_calcs(i)
    }

    pub fn set_ach_speed(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_ach_speed(i)
    }

    pub fn set_hybrid_cont_calcs(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_hybrid_cont_calcs(i)
    }

    pub fn set_fc_forced_state_rust(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_fc_forced_state_rust(i)
    }

    pub fn set_hybrid_cont_decisions(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_hybrid_cont_decisions(i)
    }

    pub fn set_fc_power(&mut self, i: usize) -> Result<(), anyhow::Error> {
        if self.sd.veh.fc_max_kw == 0.0 {
            self.sd.fc_kw_out_ach[i] = 0.0;
        } else if self.sd.veh.fc_eff_type == vehicle::H2FC {
            self.sd.fc_kw_out_ach[i] = min(
                self.sd.cur_max_fc_kw_out[i],
                max(
                    0.0,
                    self.sd.mc_elec_kw_in_ach[i] + self.sd.aux_in_kw[i]
                        - self.sd.ess_kw_out_ach[i]
                        - self.sd.roadway_chg_kw_out_ach[i],
                ),
            );
        } else if self.sd.veh.no_elec_sys
            || self.sd.veh.no_elec_aux
            || self.sd.high_acc_fc_on_tag[i]
        {
            self.sd.fc_kw_out_ach[i] = min(
                self.sd.cur_max_fc_kw_out[i],
                max(
                    0.0,
                    self.sd.trans_kw_in_ach[i] - self.sd.mc_mech_kw_out_ach[i]
                        + self.sd.aux_in_kw[i],
                ),
            );
        } else {
            self.sd.fc_kw_out_ach[i] = min(
                self.sd.cur_max_fc_kw_out[i],
                max(
                    0.0,
                    self.sd.trans_kw_in_ach[i] - self.sd.mc_mech_kw_out_ach[i],
                ),
            );
        }

        if self.sd.veh.fc_max_kw == 0.0 {
            self.sd.fc_kw_out_ach_pct[i] = 0.0;
        } else {
            self.sd.fc_kw_out_ach_pct[i] = self.sd.fc_kw_out_ach[i] / self.sd.veh.fc_max_kw;
        }

        if self.sd.fc_kw_out_ach[i] == 0.0 {
            self.sd.fc_kw_in_ach[i] = 0.0;
            self.sd.fc_kw_out_ach_pct[i] = 0.0;
        } else {
            if let FcModelTypes::Internal(fc_temp_eff_model, fc_temp_eff_comp) =
                &self.vehthrm.fc_model
            {
                if let FcTempEffModel::Linear(FcTempEffModelLinear {
                    offset,
                    slope,
                    minimum,
                }) = fc_temp_eff_model
                {
                    self.state.fc_eta_temp_coeff =
                        max(*minimum, min(1.0, offset + slope * self.state.fc_te_deg_c));
                }

                if let FcTempEffModel::Exponential(FcTempEffModelExponential {
                    offset,
                    lag,
                    minimum,
                }) = fc_temp_eff_model
                {
                    match fc_temp_eff_comp {
                        FcTempEffComponent::FuelConverter => {
                            self.state.fc_eta_temp_coeff = (1.0
                                - f64::exp(-1.0 / lag * (self.state.fc_te_deg_c - offset)))
                            .max(*minimum);
                        }
                        FcTempEffComponent::CatAndFC => {
                            if self.state.cat_te_deg_c < self.vehthrm.cat_te_lightoff_deg_c {
                                self.state.fc_eta_temp_coeff = (1.0
                                    - f64::exp(-1.0 / lag * (self.state.fc_te_deg_c - offset)))
                                .max(*minimum);
                                // reduce efficiency to account for catalyst not being lit off
                                self.state.fc_eta_temp_coeff *= self.vehthrm.cat_fc_eta_coeff;
                            }
                        }
                        FcTempEffComponent::Catalyst => {
                            self.state.fc_eta_temp_coeff = (1.0
                                - f64::exp(-1.0 / lag * (self.state.cat_te_deg_c - offset)))
                            .max(*minimum);
                        }
                    }
                }
            }

            if self.sd.fc_kw_out_ach[i] == ndarrmax(&self.sd.veh.input_kw_out_array) {
                self.sd.fc_kw_in_ach[i] = self.sd.fc_kw_out_ach[i]
                    / (self.sd.veh.fc_eff_array.last().unwrap() * self.state.fc_eta_temp_coeff)
            } else {
                self.sd.fc_kw_in_ach[i] = self.sd.fc_kw_out_ach[i]
                    / (self.sd.veh.fc_eff_array[max(
                        1.0,
                        (first_grtr(
                            &self.sd.veh.fc_kw_out_array,
                            min(
                                self.sd.fc_kw_out_ach[i],
                                ndarrmax(&self.sd.veh.input_kw_out_array) - 0.001,
                            ),
                        )
                        .unwrap()
                            - 1) as f64,
                    ) as usize])
                    / self.state.fc_eta_temp_coeff
            }
        }

        // fs out = fc in
        self.sd.fs_kw_out_ach[i] = self.sd.fc_kw_in_ach[i];

        self.sd.fs_kwh_out_ach[i] = self.sd.fs_kw_out_ach[i] * self.sd.cyc.dt_s_at_i(i) / 3.6e3;
        Ok(())
    }

    pub fn set_time_dilation(&mut self, i: usize) -> Result<(), anyhow::Error> {
        self.sd.set_time_dilation(i)
    }

    pub fn set_post_scalars(&mut self) -> Result<(), anyhow::Error> {
        self.sd.set_post_scalars()
    }
}

#[add_pyo3_api(
    #[new]
    pub fn __new__(
        amb_te_deg_c: Option<f64>,
        fc_te_deg_c_init: Option<f64>,
        cab_te_deg_c_init: Option<f64>,
        exhport_te_deg_c_init: Option<f64>,
        cat_te_deg_c_init: Option<f64>,
    ) -> Self {
        Self::new(
            amb_te_deg_c,
            fc_te_deg_c_init,
            cab_te_deg_c_init,
            exhport_te_deg_c_init,
            cat_te_deg_c_init,
        )
    }
)]
#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, HistoryVec)]
/// Struct containing thermal state variables for all thermal components
pub struct ThermalState {
    // fuel converter (engine) variables
    /// fuel converter (engine) temperature [°C]
    pub fc_te_deg_c: f64,
    /// fuel converter temperature efficiency correction
    pub fc_eta_temp_coeff: f64,
    /// fuel converter heat generation per total heat release minus shaft power
    pub fc_qdot_per_net_heat: f64,
    /// fuel converter heat generation [kW]
    pub fc_qdot_kw: f64,
    /// fuel converter convection to ambient [kW]
    pub fc_qdot_to_amb_kw: f64,
    /// fuel converter heat loss to heater core [kW]
    pub fc_qdot_to_htr_kw: f64,
    /// heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
    pub fc_htc_to_amb: f64,
    /// lambda (air/fuel ratio normalized w.r.t. stoich air/fuel ratio) -- 1 is reasonable default
    pub fc_lambda: f64,
    /// lambda-dependent adiabatic flame temperature
    pub fc_te_adiabatic_deg_c: f64,

    // cabin (cab) variables
    /// cabin temperature [°C]
    pub cab_te_deg_c: f64,
    /// previous cabin temperature [°C]
    pub cab_prev_te_deg_c: f64,
    /// cabin solar load [kw]
    pub cab_qdot_solar_kw: f64,
    /// cabin convection to ambient [kw]
    pub cab_qdot_to_amb_kw: f64,
    /// heat transfer to cabin from hvac system
    pub cab_qdot_from_hvac_kw: f64,
    /// aux load from hvac
    pub cab_hvac_pwr_aux_kw: f64,

    // exhaust variables
    /// exhaust mass flow rate [kg/s]
    pub exh_mdot: f64,
    /// exhaust enthalpy flow rate [kw]
    pub exh_hdot_kw: f64,

    /// exhaust port (exhport) variables
    /// exhaust temperature at exhaust port inlet
    pub exhport_exh_te_in_deg_c: f64,
    /// heat transfer from exhport to amb [kw]
    pub exhport_qdot_to_amb: f64,
    /// catalyst temperature [°C]
    pub exhport_te_deg_c: f64,
    /// convection from exhaust to exhport [W]
    /// positive means exhport is receiving heat
    pub exhport_qdot_from_exh: f64,
    /// net heat generation in cat [W]
    pub exhport_qdot_net: f64,

    // catalyst (cat) variables
    /// catalyst heat generation [W]
    pub cat_qdot: f64,
    /// catalytic converter convection coefficient to ambient [W / (m ** 2 * K)]
    pub cat_htc_to_amb: f64,
    /// heat transfer from catalyst to ambient [W]
    pub cat_qdot_to_amb: f64,
    /// catalyst temperature [°C]
    pub cat_te_deg_c: f64,
    /// exhaust temperature at cat inlet
    pub cat_exh_te_in_deg_c: f64,
    /// catalyst external reynolds number
    pub cat_re_ext: f64,
    /// convection from exhaust to cat [W]
    /// positive means cat is receiving heat
    pub cat_qdot_from_exh: f64,
    /// net heat generation in cat [W]
    pub cat_qdot_net: f64,

    /// ambient temperature
    pub amb_te_deg_c: f64,
    #[serde(skip)]
    pub orphaned: bool,
}

impl ThermalState {
    pub fn new(
        amb_te_deg_c: Option<f64>,
        fc_te_deg_c_init: Option<f64>,
        cab_te_deg_c_init: Option<f64>,
        exhport_te_deg_c_init: Option<f64>,
        cat_te_deg_c_init: Option<f64>,
    ) -> Self {
        // Note default temperature is defined twice, see default()
        let default_te_deg_c: f64 = 22.0;
        let amb_te_deg_c = amb_te_deg_c.unwrap_or(default_te_deg_c);
        Self {
            amb_te_deg_c,
            fc_te_deg_c: fc_te_deg_c_init.unwrap_or(amb_te_deg_c),
            cab_te_deg_c: cab_te_deg_c_init.unwrap_or(amb_te_deg_c),
            cab_prev_te_deg_c: cab_te_deg_c_init.unwrap_or(amb_te_deg_c),
            exhport_te_deg_c: exhport_te_deg_c_init.unwrap_or(amb_te_deg_c),
            cat_te_deg_c: cat_te_deg_c_init.unwrap_or(amb_te_deg_c),
            // fc_te_adiabatic_deg_c // chad is pretty sure 'fc_te_adiabatic_deg_c' gets overridden in first time step
            ..Default::default()
        }
    }
}

impl Default for ThermalState {
    fn default() -> Self {
        // Note default temperature is defined twice, see new()
        let default_te_deg_c: f64 = 22.0;

        Self {
            fc_te_deg_c: default_te_deg_c, // overridden by new()
            fc_eta_temp_coeff: 0.0,
            fc_qdot_per_net_heat: 0.0,
            fc_qdot_kw: 0.0,
            fc_qdot_to_amb_kw: 0.0,
            fc_qdot_to_htr_kw: 0.0,
            fc_htc_to_amb: 0.0,
            fc_lambda: 1.0,
            fc_te_adiabatic_deg_c: default_te_deg_c, // this needs to be calculated, get Chad to revisit

            cab_te_deg_c: default_te_deg_c, // overridden by new()
            cab_prev_te_deg_c: default_te_deg_c,
            cab_qdot_solar_kw: 0.0,
            cab_qdot_to_amb_kw: 0.0,
            cab_qdot_from_hvac_kw: 0.0,
            cab_hvac_pwr_aux_kw: 0.0,

            exh_mdot: 0.0,
            exh_hdot_kw: 0.0,

            exhport_exh_te_in_deg_c: default_te_deg_c,
            exhport_qdot_to_amb: 0.0,
            exhport_te_deg_c: default_te_deg_c, // overridden by new()
            exhport_qdot_from_exh: 0.0,
            exhport_qdot_net: 0.0,

            cat_qdot: 0.0,
            cat_htc_to_amb: 0.0,
            cat_qdot_to_amb: 0.0,
            cat_te_deg_c: default_te_deg_c, // overridden by new()
            cat_exh_te_in_deg_c: default_te_deg_c,
            cat_re_ext: 0.0,
            cat_qdot_from_exh: 0.0,
            cat_qdot_net: 0.0,
            amb_te_deg_c: default_te_deg_c, // overridden by new()

            orphaned: false,
        }
    }
}
