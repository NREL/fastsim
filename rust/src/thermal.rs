//! Module for simulating thermal behavior of powertrains

use proc_macros::{add_pyo3_api, HistoryVec};
use ndarray::Array1;
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::{Deserialize, Serialize};
use serde_json;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::f64::consts::PI;

use crate::air::AirProperties;
use crate::simdrive;
use crate::utils::Pyo3VecF64;

use crate::utils::*;

/// Whether FC thermal modeling is handled by FASTSim
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FcModelTypes {
    /// Thermal modeling of fuel converter is handled inside FASTSim
    Internal(FcTempEffModel, FcTempEffComponent),
    /// Thermal modeling of fuel converter will be overriden by wrapper code
    External,
}

impl Default for FcModelTypes {
    fn default() -> Self {
        FcModelTypes::Internal(FcTempEffModel::default(), FcTempEffComponent::default())
    }
}

/// Which commponent temperature affects FC efficency
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FcTempEffComponent {
    /// FC efficiency is purely dependent on cat temp
    Catalyst,
    /// FC efficency is dependent on both cat and FC temp
    Hybrid,
    /// FC efficiency is dependent on FC temp only
    FuelConverter,
}

impl Default for FcTempEffComponent {
    fn default() -> Self {
        FcTempEffComponent::FuelConverter
    }
}

/// Model variants for how FC efficiency depends on temperature
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FcTempEffModel {
    /// Linear temperature dependence
    Linear { offset: f64, slope: f64, min: f64 },
    /// Exponential temperature dependence
    Exponential { offset: f64, lag: f64, min: f64 },
}

impl Default for FcTempEffModel {
    fn default() -> Self {
        // todo: check on reasonableness of default values
        FcTempEffModel::Exponential {
            offset: 0.0,
            lag: 25.0,
            min: 0.2,
        }
    }
}

/// Whether compontent thermal model is handled by FASTSim
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ComponentModelTypes {
    /// Component temperature is handled inside FASTSim
    Internal,
    /// Component temperature will be overriden by wrapper code
    External,
}

impl Default for ComponentModelTypes {
    fn default() -> Self {
        ComponentModelTypes::Internal
    }
}

/// Struct for containing vehicle thermal (and related) parameters.
#[pyclass]
#[add_pyo3_api]
#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct VehicleThermal {
    // fuel converter / engine
    /// parameter fuel converter thermal mass [kJ/K]
    pub fc_c_kj__k: f64,
    /// parameter for engine characteristic length [m] for heat transfer calcs
    pub fc_l: f64,
    /// parameter for heat transfer coeff [W / (m ** 2 * K)] from eng to ambient during vehicle stop
    pub fc_htc_to_amb_stop: f64,
    /// coeff. for fraction of combustion heat that goes to fuel converter (engine)
    /// thermal mass. Remainder goes to environment (e.g. via tailpipe)
    pub fc_coeff_from_comb: f64,
    /// parameter for temperature [°C] at which thermostat starts to open
    pub tstat_te_sto_deg_c: f64,
    /// temperature delta [°C] over which thermostat is partially open
    pub tstat_te_delta_deg_c: f64,
    /// radiator effectiveness -- ratio of active heat rejection from
    /// radiator to passive heat rejection
    pub rad_eps: f64,

    /// temperature-dependent efficiency
    /// fuel converter (engine or fuel cell) thermal model type
    #[api(skip_get, skip_set)]
    pub fc_model: FcModelTypes,

    // battery
    /// battery thermal mass [kJ/K]
    pub ess_c_kj_k: f64,
    /// effective (incl. any thermal management system) heat transfer coefficient from battery to ambient
    pub ess_htc_to_amb: f64,
    // battery controls
    // TODO:
    // need to flesh this out

    // cabin
    /// cabin model internal or external w.r.t. fastsim
    #[api(skip_get, skip_set)]
    pub cabin_model: ComponentModelTypes,
    /// parameter for cabin thermal mass [kJ/K]
    pub cab_c_kj__k: f64,
    /// cabin length [m], modeled as a flat plate
    pub cab_l_length: f64,
    /// cabin width [m], modeled as a flat plate
    pub cab_l_width: f64,
    /// cabin shell thermal resistance [m **2 * K / W]
    pub cab_r_to_amb: f64,
    /// parameter for heat transfer coeff [W / (m ** 2 * K)] from cabin to ambient during
    /// vehicle stop
    pub cab_htc_to_amb_stop: f64,
    // cabin controls
    // TODO
    // need to flesh this out

    // exhaust port
    /// 'external' (effectively no model) is default
    /// exhaust port model type
    #[api(skip_get, skip_set)]
    pub exhport_model: ComponentModelTypes,
    /// thermal conductance [W/K] for heat transfer to ambient
    pub exhport_ha_to_amb: f64,
    /// thermal conductance [W/K] for heat transfer from exhaust
    pub exhport_ha_int: f64,
    /// exhaust port thermal capacitance [kJ/K]
    pub exhport_c_kj__k: f64,

    // catalytic converter (catalyst)
    #[api(skip_get, skip_set)]
    pub cat_model: ComponentModelTypes,
    /// diameter [m] of catalyst as sphere for thermal model
    pub cat_l: f64,
    /// catalyst thermal capacitance [kJ/K]
    pub cat_c_kj__K: f64,
    /// parameter for heat transfer coeff [W / (m ** 2 * K)] from catalyst to ambient
    /// during vehicle stop
    pub cat_htc_to_amb_stop: f64,
    /// lightoff temperature to be used when fc_temp_eff_component == 'hybrid'
    pub cat_te_lightoff_deg_c: f64,
    /// cat engine efficiency coeff. to be used when fc_temp_eff_component == 'hybrid'
    pub cat_fc_eta_coeff: f64,

    // model choices
    /// HVAC model type
    #[api(skip_get, skip_set)]
    pub hvac_model: ComponentModelTypes,

    /// for pyo3 api
    pub orphaned: bool,
}

impl Default for VehicleThermal {
    fn default() -> Self {
        VehicleThermal {
            fc_c_kj__k: 150.0,
            fc_l: 1.0,
            fc_htc_to_amb_stop: 50.0,
            fc_coeff_from_comb: 1e-4,
            tstat_te_sto_deg_c: 85.0,
            tstat_te_delta_deg_c: 5.0,
            rad_eps: 5.0,
            fc_model: FcModelTypes::Internal(
                FcTempEffModel::Exponential {
                    offset: 0.0,
                    lag: 25.0,
                    min: 0.2,
                },
                FcTempEffComponent::FuelConverter,
            ),
            ess_c_kj_k: 200.0,   // similar size to engine
            ess_htc_to_amb: 5.0, // typically well insulated from ambient inside cabin
            cabin_model: ComponentModelTypes::Internal,
            cab_c_kj__k: 125.0,
            cab_l_length: 2.0,
            cab_l_width: 2.0,
            cab_r_to_amb: 0.02,
            cab_htc_to_amb_stop: 10.0,
            exhport_model: ComponentModelTypes::External, // turned off by default
            exhport_ha_to_amb: 5.0,
            exhport_ha_int: 100.0,
            exhport_c_kj__k: 10.0,
            cat_model: ComponentModelTypes::External, // turned off by default
            cat_l: 0.50,
            cat_c_kj__K: 15.0,
            cat_htc_to_amb_stop: 10.0,
            cat_te_lightoff_deg_c: 400.0,
            cat_fc_eta_coeff: 0.3,                     // revisit this
            hvac_model: ComponentModelTypes::External, // turned off by default
            orphaned: false,
        }
    }
}

pub const VEHICLE_THERMAL_DEFAULT_FOLDER: &str = "fastsim/resources";

impl VehicleThermal {
    impl_serde!(VehicleThermal, VEHICLE_THERMAL_DEFAULT_FOLDER);

    pub fn from_file(filename: &str) -> Self {
        Self::from_file_parser(filename).unwrap()
    }

    /// derived temperature [ºC] at which thermostat is fully open
    pub fn tstat_te_fo_deg_c(self) -> f64 {
        self.tstat_te_sto_deg_c + self.tstat_te_delta_deg_c
    }

    /// parameter for engine surface area [m**2] for heat transfer calcs
    pub fn fc_area_ext(self) -> f64
    {
        PI * self.fc_l.powf(2.0/4.0)
    }
    
    /// parameter for catalyst surface area [m**2] for heat transfer calcs
    pub fn cat_area_ext(self) -> f64
    {
        PI * self.cat_l.powf(2.0/4.0)
    }
}

#[add_pyo3_api(
    /// method for instantiating SimDriveHot
    #[new]
    pub fn __new__(filename: &str) -> Self {
        Self::from_file_parser(filename).unwrap()
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
        self.sim_drive(init_soc, aux_in_kw_override);
        Ok(())
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
        self.step();
        Ok(())
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
        self.set_comp_lims(i);
        Ok(())
    }

    #[pyo3(name = "set_power_calcs")]
    /// Calculate power requirements to meet cycle and determine if
    /// cycle can be met.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_power_calcs_py(&mut self, i: usize) -> PyResult<()> {
        self.set_power_calcs(i);
        Ok(())
    }

    #[pyo3(name = "set_ach_speed")]
    // Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
    // Arguments
    // ------------
    // i: index of time step
    pub fn set_ach_speed_py(&mut self, i: usize) -> PyResult<()> {
        self.set_ach_speed(i);
        Ok(())
    }

    #[pyo3(name = "set_hybrid_cont_calcs")]
    /// Hybrid control calculations.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_calcs_py(&mut self, i: usize) -> PyResult<()> {
        self.set_hybrid_cont_calcs(i);
        Ok(())
    }

    #[pyo3(name = "set_fc_forced_state")]
    /// Calculate control variables related to engine on/off state
    /// Arguments
    /// ------------
    /// i: index of time step
    /// `_py` extension is needed to avoid name collision with getter/setter methods
    pub fn set_fc_forced_state_py(&mut self, i: usize) -> PyResult<()> {
        self.set_fc_forced_state_rust(i);
        Ok(())
    }

    #[pyo3(name = "set_hybrid_cont_decisions")]
    /// Hybrid control decisions.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_hybrid_cont_decisions_py(&mut self, i: usize) -> PyResult<()> {
        self.set_hybrid_cont_decisions(i);
        Ok(())
    }

    #[pyo3(name = "set_fc_power")]
    /// Sets power consumption values for the current time step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_fc_power_py(&mut self, i: usize) -> PyResult<()> {
        self.set_fc_power(i);
        Ok(())
    }

    #[pyo3(name = "set_time_dilation")]
    /// Sets the time dilation for the current step.
    /// Arguments
    /// ------------
    /// i: index of time step
    pub fn set_time_dilation_py(&mut self, i: usize) -> PyResult<()> {
        self.set_time_dilation(i);
        Ok(())
    }

    #[pyo3(name = "set_post_scalars")]
    /// Sets scalar variables that can be calculated after a cycle is run.
    /// This includes mpgge, various energy metrics, and others
    pub fn set_post_scalars_py(&mut self) -> PyResult<()> {
        self.set_post_scalars();
        Ok(())
    }

    /// Return length of time arrays
    pub fn len(&self) -> usize {
        self.sd.cyc.time_s.len()
    }    

    /// added to make clippy happy
    /// not sure whether there is any benefit to this or not for our purposes
    /// Return self.cyc.time_is.is_empty()
    pub fn is_empty(&self) -> bool {
        self.sd.cyc.time_s.is_empty()
    }
)]
#[pyclass]
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SimDriveHot {
    #[api(has_orphaned)]
    sd: simdrive::RustSimDrive,
    vehthrm: VehicleThermal,
    #[api(skip_get, skip_set)]
    #[serde(skip)]
    air: AirProperties,
    state: ThermalState,
    history: ThermalStateHistoryVec,
}

pub const SIMDRIVEHOT_DEFAULT_FOLDER: &str = "fastsim/resources";

impl SimDriveHot {
    impl_serde!(SimDriveHot, SIMDRIVEHOT_DEFAULT_FOLDER);

    pub fn from_file(filename: &str) -> Self {
        Self::from_file_parser(filename).unwrap()
    }

    pub fn gap_to_lead_vehicle_m(&self) -> Array1<f64> {
        self.sd.gap_to_lead_vehicle_m()
    }

    pub fn sim_drive(&mut self, init_soc: Option<f64>, aux_in_kw_override: Option<Array1<f64>>) {
        self.sd.sim_drive(init_soc, aux_in_kw_override).unwrap();
    }

    pub fn walk(&mut self, init_soc: f64, aux_in_kw_override: Option<Array1<f64>>) {
        self.sd.walk(init_soc, aux_in_kw_override).unwrap();
    }

    pub fn init_for_step(&mut self, init_soc: f64, aux_in_kw_override: Option<Array1<f64>>) {
        self.sd.init_for_step(init_soc, aux_in_kw_override).unwrap();
    }

    pub fn set_speed_for_target_gap_using_idm(&mut self, i: usize) {
        self.sd.set_speed_for_target_gap_using_idm(i);
    }

    pub fn set_speed_for_target_gap(&mut self, i: usize) {
        self.sd.set_speed_for_target_gap(i);
    }

    pub fn step(&mut self) {
        self.sd.step().unwrap();
        self.history.push(self.state);
    }

    pub fn solve_step(&mut self, i: usize) {
        self.sd.solve_step(i).unwrap();
    }

    pub fn set_thermal_calcs(&mut self, i: usize) {
        // most of the thermal equations are at [i-1] because the various thermally 
        // sensitive component efficiencies dependent on the [i] temperatures, but 
        // these are in turn dependent on [i-1] heat transfer processes  
        // verify that valid option is specified

        if let FcModelTypes::Internal(..) = &self.vehthrm.fc_model {self.set_fc_thermal_calcs(i);}

        if self.vehthrm.hvac_model ==
            ComponentModelTypes::Internal {
                todo!()
                // self.fc_qdot_to_htr_kw[i] = 0.0 // placeholder
            }

        if self.vehthrm.cabin_model == ComponentModelTypes::Internal {
                self.set_cab_thermal_calcs(i);
        }

        if self.vehthrm.exhport_model ==
            ComponentModelTypes::Internal {
                self.set_exhport_thermal_calcs(i)
        }

        if self.vehthrm.cat_model ==
            ComponentModelTypes::Internal {
                self.set_cat_thermal_calcs(i)
        }

        // if self.vehthrm.fc_model == 'internal':
        //     // Energy balance for fuel converter
        //     self.fc_te_degC[i] = self.fc_te_degC[i-1] + (
        //        self.fc_qdot_kw[i] - self.fc_qdot_to_amb_kw[i] - self.fc_qdot_to_htr_kw[i]) / self.vehthrm.fc_C_kJ__K * self.cyc.dt_s[i]
       
    }

    /// Solve fuel converter thermal behavior assuming convection parameters of sphere.
    pub fn set_fc_thermal_calcs(&mut self, i: usize) {
        // Constitutive equations for fuel converter
        // calculation of adiabatic flame temperature
        self.state.fc_te_adiabatic_deg_c = self.air.get_te_from_h(
            ((1.0 + self.state.fc_lambda * self.sd.props.fuel_afr_stoich) * self.air.get_h(self.state.amb_te_deg_c) + 
                self.sd.props.get_fuel_lhv_kj_per_kg() * 1e3 * self.state.fc_lambda.min(1.0)
            ) / (1.0 + self.state.fc_lambda * self.sd.props.fuel_afr_stoich)
        );

        // limited between 0 and 1, but should really not get near 1
        self.state.fc_qdot_per_net_heat = 
            (self.vehthrm.fc_coeff_from_comb * (self.state.fc_te_adiabatic_deg_c - self.state.fc_te_deg_c)).min(1.0).max(0.0);

        // heat generation 
        self.state.fc_qdot_kw = self.state.fc_qdot_per_net_heat * (self.sd.fc_kw_in_ach[i-1] - self.sd.fc_kw_out_ach[i-1]);

        // film temperature for external convection calculations
        let fc_air_film_te_deg_c = 0.5 * (self.state.fc_te_deg_c + self.state.amb_te_deg_c);
    
        // density * speed * diameter / dynamic viscosity
        let fc_air_film_re = self.air.get_rho(fc_air_film_te_deg_c, None) * self.mps_ach[i-1] * self.vehthrm.fc_l / 
            self.air.get_mu(fc_air_film_te_deg_c);

        // calculate heat transfer coeff. from engine to ambient [W / (m ** 2 * K)]
        if self.sd.mps_ach[i-1] < 1.0 {
            // if stopped, scale based on thermostat opening and constant convection
            self.state.fc_htc_to_amb = np.interp(self.fc_te_deg_c[i-1], 
                [self.vehthrm.tstat_te_sto_deg_c, self.vehthrm.tstat_te_fo_deg_c],
                [self.vehthrm.fc_htc_to_amb_stop, self.vehthrm.fc_htc_to_amb_stop * self.vehthrm.rad_eps])
        } else {
            // Calculate heat transfer coefficient for sphere, 
            // from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            fc_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(fc_air_film_re);
            fc_htc_to_ambSphere = (fc_sphere_conv_params[0] * fc_air_film_re ** fc_sphere_conv_params[1]) *
                self.air.get_pr(fc_air_film_te_deg_c).powf(1.0/3.0) *
                self.air.get_k(fc_air_film_te_deg_c) / self.vehthrm.fc_l;
            self.fc_htc_to_amb[i] = interpolate(
                &self.state.fc_te_deg_c,
                [self.vehthrm.tstat_te_sto_deg_c, self.vehthrm.tstat_te_fo_deg_c],
                [fc_htc_to_ambSphere, fc_htc_to_ambSphere * self.vehthrm.rad_eps],
                false
            )
        }

        self.fc_qdot_to_amb_kW[i] = self.fc_htc_to_amb[i] * 1e-3 * self.vehthrm.fc_area_ext * (self.fc_te_deg_c[i-1] - self.amb_te_deg_c[i-1])


    }

    /// Solve cabin thermal behavior.
    pub fn set_cab_thermal_calcs(&mut self, i: usize) {
        // flat plate model for isothermal, mixed-flow from Incropera and deWitt, Fundamentals of Heat and Mass
        // Transfer, 7th Edition
        let cab_te_film_ext_deg_c: f64 = 0.5 * (self.state.cab_te_deg_c + self.state.amb_te_deg_c);
        let re_l: f64 = self.air.get_rho(cab_te_film_ext_deg_c, None) * self.sd.mps_ach[i-1] * self.vehthrm.cab_l_length / self.air.get_mu(cab_te_film_ext_deg_c);
        let re_l_crit: f64 = 5.0e5;  // critical Re for transition to turbulence

        let mut nu_l_bar: f64 = 0.0;
        let a: f64 = 0.0;
        if re_l < re_l_crit {
            // equation 7.30
            nu_l_bar = 0.664 * re_l.powf(0.5) * self.air.get_pr(cab_te_film_ext_deg_c).powf(1.0/3.0);
        } else {
            // equation 7.38
            a = 871.0;  // equation 7.39
            nu_l_bar = (0.037 * re_l.powf(0.8) - a) * self.air.get_pr(cab_te_film_ext_deg_c);
        }
        
        if self.sd.mph_ach[i-1] > 2.0 {        
            self.state.cab_qdot_to_amb_kw = 1e-3 * (self.vehthrm.cab_l_length * self.vehthrm.cab_l_width) / (
                1.0 / (nu_l_bar * self.air.get_k(cab_te_film_ext_deg_c) / self.vehthrm.cab_l_length) + self.vehthrm.cab_r_to_amb
                ) * (self.state.cab_te_deg_c - self.state.amb_te_deg_c);
        } else {
            self.state.cab_qdot_to_amb_kw = 1e-3 * (self.vehthrm.cab_l_length * self.vehthrm.cab_l_width) / (
                1.0 / self.vehthrm.cab_htc_to_amb_stop + self.vehthrm.cab_r_to_amb
                ) * (self.state.cab_te_deg_c - self.state.amb_te_deg_c);
        }
        
        self.state.cab_te_deg_c = self.state.cab_te_deg_c + 
            (self.state.fc_qdot_to_htr_kw - self.state.cab_qdot_to_amb_kw) / self.vehthrm.cab_c_kj__k * self.sd.cyc.dt_s()[i];
    }

    /// Solve exhport thermal behavior.
    pub fn set_exhport_thermal_calcs(&mut self, i: usize) {
        // lambda index may need adjustment, depending on how this ends up being modeled.
        self.state.exh_mdot = self.sd.fs_kw_out_ach[i-1] / self.sd.props.get_fuel_lhv_kj_per_kg() * (1.0 + self.sd.props.fuel_afr_stoich * self.state.fc_lambda);
        self.state.exh_hdot_kw = (1.0 - self.state.fc_qdot_per_net_heat) * (self.sd.fc_kw_in_ach[i-1] - self.sd.fc_kw_out_ach[i-1]);
        
        if self.state.exh_mdot > 5e-4 {
            self.state.exhport_exh_te_in_deg_c = min(
                self.air.get_te_from_h(self.state.exh_hdot_kw * 1e3 / self.state.exh_mdot),
                self.state.fc_te_adiabatic_deg_c);
        } else {
            // when flow is small, assume inlet temperature is temporally constant
            self.state.exhport_exh_te_in_deg_c = self.state.exhport_exh_te_in_deg_c;
        }

        // calculate heat transfer coeff. from exhaust port to ambient [W / (m ** 2 * K)]
        if (self.state.exhport_te_deg_c - self.state.fc_te_deg_c) > 0.0 {
            // if exhaust port is hotter than ambient, make sure heat transfer cannot violate the second law
            self.state.exhport_qdot_to_amb = min(
                // nominal heat transfer to amb
                self.vehthrm.exhport_ha_to_amb * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c),
                // max possible heat transfer to amb
                self.vehthrm.exhport_c_kj__k * 1e3 * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        } else {
            // exhaust port cooler than the ambient
            self.state.exhport_qdot_to_amb = max(
                // nominal heat transfer to amb
                self.vehthrm.exhport_ha_to_amb * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c),
                // max possible heat transfer to amb
                self.vehthrm.exhport_c_kj__k * 1e3 * (self.state.exhport_te_deg_c - self.state.fc_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        }

        if (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c) > 0.0 {
            // exhaust hotter than exhaust port
            self.state.exhport_qdot_from_exh = min(
                // nominal heat transfer to exhaust port
                self.vehthrm.exhport_ha_int * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c),
                min(
                    // max possible heat transfer from exhaust
                    self.state.exh_mdot * (self.air.get_h(self.state.exhport_exh_te_in_deg_c) - self.air.get_h(self.state.exhport_te_deg_c)),
                    // max possible heat transfer to exhaust port
                    self.vehthrm.exhport_c_kj__k * 1e3 * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c) / self.sd.cyc.dt_s()[i]
                )
            );
        } else {
            // exhaust cooler than exhaust port
            self.state.exhport_qdot_from_exh = max(
                // nominal heat transfer to exhaust port
                self.vehthrm.exhport_ha_int * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c),
                max(
                    // max possible heat transfer from exhaust
                    self.state.exh_mdot * (self.air.get_h(self.state.exhport_exh_te_in_deg_c) - self.air.get_h(self.state.exhport_te_deg_c)),
                    // max possible heat transfer to exhaust port
                    self.vehthrm.exhport_c_kj__k * 1e3 * (self.state.exhport_exh_te_in_deg_c - self.state.exhport_te_deg_c) / self.sd.cyc.dt_s()[i]
                )
            );
        }

        self.state.exhport_qdot_net = self.state.exhport_qdot_from_exh - self.state.exhport_qdot_to_amb;
        self.state.exhport_te_deg_c = 
            self.state.exhport_te_deg_c + self.state.exhport_qdot_net / (self.vehthrm.exhport_c_kj__k * 1e3) * self.sd.cyc.dt_s()[i];
    }

    /// Solve catalyst thermal behavior.
    pub fn set_cat_thermal_calcs(&mut self, i: usize) {
        // external or internal model handling catalyst thermal behavior

        // Constitutive equations for catalyst
        // catalyst film temperature for property calculation
        let cat_te_ext_film_deg_c: f64 = 0.5 * (self.state.cat_te_deg_c + self.state.amb_te_deg_c);
        // density * speed * diameter / dynamic viscosity
        self.state.cat_re_ext =
            self.air.get_rho(cat_te_ext_film_deg_c, None) * self.sd.mps_ach[i-1] * self.vehthrm.cat_l 
            / self.air.get_mu(cat_te_ext_film_deg_c);

        // calculate heat transfer coeff. from cat to ambient [W / (m ** 2 * K)]
        if self.sd.mps_ach[i-1] < 1.0 {
            // if stopped, scale based on constant convection
            self.state.cat_htc_to_amb = self.vehthrm.cat_h_to_amb_stop;
        } else {
            // if moving, scale based on speed dependent convection and thermostat opening
            // Nusselt number coefficients from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
            cat_sphere_conv_params = self.conv_calcs.get_sphere_conv_params(self.state.cat_re_ext[i]);
            cat_htc_to_ambSphere = (cat_sphere_conv_params[0] * self.cat_Re_ext[i].powf(cat_sphere_conv_params[1])
                ) * self.air.get_Pr(cat_te_ext_film_deg_c).powf(1.0/3.0) * self.air.get_k(cat_te_ext_film_deg_c) / self.vehthrm.cat_l;
            self.state.fc_htc_to_amb = cat_htc_to_ambSphere
        }

        if (self.state.cat_te_deg_c - self.state.amb_te_deg_c) > 0.0 {
            // cat hotter than ambient
            self.state.cat_qdot_to_amb = min(
                // nominal heat transfer to ambient
                self.state.cat_htc_to_amb * self.vehthrm.cat_area_ext() * (self.state.cat_te_deg_c - self.state.amb_te_deg_c),
                // max possible heat transfer to ambient
                self.vehthrm.cat_c_kj__K * 1e3 * (self.state.cat_te_deg_c - self.state.amb_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        } else {
            // ambient hotter than cat (less common)
            self.state.cat_qdot_to_amb = max(
                // nominal heat transfer to ambient
                self.state.cat_htc_to_amb * self.vehthrm.cat_area_ext() * (self.state.cat_te_deg_c - self.state.amb_te_deg_c),
                // max possible heat transfer to ambient
                self.vehthrm.cat_c_kj__K * 1e3 * (self.state.cat_te_deg_c - self.state.amb_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        }
        
        if self.state.exh_mdot > 5e-4 {
            self.state.cat_exh_te_in_deg_c = min(
                self.air.get_te_from_h((self.state.exh_hdot_kw * 1e3 - self.state.exhport_qdot_from_exh) / self.state.exh_mdot),
                self.state.fc_te_adiabatic_deg_c
            );
        } else {
            // when flow is small, assume inlet temperature is temporally constant
            self.state.cat_exh_te_in_deg_c = self.state.cat_exh_te_in_deg_c
        }

        if (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c) > 0.0 {
            // exhaust hotter than cat
            self.state.cat_qdot_from_exh = min(
                // limited by exhaust heat capacitance flow
                self.state.exh_mdot * (self.air.get_h(self.state.cat_exh_te_in_deg_c) - self.air.get_h(self.state.cat_te_deg_c)),
                // limited by catalyst thermal mass temperature change
                self.vehthrm.cab_c_kj__k * 1e3 * (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        } else {
            // cat hotter than exhaust (less common)
            self.state.cat_qdot_from_exh = max(
                // limited by exhaust heat capacitance flow
                self.state.exh_mdot * (self.air.get_h(self.state.cat_exh_te_in_deg_c) - self.air.get_h(self.state.cat_te_deg_c)),
                // limited by catalyst thermal mass temperature change
                self.vehthrm.cat_c_kj__K * 1e3 * (self.state.cat_exh_te_in_deg_c - self.state.cat_te_deg_c) / self.sd.cyc.dt_s()[i]
            );
        }

        // catalyst heat generation
        self.state.cat_qdot = 0.0;  // TODO: put something substantive here eventually

        // net heat generetion/transfer in cat
        self.state.cat_qdot_net = self.state.cat_qdot + self.state.cat_qdot_from_exh - self.state.cat_qdot_to_amb;

        self.state.cat_te_deg_c = self.state.cat_te_deg_c + self.state.cat_qdot_net * 1e-3 / self.vehthrm.cat_c_kj__k * self.sd.cyc.dt_s()[i];
    }

    pub fn set_misc_calcs(&mut self, i: usize) {
        self.sd.set_misc_calcs(i).unwrap();
    }

    pub fn set_comp_lims(&mut self, i: usize) {
        self.sd.set_comp_lims(i).unwrap();
    }

    pub fn set_power_calcs(&mut self, i: usize) {
        self.sd.set_power_calcs(i).unwrap();
    }

    pub fn set_ach_speed(&mut self, i: usize) {
        self.sd.set_ach_speed(i).unwrap();
    }

    pub fn set_hybrid_cont_calcs(&mut self, i: usize) {
        self.sd.set_hybrid_cont_calcs(i).unwrap();
    }

    pub fn set_fc_forced_state_rust(&mut self, i: usize) {
        self.sd.set_fc_forced_state_rust(i).unwrap();
    }

    pub fn set_hybrid_cont_decisions(&mut self, i: usize) {
        self.sd.set_hybrid_cont_decisions(i).unwrap();
    }

    pub fn set_fc_power(&mut self, i: usize) {
        self.sd.set_fc_power(i).unwrap();
    }
    pub fn set_time_dilation(&mut self, i: usize) {
        self.sd.set_time_dilation(i).unwrap();
    }

    pub fn set_post_scalars(&mut self) {
        self.sd.set_post_scalars().unwrap();
    }
}


#[pyclass]
#[add_pyo3_api]
#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, HistoryVec)]
/// Struct containing thermal state variables for all thermal components
pub struct ThermalState {
    // fuel converter (engine) variables
    /// fuel converter (engine) temperature [°C]
    fc_te_deg_c: f64, 
    /// fuel converter temperature efficiency correction
    fc_eta_temp_coeff: f64,
    /// fuel converter heat generation per total heat release minus shaft power
    fc_qdot_per_net_heat: f64,
    /// fuel converter heat generation [kW]
    fc_qdot_kw: f64,
    /// fuel converter convection to ambient [kW]
    fc_qdot_to_amb_kw: f64,
    /// fuel converter heat loss to heater core [kW]
    fc_qdot_to_htr_kw: f64,
    /// heat transfer coeff [W / (m ** 2 * K)] to amb after arbitration
    fc_htc_to_amb: f64,
    /// lambda (air/fuel ratio normalized w.r.t. stoich air/fuel ratio) -- 1 is reasonable default
    fc_lambda: f64,
    /// lambda-dependent adiabatic flame temperature
    fc_te_adiabatic_deg_c: f64,

    // cabin (cab) variables
    /// cabin temperature [°C]
    cab_te_deg_c: f64,
    /// cabin solar load [kw]
    cab_qdot_solar_kw: f64,
    /// cabin convection to ambient [kw]
    cab_qdot_to_amb_kw: f64, 

    // exhaust variables
    /// exhaust mass flow rate [kg/s]
    exh_mdot: f64,
    /// exhaust enthalpy flow rate [kw]
    exh_hdot_kw: f64,

    /// exhaust port (exhport) variables
    /// exhaust temperature at exhaust port inlet 
    exhport_exh_te_in_deg_c: f64,
    /// heat transfer from exhport to amb [kw]
    exhport_qdot_to_amb: f64,
    /// catalyst temperature [°C]
    exhport_te_deg_c: f64,
    /// convection from exhaust to exhport [W] 
    /// positive means exhport is receiving heat
    exhport_qdot_from_exh: f64,
    /// net heat generation in cat [W]
    exhport_qdot_net: f64,

    // catalyst (cat) variables
    /// catalyst heat generation [W]
    cat_qdot: f64,
    /// catalytic converter convection coefficient to ambient [W / (m ** 2 * K)]
    cat_htc_to_amb: f64,
    /// heat transfer from catalyst to ambient [W]
    cat_qdot_to_amb: f64,
    /// catalyst temperature [°C]
    cat_te_deg_c: f64,
    /// exhaust temperature at cat inlet
    cat_exh_te_in_deg_c: f64,
    /// catalyst external reynolds number
    cat_re_ext: f64,
    /// convection from exhaust to cat [W] 
    /// positive means cat is receiving heat
    cat_qdot_from_exh: f64,
    /// net heat generation in cat [W]
    cat_qdot_net: f64,

    /// ambient temperature
    amb_te_deg_c: f64,
}