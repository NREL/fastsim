use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
#[cfg(feature = "pyo3")]
use crate::utils;
#[cfg(feature = "pyo3")]
use crate::utils::Pyo3VecF64;
use proc_macros::{add_pyo3_api, HistoryVec};
use std::f64::consts::PI;

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
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum FcTempEffComponent {
    /// FC efficiency is purely dependent on cat temp
    Catalyst,
    /// FC efficency is dependent on both cat and FC temp
    CatAndFC,
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
    Linear(FcTempEffModelLinear),
    /// Exponential temperature dependence
    Exponential(FcTempEffModelExponential),
}

impl Default for FcTempEffModel {
    fn default() -> Self {
        FcTempEffModel::Exponential(FcTempEffModelExponential::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FcTempEffModelLinear {
    pub offset: f64,
    pub slope: f64,
    pub minimum: f64,
}

impl Default for FcTempEffModelLinear {
    fn default() -> Self {
        Self {
            offset: 0.0,
            slope: 25.0,
            minimum: 0.2,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FcTempEffModelExponential {
    /// temperature at which `fc_eta_temp_coeff` begins to grow
    pub offset: f64,
    /// exponential lag parameter
    pub lag: f64,
    /// minimum value that `fc_eta_temp_coeff` can take
    pub minimum: f64,
}

impl Default for FcTempEffModelExponential {
    fn default() -> Self {
        Self {
            offset: 0.0,
            lag: 25.0,
            minimum: 0.2,
        }
    }
}

/// Struct containing parameters and one time-varying variable for HVAC model
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, HistoryVec)]
#[add_pyo3_api(
    #[classmethod]
    #[pyo3(name = "default")]
    pub fn default_py(_cls: &PyType) -> PyResult<Self> {
        Ok(Self::default())
    }
)]
pub struct HVACModel {
    /// set temperature for component (e.g. cabin, ESS)
    pub te_set_deg_c: f64,
    /// proportional control effort [kW / °C]
    pub p_cntrl_kw_per_deg_c: f64,
    /// integral control effort [kW / (°C-seconds)]
    pub i_cntrl_kw_per_deg_c_scnds: f64,
    /// derivative control effort [kW / (°C/second) = kJ / °C]
    pub d_cntrl_kj_per_deg_c: f64,
    /// Saturation value for integral control [kW].
    /// Whenever `i_cntrl_kw` hit this value, it stops accumulating
    pub cntrl_max_kw: f64,
    /// deadband range.  any cabin temperature within this range of
    /// `te_set_deg_c` results in no HVAC power draw
    pub te_deadband_deg_c: f64,
    /// current proportional control amount
    pub p_cntrl_kw: f64,
    /// current integral control amount
    pub i_cntrl_kw: f64,
    /// current derivative control amount
    pub d_cntrl_kw: f64,
    /// coefficient between 0 and 1 to calculate HVAC efficiency by multiplying by
    /// coefficient of performance (COP)
    pub frac_of_ideal_cop: f64,
    /// whether heat comes from fuel converter
    pub use_fc_waste_heat: bool,
    /// max cooling aux load
    pub pwr_max_aux_load_for_cooling_kw: f64,
    /// coefficient of performance of vapor compression cycle
    pub cop: f64,
    #[serde(skip)]
    orphaned: bool,
}

impl Default for HVACModel {
    fn default() -> Self {
        Self {
            te_set_deg_c: 22.0,
            p_cntrl_kw_per_deg_c: 0.1,
            i_cntrl_kw_per_deg_c_scnds: 0.01,
            d_cntrl_kj_per_deg_c: 0.1,
            cntrl_max_kw: 5.0,
            te_deadband_deg_c: 1.0,
            p_cntrl_kw: 0.0,
            i_cntrl_kw: 0.0,
            d_cntrl_kw: 0.0,
            frac_of_ideal_cop: 0.075, // this is based on Chad's engineering judgment
            use_fc_waste_heat: true,
            pwr_max_aux_load_for_cooling_kw: 5.0,
            cop: 0.0,
            orphaned: Default::default(),
        }
    }
}

/// Whether HVAC model is handled by FASTSim (internal) or not
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum CabinHvacModelTypes {
    /// HVAC is modeled natively
    Internal(HVACModel),
    External,
}

/// Whether compontent thermal model is handled by FASTSim
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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

#[cfg_attr(feature = "pyo3", pyfunction)]
/// Given Reynolds number `re`, return C and m to calculate Nusselt number for
/// sphere, from Incropera's Intro to Heat Transfer, 5th Ed., eq. 7.44
pub fn get_sphere_conv_params(re: f64) -> (f64, f64) {
    let (c, m) = if re < 4.0 {
        (0.989, 0.330)
    } else if re < 40.0 {
        (0.911, 0.385)
    } else if re < 4e3 {
        (0.683, 0.466)
    } else if re < 40e3 {
        (0.193, 0.618)
    } else {
        (0.027, 0.805)
    };
    (c, m)
}

/// Struct for containing vehicle thermal (and related) parameters.
#[allow(non_snake_case)]
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[add_pyo3_api(
    #[classmethod]
    #[pyo3(name = "default")]
    pub fn default_py(_cls: &PyType) -> Self {
        Default::default()
    }

    pub fn set_cabin_hvac_model_internal(
        &mut self,
        hvac_model: HVACModel
    ) -> PyResult<()>{
        Ok(check_orphaned_and_set!(self, cabin_hvac_model, CabinHvacModelTypes::Internal(hvac_model))?)
    }

    pub fn get_cabin_model_internal(&self, ) -> PyResult<HVACModel> {
        if let CabinHvacModelTypes::Internal(hvac_model) = &self.cabin_hvac_model {
            Ok(hvac_model.clone())
        } else {
            Err(PyAttributeError::new_err("HvacModelTypes::External variant currently used."))
        }
    }

    pub fn set_cabin_hvac_model_external(&mut self, ) -> PyResult<()> {
        Ok(check_orphaned_and_set!(self, cabin_hvac_model, CabinHvacModelTypes::External)?)
    }

    pub fn set_fc_model_internal_exponential(
        &mut self,
        offset: f64,
        lag: f64,
        minimum: f64,
        fc_temp_eff_component: String
    ) -> PyResult<()>{
        let fc_temp_eff_comp = match fc_temp_eff_component.as_str() {
            "FuelConverter" => FcTempEffComponent::FuelConverter,
            "Catalyst" => FcTempEffComponent::Catalyst,
            "CatAndFC" => FcTempEffComponent::CatAndFC,
            _ => panic!("Invalid option for fc_temp_eff_component.")
        };

        Ok(check_orphaned_and_set!(
            self,
            fc_model,
            FcModelTypes::Internal(
                FcTempEffModel::Exponential(
                    FcTempEffModelExponential{ offset, lag, minimum }),
                    fc_temp_eff_comp
            )
        )?)
    }

    #[setter]
    pub fn set_fc_exp_offset(&mut self, new_offset: f64) -> PyResult<()> {
        if !self.orphaned {
            self.fc_model = if let FcModelTypes::Internal(fc_temp_eff_model, fc_temp_eff_comp) = &self.fc_model {
                // If model is internal
                if let FcTempEffModel::Exponential(FcTempEffModelExponential{ offset: _, lag, minimum }) = fc_temp_eff_model {
                    // If model is exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ offset: new_offset, lag: *lag, minimum: *minimum }),
                        fc_temp_eff_comp.clone())
                } else {
                    // If model is not exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ offset: new_offset, ..FcTempEffModelExponential::default() }),
                        fc_temp_eff_comp.clone())
                }
            }  else {
                // If model is not internal
                FcModelTypes::Internal(FcTempEffModel::Exponential
                    (FcTempEffModelExponential{ offset: new_offset, ..FcTempEffModelExponential::default() }),
                    FcTempEffComponent::default())
            };
            Ok(())
        } else {
            Err(PyAttributeError::new_err(utils::NESTED_STRUCT_ERR))
        }
    }

    #[setter]
    pub fn set_fc_exp_lag(&mut self, new_lag: f64) -> PyResult<()>{
        if !self.orphaned {
            self.fc_model = if let FcModelTypes::Internal(fc_temp_eff_model, fc_temp_eff_comp) = &self.fc_model {
                // If model is internal
                if let FcTempEffModel::Exponential(FcTempEffModelExponential{ offset, lag: _, minimum }) = fc_temp_eff_model {
                    // If model is exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ offset: *offset, lag: new_lag, minimum: *minimum }),
                        fc_temp_eff_comp.clone())
                } else {
                    // If model is not exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ lag: new_lag, ..FcTempEffModelExponential::default() }),
                        fc_temp_eff_comp.clone())
                }
            }  else {
                // If model is not internal
                FcModelTypes::Internal(FcTempEffModel::Exponential
                    (FcTempEffModelExponential{ lag: new_lag, ..FcTempEffModelExponential::default() }),
                    FcTempEffComponent::default())
            };
            Ok(())
        } else {
            Err(PyAttributeError::new_err(utils::NESTED_STRUCT_ERR))
        }
    }

    #[setter]
    pub fn set_fc_exp_minimum(&mut self, new_minimum: f64) -> PyResult<()> {
        if !self.orphaned {
            self.fc_model = if let FcModelTypes::Internal(fc_temp_eff_model, fc_temp_eff_comp) = &self.fc_model {
                // If model is internal
                if let FcTempEffModel::Exponential(FcTempEffModelExponential{ offset, lag, minimum: _ }) = fc_temp_eff_model {
                    // If model is exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ offset: *offset, lag: *lag, minimum: new_minimum }),
                        fc_temp_eff_comp.clone())
                } else {
                    // If model is not exponential
                    FcModelTypes::Internal(FcTempEffModel::Exponential
                        (FcTempEffModelExponential{ minimum: new_minimum, ..FcTempEffModelExponential::default() }),
                        fc_temp_eff_comp.clone())
                }
            }  else {
                // If model is not internal
                FcModelTypes::Internal(FcTempEffModel::Exponential
                    (FcTempEffModelExponential{ minimum: new_minimum, ..FcTempEffModelExponential::default() }),
                    FcTempEffComponent::default())
            };
            Ok(())
        } else {
            Err(PyAttributeError::new_err(utils::NESTED_STRUCT_ERR))
        }
    }

    #[getter]
    pub fn get_fc_exp_offset(&mut self) -> PyResult<f64> {
        if let FcModelTypes::Internal(FcTempEffModel::Exponential(FcTempEffModelExponential{ offset, ..}), ..) = &self.fc_model {
            Ok(*offset)
        } else {
            Err(PyAttributeError::new_err("fc_model is not Exponential"))
        }
    }

    #[getter]
    pub fn get_fc_exp_lag(&mut self) -> PyResult<f64> {
        if let FcModelTypes::Internal(FcTempEffModel::Exponential(FcTempEffModelExponential{ lag, ..}), ..) = &self.fc_model {
            Ok(*lag)
        } else {
            Err(PyAttributeError::new_err("fc_model is not Exponential"))
        }
    }

    #[getter]
    pub fn get_fc_exp_minimum(&mut self) -> PyResult<f64> {
        if let FcModelTypes::Internal(FcTempEffModel::Exponential(FcTempEffModelExponential{ minimum, ..}), ..) = &self.fc_model {
            Ok(*minimum)
        } else {
            Err(PyAttributeError::new_err("fc_model is not Exponential"))
        }
    }

    // TODO: make setters for all the other enum stuff
)]
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
    pub cabin_hvac_model: CabinHvacModelTypes,
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

    /// for pyo3 api
    #[serde(skip)]
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
            fc_model: FcModelTypes::default(),
            ess_c_kj_k: 200.0,   // similar size to engine
            ess_htc_to_amb: 5.0, // typically well insulated from ambient inside cabin
            cabin_hvac_model: CabinHvacModelTypes::External, // turned off by default
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
            cat_fc_eta_coeff: 0.3, // revisit this
            orphaned: false,
        }
    }
}

impl VehicleThermal {
    /// derived temperature [ºC] at which thermostat is fully open
    pub fn tstat_te_fo_deg_c(&self) -> f64 {
        self.tstat_te_sto_deg_c + self.tstat_te_delta_deg_c
    }

    /// parameter for engine surface area [m**2] for heat transfer calcs
    pub fn fc_area_ext(&self) -> f64 {
        PI * self.fc_l.powf(2.0 / 4.0)
    }

    /// parameter for catalyst surface area [m**2] for heat transfer calcs
    pub fn cat_area_ext(&self) -> f64 {
        PI * self.cat_l.powf(2.0 / 4.0)
    }
}
