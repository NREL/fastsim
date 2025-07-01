use crate::imports::*;

use super::*;
lazy_static! {
    /// room temperature
    pub static ref TE_STD_AIR: si::Temperature = (22. + 273.15) * uc::KELVIN;
    /// pressure of air at 180 m and 22 C
    pub static ref STD_PRESSURE_AIR: si::Pressure = Air::std_pressure_at_elev(*H_STD);
    /// density of air at 180 m ASL and 22 C
    pub static ref STD_DENSITY_AIR: si::MassDensity = *STD_PRESSURE_AIR / (*R_AIR * *TE_STD_AIR);
    /// ideal gas constant for air
    pub static ref R_AIR: si::SpecificHeatCapacity = 287.0 * uc::J_PER_KG_K;
    /// standard elevation above sea level
    pub static ref H_STD: si::Length = 180.0 * uc::M;
}

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
#[serde(deny_unknown_fields)]
pub struct Air {}
impl Init for Air {}
impl SerdeAPI for Air {}

#[pyo3_api]
impl Air {
    #[new]
    fn __new__() -> Self {
        Self {}
    }
    /// Returns density of air \[kg/m^3\]
    /// Source: <https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html>  
    ///
    /// # Equations used
    /// T = 15.04 - .00649 * h  
    /// p = 101.29 * [(T + 273.1)/288.08]^5.256  
    ///
    /// # Arguments  
    /// * `te_air_deg_c` - optional ambient temperature [°C] of air, defaults to 22 C
    /// * `h_m` - optional elevation \[m\] above sea level, defaults to 180 m
    #[staticmethod]
    #[pyo3(name = "get_density")]
    #[pyo3(signature = (te_air_deg_c=None, h_m=None))]
    pub fn get_density_py(te_air_deg_c: Option<f64>, h_m: Option<f64>) -> f64 {
        Self::get_density(
            te_air_deg_c.map(|te_air_deg_c| (te_air_deg_c + 273.15) * uc::KELVIN),
            h_m.map(|h_m| h_m * uc::M),
        )
        .get::<si::kilogram_per_cubic_meter>()
    }

    /// Returns thermal conductivity [W/(m*K)] of air
    /// # Arguments
    /// - `te_air`: temperature [°C] of air
    #[pyo3(name = "get_therm_cond")]
    #[staticmethod]
    pub fn get_therm_cond_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_therm_cond((te_air + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::watt_per_meter_kelvin>(),
        )
    }

    /// Returns constant pressure specific heat [J/(kg*K)] of air
    /// # Arguments
    /// - `te_air`: temperature [°C] of air
    #[pyo3(name = "get_specific_heat_cp")]
    #[staticmethod]
    pub fn get_specific_heat_cp_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_specific_heat_cp((te_air + uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::joule_per_kilogram_kelvin>(),
        )
    }

    /// Returns specific enthalpy [J/kg] of air  
    /// # Arguments  
    /// - `te_air`: temperature [°C] of air
    #[pyo3(name = "get_specific_enthalpy")]
    #[staticmethod]
    pub fn get_specific_enthalpy_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_specific_enthalpy((te_air - uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::joule_per_kilogram>(),
        )
    }

    /// Returns specific energy [J/kg] of air  
    /// # Arguments  
    /// - `te_air`: temperature [°C] of air
    #[pyo3(name = "get_specific_energy")]
    #[staticmethod]
    pub fn get_specific_energy_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_specific_energy((te_air - uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::joule_per_kilogram>(),
        )
    }

    /// Returns thermal Prandtl number of air
    /// # Arguments
    /// - `te_air`: temperature [°C] of air     
    #[pyo3(name = "get_pr")]
    #[staticmethod]
    pub fn get_pr_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(Self::get_pr((te_air - uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?.get::<si::ratio>())
    }

    /// Returns dynamic viscosity \[Pa*s\] of air
    /// # Arguments
    /// te_air: temperature [°C] of air
    #[pyo3(name = "get_dyn_visc")]
    #[staticmethod]
    pub fn get_dyn_visc_py(te_air: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_dyn_visc((te_air - uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::pascal_second>(),
        )
    }

    /// Returns temperature [°C] of air
    /// # Arguments
    /// - `h`: specific enthalpy of air \[J/kg\]
    #[pyo3(name = "get_te_from_h")]
    #[staticmethod]
    pub fn get_te_from_h_py(h: f64) -> anyhow::Result<f64> {
        Ok(Self::get_te_from_h(h * uc::J_PER_KG)?.get::<si::degree_celsius>())
    }

    /// Returns temperature [°C] of air
    /// # Arguments
    /// - `u`: specific energy of air \[J/kg\]
    #[pyo3(name = "get_te_from_u")]
    #[staticmethod]
    pub fn get_te_from_u_py(u: f64) -> anyhow::Result<f64> {
        Ok(Self::get_te_from_u(u * uc::J_PER_KG)?.get::<si::degree_celsius>())
    }
}

impl Air {
    /// Returns density of air with computational optimizations for default inputs
    /// Source: <https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html>  
    /// Note that if `None` is passed for either argument, function evaluation should be faster
    ///
    /// # Equations used
    /// - T = 15.04 - 0.00649 * h  
    /// - p = 101.29 * ((T + 273.1) / 288.08) ^ 5.256  
    ///
    /// # Arguments  
    /// * `te_air` - ambient temperature of air, defaults to 22 C
    /// * `h` - elevation above sea level, defaults to 180 m
    pub fn get_density(te_air: Option<si::Temperature>, h: Option<si::Length>) -> si::MassDensity {
        let std_pressure_at_elev = Self::std_pressure_at_elev;
        match (h, te_air) {
            (None, None) => *STD_DENSITY_AIR,
            (None, Some(te_air)) => *STD_PRESSURE_AIR / *R_AIR / te_air,
            (Some(h_val), None) => std_pressure_at_elev(h_val) / *R_AIR / *TE_STD_AIR,
            (Some(h_val), Some(te_air)) => std_pressure_at_elev(h_val) / *R_AIR / te_air,
        }
    }

    fn std_temp_at_elev(h: si::Length) -> si::Temperature {
        (15.04 - 0.00649 * h.get::<si::meter>() + 273.15) * uc::KELVIN
    }

    fn std_pressure_at_elev(h: si::Length) -> si::Pressure {
        (101.29e3 * uc::PASCAL)
            * ((Self::std_temp_at_elev(h) / (288.08 * uc::KELVIN))
                .get::<si::ratio>()
                .powf(5.256))
    }

    /// Returns thermal conductivity of air
    /// # Arguments
    /// - `te_air`: temperature of air
    pub fn get_therm_cond(te_air: si::Temperature) -> anyhow::Result<si::ThermalConductivity> {
        Ok(
            asp::THERMAL_CONDUCTIVITY_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])?
                * uc::WATT_PER_METER_KELVIN,
        )
    }

    /// Returns constant pressure specific heat of air
    /// # Arguments
    /// - `te_air`: temperature of air
    pub fn get_specific_heat_cp(
        te_air: si::Temperature,
    ) -> anyhow::Result<si::SpecificHeatCapacity> {
        Ok(asp::C_P_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])? * uc::J_PER_KG_K)
    }

    /// Returns specific enthalpy of air  
    /// # Arguments  
    /// - `te_air`: temperature of air
    pub fn get_specific_enthalpy(te_air: si::Temperature) -> anyhow::Result<si::SpecificEnergy> {
        Ok(asp::ENTHALPY_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])? * uc::J_PER_KG)
    }

    /// Returns specific energy of air  
    /// # Arguments  
    /// - `te_air`: temperature of air
    pub fn get_specific_energy(te_air: si::Temperature) -> anyhow::Result<si::SpecificEnergy> {
        Ok(asp::ENERGY_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])? * uc::J_PER_KG)
    }

    /// Returns thermal Prandtl number of air
    /// # Arguments
    /// - `te_air`: temperature of air     
    pub fn get_pr(te_air: si::Temperature) -> anyhow::Result<si::Ratio> {
        Ok(asp::PRANDTL_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])? * uc::R)
    }

    /// Returns dynamic viscosity \[Pa*s\] of air
    /// # Arguments
    /// te_air: temperature of air
    pub fn get_dyn_visc(te_air: si::Temperature) -> anyhow::Result<si::DynamicViscosity> {
        Ok(
            asp::DYN_VISC_INTERP.interpolate(&[te_air.get::<si::degree_celsius>()])?
                * uc::PASCAL_SECOND,
        )
    }

    /// Returns temperature of air
    /// # Arguments
    /// `h`: specific enthalpy of air \[J/kg\]
    pub fn get_te_from_h(h: si::SpecificEnergy) -> anyhow::Result<si::Temperature> {
        Ok(asp::TEMP_FROM_ENTHALPY.interpolate(&[h.get::<si::joule_per_kilogram>()])? * uc::KELVIN)
    }

    /// Returns temperature of air
    /// # Arguments
    /// `u`: specific energy of air \[J/kg\]
    pub fn get_te_from_u(u: si::SpecificEnergy) -> anyhow::Result<si::Temperature> {
        Ok(asp::TEMP_FROM_ENERGY.interpolate(&[u.get::<si::joule_per_kilogram>()])? * uc::KELVIN)
    }
}

use air_static_props as asp;

/// Air fluid properties for calculations.  
///
/// Values obtained via (in Python, after running `pip install CoolProp`):
/// ```python
/// from CoolProp.CoolProp import PropsSI
/// import numpy as np
/// import pandas as pd
/// T_degC = np.logspace(1, np.log10(5e3 + 70), 25) - 70
/// T = T_degC + 273.15
/// prop_dict = {
///     'T [°C]': T_degC,
///     'h [J/kg]': [0] * len(T),
///     'u [J/kg]': [0] * len(T),
///     'k [W/(m*K)]': [0] * len(T),
///     'rho [kg/m^3]': [0] * len(T),
///     'c_p [J/(kg*K)]': [0] * len(T),
///     'mu [Pa*s]': [0] * len(T),
/// }
///
/// species = "Air"
///
/// for i, _ in enumerate(T_degC):
///     prop_dict['h [J/kg]'][i] = f"{PropsSI('H', 'P', 101325, 'T', T[i], species):.5g}" # specific enthalpy [J/(kg*K)]
///     prop_dict['u [J/kg]'][i] = f"{PropsSI('U', 'P', 101325, 'T', T[i], species):.5g}" # specific enthalpy [J/(kg*K)]
///     prop_dict['k [W/(m*K)]'][i] = f"{PropsSI('L', 'P', 101325, 'T', T[i], species):.5g}" # thermal conductivity [W/(m*K)]
///     prop_dict['rho [kg/m^3]'][i] = f"{PropsSI('D', 'P', 101325, 'T', T[i], species):.5g}" # density [kg/m^3]
///     prop_dict['c_p [J/(kg*K)]'][i] = f"{PropsSI('C', 'P', 101325, 'T', T[i], species):.5g}" # density [kg/m^3]
///     prop_dict['mu [Pa*s]'][i] = f"{PropsSI('V', 'P', 101325, 'T', T[i], species):.5g}" # viscosity [Pa*s]
///
/// prop_df = pd.DataFrame(data=prop_dict)
/// pd.set_option('display.float_format', lambda x: '%.3g' % x)
/// prop_df = prop_df.apply(np.float64)
/// ```
mod air_static_props {
    use super::*;
    lazy_static! {
        /// Array of temperatures at which properties are evaluated
        static ref TEMPERATURE_DEG_C_VALUES: Array1<f64> = array![
            -60.,
            -57.03690616,
            -53.1958198,
            -48.21658352,
            -41.7619528,
            -33.39475442,
            -22.54827664,
            -8.48788571,
            9.73873099,
            33.36606527,
            63.99440042,
            103.69819869,
            155.16660498,
            221.88558305,
            308.37402042,
            420.48979341,
            565.82652205,
            754.22788725,
            998.45434496,
            1315.04739396,
            1725.44993435,
            2257.45859876,
            2947.10642291,
            3841.10336915,
            5000.
        ];
        pub static ref TEMP_FROM_ENTHALPY: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            ENTHALPY_VALUES.view(),
            TEMPERATURE_DEG_C_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        pub static ref TEMP_FROM_ENERGY: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            ENERGY_VALUES.view(),
            TEMPERATURE_DEG_C_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// Thermal conductivity values of air corresponding to temperature values
        /// \[W/m-K\]
        static ref THERMAL_CONDUCTIVITY_VALUES: Array1<f64> = array![
            0.019597,
            0.019841,
            0.020156,
            0.020561,
            0.021083,
            0.021753,
            0.022612,
            0.023708,
            0.025102,
            0.026867,
            0.02909,
            0.031875,
            0.035342,
            0.039633,
            0.044917,
            0.051398,
            0.059334,
            0.069059,
            0.081025,
            0.095855,
            0.11442,
            0.13797,
            0.16828,
            0.20795,
            0.26081,
        ];
        pub static ref THERMAL_CONDUCTIVITY_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            THERMAL_CONDUCTIVITY_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// Specific heat values of air corresponding to temperature values
        /// \[J/kg-K\]
        static ref C_P_VALUES: Array1<f64> = array![
            1006.2,
            1006.1,
            1006.,
            1005.9,
            1005.7,
            1005.6,
            1005.5,
            1005.6,
            1005.9,
            1006.6,
            1008.3,
            1011.6,
            1017.9,
            1028.9,
            1047.,
            1073.4,
            1107.6,
            1146.1,
            1184.5,
            1219.5,
            1250.1,
            1277.1,
            1301.7,
            1324.5,
            1347.,
        ];
        pub static ref C_P_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            C_P_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// \[J/kg\]
        static ref ENTHALPY_VALUES: Array1<f64> = array![
            338940.,
            341930.,
            345790.,
            350800.,
            357290.,
            365710.,
            376610.,
            390750.,
            409080.,
            432860.,
            463710.,
            503800.,
            556020.,
            624280.,
            714030.,
            832880.,
            991400.,
            1203800.,
            1488700.,
            1869600.,
            2376700.,
            3049400.,
            3939100.,
            5113600.,
            6662000.
        ];
        pub static ref ENTHALPY_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            ENTHALPY_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// \[J/kg\]
        pub static ref ENERGY_VALUES: Array1<f64> = array![
            277880.,
            280000.,
            282760.,
            286330.,
            290960.,
            296960.,
            304750.,
            314840.,
            327920.,
            344890.,
            366940.,
            395620.,
            433040.,
            482140.,
            547050.,
            633700.,
            750490.,
            908830.,
            1123600.,
            1413600.,
            1802900.,
            2322900.,
            3014700.,
            3932500.,
            5148300.,
        ];
        pub static ref ENERGY_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            ENERGY_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// \[Pa-s\]
        static ref DYN_VISCOSITY_VALUES: Array1<f64> = array![
            1.4067e-05,
            1.4230e-05,
            1.4440e-05,
            1.4711e-05,
            1.5058e-05,
            1.5502e-05,
            1.6069e-05,
            1.6791e-05,
            1.7703e-05,
            1.8850e-05,
            2.0283e-05,
            2.2058e-05,
            2.4240e-05,
            2.6899e-05,
            3.0112e-05,
            3.3966e-05,
            3.8567e-05,
            4.4049e-05,
            5.0595e-05,
            5.8464e-05,
            6.8036e-05,
            7.9878e-05,
            9.4840e-05,
            1.1423e-04,
            1.4006e-04
        ];
        pub static ref DYN_VISC_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            DYN_VISCOSITY_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        static ref PRANDTL_VALUES: Array1<f64> = DYN_VISCOSITY_VALUES
            .iter()
            .zip(C_P_VALUES.iter())
            .zip(THERMAL_CONDUCTIVITY_VALUES.iter())
            .map(|((mu, c_p), k)| mu * c_p / k)
            .collect();
        pub static ref PRANDTL_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
            TEMPERATURE_DEG_C_VALUES.view(),
            PRANDTL_VALUES.view(),
            strategy::Linear,
            Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
    }
}

use octane_static_props as osp;

/// Octane (as a surrogate for gasoline) fluid properties for calculations.  
///
/// Values obtained via (in Python, after running `pip install CoolProp`):
/// ```python
/// from CoolProp.CoolProp import PropsSI
/// import numpy as np
/// import pandas as pd
/// T_degC = np.logspace(1, np.log10(5e3 + 70), 25) - 50
/// T = T_degC + 273.15
/// prop_dict = {
///     'T [°C]': T_degC,
///     'h [J/kg]': [0] * len(T),
///     'u [J/kg]': [0] * len(T),
///     'k [W/(m*K)]': [0] * len(T),
///     'rho [kg/m^3]': [0] * len(T),
///     'c_p [J/(kg*K)]': [0] * len(T),
///     'mu [Pa*s]': [0] * len(T),
/// }
///
/// species = "Octane"
///
/// for i, _ in enumerate(T_degC):
///     prop_dict['h [J/kg]'][i] = f"{PropsSI('H', 'P', 101325, 'T', T[i], species):.5g}" # specific enthalpy [J/(kg*K)]
///     prop_dict['u [J/kg]'][i] = f"{PropsSI('U', 'P', 101325, 'T', T[i], species):.5g}" # specific enthalpy [J/(kg*K)]
///     prop_dict['k [W/(m*K)]'][i] = f"{PropsSI('L', 'P', 101325, 'T', T[i], species):.5g}" # thermal conductivity [W/(m*K)]
///     prop_dict['rho [kg/m^3]'][i] = f"{PropsSI('D', 'P', 101325, 'T', T[i], species):.5g}" # density [kg/m^3]
///     prop_dict['c_p [J/(kg*K)]'][i] = f"{PropsSI('C', 'P', 101325, 'T', T[i], species):.5g}" # density [kg/m^3]
///     prop_dict['mu [Pa*s]'][i] = f"{PropsSI('V', 'P', 101325, 'T', T[i], species):.5g}" # viscosity [Pa*s]
///
/// prop_df = pd.DataFrame(data=prop_dict)
/// pd.set_option('display.float_format', lambda x: '%.3g' % x)
/// prop_df = prop_df.apply(np.float64)
/// ```
mod octane_static_props {
    use super::*;
    lazy_static! {
        /// Array of temperatures at which properties are evaluated
        static ref TEMPERATURE_DEG_C_VALUES: Array1<f64> = array![
            -4.00000000e+01,
            -3.70369062e+01,
            -3.31958198e+01,
            -2.82165835e+01,
            -2.17619528e+01,
            -1.33947544e+01,
            -2.54827664e+00,
            1.15121143e+01,
            2.97387310e+01,
            5.33660653e+01,
            8.39944004e+01,
            1.23698199e+02,
            1.75166605e+02,
            2.41885583e+02,
            3.28374020e+02,
            4.40489793e+02,
            5.85826522e+02,
            7.74227887e+02,
            1.01845434e+03,
            1.33504739e+03,
            1.74544993e+03,
            2.27745860e+03,
            2.96710642e+03,
            3.86110337e+03,
            5.02000000e+03
        ];
        pub static ref TEMP_FROM_ENERGY: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
           ENERGY_VALUES.view(),
           TEMPERATURE_DEG_C_VALUES.view(),
           strategy::Linear,
           Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
        /// \[J/kg\]
        pub static ref ENERGY_VALUES: Array1<f64> = array![
            -3.8247e+05,
            -3.7645e+05,
            -3.6862e+05,
            -3.5841e+05,
            -3.4507e+05,
            -3.2760e+05,
            -3.0464e+05,
            -2.7432e+05,
            -2.3400e+05,
            -1.7991e+05,
            -1.0649e+05,
            -5.3074e+03,
            3.8083e+05,
            5.3958e+05,
            7.6926e+05,
            1.1024e+06,
            1.5836e+06,
            2.2729e+06,
            3.2470e+06,
            4.6015e+06,
            6.4541e+06,
            8.9500e+06,
            1.2272e+07,
            1.6654e+07,
            2.2399e+07,
        ];
        pub static ref ENERGY_INTERP: Interp1DViewed<&'static f64, strategy::Linear> = Interp1D::new(
           TEMPERATURE_DEG_C_VALUES.view(),
           ENERGY_VALUES.view(),
           strategy::Linear,
           Extrapolate::Error,
        ).unwrap_or_else(|_| panic!("Failed to construct gas properties vec"));
    }
}

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
#[serde(deny_unknown_fields)]
pub struct Octane {}
impl Init for Octane {}
impl SerdeAPI for Octane {}

#[pyo3_api]
impl Octane {
    /// Returns specific energy [J/kg] of octane  
    /// # Arguments  
    /// - `te_octane`: temperature [°C] of octane
    #[pyo3(name = "get_specific_energy")]
    #[staticmethod]
    pub fn get_specific_energy_py(te_octane: f64) -> anyhow::Result<f64> {
        Ok(
            Self::get_specific_energy((te_octane - uc::CELSIUS_TO_KELVIN) * uc::KELVIN)?
                .get::<si::joule_per_kilogram>(),
        )
    }

    /// Returns temperature [°C] of octane
    /// # Arguments
    /// - `u`: specific energy of octane \[J/kg\]
    #[pyo3(name = "get_te_from_u")]
    #[staticmethod]
    pub fn get_te_from_u_py(u: f64) -> anyhow::Result<f64> {
        Ok(Self::get_te_from_u(u * uc::J_PER_KG)?.get::<si::degree_celsius>())
    }
}

impl Octane {
    /// Returns specific energy of octane  
    /// # Arguments  
    /// - `te_octane`: temperature of octane
    pub fn get_specific_energy(te_octane: si::Temperature) -> anyhow::Result<si::SpecificEnergy> {
        Ok(
            osp::ENERGY_INTERP.interpolate(&[te_octane.get::<si::degree_celsius>()])?
                * uc::J_PER_KG,
        )
    }

    /// Returns temperature of octane
    /// # Arguments
    /// `u`: specific energy of octane \[J/kg\]
    pub fn get_te_from_u(u: si::SpecificEnergy) -> anyhow::Result<si::Temperature> {
        Ok(osp::TEMP_FROM_ENERGY.interpolate(&[u.get::<si::joule_per_kilogram>()])? * uc::KELVIN)
    }
}

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
