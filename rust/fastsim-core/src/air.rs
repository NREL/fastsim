//! Module containing models for air properties.

use crate::utils::interpolate as interp;
use ndarray::Array1;

pub const R_AIR: f64 = 287.0; // J/(kg*K)

/// Fluid Properties for calculations.  
///
/// Values obtained via (in Python):
/// ```python
/// >>> from CoolProp.CoolProp import PropsSI
/// >>> import numpy as np
/// >>> import pandas as pd
/// >>> T_degC = np.logspace(1, np.log10(5e3 + 70), 25) - 70
/// >>> T = T_degC + 273.15
/// >>> prop_dict = {
/// >>>     'T [°C]': T_degC,
/// >>>     'h [J/kg]': [0] * len(T),
/// >>>     'k [W/(m*K)]': [0] * len(T),
/// >>>     'rho [kg/m^3]': [0] * len(T),
/// >>>     'c_p [J/(kg*K)]': [0] * len(T),
/// >>>     'mu [Pa*s]': [0] * len(T),
/// >>> }
///
/// >>> for i, _ in enumerate(T_degC):
/// >>>     prop_dict['h [J/kg]'][i] = f"{PropsSI('H', 'P', 101325, 'T', T[i], 'Air'):.5g}" # specific enthalpy [J/(kg*K)]
/// >>>     prop_dict['k [W/(m*K)]'][i] = f"{PropsSI('L', 'P', 101325, 'T', T[i], 'Air'):.5g}" # thermal conductivity [W/(m*K)]
/// >>>     prop_dict['rho [kg/m^3]'][i] = f"{PropsSI('D', 'P', 101325, 'T', T[i], 'Air'):.5g}" # density [kg/m^3]
/// >>>     prop_dict['c_p [J/(kg*K)]'][i] = f"{PropsSI('C', 'P', 101325, 'T', T[i], 'Air'):.5g}" # density [kg/m^3]
/// >>>     prop_dict['mu [Pa*s]'][i] = f"{PropsSI('V', 'P', 101325, 'T', T[i], 'Air'):.5g}" # viscosity [Pa*s]
///
/// >>> prop_df = pd.DataFrame(data=prop_dict)
/// >>> pd.set_option('display.float_format', lambda x: '%.3g' % x)
/// >>> prop_df = prop_df.apply(np.float64)
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AirProperties {
    /// Private array of temperatures [°C] at which properties are evaluated ()
    te_array_degc: Array1<f64>,
    /// Private thermal conductivity of air \[W / (m * K)\]
    k_array: Array1<f64>,
    /// Private specific heat of air \[J / (kg * K)\]
    c_p_array: Array1<f64>,
    /// Private specific enthalpy of air \[J / kg\] w.r.t. 0K reference
    h_array: Array1<f64>,
    /// Private dynamic viscosity of air \[Pa * s\]
    mu_array: Array1<f64>,
    /// Private Prandtl number of air
    pr_array: Array1<f64>,
}

impl AirProperties {
    /// Returns density [kg/m^3] of air  
    /// Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html  
    /// T = 15.04 - .00649 * h  
    /// p = 101.29 * [(T + 273.1)/288.08]^5.256  
    /// Arguments:  
    /// ----------  
    /// te_air: f64  
    ///     ambient temperature \[°C\] of air   
    /// h=180: Option<f64>  
    ///     evelation \[m\] above sea level   
    pub fn get_rho(&self, te_air: f64, h: Option<f64>) -> f64 {
        let h = h.unwrap_or(180.0);
        let te_standard = 15.04 - 0.00649 * h; // \[degC\]
        let p = 101.29e3 * ((te_standard + 273.1) / 288.08).powf(5.256); // \[Pa\]
        p / (R_AIR * (te_air + 273.15)) // [kg/m**3]
    }

    /// Returns thermal conductivity [W/(m*K)] of air
    /// Arguments:
    /// ----------
    /// te_air: Float
    ///     temperature [°C] of air
    pub fn get_k(&self, te_air: f64) -> f64 {
        interp(&te_air, &self.te_array_degc, &self.k_array, false)
    }

    /// Returns specific heat [J/(kg*K)] of air
    /// Arguments:
    /// ----------
    /// te_air: f64
    ///     temperature [°C] of air
    pub fn get_cp(&self, te_air: f64) -> f64 {
        interp(&te_air, &self.te_array_degc, &self.c_p_array, false)
    }

    /// Returns specific enthalpy [J/kg] of air  
    /// Arguments:  
    /// ----------
    /// te_air: f64
    ///     temperature [°C] of air
    pub fn get_h(&self, te_air: f64) -> f64 {
        interp(&te_air, &self.te_array_degc, &self.h_array, false)
    }

    /// Returns thermal Prandtl number of air
    /// Arguments:
    /// ----------
    /// te_air: f64
    ///     temperature [°C] of air     
    pub fn get_pr(&self, te_air: f64) -> f64 {
        interp(&te_air, &self.te_array_degc, &self.pr_array, false)
    }

    /// Returns dynamic viscosity \[Pa*s\] of air
    /// Arguments:
    /// ----------
    /// te_air: f64
    ///     temperature [°C] of air
    pub fn get_mu(&self, te_air: f64) -> f64 {
        interp(&te_air, &self.te_array_degc, &self.mu_array, false)
    }

    /// Returns temperature [°C] of air
    /// Arguments:
    /// ----------
    /// h: Float
    ///     specific enthalpy [J/kg] of air
    pub fn get_te_from_h(&self, h: f64) -> f64 {
        interp(&h, &self.h_array, &self.te_array_degc, false)
    }
}

impl Default for AirProperties {
    fn default() -> Self {
        let te_array_degc = Array1::from_vec(
            [
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
                5000.,
            ]
            .into(),
        );
        let k_array = Array1::from_vec(
            [
                0.019597, 0.019841, 0.020156, 0.020561, 0.021083, 0.021753, 0.022612, 0.023708,
                0.025102, 0.026867, 0.02909, 0.031875, 0.035342, 0.039633, 0.044917, 0.051398,
                0.059334, 0.069059, 0.081025, 0.095855, 0.11442, 0.13797, 0.16828, 0.20795,
                0.26081,
            ]
            .into(),
        );
        let c_p_array = Array1::from_vec(
            [
                1006.2, 1006.1, 1006., 1005.9, 1005.7, 1005.6, 1005.5, 1005.6, 1005.9, 1006.6,
                1008.3, 1011.6, 1017.9, 1028.9, 1047., 1073.4, 1107.6, 1146.1, 1184.5, 1219.5,
                1250.1, 1277.1, 1301.7, 1324.5, 1347.,
            ]
            .into(),
        );
        let h_array = Array1::from_vec(
            [
                338940., 341930., 345790., 350800., 357290., 365710., 376610., 390750., 409080.,
                432860., 463710., 503800., 556020., 624280., 714030., 832880., 991400., 1203800.,
                1488700., 1869600., 2376700., 3049400., 3939100., 5113600., 6662000.,
            ]
            .into(),
        );
        let mu_array = Array1::from_vec(
            [
                1.4067e-05, 1.4230e-05, 1.4440e-05, 1.4711e-05, 1.5058e-05, 1.5502e-05, 1.6069e-05,
                1.6791e-05, 1.7703e-05, 1.8850e-05, 2.0283e-05, 2.2058e-05, 2.4240e-05, 2.6899e-05,
                3.0112e-05, 3.3966e-05, 3.8567e-05, 4.4049e-05, 5.0595e-05, 5.8464e-05, 6.8036e-05,
                7.9878e-05, 9.4840e-05, 1.1423e-04, 1.4006e-04,
            ]
            .into(),
        );

        let pr_array = mu_array.clone() * c_p_array.clone() / k_array.clone();

        Self {
            te_array_degc,
            k_array,
            c_p_array,
            h_array,
            mu_array,
            pr_array,
        }
    }
}
