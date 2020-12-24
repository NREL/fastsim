"""Various optional utilities that may support some applications of FASTSim."""

from scipy.optimize import curve_fit
import numpy as np
from fastsim import parameters as params
import seaborn as sns
import matplotlib.pyplot as plt

from fastsim import parameters

sns.set()

props = parameters.PhysicalProperties()
R_air = 287  # J/(kg*K)

def get_rho_air(elevation_m, temperature_degC, full_output=False):
    """Returns air density [kg/m**3] for given elevation and temperature.
    Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html"""
    #     T = 15.04 - .00649 * h
    #     p = 101.29 * [(T + 273.1)/288.08]^5.256
    T_standard = 15.04 - 0.00649 * elevation_m  # nasa [degC]
    p = 101.29e3 * ((T_standard + 273.1) / 288.08) ** 5.256  # nasa [Pa]
    rho = p / (R_air * (temperature_degC + 273.15))  # [kg/m**3]

    if not(full_output):
        return rho
    else:
        return rho, p, T_standard

def abc_to_drag_coeffs(veh_kg, veh_fa_m2, a, b, c, show_plots=False):
    """For a given vehicle mass; frontal area; and target A, B, and C 
    coefficients; calculate and return drag and rolling resistance 
    coefficients.

    Arguments:
    ----------
    veh_kg: vehicle mass [kg]
    veh_fa_m2: vehicle frontal area [m^2]
    a, b, c: coastdown coefficients for road load [lb] vs speed [mph]
    show_plots: if True, plots are shown

    It may be worthwhile to have this use get_rho_air() in the future. 
    """

    speed_mph = np.linspace(0, 70, 500)
    dyno_func_lb = np.poly1d([c, b, a])  # polynomial function for pounds vs speed
    dyno_lb = dyno_func_lb(speed_mph)

    def model_func_lb(speed_mps, dragCoef, wheelRrCoef):
        """fastsim-style solution for drag force on vehicle.
        Arguments:
        ---------
        speed: array of vehicle speeds [mps]
        dragCoef: drag coefficient [-]
        wheelRrCoef: rolling resistance coefficient [-]
        """

        out = (veh_kg * props.gravityMPerSec2 * wheelRrCoef +
            0.5 * props.airDensityKgPerM3 * dragCoef * veh_fa_m2
            * speed_mps ** 2) / 4.448
        return out

    (dragCoef, wheelRrCoef), pcov = curve_fit(model_func_lb,
                                              xdata=speed_mph / params.mphPerMps,
                                            ydata=dyno_func_lb(speed_mph),
                                            p0=[0.3, 0.01])
    model_lb = model_func_lb(speed_mph / params.mphPerMps, dragCoef, wheelRrCoef)

    if show_plots:
        plt.figure()    
        plt.plot(speed_mph, dyno_lb, label='dyno')
        plt.plot(speed_mph, model_lb, label='model')
        plt.legend()
        plt.xlabel('Speed [mph]')
        plt.ylabel('Road Load [lb]')

    return dragCoef, wheelRrCoef

def drag_coeffs_to_abc(veh_kg, veh_fa_m2, dragCoef, wheelRrCoef, show_plots=False):
    """For a given vehicle mass, frontal area, dragCoef, and wheelRrCoef, 
    calculate and return ABCs.

    Arguments:
    ----------
    veh_kg: vehicle mass [kg]
    veh_fa_m2: vehicle frontal area [m^2]
    show_plots: if True, plots are shown

    Returns:
    a, b, c: coastdown coefficients for road load [lb] vs speed [mph]

    It may be worthwhile to have this use get_rho_air() in the future. 
    """

    speed_mph = np.linspace(0, 70, 500)

    def model_func_lb(speed_mps, dragCoef, wheelRrCoef):
        """fastsim-style solution for drag force on vehicle.
        Arguments:
        ---------
        speed: array of vehicle speeds [mps]
        dragCoef: drag coefficient [-]
        wheelRrCoef: rolling resistance coefficient [-]
        """

        out = (veh_kg * props.gravityMPerSec2 * wheelRrCoef +
               0.5 * props.airDensityKgPerM3 * dragCoef * veh_fa_m2
               * speed_mps ** 2) / 4.448
        return out

    model_lb = model_func_lb(
        speed_mph / params.mphPerMps, dragCoef, wheelRrCoef)

    # polynomial function for pounds vs speed
    dyno_func_lb = lambda speed_mph, a, b, c: np.poly1d([c, b, a])(speed_mph)

    (a, b, c), pcov = curve_fit(dyno_func_lb,
                        xdata=speed_mph,
                        ydata=model_lb,
                        p0=[10, 0.1, 0.01])
    dyno_lb = dyno_func_lb(speed_mph, a, b, c)

    if show_plots:
        plt.figure()
        plt.plot(speed_mph, dyno_lb, label='dyno')
        plt.plot(speed_mph, model_lb, label='model', linestyle='--')
        plt.legend()
        plt.xlabel('Speed [mph]')
        plt.ylabel('Road Load [lb]')

    return a, b, c

