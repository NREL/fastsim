"""Various optional utilities that may support some applications of FASTSim."""

from typing import Callable
import inspect
from scipy.optimize import minimize
import numpy as np
from fastsim import parameters as params
import seaborn as sns
import matplotlib.pyplot as plt
import re
from typing import Tuple

from fastsim import parameters

sns.set()

props = parameters.PhysicalProperties()
R_air = 287  # J/(kg*K)


def get_rho_air(temperature_degC, elevation_m=180):
    """Returns air density [kg/m**3] for given elevation and temperature.
    Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html
    Arguments:
    ----------
    temperature_degC : ambient temperature [°C]
    elevation_m : elevation above sea level [m].
        Default 180 m is for Chicago, IL"""
    #     T = 15.04 - .00649 * h
    #     p = 101.29 * [(T + 273.1)/288.08]^5.256
    T_standard = 15.04 - 0.00649 * elevation_m  # nasa [degC]
    p = 101.29e3 * ((T_standard + 273.1) / 288.08) ** 5.256  # nasa [Pa]
    rho = p / (R_air * (temperature_degC + 273.15))  # [kg/m**3]

    return rho

# TODO: implement these functions with sim_drive and a vehicle


def abc_to_drag_coeffs(veh,
                       a_lbf: float, b_lbf__mph: float, c_lbf__mph2: float, show_plots: bool = False,
                       use_rust=True) -> Tuple[float, float]:
    """For a given vehicle and target A, B, and C
    coefficients; calculate and return drag and rolling resistance
    coefficients.

    Arguments:
    ----------
    veh: vehicle.Vehicle with all parameters correct except for drag and rolling resistance coefficients
    a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    show_plots: if True, plots are shown
    use_rust: if True, use rust implementation of drag coefficient calculation.

    It may be worthwhile to have this use get_rho_air() in the future.
    """

    import fastsim as fsim  # to avoid circular import

    cd_len = 300

    cyc = fsim.cycle.Cycle.from_dict({
        'time_s': np.arange(0, cd_len),
        'mps': np.linspace(70.0 / parameters.MPH_PER_MPS, 0, cd_len)
    })

    if use_rust:
        cyc = cyc.to_rust()
        veh = veh.to_rust()

    # polynomial function for pounds vs speed
    dyno_func_lb = np.poly1d([c_lbf__mph2, b_lbf__mph, a_lbf])

    def get_err(x):
        """fastsim-style solution for drag force on vehicle.
        Arguments:
        ---------
        x: (speed: array of vehicle speeds [mps], dragCoef: drag coefficient [-])
        wheelRrCoef: rolling resistance coefficient [-]
        """
        drag_coef, wheel_rr_coef = x
        veh.drag_coef = drag_coef
        veh.wheel_rr_coef = wheel_rr_coef

        if use_rust:
            sd_coast = fsim.simdrive.RustSimDrive(cyc, veh)
        else:
            sd_coast = fsim.simdrive.SimDrive(cyc, veh)
        sd_coast.impose_coast = [True] * len(sd_coast.impose_coast)
        sd_coast.sim_drive()

        cutoff = np.where(np.array(sd_coast.mps_ach) < 0.1)[0][0]

        err = fsim.cal.get_error_val(
            ((np.array(sd_coast.drag_kw) + np.array(sd_coast.cyc_rr_kw)) /
                np.array(sd_coast.mps_ach))[:cutoff],
            (dyno_func_lb(sd_coast.mph_ach) * fsim.params.N_PER_LBF)[:cutoff],
            cyc.time_s[:cutoff],
            normalize=False
        )

        return err

    res = minimize(get_err, x0=np.array([0.3, 0.01]))
    (drag_coef, wheel_rr_coef) = res.x

    veh.drag_coef, veh.wheel_rr_coef = drag_coef, wheel_rr_coef
    if use_rust:
        sd_coast = fsim.simdrive.RustSimDrive(cyc, veh)
    else:
        sd_coast = fsim.simdrive.SimDrive(cyc, veh)
    sd_coast.impose_coast = [True] * len(sd_coast.impose_coast)
    sd_coast.sim_drive()

    if show_plots:
        plt.figure()
        plt.plot(
            sd_coast.mph_ach,
            np.array(sd_coast.trans_kw_out_ach) /
            np.array(sd_coast.mps_ach) / params.N_PER_LBF,
            label='sim_drive')
        plt.plot(sd_coast.mph_ach, dyno_func_lb(
            sd_coast.mph_ach), label='ABCs')
        plt.legend()
        plt.xlabel('Speed [mph]')
        plt.ylabel('Road Load [lb]')

        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(cyc.time_s,
                   np.array(sd_coast.trans_kw_out_ach) /
                   np.array(sd_coast.mps_ach)
                   )
        ax[0].set_ylabel("Road Load [N]")

        ax[1].plot(cyc.time_s, sd_coast.trans_kw_out_ach, label='ach')
        ax[1].plot(cyc.time_s, sd_coast.cur_max_trans_kw_out, label='max')
        ax[1].set_ylabel('trans kw')

        ax[-1].plot(cyc.time_s, sd_coast.mph_ach)
        ax[-1].set_ylabel("mph")
        ax[-1].set_xlabel('Time [s]')

    return drag_coef, wheel_rr_coef

# TODO, make drag_coeffs and abcs generate plots of drag force vs speed
# and implement units in the inputs


def drag_coeffs_to_abc(veh_kg: float, veh_fa_m2: float, drag_coef: float, wheel_rr_coef: float, show_plots: bool = False) -> Tuple[float, float, float]:
    """For a given vehicle mass, frontal area, dragCoef, and wheelRrCoef,
    calculate and return ABCs.

    Arguments:
    ----------
    veh_kg: vehicle mass [kg]
    veh_fa_m2: vehicle frontal area [m^2]
    show_plots: if True, plots are shown

    Returns:
    a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]

    It may be worthwhile to have this use get_rho_air() in the future.
    """

    speed_mph = np.linspace(0, 70, 500)

    def model_func_lb(speed_mps, drag_coef, wheel_rr_coef):
        """fastsim-style solution for drag force on vehicle.
        Arguments:
        ---------
        speed: array of vehicle speeds [mps]
        dragCoef: drag coefficient [-]
        wheelRrCoef: rolling resistance coefficient [-]
        """

        out = (veh_kg * props.a_grav_mps2 * wheel_rr_coef +
               0.5 * props.air_density_kg_per_m3 * drag_coef * veh_fa_m2
               * speed_mps ** 2) / 4.448
        return out

    model_lb = model_func_lb(
        speed_mph / params.MPH_PER_MPS, drag_coef, wheel_rr_coef)

    # polynomial function for pounds vs speed
    def dyno_func_lb(speed_mph, a, b, c): return np.poly1d(
        [c, b, a])(speed_mph)

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


def l__100km_to_mpg(l__100km):
    """Given fuel economy in L/100km, returns mpg."""

    mpg = 1 / (l__100km / 3.785 / 100 / 1_000 * params.M_PER_MI)

    return mpg


def mpg_to_l__100km(mpg):
    """Given fuel economy in mpg, returns L/100km."""

    l__100km = 1 / (mpg / 3.785 * params.M_PER_MI / 1_000 / 100)

    return l__100km


def rollav(x, y, width=10):
    """
    Returns x-weighted backward-looking rolling average of y.  
    Good for resampling data that needs to preserve cumulative information.
    Arguments:
    ----------
    x : x data
    y : y data (`len(y) == len(x)` must be True)
    width: rolling average width
    """

    assert(len(x) == len(y))

    dx = np.concatenate([0, x.diff()])

    yroll = np.zeros(len(x))
    yroll[0] = y[0]

    for i in range(1, len(x)):
        if i < width:
            yroll[i] = (
                dx[:i] * y[:i]).sum() / (x[i] - x[0])
        else:
            yroll[i] = (
                dx[i-width:i] * y[i-width:i]).sum() / (
                    x[i] - x[i-width])
    return yroll


def camel_to_snake(name):
    "Given camelCase, returns snake_case."
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
