"""Auxiliary functions that require fastsim and provide faster access FASTSim vehicle properties."""
import fastsim as fsim
from fastsim.vehicle import Vehicle
from fastsim import parameters as params
from scipy.optimize import minimize, curve_fit
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from fastsim.utilities import get_rho_air

props = params.PhysicalProperties()
R_air = 287  # J/(kg*K)
    

def abc_to_drag_coeffs(veh: Vehicle,
                       a_lbf: float, b_lbf__mph: float, c_lbf__mph2: float,
                       custom_rho: bool = False,
                       custom_rho_temp_degC: float = 20.,
                       custom_rho_elevation_m: float = 180.,
                       simdrive_optimize: bool = True,
                       show_plots: bool = False,
                       use_rust=True) -> Tuple[float, float]:
    """For a given vehicle and target A, B, and C
    coefficients; calculate and return drag and rolling resistance
    coefficients.

    Arguments:
    ----------
    veh: vehicle.Vehicle with all parameters correct except for drag and rolling resistance coefficients
    a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    custom_rho: if True, use `fastsim.utilities.get_rho_air()` to calculate the current ambient density
    custom_rho_temp_degC: ambient temperature [degree C] for `get_rho_air()`; 
        will only be used when `custom_rho` is True
    custom_rho_elevation_m: location elevation [degree C] for `get_rho_air()`; 
        will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
    simdrive_optimize: if True, use `SimDrive` to optimize the drag and rolling resistance; 
        otherwise, directly use target A, B, C to calculate the results
    show_plots: if True, plots are shown
    use_rust: if True, use rust implementation of drag coefficient calculation.
    """

    # TODO: allows air density read APIs for whole project; `get_rho_air()` not used for `SimDrive` yet
    cur_ambient_air_density_kg__m3 = get_rho_air(
        custom_rho_temp_degC, custom_rho_elevation_m) if custom_rho else props.air_density_kg_per_m3

    vmax_mph = 70.0

    a_newton = a_lbf * params.N_PER_LBF
    b_newton__mps = b_lbf__mph * params.N_PER_LBF * params.MPH_PER_MPS
    c_newton__mps2 = c_lbf__mph2 * params.N_PER_LBF * \
        params.MPH_PER_MPS * params.MPH_PER_MPS

    cd_len = 300

    cyc = fsim.cycle.Cycle.from_dict({
        'time_s': np.arange(0, cd_len),
        'mps': np.linspace(vmax_mph / params.MPH_PER_MPS, 0, cd_len)
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
            (1000 * (np.array(sd_coast.drag_kw) + np.array(sd_coast.rr_kw)) /
                np.array(sd_coast.mps_ach))[:cutoff],
            (dyno_func_lb(sd_coast.mph_ach) * fsim.params.N_PER_LBF)[:cutoff],
            np.array(cyc.time_s)[:cutoff],
        )

        return err

    if simdrive_optimize:
        res = minimize(get_err, x0=np.array([0.3, 0.01]))
        (drag_coef, wheel_rr_coef) = res.x

        # TODO: Surpress unnecessary excessive warnings on screen

    else:
        drag_coef = c_newton__mps2 / \
            (0.5 * veh.frontal_area_m2 * cur_ambient_air_density_kg__m3)
        wheel_rr_coef = a_newton / veh.veh_kg / props.a_grav_mps2

    veh.drag_coef, veh.wheel_rr_coef = drag_coef, wheel_rr_coef

    sd_coast = fsim.simdrive.RustSimDrive(
        cyc, veh) if use_rust else fsim.simdrive.SimDrive(cyc, veh)
    sd_coast.impose_coast = [True] * len(sd_coast.impose_coast)
    sd_coast.sim_drive()

    cutoff_val = np.where(np.array(sd_coast.mps_ach) < 0.1)[0][0]

    if show_plots:
        plt.figure()
        plt.plot(
            np.array(sd_coast.mph_ach)[:cutoff_val],
            (1000 * (np.array(sd_coast.drag_kw) + np.array(sd_coast.rr_kw)) /
             np.array(sd_coast.mps_ach) / fsim.params.N_PER_LBF)[:cutoff_val],
            label='sim_drive simulated road load')
        plt.plot(np.array(sd_coast.mph_ach)[:cutoff_val], (dyno_func_lb(
            sd_coast.mph_ach))[:cutoff_val], label='ABCs calculated road load')
        plt.legend()
        plt.xlabel('Speed [mph]')
        plt.ylabel('Road Load [lb]')
        plt.title("Simulated vs Calculated Road Load with Speed")
        plt.show()

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(np.array(cyc.time_s)[:cutoff_val],
                   (1000 * (np.array(sd_coast.drag_kw) + np.array(sd_coast.rr_kw)
                            ) / np.array(sd_coast.mps_ach))[:cutoff_val]
                   )
        ax[0].set_ylabel("Road Load [N]")

        ax[-1].plot(np.array(cyc.time_s)[:cutoff_val],
                    np.array(sd_coast.mph_ach)[:cutoff_val])
        ax[-1].set_ylabel("mph")
        ax[-1].set_xlabel('Time [s]')
        plt.show()
    return drag_coef, wheel_rr_coef


def drag_coeffs_to_abc(veh,
                       custom_rho: bool = False,
                       custom_rho_temp_degC: float = 20.,
                       custom_rho_elevation_m: float = 180.,
                       fit_with_curve: bool = False,
                       show_plots: bool = False) -> Tuple[float, float, float]:
    """For a given vehicle mass, frontal area, dragCoef, and wheelRrCoef,
    calculate and return ABCs.

    Arguments:
    ----------
    veh: vehicle.Vehicle with correct drag and rolling resistance
    custom_rho: if True, use `fastsim.utilities.get_rho_air()` to calculate the current ambient density
    custom_rho_temp_degC: ambient temperature [degree C] for `get_rho_air()`; will only be used when `custom_rho` is True
    custom_rho_elevation_m: location elevation [degree C] for `get_rho_air()`; will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
    fit_with_curve: if True, use `scipy.curve_fit` to get A, B, Cs; otherwise, directly calculate A, B, Cs from given drag and rolling resistance
    show_plots: if True, plots are shown

    Returns:
    a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    """
    # TODO: allows air density read APIs for whole project; `get_rho_air()` not used for `SimDrive` yet
    cur_ambient_air_density_kg__m3 = get_rho_air(
        custom_rho_temp_degC, custom_rho_elevation_m) if custom_rho else props.air_density_kg_per_m3

    vmax_mph = 70.0

    speed_mph = np.linspace(0, vmax_mph, 500)
    veh_kg = veh.veh_kg
    veh_fa_m2 = veh.frontal_area_m2
    drag_coef = veh.drag_coef
    wheel_rr_coef = veh.wheel_rr_coef

    c_newton__mps2 = drag_coef * \
        (0.5 * veh_fa_m2 * cur_ambient_air_density_kg__m3)
    b_newton__mps = 0.0
    a_newton = wheel_rr_coef * veh_kg * props.a_grav_mps2

    def model_func_lb(speed_mps, drag_coef, wheel_rr_coef):
        """fastsim-style solution for drag force on vehicle.
        Arguments:
        ---------
        speed: array of vehicle speeds [mps]
        dragCoef: drag coefficient [-]
        wheelRrCoef: rolling resistance coefficient [-]
        """
        out = (veh_kg * props.a_grav_mps2 * wheel_rr_coef +
               0.5 * cur_ambient_air_density_kg__m3 * drag_coef * veh_fa_m2
               * speed_mps ** 2) / 4.448
        return out

    model_lb = model_func_lb(
        speed_mph / params.MPH_PER_MPS, drag_coef, wheel_rr_coef)

    # polynomial function for pounds vs speed
    def dyno_func_lb(speed_mph, a, b, c):
        return np.poly1d([c, b, a])(speed_mph)
    if fit_with_curve:
        (a_lbf, b_lbf__mph, c_lbf__mph2), pcov = curve_fit(dyno_func_lb,
                                                           xdata=speed_mph,
                                                           ydata=model_lb,
                                                           p0=[10, 0.1, 0.01])
    else:
        a_lbf = a_newton / params.N_PER_LBF
        b_lbf__mph = b_newton__mps / params.N_PER_LBF / params.MPH_PER_MPS
        c_lbf__mph2 = c_newton__mps2 / params.N_PER_LBF / \
            params.MPH_PER_MPS / params.MPH_PER_MPS
    dyno_lb = dyno_func_lb(speed_mph, a_lbf, b_lbf__mph, c_lbf__mph2)

    if show_plots:
        plt.figure()
        plt.plot(speed_mph, dyno_lb, label='dyno')
        plt.plot(speed_mph, model_lb, label='model', linestyle='--')
        plt.legend()
        plt.xlabel('Speed [mph]')
        plt.ylabel('Road Load [lb]')
        plt.show()

    return a_lbf, b_lbf__mph, c_lbf__mph2


def set_nested_values(nested_struct, **kwargs):
    nested_struct.reset_orphaned()
    for key, value in kwargs.items():
        setattr(nested_struct, key, value)
    return nested_struct
