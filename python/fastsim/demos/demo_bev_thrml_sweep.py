"""BEV thermal demo with cold start and cold ambient conditions."""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import fastsim as fsim

sns.set_theme()


# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

celsius_to_kelvin = 273.15
temp_amb_and_init = -6.7 + celsius_to_kelvin
# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%


def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


# array of ambient temperatures in kelvin
te_amb_arr_k: list[float] = [t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, 50)]
# array of init temperatures in kelvin
te_batt_and_cab_init_arr_k: list[float] = [
    t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, 50)
]

cyc_key = "cycle"
te_amb_key = "te_amb [*C]"
te_init_key = "te_init [*C]"
ecr_key = "ECR [kW-hr/100mi]"
udds = "udds"
hwfet = "hwfet"


def sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep ambient and initial conditions"""
    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)

    # full factorial of results
    res_list = []
    # results filtered for feasibility
    res_list_feasible = []
    for cyc_str in [udds, hwfet]:
        for te_amb_k in te_amb_arr_k:
            for te_init_k in te_batt_and_cab_init_arr_k:
                cyc = fsim.Cycle.from_resource(cyc_str + ".csv")
                cyc_dict = cyc.to_pydict()
                cyc_dict["temp_amb_air_kelvin"] = [te_amb_k] * cyc.len()
                cyc = fsim.Cycle.from_pydict(cyc_dict)

                # setup initial conditions
                veh_dict = veh.to_pydict()
                veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
                    "temperature_kelvin"
                ] = te_init_k
                veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
                    "temp_prev_kelvin"
                ] = te_init_k
                veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = te_init_k
                veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = te_init_k

                # simulate cycle
                veh = fsim.Vehicle.from_pydict(veh_dict)
                sd = fsim.SimDrive(veh, cyc, None)
                try_walk(sd, f"`sd_prep`, te_amb: {te_amb_k}, te_init: {te_init_k}")
                veh_dict_solved = sd.to_pydict()["veh"]

                new_row = {
                    cyc_key: cyc_str,
                    te_amb_key: te_amb_k - celsius_to_kelvin,
                    te_init_key: te_init_k - celsius_to_kelvin,
                    ecr_key: veh_dict_solved["pt_type"]["BEV"]["res"]["state"][
                        "energy_out_chemical_joules"
                    ]
                    / 1_000
                    / 3_600
                    / (veh_dict_solved["state"]["dist_meters"] / 1e3 / 1.61)
                    * 100.0,
                }
                res_list.append(new_row)
                feasible = (
                    # if hot ambient, init temp must be at or above reasonable HVAC setpoint
                    ((te_init_k - celsius_to_kelvin) >= 17.0) & ((te_amb_k + 5) >= te_init_k)
                    |
                    # if cold ambient, init temp must be at or above reasonable HVAC setpoint
                    ((te_init_k - celsius_to_kelvin) <= 27.0) & ((te_amb_k - 5) <= te_init_k)
                )
                if feasible:
                    res_list_feasible.append(new_row)

    df_res = pd.DataFrame(res_list)
    df_feasible = pd.DataFrame(res_list_feasible)

    return df_res, df_feasible


df_res, df_feasible = sweep()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"

# if environment var `SAVE_PLOTS=true` is set, plots are saved
SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "true").lower() == "true"

# plot ECR v. init for a sweep of amb
te_amb_step = int(len(te_amb_arr_k) / 10)
te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

te_init_step = int(len(te_amb_arr_k) / 10)
te_init_short_deg_c = [te_init_k - celsius_to_kelvin for te_init_k in te_batt_and_cab_init_arr_k][
    ::te_init_step
]


def plot_sweep(cyc: str, x_var: str, par_var_sweep: list[float]) -> tuple[plt.Figure, plt.Axes]:
    """Plot sweep of ambient and initial temperatures, parameteric style"""
    allowed_cycs = ["udds", "hwfet"]
    assert cyc in allowed_cycs
    allowed_x_vars = {
        te_amb_key,
        te_init_key,
    }
    assert x_var in allowed_x_vars
    par_var = te_init_key if x_var == te_amb_key else te_amb_key
    var_to_title = {te_amb_key: "Amb.", te_init_key: "Init."}

    fig, ax = plt.subplots()
    fig.suptitle(
        cyc.upper() + f" ECR v. {var_to_title[x_var]} and {var_to_title[par_var]} Temp.",
    )
    for par_var_val in par_var_sweep:
        df_fltrd = df_res[(df_res[par_var] == par_var_val) & (df_res[cyc_key] == cyc)]
        df_feas_fltrd = df_feasible[
            (df_feasible[par_var] == par_var_val) & (df_feasible[cyc_key] == cyc)
        ]
        line = ax.plot(
            df_feas_fltrd[x_var],
            df_feas_fltrd[ecr_key],
            label=f"{par_var_val:.1f}",
        )[0]
        ax.plot(
            df_fltrd[x_var],
            df_fltrd[ecr_key],
            color=line.get_color(),
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            df_feas_fltrd[x_var],
            df_feas_fltrd[ecr_key],
            marker=".",
            color=line.get_color(),
            linestyle=None,
        )
    ax.set_xlabel(var_to_title[x_var] + "Temp. [*C]")
    ax.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
    ax.legend(title=par_var)
    plt.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(
            Path(__file__).parent
            / (cyc.upper() + f" ECR v. {var_to_title[x_var]} and {var_to_title[par_var]} Temp.svg"),
        )

    if SHOW_PLOTS:
        plt.show()

    return fig, ax


fig0, ax0 = plot_sweep(udds, te_init_key, te_amb_short_deg_c)
fig1, ax1 = plot_sweep(udds, te_amb_key, te_init_short_deg_c)
fig2, ax2 = plot_sweep(hwfet, te_init_key, te_amb_short_deg_c)
fig3, ax3 = plot_sweep(hwfet, te_amb_key, te_init_short_deg_c)
