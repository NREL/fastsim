"""
---
execute:
  skip: true
---

# BEV Thermal Sweep

This demo sweeps ambient and initial temperatures for a thermal
Battery Electric Vehicle (BEV) across Urban Dynamometer Driving
Schedule (UDDS) and Highway Fuel Economy Test (HWFET) cycles,
computing Energy Consumption Rate (ECR, kW-hr/100mi) for each
combination.
"""

# %%
import os
from collections.abc import Hashable
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import fastsim as fsim

# %%
sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

"""
## Constants and Sweep Configuration

The sweep varies ambient temperature from -7 to 40 C and initial
cabin/battery temperature from -7 to 45 C. Each combination is
simulated over both UDDS and HWFET cycles.
"""

# %%
cyc_key = "cycle"
te_amb_key = "te_amb [*C]"
te_init_key = "te_init [*C]"
ecr_key = "ECR [kW-hr/100mi]"
udds = "udds"
hwfet = "hwfet"

celsius_to_kelvin = 273.15
mph_per_mps = 2.24

sweep_size = 10
n_proc = 4

te_amb_arr_k: list[float] = [
    t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, sweep_size)
]
te_batt_and_cab_init_arr_k: list[float] = [
    t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, sweep_size)
]

"""
## Helper Functions
"""


# %%
def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


def solve_row(iterrow: tuple[Hashable, pd.Series]) -> dict[str, Any]:
    """Solve a single row of the DOE dataframe"""
    row = iterrow[1]
    if cast(int, iterrow[0]) % 500 == 0:
        print()
        print(fsim.utils.utilities.print_dt())
        print(row)
    cyc_str = row[cyc_key]
    te_amb_k = row[te_amb_key] + celsius_to_kelvin
    te_init_k = row[te_init_key] + celsius_to_kelvin
    cyc = fsim.Cycle.from_resource(cyc_str + ".csv")
    cyc_dict = cyc.to_pydict()
    cyc_dict["temp_amb_air_kelvin"] = [te_amb_k] * cyc.len()
    cyc = fsim.Cycle.from_pydict(cyc_dict)

    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)
    veh_dict = veh.to_pydict()

    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
        "temperature_kelvin"
    ] = te_init_k
    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"]["temp_prev_kelvin"] = (
        te_init_k
    )
    veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = te_init_k
    veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = te_init_k

    veh = fsim.Vehicle.from_pydict(veh_dict)
    sd = fsim.SimDrive(veh, cyc, None)
    try_walk(sd, f"`sd_prep`, te_amb: {te_amb_k}, te_init: {te_init_k}")
    veh_dict_solved = sd.to_pydict()["veh"]

    new_row = {
        cyc_key: cyc_str,
        te_amb_key: te_amb_k - celsius_to_kelvin,
        te_init_key: te_init_k - celsius_to_kelvin,
        ecr_key: veh_dict_solved["pt_type"]["BEV"]["res"]["state"]["energy_out_chemical_joules"]
        / 1_000
        / 3_600
        / (veh_dict_solved["state"]["dist_meters"] / 1e3 / 1.61)
        * 100.0,
        "sd": sd.to_pydict(),
    }

    return new_row


"""
## Build and Run Sweep

Build a full factorial Design of Experiments (DOE) across cycles, ambient temperatures, and
initial temperatures, then solve each combination. Results are filtered
to feasible combinations based on ambient/initial temperature proximity.
"""


# %%
def setup_sweep() -> pd.DataFrame:
    """Set up full factorial of ambient and initial conditions"""
    res_list = []
    for cyc_str in [udds, hwfet]:
        for te_amb_k in te_amb_arr_k:
            for te_init_k in te_batt_and_cab_init_arr_k:
                new_row = {
                    cyc_key: cyc_str,
                    te_amb_key: te_amb_k - celsius_to_kelvin,
                    te_init_key: te_init_k - celsius_to_kelvin,
                }
                res_list.append(new_row)

    return pd.DataFrame(res_list)


def sweep(df: pd.DataFrame, n_proc: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep ambient and initial conditions, returning all and feasible results"""
    if n_proc is not None:
        with Pool(n_proc) as pool:
            res_list = pool.map(solve_row, df.iterrows())
    else:
        res_list = [solve_row(row) for row in df.iterrows()]

    df_res = pd.DataFrame(res_list)

    res_list_feasible = []
    for i, row in df_res.iterrows():
        te_amb_k = row[te_amb_key] + celsius_to_kelvin
        te_init_k = row[te_init_key] + celsius_to_kelvin
        feasible = (
            ((te_init_k - celsius_to_kelvin) >= 17.0) & ((te_amb_k + 5) >= te_init_k)
            |
            ((te_init_k - celsius_to_kelvin) <= 27.0) & ((te_amb_k - 5) <= te_init_k)
        )
        if feasible:
            res_list_feasible.append(row)

    df_feasible = pd.DataFrame(res_list_feasible)

    return df_res, df_feasible


"""
## Plot ECR Sweep Results

ECR plotted against ambient and initial temperatures for UDDS and HWFET
cycles. Solid lines show feasible combinations, dashed lines show all
combinations.
"""


# %%
def plot_sweep(
    df: pd.DataFrame,
    df_feas: pd.DataFrame,
    cyc: str,
    x_var: str,
    par_var_sweep: list[float],
    show_plots: bool = False,
    save_figs: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot ECR sweep, parametric style"""
    par_var = te_init_key if x_var == te_amb_key else te_amb_key
    var_to_title = {te_amb_key: "Amb.", te_init_key: "Init."}

    fig, ax = plt.subplots()
    if not (show_plots) and not (save_figs):
        return (fig, ax)
    title_str = cyc.upper() + f" ECR v. {var_to_title[x_var]} and {var_to_title[par_var]} Temp."
    fig.suptitle(title_str)
    for par_var_val in par_var_sweep:
        df_fltrd = df[(df[par_var] == par_var_val) & (df[cyc_key] == cyc)]
        df_feas_fltrd = df_feas[(df_feas[par_var] == par_var_val) & (df_feas[cyc_key] == cyc)]
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
    ax.set_ylabel("ECR [kW-hr/100mi]")
    ax.legend(title=par_var)
    plt.tight_layout()

    if save_figs:
        fig.savefig(Path(__file__).parent / (title_str + ".svg"))
    if show_plots:
        plt.show()

    return fig, ax


"""
## Plot Cross Effects

Change in ECR per change in temperature (dECR/dT) for UDDS and HWFET
cycles.
"""


# %%
def plot_sweep_cross_effects(
    df: pd.DataFrame,
    df_feas: pd.DataFrame,
    cyc: str,
    x_var: str,
    par_var_sweep: list[float],
    show_plots: bool = False,
    save_figs: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot ECR sensitivity to temperature changes"""
    par_var = te_init_key if x_var == te_amb_key else te_amb_key
    var_to_title = {te_amb_key: "Amb.", te_init_key: "Init."}

    fig, ax = plt.subplots()
    if not (show_plots) and not (save_figs):
        return (fig, ax)
    title_str = cyc.upper() + f" dECR per d{var_to_title[x_var]}"
    fig.suptitle(title_str)
    for par_var_val in par_var_sweep:
        df_fltrd = df[(df[par_var] == par_var_val) & (df[cyc_key] == cyc)]
        df_feas_fltrd = df_feas[(df_feas[par_var] == par_var_val) & (df_feas[cyc_key] == cyc)]
        d_ecr_d_x_var = np.diff(df_fltrd[ecr_key]) / np.diff(df_fltrd[x_var])
        d_ecr_d_x_var_feas = np.diff(df_feas_fltrd[ecr_key]) / np.diff(df_feas_fltrd[x_var])
        line = ax.plot(
            df_feas_fltrd[x_var][1:],
            d_ecr_d_x_var_feas,
            label=f"{par_var_val:.1f}",
        )[0]
        ax.plot(
            df_fltrd[x_var][1:],
            d_ecr_d_x_var,
            color=line.get_color(),
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            df_feas_fltrd[x_var][1:],
            d_ecr_d_x_var_feas,
            marker=".",
            color=line.get_color(),
            linestyle=None,
        )
    ax.set_xlabel(var_to_title[x_var] + "Temp. [*C]")
    ax.set_ylabel(f"dECR [kW-hr/100mi] / d{var_to_title[x_var]}")
    ax.legend(title=par_var)
    plt.tight_layout()

    if save_figs:
        fig.savefig(Path(__file__).parent / (title_str + ".svg"))
    if show_plots:
        plt.show()

    return fig, ax


"""
## Cross-Effect Deltas

Percent increase in ECR across the sweep range when one temperature
variable is held near 22-24 C.
"""


# %%
def print_cross_delta(df: pd.DataFrame, cycle: str, fixed_var: str) -> None:
    """Print percent increase in ECR when one variable is fixed between 22-24 C"""
    ecr = df[((df[fixed_var] > 22.0) & (df[fixed_var] < 24.0)) & (df[cyc_key] == cycle)][ecr_key]
    ecr_delta = (ecr.max() - ecr.min()) / ecr.min()
    print(
        f"Percent increase between lowest and highest ECR for {cycle} and fixed {fixed_var}:"
        + f" {ecr_delta:.5%}",
    )


# %%
if __name__ == "__main__":
    df_doe = setup_sweep()
    df_res, df_feasible = sweep(df_doe, n_proc)

    te_amb_step = int(len(te_amb_arr_k) / 10) if len(te_amb_arr_k) > 10 else 1
    te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

    te_init_step = (
        int(len(te_batt_and_cab_init_arr_k) / 10) if len(te_batt_and_cab_init_arr_k) > 10 else 1
    )
    te_init_short_deg_c = [
        te_init_k - celsius_to_kelvin for te_init_k in te_batt_and_cab_init_arr_k
    ][::te_init_step]

    plot_sweep(df_res, df_feasible, udds, te_init_key, te_amb_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep(df_res, df_feasible, udds, te_amb_key, te_init_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep(df_res, df_feasible, hwfet, te_init_key, te_amb_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep(df_res, df_feasible, hwfet, te_amb_key, te_init_short_deg_c, SHOW_PLOTS, SAVE_FIGS)

    plot_sweep_cross_effects(
        df_res, df_feasible, udds, te_init_key, te_amb_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep_cross_effects(
        df_res, df_feasible, udds, te_amb_key, te_init_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep_cross_effects(
        df_res, df_feasible, hwfet, te_init_key, te_amb_short_deg_c, SHOW_PLOTS, SAVE_FIGS)
    plot_sweep_cross_effects(
        df_res, df_feasible, hwfet, te_amb_key, te_init_short_deg_c, SHOW_PLOTS, SAVE_FIGS)

    print("Cross-effect deltas w.r.t. full dataframe")
    print_cross_delta(df_res, udds, te_init_key)
    print_cross_delta(df_res, udds, te_amb_key)
    print_cross_delta(df_res, hwfet, te_init_key)
    print_cross_delta(df_res, hwfet, te_amb_key)

    print("\nCross-effect deltas w.r.t. feasible dataframe")
    print_cross_delta(df_feasible, udds, te_init_key)
    print_cross_delta(df_feasible, udds, te_amb_key)
    print_cross_delta(df_feasible, hwfet, te_init_key)
    print_cross_delta(df_feasible, hwfet, te_amb_key)
