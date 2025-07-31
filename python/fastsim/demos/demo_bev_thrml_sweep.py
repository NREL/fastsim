"""BEV thermal demo with cold start and cold ambient conditions."""

# %%
import argparse
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

sns.set_theme()

cyc_key = "cycle"
te_amb_key = "te_amb [*C]"
te_init_key = "te_init [*C]"
ecr_key = "ECR [kW-hr/100mi]"
udds = "udds"
hwfet = "hwfet"

celsius_to_kelvin = 273.15
temp_amb_and_init = -6.7 + celsius_to_kelvin
mph_per_mps = 2.24

te_amb_sweep_size = 50
te_init_sweep_size = te_amb_sweep_size


def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


def setup_sweep() -> pd.DataFrame:
    """Set up sweep of ambient and initial conditions"""
    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)

    # full factorial of results
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

    df_res = pd.DataFrame(res_list)

    return df_res


def sweep(df: pd.DataFrame, n_proc: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep ambient and initial conditions

    # Arguments:
    # - `df`: dataframe of DOE
    # - `n_proc`: number of parallel processes

    """
    # full factorial of results

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
            # if hot ambient, init temp must be at or above reasonable HVAC setpoint
            ((te_init_k - celsius_to_kelvin) >= 17.0) & ((te_amb_k + 5) >= te_init_k)
            |
            # if cold ambient, init temp must be at or above reasonable HVAC setpoint
            ((te_init_k - celsius_to_kelvin) <= 27.0) & ((te_amb_k - 5) <= te_init_k)
        )
        if feasible:
            res_list_feasible.append(row)

    df_feasible = pd.DataFrame(res_list_feasible)

    return df_res, df_feasible


def plot_sweep_cross_effects(
    df: pd.DataFrame,
    cyc: str,
    x_var: str,
    par_var_sweep: list[float],
    show_plots: bool = False,
    save_plots: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
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
    title_str = cyc.upper() + f"\\deltaECR / \\delta{x_var}"
    fig.suptitle(
        title_str,
    )
    for par_var_val in par_var_sweep:
        df_fltrd = df[(df[par_var] == par_var_val) & (df[cyc_key] == cyc)]
        df_feas_fltrd = df_feasible[
            (df_feasible[par_var] == par_var_val) & (df_feasible[cyc_key] == cyc)
        ]
        d_ecr_d_x_var = np.diff(df_fltrd[ecr_key]) / np.diff(df_fltrd[x_var])
        d_ecr_d_x_var_feas = np.diff(df_feas_fltrd[ecr_key]) / np.diff(df_feas_fltrd[x_var])
        line = ax.plot(
            df_feas_fltrd[x_var],
            d_ecr_d_x_var_feas,
            label=f"{par_var_val:.1f}",
        )[0]
        ax.plot(
            df_fltrd[x_var],
            d_ecr_d_x_var,
            color=line.get_color(),
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            df_feas_fltrd[x_var],
            d_ecr_d_x_var,
            marker=".",
            color=line.get_color(),
            linestyle=None,
        )
    ax.set_xlabel(var_to_title[x_var] + "Temp. [*C]")
    ax.set_ylabel(f"\\deltaECR [kW-hr/100mi] / \\delta{x_var}")
    ax.legend(title=par_var)
    plt.tight_layout()

    if save_plots:
        fig.savefig(
            Path(__file__).parent / (title_str + ".svg"),
        )

    if show_plots:
        plt.show()

    return fig, ax


def solve_row(iterrow: tuple[Hashable, pd.Series]) -> dict[str, Any]:
    """Solve row of dataframe and return result"""
    row = iterrow[1]
    if cast(int, iterrow[0]) % 250 == 0:
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

    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)
    veh_dict = veh.to_pydict()

    # setup initial conditions
    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
        "temperature_kelvin"
    ] = te_init_k
    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"]["temp_prev_kelvin"] = (
        te_init_k
    )
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
        ecr_key: veh_dict_solved["pt_type"]["BEV"]["res"]["state"]["energy_out_chemical_joules"]
        / 1_000
        / 3_600
        / (veh_dict_solved["state"]["dist_meters"] / 1e3 / 1.61)
        * 100.0,
        "sd": sd.to_pydict(),
    }

    return new_row


def plot_time_series(
    df: pd.DataFrame,
    verbose: bool = False,
    show_plots: bool = False,
    save_plots: bool = False,
) -> None:
    """Plot time series temperature data"""
    for i, row in df.iterrows():
        if verbose:
            print(row)
        sd = row["sd"]
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            sd["cyc"]["time_seconds"],
            np.array(sd["veh"]["cabin"]["LumpedCabin"]["history"]["temperature_kelvin"])
            - celsius_to_kelvin,
            label="cabin",
        )
        ax[0].plot(
            sd["cyc"]["time_seconds"],
            np.array(
                sd["veh"]["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["history"][
                    "temperature_kelvin"
                ],
            )
            - celsius_to_kelvin,
            label="battery",
        )
        ax[0].plot(
            sd["cyc"]["time_seconds"],
            np.array(sd["cyc"]["temp_amb_air_kelvin"]) - celsius_to_kelvin,
            label="ambient",
        )
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel("Temp. [°C]")
        ax[0].legend()

        ax[1].plot(
            sd["veh"]["history"]["time_seconds"],
            np.array(sd["veh"]["pt_type"]["BEV"]["res"]["history"]["soc"]),
        )
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Batt. SOC")

        ax[-1].plot(
            sd["veh"]["history"]["time_seconds"],
            np.array(sd["veh"]["history"]["speed_ach_meters_per_second"]) * mph_per_mps,
        )
        ax[-1].set_xlabel("Time [s]")
        ax[-1].set_ylabel("Speed [mph]")
        plt.tight_layout()

        if save_plots:
            save_str = (
                f"cyc - {row[cyc_key]}, te_init - {row[te_init_key]}, te_amb - {row[te_amb_key]}"
            )
            fig.savefig(
                Path(__file__).parent / f"{save_str}.svg",
            )

        if show_plots:
            plt.show()
        plt.close()


def plot_sweep(
    df: pd.DataFrame,
    cyc: str,
    x_var: str,
    par_var_sweep: list[float],
    show_plots: bool = False,
    save_plots: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
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
        df_fltrd = df[(df[par_var] == par_var_val) & (df[cyc_key] == cyc)]
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
    ax.set_ylabel("ECR [kW-hr/100mi]")
    ax.legend(title=par_var)
    plt.tight_layout()

    if save_plots:
        fig.savefig(
            Path(__file__).parent
            / (cyc.upper() + f" ECR v. {var_to_title[x_var]} and {var_to_title[par_var]} Temp.svg"),
        )

    if show_plots:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sweep of ambient and initial temperatures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("n_proc", type=int, default=4, help="Number of parallel processes.")

    # if environment var `SHOW_PLOTS=false` is set, no plots are shown
    SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
    # if environment var `SAVE_FIGS=true` is set, save plots
    SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

    # array of ambient temperatures in kelvin
    te_amb_arr_k: list[float] = [
        t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, te_amb_sweep_size)
    ]
    # array of init temperatures in kelvin
    te_batt_and_cab_init_arr_k: list[float] = [
        t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, te_init_sweep_size)
    ]

    args = parser.parse_args()
    n_proc: int = args.n_proc  # type: ignore[attr-defined]

    print(f"\nRunning sweep with {n_proc} parallel processes")
    df_doe = setup_sweep()
    df_res, df_feasible = sweep(df_doe, n_proc)

    # plot ECR v. init for a sweep of amb
    te_amb_step = int(len(te_amb_arr_k) / 10) if len(te_amb_arr_k) > 10 else 1
    te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

    te_init_step = (
        int(len(te_batt_and_cab_init_arr_k) / 10) if len(te_batt_and_cab_init_arr_k) > 10 else 1
    )
    te_init_short_deg_c = [
        te_init_k - celsius_to_kelvin for te_init_k in te_batt_and_cab_init_arr_k
    ][::te_init_step]

    print("\nPlotting sweep results")
    fig0, ax0 = plot_sweep(df_res, udds, te_init_key, te_amb_short_deg_c)
    fig1, ax1 = plot_sweep(df_res, udds, te_amb_key, te_init_short_deg_c)
    fig2, ax2 = plot_sweep(df_res, hwfet, te_init_key, te_amb_short_deg_c)
    fig3, ax3 = plot_sweep(df_res, hwfet, te_amb_key, te_init_short_deg_c)

    # print("Plotting time series")
    # plot_time_series(df_res)

    # perform some calculations
    fixed_init_udds_ecr = df_res[
        ((df_res[te_init_key] > 22.0) & (df_res[te_init_key] < 24.0)) & (df_res[cyc_key] == udds)
    ][ecr_key]
    ecr_delta_fixed_init_udds = (
        fixed_init_udds_ecr.max() - fixed_init_udds_ecr.min()
    ) / fixed_init_udds_ecr.min()
    print(
        "Percent increase beween lowwest and highest ECR for udds and fixed initial temp:"
        + f" {ecr_delta_fixed_init_udds:.5%}",
    )

    fixed_amb_udds_ecr = df_res[
        ((df_res[te_amb_key] > 22.0) & (df_res[te_amb_key] < 24.0)) & (df_res[cyc_key] == udds)
    ][ecr_key]
    ecr_delta_fixed_amb_udds = (
        fixed_amb_udds_ecr.max() - fixed_amb_udds_ecr.min()
    ) / fixed_amb_udds_ecr.min()
    print(
        "Percent increase beween lowwest and highest ECR for udds and fixed ambient temp:"
        + f" {ecr_delta_fixed_amb_udds:.5%}",
    )

    fixed_init_hwfet_ecr = df_res[
        ((df_res[te_init_key] > 22.0) & (df_res[te_init_key] < 24.0)) & (df_res[cyc_key] == hwfet)
    ][ecr_key]
    ecr_delta_fixed_init_hwfet = (
        fixed_init_hwfet_ecr.max() - fixed_init_hwfet_ecr.min()
    ) / fixed_init_hwfet_ecr.min()
    print(
        "Percent increase beween lowwest and highest ECR for hwfet and fixed initial temp:"
        + f" {ecr_delta_fixed_init_hwfet:.5%}",
    )

    fixed_amb_hwfet_ecr = df_res[
        ((df_res[te_amb_key] > 22.0) & (df_res[te_amb_key] < 24.0)) & (df_res[cyc_key] == hwfet)
    ][ecr_key]
    ecr_delta_fixed_amb_hwfet = (
        fixed_amb_hwfet_ecr.max() - fixed_amb_hwfet_ecr.min()
    ) / fixed_amb_hwfet_ecr.min()
    print(
        "Percent increase beween lowwest and highest ECR for hwfet and fixed ambient temp:"
        + f" {ecr_delta_fixed_amb_hwfet:.5%}",
    )
