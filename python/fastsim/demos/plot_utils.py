from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import fastsim as fsim

figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
BASE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
BASE_LINE_STYLES = ["--", "-.", ":"]

COLOR = "color"
LINESTYLE = "linestyle"
DEF_PAIR_ATTR = COLOR


def get_paired_cycler(pair_attr: str = DEF_PAIR_ATTR):
    """
    # Arguments:
    - `pair_attr`: whether the paired lines should match in `"color"` or `"linestyle"`
    """
    assert (pair_attr == COLOR) or (pair_attr == LINESTYLE)
    # construct array of repeated
    if pair_attr == LINESTYLE:
        series_list = BASE_COLORS
    else:
        series_list = BASE_LINE_STYLES
    series = [[c, c] for c in series_list]
    series = [x for sublist in series for x in sublist]

    if pair_attr == LINESTYLE:
        pairs = (BASE_LINE_STYLES[:2] * int(np.ceil(len(series) / 2)))[: len(series)]
    else:
        pairs = (BASE_COLORS[:2] * int(np.ceil(len(series) / 2)))[: len(series)]

    paired_cycler = cycler(color=pairs if pair_attr == COLOR else series) + cycler(
        linestyle=pairs if pair_attr == LINESTYLE else series,
    )
    return paired_cycler


def get_uni_cycler(pair_attr: str = DEF_PAIR_ATTR):
    """
    # Arguments:
    - `pair_attr`: ensures consistent behavior with `get_paired_cycler`
    """
    assert (pair_attr == COLOR) or (pair_attr == LINESTYLE)
    if pair_attr == COLOR:
        colors = BASE_COLORS
        linestyles = ["--"] * len(colors)
    else:
        linestyles = BASE_LINE_STYLES
        colors = [BASE_COLORS[0]] * len(linestyles)
    uni_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    return uni_cycler


def plot_bev_temperatures(
    df: pd.DataFrame, save_figs: bool, show_plots: bool
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Component Temperatures")

    ax[0].set_prop_cycle(get_uni_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.cabin.LumpedCabin.history.temperature_kelvin"] - 273.15,
        label="cabin",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df[
            "veh.pt_type.BatteryElectricVehicle.res.thrml."
            + "RESLumpedThermal.history.temperature_kelvin"
        ]
        - 273.15,
        label="res",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["cyc.temp_amb_air_kelvin"] - 273.15,
        label="amb",
    )
    ax[0].set_ylabel("Temperatures [°C]")
    ax[0].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if save_figs:
        plt.savefig(Path("./plots/temps.svg"))
    if show_plots:
        plt.show()

    return fig, ax


def plot_bev_hvac_pwr(df: pd.DataFrame, save_figs: bool, show_plots: bool) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Thermal Management Power Demand")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.hvac.LumpedCabinAndRES.history.pwr_thrml_hvac_to_cabin_watts"],
        label="hvac thrml to cabin",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.hvac.LumpedCabinAndRES.history.pwr_aux_for_cab_hvac_watts"],
        label="hvac aux for cabin",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.hvac.LumpedCabinAndRES.history.pwr_thrml_hvac_to_res_watts"],
        label="hvac thrml to battery",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.hvac.LumpedCabinAndRES.history.pwr_aux_for_res_hvac_watts"],
        label="hvac aux for battery",
    )
    ax[0].set_ylabel("Power [W]")
    ax[0].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if save_figs:
        plt.savefig(Path("./plots/hvac pwr.svg"))
    if show_plots:
        plt.show()

    return fig, ax


def plot_bev_res_pwr(df: pd.DataFrame, save_figs: bool, show_plots: bool) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"],
        label="soc",
    )
    ax[1].set_ylabel("SOC")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["cyc.speed_meters_per_second"],
        label="cyc",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()
    if save_figs:
        plt.savefig(Path("./plots/res_pwr.svg"))
    if show_plots:
        plt.show()

    return fig, ax


def plot_bev_res_energy(df: pd.DataFrame, save_figs: bool, show_plots: bool) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Energy [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"],
        label="soc",
    )
    ax[1].set_ylabel("SOC")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["cyc.speed_meters_per_second"],
        label="cyc",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()
    if save_figs:
        plt.savefig(Path("./plots/res_energy.svg"))
    if show_plots:
        plt.show()

    return fig, ax


def plot_road_loads(
    df: pd.DataFrame,
    veh: fsim.Vehicle,
    save_figs: bool,
    show_plots: bool,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_uni_cycler())
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_drag_watts"] / 1e3,
        label="drag",
    )
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_rr_watts"] / 1e3,
        label="rr",
    )
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_tractive_watts"] / 1e3,
        label="total",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach. Speed [m/s]")

    plt.tight_layout()
    if save_figs:
        plt.savefig(Path("./plots/road_loads.svg"))
    if show_plots:
        plt.show()

    return fig, ax
