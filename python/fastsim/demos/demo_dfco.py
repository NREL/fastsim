"""Demonstrate using and activating decel fuel cut-off (DFCO)."""

import os
import time
from pathlib import Path

# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import fastsim as fsim
from fastsim.demos.plot_utils import (
    figsize_3_stacked,
    get_paired_cycler,
)

sns.set_theme()

# Plot Related Data
baselinestyles = [
    "--",
    "-.",
]

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

METERS_PER_MILE = 1609.34
MJ_PER_GGE = 125.0

# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%

# load 2026 Chrysler Pacifica Select -- No DFCO
veh = fsim.Vehicle.from_resource("2026_Chrysler_Pacifica_Select.yaml")
veh.set_dfco_params(enabled=False, min_dfco_speed_m_per_s=0.0, max_accel_for_dfco_m_per_s2=0.0)
veh.set_save_interval(1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# Instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)
t0 = time.perf_counter()
sd.walk()
t1 = time.perf_counter()
dt_fsim3_conv = t1 - t0
print(f"NORMAL: fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{dt_fsim3_conv} s")
df = sd.to_dataframe()

# load 2026 Chrysler Pacifica Select -- Include DFCO
veh_dfco = fsim.Vehicle.from_resource("2026_Chrysler_Pacifica_Select.yaml")
veh_dfco.set_dfco_params(
    enabled=True,
    min_dfco_speed_m_per_s=11.176,
    max_accel_for_dfco_m_per_s2=-0.2,
)
veh_dfco.set_save_interval(1)

# load cycle from file
cyc_dfco = fsim.Cycle.from_resource("udds.csv")

# Instantiate `SimDrive` simulation object
sd_dfco = fsim.SimDrive(veh_dfco, cyc_dfco)
t0 = time.perf_counter()
sd_dfco.walk()
t1 = time.perf_counter()
dt_fsim3_conv_dfco = t1 - t0
print(
    "DFCO: fastsim-3 `sd.walk()` elapsed time with `save_interval` "
    + f"of 1:\n{dt_fsim3_conv_dfco} s",
)
df_dfco = sd_dfco.to_dataframe()

# Determine miles per gallon
cyc_dict = cyc.to_pydict()
distance_m = cyc_dict["dist_meters"][-1]
distance_mi = distance_m / METERS_PER_MILE
fuel_mj = df["veh.pt_type.Conv.fc.history.energy_fuel_joules"][-1] / 1e6
fuel_dfco_mj = df_dfco["veh.pt_type.Conv.fc.history.energy_fuel_joules"][-1] / 1e6
gge_gal = fuel_mj / MJ_PER_GGE
gge_dfco_gal = fuel_dfco_mj / MJ_PER_GGE
fuel_economy_mpg = distance_mi / gge_gal
fuel_economy_dfco_mpg = distance_mi / gge_dfco_gal

percent_reduction = (fuel_mj - fuel_dfco_mj) * 100.0 / fuel_mj

print(f"Conventional Vehicle Fuel Economy: {fuel_economy_mpg} mpg")
print(f"Conventional w/ DFCO             : {fuel_economy_dfco_mpg} mpg")
print(f"DFCO Reduction in Fuel Use (Conv): {percent_reduction} %")


def plot_fc_pwr(df: pd.DataFrame, df_dfco: pd.DataFrame) -> tuple[Figure, Axes]:
    """Plot fuel converter powers."""
    num_subplots = 3
    fig, ax = plt.subplots(num_subplots, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")
    tag = "Conv"

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.Conv.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.Conv.fc.history.pwr_aux_watts"]
        )
        / 1e3,
        label="f3 shaft",
    )
    ax[0].plot(
        df_dfco["cyc.time_seconds"],
        (
            df_dfco[f"veh.pt_type.{tag}.fc.history.pwr_prop_watts"]
            + df_dfco[f"veh.pt_type.{tag}.fc.history.pwr_aux_watts"]
        )
        / 1e3,
        label="f3 shaft (dfco)",
    )
    ax[0].set_ylabel("FC Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.Conv.fc.history.pwr_fuel_watts"] / 1e3,
        label="f3 fuel",
    )
    ax[1].plot(
        df_dfco["cyc.time_seconds"],
        df_dfco[f"veh.pt_type.{tag}.fc.history.pwr_fuel_watts"] / 1e3,
        label="f3 fuel (dfco)",
    )
    ax[1].set_ylabel("FC Power [kW]")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[2].plot(
        df_dfco["cyc.time_seconds"],
        df_dfco["veh.history.speed_ach_meters_per_second"],
        label="f3 (dfco)",
    )
    ax[2].legend()
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[2].get_xlim()[0], ax[2].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[2].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/fc_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


fig, ax = plot_fc_pwr(df, df_dfco)
