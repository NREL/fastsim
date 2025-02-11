# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path
import time
import json
import os
from typing import Tuple
import fastsim as fsim

sns.set_theme()

from plot_utils  import *

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"     
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

# `fastsim3` -- load vehicle and cycle, build simulation, and run 
# %%

# load 2020 Chevrolet Bolt BEV from file
veh = fsim.Vehicle.from_file(
    fsim.package_root() / "../../cal_and_val/thermal/f3-vehicles/2020 Chevrolet Bolt EV.yaml"
)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s")
df = sd.to_dataframe()

# # Visualize results

def plot_res_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df['cyc.time_seconds'],
        df["veh.pt_type.BatteryElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df['cyc.time_seconds'],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"],
        label="soc",
    )
    ax[1].set_ylabel("SOC")
    ax[1].legend()
    
    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df['cyc.time_seconds'],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].plot(
        df['cyc.time_seconds'],
        df["cyc.speed_meters_per_second"],
        label="cyc",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax

def plot_res_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df['cyc.time_seconds'],
        df["veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Energy [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df['cyc.time_seconds'],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"],
        label="soc",
    )
    ax[1].set_ylabel("SOC")
    ax[1].legend()
    
    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df['cyc.time_seconds'],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].plot(
        df['cyc.time_seconds'],
        df["cyc.speed_meters_per_second"],
        label="cyc",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax

def plot_road_loads() -> Tuple[Figure, Axes]: 
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"][::veh.save_interval],
        df["veh.history.pwr_drag_watts"] / 1e3,
        label="drag",
    )
    ax[0].plot(
        df["cyc.time_seconds"][::veh.save_interval],
        df["veh.history.pwr_rr_watts"] / 1e3,
        label="rr",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"][::veh.save_interval],
        df["veh.history.speed_ach_meters_per_second"],
        label="ach",
    )
    ax[-1].plot(
        df['cyc.time_seconds'],
        df["cyc.speed_meters_per_second"],
        label="cyc",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach. Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/road_loads.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax

fig, ax = plot_res_pwr() 
fig, ax = plot_res_energy()
fig, ax = plot_road_loads()

# %%
# example for how to use set_default_pwr_interp() method for veh.res
res = fsim.ReversibleEnergyStorage.from_pydict(sd.to_pydict()['veh']['pt_type']['BatteryElectricVehicle']['res'])
res.set_default_pwr_interp()

# %%
