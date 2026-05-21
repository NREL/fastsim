"""
---
execute:
  skip: true
---

# Battery Electric Vehicle Demo

TODO: overview of what this demo covers
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import fastsim as fsim

# %%
sns.set_theme()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

"""
## Setup and Simulation

TODO: describe loading vehicle and cycle, configuring save_interval, running simulation
"""

# %%
# load 2022 Renault Zoe from file
veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# %%
# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)
sd.walk()

df = sd.to_dataframe()
sd_dict = sd.to_pydict(flatten=True)

"""
## Visualize Results

TODO: describe what these plots show for a battery electric vehicle
"""


# %%
def plot_res_pwr():
    """Plot reversible energy storage powers"""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    plt.suptitle("Reversible Energy Storage Power")

    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BEV.res.history.pwr_out_electrical_watts"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Power [kW]")
    ax[0].legend()

    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BEV.res.history.soc"],
    )
    ax[1].set_ylabel("SOC")

    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
    )
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Ach Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


"""
TODO: describe reversible energy storage power plot
"""

# %%
fig, ax = plot_res_pwr()


# %%
def plot_res_energy():
    """Plot reversible energy storage energies"""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    plt.suptitle("Reversible Energy Storage Energy")

    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BEV.res.history.energy_out_electrical_joules"] / 1e6,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Energy [MJ]")
    ax[0].legend()

    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BEV.res.history.soc"],
    )
    ax[1].set_ylabel("SOC")

    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
    )
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Ach Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


"""
TODO: describe reversible energy storage energy plot
"""

# %%
fig, ax = plot_res_energy()


# %%
def plot_road_loads():
    """Plot road loads"""
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    plt.suptitle("Road Loads")

    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_drag_watts"] / 1e3,
        label="drag",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_rr_watts"] / 1e3,
        label="rolling resistance",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
    )
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Ach. Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/road_loads.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


"""
TODO: describe road loads plot
"""

# %%
fig, ax = plot_road_loads()
