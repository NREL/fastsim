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

from plot_utils import *

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%

# load 2012 Ford Fusion from file
veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
veh_no_save = veh.copy()
veh_no_save.set_save_interval(None)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# instantiate `SimDrive` simulation object
sd0 = fsim.SimDrive(veh, cyc)
sd = sd0.copy()

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s")
df = sd.to_dataframe()
sd_dict = sd.to_pydict(flatten=True)

# instantiate `SimDrive` simulation object
sd_no_save = fsim.SimDrive(veh_no_save, cyc)

# simulation start time
t0 = time.perf_counter()
# run simulation
sd_no_save.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si_none = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of None:\n{t_fsim3_si_none:.2e} s")

# %%
# # `fastsim-2` benchmarking

sd2 = sd0.to_fastsim2()
t0 = time.perf_counter()
sd2.sim_drive()
t1 = time.perf_counter()
t_fsim2 = t1 - t0
print(f"fastsim-2 `sd.walk()` elapsed time: {t_fsim2:.2e} s")
print(
    "`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of 1:"
)
print(f"{t_fsim2 / t_fsim3_si1:.3g}x")
print(
    "`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of `None`:"
)
print(f"{t_fsim2 / t_fsim3_si_none:.3g}x")

# # Visualize results


def plot_res_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3,
        label="f3 electrical out",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.ess_kw_out_ach.tolist()),
        label="f2 electrical out",
    )
    ax[0].set_ylabel("RES Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3
        - np.array(sd2.ess_kw_out_ach.tolist()),
        label="f3 res kw out",
    )
    ax[1].set_ylabel("RES Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"]
        - (
            df["veh.pt_type.BatteryElectricVehicle.res.history.soc"][0]
            - np.array(sd2.soc.tolist())[0]
        ),
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.soc.tolist()),
        label="f2 soc",
    )
    ax[2].set_ylabel("SOC")
    ax[2].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist()),
        np.array(sd2.mps_ach.tolist()),
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_res_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"] / 1e3,
        label="f3 electrical out",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.cumsum(
            np.array(sd2.ess_kw_out_ach.tolist()) * np.diff(sd2.cyc.time_s.tolist(), prepend=0)
        ),
        label="f2 electrical out",
    )
    ax[0].set_ylabel("RES Energy [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"] / 1e3
        - np.cumsum(
            np.array(sd2.ess_kw_out_ach.tolist()) * np.diff(sd2.cyc.time_s.tolist(), prepend=0)
        ),
        label="electrical out",
    )
    ax[1].set_ylim(
        -np.max(
            np.abs(
                sd_dict[
                    "veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"
                ]
            )
        )
        * 1e-3
        * 0.1,
        np.max(
            np.abs(
                sd_dict[
                    "veh.pt_type.BatteryElectricVehicle.res.history.energy_out_electrical_joules"
                ]
            )
        )
        * 1e-3
        * 0.1,
    )
    ax[1].set_ylabel("RES Energy\nDelta (f3-f2) [kJ]\n+/- 10% Range")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.BatteryElectricVehicle.res.history.soc"]
        - (
            df["veh.pt_type.BatteryElectricVehicle.res.history.soc"][0]
            - np.array(sd2.soc.tolist())[0]
        ),
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.soc.tolist()),
        label="f2 soc",
    )
    ax[2].set_ylabel("SOC")
    ax[2].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist()),
        np.array(sd2.mps_ach.tolist()),
        label="f2",
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
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        np.array(df["cyc.time_seconds"])[:: veh.save_interval],
        np.array(df["veh.history.pwr_drag_watts"]) / 1e3,
        label="f3 drag",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.drag_kw.tolist()),
        label="f2 drag",
    )
    ax[0].plot(
        np.array(df["cyc.time_seconds"])[:: veh.save_interval],
        np.array(df["veh.history.pwr_rr_watts"]) / 1e3,
        label="f3 rr",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.rr_kw.tolist()),
        label="f2 rr",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        np.array(df["cyc.time_seconds"])[:: veh.save_interval],
        np.array(df["veh.history.pwr_drag_watts"]) / 1e3 - np.array(sd2.drag_kw.tolist()),
        label="drag",
        linestyle=BASE_LINE_STYLES[0],
    )
    ax[1].plot(
        np.array(df["cyc.time_seconds"])[:: veh.save_interval],
        np.array(df["veh.history.pwr_rr_watts"]) / 1e3 - np.array(sd2.rr_kw.tolist()),
        label="rr",
        linestyle=BASE_LINE_STYLES[1],
    )
    # ax[1].text(
    #     500, -0.125, "Drag error is due to more\naccurate air density model .")
    ax[1].set_ylabel("Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        np.array(df["cyc.time_seconds"])[:: veh.save_interval],
        np.array(df["veh.history.speed_ach_meters_per_second"]),
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist()),
        np.array(sd2.mps_ach.tolist()),
        label="f2",
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
res = fsim.ReversibleEnergyStorage.from_pydict(
    sd.to_pydict()["veh"]["pt_type"]["BatteryElectricVehicle"]["res"]
)
res.set_default_pwr_interp()


def test_this_file():
    """to trigger automated testing"""
    pass
