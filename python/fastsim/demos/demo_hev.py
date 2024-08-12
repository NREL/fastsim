# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc_params
from cycler import cycler
import seaborn as sns
from pathlib import Path
import time
import json
import os
from typing import Tuple
import fastsim as fsim

sns.set_theme()

# if enivronment var `DEBUG_LOG=true` is set, turns on debug logging
DEBUG_LOG = os.environ.get("DEBUG_LOG", "false").lower() == "true"     
# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"     
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

# `fastsim3` -- load vehicle and cycle, build simulation, and run 
# %%

# load 2016 Toyota Prius Two from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2016_TOYOTA_Prius_Two.yaml")
)
veh_no_save = veh.copy()
fsim.set_param_from_path(veh_no_save, "save_interval", None)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
fsim.set_param_from_path(veh, "save_interval", 1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# instantiate `SimDrive` simulation object
sd0 = fsim.SimDrive(veh, cyc)
sd = sd0.copy()

# simulation start time
t0 = time.perf_counter()
# run simulation
if DEBUG_LOG:
    with fsim.utils.with_logging():
        sd.walk()
else:
    sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s")

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

# `fastsim-2` benchmarking
# %%

sd2 = sd0.to_fastsim2()
t0 = time.perf_counter()
with fsim.utils.without_logging(): # suppresses known warning
    sd2.sim_drive()
t1 = time.perf_counter()
t_fsim2 = t1 - t0
print(f"fastsim-2 `sd.walk()` elapsed time: {t_fsim2:.2e} s")
print("`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of 1:")
print(f"{t_fsim2/t_fsim3_si1:.3g}x")
print("`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of `None`:")
print(f"{t_fsim2/t_fsim3_si_none:.3g}x")
# Visualize results

# %%
def plot_road_loads() -> Tuple[Figure, Axes]: 
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.pwr_drag_watts) / 1e3,
        label="f3 drag",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.drag_kw.tolist()),
        label="f2 drag",
    )
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.pwr_rr_watts) / 1e3,
        label="f3 rr",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.rr_kw.tolist()),
        label="f2 rr",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.pwr_drag_watts) /
        1e3 - np.array(sd2.drag_kw.tolist()),
        label="drag",
        linestyle=baselinestyles[0],
    )
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.pwr_rr_watts) /
        1e3 - np.array(sd2.rr_kw.tolist()),
        label="rr",
        linestyle=baselinestyles[1],
    )
    ax[1].text(
        500, -0.125, "Drag error is due to more\naccurate air density model .")
    ax[1].set_ylabel("Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
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
    plt.show()

    return fig, ax

def plot_fc_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_propulsion_watts) +
            np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.fc_kw_out_ach.tolist()),
        label="f2 shaft",
    )
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.fc.history.pwr_fuel_watts) / 1e3,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.fs_kw_out_ach.tolist()),
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_propulsion_watts) +
            np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="shaft",
        linestyle=baselinestyles[0]
    )
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_propulsion_watts) +
            np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="fuel",
        linestyle=baselinestyles[1]
    )
    ax[1].set_ylabel("FC Power\nDelta (f3-f2) [kW]")
    ax[1].legend()
    
    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
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
        plt.savefig(Path("./plots/fc_pwr.svg"))
    plt.show()

    return fig, ax

def plot_fc_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.energy_propulsion_joules) +
            np.array(sd.veh.fc.history.energy_aux_joules)) / 1e6,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.fc_cumu_mj_out_ach.tolist()),
        label="f2 shaft",
    )
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.fc.history.energy_fuel_joules) / 1e6,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.fs_cumu_mj_out_ach.tolist()),
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Energy [MJ]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.energy_propulsion_joules) +
            np.array(sd.veh.fc.history.energy_aux_joules)) / 1e6 - np.array(sd2.fc_cumu_mj_out_ach.tolist()),
        label="shaft",
        linestyle=baselinestyles[0]
    )
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.energy_propulsion_joules) +
            np.array(sd.veh.fc.history.energy_aux_joules)) / 1e6 - np.array(sd2.fc_cumu_mj_out_ach.tolist()),
        label="fuel",
        linestyle=baselinestyles[1]
    )
    ax[1].set_ylabel("FC Energy\nDelta (f3-f2) [MJ]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
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
        plt.savefig(Path("./plots/fc_energy.svg"))
    plt.show()

    return fig, ax

def plot_res_pwr() -> Tuple[Figure, Axes]: 
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Battery Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.pwr_out_electrical_watts) / 1e3,
        label="f3 batt elec",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.ess_kw_out_ach.tolist()),
        label="f2 batt elec",
    )
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.pwr_out_chemical_watts) / 1e3,
        label="f3 batt chem",
    )
    ax[0].set_ylabel("RES (battery) Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.pwr_out_electrical_watts) / 1e3 - np.array(sd2.ess_kw_out_ach.tolist()),
        label="batt elec",
        linestyle=baselinestyles[0]
    )
    ax[1].set_ylabel("FC Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
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
        plt.savefig(Path("./plots/battery_pwr.svg"))
    plt.show()

    return fig, ax

def plot_res_energy() -> Tuple[Figure, Axes]: 
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Battery Energy")

    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.energy_out_electrical_joules) / 1e6,
        label="f3 batt elec",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.ess_kw_out_ach.tolist()),
        label="f2 batt elec",
    )
    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.energy_out_chemical_joules) / 1e6,
        label="f3 batt chem",
    )
    ax[0].set_ylabel("RES (battery) Energy [MJ]")
    ax[0].legend()

    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.res.history.energy_out_electrical_joules) / 1e6 - np.array(sd2.ess_kw_out_ach.tolist()),
        label="batt elec",
        linestyle=baselinestyles[0]
    )
    ax[1].set_ylabel("FC Energy\nDelta [MJ]")
    ax[1].legend()

    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
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
        plt.savefig(Path("./plots/battery_energy.svg"))
    plt.show()

    return fig, ax

def plot_pwr_split() -> Tuple[Figure, Axes]: ...

figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
baselinestyles = ["--", "-.",]

def get_paired_cycler():
    colors = [[c, c] for c in base_colors]
    colors = [x for sublist in colors for x in sublist]
    linestyles = (baselinestyles * int(np.ceil(len(colors) / len(baselinestyles))))[:len(colors)]
    paired_cycler = (
        cycler(color=colors) +
        cycler(linestyle=linestyles)
    )
    return paired_cycler

def get_uni_cycler():
    colors = base_colors
    baselinestyles = ["--",]
    linestyles = (baselinestyles * int(np.ceil(len(colors) / len(baselinestyles))))[:len(colors)]
    uni_cycler = (
        cycler(color=colors) +
        cycler(linestyle=linestyles)
    )
    return uni_cycler

if SHOW_PLOTS:
    fig, ax = plot_road_loads()
    fig, ax = plot_fc_pwr()
    fig, ax = plot_fc_energy()
    fig, ax = plot_res_pwr()
    fig, ax = plot_res_energy()
    




# %%
