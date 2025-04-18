# %%

from plot_utils import *
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


# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%

# load 2012 Ford Fusion from file
veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
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


def plot_fc_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.ConventionalVehicle.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.ConventionalVehicle.fc.history.pwr_aux_watts"]
        )
        / 1e3,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.fc_kw_out_ach.tolist()),
        label="f2 shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.ConventionalVehicle.fc.history.pwr_fuel_watts"] / 1e3,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.fs_kw_out_ach.tolist()),
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.ConventionalVehicle.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.ConventionalVehicle.fc.history.pwr_aux_watts"]
        )
        / 1e3
        - np.array(sd2.fc_kw_out_ach.tolist()),
        label="shaft",
    )
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.ConventionalVehicle.fc.history.pwr_fuel_watts"] / 1e3
        - np.array(sd2.fs_kw_out_ach.tolist()),
        label="fuel",
    )
    ax[1].set_ylabel("FC Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

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
        plt.savefig(Path("./plots/fc_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_fc_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        (
            df["veh.pt_type.ConventionalVehicle.fc.history.energy_prop_joules"]
            + df["veh.pt_type.ConventionalVehicle.fc.history.energy_aux_joules"]
        )
        / 1e6,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.fc_cumu_mj_out_ach.tolist()),
        label="f2 shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.pt_type.ConventionalVehicle.fc.history.energy_fuel_joules"] / 1e6,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.fs_cumu_mj_out_ach.tolist()),
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Energy [MJ]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        (
            df["veh.pt_type.ConventionalVehicle.fc.history.energy_prop_joules"]
            + df["veh.pt_type.ConventionalVehicle.fc.history.energy_aux_joules"]
        )
        / 1e6
        - np.array(sd2.fc_cumu_mj_out_ach.tolist()),
        label="shaft",
    )
    ax[1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.pt_type.ConventionalVehicle.fc.history.energy_fuel_joules"] / 1e6
        - np.array(sd2.fs_cumu_mj_out_ach.tolist()),
        label="fuel",
    )
    ax[1].set_ylim(
        (
            -sd_dict["veh.pt_type.ConventionalVehicle.fc.state.energy_fuel_joules"] * 1e-6 * 0.1,
            sd_dict["veh.pt_type.ConventionalVehicle.fc.state.energy_fuel_joules"] * 1e-6 * 0.1,
        )
    )
    ax[1].set_ylabel("FC Energy\nDelta (f3-f2) [MJ]\n+/- 10% Range")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
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
        plt.savefig(Path("./plots/fc_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_road_loads() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_drag_watts"] / 1e3,
        label="f3 drag",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval],
        np.array(sd2.drag_kw.tolist()),
        label="f2 drag",
    )
    ax[0].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_rr_watts"] / 1e3,
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
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_drag_watts"] / 1e3 - np.array(sd2.drag_kw.tolist()),
        label="drag",
        linestyle=BASE_LINE_STYLES[0],
    )
    ax[1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
        df["veh.history.pwr_rr_watts"] / 1e3 - np.array(sd2.rr_kw.tolist()),
        label="rr",
        linestyle=BASE_LINE_STYLES[1],
    )
    ax[1].set_ylabel("Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"][:: veh.save_interval],
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
    ax[-1].set_ylabel("Ach. Speed [m/s]")

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/road_loads.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


fig, ax = plot_fc_pwr()
fig, ax = plot_fc_energy()
fig, ax = plot_road_loads()

# # Benchmarking
#
# ## CPU Performance
# See above for cpu benchmarking.
#
# ## Memory Profiling
# Within [benchmarks](./../../../benchmarks/), here are the results of running the benchmarking tests.
#
# These benchmarks show that `fastsim-3` uses about 63% less memory than `fastsim-2`, which results in a substantially greater potential to leverage HPC and cloud computing via multiprocessing parallelization, enabling more capacity to use FASTSim for large scale simulations.
#
# ### python -m memory_profiler f2.py
#
# ```
# Filename: /Users/cbaker2/Documents/GitHub/fastsim-3/benchmarks/./f2.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
# 10  163.438 MiB  163.438 MiB           1   @profile(precision=3)
# 11                                         def build_and_run_sim_drive():
# 12  164.234 MiB    0.797 MiB           2       veh = fsr.RustVehicle.from_file(
# 13  163.438 MiB    0.000 MiB           1           str(fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml")
# 14                                             )
# 15  164.375 MiB    0.141 MiB           1       cyc = fsr.RustCycle.from_resource("cycles/udds.csv")
# 16  165.453 MiB    1.078 MiB           1       sd = fsr.RustSimDrive(cyc, veh)
# 17  165.656 MiB    0.203 MiB           1       sd.sim_drive()
# ```
#
# ### python -m memory_profiler f3-save-int-1.py
#
# ```
# Filename: /Users/cbaker2/Documents/GitHub/fastsim-3/benchmarks/./f3-save-int-1.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#     5   61.562 MiB   61.562 MiB           1   @profile(precision=3)
#     6                                         def build_and_run_sim_drive():
#     7   62.125 MiB    0.562 MiB           2       veh = fsim.Vehicle.from_file(
#     8
#     9   61.562 MiB    0.000 MiB           1           fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml"
#     10                                             )
#     11   62.125 MiB    0.000 MiB           1       veh.save_interval = 1
#     12   62.312 MiB    0.188 MiB           1       cyc = fsim.Cycle.from_resource("cycles/udds.csv")
#     13   62.406 MiB    0.094 MiB           1       sd = fsim.SimDrive(veh, cyc)
#     14   62.953 MiB    0.547 MiB           1       sd.walk()
# ```
#
# ### python -m memory_profiler f3-save-int-none.py
#
# ```
# Filename: /Users/cbaker2/Documents/GitHub/fastsim-3/benchmarks/./f3-save-int-none.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#      5   61.203 MiB   61.203 MiB           1   @profile(precision=3)
#      6                                         def build_and_run_sim_drive():
#      7   61.766 MiB    0.562 MiB           2       veh = fsim.Vehicle.from_file(
#      8
#      9   61.203 MiB    0.000 MiB           1           fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml"
#     10                                             )
#     11   61.766 MiB    0.000 MiB           1       veh.save_interval = None
#     12   61.938 MiB    0.172 MiB           1       cyc = fsim.Cycle.from_resource("cycles/udds.csv")
#     13   62.031 MiB    0.094 MiB           1       sd = fsim.SimDrive(veh, cyc)
#     14   62.031 MiB    0.000 MiB           1       sd.walk()
# ```


# %%
