# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_params
from cycler import cycler
import seaborn as sns
import time
import json
import os
import fastsim as fsim

sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true" 
SHOW_PLOTS = False

# %% [markdown]

# `fastsim3` -- load vehicle and cycle, build simulation, and run 
# %%

# load 2012 Ford Fusion from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.save_interval = 1

# load cycle from file
cyc = fsim.Cycle.from_resource("cycles/udds.csv")

print(veh.param_path_list())

print(veh.history_path_list())

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)

# print([attr for attr in sd.__dir__() if not attr.startswith("__") and not callable(getattr(sd,attr))])
# print(sd.param_path_list())

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
print(f"fastsim-3 `sd.walk()` elapsed time: {t1-t0:.2e} s")

# %% [markdown]  

# `fastsim-2` benchmarking

# %%

sd2 = sd.to_fastsim2()
t0 = time.perf_counter()
sd2.sim_drive()
t1 = time.perf_counter()
print(f"fastsim-2 `sd.walk()` elapsed time: {t1-t0:.2e} s")

# %% [markdown]

# Visualize results

# %%

if SHOW_PLOTS:
    figsize_3_stacked = (10, 9)

    # set up cycling of colors and linestyles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf']
    baselinestyles = ["--", "-.", ":"]
    linestyles = [[c, c] for c in baselinestyles]
    linestyles = [x for sublist in linestyles for x in sublist]
    default_cycler = (
        cycler(color=colors[:len(linestyles)]) +
        cycler(linestyle=linestyles)
    )
    plt.rc('axes', prop_cycle=default_cycler)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")

    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_tractive_watts) +
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

    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_tractive_watts) +
         np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="shaft",
        linestyle=baselinestyles[0]
    )
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_tractive_watts) +
         np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="fuel",
        linestyle=baselinestyles[1]
    )
    ax[1].set_ylabel("FC Power\nDelta [kW]")
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
    plt.show()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Energy")

    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.energy_tractive_joules) +
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
    ax[0].text(
        200, 
        13, 
        "Discrepancy mostly due to switch to linear interpolation\n" + 
        "from left-hand interpolation resulting in more accurate\n" + 
        "handling of idling conditions.",
    )
    ax[0].set_ylabel("FC Energy [MJ]")
    ax[0].legend()

    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_tractive_watts) +
         np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="shaft",
        linestyle=baselinestyles[0]
    )
    ax[1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        (np.array(sd.veh.fc.history.pwr_tractive_watts) +
         np.array(sd.veh.fc.history.pwr_aux_watts)) / 1e3 - np.array(sd2.fc_kw_out_ach.tolist()),
        label="fuel",
        linestyle=baselinestyles[1]
    )
    ax[1].set_ylabel("FC Power\nDelta [kW]")
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
    plt.show()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

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
    ax[1].text(500, -0.125, "Drag error is due to intentional\nair density model change.")
    ax[1].set_ylabel("Power\nDelta [kW]")
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
    ax[-1].set_ylabel("Ach. Speed [m/s]")
    plt.show()

# %%
