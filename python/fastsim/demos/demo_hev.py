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

# load 2016 Toyota Prius Two from file
veh = fsim.Vehicle.from_resource("2016_TOYOTA_Prius_Two.yaml")
# veh_dict = veh.to_pydict(flatten=False)
# veh_dict['pt_type']['HybridElectricVehicle'][
#     'pt_cntrl']['RGWDB']['speed_soc_disch_buffer'] += 0.0
# speed_disch_soc_buffer = veh_dict['pt_type']['HybridElectricVehicle'][
#     'pt_cntrl']['RGWDB']['speed_soc_disch_buffer']
# veh_dict['pt_type']['HybridElectricVehicle'][
#     'pt_cntrl']['RGWDB']['speed_soc_fc_on_buffer'] = speed_disch_soc_buffer * 1.1
# veh = veh.from_pydict(veh_dict)

veh_no_save = veh.copy()
veh_no_save.set_save_interval(None)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh_no_save.set_save_interval(1)


# load cycle from file
# TODO make it so that the cycles in resources have `name` populated
cyc = fsim.Cycle.from_resource("udds.csv")

# instantiate `SimDrive` simulation object
sd0 = fsim.SimDrive(veh, cyc)
sd = sd0.copy()
# sd_dict['sim_params']['trace_miss_opts'] = 'Error'
# sd = fsim.SimDrive.from_pydict(sd_dict)

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s")

# %%

# plt_slice = slice(200)
# df = sd.to_dataframe()[plt_slice]
df = sd.to_dataframe()
plt_slice = slice(0, len(df))
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

# `fastsim-2` benchmarking
# %%

sd2 = sd0.to_fastsim2()
t0 = time.perf_counter()
with fsim.utils.without_logging():  # suppresses known warning
    sd2.sim_drive()
t1 = time.perf_counter()
t_fsim2 = t1 - t0
print(f"fastsim-2 `sim_drive()` elapsed time: {t_fsim2:.2e} s")
print(
    "`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of 1:"
)
print(f"{t_fsim2 / t_fsim3_si1:.3g}x")
print(
    "`fastsim-3` speedup relative to `fastsim-2` (should be greater than 1) for `save_interval` of `None`:"
)
print(f"{t_fsim2 / t_fsim3_si_none:.3g}x")
# Visualize results

# %%


def plot_road_loads() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Road Loads")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_drag_watts"] / 1e3,
        label="f3 drag",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.drag_kw.tolist())[plt_slice],
        label="f2 drag",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_rr_watts"] / 1e3,
        label="f3 rr",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.rr_kw.tolist())[plt_slice],
        label="f2 rr",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_drag_watts"] / 1e3 - np.array(sd2.drag_kw.tolist())[plt_slice],
        label="drag",
        linestyle=baselinestyles[0],
    )
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.history.pwr_rr_watts"] / 1e3 - np.array(sd2.rr_kw.tolist())[plt_slice],
        label="rr",
        linestyle=baselinestyles[1],
    )
    ax[1].set_ylabel("Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist())[plt_slice],
        np.array(sd2.mps_ach.tolist())[plt_slice],
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/road_loads.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_fc_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_aux_watts"]
        )
        / 1e3,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.fc_kw_out_ach.tolist())[plt_slice],
        label="f2 shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_fuel_watts"] / 1e3,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.fs_kw_out_ach.tolist())[plt_slice],
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_aux_watts"]
        )
        / 1e3
        - np.array(sd2.fc_kw_out_ach.tolist())[plt_slice],
        label="shaft",
        linestyle=baselinestyles[0],
    )
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_fuel_watts"] / 1e3
        - np.array(sd2.fs_kw_out_ach.tolist())[plt_slice],
        label="fuel",
        linestyle=baselinestyles[1],
    )
    ax[1].set_ylabel("FC Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.soc.tolist())[plt_slice],
        label="f2 soc",
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_disch_buffer"],
        label="f3 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.accel_buff_soc.tolist())[plt_slice],
        label="f2 accel buffer",
        alpha=0.5,
    )
    # ax[2].plot(
    #     df["cyc.time_seconds"],
    #     df["veh.pt_type.HybridElectricVehicle.res.history.soc_regen_buffer"],
    #     label='f3 regen buffer',
    #     alpha=0.5,
    # )
    # ax[2].plot(
    #     np.array(sd2.cyc.time_s.tolist())[::veh.save_interval][plt_slice],
    #     np.array(sd2.regen_buff_soc.tolist())[plt_slice],
    #     label='f2 regen buffer',
    #     alpha=0.5,
    # )
    # ax[2].plot(
    #     df["cyc.time_seconds"],
    #     df['veh.pt_type.HybridElectricVehicle.fc.history.eff'],
    #     label='f3 FC eff',
    # )
    # f2_fc_eff = (np.array(sd2.fc_kw_out_ach.tolist()) /
    #              np.array(sd2.fc_kw_in_ach.tolist()))[plt_slice]
    # ax[2].plot(
    #     np.array(sd2.cyc.time_s.tolist())[::veh.save_interval][plt_slice],
    #     f2_fc_eff,
    #     label='f2 FC eff',
    # )
    ax[2].set_ylabel("[-]")
    ax[2].legend(loc="center right")

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist())[plt_slice],
        np.array(sd2.mps_ach.tolist())[plt_slice],
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/fc_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_fc_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.energy_prop_joules"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.energy_aux_joules"]
        )
        / 1e6,
        label="f3 shaft",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.fc_cumu_mj_out_ach.tolist())[plt_slice],
        label="f2 shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.energy_fuel_joules"] / 1e6,
        label="f3 fuel",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.fs_cumu_mj_out_ach.tolist())[plt_slice],
        label="f2 fuel",
    )
    ax[0].set_ylabel("FC Energy [MJ]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.energy_prop_joules"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.energy_aux_joules"]
        )
        / 1e6
        - np.array(sd2.fc_cumu_mj_out_ach.tolist())[plt_slice],
        label="shaft",
        linestyle=baselinestyles[0],
    )
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.energy_fuel_joules"] / 1e6
        - np.array(sd2.fs_cumu_mj_out_ach.tolist())[plt_slice],
        label="fuel",
        linestyle=baselinestyles[1],
    )
    ax[1].set_ylim(
        (
            -sd_dict["veh.pt_type.HybridElectricVehicle.fc.state.energy_fuel_joules"] * 1e-6 * 0.1,
            sd_dict["veh.pt_type.HybridElectricVehicle.fc.state.energy_fuel_joules"] * 1e-6 * 0.1,
        )
    )
    ax[1].set_ylabel("FC Energy\nDelta (f3-f2) [MJ]\n+/- 10% Range")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.soc.tolist())[plt_slice],
        label="f2 soc",
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_disch_buffer"],
        label="f3 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.accel_buff_soc.tolist())[plt_slice],
        label="f2 accel buffer",
        alpha=0.5,
    )
    # ax[2].plot(
    #     df["cyc.time_seconds"],
    #     df["veh.pt_type.HybridElectricVehicle.res.history.soc_regen_buffer"],
    #     label='f3 regen buffer',
    #     alpha=0.5,
    # )
    # ax[2].plot(
    #     np.array(sd2.cyc.time_s.tolist())[::veh.save_interval][plt_slice],
    #     np.array(sd2.regen_buff_soc.tolist())[plt_slice],
    #     label='f2 regen buffer',
    #     alpha=0.5,
    # )
    # ax[2].plot(
    #     df["cyc.time_seconds"],
    #     df['veh.pt_type.HybridElectricVehicle.fc.history.eff'],
    #     label='f3 FC eff',
    # )
    # f2_fc_eff = (np.array(sd2.fc_kw_out_ach.tolist()) /
    #              np.array(sd2.fc_kw_in_ach.tolist()))[plt_slice]
    # ax[2].plot(
    #     np.array(sd2.cyc.time_s.tolist())[::veh.save_interval][plt_slice],
    #     f2_fc_eff,
    #     label='f2 FC eff',
    # )
    ax[2].set_ylabel("[-]")
    ax[2].legend(loc="center right")

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist())[plt_slice],
        np.array(sd2.mps_ach.tolist())[plt_slice],
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path(f"./plots/fc_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_res_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Battery Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3,
        label="f3 batt elec",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.ess_kw_out_ach.tolist())[plt_slice],
        label="f2 batt elec",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.pwr_out_chemical_watts"] / 1e3,
        label="f3 batt chem",
    )
    ax[0].set_ylabel("RES (battery) Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_uni_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3
        - np.array(sd2.ess_kw_out_ach.tolist())[plt_slice],
        label="batt elec",
        linestyle=baselinestyles[0],
    )
    ax[1].set_ylabel("RES Power\nDelta (f3-f2) [kW]")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.soc.tolist())[plt_slice],
        label="f2 soc",
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_disch_buffer"],
        label="f3 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.accel_buff_soc.tolist())[plt_slice],
        label="f2 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_regen_buffer"],
        label="f3 regen buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.regen_buff_soc.tolist())[plt_slice],
        label="f2 regen buffer",
        alpha=0.5,
    )
    ax[2].set_ylabel("[-]")
    ax[2].legend(loc="center right")

    ax[-1].set_prop_cycle(get_paired_cycler())
    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist())[plt_slice],
        np.array(sd2.mps_ach.tolist())[plt_slice],
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/battery_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_res_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Battery Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.energy_out_electrical_joules"] / 1e6,
        label="f3 batt elec",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        (
            (
                np.array(sd2.ess_kw_out_ach.tolist())
                * np.diff(np.array(sd2.cyc.time_s.tolist()), prepend=0.0)
            ).cumsum()
            / 1e3
        )[plt_slice],
        label="f2 batt elec",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.energy_out_chemical_joules"] / 1e6,
        label="f3 batt chem",
    )
    ax[0].set_ylabel("RES (battery) Energy [MJ]")
    ax[0].legend()

    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.energy_out_electrical_joules"] / 1e6
        - np.array(sd2.ess_kw_out_ach.tolist())[plt_slice],
        label="batt elec",
        linestyle=baselinestyles[0],
    )
    ax[1].set_ylabel("RES Energy\nDelta [MJ]")
    ax[1].legend()

    ax[2].set_prop_cycle(get_paired_cycler())
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
        label="f3 soc",
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.soc.tolist())[plt_slice],
        label="f2 soc",
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_disch_buffer"],
        label="f3 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.accel_buff_soc.tolist())[plt_slice],
        label="f2 accel buffer",
        alpha=0.5,
    )
    ax[2].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_regen_buffer"],
        label="f3 regen buffer",
        alpha=0.5,
    )
    ax[2].plot(
        np.array(sd2.cyc.time_s.tolist())[:: veh.save_interval][plt_slice],
        np.array(sd2.regen_buff_soc.tolist())[plt_slice],
        label="f2 regen buffer",
        alpha=0.5,
    )
    ax[2].set_ylabel("[-]")
    ax[2].legend(loc="center right")

    ax[-1].plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        label="f3",
    )
    ax[-1].plot(
        np.array(sd2.cyc.time_s.tolist())[plt_slice],
        np.array(sd2.mps_ach.tolist())[plt_slice],
        label="f2",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Ach Speed [m/s]")
    x_min, x_max = ax[-1].get_xlim()[0], ax[-1].get_xlim()[1]
    x_max = (x_max - x_min) * 1.15
    ax[-1].set_xlim([x_min, x_max])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(Path("./plots/battery_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


# def plot_pwr_split() -> Tuple[Figure, Axes]: ...


figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
base_colors = [
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
baselinestyles = [
    "--",
    "-.",
]

fig, ax = plot_road_loads()
fig, ax = plot_fc_pwr()
fig, ax = plot_fc_energy()
fig, ax = plot_res_pwr()
fig, ax = plot_res_energy()

# %%
