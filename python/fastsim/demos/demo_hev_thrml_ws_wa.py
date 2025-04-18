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

celsius_to_kelvin = 273.15
temp_amb = 38.0 + celsius_to_kelvin
temp_init = 45.0 + celsius_to_kelvin
# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%

# load 2021 Hyundai Sonata HEV from file
veh_dict = fsim.Vehicle.from_file(
    fsim.package_root()
    / "../../cal_and_val/thermal/f3-vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml"
).to_pydict()
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_init
veh_dict["pt_type"]["HybridElectricVehicle"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    "temperature_kelvin"
] = temp_init
veh_dict["pt_type"]["HybridElectricVehicle"]["fc"]["thrml"]["FuelConverterThermal"]["state"][
    "temperature_kelvin"
] = temp_init
veh = fsim.Vehicle.from_pydict(veh_dict)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)


# load cycle from file
cyc_dict = fsim.Cycle.from_resource("udds.csv").to_pydict()
cyc_dict["temp_amb_air_kelvin"] = [temp_amb] * len(cyc_dict["time_seconds"])
cyc = fsim.Cycle.from_pydict(cyc_dict)

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

# %%
df = sd.to_dataframe(allow_partial=True)
sd_dict = sd.to_pydict(flatten=True)
# # Visualize results


def plot_temperatures() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Component Temperatures")

    ax[0].set_prop_cycle(get_uni_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["cyc.temp_amb_air_kelvin"] - 273.15,
        label="amb",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.cabin.LumpedCabin.history.temperature_kelvin"] - 273.15,
        label="cabin",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df[
            "veh.pt_type.HybridElectricVehicle.res.thrml."
            + "RESLumpedThermal.history.temperature_kelvin"
        ]
        - 273.15,
        label="res",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df[
            "veh.pt_type.HybridElectricVehicle.fc.thrml."
            + "FuelConverterThermal.history.temperature_kelvin"
        ]
        - 273.15,
        label="fc",
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
    if SAVE_FIGS:
        plt.savefig(Path("./plots/temps.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_fc_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_prop_watts"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_aux_watts"]
        )
        / 1e3,
        label="shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.pwr_fuel_watts"] / 1e3,
        label="fuel",
    )
    ax[0].set_ylabel("FC Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
        label="soc",
    )
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc_disch_buffer"],
        label="accel buffer",
        alpha=0.5,
    )
    # ax[1].plot(
    #     df["cyc.time_seconds"],
    #     df["veh.pt_type.HybridElectricVehicle.res.history.soc_regen_buffer"],
    #     label='regen buffer',
    #     alpha=0.5,
    # )
    # ax[1].plot(
    #     df["cyc.time_seconds"],
    #     df['veh.pt_type.HybridElectricVehicle.fc.history.eff'],
    #     label='FC eff',
    # )
    ax[1].set_ylabel("[-]")
    ax[1].legend(loc="center right")

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
    if SAVE_FIGS:
        plt.savefig(Path("./plots/fc_pwr.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_fc_energy() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Fuel Converter Energy")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        (
            df["veh.pt_type.HybridElectricVehicle.fc.history.energy_prop_joules"]
            + df["veh.pt_type.HybridElectricVehicle.fc.history.energy_aux_joules"]
        )
        / 1e6,
        label="shaft",
    )
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.fc.history.energy_fuel_joules"] / 1e6,
        label="fuel",
    )
    ax[0].set_ylabel("FC Energy [MJ]")
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
    if SAVE_FIGS:
        plt.savefig(Path(f"./plots/fc_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_res_pwr() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)
    plt.suptitle("Reversible Energy Storage Power")

    ax[0].set_prop_cycle(get_paired_cycler())
    ax[0].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.pwr_out_electrical_watts"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Power [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
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
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.energy_out_electrical_joules"] / 1e3,
        label="electrical out",
    )
    ax[0].set_ylabel("RES Energy [kW]")
    ax[0].legend()

    ax[1].set_prop_cycle(get_paired_cycler())
    ax[1].plot(
        df["cyc.time_seconds"],
        df["veh.pt_type.HybridElectricVehicle.res.history.soc"],
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
    if SAVE_FIGS:
        plt.savefig(Path("./plots/res_energy.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


def plot_road_loads() -> Tuple[Figure, Axes]:
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
    if SAVE_FIGS:
        plt.savefig(Path("./plots/road_loads.svg"))
    if SHOW_PLOTS:
        plt.show()

    return fig, ax


fig_fc_pwr, ax_fc_pwr = plot_fc_pwr()
fig_fc_energy, ax_fc_energy = plot_fc_energy()
fig_res_pwr, ax_res_pwr = plot_res_pwr()
fig_res_energy, ax_res_energy = plot_res_energy()
fig_temps, ax_temps = plot_temperatures()
fig, ax = plot_road_loads()

# %%

# %%
# example for how to use set_default_pwr_interp() method for veh.res
res = fsim.ReversibleEnergyStorage.from_pydict(
    sd.to_pydict()["veh"]["pt_type"]["HybridElectricVehicle"]["res"]
)
res.set_default_pwr_interp()

# %%
