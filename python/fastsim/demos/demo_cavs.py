"""
Demonstration of Connected Automated Vehicle (CAV) Functionality in FASTSim

This module demonstrates:
- cycle manipulation utilities
- eco-approach: utilizing vehicle coasting to conserve fuel use
- eco-cruise: use of trajectories to remove unnecessary accelerations
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import fastsim as fsim
import fastsim.demos.plot_utils as pu

sns.set_theme()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"
LIST_COLUMN_OPTIONS = False


def microtrip_demo():
    """Run a demonstration of cycle manipulation utilities"""
    cycle_name = "udds"
    cycle = fsim.Cycle.from_resource(f"{cycle_name}.csv")

    microtrips = cycle.to_microtrips(None)

    if SHOW_PLOTS:
        max_microtrips = 4
        fig, ax = plt.subplots()
        num = min(max_microtrips, len(microtrips))
        for idx, mt in enumerate(microtrips):
            color = pu.BASE_COLORS[idx % len(pu.BASE_COLORS)]
            line = pu.BASE_LINE_STYLES[idx % len(pu.BASE_LINE_STYLES)]
            ax.plot(
                mt.time_s,
                mt.speed_m_per_s,
                marker=".",
                color=color,
                linestyle=line,
                label=f"#{idx + 1}",
            )
            if idx >= max_microtrips:
                break
        ax.set_title(f"First {num} Microtrips of {cycle_name.upper()}")
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)


def coasting_demo():
    """Run a demonstration of a coasting maneuver"""
    # veh = fsim.Vehicle.from_resource("2022_Renault_Zoe_ZE50_R135.yaml")
    coast_speed_mps = 20.0
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    cyc = fsim.Cycle.from_resource("udds.csv")
    # add 100 seconds to the cycle time to allow for delay caused by
    # coasting
    cyc = cyc.extend_time(absolute_time_s=100.0, time_fraction=None)
    cyc0 = cyc.copy()
    man = fsim.Maneuver.create_from(cyc, veh.copy())
    d = man.to_pydict()
    d["coast_allow"] = True
    d["coast_start_speed_meters_per_second"] = coast_speed_mps
    man = fsim.Maneuver.from_pydict(d)
    d = man.to_pydict()
    print(f"coast_allow: {d['coast_allow']}")
    cyc = man.apply_maneuvers()
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    if SHOW_PLOTS:
        df = sd.to_dataframe()
        if LIST_COLUMN_OPTIONS:
            print("Available Columns:")
            for column_name in df.columns:
                print(f"- {column_name}")
        fig, ax = plt.subplots()
        ax.plot(cyc0.time_s, cyc0.speed_m_per_s, "k-", label="original")
        ax.plot(
            np.array(df["cyc.time_seconds"])[:: veh.save_interval],
            np.array(df["veh.history.speed_ach_meters_per_second"]),
            "b:", label="coast")
        ax.set_title(f"Coasting behavior from {coast_speed_mps} m/s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Speed [m/s]")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)


if __name__ == "__main__":
    microtrip_demo()
    coasting_demo()
    print("Done!")
