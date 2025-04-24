"""
Demonstration of Connected Automated Vehicle (CAV) Functionality in FASTSim

This module demonstrates:
- cycle manipulation utilities
- eco-approach: utilizing vehicle coasting to conserve fuel use
- eco-cruise: use of trajectories to remove unnecessary accelerations
"""

import os

import matplotlib.pyplot as plt
import plot_utils as pu
import seaborn as sns

import fastsim as fsim

sns.set_theme()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"


def main():
    """Run a demonstration of cycle manipulation utilities"""
    cycle_name = "udds"
    cycle = fsim.Cycle.from_resource(f"{cycle_name}.csv")

    microtrips = cycle.to_microtrips(None)

    if SHOW_PLOTS:
        fig, ax = plt.subplots()
        num = min(4, len(microtrips))
        for idx, mt in enumerate(microtrips):
            color = pu.BASE_COLORS[idx % len(pu.BASE_COLORS)]
            line = pu.BASE_LINE_STYLES[idx % len(pu.BASE_LINE_STYLES)]
            ax.plot(mt.time_s, mt.speed_m_per_s,
                    marker=".", color=color, linestyle=line,
                    label=f"#{idx + 1}")
            if idx >= 4:
                break
        ax.set_title(f"First {num} Microtrips of {cycle_name.upper()}")
        ax.set_ylabel("Speed (m/s)")
        ax.set_xlabel("Time (s)")
        ax.legend()
        plt.show(block=True)


if __name__ == "__main__":
    main()
    print("Done!")
