"""A module that demonstrates trace-miss correction."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import fastsim as fsim

sns.set_theme()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"


def trace_miss_demo():
    """Run a vehicle over a cycle with a trace miss + correction"""
    cyc_d = {
        "time_seconds": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "speed_meters_per_second": [0.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0],
    }
    cyc = fsim.Cycle.from_pydict(cyc_d)
    cyc0 = cyc.copy()
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(1)
    params = fsim.SimParams.default().to_pydict()
    # Set the trace miss option to use.
    # NOTE: can also choose "Allow". However, 'Allow' will not attempt to
    # re-rendezvous with the reference trace whereas 'Correct' will.
    params["trace_miss_opts"] = "Correct"  # "Allow"
    # This is the maximum number of time steps with which to re-rendezvous with
    # the reference trace. The trajectory with the "gentlest" acceleration will
    # be chosen up to the maximum number of time steps.
    params["trace_miss_correct_max_steps"] = 6
    sd = fsim.SimDrive(veh, cyc, fsim.SimParams.from_pydict(params))
    sd.walk()
    if SHOW_PLOTS:
        c0 = cyc0.to_pydict()
        df = sd.to_dataframe()
        fig, ax = plt.subplots()
        ax.plot(
            np.array(c0["time_seconds"]),
            np.array(c0["speed_meters_per_second"]), "k-", label="original")
        ax.plot(
            np.array(df["cyc.time_seconds"]),
            np.array(df["veh.history.speed_ach_meters_per_second"]),
            "b:", label="modified")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Speed [m/s]")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)
        # By Distance
        fig, ax = plt.subplots()
        ax.plot(
            np.array(c0["dist_meters"]),
            np.array(c0["speed_meters_per_second"]), "k-", label="original")
        ax.plot(
            np.array(df["cyc.dist_meters"]),
            np.array(df["veh.history.speed_ach_meters_per_second"]),
            "b:", label="modified")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Speed [m/s]")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)
        # Time/Distance Plot
        fig, ax = plt.subplots()
        ax.plot(
            np.array(c0["time_seconds"]),
            np.array(c0["dist_meters"]), "k-", label="original")
        ax.plot(
            np.array(df["cyc.time_seconds"]),
            np.array(df["cyc.dist_meters"]), "b:", label="modified")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")
        ax.legend()
        fig.tight_layout()
        plt.show(block=True)


if __name__ == "__main__":
    trace_miss_demo()
    print("Done!")
