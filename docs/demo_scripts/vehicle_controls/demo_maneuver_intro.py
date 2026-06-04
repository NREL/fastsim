"""
---
execute:
  skip: true
---

# CAVS: Intro to Maneuvers

The `Maneuver` struct applies Connected Automated Vehicle (CAV) maneuvers
to a drive cycle and returns the modified cycle. This demo walks through
the core Maneuver workflow.
"""

# %%
import os

import matplotlib.pyplot as plt
import seaborn as sns

import fastsim as fsim

# %%
sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"

"""
## Creating a Maneuver

`Maneuver.create_from` creates a Maneuver from a cycle and vehicle.
It copies chassis data from the vehicle and solver settings from
`SimParams` defaults.
"""

# %%
veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
veh.set_save_interval(1)

cyc = fsim.Cycle.from_resource("udds.csv")
cyc0 = cyc.copy()

man = fsim.Maneuver.create_from(cyc, veh.copy())

"""
## Inspecting and Modifying Parameters

`to_pydict()` serializes the Maneuver to a Python dictionary.
Parameters can be modified in the dictionary and then loaded back
with `Maneuver.from_pydict()`.

The Maneuver struct has two groups of configurable parameters:

**Coasting** (`coast_*`): controls whether and how coasting is applied
to the cycle.

**IDM** (`idm_*`): Intelligent Driver Model, Adaptive Cruise Control
version.
"""

# %%
d = man.to_pydict()

print("Coasting parameters:")
for k in sorted(d):
    if k.startswith("coast"):
        print(f"  {k}: {d[k]}")

print("\nIDM parameters:")
for k in sorted(d):
    if k.startswith("idm"):
        print(f"  {k}: {d[k]}")

"""
## Applying Maneuvers

Enable coasting and call `apply_maneuvers()` to get a modified cycle.
The returned `Cycle` can then be passed to `SimDrive` for simulation.
"""

# %%
d["coast_allow"] = True
d["coast_start_speed_meters_per_second"] = 15.0
man = fsim.Maneuver.from_pydict(d)

cyc_modified = man.apply_maneuvers()

"""
## Comparing Original and Modified Cycles

The modified cycle deviates from the original where coasting is applied.
"""

# %%
if SHOW_PLOTS:
    c0 = cyc0.to_pydict()
    cm = cyc_modified.to_pydict()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(c0["time_seconds"], c0["speed_meters_per_second"], label="Original")
    ax.plot(cm["time_seconds"], cm["speed_meters_per_second"], linestyle="--", label="Modified")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Original vs. Modified Cycle")
    ax.legend()
    plt.tight_layout()
    plt.show()

"""
## Simulating with the Modified Cycle
"""

# %%
sd = fsim.SimDrive(veh, cyc_modified)
sd.walk()

# %%
if SHOW_PLOTS:
    df = sd.to_dataframe()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(c0["time_seconds"], c0["speed_meters_per_second"], label="Original Cycle")
    ax.plot(
        df["cyc.time_seconds"],
        df["veh.history.speed_ach_meters_per_second"],
        linestyle="--",
        label="Achieved Speed",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Achieved Speed with Coasting Maneuver")
    ax.legend()
    plt.tight_layout()
    plt.show()
