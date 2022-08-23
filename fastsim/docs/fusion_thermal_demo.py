# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import sys
import os
from pathlib import Path
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# local modules
import fastsim as fsim
import fastsimrust as fsr


# %% Case with no cabin thermal modeling or HVAC

fusion = fsr.RustVehicle.from_file(str(fsim.vehicle.VEHICLE_DIR / "2012_Ford_Fusion.yaml"))
fusion_thermal = fsr.VehicleThermal.from_file(str(fsim.vehicle.VEHICLE_DIR / "thermal/2012_Ford_Fusion_thrml.yaml"))
cyc = fsr.RustCycle.from_file(str(fsim.cycle.CYCLES_DIR / "udds.csv"))

# no arguments use default of 22°C
init_thermal_state = fsr.ThermalState()

sdh = fsr.SimDriveHot(cyc, fusion, fusion_thermal, init_thermal_state)

t0 = time.perf_counter()
sdh.sim_drive()
t1 = time.perf_counter()

print(f"Elapsed time: {t1 - t0:.3g} s")

# %% 

fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

ax[0].plot(sdh.sd.cyc.time_s, sdh.history.fc_te_deg_c)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Engine Temp. [°C]")

ax[1].plot(sdh.sd.cyc.time_s, sdh.sd.fs_kw_out_ach)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Fuel Power [kW]")

ax[-1].plot(sdh.sd.cyc.time_s, sdh.sd.mph_ach)
ax[-1].set_xlabel("Time")
ax[-1].set_ylabel("Speed [mph]")
