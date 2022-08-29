# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import fastsimrust as fsr
import fastsim as fsim
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from pathlib import Path
import os
import sys
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
sns.set()


# local modules


# %% Case with no cabin thermal modeling or HVAC

fusion = fsr.RustVehicle.from_file(
    str(fsim.vehicle.VEHICLE_DIR / "2012_Ford_Fusion.yaml"))
fusion_thermal_base = fsr.VehicleThermal.from_file(
    str(fsim.vehicle.VEHICLE_DIR / "thermal/2012_Ford_Fusion_thrml.yaml"))
cyc = fsr.RustCycle.from_file(str(fsim.cycle.CYCLES_DIR / "udds.csv"))

# no arguments use default of 22°C
init_thermal_state = fsr.ThermalState()

sdh = fsr.SimDriveHot(cyc, fusion, fusion_thermal_base, init_thermal_state)

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

# %% Case with cabin heating

hvac_model = fsr.HVACModel.default()
fusion_thermal_htng = fusion_thermal_base.copy()
fusion_thermal_htng.set_cabin_hvac_model_internal(hvac_model)
cyc = fsr.RustCycle.from_file(str(fsim.cycle.CYCLES_DIR / "udds.csv"))

init_thermal_state = fsr.ThermalState(amb_te_deg_c=-5.0)

sdh = fsr.SimDriveHot(cyc, fusion, fusion_thermal_htng, init_thermal_state)

t0 = time.perf_counter()
sdh.sim_drive()
t1 = time.perf_counter()

print(f"Elapsed time: {t1 - t0:.3g} s")

# %%

fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

plt.suptitle('Cold Start, Cold Ambient')
ax[0].plot(sdh.sd.cyc.time_s, sdh.history.fc_te_deg_c)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Engine\nTemp. [°C]")

ax[1].plot(sdh.sd.cyc.time_s, sdh.sd.fs_kw_out_ach)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Fuel Power [kW]")

ax[2].plot(sdh.sd.cyc.time_s, sdh.history.cab_te_deg_c)
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Cabin\nTemp. [°C]")

ax[3].plot(sdh.sd.cyc.time_s,
           sdh.history.cab_qdot_from_hvac_kw, label='to cabin')
ax[3].plot(sdh.sd.cyc.time_s, sdh.sd.aux_in_kw, label='aux')
ax[3].legend()
ax[3].set_xlabel("Time")
ax[3].set_ylabel("Climate Power [kW]")

ax[-1].plot(sdh.sd.cyc.time_s, sdh.sd.mph_ach)
ax[-1].set_xlabel("Time")
ax[-1].set_ylabel("Speed [mph]")
plt.tight_layout()
plt.savefig("plots/fusion udds cold start.png")
plt.savefig("plots/fusion udds cold start.svg")

# %% Case with cabin cooling

hvac_model = fsr.HVACModel.default()
fusion_thermal_clng = fusion_thermal_base.copy()
fusion_thermal_clng.set_cabin_hvac_model_internal(hvac_model)
cyc = fsr.RustCycle.from_file(str(fsim.cycle.CYCLES_DIR / "udds.csv"))

init_thermal_state = fsr.ThermalState(amb_te_deg_c=40.0)

sdh = fsr.SimDriveHot(cyc, fusion, fusion_thermal_clng, init_thermal_state)

t0 = time.perf_counter()
sdh.sim_drive()
t1 = time.perf_counter()

print(f"Elapsed time: {t1 - t0:.3g} s")

# %%

fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

plt.suptitle('Hot Start, Hot Ambient')
ax[0].plot(sdh.sd.cyc.time_s, sdh.history.fc_te_deg_c)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Engine\nTemp. [°C]")

ax[1].plot(sdh.sd.cyc.time_s, sdh.sd.fs_kw_out_ach)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Fuel Power [kW]")

ax[2].plot(sdh.sd.cyc.time_s, sdh.history.cab_te_deg_c)
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Cabin\nTemp. [°C]")

ax[3].plot(sdh.sd.cyc.time_s,
           sdh.history.cab_qdot_from_hvac_kw, label='to cabin')
ax[3].plot(sdh.sd.cyc.time_s, sdh.sd.aux_in_kw, label='aux')
ax[3].legend()
ax[3].set_xlabel("Time")
ax[3].set_ylabel("Climate Power [kW]")

ax[-1].plot(sdh.sd.cyc.time_s, sdh.sd.mph_ach)
ax[-1].set_xlabel("Time")
ax[-1].set_ylabel("Speed [mph]")
plt.tight_layout()
plt.savefig("plots/fusion udds hot start.png")
plt.savefig("plots/fusion udds hot start.svg")


# %% sweep ambient

hvac_model = fsr.HVACModel.default()
fusion_thermal_hvac = fusion_thermal_base.copy()
fusion_thermal_hvac.set_cabin_hvac_model_internal(hvac_model)
cyc = fsr.RustCycle.from_file(str(fsim.cycle.CYCLES_DIR / "udds.csv"))

mpg = []
mpg_no_hvac = []
amb_te_deg_c_arr = np.linspace(-15, 40, 100)

t0 = time.perf_counter()
for amb_te_deg_c in amb_te_deg_c_arr:
    init_thermal_state = fsr.ThermalState(amb_te_deg_c=amb_te_deg_c)
    sdh = fsr.SimDriveHot(cyc, fusion, fusion_thermal_hvac, init_thermal_state)
    sdh.sim_drive()
    mpg.append(sdh.sd.mpgge)
    sdh_no_hvac = fsr.SimDriveHot(cyc, fusion, fusion_thermal_base, init_thermal_state)
    sdh_no_hvac.sim_drive()
    mpg_no_hvac.append(sdh_no_hvac.sd.mpgge)

sdh_no_thrml = fsr.RustSimDrive(cyc, fusion)
sdh_no_thrml.sim_drive()

t1 = time.perf_counter()

print(f"Elapsed time: {t1 - t0:.3g} s")

# %%

colors = ['#7fc97f', '#beaed4', '#fdc086']

fig, ax = plt.subplots()
plt.suptitle('2012 Ford Fusion V6 FE v. Ambient/Init. Temp.')
ax.plot(amb_te_deg_c_arr, mpg, color=colors[0], label='w/ HVAC')
ax.plot(amb_te_deg_c_arr, mpg_no_hvac, color=colors[1], label='w/o HVAC')
ax.axhline(sdh_no_thrml.mpgge, color=colors[2], label='no thermal')
ax.legend()
ax.set_xlabel('Ambient/Init. Temp [°C]')
ax.set_ylabel('Fuel Economy [mpg]')
plt.tight_layout()
plt.savefig("plots/fusion FE vs temp sweep.png")
plt.savefig("plots/fusion FE vs temp sweep.svg")
# %%
