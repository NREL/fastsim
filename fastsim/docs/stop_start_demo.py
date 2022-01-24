# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Notebook for assessing stop/start and dfco impacts

# %%
import sys
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import seaborn as sns

sns.set()


# %%
# local modules
from fastsim import simdrive, cycle, vehicle


# %%
t0 = time.time()
# cyc = cycle.Cycle(cyc_dict=
#                   cycle.clip_by_times(cycle.Cycle("udds").get_cyc_dict(), 130))
cyc = cycle.Cycle('udds').get_cyc_dict()
cyc = cycle.Cycle(cyc_dict=cycle.clip_by_times(cyc, 130))
cyc_jit = cyc.get_numba_cyc()
print(f"Elapsed time: {time.time() - t0:.3e} s")


# %%
t0 = time.time()
vehno = 1
veh0 = vehicle.Vehicle(vehno).get_numba_veh()
print(f"Elapsed time: {time.time() - t0:.3e} s")


# %%
t0 = time.time()
# veh1 = vehicle.Vehicle(28).get_numba_veh()
veh1 = vehicle.Vehicle(vehno)
veh1.stopStart = True
veh1.maxMotorKw = 1
veh1.maxEssKw = 5
veh1.maxEssKwh = 1
veh1.set_init_calcs()
veh1.vehKg = veh0.vehKg
veh1 = veh1.get_numba_veh()
print(f"Elapsed time: {time.time() - t0:.3e} s")


# %%
t0 = time.time()
sim_drive0 = simdrive.SimDriveJit(cyc_jit, veh0)
sim_drive0.sim_drive()
sim_drive1 = simdrive.SimDriveJit(cyc_jit, veh1)
sim_drive1.sim_drive()
print(f"Elapsed time: {time.time() - t0:.3e} s")


# %%
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9,5))
ax0.plot(cyc.cycSecs, sim_drive0.fcKwInAch, 
         label='base')
ax0.plot(cyc.cycSecs, sim_drive1.fcKwInAch, 
         label='stop-start', linestyle='--')
# ax.plot(cyc.cycSecs, dfco_fcKwOutAchPos, label='dfco', linestyle='--', color='blue')
ax0.legend(loc='upper left')
ax0.set_ylabel('Fuel Power [kW]')

ax2 = ax1.twinx()
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.plot(cyc.cycSecs, sim_drive1.canPowerAllElectrically, 
        color='red')
ax2.set_ylabel('SS active')
ax2.grid()

ax1.plot(cyc.cycSecs, cyc.cycMph)
ax1.yaxis.label.set_color('blue')
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel('Speed [mph]')
ax1.set_ylim([0, 35])
ax1.set_xlabel('Time [s]')

# %%
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9,5))
ax0.plot(cyc.cycSecs, (sim_drive0.fcKwInAch * cyc.secs).cumsum() / 1e3, 
         label='base')
ax0.plot(cyc.cycSecs, (sim_drive1.fcKwInAch * cyc.secs).cumsum() / 1e3, 
         label='stop-start')
ax0.legend(loc='upper left')
ax0.set_ylabel('Fuel Energy [MJ]')

ax2 = ax1.twinx()
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.plot(cyc.cycSecs, sim_drive1.canPowerAllElectrically, 
        color='red', alpha=0.25)
ax2.set_ylabel('SS active')
ax2.set_xlim(ax0.get_xlim())
ax2.set_yticks([0, 1])
ax2.grid()

ax1.plot(cyc.cycSecs, cyc.cycMph)
ax1.yaxis.label.set_color('blue')
ax1.tick_params(axis='y', colors='blue')
ax1.set_ylabel('Speed [mph]')
ax1.set_xlabel('Time [s]')

diff = ((sim_drive0.fcKwOutAch * cyc.secs).sum() - 
    (sim_drive1.fcKwOutAch * cyc.secs).sum()) / (
    sim_drive0.fcKwOutAch * cyc.secs).sum()

print(f'Stop/start produces a {diff:.2%} reduction in fuel consumption.\n')
# %%
