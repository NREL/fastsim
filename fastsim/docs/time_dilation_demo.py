# %%

import sys
import os
from pathlib import Path
import numpy as np
import time
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
sns.set()

# local modules
from fastsim import simdrive, vehicle, cycle
from fastsim import parameters as params

# importlib.reload(simdrive)

# %%

t0 = time.time()
cyc = cycle.Cycle.from_dict(cyc_dict=cycle.clip_by_times(
   cycle.Cycle.from_file('longHaulDriveCycle').get_cyc_dict(),
    t_end=18_000, t_start=1_800))
print('Time to load cycle file: {:.3f} s'.format(time.time() - t0))


t0 = time.time()
veh = vehicle.Vehicle.from_file('Line_Haul_Conv.csv')
veh.veh_kg *= 2
print('Time to load vehicle: {:.3f} s'.format(time.time() - t0))


t0 = time.time()

sd_fixed = simdrive.SimDrive(cyc, veh)
sim_params = sd_fixed.sim_params
sim_params.missed_trace_correction=True
# sim_params.min_time_dilation = 1
sim_params.max_time_dilation = 0.1
# sim_params.time_dilation_tol = 1e-1

sd_base = simdrive.SimDrive(cyc, veh)

sd_fixed.sim_drive() 
sd_base.sim_drive()

t_delta = time.time() - t0

print('Time to run sim_drive: {:.3f} s'.format(t_delta))
print('Mean number of trace miss iterations: {:.3f}'.format(sd_fixed.trace_miss_iters.mean()))
print('Distance percent error w.r.t. base cycle: {:.3%}'.format(
    (sd_fixed.dist_m.sum() - cyc.dist_m.sum()) / cyc.dist_m.sum()))

# elevation delta based on dilated cycle secs
delta_elev_dilated = (sd_fixed.cyc.grade * sd_fixed.cyc.dt_s * sd_fixed.cyc.mps).sum()
# elevation delta based on dilated cycle secs
delta_elev_achieved = (sd_fixed.cyc.grade *
                      sd_fixed.cyc.dt_s * sd_fixed.mps_ach).sum()

# PLOTS

# speed

plt.plot(cyc.time_s, cyc.mps, label='trace')
plt.plot(sd_fixed.cyc.time_s, sd_fixed.mps_ach,
         label='dilated', linestyle='--')
# plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Time, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

plt.figure()
plt.plot(cyc.mps, label='trace')
plt.plot(sd_fixed.mps_ach, label='dilated', linestyle='--')
# plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Index, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

# distance

plt.figure()
plt.plot(cyc.time_s, (cyc.mps * cyc.dt_s).cumsum() / 1e3, label='trace')
plt.plot(sd_fixed.cyc.time_s, (sd_fixed.mps_ach *
                                 sd_fixed.cyc.dt_s).cumsum() / 1e3, label='dilated', linestyle='--')
plt.plot(sd_base.cyc.time_s, (sd_base.mps_ach *
                                 sd_base.cyc.dt_s).cumsum() / 1e3, label='base', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.title('Distance v. Time, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

plt.figure()
plt.plot((cyc.mps * cyc.dt_s).cumsum() / 1e3, label='trace')
plt.plot((sd_fixed.mps_ach * sd_fixed.cyc.dt_s).cumsum() / 1e3,
         label='dilated', linestyle='--')
plt.plot((sd_base.mps_ach * sd_base.cyc.dt_s).cumsum() / 1e3,
         label='base', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance [km]')
plt.title('Distance v. Index, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

plt.figure()
plt.plot(sd_fixed.cyc.time_s,
    (np.interp(
    sd_fixed.cyc.time_s, 
    cyc.time_s, 
    cyc.dist_m.cumsum()) - sd_fixed.dist_m.cumsum())
         / 1e3)
# plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Distance (trace - achieved) [km]')
plt.title('Trace Miss v. Time, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.tight_layout()
plt.show()

plt.figure()
plt.plot((cyc.dist_m.cumsum() -
         sd_fixed.dist_m.cumsum()))
# plt.grid()
plt.xlabel('Index')
plt.ylabel('Distance (trace - achieved) [m]')
plt.title('Trace Miss v. Index, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.tight_layout()
plt.show()

# elevation change

plt.figure()
plt.plot(cyc.time_s, (cyc.grade * cyc.mps * cyc.dt_s).cumsum(), label='trace')
plt.plot(sd_fixed.cyc.time_s, (cyc.grade * cyc.dt_s *
                                 sd_fixed.mps_ach).cumsum(), label='undilated', linestyle='--')
plt.plot(sd_fixed.cyc.time_s, (sd_fixed.cyc.grade * sd_fixed.cyc.dt_s *
                                 sd_fixed.mps_ach).cumsum(), label='dilated', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Time, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()


plt.figure()
plt.plot((cyc.grade * cyc.mps *
                       cyc.dt_s).cumsum(), label='trace')
plt.plot((cyc.grade * cyc.dt_s * sd_fixed.mps_ach).cumsum(), label='undilated', linestyle='--')
plt.plot((sd_fixed.cyc.grade * sd_fixed.cyc.dt_s *
                                 sd_fixed.mps_ach).cumsum(), label='dilated', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Index, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

# grade

plt.figure()
plt.plot(cyc.time_s, cyc.grade, label='trace')
plt.plot(sd_fixed.cyc.time_s, sd_fixed.cyc.grade, label='dilated', linestyle='--')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Grade [-]')
plt.title('Grade v. Time, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

plt.figure()
plt.plot(cyc.grade, label='trace')
plt.plot(sd_fixed.cyc.grade, label='dilated', linestyle='--')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Grade [-]')
plt.title('Grade v. Index, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()

# time dilation

plt.figure()
plt.plot(sd_fixed.cyc.dt_s)
# plt.grid()
plt.xlabel('Index')
plt.ylabel('Time Dilation')
plt.title('Time Dilation, veh wt = {:,.0f} lbs'.format(round(veh.veh_kg * 2.205 / 1000) * 1000))
plt.show()
# %%
