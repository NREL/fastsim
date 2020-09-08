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

t0 = time.time()
cyc = cycle.Cycle(cyc_dict=cycle.clip_by_times(
    cycle.Cycle(cyc_file_path=Path('../cycles/longHaulDriveCycle.csv')).get_cyc_dict(),
    t_start=1_800, t_end=18_000))
cyc_jit = cyc.get_numba_cyc()
print('Time to load cycle file: {:.3f} s'.format(time.time() - t0))


t0 = time.time()
# veh = vehicle.Vehicle(26)
veh = vehicle.Vehicle(veh_file=Path('../vehdb/Line Haul Conv.csv'))
veh.vehKg *= 2
veh_jit = veh.get_numba_veh()
print('Time to load vehicle: {:.3f} s'.format(time.time() - t0))


t0 = time.time()

sim_drive_params = simdrive.SimDriveParams(missed_trace_correction=True)
sim_drive_params.min_time_dilation = 1
sim_drive_params.time_dilation_tol = 1e-1
sd_fixed = simdrive.SimDriveJit(cyc_jit, veh_jit, sim_drive_params)
sd_base = simdrive.SimDriveJit(cyc_jit, veh_jit)

sd_fixed.sim_drive() 
sd_base.sim_drive()

t_delta = time.time() - t0

print('Time to run sim_drive: {:.3f} s'.format(t_delta))
print('Mean number of trace miss iterations: {:.3f}'.format(sd_fixed.trace_miss_iters.mean()))
print('Distance percent error w.r.t. base cycle: {:.3%}'.format(
    (sd_fixed.distMeters.sum() - cyc.cycDistMeters.sum()) / cyc.cycDistMeters.sum()))

# elevation delta based on dilated cycle secs
delta_elev_dilated = (sd_fixed.cyc.cycGrade * sd_fixed.cyc.secs * sd_fixed.cyc.cycMps).sum()
# elevation delta based on dilated cycle secs
delta_elev_achieved = (sd_fixed.cyc.cycGrade *
                      sd_fixed.cyc.secs * sd_fixed.mpsAch).sum()

# PLOTS

# speed

plt.plot(cyc.cycSecs, cyc.cycMps, label='trace')
plt.plot(sd_fixed.cyc.cycSecs, sd_fixed.mpsAch,
         label='dilated', linestyle='--')
# plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Time, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.svg')
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.png')
plt.show()

plt.figure()
plt.plot(cyc.cycMps, label='trace')
plt.plot(sd_fixed.mpsAch, label='dilated', linestyle='--')
# plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Index, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v index.png')
plt.show()

# distance

plt.figure()
plt.plot(cyc.cycSecs, (cyc.cycMps * cyc.secs).cumsum() / 1e3, label='trace')
plt.plot(sd_fixed.cyc.cycSecs, (sd_fixed.mpsAch *
                                 sd_fixed.cyc.secs).cumsum() / 1e3, label='dilated', linestyle='--')
plt.plot(sd_base.cyc.cycSecs, (sd_base.mpsAch *
                                 sd_base.cyc.secs).cumsum() / 1e3, label='base', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.title('Distance v. Time, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v time.png')
plt.show()

plt.figure()
plt.plot((cyc.cycMps * cyc.secs).cumsum() / 1e3, label='trace')
plt.plot((sd_fixed.mpsAch * sd_fixed.cyc.secs).cumsum() / 1e3,
         label='dilated', linestyle='--')
plt.plot((sd_base.mpsAch * sd_base.cyc.secs).cumsum() / 1e3,
         label='base', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance [km]')
plt.title('Distance v. Index, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v index.png')
plt.show()

plt.figure()
plt.plot(sd_fixed.cyc.cycSecs,
    (np.interp(
    sd_fixed.cyc.cycSecs, 
    cyc.cycSecs, 
    cyc.cycDistMeters.cumsum()) - sd_fixed.distMeters.cumsum())
         / 1e3)
# plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Distance (trace - achieved) [km]')
plt.title('Trace Miss v. Time, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.tight_layout()
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.png')
plt.show()

plt.figure()
plt.plot((cyc.cycDistMeters.cumsum() -
         sd_fixed.distMeters.cumsum()))
# plt.grid()
plt.xlabel('Index')
plt.ylabel('Distance (trace - achieved) [m]')
plt.title('Trace Miss v. Index, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.tight_layout()
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v index.png')
plt.show()

# elevation change

plt.figure()
plt.plot(cyc.cycSecs, (cyc.cycGrade * cyc.cycMps * cyc.secs).cumsum(), label='trace')
plt.plot(sd_fixed.cyc.cycSecs, (cyc.cycGrade * cyc.secs *
                                 sd_fixed.mpsAch).cumsum(), label='undilated', linestyle='--')
plt.plot(sd_fixed.cyc.cycSecs, (sd_fixed.cyc.cycGrade * sd_fixed.cyc.secs *
                                 sd_fixed.mpsAch).cumsum(), label='dilated', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Time, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v time.png')
plt.show()


plt.figure()
plt.plot((cyc.cycGrade * cyc.cycMps *
                       cyc.secs).cumsum(), label='trace')
plt.plot((cyc.cycGrade * cyc.secs * sd_fixed.mpsAch).cumsum(), label='undilated', linestyle='--')
plt.plot((sd_fixed.cyc.cycGrade * sd_fixed.cyc.secs *
                                 sd_fixed.mpsAch).cumsum(), label='dilated', linestyle='-.')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Index, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.png')
plt.show()

# grade

plt.figure()
plt.plot(cyc.cycSecs, cyc.cycGrade, label='trace')
plt.plot(sd_fixed.cyc.cycSecs, sd_fixed.cyc.cycGrade, label='dilated', linestyle='--')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Grade [-]')
plt.title('Grade v. Time, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.png')
plt.show()

plt.figure()
plt.plot(cyc.cycGrade, label='trace')
plt.plot(sd_fixed.cyc.cycGrade, label='dilated', linestyle='--')
# plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Grade [-]')
plt.title('Grade v. Index, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.png')
plt.show()

# time dilation

plt.figure()
plt.plot(sd_fixed.cyc.secs)
# plt.grid()
plt.xlabel('Index')
plt.ylabel('Time Dilation')
plt.title('Time Dilation, veh wt = {:,.0f} lbs'.format(round(veh.vehKg * 2.205 / 1000) * 1000))
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.png')
plt.show()


