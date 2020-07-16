import sys
import os
from pathlib import Path
# allow it to find simdrive module
fsimpath=str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)
import numpy as np
import time
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import importlib

# local modules
from fastsim import simdrive, vehicle, cycle
from fastsim import globalvars as gl

# importlib.reload(simdrive)

t0 = time.time()
try: # if the cycle has already been loaded, just use it
    cyc = cycle.Cycle(cyc_dict={'cycSecs': cyc_df['cycSecs'].values.copy(),
                                'cycMps': cyc_df['cycMps'].values.copy(),
                                'cycGrade': cyc_df['grade_decimalOfPercent'].values.copy()})
    cyc_jit = cyc.get_numba_cyc()
except: # if that fails, load it
    cyc_df = pd.read_csv(
        r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\longHaulDriveCycle (1).csv')
    cyc_df.drop(columns=['Unnamed: 0'], inplace=True)
    cyc_df = cyc_df.iloc[200:16_104]
    cyc_df.reset_index(inplace=True)
    cyc_df['TimeStamp'] = pd.to_datetime(cyc_df['TimeStamp'])
    cyc_df['cycSecs'] = (cyc_df['TimeStamp'] - cyc_df.loc[0, 'TimeStamp']).dt.total_seconds()
    cyc_df['cycMps'] = cyc_df['Speed_Mph'] / gl.mphPerMps
    cyc_df['delta elevation [m]'] = (cyc_df['elevation_Feet'] - \
        cyc_df.iloc[0]['elevation_Feet']) / 3.28

cyc = cycle.Cycle(cyc_dict={'cycSecs': cyc_df['cycSecs'].values.copy(),
                            'cycMps': cyc_df['cycMps'].values.copy(),
                            'cycGrade': cyc_df['grade_decimalOfPercent'].values.copy()})
cyc_jit = cyc.get_numba_cyc()

print('Time to load cycle: {:.3f} s'.format(time.time() - t0))

t0 = time.time()
veh = vehicle.Vehicle(1)
veh.vehKg = 15e3
veh_jit = veh.get_numba_veh()
print('Time to load vehicle: {:.3f} s'.format(time.time() - t0))

t0 = time.time()

sim_drive_params = simdrive.SimDriveParams(missed_trace_correction=True)
sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit, sim_drive_params)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit, sim_drive_params) 
# cyc_jit is necessary even for SimDriveClassic to get correct behavior 
# in overriding self.cyc.secs

sim_drive.sim_drive() 

print('Time to run sim_drive: {:.3f} s'.format(time.time() - t0))

# elevation delta based on dilated cycle secs
delta_elev_dilated = (sim_drive.cyc.cycGrade * sim_drive.cyc.secs * sim_drive.cyc.cycMps).sum()
# elevation delta based on dilated cycle secs
delta_elev_achieved = (sim_drive.cyc.cycGrade *
                      sim_drive.cyc.secs * sim_drive.mpsAch).sum()

# PLOTS

# speed

plt.plot(cyc.cycSecs, cyc.cycMps, label='base')
plt.plot(sim_drive.cyc.cycSecs, sim_drive.mpsAch,
         label='dilated', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [mps]')
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.svg')
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.png')

plt.figure()
plt.plot(cyc.cycMps, label='base')
plt.plot(sim_drive.mpsAch, label='dilated', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Speed [mps]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v index.png')

# distance

plt.figure()
plt.plot(cyc.cycSecs, (cyc.cycMps * cyc.secs).cumsum() / 1e3, label='base')
plt.plot(sim_drive.cyc.cycSecs, (sim_drive.mpsAch *
                                 sim_drive.cyc.secs).cumsum() / 1e3, label='dilated', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v time.png')

plt.figure()
plt.plot((cyc.cycMps * cyc.secs).cumsum() / 1e3, label='base')
plt.plot((sim_drive.mpsAch * sim_drive.cyc.secs).cumsum() / 1e3,
         label='dilated', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance [km]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist v index.png')

plt.figure()
plt.plot(sim_drive.cyc.cycSecs,
    (np.interp(
    sim_drive.cyc.cycSecs, 
    cyc.cycSecs, 
    cyc.cycDistMeters.cumsum()) - sim_drive.distMeters.cumsum())
         / 1e3, label='base')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance (trace - achieved) [km]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.png')

plt.figure()
plt.plot((cyc.cycDistMeters.cumsum() -
         sim_drive.distMeters.cumsum()) / 1e3, label='base')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance (trace - achieved) [km]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v index.png')

# elevation change

plt.figure()
plt.plot(cyc.cycSecs, (cyc.cycGrade * cyc.cycMps * cyc.secs).cumsum(), label='base')
plt.plot(sim_drive.cyc.cycSecs, (cyc.cycGrade * cyc.secs *
                                 sim_drive.mpsAch).cumsum(), label='undilated', linestyle='--')
plt.plot(sim_drive.cyc.cycSecs, (sim_drive.cyc.cycGrade * sim_drive.cyc.secs *
                                 sim_drive.mpsAch).cumsum(), label='achieved', linestyle='-.')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Delta Elevation [m]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v time.png')


plt.figure()
plt.plot((cyc.cycGrade * cyc.cycMps *
                       cyc.secs).cumsum(), label='base')
plt.plot((cyc.cycGrade * cyc.secs * sim_drive.mpsAch).cumsum(), label='undilated', linestyle='--')
plt.plot((sim_drive.cyc.cycGrade * sim_drive.cyc.secs *
                                 sim_drive.mpsAch).cumsum(), label='achieved', linestyle='-.')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Delta Elevation [m]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.png')

# grade

plt.figure()
plt.plot(cyc.cycSecs, cyc.cycGrade, label='base')
plt.plot(sim_drive.cyc.cycSecs, sim_drive.cyc.cycGrade, label='achieved', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Grade [-]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.png')

plt.figure()
plt.plot(cyc.cycGrade, label='base')
plt.plot(sim_drive.cyc.cycGrade, label='achieved', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Grade [-]')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.png')

# time dilation

plt.figure()
plt.plot(sim_drive.cyc.secs)
plt.grid()
plt.xlabel('Index')
plt.ylabel('Time Dilation')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.png')

