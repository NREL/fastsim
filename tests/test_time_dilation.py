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
from fastsim import parameters as params

# importlib.reload(simdrive)

t0 = time.time()
try: # if the cycle has already been loaded, just use it
    cyc = cycle.Cycle(cyc_dict={'cycSecs': cyc_df['cycSecs'].values.copy(),
                                'cycMps': cyc_df['cycMps'].values.copy(),
                                'cycGrade': cyc_df['grade_decimalOfPercent'].values.copy()})
    cyc_jit = cyc.get_numba_cyc()
except: # if that fails, load it
    cyc_df = pd.read_csv(os.path.join('..', 'cycles', 'longHaulDriveCycle.csv'))
    cyc_df.drop(columns=['Unnamed: 0'], inplace=True)
    # cyc_df = cyc_df.iloc[200:16_104]
    # cyc_df.reset_index(inplace=True)
    cyc_df['TimeStamp'] = pd.to_datetime(cyc_df['TimeStamp'])
    cyc_df['cycSecs'] = (cyc_df['TimeStamp'] - cyc_df.loc[0, 'TimeStamp']).dt.total_seconds()
    cyc_df['cycMps'] = cyc_df['Speed_Mph'] / params.mphPerMps
    cyc_df['delta elevation [m]'] = (cyc_df['elevation_Feet'] - \
        cyc_df.iloc[0]['elevation_Feet']) / 3.28

print('Time to load cycle file: {:.3f} s'.format(time.time() - t0))

cyc = cycle.Cycle(cyc_dict={'cycSecs': cyc_df.loc[:19e3, 'cycSecs'].values.copy(),
                            'cycMps': cyc_df.loc[:19e3, 'cycMps'].values.copy(),
                            'cycGrade': cyc_df.loc[:19e3, 'grade_decimalOfPercent'].values.copy()})
cyc_jit = cyc.get_numba_cyc()


t0 = time.time()
veh = vehicle.Vehicle(26)
veh.vehKg *= 4
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
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Time')
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.svg')
plt.savefig(r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\speed v time.png')
plt.show()

plt.figure()
plt.plot(cyc.cycMps, label='trace')
plt.plot(sd_fixed.mpsAch, label='dilated', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel('Index')
plt.ylabel('Speed [mps]')
plt.title('Speed v. Index')
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
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance [km]')
plt.title('Distance v. Time')
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
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance [km]')
plt.title('Distance v. Index')
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
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Distance (trace - achieved) [km]')
plt.title('Trace Miss v. Time')
plt.tight_layout()
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\dist diff v time.png')
plt.show()

plt.figure()
plt.plot((cyc.cycDistMeters.cumsum() -
         sd_fixed.distMeters.cumsum()))
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Distance (trace - achieved) [m]')
plt.title('Trace Miss v. Index')
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
                                 sd_fixed.mpsAch).cumsum(), label='achieved', linestyle='-.')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Time')
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
                                 sd_fixed.mpsAch).cumsum(), label='achieved', linestyle='-.')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Delta Elevation [m]')
plt.title('Delta Elev. v. Index')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\elev v index.png')
plt.show()

# grade

plt.figure()
plt.plot(cyc.cycSecs, cyc.cycGrade, label='trace')
plt.plot(sd_fixed.cyc.cycSecs, sd_fixed.cyc.cycGrade, label='achieved', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Grade [-]')
plt.title('Grade v. Time')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v time.png')
plt.show()

plt.figure()
plt.plot(cyc.cycGrade, label='trace')
plt.plot(sd_fixed.cyc.cycGrade, label='achieved', linestyle='--')
plt.grid()
plt.legend(loc='upper left')
plt.xlabel('Index')
plt.ylabel('Grade [-]')
plt.title('Grade v. Index')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\grade v index.png')
plt.show()

# time dilation

plt.figure()
plt.plot(sd_fixed.cyc.secs)
plt.grid()
plt.xlabel('Index')
plt.ylabel('Time Dilation')
plt.title('Time Dilation')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.svg')
plt.savefig(
    r'C:\Users\cbaker2\Documents\Projects\FASTSim\MDHD\plots\time dilation.png')
plt.show()


