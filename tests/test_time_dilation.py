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

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(time.time() - t0)

t0 = time.time()
veh = vehicle.Vehicle(1)
veh_jit = veh.get_numba_veh()
print(time.time() - t0)

t0 = time.time()

sim_drive_params = simdrive.SimDriveParams(missed_trace_correction=True)
veh_jit.vehKg = 10e3
veh.vehKg = 10e3
# sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit, sim_drive_params)
sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit, sim_drive_params)

sim_drive.sim_drive() 

print(time.time() - t0) 

plt.plot(cyc.cycSecs, cyc.cycMps, label='base')
plt.plot(sim_drive.cyc.cycSecs, sim_drive.mpsAch,
         label='dilated', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel('Time [s]\nWhat is time, anyway? Just a human construct.')
plt.ylabel('Speed [mps]')

plt.figure()
plt.plot(cyc.cycSecs, (cyc.cycMps * cyc.secs).cumsum(), label='base')
plt.plot(sim_drive.cyc.cycSecs, (sim_drive.mpsAch *
                                 sim_drive.cyc.secs).cumsum(), label='dilated', linestyle='--')
plt.grid()
plt.legend()
plt.xlabel('Time [s]\nWhat is time, anyway? Just a human construct.')
plt.ylabel('Distance [m]')
