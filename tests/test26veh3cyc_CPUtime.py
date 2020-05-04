"""Test script that times runs 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles.
This is to be compared to the same script from tag:baseline (244758fac88e22e4565ed369ec422...)"""

import pandas as pd
import time
import numpy as np
import os
import sys
import importlib

from pathlib import Path
fsimpath = str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)

# local modules
from fastsim import SimDrive
importlib.reload(SimDrive)

t0 = time.time()

cycles = ['udds', 'hwfet', 'us06']
vehicles = np.arange(1, 27)

print('Instantiating classes.')
print()
veh = SimDrive.Vehicle(1)
veh_jit = veh.get_numba_veh()
cyc = SimDrive.Cycle('udds')
cyc_jit = cyc.get_numba_cyc()

iter = 0
for vehno in vehicles:
    print('vehno =', vehno)
    if vehno == 2:
        t0a = time.time()
    for cycname in cycles:
        if not((vehno == 1) and (cycname == 'udds')):
            cyc.set_standard_cycle(cycname)
            cyc_jit = cyc.get_numba_cyc()
            veh.load_veh(vehno)
            veh_jit = veh.get_numba_veh()
        sim_drive = SimDrive.SimDriveJit(cyc_jit, veh_jit)
        sim_drive.sim_drive(-1)

t1 = time.time()
print()
print('Elapsed time: ', round(t1 - t0, 2), 's')
print('Elapsed time since first vehicle: ', round(t1 - t0a, 2), 's')
