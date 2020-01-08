"""Test script that times runs 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles.
This is to be compared to the same script from tag:baseline (244758fac88e22e4565ed369ec422...)"""

import pandas as pd
import time
import numpy as np
import sys
import importlib

sys.path.append('../src')

# local modules
import SimDrive
importlib.reload(SimDrive)
import LoadData
importlib.reload(LoadData)

t0 = time.time()

cycles = ['udds', 'hwfet', 'us06']
vehicles = np.arange(1, 27)

print('Instantiating classes.')
print()
veh = LoadData.Vehicle(1)
cyc = LoadData.Cycle('udds')
sim_drive = SimDrive.SimDrive()

iter = 0
for vehno in vehicles:
    print('vehno =', vehno)
    for cycname in cycles:
        if not((vehno == 1) and (cycname == 'udds')):
            cyc.set_standard_cycle(cycname)
            veh.load_vnum(vehno)

        sim_drive.sim_drive(cyc, veh)
        sim_drive.get_diagnostics(cyc)

t1 = time.time()
print()
print('Elapsed time: ', round(t1 - t0, 2), 's')
