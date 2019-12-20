"""Test script that times runs 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles."""

import time
import numpy as np
import sys
import importlib

sys.path.append('../src')

# local modules
import FASTSim
importlib.reload(FASTSim)

t0 = time.time()

cycles = ['udds', 'hwfet', 'us06']
vehicles = np.arange(1, 27)

iter = 0
for vehno in vehicles:
    print('vehno =', vehno)
    for cycname in cycles:
        cyc = FASTSim.get_standard_cycle(cycname)
        veh = FASTSim.get_veh(vehno)

        FASTSim.sim_drive(cyc, veh)

t1 = time.time()
print()
print('Elapsed time: ', round(t1 - t0, 2), 's')
