"""Test script that times runs 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles.
This is to be compared to the same script from tag:baseline (244758fac88e22e4565ed369ec422...).
From command line, pass True (default if left blank) or False argument to use JIT compilation or not, respectively."""

import pandas as pd
import time
import numpy as np
import re
import os
import sys
import importlib

# local modules
from fastsim import simdrive, vehicle, cycle
importlib.reload(simdrive)

def main(use_jitclass=True):
    t0 = time.time()

    cyc_names = ['udds', 'hwfet', 'us06']
    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()
    veh = vehicle.Vehicle(1)
    veh_jit = veh.get_numba_veh()
    
    cycs_jit = {cyc_name:cycle.Cycle(cyc_name).get_numba_cyc() for cyc_name in cyc_names}

    iter = 0
    for vehno in vehicles:
        print('vehno =', vehno)
        if vehno == 2:
            t0a = time.time()
        if not(vehno == 1):
            veh.load_veh(vehno)
            veh_jit = veh.get_numba_veh()
        print(veh.Scenario_name, '\n')

        for cyc_name in cyc_names:
            if use_jitclass:
                sim_drive = simdrive.SimDriveJit(cycs_jit[cyc_name], veh_jit)
                sim_drive.sim_drive()
            else:
                sim_drive = simdrive.SimDriveClassic(cycs_jit[cyc_name], veh_jit)
                sim_drive.sim_drive()

    t1 = time.time()
    print()
    print('Elapsed time: {:.2f} s'.format(t1 - t0))
    print('Elapsed time since first vehicle: {:.2f} s'.format(t1 - t0a, 2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if re.match('(?i)true', sys.argv[1]):
            use_jitclass = True
            print('Using numba JIT compilation.')
        else:
            use_jitclass = False
            print('Skipping numba JIT compilation.')

        main(use_jitclass=use_jitclass)
    else:
        print('Using numba JIT compilation.')
        main()
