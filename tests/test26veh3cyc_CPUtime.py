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

from pathlib import Path
fsimpath = str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)

# local modules
from fastsim import simdrive, vehicle, cycle
importlib.reload(simdrive)

def run_test26veh3cyc_CPUtime(use_jitclass=True):
    t0 = time.time()

    cycles = ['udds', 'hwfet', 'us06']
    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()
    veh = vehicle.Vehicle(1)
    veh_jit = veh.get_numba_veh()
    cyc = cycle.Cycle('udds')
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
                if use_jitclass:
                    sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
                    sim_drive.sim_drive(-1)
                else:
                    sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
                    sim_drive.sim_drive()

    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')
    print('Elapsed time since first vehicle: ', round(t1 - t0a, 2), 's')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if re.match('(?i)true', sys.argv[1]):
            use_jitclass = True
            print('Using numba JIT compilation.')
        else:
            use_jitclass = False
            print('Skipping numba JIT compilation.')

        run_test26veh3cyc_CPUtime(use_jitclass=use_jitclass)
    else:
        print('Using numba JIT compilation.')
        run_test26veh3cyc_CPUtime()
