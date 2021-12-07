"""
Script for demonstrating parallelization of FASTSim.  
Optional positional arguments:
    - processes: int, number of processes
"""

import multiprocessing as mp
import time
import datetime
import sys
import numpy as np

import fastsim as fsim

veh = fsim.vehicle.Vehicle(11, verbose=False)
# contrived cycles for this example
cycs = [fsim.cycle.Cycle('udds')] * 1_000

def run_sd(cyc_num: int):
    # convert to jit inside the parallelized function
    cyc_jit = cycs[cyc_num].get_numba_cyc()
    veh_jit = veh.get_numba_veh()
    # this example is not using simdrivehot, but the changes should be miminal
    sd = fsim.simdrive.SimDriveJit(cyc_jit, veh_jit)
    sd.sim_drive()
    if cyc_num % 100 == 0:
        print(f"Finished cycle {cyc_num}")
    
    return sd.mpgge

if __name__ == '__main__':
    print(datetime.datetime.now())
    t0 = time.time()

    assert len(sys.argv) <= 2, 'Expected at most 2 arguments.'

    if len(sys.argv) == 2:
        processes = int(sys.argv[1])
    else:
        processes = 4 # results in 250 runs of udds per process, which appears to be optimal

    print(f"Running with {processes} pool processes.")
    with mp.Pool(processes=processes) as pool:
        mpgges = pool.map(run_sd, np.arange(len(cycs)))
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.5g}")

