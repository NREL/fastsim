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

veh = fsim.vehicle.Vehicle.from_vehdb(11, verbose=False)
# contrived cycles for this example
cycs = [fsim.cycle.Cycle.from_file('udds')] * 1_000

def run_sd(cyc_num: int):
    # convert to compiled version inside the parallelized function to allow for pickling
    cyc = cycs[cyc_num]
    sd = fsim.simdrive.SimDriveClassic(cyc, veh)
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
        # processes = 4 results in 250 runs of udds per process, which appears to be optimal
        # this is equivalent to 250 runs * 1,400 s / run = 350,000 s per process
        processes = 4 

    print(f"Running with {processes} pool processes.")
    with mp.Pool(processes=processes) as pool:
        mpgges = pool.map(run_sd, np.arange(len(cycs)))
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.5g}")

