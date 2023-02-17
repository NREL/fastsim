"""
Script for demonstrating parallelization of FASTSim.  
Optional positional arguments:
    - processes: int, number of processes
"""

import multiprocessing as mp
import time
import datetime
import warnings
import numpy as np
import argparse

import fastsim as fsim

veh = fsim.vehicle.Vehicle.from_vehdb(11)
# contrived cycles for this example
cycs = [fsim.cycle.Cycle.from_file('udds')] * 1_000

def run_sd(cyc_num: int):
    # convert to compiled version inside the parallelized function to allow for pickling
    cyc = cycs[cyc_num]
    sd = fsim.simdrive.RustSimDrive(cyc.to_rust(), veh.to_rust())
    sd.sim_drive()
    if cyc_num % 100 == 0:
        print(f"Finished cycle {cyc_num}")
    
    return sd.mpgge

parser = argparse.ArgumentParser(description='Parallelized FASTSim demo')
parser.add_argument('-p', '--processes', type=int, default=4)

if __name__ == '__main__':
    print(datetime.datetime.now())
    t0 = time.time()

    # processes = 4 results in 250 runs of udds per process, which appears to be optimal
    # this is equivalent to 250 runs * 1,400 s / run = 350,000 s per process
    args, unknown = parser.parse_known_args()
    processes = args.processes
    if len(unknown) > 0:
        warnings.warn(f"Unknown arguments: {unknown}")

    print(f"Running with {processes} pool processes.")
    with mp.Pool(processes=processes) as pool:
        mpgges = pool.map(run_sd, np.arange(len(cycs)))
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.5g}")

