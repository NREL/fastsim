# Demonstrate the use of acceleration test

import sys
import os
import numpy as np

from fastsim import simdrive, vehicle, cycle
from fastsim import parameters as params

def create_accel_cyc(length_in_seconds=300, spd_mph=89.48, grade=0.0, hz=10):
    """
    Create a synthetic Drive Cycle for acceleration targeting.
    Defaults to a 15 second acceleration cycle. Should be adjusted based on target acceleration time
    and initial vehicle acceleration time, so that time isn't wasted on cycles that are needlessly long.

    spd_mph @ 89.48 FASTSim XL version mph default speed for acceleration cycles
    grade @ 0 and hz @ 10 also matches XL version settings
    """

    mps = np.array([(1 / params.MPH_PER_MPS) * float(spd_mph)] * (length_in_seconds * hz))
    mps[0] = 0.
    mps = np.array(mps)
    time_s = np.arange(0, length_in_seconds, 1. / hz)
    grade = np.array([float(grade)] * (length_in_seconds * hz))
    road_type = np.zeros(length_in_seconds * hz)
    cyc = {'mps': mps, 'time_s': time_s, 'grade': grade, 'road_type':road_type}
    return cyc

def main():
    """
    Arguments:
    ----------
    """

    # just use first vehicle in default database
    for i in range(1,27):
        veh = vehicle.Vehicle.from_vehdb(i).to_rust()
        accel_cyc = cycle.Cycle.from_dict(cyc_dict=create_accel_cyc()).to_rust()
        sd_accel = simdrive.RustSimDrive(accel_cyc, veh)
        
        simdrive.run_simdrive_for_accel_test(sd_accel)
        if (np.array(sd_accel.mph_ach) >= 60).any():
                net_accel = np.interp(
                    x=60, xp=sd_accel.mph_ach, fp=sd_accel.cyc.time_s)
        else:
            # in case vehicle never exceeds 60 mph, penalize it a lot with a high number
            print(veh.scenario_name + ' never achieves 60 mph.')
            net_accel = 1e3        
        
        print('vehicle {}: acceleration [s] {:.3f}'.format(i, net_accel))

if __name__=='__main__':
    main()
