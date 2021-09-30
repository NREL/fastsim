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

    cycMps = np.array([(1 / params.mphPerMps) * float(spd_mph)] * (length_in_seconds * hz))
    cycMps[0] = 0.
    cycMps = np.array(cycMps)
    cycSecs = np.arange(0, length_in_seconds, 1. / hz)
    cycGrade = np.array([float(grade)] * (length_in_seconds * hz))
    cycRoadType = np.zeros(length_in_seconds * hz)
    cyc = {'cycMps': cycMps, 'cycSecs': cycSecs, 'cycGrade': cycGrade, 'cycRoadType':cycRoadType}
    return cyc

def main(use_jitclass=True):
    """
    Arguments:
    ----------
    use_jitclass : Boolean
        if True, use numba jitclass"""

    # just use first vehicle in default database
    for i in range(1,27):
        if use_jitclass:
            veh = vehicle.Vehicle(i).get_numba_veh()
            accel_cyc = cycle.Cycle(std_cyc_name=None,
                cyc_dict=create_accel_cyc()).get_numba_cyc()
            accel_out = simdrive.SimAccelTestJit(accel_cyc, veh)

        else:
            veh = vehicle.Vehicle(i)
            accel_cyc = cycle.Cycle(std_cyc_name=None,
                cyc_dict=create_accel_cyc())
            accel_out = simdrive.SimAccelTest(accel_cyc, veh)
        
        accel_out.sim_drive()
        acvhd_0_to_acc_speed_secs = simdrive.SimDrivePost(accel_out).get_output()['ZeroToSixtyTime_secs']
        print('vehicle {}: acceleration [s] {:.3f}'.format(i, acvhd_0_to_acc_speed_secs))

if __name__=='__main__':
    main()
