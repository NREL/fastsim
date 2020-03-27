from src import SimDrive
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np

"""
Changes:
SimDrive.Vehicle.load_vnum
    now allows dynamic selection of vehicle input files for reading and initializing
    no longer stuck with the one default file

"""

def create_accel_cyc(length_in_seconds=300, spd_mph=89.48, grade=0.0, hz=10):
    """
    Create a synthetic Drive Cycle for acceleration targeting.
    Defaults to a 15 second acceleration cycle. Should be adjusted based on target acceleration time
    and initial vehicle acceleration time, so that time isn't wasted on cycles that are needlessly long.

    spd_mph @ 89.48 FASTSim XL version mph default speed for acceleration cycles
    grade @ 0 and hz @ 10 also matches XL version settings
    """
    mphPerMps = 2.23694
    cycMps = [(1/mphPerMps)*float(spd_mph)]*(length_in_seconds*hz)
    cycMps[0] = 0.
    cycMps = np.asarray(cycMps)
    cycSecs = np.arange(0, length_in_seconds, 1./hz)
    cycGrade = np.asarray([float(grade)]*(length_in_seconds*hz))
    cycRoadType = np.zeros(length_in_seconds*hz)
    cyc = {'cycMps': cycMps, 'cycSecs': cycSecs, 'cycGrade': cycGrade, 'cycRoadType':cycRoadType}
    return cyc

def run_vanilla_test(vnum, vehicle_input_file):
    """
    demonstrate that "vanilla" version - non Numba extended - yields
    MPS estimates that are close using np.roots and newtons method
    :return:
    """
    v_accel_cyc = SimDrive.Cycle(std_cyc_name=None,
                                 cyc_dict=create_accel_cyc())
    vanilla_vehicle = SimDrive.Vehicle()
    vanilla_vehicle.load_vnum(vnum, vehicle_input_file)

def run_numba_test(vnum, vehicle_input_file):
    """
    demonstrate that Numba extended version yields
    MPS estimates that are close using np.roots and newtons method
    :return:
    """
    n_accel_cyc = SimDrive.Cycle(std_cyc_name=None,
                                 cyc_dict=create_accel_cyc())
    n_accel_cyc=n_accel_cyc.get_numba_cyc()
    numba_vehicle = SimDrive.Vehicle()
    numba_vehicle.load_vnum(vnum, vehicle_input_file)
    numba_vehicle=numba_vehicle.get_numba_veh()

def main():
    vehicle_input_file=r'FASTSim_py_veh_db.csv'
    vehdf=pd.read_csv(vehicle_input_file)
    numvehicles = len(vehdf)
    vehicles=[]
    ts=time.time()
    for vnum in range(1, numvehicles+1):
        run_numba_test(vnum, vehicle_input_file)
        run_vanilla_test(vnum, vehicle_input_file)
    print('time to run all [s] {}'.format(time.time()-ts))

if __name__=='__main__':
    main()