"""Test script that saves results from 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles. 
From command line, pass True (default if left blank) or False argument to use JIT compilation or not, respectively."""

import pandas as pd
import time
import numpy as np
import re
import os
import sys
import inspect
from pathlib import Path

# local modules
from fastsim import simdrive, vehicle, cycle


def main(use_jitclass=True, err_tol=1e-4):
    """Runs test test for 26 vehicles and 3 cycles.  
    Test compares cumulative positive and negative energy 
    values to a benchmark from earlier.
    
    Arguments:
    ----------
    use_jitclass : use numba or not, default True
    err_tol : error tolerance
        default of 1e-4 was selected to prevent minor errors from showing.  
        As of 31 December 2020, a recent python update caused errors that 
        are smaller than this and therefore ok to neglect.
    """
    t0 = time.time()

    cycles = ['udds', 'hwfet', 'us06']
    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()
    veh = vehicle.Vehicle(1)
    if use_jitclass:
        veh_jit = veh.get_numba_veh()
    cyc = cycle.Cycle('udds')
    if use_jitclass:
        cyc_jit = cyc.get_numba_cyc()

    energyAuditErrors = []

    iter = 0
    for vehno in vehicles:
        print('vehno =', vehno)
        for cycname in cycles:
            if not((vehno == 1) and (cycname == 'udds')):
                cyc.set_standard_cycle(cycname)
                if use_jitclass:
                    cyc_jit = cyc.get_numba_cyc()
                veh.load_veh(vehno)
                if use_jitclass:
                    veh_jit = veh.get_numba_veh()

            if use_jitclass:
                sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
                sim_drive.sim_drive()
            else:
                sim_drive = simdrive.SimDriveClassic(cyc, veh)
                sim_drive.sim_drive()
                
            sim_drive_post = simdrive.SimDrivePost(sim_drive)
            # sim_drive_post.set_battery_wear()
            diagno = sim_drive_post.get_diagnostics()
            energyAuditErrors.append(sim_drive.energyAuditError)

            if iter == 0:
                dict_diag = {}
                dict_diag['vnum'] = [vehno]
                dict_diag['cycle'] = [cycname]
                for key in diagno.keys():
                    dict_diag[key] = [diagno[key]]
                iter += 1

            else:
                dict_diag['vnum'].append(vehno)
                dict_diag['cycle'].append(cycname)
                for key in diagno.keys():
                    dict_diag[key].append(diagno[key])

    df = pd.DataFrame.from_dict(dict_diag)

    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')

    df0 = pd.read_csv(Path(simdrive.__file__).parent.resolve() / 'resources/master_benchmark_vars.csv')

    # make sure both dataframes have the same columns
    new_cols = {col for col in df.columns} - {col for col in df0.columns}
    df.drop(columns=new_cols, inplace=True)
    old_cols = {col for col in df0.columns} - {col for col in df.columns}
    df0.drop(columns=old_cols, inplace=True)

    from math import isclose

    df_err = df.copy()
    abs_err = []
    for idx in df.index:
        for col in df.columns[2:]:
            if not(isclose(df.loc[idx, col], df0.loc[idx, col], rel_tol=err_tol, abs_tol=err_tol)):
                df_err.loc[idx, col] = (df.loc[idx, col] - df0.loc[idx, col]) / df0.loc[idx, col]
                abs_err.append(np.abs(df_err.loc[idx, col]))
                print(str(round(df_err.loc[idx, col] * 100, 5)) + '% for')
                print('New Value: ' + str(round(df.loc[idx, col], 15)))
                print('vnum = ' + str(df.loc[idx, 'vnum']))            
                print('cycle = ' + str(df.loc[idx, 'cycle']))            
                print('idx =', idx, ', col =', col)
                print()
            else:
                df_err.loc[idx, col] = 0

    abs_err = np.array(abs_err)
    if len(abs_err) > 0:
        print('\nmax error =', str(round(abs_err.max() * 100, 4)) + '%')
    else: 
        print(f'No errors exceed the {err_tol:.3g} tolerance threshold.')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if re.match('(?i)true', sys.argv[1]):
            use_jitclass = True
            print('Using numba JIT compilation.')
        else:
            use_jitclass = False
            print('Skipping numba JIT compilation.')
        if len(sys.argv) > 2:
            err_tol = float(sys.argv[2])
            print(f"Using error tolerance of {err_tol:.3g}.")
        else:
            err_tol = list(inspect.signature(main).parameters.values())[1].default
            print(f"Using error default tolerance of {err_tol:.3g}.")

        main(use_jitclass=use_jitclass, err_tol=err_tol)
    else:
        print('Using numba JIT compilation.')
        main()
