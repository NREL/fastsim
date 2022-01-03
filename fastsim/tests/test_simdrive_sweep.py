"""Test script that saves results from 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles. 
From command line, pass True (default if left blank) or False argument to use JIT compilation or not, respectively."""

import pandas as pd
import time
import numpy as np
import re
import os
import sys
from pathlib import Path
import unittest

# local modules
from fastsim import simdrive, vehicle, cycle


def main(use_jitclass=True, err_tol=1e-4, verbose=True):
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

    Returns:
    --------
    df_err : pandas datafram, fractional errors
    df : pandas dataframe, new values
    df0 : pandas dataframe, original benchmark values
    """
    t0 = time.time()

    print('Running vehicle sweep.\n')

    cyc_names = ['udds', 'hwfet', 'us06']
    cycs = {
        cyc_name: cycle.Cycle(cyc_name).get_numba_cyc() if use_jitclass else 
            cycle.Cycle(cyc_name) for cyc_name in cyc_names
        }

    vehnos = np.arange(1, 27)

    veh = vehicle.Vehicle(1, verbose=False)
    if use_jitclass:
        veh = veh.get_numba_veh()
    energyAuditErrors = []

    dict_diag = {}
    t0a = 0 
    iter = 0
    for vehno in vehnos:
        if verbose:
            print('vehno =', vehno)
        if vehno == 2:
            t0a = time.time()
        for cyc_name, cyc in cycs.items():
            if not(vehno == 1):
                veh = vehicle.Vehicle(vehno, verbose=False)
                if use_jitclass:
                    veh = veh.get_numba_veh()

            if use_jitclass:
                sim_drive = simdrive.SimDriveJit(cyc, veh)
                sim_drive.sim_drive()
            else:
                sim_drive = simdrive.SimDriveClassic(cyc, veh)
                sim_drive.sim_drive()
                
            sim_drive_post = simdrive.SimDrivePost(sim_drive)
            # sim_drive_post.set_battery_wear()
            diagno = sim_drive_post.get_diagnostics()
            energyAuditErrors.append(sim_drive.energyAuditError)

            if iter == 0:
                dict_diag['vnum'] = [vehno]
                dict_diag['Scenario_name'] = [veh.Scenario_name]
                dict_diag['cycle'] = [cyc_name]
                for key in diagno.keys():
                    dict_diag[key] = [diagno[key]]
                iter += 1

            else:
                dict_diag['vnum'].append(vehno)
                dict_diag['Scenario_name'].append(veh.Scenario_name)
                dict_diag['cycle'].append(cyc_name)
                for key in diagno.keys():
                    dict_diag[key].append(diagno[key])

    df = pd.DataFrame.from_dict(dict_diag)

    t1 = time.time()
    print()
    print('Elapsed time: {:.2f} s'.format(t1 - t0))
    print('Elapsed time since first vehicle: {:.2f} s'.format(t1 - t0a, 2))


    df0 = pd.read_csv(Path(simdrive.__file__).parent.resolve() / 'resources/master_benchmark_vars.csv')

    # make sure new dataframe does not incude newly added or deprecated columns
    new_cols = {col for col in df.columns} - {col for col in df0.columns}
    old_cols = {col for col in df0.columns} - {col for col in df.columns}
    
    from math import isclose

    df_err = df.copy().drop(columns=list(new_cols) + list(old_cols))
    abs_err = []
    for idx in df.index:
        for col in df_err.columns[2:]:
            if not(isclose(df.loc[idx, col], df0.loc[idx, col], rel_tol=err_tol, abs_tol=err_tol)):
                df_err.loc[idx, col] = (df.loc[idx, col] - df0.loc[idx, col]) / df0.loc[idx, col]
                abs_err.append(np.abs(df_err.loc[idx, col]))
                print(f"{df_err.loc[idx, col]:.5%} error for {col}")
                print(f"vehicle: {vehicle.DEFAULT_VEHDF[vehicle.DEFAULT_VEHDF['Selection'] == df.loc[idx, 'vnum']]['Scenario name'].values[0]}")
                print(f"cycle: {df.loc[idx, 'cycle']}")         
                print('New Value: ' + str(round(df.loc[idx, col], 15)))
                print('Old Value: ' + str(round(df0.loc[idx, col], 15)))
                print()
            else:
                df_err.loc[idx, col] = 0

    abs_err = np.array(abs_err)
    if len(abs_err) > 0:
        print(f'\nmax error = {abs_err.max():.3%}')
    else: 
        print(f'No errors exceed the {err_tol:.3g} tolerance threshold.')

    return df_err, df, df0

class TestSimDriveSweep(unittest.TestCase):
    def test_with_jit(self):
        "Compares jit results against benchmark."
        print(f"Running {type(self)}.test_with_jit.") 
        df_err, _, _ = main(use_jitclass=True, verbose=False)
        self.assertEqual(df_err.iloc[:, 2:].max().max(), 0)

    # this implicitly works if test_with_jit works, but may 
    # def test_without_jit(self):
    #     "Compares non-jit results against benchmark."
    #     print('Running TestSimDriveSweep.test_without_jit')
    #     df_err, _, _ = main(use_jitclass=False)
    #     self.assertEqual(df_err.iloc[:, 2:].max().max(), 0)
        
if __name__ == '__main__':
    df_err, df, df0 = main(use_jitclass=False)