"""Module for comparing python results with Excel."""

"""Test script that runs all the vehicles in both Excel and Python FASTSim for both UDDS and HWFET cycles."""


# local modules
import xlwings as xw
import pandas as pd
import time
import numpy as np
import os
import json
import sys
import importlib
from pathlib import Path
import xlwings as xw
from math import isclose
import importlib
import pickle
from fastsim import simdrive, vehicle, cycle, simdrivelabel
importlib.reload(simdrivelabel) # useful for debugging


def run_python(use_jit=False, verbose=True):
    """Runs python fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions.
    
    Arguments:
    **********
    use_jit : Boolean
        if True, use numba jitclass
    verbose : Boolean
        if True, print progress
    """

    t0 = time.time()

    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()

    res_python = {}

    for vehno in vehicles:
        if use_jit:
            veh = vehicle.Vehicle(vehno).get_numba_veh()
        else:
            veh = vehicle.Vehicle(vehno)
        if verbose:
            print('Running ' + veh.Scenario_name)
        res_python[veh.Scenario_name] = simdrivelabel.get_label_fe(veh)

    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')

    return res_python


def run_excel(prev_res_path=Path(__file__).resolve().parents[1] / 'resources' / 'res_excel.json', 
    rerun_excel=False):
    """Runs excel fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions.
    Arguments: 
    -----------
    prev_res_path : path (str) to prevous results in pickle (*.p) file
    rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open"""

    if not(rerun_excel):
        print("Loading Excel results.")
        if '.p' in str(prev_res_path):
            res_excel = pickle.load(open(prev_res_path, 'rb'))
        elif '.json' in str(prev_res_path):
            res_excel = json.load(open(prev_res_path, 'r'))
    else:  
        print("Running Excel")
        t0 = time.time()

        # initial setup
        wb = xw.Book('FASTSim.xlsm')  # FASTSim.xlsm must be open
        # capture sheets with results of interest
        sht_veh = wb.sheets('VehicleIO')
        sht_vehnames = wb.sheets('SavedVehs')
        # setup macros to run from python
        app = wb.app
        load_veh_macro = app.macro("FASTSim.xlsm!reloadVehInfo")
        run_macro = app.macro("FASTSim.xlsm!run.run")

        vehicles = np.arange(1, 28)
        res_excel = {}

        for vehno in vehicles:
            print('vehno =', vehno)
            # running a particular vehicle and getting the result
            res_dict = {}
            # set excel to run vehno
            sht_veh.range('C6').value = vehno
            load_veh_macro()
            run_macro()
            # lab results (unadjusted)
            res_dict['labUddsMpgge'] = sht_veh.range('labUddsMpgge').value
            res_dict['labHwyMpgge'] = sht_veh.range('labHwyMpgge').value
            res_dict['labCombMpgge'] = sht_veh.range('labCombMpgge').value
            res_dict['labUddsKwhPerMile'] = sht_veh.range(
                'labUddsKwhPerMile').value
            res_dict['labHwyKwhPerMile'] = sht_veh.range('labHwyKwhPerMile').value
            res_dict['labCombKwhPerMile'] = sht_veh.range(
                'labCombKwhPerMile').value

            # adjusted results
            res_dict['adjUddsMpgge'] = sht_veh.range('adjUddsMpgge').value
            res_dict['adjHwyMpgge'] = sht_veh.range('adjHwyMpgge').value
            res_dict['adjCombMpgge'] = sht_veh.range('adjCombMpgge').value
            res_dict['adjUddsKwhPerMile'] = sht_veh.range(
                'adjUddsKwhPerMile').value
            res_dict['adjHwyKwhPerMile'] = sht_veh.range('adjHwyKwhPerMile').value
            res_dict['adjCombKwhPerMile'] = sht_veh.range(
                'adjCombKwhPerMile').value
            
            # performance
            res_dict['netAccel'] = sht_veh.range('netAccel').value

            for key in res_dict.keys():
                if (res_dict[key] == '') | (res_dict[key] == None):
                    res_dict[key] = 0

            res_excel[sht_vehnames.range('B' + str(vehno + 2)).value] = res_dict

        t1 = time.time()
        print()
        print('Elapsed time: ', round(t1 - t0, 2), 's')

    return res_excel


def compare(res_python, res_excel, err_tol=0.001):
    """Finds common vehicle names in both excel and python 
    (hypothetically all of them, but there may be discrepancies) and then compares
    fuel economy results.  
    Arguments: results from run_python and run_excel
    Returns dict of comparsion results.
    
    Arguments:
    ----------
    res_python : output of run_python
    res_excel : output of run_excel
    err_tol : (float) error tolerance, default=1e-3"""

    common_names = set(res_python.keys()) & set(res_excel.keys())

    res_keys = ['labUddsMpgge', 'labHwyMpgge', 'labCombMpgge',
                'labUddsKwhPerMile', 'labHwyKwhPerMile', 'labCombKwhPerMile',
                'adjUddsMpgge', 'adjHwyMpgge', 'adjCombMpgge',
                'adjUddsKwhPerMile', 'adjHwyKwhPerMile', 'adjCombKwhPerMile', 
                'netAccel', ]

    res_comps = {}
    for vehname in common_names:
        print('\n')
        print(vehname)
        print('***'*7)
        res_comp = {}
        # if res_python[vehname]['veh'].vehPtType != 3:
        if True:
            for res_key in res_keys:
                if (type(res_python[vehname][res_key]) != np.ndarray) and not(
                    isclose(res_python[vehname][res_key],
                                res_excel[vehname][res_key],
                                rel_tol=err_tol, abs_tol=err_tol)):
                    res_comp[res_key + '_frac_err'] = (
                        res_python[vehname][res_key] -
                        res_excel[vehname][res_key]) / res_excel[vehname][res_key]
                else:
                    res_comp[res_key + '_frac_err'] = 0.0
                if res_comp[res_key + '_frac_err'] != 0.0:
                    print(
                        res_key + ' error = {:.3g}%'.format(res_comp[res_key + '_frac_err'] * 100))

            if (np.array(list(res_comp.values())) == 0).all():
                print(f'All values within error tolerance of {err_tol:.3g}')

            res_comps[vehname] = res_comp.copy()
        else:
            print('You ran a PHEV, which is not working yet in python')
    return res_comps


def main(use_jit=True, err_tol=0.001, 
    prev_res_path=Path(__file__).resolve().parents[1] / 'resources' / 'res_excel.json',
    rerun_excel=False):
    """Function for running both python and excel and then comparing
    Arguments:
    **********
    use_jit : Boolean
        if True, use numba jitclass
    err_tol : (float) error tolerance, default=1e-3
    prev_res_path : path (str) to prevous results in pickle (*.p) file
    rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open"""

    res_python = run_python(verbose=False, use_jit=use_jit)
    res_excel = run_excel(prev_res_path=prev_res_path, rerun_excel=rerun_excel)
    res_comps = compare(res_python, res_excel)

if __name__ == '__main__':
    main()
