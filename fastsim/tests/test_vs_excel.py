"""Module for comparing python results with Excel by running all the vehicles 
in both Excel (uses archived results if Excel version not available) and 
Python FASTSim for both UDDS and HWFET cycles."""


# local modules
import pandas as pd
import time
import numpy as np
import os
import json
import sys
import importlib
from pathlib import Path
from math import isclose
import importlib
import unittest

import fastsim as fsim
from fastsim import simdrive, vehicle, cycle, simdrivelabel
from fastsim.rustext import RUST_AVAILABLE, warn_rust_unavailable
importlib.reload(simdrivelabel) # useful for debugging


RUN_PYTHON = False
RUN_RUST = True
_USE_RUST_LIST = []
if RUN_PYTHON:
    _USE_RUST_LIST.append(False)
if RUN_RUST and RUST_AVAILABLE:
    _USE_RUST_LIST.append(True)

if 'xlwings' in sys.modules:
    xw_success = True
    import xlwings as xw
else:
    xw_success = False


def run(vehicles=np.arange(1, 27), verbose=True, use_rust=False):
    """
    Runs python fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions.
    
    Arguments:
    **********
    verbose : Boolean
        if True, print progress
    use_rust: Boolean, if True, use Rust versions of classes
    """
    if use_rust and not RUST_AVAILABLE:
        warn_rust_unavailable(__file__)

    t0 = time.time()

    print('Running vehicle sweep.')
    print()

    res_python = {}
    def to_rust(obj):
        if use_rust and RUST_AVAILABLE:
            return obj.to_rust()
        return obj

    for vehno in vehicles:
        veh = to_rust(vehicle.Vehicle.from_vehdb(vehno))
        if verbose:
            print('Running ' + veh.scenario_name)
        res_python[veh.scenario_name] = simdrivelabel.get_label_fe(veh, verbose=False, use_rust=use_rust)

    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')

    return res_python


PREV_RES_PATH = Path(__file__).resolve().parents[1] / 'resources' / 'res_excel.json'


def run_excel(vehicles=np.arange(1, 28), prev_res_path=PREV_RES_PATH, rerun_excel=False):
    """
    Runs excel fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions.
    Arguments: 
    -----------
    prev_res_path : path (str) to prevous results in pickle (*.p) file
    rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open
    """

    if not(rerun_excel) and prev_res_path:
        print("Loading archived Excel results.")
        with open(prev_res_path, 'r') as file:
            res_excel = json.load(file)
    elif xw_success:  
        print("Running Excel.")
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
            res_dict['UF'] = sht_veh.range('UF').value
            res_dict['netRangeMiles'] = sht_veh.range('netRangeMiles').value

            for key in res_dict.keys():
                if (res_dict[key] == '') | (res_dict[key] == None):
                    res_dict[key] = 0

            res_excel[sht_vehnames.range('B' + str(vehno + 2)).value] = res_dict

        t1 = time.time()
        print()
        print('Elapsed time: ', round(t1 - t0, 2), 's')
    else:
        print("""Warning: cannot run test_vs_excel.run_excel()
        because xlwings is not installed. Run the command:
        `pip install xlwings` if compatible with your OS.""")

    if rerun_excel:
        with open(prev_res_path, 'w') as file:
            json.dump(res_excel, file)

    return res_excel

# vehicles for which fairly large discrepancies in efficiencies are expected
KNOWN_ERROR_LIST = ['Regional Delivery Class 8 Truck']


def compare(res_python, res_excel, err_tol=0.001, verbose=True):
    """
    Finds common vehicle names in both excel and python 
    (hypothetically all of them, but there may be discrepancies) and then compares
    fuel economy results.  
    Arguments: results from run_python and run_excel
    Returns dict of comparsion results.
    
    Arguments:
    ----------
    res_python : output of run_python
    res_excel : output of run_excel
    err_tol : (float) error tolerance, default=1e-3
    verbose : Boolean
        if True, print progress
    """

    common_names = set(res_python.keys()) & set(res_excel.keys())

    res_keys = ['labUddsMpgge', 'labHwyMpgge', 'labCombMpgge',
                'labUddsKwhPerMile', 'labHwyKwhPerMile', 'labCombKwhPerMile',
                'adjUddsMpgge', 'adjHwyMpgge', 'adjCombMpgge',
                'adjUddsKwhPerMile', 'adjHwyKwhPerMile', 'adjCombKwhPerMile', 
                'netAccel', ]

    res_comps = {}
    for vehname in common_names:
        if verbose:
            print('\n')
            print(vehname)
            print('***'*7)
        res_comp = {}

        if (vehname in KNOWN_ERROR_LIST) and verbose:
            print("Discrepancy in model year between Excel and Python")
            print("is probably the root cause of efficiency errors below.")

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
                error = res_comp[res_key + '_frac_err'] * 100
                if verbose:
                    print(f"{vehname} - {res_key} error = {error:.3g}%")

        if (np.array(res_comp.values()) == 0).all() and verbose:
            print(f'All values within error tolerance of {err_tol:.3g}')

        res_comps[vehname] = res_comp.copy()
    return res_comps


def main(err_tol=0.001, prev_res_path=PREV_RES_PATH, rerun_excel=False, verbose=False):
    """
    Function for running both python and excel and then comparing
    Arguments:
    **********
    err_tol : (float) error tolerance, default=1e-3
    prev_res_path : path (str) to prevous results in pickle (*.p) file
    rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open
    verbose : Boolean
        if True, print progress
    """

    if xw_success and rerun_excel:
        res_python = run(verbose=verbose)
        res_excel = run_excel(prev_res_path=prev_res_path,
                              rerun_excel=rerun_excel)
        res_comps = compare(res_python, res_excel)
    elif not(rerun_excel):
        res_python = run(verbose=verbose)
        res_excel = run_excel(prev_res_path=prev_res_path, rerun_excel=rerun_excel)
        res_comps = compare(res_python, res_excel)
    else:
        print("""Warning: cannot run test_vs_excel.run_excel()
        because xlwings is not installed. Run the command:
        `pip install xlwings` if compatible with your OS.""")

    return res_comps

# 2.2% acceleration error between python and excel is ok because python interpolates
ACCEL_ERR_TOL = 0.022 

class TestExcel(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def test_vs_excel(self):
        "Compares results against archived Excel results."
        for use_rust in _USE_RUST_LIST:
            print(f"Running {type(self)} (Rust: {use_rust})")
            res_python = run(verbose=True, use_rust=use_rust)
            res_excel = run_excel(prev_res_path=PREV_RES_PATH,
                                rerun_excel=False)
            res_comps = compare(res_python, res_excel, verbose=False)

            failed_tests = []
            for veh_key, veh_val in res_comps.items():
                if veh_key not in KNOWN_ERROR_LIST:
                    for attr_key, attr_val in veh_val.items():
                        if attr_key == 'netAccel_frac_err':
                            if ((abs(attr_val) - ACCEL_ERR_TOL) > 0.0):
                                failed_tests.append(veh_key + '.' + attr_key)
                        elif attr_val != 0:
                            failed_tests.append(veh_key + '.' + attr_key)

            self.assertEqual(failed_tests, [])

if __name__ == "__main__":
    unittest.main()
