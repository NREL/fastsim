"""Test script that runs all the vehicles in both Excel and Python FASTSim for both UDDS and HWFET cycles."""

import xlwings as xw
import pandas as pd
import time
import numpy as np
import os
import sys
import importlib
import xlwings as xw
from math import isclose

# local modules
from fastsim import simdrive, vehicle, cycle, simdrivelabel


def run_python_fastsim():
    """Runs python fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions."""

    t0 = time.time()

    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()

    res_python = {}

    for vehno in vehicles:
        print('vehno =', vehno)
        veh_jit = vehicle.Vehicle(vehno).get_numba_veh()
        res_python[veh_jit.Scenario_name] = simdrivelabel.get_label_fe(veh_jit)
                                   
    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')

    return res_python


def run_excel_fastsim():
    """Runs excel fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions."""

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

    vehicles = np.arange(1, 39)
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
        res_dict['labUddsKwhPerMile'] = sht_veh.range('labUddsKwhPerMile').value
        res_dict['labHwyKwhPerMile'] = sht_veh.range('labHwyKwhPerMile').value
        res_dict['labCombKwhPerMile'] = sht_veh.range('labCombKwhPerMile').value

        # adjusted results
        res_dict['adjUddsMpgge'] = sht_veh.range('adjUddsMpgge').value
        res_dict['adjHwyMpgge'] = sht_veh.range('adjHwyMpgge').value
        res_dict['adjCombMpgge'] = sht_veh.range('adjCombMpgge').value
        res_dict['adjUddsKwhPerMile'] = sht_veh.range('adjUddsKwhPerMile').value
        res_dict['adjHwyKwhPerMile'] = sht_veh.range('adjHwyKwhPerMile').value
        res_dict['adjCombKwhPerMile'] = sht_veh.range('adjCombKwhPerMile').value

        for key in res_dict.keys():
            if (res_dict[key] == '') | (res_dict[key] == None):
                res_dict[key] = 0
                res_excel[sht_vehnames.range('B' + str(vehno + 2)).value] = res_dict

    t1 = time.time()
    print()
    print('Elapsed time: ', round(t1 - t0, 2), 's')

    return res_excel


def compare(res_python, res_excel):
    """Finds common vehicle names in both excel and python 
    (hypothetically all of them, but there may be discrepancies) and then compares
    fuel economy results.  
    Arguments: results from run_python_fastsim and run_excel_fastsim
    Returns dict of comparsion results."""

    common_names = set(res_python.keys()) & set(res_excel.keys())

    ERR_TOL = 0.02

    res_keys = ['labUddsMpgge', 'labHwyMpgge', 'labCombMpgge', 
        'labUddsKwhPerMile', 'labHwyKwhPerMile', 'labCombKwhPerMile',
        'adjUddsMpgge', 'adjHwyMpgge', 'adjCombMpgge', 
        'adjUddsKwhPerMile', 'adjHwyKwhPerMile', 'adjCombKwhPerMile', ]

    res_comps = {}
    for vehname in common_names:
        print('')
        print(vehname)
        print('***'*5)
        for res_key in res_keys:
            res_comp = {}
            if not(isclose(res_python[vehname][res_key], 
                res_excel[vehname][res_key],
                rel_tol=ERR_TOL, abs_tol=ERR_TOL)):
                try:
                    res_comp[res_key + '_frac_err'] = (
                        res_python[vehname][res_key] -
                        res_excel[vehname][res_key]) / res_excel[vehname][res_key]
                except:
                    res_comp[res_key + '_frac_err'] = np.float64('666'*6)
                print(
                    res_key + ' error = {:.3g}%'.format(res_comp[res_key + '_frac_err'] * 100))
                print('')
            else:
                res_comp[res_key + '_frac_err'] = 0.0
        if (np.array(list(res_comp.values())) == 0).all():
            print(f'All values within error tolerance of {ERR_TOL:.3g}')

        res_comps[vehname] = res_comp
    return res_comps

if __name__ == "__main__":
    res_python = run_python_fastsim()
    res_excel = run_excel_fastsim()
    res_comp = compare(res_python, res_excel)
