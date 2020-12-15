"""Test script that runs all the vehicles in both Excel and Python FASTSim for both UDDS and HWFET cycles."""

import pandas as pd
import time
import numpy as np
import os
import sys
import importlib
try:
    import xlwings as xw
except:
    print('You need to install xlwings by running `pip install xlwings`')
    print('in a python environment.')
    raise
from math import isclose

# local modules
from fastsim import simdrive, vehicle, cycle


def run_python_fastsim():
    """Runs python fastsim through 26 vehicles and returns list of dictionaries 
    containing scenario descriptions."""

    t0 = time.time()

    cycles = ['udds', 'hwfet']
    vehicles = np.arange(1, 27)

    print('Instantiating classes.')
    print()
    veh = vehicle.Vehicle(1)
    veh_jit = veh.get_numba_veh()
    cyc = cycle.Cycle('udds')
    cyc_jit = cyc.get_numba_cyc()

    res_python = {}

    for vehno in vehicles:
        print('vehno =', vehno)
        res_dict = {}
        for cycname in cycles:
            if not((vehno == 1) and (cycname == 'udds')):
                cyc.set_standard_cycle(cycname)
                cyc_jit = cyc.get_numba_cyc()
                veh.load_veh(vehno)
                veh_jit = veh.get_numba_veh()
            sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
            sim_drive.sim_drive()

            res_dict['fe_' + cycname] = sim_drive.mpgge
            res_dict['kW_hr__mi_' + cycname] = sim_drive.electric_kWh_per_mi
        res_python[veh.Scenario_name] = res_dict
                                   
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
    sht_veh = wb.sheets('VehicleIO')
    sht_udds = wb.sheets('UDDS')
    sht_hwy = wb.sheets('HWY')
    sht_vehnames = wb.sheets('SavedVehs')
    app = wb.app
    load_veh_macro = app.macro("FASTSim.xlsm!reloadVehInfo")
    run_macro = app.macro("FASTSim.xlsm!run.run")

    vehicles = np.arange(1, 27)
    res_excel = {}
    
    for vehno in vehicles:
        print('vehno =', vehno)
        # running a particular vehicle and getting the result
        res_dict = {}
        sht_veh.range('C6').value = vehno
        load_veh_macro()
        run_macro()
        res_dict['fe_udds'] = sht_udds.range(
            'C118').value if sht_udds.range('C118').value != None else 0
        res_dict['fe_hwfet'] = sht_hwy.range(
            'C118').value if sht_hwy.range('C118').value != None else 0
        res_dict['kW_hr__mi_udds'] = sht_udds.range('C120').value
        res_dict['kW_hr__mi_hwfet'] = sht_hwy.range('C120').value

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
    
    res_comps = {}
    for vehname in common_names:
        res_comp = {}
        if not(isclose(res_python[vehname]['kW_hr__mi_udds'], res_excel[vehname]['kW_hr__mi_udds'], rel_tol=1e-6, abs_tol=1e-6)):
            res_comp['udds_elec_per_err'] = (
                res_python[vehname]['kW_hr__mi_udds'] -
                res_excel[vehname]['kW_hr__mi_udds']) / res_excel[vehname]['kW_hr__mi_udds'] * 100
        else:
            res_comp['udds_elec_per_err'] = 0.0

        if not(isclose(res_python[vehname]['kW_hr__mi_hwfet'], res_excel[vehname]['kW_hr__mi_hwfet'], rel_tol=1e-6, abs_tol=1e-6)):
            res_comp['hwfet_elec_per_err'] = (
                res_python[vehname]['kW_hr__mi_hwfet'] -
                res_excel[vehname]['kW_hr__mi_hwfet']) / res_excel[vehname]['kW_hr__mi_hwfet'] * 100
        else:
            res_comp['hwfet_elec_per_err'] = 0.0

        if not(isclose(res_python[vehname]['fe_udds'], res_excel[vehname]['fe_udds'], rel_tol=1e-6, abs_tol=1e-6)):
            res_comp['udds_perc_err'] = (
                res_python[vehname]['fe_udds'] - res_excel[vehname]['fe_udds']) / res_excel[vehname]['fe_udds'] * 100
        else: 
            res_comp['udds_perc_err'] = 0.0

        if not(isclose(res_python[vehname]['fe_hwfet'], res_excel[vehname]['fe_hwfet'], rel_tol=1e-6, abs_tol=1e-6)):
            res_comp['hwy_perc_err'] = (
                res_python[vehname]['fe_hwfet'] - res_excel[vehname]['fe_hwfet']) / res_excel[vehname]['fe_hwfet'] * 100
        else:
            res_comp['hwy_perc_err'] = 0.0

        res_comps[vehname] = res_comp
        print('')
        print(vehname)
        
        print('FE % Error, UDDS: {:.2f}'.format(res_comps[vehname]['udds_perc_err']))
        print('FE % Error, HWY: {:.2f}'.format(res_comps[vehname]['hwy_perc_err']))
        print('kW-hr/mi % Error, UDDS: {:.4f}'.format(res_comps[vehname]['udds_elec_per_err']))
        print('wK-hr/mi % Error, HWY: {:.4f}'.format(res_comps[vehname]['hwfet_elec_per_err']))
    return res_comps

if __name__ == "__main__":
    _ = compare(run_python_fastsim(), run_excel_fastsim())