import sys
import os
from pathlib import Path
# allow it to find simdrive module
fsimpath=str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)
import numpy as np
from scipy.integrate import cumtrapz, trapz
import autograd.numpy as anp
import time
import pandas as pd
import re
import matplotlib.pyplot as plt
import importlib
from collections import ChainMap
from inspect import signature

# pymoo stuff
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import autograd.numpy as anp
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.visualization.scatter import Scatter

# local modules
from fastsim import simdrivehot, simdrive, vehicle, cycle

# load the vehicle test data
def load_test_data():
    test_data_path = 'C:\\Users\\cbaker2\\Documents\\TestData\\FordFusionTestData\\'
    t0 = time.time()
    datadirs = [test_data_path + 'udds_tests\\',
                test_data_path + 'us06_tests\\']
    tests = [os.listdir(f) for f in datadirs]

    dflist = [{j: pd.read_excel(i+j, sheet_name='Data')
            for j in tests[datadirs.index(i)]} for i in datadirs]
    print('Elapsed time to read data files: {:.3e} s'.format(time.time() - t0))
    dfdict0 = dict(ChainMap(*reversed(dflist)))
    dfdict1 = {}

    for key in dfdict0.keys():
        repkey = key[9:].replace('.xlsx', '').replace(' ', '').replace('_', ' ')
        #     cyclekey = re.match('\S+(?=x\d)', repkey)[0] # grab cycle name without repetition suffix
        # grab cycle name with repetition suffix
        cyclekey = re.match('\S+', repkey)[0]
        tempkey = re.search('\d{1,2}F', repkey)[0]
        startkey = re.search('(?<=_)(c|h)s', key)[0]
        dfdict1[cyclekey + ' ' + tempkey + ' ' + startkey] = dfdict0[key]
    del dfdict0
    times = np.arange(0, 1500, 250)

    # dict for results dataframes
    resdfdict = {}
    dfdict = {}

    cs_keys = [key for key in dfdict1.keys() if re.search('\scs$', key)]

    for k in cs_keys:
        # print(k)

        # Remove rows with wacky time
        steps = dfdict1[k]['DAQ_Time[s]'].diff()
        timejumps = np.where(steps < 0)[0]
        timejump = timejumps[-1]
        dfdict[k] = dfdict1[k].drop(np.arange(timejump), axis=0)
        dfdict[k].index -= dfdict[k].index.min()

        # Remove extra columns
        dfdict[k].insert(int(np.where(dfdict[k].columns == 'Dyno_Spd[mph]')[
                        0][0]), 'Dyno_Spd[mps]', dfdict[k]['Dyno_Spd[mph]'] * 0.44704)

        dfdict[k]['Tractive_Power[W]'] = dfdict[k]['Dyno_TractiveForce[N]'] * \
            dfdict[k]['Dyno_Spd[mps]']
        dfdict[k]['PosTractive_Power[W]'] = dfdict[k]['Tractive_Power[W]']
        # negindices = np.where(dfdict[k]['Tractive_Power[W]'] < 0)[0]
        negindices = dfdict[k]['Tractive_Power[W]'] < 0
        dfdict[k].loc[negindices, 'PosTractive_Power[W]'] = 0

        dfdict[k]['Kinetic_Power_Density[W/kg]'] = dfdict[k]['Dyno_Spd[mps]'] * dfdict[k]['Dyno_Spd[mps]'].diff() \
            / dfdict[k]['DAQ_Time[s]'].diff()
        dfdict[k]['Kinetic_Power_Density[W/kg]'].fillna(0, inplace=True)

        dfdict[k]['PosKinetic_Power_Density[W/kg]'] = dfdict[k]['Kinetic_Power_Density[W/kg]']
        dfdict[k].loc[dfdict[k]['Kinetic_Power_Density[W/kg]']
                    <= 0, 'PosKinetic_Power_Density[W/kg]'] = 0

        dfdict[k]['PosTractive_Energy[J]'] = np.concatenate((np.array([0]), cumtrapz(
            dfdict[k]['PosTractive_Power[W]'], x=dfdict[k]['DAQ_Time[s]'])), axis=0)
        dfdict[k]['Distance[m]'] = np.concatenate((np.array([0]), cumtrapz(
            dfdict[k]['Dyno_Spd[mps]'], x=dfdict[k]['DAQ_Time[s]'])), axis=0)

        dfdict[k]['PosKinetic_Energy_Density[J/kg]'] = np.concatenate((np.array([0]),
                                                                    cumtrapz(dfdict[k]['PosKinetic_Power_Density[W/kg]'].values,
                                                                                x=dfdict[k]['DAQ_Time[s]'].values)), axis=0)


        dfdict[k].index.set_names('time_step', inplace=True)

        # bring temperature into data frame from filenames
        dfdict[k]['Temp_Amb[C]'] = round(
            (int(re.search('\d{1,2}(?=F)', k)[0]) - 32) * 5 / 9, 1)

        # compute deltaT at times
        indices = [np.where(dfdict[k]['DAQ_Time[s]'] >= t)[0][0] for t in times]

        # resdfdict[k].drop(axis=0, index=0, inplace=True) # probably want this initial row
        dfdict[k].set_index([dfdict[k].index, 'Temp_Amb[C]'], inplace=True)

    keys = list(dfdict.keys())
    del dfdict1

    df = pd.concat(dfdict, sort=False)
    df.index.set_names('filename', level=0, inplace=True)
    df = df.reorder_levels(['filename', 'Temp_Amb[C]', 'time_step'])
    cols0 = list(df.columns.values)
    # reorder columns in terms of independent and dependent variables
    cols = [cols0[0]] + cols0[-3:] + cols0[2:-3]
    df = df[cols]
    df['Powertrain Efficiency'] = df['Dyno_Power_Calc[kW]'] / df['Fuel_Power_Calc[kW]']
    df.drop(columns=[col for col in df.columns if re.search(
        'Unnamed', col)], inplace=True)
    
    return df

df = load_test_data()

# load the vehicle
t0 = time.time()
veh = vehicle.Vehicle(9) # this needs to come from the file for the Fusion after Fusion data is converted to standalone file type
veh_jit = veh.get_numba_veh()
print(f"Vehicle load time: {time.time() - t0:.3f} s")

# create drive cycles from vehicle test data in a dict

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(f"Cycle load time: {time.time() - t0:.3f} s")

t0 = time.time()
sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit)
sim_drive.sim_drive() 

print(f"Sim drive time: {time.time() - t0:.3f} s")

idx = pd.IndexSlice

cyc_name = 'us06x4 0F cs'

def get_error_for_cycle(x):
    """Function for running a single cycle and returning the error."""
    # unpack input parameters
    fcThrmMass, fcDiam, hFcToAmbStop, hFcToAmbRad = x

    # create cycle.Cycle()
    test_time_steps = df.loc[idx[cyc_name, :, :], 'DAQ_Time[s]'].values
    
    cycSecs = np.arange(0, round(test_time_steps[-1], 0))
    cycMps = np.interp(cycSecs, 
        test_time_steps, 
        df.loc[idx[cyc_name, :, :], 'Dyno_Spd[mps]'].values)

    cyc = cycle.Cycle(cyc_dict={'cycSecs':cycSecs, 'cycMps':cycMps})
    cyc_jit = cyc.get_numba_cyc()

    # simulate
    sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit)
    sim_drive.teAmbDegC = np.interp(cycSecs,
            test_time_steps,
            df.loc[idx[cyc_name, :, :], 'Cell_Temp[C]'].values)
    sim_drive.sim_drive()

    # calculate error
    fc_te_err = trapz(y=abs(sim_drive.teFcDegC - np.interp(
        cyc.cycSecs, test_time_steps, 'CylinderHeadTempC'
        )), x=self.cyc.cycSecs) * self.cyc.cycSecs[-1]
    fc_dte_err = trapz(y=abs(sim_drive.teFcDegC - np.interp(
        cyc.cycSecs, test_time_steps, 'CylinderHeadTempC'
        )), x=self.cyc.cycSecs) * self.cyc.cycSecs[-1]

    return fc_te_err, fc_dte_err

no_args = signature(get_error_for_cycle).parameters

# test function and get number of outputs
no_outs = len()

class ThermalProblem(Problem):
    "Class for creating PyMoo problem for FASTSimHot vehicle."

    def __init__(self, **kwargs):
        super().__init__(n_var=5, n_obj=2,
                         # lower bounds
                         xl=anp.array([50e3, 0.1, 0.01, 10, 250, 0.2]),
                         # upper bounds
                         xu=anp.array([300e3, 0.95, 1, 200, 1500, 0.5]),
                         **kwargs,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        err_arr = np.array([get_error_for_cycle(x)])
        f1 = err_arr[:, 0]
        f2 = err_arr[:, 1]

        out['F'] = anp.column_stack([f1, f2])


problem = ThermalProblem(parallelization=("threads", 6))
algorithm = NSGA2(pop_size=6, eliminate_duplicates=True)
t0 = time.time()
res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=True)
t1 = time.time()
print('res.X:\n', res.X)
print('res.F:\n', res.F)

print('Elapsed time:', round(t1 - t0, 2))
print('Real time ratio:', round((stop_time - start_time) / (t1 - t0), 0))
print('Error function tested successfully.')
