# %%
import sys
import os
from pathlib import Path
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
import pickle
import seaborn as sns

# pymoo stuff
from pymoo.optimize import minimize
from pymoo.algorithms.nsga3 import NSGA3 as NSGA
from pymoo.factory import get_reference_directions
import autograd.numpy as anp
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

# local modules
from fastsim import simdrivehot, simdrive, vehicle, cycle, utils
import docs.hot_utilities as hot_util

#%%

# load the vehicle
t0 = time.time()
veh = vehicle.Vehicle(veh_file=Path('../vehdb/2012 Ford Fusion.csv'))
veh_jit = veh.get_numba_veh()
veh_jit.dragCoef, veh_jit.wheelRrCoef = utils.abc_to_drag_coeffs(3625 / 2.2,
                                                                 veh.frontalAreaM2,
                                                                 35.55, 0.2159, 0.0182)
veh_jit.fcEffArray *= 1 / 1.0539  # correcting for remaining difference
veh_jit.auxKw = 1.1
print(f"Vehicle load time: {time.time() - t0:.3f} s")

# create drive cycles from vehicle test data in a dict

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(f"Cycle load time: {time.time() - t0:.3f} s")

# run it once before tuning to compile
t0 = time.time()
sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit, 
    teAmbDegC=np.ones(len(cyc.cycSecs), dtype=np.float64) * 0, teFcInitDegC=0, teCabInitDegC=0)
sim_drive.sim_drive()

print(f"Sim drive time: {time.time() - t0:.3f} s")

df = hot_util.load_test_data()
idx = pd.IndexSlice # used to slice multi index 

tuning_cyc_names = ['us06x2 72F cs', 'us06x2 20F cs', 'uddsx4 0F cs']
for cyc_name in tuning_cyc_names:
    for item in error_vars:
        df.loc[idx[cyc_name, :, :], item[1]] = rollav(
            df.loc[idx[cyc_name, :, :], item[1]])

#%%

# list of parameter names to be modified to obtain objectives
params = ['fcThrmMass', 'fcDiam', 'hFcToAmbStop', 'radiator_eff',
          'fcTempEffOffset', 'fcTempEffSlope', 'teTStatDeltaDegC', 'teTStatSTODegC']
lower_bounds = anp.array([50, 0.1, 1, 2, 0.1, 0.0001, 1, 75])
upper_bounds = anp.array([500, 5, 200, 50, 0.95, 0.1, 15, 95])

# list of tuples of pairs of objective errors to minimize in the form of 
# [('model signal1', 'test signal1'), ('model signal2', 'test signal2'), ...].  
error_vars = [('teFcDegC', 'CylinderHeadTempC'),
              ('fcKwInAch', 'Fuel_Power_Calc[kW]'),
              ]

def rollav(data, width=10):
    """Rolling mean for `data` with `width`"""
    out = np.zeros(len(data))
    out[0] = data[0]
    for i in range(1, len(data)):
        if i < width:
            out[i] = data[:i].mean()
        else:
            out[i] = data[i-width:i].mean()
    return out

def get_error_val(model, test, model_time_steps, test_time_steps):
    """Returns time-averaged error for model and test signal.
    Arguments:
    ----------
    model: array of values for signal from model
    model_time_steps: array (or scalar for constant) of values for model time steps [s]
    test: array of values for signal from test
    
    Output: 
    -------
    err: integral of absolute value of difference between model and 
    test per time"""

    err = trapz(
        y=abs(model - np.interp(
            x=model_time_steps, xp=test_time_steps, fp=test)), 
        x=model_time_steps) / model_time_steps[-1]

    return err

def get_error_for_cycle(x):
    """Function for running a single cycle and returning the error."""
    # create cycle.Cycle()
    errors = []     
    for cyc_name in tuning_cyc_names:
        test_time_steps = df.loc[idx[cyc_name, :, :], 'DAQ_Time[s]'].values
        test_te_amb = df.loc[idx[cyc_name, :, :], 'Cell_Temp[C]'].values

        cycSecs = np.arange(0, round(test_time_steps[-1], 0))
        cycMps = np.interp(cycSecs, 
            test_time_steps, 
            df.loc[idx[cyc_name, :, :], 'Dyno_Spd[mps]'].values)

        cyc = cycle.Cycle(cyc_dict={'cycSecs':cycSecs, 'cycMps':cycMps})
        cyc_jit = cyc.get_numba_cyc()

        sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit,
                        teAmbDegC=np.interp(cycSecs, test_time_steps, test_te_amb),
                        teFcInitDegC=df.loc[idx[cyc_name, :, 0], 'CylinderHeadTempC'][0]
        )   

        # unpack input parameters
        for i in range(len(x)):
            sim_drive.__setattr__(params[i], x[i])

        sim_drive.teTStatFODegC = sim_drive.teTStatSTODegC + sim_drive.teTStatDeltaDegC
        sim_drive.sim_drive()

        # calculate error
        for i in range(len(error_vars)):
            model_err_var = sim_drive.__getattribute__(error_vars[i][0])
            test_err_var = df.loc[idx[cyc_name, :, :], error_vars[i][1]].values

            err = get_error_val(model_err_var, test_err_var, 
                model_time_steps=cycSecs, test_time_steps=test_time_steps)

            errors.append(err)
        # normalized fuel error
        fuel_err = abs(np.trapz(y=df.loc[idx[cyc_name, :, :], 'Fuel_Power_Calc[kW]'], x=test_time_steps) - 
                    np.trapz(y=sim_drive.fcKwInAch, x=cycSecs)) / \
                        np.trapz(y=df.loc[idx[cyc_name, :, :], 'Fuel_Power_Calc[kW]'], x=test_time_steps)
        errors.append(fuel_err)

    return tuple(errors)

no_args = len(params)
# no_args = signature(get_error_for_cycle).parameters # another possible way to do this

# get number of outputs
no_outs = len(error_vars) + 1
n_obj = no_outs * len(tuning_cyc_names)

class ThermalProblem(Problem):
    "Class for creating PyMoo problem for FASTSimHot vehicle."

    def __init__(self, **kwargs):
        super().__init__(n_var=no_args, n_obj=n_obj,
                         # lower bounds
                         xl=lower_bounds,
                         # upper bounds
                         xu=upper_bounds,
                         **kwargs,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        err_arr = np.array([get_error_for_cycle(x)])
        # f1 = err_arr[:, 0]    
        # f2 = err_arr[:, 1]
        # out['F'] = anp.column_stack([f1, f2])        
        f = []
        for i in range(err_arr.shape[1]):
            f.append(err_arr[:, i])
        out['F'] = anp.column_stack(f)

#%% 

run_optimization = False

if run_optimization:
    print('Running optimization.')
    problem = ThermalProblem(parallelization=("threads", 1))
    # See https://pymoo.org/algorithms/nsga3.html
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=3)
    algorithm = NSGA(pop_size=200, eliminate_duplicates=True, ref_dirs=ref_dirs)
    t0 = time.time()    
    res = minimize(problem,
                algorithm,
                ('n_gen', 25),
                seed=1,
                verbose=True,)
                #    save_history=True)
    t1 = time.time()
    print(f'\nElapsed time for optimization: {t1 - t0:.2e} s')
    print('\nParameter pareto sets:')
    print(np.array2string(res.X, precision=3, separator=', '))
    print('Results pareto sets:')
    print(np.array2string(res.F, precision=3, separator=', '))

    # write results to file
    with open('tuning_res.txt', 'w') as f:
        f.write('\nParameter pareto sets:\n')
        f.write(np.array2string(res.X, precision=3, separator=', ') + '\n')
        f.write('\nResults pareto sets:\n')
        f.write(np.array2string(res.F, precision=3, separator=', ') + '\n')

    # pickle results
    pickle.dump(res, open('res.p', 'wb'))

# %% 

res = pickle.load(open('res.p', 'rb'))

# %%
# get pareto objectives in a pandas dataframe
pareto_list = []
for pareto_objs in res.F:
    pareto_list.append(pareto_objs.tolist())

columns = [cyc_name + ' ' + var_name for cyc_name in tuning_cyc_names for var_name in ['temp', 'fuel kW', 'fuel kJ']]

df_res = pd.DataFrame(data=pareto_list, columns=columns)

print(df_res.filter(regex='temp').sum(axis=1).sort_values())
print('\n')
print(df_res.filter(regex='fuel kJ').sum(axis=1).sort_values())

#%%
# plot traces 

validation_cyc_names = [name for name in df.index.levels[0] if re.search('(0|20|72)F', name)]
sns.set(font_scale=2)

def plot_cyc_traces(x, show_plots=False):
    print('\nPlotting time traces.')
    for cyc_name in validation_cyc_names:
        test_time_steps = df.loc[idx[cyc_name, :, :], 'DAQ_Time[s]'].values
        test_te_amb = df.loc[idx[cyc_name, :, :], 'Cell_Temp[C]'].values

        cycSecs = np.arange(0, round(test_time_steps[-1], 0))
        cycMps = np.interp(cycSecs, 
            test_time_steps, 
            df.loc[idx[cyc_name, :, :], 'Dyno_Spd[mps]'].values)

        cyc = cycle.Cycle(cyc_dict={'cycSecs':cycSecs, 'cycMps':cycMps})
        cyc_jit = cyc.get_numba_cyc()

        sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit,
                        teAmbDegC=np.interp(
                        cycSecs, test_time_steps, test_te_amb),
                        teFcInitDegC=df.loc[idx[cyc_name, :, 0], 'CylinderHeadTempC'][0])
        sd_base = simdrive.SimDriveJit(cyc_jit, veh_jit)
        sd_base.sim_drive()

        # unpack input parameters
        for i in range(len(x)):
            sim_drive.__setattr__(params[i], x[i])
        sim_drive.teTStatFODegC = sim_drive.teTStatSTODegC + sim_drive.teTStatDeltaDegC
        sim_drive.sim_drive()
        
        fuel_frac_err = (np.trapz(x=cyc.cycSecs, y=sim_drive.fcKwInAch) -
                         np.trapz(x=test_time_steps,
                                  y=df.loc[idx[cyc_name, :, :], 'Fuel_Power_Calc[kW]'])) /\
                        np.trapz(x=test_time_steps,
                            y=df.loc[idx[cyc_name, :, :], 'Fuel_Power_Calc[kW]'])
        temp_err = get_error_val(
            sim_drive.teFcDegC, df.loc[idx[cyc_name,
                                           :, :], 'CylinderHeadTempC'],
            cycSecs, test_time_steps) 

        less_more = 'less' if fuel_frac_err < 0 else 'more'
        print('\n' + cyc_name)
        print(f"Model uses {abs(fuel_frac_err):.2%} " +
              less_more + " fuel than test.")
        print(f"Model temperature error: {temp_err:.2f} ºC")

        if show_plots:
            # temperature plot
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
            ax1.plot(cyc.cycSecs, sim_drive.teFcDegC, label='model')
            ax1.plot(test_time_steps, df.loc[idx[cyc_name,
                                                :, :], 'CylinderHeadTempC'], label='test')
            ax1.set_ylabel('FC Temp. [$^\circ$C]')
            ax1.legend()
            if cyc_name in tuning_cyc_names:
                title = cyc_name + ' tuning'
            else:
                title = cyc_name + ' validation'
            ax1.set_title(title)
            ax2.plot(cyc.cycSecs, sim_drive.mpsAch)
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Speed [mps]')
            plt.savefig('plots/' + title + ' temp.svg')
            plt.savefig('plots/' + title + ' temp.png')

            # fuel energy plot
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
            ax1.plot(cyc.cycSecs[1:], cumtrapz(x=cyc.cycSecs,
                                            y=sim_drive.fcKwInAch * 1e-3), label='thermal')
            ax1.plot(cyc.cycSecs[1:], cumtrapz(x=cyc.cycSecs,
                                               y=sd_base.fcKwInAch * 1e-3), label='no thermal')
            ax1.plot(test_time_steps[1:], cumtrapz(
                x=test_time_steps, y=df.loc[idx[cyc_name, :, :], 'Fuel_Power_Calc[kW]'] * 1e-3), label='test')
            ax1.set_ylabel('Fuel Energy [MJ]')
            ax1.legend()
            if cyc_name in tuning_cyc_names:
                title = cyc_name + ' tuning'
            else:
                title = cyc_name + ' validation'
            ax1.set_title(title)
            ax2.plot(cyc.cycSecs, sim_drive.mpsAch, label='model')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Speed [mps]')
            plt.savefig('plots/' + title + ' energy.svg')
            plt.savefig('plots/' + title + ' energy.png')