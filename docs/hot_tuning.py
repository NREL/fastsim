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
from fastsim import simdrivehot, simdrive, vehicle, cycle, utils
import docs.hot_utilities as hot_util

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

#%%

cyc_name = 'us06x4 0F cs'

# list of parameter names to be modified to obtain objectives
params = ['fcThrmMass', 'fcDiam', 'hFcToAmbStop', 'radiator_eff', 'fcTempEffOffset', 'fcTempEffSlope']
lower_bounds = anp.array([50, 0.1, 1, 2, 0.25, 0.0001])
upper_bounds = anp.array([500, 5, 200, 50, 0.95, 0.1])

# list of tuples of pairs of objective errors to minimize in the form of 
# [('model signal1', 'test signal1'), ('model signal2', 'test signal2'), ...].  
error_vars = [('teFcDegC', 'CylinderHeadTempC'),
              ('fcKwInAch', 'Fuel_Power_Calc[kW]'),
              ]

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
    test_time_steps = df.loc[idx[cyc_name, :, :], 'DAQ_Time[s]'].values
    test_te_amb = df.loc[idx[cyc_name, :, :], 'Cell_Temp[C]'].values
    # fix this to actually calculate the rolling mean for 10 steps (~1 s)
    df.loc[idx[cyc_name, :, :],
           'Fuel_Power_Calc_rollav[kW]'] = df.loc[idx[cyc_name, :, ], 'Fuel_Power_Calc[kW]']
    
    cycSecs = np.arange(0, round(test_time_steps[-1], 0))
    cycMps = np.interp(cycSecs, 
        test_time_steps, 
        df.loc[idx[cyc_name, :, :], 'Dyno_Spd[mps]'].values)

    cyc = cycle.Cycle(cyc_dict={'cycSecs':cycSecs, 'cycMps':cycMps})
    cyc_jit = cyc.get_numba_cyc()

    # simulate
    # try:
    # some conditions cause SimDriveHotJit to have divide by zero errors
    sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit,
                    teAmbDegC=np.interp(cycSecs, test_time_steps, test_te_amb),
                    teFcInitDegC=df.loc[idx[cyc_name, :, 0], 'CylinderHeadTempC'][0]
    )   

    sim_drive.sim_drive()

    # except:
    #     sim_drive=simdrivehot.SimDriveHot(cyc_jit, veh_jit,
    #                     teAmbDegC = np.interp(cycSecs, test_time_steps, test_te_amb),
    #                     teFcInitDegC = df.loc[idx[cyc_name, :, 0], 'CylinderHeadTempC'][0]
    #     )

    #     sim_drive.sim_drive()


    # unpack input parameters
    for i in range(len(x)):
        sim_drive.__setattr__(params[i], x[i])

    sim_drive.sim_drive()

    # calculate error
    errors = []
    for i in range(len(error_vars)):
        model_err_var = sim_drive.__getattribute__(error_vars[i][0])
        test_err_var = df.loc[idx[cyc_name, :, :], error_vars[i][1]].values

        err = get_error_val(model_err_var, test_err_var, 
            model_time_steps=cycSecs, test_time_steps=test_time_steps)

        errors.append(err)

    return tuple(errors)

no_args = len(params)
# no_args = signature(get_error_for_cycle).parameters # another possible way to do this

# test function and get number of outputs
no_outs = len(error_vars)

class ThermalProblem(Problem):
    "Class for creating PyMoo problem for FASTSimHot vehicle."

    def __init__(self, **kwargs):
        super().__init__(n_var=no_args, n_obj=no_outs,
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

print('Running optimization.')
problem = ThermalProblem(parallelization=("threads", 6))
algorithm = NSGA2(pop_size=12, eliminate_duplicates=True)
t0 = time.time()    
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=True)
t1 = time.time()
print('\nParameter pareto sets:')
print(np.array2string(res.X, precision=3, separator=', '))
print('Results pareto sets:')
print(np.array2string(res.F, precision=3, separator=', '))

