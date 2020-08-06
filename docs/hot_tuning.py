import sys
import os
from pathlib import Path
# allow it to find simdrive module
fsimpath=str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
import shutil

# pymoo stuff
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import autograd.numpy as anp
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.visualization.scatter import Scatter

# local modules
from fastsim import simdrivehot, simdrive, vehicle, cycle

# load the vehicle
t0 = time.time()
veh = vehicle.Vehicle(9) # this needs to come from the file for the Fusion after Fusion data is converted to standalone file type
veh_jit = veh.get_numba_veh()
print(f"Vehicle load time: {time.time() - t0:.3f} s")

# load the vehicle test data
test_data_path = r'C:\Users\cbaker2\Documents\TestData\FordFusionTestData'

# create drive cycles from vehicle test data in a list

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(f"Cycle load time: {time.time() - t0:.3f} s")

t0 = time.time()
sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit)
sim_drive.sim_drive() 

print(f"Sim drive time: {time.time() - t0:.3f} s")


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
