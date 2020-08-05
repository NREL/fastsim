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
# importlib.reload(simdrive)
# importlib.reload(cycle)

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(f"Cycle load time: {time.time() - t0:.3f} s")


# ### Load Powertrain Model
# 
# A vehicle database in CSV format is required to be in the working directory where FASTSim is running (i.e. the same directory as this notebook). The "get_veh" function selects the appropriate vehicle attributes from the database and contructs the powertrain model (engine efficiency map, etc.). An integer value corresponds to each vehicle in the database. To add a new vehicle, simply populate a new row to the vehicle database CSV.


t0 = time.time()
veh = vehicle.Vehicle(9)
veh_jit = veh.get_numba_veh()
print(f"Vehicle load time: {time.time() - t0:.3f} s")


# ### Run FASTSim
# 
# The "sim_drive" function takes the drive cycle and vehicle models defined above as inputs. The output is a dictionary of time series and scalar values described the simulation results. Typically of interest is the "gge" key, which is an array of time series energy consumption data at each time step in the drive cycle.
# 
# If running FASTSim in batch over many drive cycles, the output from "sim_drive" can be written to files or database for batch post-processing. 


t0 = time.time()
importlib.reload(simdrivehot)
sim_drive = simdrivehot.SimDriveHotJit(cyc_jit, veh_jit)
sim_drive.sim_drive() 

print(f"Sim drive time: {time.time() - t0:.3f} s")
