# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_params
from cycler import cycler
import seaborn as sns
import time
import json
import os
import fastsim as fsim

# %% [markdown]

# `fastsim3` -- load vehicle, cycle and simdrive, and demonstrate usage of methods
# `param_path_list()` and `history_path_list()`
# %%

# load 2012 Ford Fusion from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.save_interval = 1

# load cycle from file
cyc = fsim.Cycle.from_resource("cycles/udds.csv")

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)

# print out all subpaths for variables in SimDrive
print("List of variable paths for SimDrive: ", sd.param_path_list())

# print out all subpaths for history variables in SimDrive
print("List of history variable paths for SimDrive: ", sd.history_path_list())