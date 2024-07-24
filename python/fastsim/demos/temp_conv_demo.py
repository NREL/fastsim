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

sns.set_theme()

DEBUG_LOG = os.environ.get("DEBUG_LOG", "false").lower() == "true"     
if DEBUG_LOG:
    fsim.utils.set_log_level("DEBUG")

veh_files = [
    fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml",
    fsim.package_root() / "../../rust/tests/assets/2012_Ford_Fusion.yaml",
 ]
veh_file = next(vf for vf in veh_files if vf.exists())
veh = fsim.Vehicle.from_file(str(veh_file))
assert veh.save_interval == 1
try:
    cyc = fsim.Cycle.from_resource("cycles/udds.csv")
except:
    cyc = fsim.Cycle.from_resource("udds.csv")
    
sd_base = fsim.SimDrive(veh, cyc)

sd_si1 = sd_base.copy()
t0 = time.perf_counter()
sd_si1.walk()
t1 = time.perf_counter()
dt_f3_sd1 = t1-t0
print(f"fastsim-3 `sd1.walk()` elapsed time: {dt_f3_sd1:.3e} s")

sd_none = sd_base.copy()
veh = sd_none.veh
try:
    veh.__save_interval = None
except:
    veh.save_interval = None
t0 = time.perf_counter()
sd_none.walk()
t1 = time.perf_counter()
dt_f3_sd_none = t1 - t0
print(f"fastsim-3 `sd1.walk()` elapsed time: {dt_f3_sd_none:.3e} s")

sd2 = sd_base.to_fastsim2()
t0 = time.perf_counter()
sd2.sim_drive()
t1 = time.perf_counter()
dt_f2 = t1 - t0
print(f"fastsim-2 `sd.walk()` elapsed time: {dt_f2:.3e} s")

print(f"fastsim-3 with `save_interval = None` is {dt_f2 / dt_f3_sd_none}x faster than fastsim-2")
print(f"fastsim-3 with `save_interval = 1` is {dt_f2 / dt_f3_sd1}x faster than fastsim-2")
