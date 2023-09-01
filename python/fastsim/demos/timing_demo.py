# %%
# local modules
import fastsim as fsim

# other dependencies
import sys
import os
from pathlib import Path
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()


# The purpose of this file is to demonstrate that loading files directly into Rust is always faster,
# and often a lot faster, than loading files in python and then converting.


#%% [markdown]
# # Various ways of instantiating rust cycles
# Performance on M1 Mac:
# ```
# Time to load cycle with `cycle.Cycle.from_file("udds")`: 2.66e-03 s
# Time to load cycle with `cycle.Cycle.from_file(udds_path)`: 1.58e-03 s
# Time to load cycle with `fastsimrust.RustCycle.from_file`: 4.29e-04 s
# ```

t0 = time.perf_counter()
cyc_python = fsim.cycle.Cycle.from_file("udds").to_rust()
t1 = time.perf_counter()
print(f'Time to load cycle with `cycle.Cycle.from_file("udds")`: {t1 - t0:.2e} s')

udds_path = Path(fsim.package_root() / "resources/cycles/udds.csv")
t0 = time.perf_counter()
cyc = fsim.cycle.Cycle.from_file(udds_path).to_rust()
t1 = time.perf_counter()
print(f'Time to load cycle with `cycle.Cycle.from_file(udds_path)`: {t1 - t0:.2e} s')

t0 = time.perf_counter()
cyc = fsim.fastsimrust.RustCycle.from_file(str(udds_path))
t1 = time.perf_counter()
print(f'Time to load cycle with `fastsimrust.RustCycle.from_file`: {t1 - t0:.2e} s')


# %% [markdown]
# # Two ways of instantiating rust vehicles
# Performance on M1 Mac:
# ```
# Time to load vehicle with
# `vehicle.Vehicle.from_file(veh_csv_path)`: 9.78e-03 s
# Time to load vehicle with
# `fastsimrust.RustVehicle.from_file(veh_yaml_path)`: 5.06e-04 s
# ```

veh_csv_path = fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.csv"
t0 = time.perf_counter()
veh_python = fsim.vehicle.Vehicle.from_file(str(veh_csv_path), to_rust=True)
print(
    'Time to load vehicle with\n' + 
    f'`vehicle.Vehicle.from_file(veh_csv_path)`: {time.perf_counter() - t0:.2e} s'
)

veh_yaml_path = fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml"
t0 = time.perf_counter()
veh = fsim.fastsimrust.RustVehicle.from_file(str(veh_yaml_path))
print(
    'Time to load vehicle with\n' + 
    f'`fastsimrust.RustVehicle.from_file(veh_yaml_path)`: {time.perf_counter() - t0:.2e} s'
)

# %% [markdown]
# # Two methods of instantiating and running RustSimDrive
# These end up being about the same performance on M1 Mac:
# ```
# Time to instantiate with `simdrive.RustSimDrive`
# and simulate in rust: 1.46e-03 s
# Time to instantiate with `fastsimrust.RustSimDrive`
# and simulate in rust: 1.31e-03 s
# ```

t0 = time.perf_counter()
sdr = fsim.simdrive.RustSimDrive(cyc, veh)
sdr.sim_drive() 
t_rust = time.perf_counter() - t0
print(
    'Time to instantiate with `simdrive.RustSimDrive`\n' + 
    f'and simulate in rust: {t_rust:.2e} s')

t0 = time.perf_counter()
sdr = fsim.fastsimrust.RustSimDrive(cyc, veh)
sdr.sim_drive() 
t_rust = time.perf_counter() - t0
print(
    'Time to instantiate with `fastsimrust.RustSimDrive`\n' + 
    f'and simulate in rust: {t_rust:.2e} s')

# %% [markdown] 
# # Instantiating and running RustSimDrive with aux load override, no significant difference
# Performance no M1 Mac:
# ```
# Time to instantiate and simulate in rust with aux load array: 1.20e-03 s
# ```

t0 = time.perf_counter()
sdr = fsim.simdrive.RustSimDrive(cyc, veh)
aux_in_kw_override = np.array(cyc.time_s) / cyc.time_s[-1] * 10
sdr.sim_drive(None, aux_in_kw_override) 
t_rust = time.perf_counter() - t0
print(
    f'Time to instantiate and simulate in rust ' + 
    f'with aux load array: {t_rust:.2e} s'
)
