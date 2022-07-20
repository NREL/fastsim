# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

import fastsim as fsim
import fastsimrust as fsr

# load test data which can be obtained at
# https://www.anl.gov/taps/d3-2012-ford-fusion-v6
possible_trip_dirs = (
    Path().home() / "Documents/DynoTestData/FordFusionTestData/",
)

for trip_dir in possible_trip_dirs:
    if trip_dir.exists():
        break

rho_fuel_kg_per_ml = 0.743e-3
lhv_fuel_btu_per_lbm = 18_344
lbm_per_kg = 2.2
btu_per_kj = 0.948
lhv_fuel_kj_per_kg = lhv_fuel_btu_per_lbm * lbm_per_kg / btu_per_kj

# full data
dfs_raw = dict()
# resampled to 1 Hz
dfs = dict()
for sub in trip_dir.iterdir():
    if sub.is_dir():
        for file in sub.iterdir():
            if file.suffix == ".csv" and "_cs" in file.stem:
                print(f"loading: ", file.resolve())
                dfs_raw[file.stem] = pd.read_csv(file)
                # clip time at zero seconds
                dfs_raw[file.stem] = dfs_raw[file.stem][dfs_raw[file.stem]['Time[s]'] >= 0.0]
                dfs[file.stem] = fsim.resample(
                    dfs_raw[file.stem],
                    rate_vars=('Eng_FuelFlow_Direct[cc/s]')
                )
                dfs[file.stem]['Fuel_Power_Calc[kW]'] = dfs[
                    file.stem]["Eng_FuelFlow_Direct[cc/s]"] * rho_fuel_kg_per_ml * lhv_fuel_kj_per_kg

# %%
# plot the data
show_plots = True

if show_plots:
    for key, df in dfs_raw.items():
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        ax[0].set_title(key)
        ax[0].plot(df['Time[s]'], df["CylinderHeadTempC"], label="cyl. head")
        ax[0].plot(df['Time[s]'], df["Cell_Temp[C]"], label="ambient")
        ax[0].set_ylabel("temp [°C]")
        ax[0].legend()
        ax[1].plot(df['Time[s]'], df["Fuel_Power_Calc[kW]"], label="ambient")
        ax[1].set_ylabel("Fuel Power [kW]")
        ax[1].legend()
        ax[-1].plot(df['Time[s]'], df["Dyno_Spd[mph]"])
        ax[-1].set_ylabel("speed [mph]")    
        ax[-1].set_xlabel('time [s]')

# %% 
# Separate calibration and validation cycles

cal_cyc_patterns = ("49", "56", "73", "60", "69", "77")
dfs_cal = dict()
for key in dfs.keys():
    for pattern in cal_cyc_patterns:
        if pattern in key:
            dfs_cal[key] = dfs[key]

dfs_val_keys = set(dfs.keys()) - set(dfs_cal.keys())
dfs_val = {key: dfs[key] for key in dfs_val_keys}


# %%
# create cycles and sim_drives

veh = fsim.vehicle.Vehicle.from_file("2012_Ford_Fusion.csv").to_rust()

cycs = dict()
cal_sim_drives = dict()
val_sim_drives = dict()
for key in dfs_raw.keys():
    cycs[key] = fsim.cycle.Cycle.from_dict(
        {
            "time_s": dfs_raw[key]["Time[s]"],
            "mps": dfs_raw[key]["Dyno_Spd[mph]"] / fsim.params.MPH_PER_MPS
        }
    ).to_rust()
    if key in list(dfs_cal.keys()):
        cal_sim_drives[key] = fsr.SimDriveHot(cycs[key], veh)
    else:
        assert key in list(dfs_val.keys())
        val_sim_drives[key] = fsr.SimDriveHot(cycs[key], veh)


# %%


objectives = fsim.calibration.ModelErrors(
    sim_drives=sim_drives,
    objectives=[
        ("Fuel_Power_Calc[kW]", "fs_kw_out_ach"),
    ],
    params=[
        ("vehthrm.fc_c_kj__k"),
        ("vehthrm.fc_l"),
        ("vehthrm.fc_htc_to_amb_stop"),
        ("vehthrm.fc_coeff_from_comb"),
    ],
    verbose=False
)

problem = fsim.calibration.CalibrationProblem(
    err=objectives,
    param_bounds=[
        (1.5, 3),
    ],
)

# %%
problem.minimize()

print(problem.res.X)
