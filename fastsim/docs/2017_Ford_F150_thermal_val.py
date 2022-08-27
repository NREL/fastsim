from typing import *
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import argparse
import re

import fastsim as fsim
import fastsimrust as fsr

# TODO: check whether 4wd is engaged

f150_resampled_data_dir = Path(
    fsim.__file__).parent / "resources/DownloadableDynamometerDatabase/2017_Ford_F150_Ecoboost"

# raw data from https://www.anl.gov/taps/d3-2017-ford-f150
# f150_raw_data_dir = Path().home() / \
#     "winhome/Documents/DynoTestData/2017 Ford F150 Ecoboost/Extended Data Set/"

# dfs = {}

# RATE_VARS = ("Dyno_Spd[mph]", "Eng_FuelFlow_Direct[ccps]")

# for f in f150_raw_data_dir.iterdir():
#     raw_df = pd.read_csv(f, sep="\t")
#     df = fsim.resample(raw_df, rate_vars=RATE_VARS)
#     df.to_csv(f150_resampled_data_dir / (f.stem + ".csv"))

FC_TE_DEG_C_KEY = "Eng_cylinder_head_temperature_PCM[C]"
CAB_TE_DEG_C_KEY = "Cabin_Temp[C]"
AMB_TE_DEG_C_KEY = "Cell_Temp[C]"
SPEED_MPH_KEY = "Dyno_Spd[mph]"

# conversions
mps_per_mph = 1 / 2.237
rho_fuel_kg_per_ml = 0.743e-3
lhv_fuel_btu_per_lbm = 18_344  # from "2012FordFusionV6Overview V5.pdf"
lbm_per_kg = 2.2
btu_per_kj = 0.948
cc_per_liter = 1_000
liter_per_gal = 3.79
lhv_fuel_kj_per_kg = lhv_fuel_btu_per_lbm * lbm_per_kg / btu_per_kj

CYCLES_TO_USE = {
    # cold start, Heater controls per CFR -- not sure what this means
    "61706024": {"hvac_on": True},
    "61706021": {"hvac_on": True},  # cold start, heater on
    "61801007": {"hvac_on": False},  # cold start, heater apparently not on
    "61705016": {"hvac_on": False},  # warm ambient, warm start
    "61705017": {"hvac_on": False},  # warm ambient, warm start
    "61706012": {"hvac_on": True},  # hot ambient, cold start
    "61706013": {"hvac_on": True},  # hot ambient, warm start
    "61706019": {"hvac_on": True},  # hot ambient, warm start
}


def load_resampled_data() -> Dict[str, pd.DataFrame]:
    dfs = dict()
    for file in f150_resampled_data_dir.iterdir():
        key = file.stem.split()[0]
        # skip cycles that are not in CYCLES_TO_USE
        if not np.array([cyc_to_use == key for cyc_to_use in CYCLES_TO_USE.keys()]).any():
            continue
        print(f"loading: ", file.resolve())
        df = pd.read_csv(file)
        df = df[df['Time[s]'] > 0.0]
        # clip time at zero seconds
        df = df[df
                ['Time[s]'] >= 0.0]

        df['Fuel_Power_Calc[kW]'] = df["Eng_FuelFlow_Direct[ccps]"] * \
            rho_fuel_kg_per_ml * lhv_fuel_kj_per_kg

        df['Fuel Cumu. [Gal.]'] = (
            df["Eng_FuelFlow_Direct[ccps]"] / cc_per_liter / liter_per_gal *
            df['Time[s]'].diff().fillna(0.0)).cumsum()

        df['Fuel_Energy_Calc[MJ]'] = (
            df['Fuel_Power_Calc[kW]'] *
            df['Time[s]'].diff().fillna(0.0)
        ).cumsum() / 1e3

        df["Tractive Power [kW]"] = df['Dyno_TractiveForce[N]'] * \
            df[SPEED_MPH_KEY] * mps_per_mph / 1_000
        dfs[key] = df.copy()

    return dfs


dfs = load_resampled_data()

f_150 = fsim.vehicle.Vehicle.from_file(
    "2017_Ford_F-150_Ecoboost.csv").to_rust()
fusion_thermal = fsr.VehicleThermal.from_file(
    str(fsim.vehicle.VEHICLE_DIR / "thermal/2012_Ford_Fusion_thrml.yaml"))
f_150_thermal = fusion_thermal
f_150_thermal.fc_c_kj__k *= 3.5 / 3.5  # fusion is a 3.5 L v6!

sdh_dict = {}
fc_init_te_deg_c_arr = {}
amb_te_deg_c_arr = {}
mpg_test_dict = {}

for key, df in dfs.items():
    fc_init_te_deg_c_arr[key] = df.iloc[0][FC_TE_DEG_C_KEY]
    amb_te_deg_c_arr[key] = df[AMB_TE_DEG_C_KEY].mean()
    init_thermal_state = fsr.ThermalState(
        amb_te_deg_c=df.iloc[0][AMB_TE_DEG_C_KEY],
        fc_te_deg_c_init=df.iloc[0][FC_TE_DEG_C_KEY],
        cab_te_deg_c_init=df.iloc[0][CAB_TE_DEG_C_KEY],
    )

    cyc = fsim.cycle.Cycle.from_dict({
        "time_s": df['Time[s]'],
        "mps": df[SPEED_MPH_KEY] * mps_per_mph,
    }).to_rust()

    sdh = fsr.SimDriveHot(
        cyc, 
        f_150, 
        f_150_thermal,
        init_thermal_state, 
        amb_te_deg_c=df[AMB_TE_DEG_C_KEY]
    )
    sdh.sim_drive()
    sdh_dict[key] = sdh

    mpg_test_dict[key] = df.iloc[-1]['Distance[mi]'] / \
        df.iloc[-1]['Fuel Cumu. [Gal.]']

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(sdh.sd.cyc.time_s, np.array(sdh.sd.fs_cumu_mj_out_ach) *
               1e3 / lhv_fuel_kj_per_kg / rho_fuel_kg_per_ml / cc_per_liter / liter_per_gal)
    ax[0].plot(df['Time[s]'], df["Fuel Cumu. [Gal.]"])
    ax[0].set_ylabel("Cumu. Fuel\nEnergy [MJ]")

    mod_enrgy_tract_cumu_mj = (
        np.array(sdh.sd.cyc_trans_kw_out_req) / 1e3 * np.diff(
            sdh.sd.cyc.time_s, prepend=0.0)).cumsum()
    exp_enrgy_tract_cumu_mj = (
        df["Tractive Power [kW]"] * df['Time[s]'].diff().fillna(0.0) / 1e3).cumsum()

    ax[1].plot(sdh.sd.cyc.time_s, mod_enrgy_tract_cumu_mj)
    ax[1].plot(df['Time[s]'], exp_enrgy_tract_cumu_mj)
    ax[1].set_ylabel("Cumu. Tractive Energy [MJ]")

    ax[-1].plot(sdh.sd.cyc.time_s, sdh.sd.mph_ach, label='mod')
    ax[-1].plot(df['Time[s]'], df[SPEED_MPH_KEY], label='dyno')
    ax[-1].set_xlabel('Time [s]')
    ax[-1].set_ylabel("Speed [mph]")
    ax[-1].legend()


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(fc_init_te_deg_c_arr.values(), [
           sdh.sd.mpgge for sdh in sdh_dict.values()], marker='x')
ax.scatter(fc_init_te_deg_c_arr.values(), [
           mpg_test for mpg_test in mpg_test_dict.values()])
ax.set_xlabel("Engine Init Temp. [°C]")
ax.set_ylabel("Fuel Economy [mpg]")
