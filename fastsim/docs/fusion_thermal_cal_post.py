# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from cycler import cycler

import fastsim as fsim
import fastsimrust as fsr

import fusion_thermal_cal as ftc

trip_dir = ftc.trip_dir

cal_objectives, val_objectives, params_bounds = ftc.get_cal_and_val_objs()
# save_path = "fusion_pymoo_res_2022_08_18"
save_path = "fusion_pymoo_res_2022_08_26"  # seems to be best
# save_path = "fusion_pymoo_res_2022_08_29"
# save_path = "pymoo_res"

res_df = pd.read_csv(Path(save_path) / "pymoo_res_df.csv")
# with open(Path(save_path) / "pymoo_res.pickle", 'rb') as file:
#     res = pickle.load(file)

res_df['euclidean'] = (
    res_df.iloc[:, len(cal_objectives.params):] ** 2).sum(1).pow(1/2)

best_row = res_df['euclidean'].argmin()
# print("WARNING: overriding `best_row`")
# best_row = 8
best_df = res_df.iloc[best_row, :]
res_df['fuel euclidean'] = (res_df.filter(
    like="fs_cumu") ** 2).sum(1) ** (1 / 2)
res_df['temp euclidean'] = (res_df.filter(
    like="fc_te") ** 2).sum(1) ** (1 / 2)
param_vals = res_df.iloc[best_row, :len(cal_objectives.params)].to_numpy()

# %%

show_plots = False

cal_errs, cal_mods = cal_objectives.get_errors(
    cal_objectives.update_params(param_vals),
    plot_save_dir=Path(save_path),
    show=show_plots,
    return_mods=True,
)
val_errs, val_mods = val_objectives.get_errors(
    val_objectives.update_params(param_vals),
    plot_save_dir=Path(save_path),
    show=show_plots,
    return_mods=True,
)

# %%

# generate data for scatter plot

l_per_cc = 1e-3
gal_per_l = 1 / 3.79
gal_per_ml = gal_per_l / 1_000


def get_mgp_from_sdh(sdh: fsr.SimDriveHot) -> float:
    fuel_gal_mod = (
        (np.array(sdh.sd.fs_kw_out_ach) * np.diff(np.array(sdh.sd.cyc.time_s), prepend=0)
         ).sum() / ftc.lhv_fuel_kj_per_kg / ftc.rho_fuel_kg_per_ml * gal_per_ml)
    dist_mi_mod = np.array(sdh.sd.dist_mi).sum()
    return dist_mi_mod / fuel_gal_mod


cal_mod_mpg = []
cal_exp_mpg = []
cal_te_amb_degc = []

for key in cal_mods.keys():
    sdh = cal_mods[key]
    cal_te_amb_degc.append(np.array(sdh.state.amb_te_deg_c).mean())
    mpg = get_mgp_from_sdh(sdh)
    cal_mod_mpg.append(mpg)
    df = cal_objectives.dfs[key]
    fuel_gal_exp = (df["Eng_FuelFlow_Direct[cc/s]"] *
                    df['Time[s]'].diff().fillna(0)).sum() * l_per_cc * gal_per_l
    dist_mi_exp = (df["Dyno_Spd[mph]"] / 3_600 *
                   df['Time[s]'].diff().fillna(0)).sum()
    cal_exp_mpg.append(dist_mi_exp / fuel_gal_exp)

val_mod_mpg = []
val_exp_mpg = []
val_te_amb_degc = []

for key in val_objectives.models.keys():
    sdh = val_mods[key]
    val_te_amb_degc.append(np.array(sdh.state.amb_te_deg_c).mean())
    mpg = get_mgp_from_sdh(sdh)
    val_mod_mpg.append(mpg)
    df = val_objectives.dfs[key]
    fuel_gal_exp = (df["Eng_FuelFlow_Direct[cc/s]"] *
                    df['Time[s]'].diff().fillna(0)).sum() * l_per_cc * gal_per_l
    dist_mi_exp = (df["Dyno_Spd[mph]"] / 3_600 *
                   df['Time[s]'].diff().fillna(0)).sum()
    val_exp_mpg.append(dist_mi_exp / fuel_gal_exp)


# %%

markers = ("x", "o", "x", "o")
colors = ("#7fc97f", "#7fc97f", "#beaed4", "#beaed4")

fig, ax = plt.subplots()
ax.scatter(cal_te_amb_degc, cal_mod_mpg, label='cal mod',
           marker=markers[0], color=colors[0])
ax.scatter(cal_te_amb_degc, cal_exp_mpg, label='cal exp',
           marker=markers[1], color=colors[1], facecolors='none')
ax.scatter(val_te_amb_degc, val_mod_mpg,
           marker=markers[2], color=colors[2], label='val mod')
ax.scatter(val_te_amb_degc, val_exp_mpg, label='val exp',
           marker=markers[3], color=colors[3], facecolors='none')
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel("Ambient Temp. [°C]")
ax.set_ylabel("Fuel Economy [mpg]")
ax.legend()
plt.tight_layout()
plt.savefig(Path(save_path) / "mpg v amb scatter.png")
plt.savefig(Path(save_path) / "mpg v amb scatter.svg")

# %%
mpg_sweep = np.linspace(0, max(max(cal_exp_mpg), max(val_exp_mpg)), 50)
err_p05 = mpg_sweep * 1.05
err_p10 = mpg_sweep * 1.10
err_n05 = mpg_sweep * 0.95
err_n10 = mpg_sweep * 0.90

fig, ax = plt.subplots()
ax.set_title("2012 Ford Fusion V6 Model Accuracy")
ax.scatter(cal_exp_mpg, cal_mod_mpg, label='calibration',
           color=colors[0])
ax.scatter(val_exp_mpg, val_mod_mpg,
           color=colors[2], label='validation')
plt.plot(mpg_sweep, err_p05, linestyle='-.', label='+/- 5%', color='black')
plt.plot(mpg_sweep, err_p10, linestyle='--', label='+/- 10%', color='black')
plt.plot(mpg_sweep, err_n05, linestyle='-.', color='black')
plt.plot(mpg_sweep, err_n10, linestyle='--', color='black')
# ax.set_ylim([0, ax.get_ylim()[1] * 1.1])
# ax.set_xlim([0, ax.get_xlim()[1] * 1.1])
ax.set_xlabel("Experiment FE [mpg]")
ax.set_ylabel("Model FE [mpg]")
ax.legend()
plt.tight_layout()
plt.savefig(Path(save_path) / "mpg scatter.png")
plt.savefig(Path(save_path) / "mpg scatter.svg")

# %%
