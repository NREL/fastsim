# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# local
import fastsim as fsim
from cal_bev import cal_mod_obj, val_mod_obj, save_path,  cyc_files_dict 
from cal_bev import time_column, speed_column, cell_temp_column 
from cal_bev import  mps_per_mph 

res_df = pd.read_csv(save_path / "pymoo_res_df.csv")
# TODO: change all the fuel stuff to reference SOC
res_df_fuel_energy = res_df.filter(regex="get_mod_energy_fuel")
res_df_fuel_energy_summed = res_df.filter(
    regex="get_mod_energy_fuel").sum(1)
best_row_fuel_energy = res_df_fuel_energy_summed.argmin()
param_vals_fuel_energy = res_df.iloc[
    best_row_fuel_energy,
    :len(cal_mod_obj.param_fns)].to_numpy()

param_vals_soc = res_df.iloc[
    best_row_soc,
    :len(cal_mod_obj.param_fns)].to_numpy()

best_row = res_df["euclidean"].argmin()
best_df = res_df.iloc[best_row, :]
param_vals_euclidean = res_df.iloc[
    best_row,
    :len(cal_mod_obj.param_fns)].to_numpy()

param_vals_best = param_vals_fuel_energy

# getting the solved models
(errors_cal, cvs_cal, sds_cal_solved, sds_cal) = cal_mod_obj.get_errors(
    sim_drives=cal_mod_obj.update_params(param_vals_best),
    return_mods=True,
    )
(errors_val, cvs_val, sds_val_solved, sds_val) = val_mod_obj.get_errors(
     sim_drives=val_mod_obj.update_params(param_vals_best),
     return_mods=True,
     )

 # %%

 # plotting
plot_save_path = save_path / "plots"
plot_save_path.mkdir(exist_ok=True)

for ((key, df_cal), (sd_key, sd_cal)) in zip(cal_mod_obj.dfs.items(), sds_cal_solved.items()):
    if not isinstance(sd_cal, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key
    df_cal = df_cal[:len(sd_cal['veh']['history']['time_seconds'])]

    for obj_fn in cal_mod_obj.obj_fns:
        fig, ax = plt.subplots(2, 1, sharex=True)
        cell_temp = next(iter(
            [v[cell_temp_column]
                for k, v in cyc_files_dict.items() if k.replace(".txt", "") == key]
        ))
        fig.suptitle(f"{key}\ncell temp [*C]: {cell_temp}, calibration")
        ax[0].plot(
            sd_cal['veh']['history']['time_seconds'],
            obj_fn[0](sd_cal),
            label='mod',
        )
        ax[0].plot(
            df_cal[time_column],
            obj_fn[1](df_cal),
            label='exp',
        )
        ax[0].legend()
        ax[0].set_ylabel(obj_fn[0].__name__.replace('get_mod_', ''))

        ax[1].plot(
            sd_cal['veh']['history']['time_seconds'],
            sd_cal['veh']['history']['speed_ach_meters_per_second'],
            label='mod',
        )
        ax[1].plot(
            df_cal[time_column],
            df_cal[speed_column] * mps_per_mph,
            label="exp",
        )
        ax[1].legend()
        ax[1].set_ylabel("Speed [m/s]")
        plt.savefig(plot_save_path / f"{key}_{obj_fn[0].__name__.replace('get_mod_', '')}_cal.svg")

for ((key, df_val), (sd_key, sd_val)) in zip(val_mod_obj.dfs.items(), sds_val_solved.items()):
    if not isinstance(sd_val, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key

    df_val = df_val[:len(sd_val['veh']['history']['time_seconds'])]

    for obj_fn in val_mod_obj.obj_fns:
        fig, ax = plt.subplots(2, 1, sharex=True)
        cell_temp = next(iter(
            [v[cell_temp_column]
                for k, v in cyc_files_dict.items() if k.replace(".txt", "") == key]
        ))
        fig.suptitle(f"{key}\ncell temp [*C]: {cell_temp}, validation")
        ax[0].plot(
            sd_val['veh']['history']['time_seconds'],
            obj_fn[0](sd_val),
            label='mod',
        )
        ax[0].plot(
            df_val[time_column],
            obj_fn[1](df_val),
            label='exp',
        )
        ax[0].legend()
        ax[0].set_ylabel(obj_fn[0].__name__.replace('get_mod_', ''))

        ax[1].plot(
            sd_val['veh']['history']['time_seconds'],
            sd_val['veh']['history']['speed_ach_meters_per_second'],
            label='mod',
        )
        ax[1].plot(
            df_val[time_column],
            df_val[speed_column] * mps_per_mph,
            label='exp',
        )
        ax[1].legend()
        ax[1].set_ylabel("Speed [m/s]")
        plt.savefig(plot_save_path / f"{key}_{obj_fn[0].__name__.replace('get_mod_', '')}_val.svg")
# %%
# function for plot formatting
def draw_error_zones(ax):
    """Draw 0%, ±5%, ±10% error regions on MPL Axes object"""
    xl, xu = ax.get_xlim()
    yl, yu = ax.get_ylim()
    l = min(xl, yl)
    u = max(xu, yu)
    lims = np.array([0, u * 1.01])

    # Plot 0% error diagonalx
    ax.plot(lims, lims, linestyle="dotted", color="g", label="0% error")

    # Plot ±5%, ±10% error regions with transparencies
    counter = 0
    error_1 = 0
    error_2 = 0
    error_3 = 0
    for err, alpha in zip((0.05, 0.10, 0.15), (0.35, 0.2, 0.15)):
        error = ax.fill_between(
            lims,
            lims * (1 - err),
            lims * (1 + err),
            alpha=alpha,
            color="g",
            label=f"±{err*100:.0f}% error",
        )

    ax.set_xlim(left=l, right=u)
    ax.set_ylim(bottom=l, top=u)
    # ax.legend(loc="lower right", framealpha=0.5, fontsize=8, borderpad=0.25)

    return error

# %%
# Scatter plots with temperature effects

fuel_energy_exp_cal = []
fuel_energy_mod_cal = []
for ((key, df_cal), (sd_key, sd_cal)) in zip(cal_mod_obj.dfs.items(), sds_cal_solved.items()):
    if not isinstance(sd_cal, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key

    df_cal = df_cal[:len(sd_cal['veh']['history']['time_seconds'])]

    mod_energy_fuel = get_mod_energy_fuel_megajoules(sd_cal)
    exp_energy_fuel = get_exp_energy_fuel_megajoules(df_cal)
    assert len(mod_energy_fuel) == len(exp_energy_fuel)

    fuel_energy_mod_cal.append(
        mod_energy_fuel[-1]
    )
    fuel_energy_exp_cal.append(
        exp_energy_fuel.iloc[-1]
    )

fuel_energy_exp_val = []
fuel_energy_mod_val = []

for ((key, df_val), (sd_key, sd_val)) in zip(val_mod_obj.dfs.items(), sds_val_solved.items()):
    if not isinstance(sd_val, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key

    df_val = df_val[:len(sd_val['veh']['history']['time_seconds'])]

    mod_energy_fuel = get_mod_energy_fuel_megajoules(sd_val)
    exp_energy_fuel = get_exp_energy_fuel_megajoules(df_val)
    assert len(mod_energy_fuel) == len(exp_energy_fuel)

    fuel_energy_mod_val.append(
        mod_energy_fuel[-1]
    )
    fuel_energy_exp_val.append(
        exp_energy_fuel.iloc[-1]
    )


fig, ax = plt.subplots()
fig.suptitle("Model v. Test Data With Thermal Effects")
ax.scatter(
    fuel_energy_exp_cal,
    fuel_energy_mod_cal,
    label='cal',
)
ax.scatter(
    fuel_energy_exp_val,
    fuel_energy_mod_val,
    label='val',
)
draw_error_zones(ax)
ax.set_xlabel("Test Data Fuel Used [MJ]")
ax.set_ylabel("FASTSim Fuel Used [MJ]")
ax.legend()
plt.savefig(plot_save_path / "scatter with thrml effects.svg")

# %%

# Scatter plots without temperature effects

fuel_energy_mod_cal_no_thrml = []
fuel_energy_exp_cal_no_thrml = []
for ((key, df_cal), (sd_key, sd_cal)) in zip(cal_mod_obj.dfs.items(), sds_cal.items()):
    if not isinstance(sd_cal, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key

    sd_cal_no_thrml = deepcopy(sd_cal)

    sd_cal_no_thrml['veh']['hvac'] = 'None'
    sd_cal_no_thrml['veh']['cabin'] = 'None'
    sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['fc']['thrml'] = 'None'
    sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res']['thrml'] = 'None'
    res = fsim.ReversibleEnergyStorage.from_pydict(
        sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res'], skip_init=False)
    res.set_default_pwr_interp()
    sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res'] = res.to_pydict()
    sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['temp_fc_allowed_off_kelvin'] = None
    sd_cal_no_thrml['veh']['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['temp_fc_forced_on_kelvin'] = None

    sd_cal_no_thrml = fsim.SimDrive.from_pydict(
        sd_cal_no_thrml, skip_init=False)
    try:
        sd_cal_no_thrml.walk_once()
    except Exception:
        pass
    sd_cal_no_thrml = sd_cal_no_thrml.to_pydict()

    df_cal = df_cal[:len(sd_cal_no_thrml['veh']['history']['time_seconds'])]

    mod_energy_fuel = get_mod_energy_fuel_megajoules(sd_cal_no_thrml)
    exp_energy_fuel = get_exp_energy_fuel_megajoules(df_cal)
    assert len(mod_energy_fuel) == len(exp_energy_fuel)

    fuel_energy_mod_cal_no_thrml.append(
        mod_energy_fuel[-1]
    )
    fuel_energy_exp_cal_no_thrml.append(
        exp_energy_fuel.iloc[-1]
    )

fuel_energy_exp_val_no_thrml = []
fuel_energy_mod_val_no_thrml = []
for ((key, df_val), (sd_key, sd_val)) in zip(val_mod_obj.dfs.items(), sds_val.items()):
    if not isinstance(sd_val, dict):
        print(f"skipping {key}")
        continue
    assert key == sd_key

    sd_val_no_thrml = deepcopy(sd_val)

    sd_val_no_thrml['veh']['hvac'] = 'None'
    sd_val_no_thrml['veh']['cabin'] = 'None'
    sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['fc']['thrml'] = 'None'
    sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res']['thrml'] = 'None'
    res = fsim.ReversibleEnergyStorage.from_pydict(
        sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res'], skip_init=False)
    res.set_default_pwr_interp()
    sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['res'] = res.to_pydict()
    sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['temp_fc_allowed_off_kelvin'] = None
    sd_val_no_thrml['veh']['pt_type']['HybridElectricVehicle']['pt_cntrl']['RGWDB']['temp_fc_forced_on_kelvin'] = None
    sd_val_no_thrml = fsim.SimDrive.from_pydict(
        sd_val_no_thrml, skip_init=False)
    try:
        sd_val_no_thrml.walk_once()
    except Exception:
        pass
    sd_val_no_thrml = sd_val_no_thrml.to_pydict()

    df_val = df_val[:len(sd_val_no_thrml['veh']['history']['time_seconds'])]

    mod_energy_fuel = get_mod_energy_fuel_megajoules(sd_val_no_thrml)
    exp_energy_fuel = get_exp_energy_fuel_megajoules(df_val)
    assert len(mod_energy_fuel) == len(exp_energy_fuel)

    fuel_energy_mod_val_no_thrml.append(
        mod_energy_fuel[-1]
    )
    fuel_energy_exp_val_no_thrml.append(
        exp_energy_fuel.iloc[-1]
    )

fig, ax = plt.subplots()
fig.suptitle("Model v. Test Data Without Thermal Effects")
ax.scatter(
    fuel_energy_exp_cal,
    fuel_energy_mod_cal_no_thrml,
    label='cal',
)
ax.scatter(
    fuel_energy_exp_val,
    fuel_energy_mod_val_no_thrml,
    label='val',
)
draw_error_zones(ax)
ax.set_xlabel("Test Data Fuel Used [MJ]")
ax.set_ylabel("FASTSim Fuel Used [MJ]")
ax.legend()
plt.savefig(plot_save_path / "scatter without thrml effects.svg")

# %%
