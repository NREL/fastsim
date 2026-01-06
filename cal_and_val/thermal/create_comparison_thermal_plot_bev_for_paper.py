
# download dyno data
# Find 0 degrees F UDDS cycle for the BEV -- choose whichever file in Line 114 is cold start cold ambient, make sure it is a single cycle
# run the BEV over the same speed trace as the UDDS cycle
# first, see if cal_bev.py runs

#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from copy import deepcopy

import fastsim as fsim
from fastsim.demos.plot_utils import (
    BASE_LINE_STYLES,
    figsize_3_stacked,
    get_paired_cycler,
    get_uni_cycler, 
)

# Test data columns
time_column = "Time[s]_RawFacilities"
speed_column = "Dyno_Spd[mph]"
cabin_temp_column = "Cabin_Driver_Headrest_Temp__C"
batt_temp_column = "HVBatt_pack_average_temp_HPCM2__C"
eng_clnt_temp_column = "engine_coolant_temp_PCAN__C"
cell_temp_column = "Cell_Temp[C]"
soc_column = "HVBatt_SOC_CAN4__per"

# Test data
test_data_folder = "dyno_test_data/D3 2020 Chevrolet Bolt/"
test_data_file_name = "62009051 Test Data.txt"
# test_data_file_name = "62009059 Test Data.txt"

def lighten_color(hex_color, percent=10):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    lighter_rgb = [
        min(255, int(c + (255 - c) * percent / 100)) for c in rgb
    ]
    return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)

# colors
three_colors = ['#4267ac','#52a675', '#ff595e']
three_colors_lighter = [lighten_color(c, percent=50) for c in three_colors]
two_colors = ['#4267ac', '#ff595e']

# Unit conversion constants
mps_per_mph = 0.447
celsius_to_kelvin_offset = 273.15

# other constants
pt_type_var = "BEV"
dashed_linewidth = 1.5

def df_to_cyc(df: pd.DataFrame) -> fsim.Cycle:
    cyc_dict = {
        "time_seconds": df[time_column].to_list(),
        "speed_meters_per_second": (df[speed_column] * mps_per_mph).to_list(),
        "temp_amb_air_kelvin": (df[cell_temp_column] + celsius_to_kelvin_offset).to_list(),
        # TODO: pipe solar load from `Cycle` into cabin thermal model
        # "pwr_solar_load_watts": df[],
    }
    return fsim.Cycle.from_pydict(cyc_dict, skip_init=False)

def resample_df(df: pd.DataFrame) -> pd.DataFrame:
    # filter out "before" time
    df = df[df[time_column] >= 0.0]
    dt = np.diff(df[time_column], prepend=1)
    df["cumu. dist [mph*s]"] = (dt * df[speed_column]).cumsum()
    init_speed = df[speed_column].iloc[0]
    df = df[::10]  # convert to ~1 Hz
    df.reset_index(inplace=True)
    dt_new = np.diff(df[time_column])
    df[speed_column] = np.concatenate(
        ([init_speed], np.diff(df["cumu. dist [mph*s]"]) / dt_new))
    df = df[df[time_column] < 2160]

    return df

# cycle using test data
dyno_data_df = pd.read_csv(test_data_folder + test_data_file_name, delimiter="\t")
test_data_cyc_df = resample_df(dyno_data_df)
test_data_cyc = df_to_cyc(test_data_cyc_df)

# load vehicle
veh = fsim.Vehicle.from_file(
    Path(__file__).parent / "f3-vehicles/2020 Chevrolet Bolt EV.yaml")

# setup initial conditions
veh_dict = veh.to_pydict()
veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    "temperature_kelvin"
] = dyno_data_df[batt_temp_column][0] + celsius_to_kelvin_offset
veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    "temp_prev_kelvin"
] = dyno_data_df[batt_temp_column][0] + celsius_to_kelvin_offset
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = dyno_data_df[cabin_temp_column][0] + celsius_to_kelvin_offset
veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = dyno_data_df[cabin_temp_column][0] + celsius_to_kelvin_offset
veh_deepcopy = deepcopy(veh_dict)
print("veh soc before update: ", veh_deepcopy['pt_type'][pt_type_var]['res']['state']['soc'])
veh_deepcopy['pt_type'][pt_type_var]['res']['state']['soc'] = dyno_data_df[soc_column][0] / 100.0
veh_deepcopy['pt_type'][pt_type_var]['res']['max_soc'] = dyno_data_df[soc_column][0] / 100.0

#%%
# simulate cycle

veh = fsim.Vehicle.from_pydict(veh_deepcopy, skip_init=False)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# instantiate `SimDrive` simulation object
sd0 = fsim.SimDrive(veh, test_data_cyc)
sd = sd0.copy()

sd_dict = sd.to_pydict()

# run simulation
sd.walk()

df = sd.to_dataframe()

# limit dyno data to match fastsim time range
dyno_data_df = dyno_data_df[
    dyno_data_df[time_column] <= df["cyc.time_seconds"][-1]
]

# plot figure
sns.set_theme()
fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize_3_stacked)

ax[0].set_prop_cycle(get_paired_cycler())
ax[0].plot(
    dyno_data_df[time_column],
    dyno_data_df[cabin_temp_column],
    label="Dyno Cabin Temp",
    color = three_colors_lighter[0],
    linestyle = 'solid',
)
ax[0].plot(
    dyno_data_df[time_column],
    dyno_data_df[batt_temp_column],
    label="Dyno Battery Temp",
    color = three_colors_lighter[1],
    linestyle = 'solid',
)
ax[0].plot(
    dyno_data_df[time_column],
    dyno_data_df[cell_temp_column],
    label="Dyno Ambient Temp",
    color = three_colors_lighter[2],
    linestyle = 'solid',
)
ax[0].plot(
    df["cyc.time_seconds"],
    df['veh.cabin.LumpedCabin.history.temperature_kelvin'] - celsius_to_kelvin_offset,
    label="FASTSim Cabin Temp",
    color = three_colors[0],
    # linewidth = dashed_linewidth,
)
ax[0].plot(
    df["cyc.time_seconds"],
    df['veh.pt_type.BEV.res.thrml.RESLumpedThermal.history.temperature_kelvin'] - celsius_to_kelvin_offset,
    label="FASTSim Battery Temp",
    color = three_colors[1],
    # linewidth = dashed_linewidth,
)
ax[0].plot(
    df["cyc.time_seconds"],
    df['cyc.temp_amb_air_kelvin'] - celsius_to_kelvin_offset,
    label="FASTSim Ambient Temp",
    color = three_colors[2],
    linestyle = 'dashed',
    # linewidth = dashed_linewidth,
)
ax[0].set_ylabel("Temp [°C]")
ax[0].legend()

ax[1].set_prop_cycle(get_paired_cycler())
ax[1].plot(
    np.array(dyno_data_df[time_column]),
    np.array(dyno_data_df[soc_column]),
    label="Dyno SOC",
    color = two_colors[1],
    linestyle = 'solid',
)
ax[1].plot(
    df["cyc.time_seconds"],
    df["veh.pt_type.BEV.res.history.soc"] * 100,
    label="FASTSim SOC",
    color = two_colors[0],
)
ax[1].set_ylabel("Batt. SOC [%]")
ax[1].legend()

ax[2].set_prop_cycle(get_paired_cycler())
ax[2].plot(
    np.array(dyno_data_df[time_column]),
    np.array(dyno_data_df[speed_column]),
    label="Dyno Speed",
    color = two_colors[1],
    linestyle = 'solid',
)
ax[2].plot(
    df["cyc.time_seconds"],
    df["veh.history.speed_ach_meters_per_second"] / mps_per_mph,
    label="FASTSim Speed",
    color = two_colors[0],
)
ax[2].legend()
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Speed [mph]")

plt.tight_layout()
plt.savefig(Path("./plots/res_pwr_" + test_data_file_name + ".svg"))
plt.show()