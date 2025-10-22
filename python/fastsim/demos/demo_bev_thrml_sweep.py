"""BEV thermal demo with all conditions."""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

import fastsim as fsim

sns.set_theme()


# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"
# if running file to get values for paper
RUNNING_FOR_ABSTRACT = False

#color palettes
# colors = ['#ff595e', '#ff924c', '#ffca3a', '#c5ca30', '#8ac926', '#52a675', '#1982c4', '#4267ac', '#6a4c93', '#d677b8']
colors = ['#6a4c93', '#4267ac', '#1982c4', '#52a675', '#8ac926', '#c5ca30', '#ffca3a', '#ff924c', '#ff595e', '#d677b8']

def lighten_color(hex_color, percentage=50):
    """Lighten a hex color by a given percentage."""
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    lighter_rgb = [
        min(255, int(c + (255 - c) * (percentage / 100))) for c in rgb
    ]
    return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)

lighter_colors = [lighten_color(color) for color in colors]

celsius_to_kelvin = 273.15
temp_amb_and_init = -6.7 + celsius_to_kelvin
# temperatures for comparisons in fastsim paper
cold_amb = -7.0
warm_amb = 22.0
hot_amb = 40.0
hot_start = 45.0
baseline_temp = 22.0
# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%


def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


# array of ambient temperatures in kelvin
te_amb_arr_k: list[float] = ([t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, 20)] + [22.0 + celsius_to_kelvin]) if RUNNING_FOR_ABSTRACT else [t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, 20)]
# array of init temperatures in kelvin
te_batt_and_cab_init_arr_k: list[float] = ([
    t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, 20)
] + [22.0 + celsius_to_kelvin]) if RUNNING_FOR_ABSTRACT else [
    t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, 20)
]

cyc_key = "cycle"
te_amb_key = "te_amb [°C]"
te_init_key = "te_init [°C]"
ecr_key = "ECR [kW-hr/100mi]"


def sweep(te_amb_arr, te_init_arr):
    """Sweep ambient and initial conditions"""
    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)

    res_list = []
    for cyc_str in ["udds", "hwfet"]:
        for te_amb_k in te_amb_arr:
            for te_init_k in te_init_arr:
                cyc = fsim.Cycle.from_resource(cyc_str + ".csv")
                cyc_dict = cyc.to_pydict()
                cyc_dict["temp_amb_air_kelvin"] = [te_amb_k] * cyc.len()
                cyc = fsim.Cycle.from_pydict(cyc_dict)

                # setup initial conditions
                veh_dict = veh.to_pydict()
                veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
                    "temperature_kelvin"
                ] = te_init_k
                veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
                    "temp_prev_kelvin"
                ] = te_init_k
                veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = te_init_k
                veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = te_init_k

                # simulate cycle
                veh = fsim.Vehicle.from_pydict(veh_dict)
                sd = fsim.SimDrive(veh, cyc, None)
                try_walk(sd, f"`sd_prep`, te_amb: {te_amb_k}, te_init: {te_init_k}")
                veh_dict_solved = sd.to_pydict()["veh"]

                new_row = {
                    cyc_key: cyc_str,
                    te_amb_key: te_amb_k - celsius_to_kelvin,
                    te_init_key: te_init_k - celsius_to_kelvin,
                    ecr_key: veh_dict_solved["pt_type"]["BEV"]["res"]["state"][
                        "energy_out_chemical_joules"
                    ]
                    / 1_000
                    / 3_600
                    / (veh_dict_solved["state"]["dist_meters"] / 1e3 / 1.61)
                    * 100.0,
                }
                res_list.append(new_row)

    df_res = pd.DataFrame(res_list)

    return df_res


df_res = sweep(te_amb_arr_k, te_batt_and_cab_init_arr_k)
df_baseline_comp = sweep([baseline_temp + celsius_to_kelvin], [baseline_temp + celsius_to_kelvin])

if RUNNING_FOR_ABSTRACT:
    # printing specific percent differences in Energy Consumption Rate (ECR) used for paper
    # energy consumption at 22 degrees C start and ambient for comparison
    ecr_comparison = df_res[(df_res[cyc_key] == "udds") & (df_res[te_amb_key] == warm_amb) & (df_res[te_init_key] == warm_amb)][ecr_key].values[0]
    print("ecr for 22 degrees start and ambient, for comparison:", ecr_comparison)
    # percent energy consumption of cold start (-7 degrees C) cold ambient (-7 degrees C), to baseline of 22 degrees start and ambient
    ecr_cold_start_cold_amb = df_res[(df_res[cyc_key] == "udds") & (df_res[te_amb_key] == cold_amb) & (df_res[te_init_key] == cold_amb)][ecr_key].values[0]
    print("ecr_cold_start_cold_amb:", ecr_cold_start_cold_amb)
    ecr_cold_start_cold_amb_perc_diff = (ecr_cold_start_cold_amb - ecr_comparison) / ecr_comparison * 100.0
    print(f"UDDS ECR percent difference for cold start (-7C) and cold ambient (-7C) vs. 22C start and ambient: {ecr_cold_start_cold_amb_perc_diff:.1f}%")
    # percent energy consumption of warm start (22 degrees C) cold ambient (-7 degrees C), to baseline of 22 degrees start and ambient
    ecr_warm_start_cold_amb = df_res[(df_res[cyc_key] == "udds") & (df_res[te_amb_key] == cold_amb) & (df_res[te_init_key] == warm_amb)][ecr_key].values[0]
    print("ecr_warm_start_cold_amb:", ecr_warm_start_cold_amb)
    ecr_warm_start_cold_amb_perc_diff = (ecr_warm_start_cold_amb - ecr_comparison) / ecr_comparison * 100.0
    print(f"UDDS ECR percent difference for warm start (22C) and cold ambient (-7C) vs. 22C start and ambient: {ecr_warm_start_cold_amb_perc_diff:.1f}%")
    # percent energy consumption of hot start (45 degrees C) hot ambient (40 degrees C), to baseline of 22 degrees start and ambient
    ecr_hot_start_hot_amb = df_res[(df_res[cyc_key] == "udds") & (df_res[te_amb_key] == hot_amb) & (df_res[te_init_key] == hot_start)][ecr_key].values[0]
    print("ecr_hot_start_hot_amb:", ecr_hot_start_hot_amb)
    ecr_hot_start_hot_amb_perc_diff = (ecr_hot_start_hot_amb - ecr_comparison) / ecr_comparison * 100.0
    print(f"UDDS ECR percent difference for hot start (45C) and hot ambient (45C) vs. 22C start and ambient: {ecr_hot_start_hot_amb_perc_diff:.1f}%")
    # percent energy consumption of warm start (22 degrees C) hot ambient (40 degrees C), to baseline of 22 degrees start and ambient
    ecr_warm_start_hot_amb = df_res[(df_res[cyc_key] == "udds") & (df_res[te_amb_key] == hot_amb) & (df_res[te_init_key] == warm_amb)][ecr_key].values[0]
    print("ecr_warm_start_hot_amb:", ecr_warm_start_hot_amb)
    ecr_warm_start_hot_amb_perc_diff = (ecr_warm_start_hot_amb - ecr_comparison) / ecr_comparison * 100.0
    print(f"UDDS ECR percent difference for warm start (22C) and hot ambient (45C) vs. 22C start and ambient: {ecr_warm_start_hot_amb_perc_diff:.1f}%")

def get_reasonable_init_ranges(te_amb_deg_c: float, init_temp_list) -> tuple[list[float], list[float]]:
    print("starting temp: ", te_amb_deg_c)
    if (round(te_amb_deg_c, 1) == -7.0) | (round(te_amb_deg_c, 1) == -2.1):
        dotted_indices = np.where(init_temp_list >= 27.0)
        solid_indices = np.where(init_temp_list <= 27.0)
    elif round(te_amb_deg_c, 1) == 2.9:
        dotted_indices = np.where((init_temp_list <= -2) | (init_temp_list >= 27.0))
        solid_indices = np.where((init_temp_list >= -2) & (init_temp_list <= 27.0))
    elif round(te_amb_deg_c, 1) == 7.8:
        dotted_indices = np.where((init_temp_list <= 3) | (init_temp_list >= 27.0))
        solid_indices = np.where((init_temp_list >= 3) & (init_temp_list <= 27.0))
    elif round(te_amb_deg_c, 1) == 12.8:
        dotted_indices = np.where((init_temp_list <= 7) | (init_temp_list >= 27.0))
        solid_indices = np.where((init_temp_list >= 7) & (init_temp_list <= 27.0))
    elif round(te_amb_deg_c, 1) == 17.7:
        dotted_indices = np.where((init_temp_list <= 12) | (init_temp_list >= 27.0))
        solid_indices = np.where((init_temp_list >= 12) & (init_temp_list <= 27.0))
    elif round(te_amb_deg_c, 1) == 22.7:
        dotted_indices = np.where((init_temp_list <= 17) | (init_temp_list >= 27.0))
        solid_indices = np.where((init_temp_list >= 17) & (init_temp_list <= 27.0))
    elif round(te_amb_deg_c, 1) == 27.6:
        dotted_indices = np.where((init_temp_list <= 17) | (init_temp_list >= 31.0))
        solid_indices = np.where((init_temp_list >= 17) & (init_temp_list <= 31.0))
    elif round(te_amb_deg_c, 1) == 32.6:
        dotted_indices = np.where((init_temp_list <= 17) | (init_temp_list >= 35.0))
        solid_indices = np.where((init_temp_list >= 17) & (init_temp_list <= 35.0))
    elif round(te_amb_deg_c, 1) == 37.5:
        dotted_indices = np.where((init_temp_list <= 17) | (init_temp_list >= 41.0))
        solid_indices = np.where((init_temp_list >= 17) & (init_temp_list <= 41.0))
    else:
        print("temp is none of above temps: ", te_amb_deg_c)
    return (dotted_indices, solid_indices)

def get_reasonable_amb_ranges(te_init_deg_c: float, amb_temp_list):
    print("starting temp: ", te_init_deg_c)
    if round(te_init_deg_c, 1) == -7.0:
        dotted_indices = np.where(amb_temp_list > -3)
        solid_indices = np.where(amb_temp_list <= -3)    
    elif round(te_init_deg_c, 1) == -1.5:
        dotted_indices = np.where(amb_temp_list > 3)
        solid_indices = np.where(amb_temp_list <= 3)   
    elif round(te_init_deg_c, 1) == 3.9:
        dotted_indices = np.where(amb_temp_list > 8)
        solid_indices = np.where(amb_temp_list <= 8)  
    elif round(te_init_deg_c, 1) == 9.4:
        dotted_indices = np.where(amb_temp_list > 13)
        solid_indices = np.where(amb_temp_list <= 13)       
    elif round(te_init_deg_c, 1) == 14.9:
        dotted_indices = np.where(amb_temp_list > 19)
        solid_indices = np.where(amb_temp_list <= 19)    
    elif round(te_init_deg_c, 1) == 20.4:
        dotted_indices = np.where(amb_temp_list > 31)
        solid_indices = np.where(amb_temp_list <= 31)    
    elif round(te_init_deg_c, 1) == 25.8:
        dotted_indices = []
        solid_indices = [index for index, _ in enumerate(amb_temp_list)]
    elif round(te_init_deg_c, 1) == 31.3:
        dotted_indices = np.where(amb_temp_list < 25)
        solid_indices = np.where(amb_temp_list >= 25)      
    elif round(te_init_deg_c, 1) == 36.8:
        dotted_indices = np.where(amb_temp_list < 31)
        solid_indices = np.where(amb_temp_list >= 31)   
    elif round(te_init_deg_c, 1) == 42.3:
        dotted_indices = np.where(amb_temp_list < 36)
        solid_indices = np.where(amb_temp_list >= 36)  
    else:
        print("temp is none of above temps: ", te_init_deg_c)
    return (dotted_indices, solid_indices) 

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"

# if environment var `SAVE_PLOTS=true` is set, plots are saved
SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "true").lower() == "true"

# plot ECR v. init for a sweep of amb
te_amb_step = int(len(te_amb_arr_k) / 10)
te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

fig, ax = plt.subplots()
fig.suptitle("UDDS ECR v. Init. and Amb. Temp.")
index = 0
for te_amb_deg_c in te_amb_short_deg_c:
    dotted_init_temps, solid_init_temps = get_reasonable_init_ranges(te_amb_deg_c, np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ]))
    ax.plot(
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ],
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ],
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax.plot(
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ])[solid_init_temps],
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ])[solid_init_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_amb_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax.set_xlabel("Cab. and Batt. Init. Temp. [°C]")
ax.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax.legend(title="te_amb [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig.savefig(Path(__file__).parent / "UDDS ECR v. Init. and Amb. Temp.svg")

if SHOW_PLOTS:
    plt.show()

fig1, ax1 = plt.subplots()
fig1.suptitle("HWFET ECR v. Init. and Amb. Temp.")
index = 0
for te_amb_deg_c in te_amb_short_deg_c:
    dotted_init_temps, solid_init_temps = get_reasonable_init_ranges(te_amb_deg_c, np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ]))
    ax1.plot(
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ],
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ],
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax1.plot(
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ])[solid_init_temps],
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ])[solid_init_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_amb_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax1.set_xlabel("Cab. and Batt. Init. Temp. [°C]")
ax1.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax1.legend(title="te_amb [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig1.savefig(Path(__file__).parent / "HWFET ECR v. Init. and Amb. Temp.svg")

if SHOW_PLOTS:
    plt.show()

te_init_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k]


te_init_step = int(len(te_amb_arr_k) / 10)
te_init_short_deg_c = [te_init_k - celsius_to_kelvin for te_init_k in te_batt_and_cab_init_arr_k][
    ::te_init_step
]

# plot ECR v. amb for a sweep of init
fig2, ax2 = plt.subplots()
fig2.suptitle("UDDS ECR v. Amb. and Init. Temp.")
index = 0
for te_init_deg_c in te_init_short_deg_c:
    dotted_amb_temps, solid_amb_temps = get_reasonable_amb_ranges(te_init_deg_c, np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ]))
    ax2.plot(
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ],
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ],
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax2.plot(
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ])[solid_amb_temps],
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ])[solid_amb_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_init_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax2.set_xlabel("Ambient Temp. [°C]")
ax2.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax2.legend(title="te_init [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig2.savefig(Path(__file__).parent / "UDDS ECR v. Amb. and Init. Temp.svg")

if SHOW_PLOTS:
    plt.show()

fig3, ax3 = plt.subplots()
fig3.suptitle("HWFET ECR v. Amb. and Init. Temp.")
index = 0
for te_init_deg_c in te_init_short_deg_c:
    dotted_amb_temps, solid_amb_temps = get_reasonable_amb_ranges(te_init_deg_c, np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ]))
    ax3.plot(
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ],
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ],
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax3.plot(
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ])[solid_amb_temps],
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ])[solid_amb_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_init_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax3.set_xlabel("Ambient Temp. [°C]")
ax3.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax3.legend(title="te_init [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig3.savefig(Path(__file__).parent / "HWFET ECR v. Amb. and Init. Temp.svg")

if SHOW_PLOTS:
    plt.show()

# plot ECR v. init for a sweep of amb using relative percentages rather than absolute temperatures
ecr_comparison_udds = df_baseline_comp[(df_baseline_comp[cyc_key] == "udds") & (df_baseline_comp[te_amb_key] == baseline_temp) & (df_baseline_comp[te_init_key] == baseline_temp)][ecr_key].values[0]

te_amb_step = int(len(te_amb_arr_k) / 10)
te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

fig, ax = plt.subplots()
fig.suptitle("UDDS ECR v. Init. and Amb. Temp.")
index = 0
for te_amb_deg_c in te_amb_short_deg_c:
    dotted_init_temps, solid_init_temps = get_reasonable_init_ranges(te_amb_deg_c, np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ]))
    ax.plot(
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ],
        (df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_udds) / ecr_comparison_udds * 100.0,
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax.plot(
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [°C]"
        ])[solid_init_temps],
        np.array((df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_udds) / ecr_comparison_udds * 100.0)[solid_init_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_amb_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax.set_xlabel("Cab. and Batt. Init. Temp. [°C]")
ax.set_ylabel("Percent Energy Consumption\nCompared to 22°C Baseline [%]")
ax.legend(title="te_amb [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig.savefig(Path(__file__).parent / "UDDS ECR v. Init. and Amb. Temp Percentage Difference.svg")

if SHOW_PLOTS:
    plt.show()

ecr_comparison_hwfet = df_baseline_comp[(df_baseline_comp[cyc_key] == "hwfet") & (df_baseline_comp[te_amb_key] == baseline_temp) & (df_baseline_comp[te_init_key] == baseline_temp)][ecr_key].values[0]

fig1, ax1 = plt.subplots()
fig1.suptitle("HWFET ECR v. Init. and Amb. Temp.")
index = 0
for te_amb_deg_c in te_amb_short_deg_c:
    dotted_init_temps, solid_init_temps = get_reasonable_init_ranges(te_amb_deg_c, np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ]))
    ax1.plot(
        df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ],
        (df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_hwfet) / ecr_comparison_hwfet * 100.0,
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax1.plot(
        np.array(df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [°C]"
        ])[solid_init_temps],
        np.array((df_res[(df_res["te_amb [°C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_hwfet) / ecr_comparison_hwfet * 100.0)[solid_init_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_amb_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax1.set_xlabel("Cab. and Batt. Init. Temp. [°C]")
ax1.set_ylabel("Percent Energy Consumption\nCompared to 22°C Baseline [%]")
ax1.legend(title="te_amb [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig1.savefig(Path(__file__).parent / "HWFET ECR v. Init. and Amb. Temp Percentage Difference.svg")

if SHOW_PLOTS:
    plt.show()

te_init_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k]


te_init_step = int(len(te_amb_arr_k) / 10)
te_init_short_deg_c = [te_init_k - celsius_to_kelvin for te_init_k in te_batt_and_cab_init_arr_k][
    ::te_init_step
]

# plot ECR v. amb for a sweep of init as percent to baseline rather than absolute
fig2, ax2 = plt.subplots()
fig2.suptitle("UDDS ECR v. Amb. and Init. Temp.")
index = 0
for te_init_deg_c in te_init_short_deg_c:
    dotted_amb_temps, solid_amb_temps = get_reasonable_amb_ranges(te_init_deg_c, np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ]))
    ax2.plot(
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ],
        (df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_udds) / ecr_comparison_udds * 100.0,
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax2.plot(
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [°C]"
        ])[solid_amb_temps],
        np.array((df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_udds) / ecr_comparison_udds * 100.0)[solid_amb_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_init_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax2.set_xlabel("Ambient Temp. [°C]")
ax2.set_ylabel("Percent Energy Consumption\nCompared to 22°C Baseline [%]")
ax2.legend(title="te_init [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig2.savefig(Path(__file__).parent / "UDDS ECR v. Amb. and Init. Temp Percentage Difference.svg")

if SHOW_PLOTS:
    plt.show()

fig3, ax3 = plt.subplots()
fig3.suptitle("HWFET ECR v. Amb. and Init. Temp.")
index = 0
for te_init_deg_c in te_init_short_deg_c:
    dotted_amb_temps, solid_amb_temps = get_reasonable_amb_ranges(te_init_deg_c, np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ]))
    ax3.plot(
        df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ],
        (df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_hwfet) / ecr_comparison_hwfet * 100.0,
        linestyle='dashed',
        color = lighter_colors[index],
    )
    ax3.plot(
        np.array(df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [°C]"
        ])[solid_amb_temps],
        np.array((df_res[(df_res["te_init [°C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ] - ecr_comparison_hwfet) / ecr_comparison_hwfet * 100.0)[solid_amb_temps],
        marker=".",
        linestyle='solid',
        label=f"{te_init_deg_c:.1f}",
        color = colors[index],
    )
    index += 1
ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax3.set_xlabel("Ambient Temp. [°C]")
ax3.set_ylabel("Percent Energy Consumption\nCompared to 22°C Baseline [%]")
ax3.legend(title="te_init [°C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig3.savefig(Path(__file__).parent / "HWFET ECR v. Amb. and Init. Temp Percentage Difference.svg")

if SHOW_PLOTS:
    plt.show()
