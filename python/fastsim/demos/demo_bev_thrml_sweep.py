"""BEV thermal demo with cold start and cold ambient conditions."""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import fastsim as fsim

sns.set_theme()


# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

celsius_to_kelvin = 273.15
temp_amb_and_init = -6.7 + celsius_to_kelvin
# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%


def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


# array of ambient temperatures in kelvin
te_amb_arr_k: list[float] = [t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, 20)]
# array of init temperatures in kelvin
te_batt_and_cab_init_arr_k: list[float] = [
    t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, 20)
]

cyc_key = "cycle"
te_amb_key = "te_amb [*C]"
te_init_key = "te_init [*C]"
ecr_key = "ECR [kW-hr/100mi]"


def sweep():
    """Sweep ambient and initial conditions"""
    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)

    res_list = []
    for cyc_str in ["udds", "hwfet"]:
        for te_amb_k in te_amb_arr_k:
            for te_init_k in te_batt_and_cab_init_arr_k:
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


df_res = sweep()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"

# if environment var `SAVE_PLOTS=true` is set, plots are saved
SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "true").lower() == "true"

# plot ECR v. init for a sweep of amb
te_amb_step = int(len(te_amb_arr_k) / 10)
te_amb_short_deg_c = [te_amb_k - celsius_to_kelvin for te_amb_k in te_amb_arr_k][::te_amb_step]

fig, ax = plt.subplots()
fig.suptitle("UDDS ECR v. Init. and Amb. Temp.")
for te_amb_deg_c in te_amb_short_deg_c:
    ax.plot(
        df_res[(df_res["te_amb [*C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "te_init [*C]"
        ],
        df_res[(df_res["te_amb [*C]"] == te_amb_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ],
        marker=".",
        label=f"{te_amb_deg_c:.1f}",
    )
ax.set_xlabel("Cab. and Batt. Init. Temp. [*C]")
ax.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax.legend(title="te_amb [*C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig.savefig(Path(__file__).parent / "UDDS ECR v. Init. and Amb. Temp.svg")

if SHOW_PLOTS:
    plt.show()

fig1, ax1 = plt.subplots()
fig1.suptitle("HWFET ECR v. Init. and Amb. Temp.")
for te_amb_deg_c in te_amb_short_deg_c:
    ax1.plot(
        df_res[(df_res["te_amb [*C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_init [*C]"
        ],
        df_res[(df_res["te_amb [*C]"] == te_amb_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ],
        marker=".",
        label=f"{te_amb_deg_c:.1f}",
    )
ax1.set_xlabel("Cab. and Batt. Init. Temp. [*C]")
ax1.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax1.legend(title="te_amb [*C]")
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
for te_init_deg_c in te_init_short_deg_c:
    ax2.plot(
        df_res[(df_res["te_init [*C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "te_amb [*C]"
        ],
        df_res[(df_res["te_init [*C]"] == te_init_deg_c) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ],
        marker=".",
        label=f"{te_amb_deg_c:.1f}",
    )
ax2.set_xlabel("Ambient Temp. [*C]")
ax2.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax2.legend(title="te_amb [*C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig2.savefig(Path(__file__).parent / "UDDS ECR v. Amb. and Init. Temp.svg")

if SHOW_PLOTS:
    plt.show()

fig3, ax3 = plt.subplots()
fig3.suptitle("HWFET ECR v. Amb. and Init. Temp.")
for te_init_deg_c in te_init_short_deg_c:
    ax3.plot(
        df_res[(df_res["te_init [*C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "te_amb [*C]"
        ],
        df_res[(df_res["te_init [*C]"] == te_init_deg_c) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ],
        marker=".",
        label=f"{te_amb_deg_c:.1f}",
    )
ax3.set_xlabel("Ambient Temp. [*C]")
ax3.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax3.legend(title="te_amb [*C]")
plt.tight_layout()

if SAVE_PLOTS:
    fig3.savefig(Path(__file__).parent / "HWFET ECR v. Amb. and Init. Temp.svg")

if SHOW_PLOTS:
    plt.show()
