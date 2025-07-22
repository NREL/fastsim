"""BEV thermal demo with cold start and cold ambient conditions."""

# %%
import os

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


te_amb_arr: list[float] = [t + celsius_to_kelvin for t in np.linspace(-7.0, 40.0, 20)]
te_batt_and_cab_init_arr: list[float] = [t + celsius_to_kelvin for t in np.linspace(-7.0, 45.0, 20)]


def sweep():
    """Sweep ambient and initial conditions"""
    # load 2020 Chevrolet Bolt BEV from file
    veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
    veh.set_save_interval(1)

    df_res = pd.DataFrame(
        columns=["cycle", "te_amb [*C]", "te_init [*C]", "ECR [kW-hr/100mi]"],
    )

    for cyc_str, te_amb, te_init in zip(["udds", "hwfet"], te_amb_arr, te_batt_and_cab_init_arr):
        cyc = fsim.Cycle.from_resource(cyc_str + ".csv")
        cyc_dict = cyc.to_pydict()
        cyc_dict["temp_amb_air_kelvin"] = [te_amb] * cyc.len()
        cyc = fsim.Cycle.from_pydict(cyc_dict)

        # setup initial conditions
        veh_dict = veh.to_pydict()
        veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
            "temperature_kelvin"
        ] = te_init
        veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
            "temp_prev_kelvin"
        ] = te_init
        veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = te_init
        veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = te_init

        # simulate cycle
        veh = fsim.Vehicle.from_pydict(veh_dict)
        sd = fsim.SimDrive(veh, cyc, None)
        try_walk(sd, f"`sd_prep`, te_amb: {te_amb}, te_init: {te_init}")
        veh_dict_solved = sd.to_pydict()["veh"]

        new_row = pd.Series(
            [
                cyc_str,
                te_amb - celsius_to_kelvin,
                te_init - celsius_to_kelvin,
                veh_dict_solved["pt_type"]["BEV"]["res"]["state"]["energy_out_chemical_joules"]
                / 1_000
                / 3_600
                / (veh_dict_solved["state"]["dist_meters"] / 1e3 / 1.61),
            ],
        )
        df_res = pd.concat(
            [
                df_res,
                new_row,
            ],
        )

    return df_res


df_res = sweep()

# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"

te_amb_short = te_amb_arr[:: int(len(te_amb_arr) / 5)]

fig, ax = plt.subplots()
fig.suptitle("UDDS ECR v. Init. and Amb. Temp.")
for te_amb in te_amb_short:
    ax.plot(
        df_res[(df_res["te_amb [*C]"] == te_amb) & (df_res["cycle"] == "udds")]["te_init [*C]"],
        df_res[(df_res["te_amb [*C]"] == te_amb) & (df_res["cycle"] == "udds")][
            "ECR [kW-hr/100mi]"
        ],
        label=te_amb,
    )
ax.set_xlabel("Cab. and Batt. Init. Temp. [*C]")
ax.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax.legend(title="te_amb [*C]")

fig0, ax0 = plt.subplots()
fig0.suptitle("HWFET ECR v. Init. and Amb. Temp.")
for te_amb in te_amb_short:
    ax0.plot(
        df_res[(df_res["te_amb [*C]"] == te_amb) & (df_res["cycle"] == "hwfet")]["te_init [*C]"],
        df_res[(df_res["te_amb [*C]"] == te_amb) & (df_res["cycle"] == "hwfet")][
            "ECR [kW-hr/100mi]"
        ],
        label=te_amb,
    )
ax0.set_xlabel("Cab. and Batt. Init. Temp. [*C]")
ax0.set_ylabel("Energy Consumption Rate [kW-hr/100mi]")
ax0.legend(title="te_amb [*C]")

if SHOW_PLOTS:
    plt.show()
