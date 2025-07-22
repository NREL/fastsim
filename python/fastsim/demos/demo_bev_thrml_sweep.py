"""BEV thermal demo with cold start and cold ambient conditions."""

# %%
import os

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

# load 2020 Chevrolet Bolt BEV from file
veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")
veh.set_save_interval(1)

# load cycle from file
base_cyc = fsim.Cycle.from_resource("udds.csv")

te_amb_arr: list[float] = [t + celsius_to_kelvin for t in [-6.7, -6.7, 38.0]]
te_batt_and_cab_init_arr: list[float] = [t + celsius_to_kelvin for t in [-6.7, 22.0, 45.0]]


def try_walk(sd: fsim.SimDrive, loc: str) -> None:
    """Wrap `walk` in try to enable context"""
    try:
        sd.walk()
    except Exception as err:
        raise Exception(f"{loc}:\n{err}")


# sweep ambient and initial conditions
for te_amb, te_init in zip(te_amb_arr, te_batt_and_cab_init_arr):
    cyc_dict = base_cyc.to_pydict()
    cyc_dict["temp_amb_air_kelvin"] = [te_amb] * base_cyc.len()
    cyc = fsim.Cycle.from_pydict(cyc_dict)

    # setup initial conditions
    veh_dict = veh.to_pydict()
    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
        "temperature_kelvin"
    ] = te_init
    veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"]["temp_prev_kelvin"] = (
        te_init
    )
    veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = te_init
    veh_dict["cabin"]["LumpedCabin"]["state"]["temp_prev_kelvin"] = te_init

    # simulate cycle
    veh = fsim.Vehicle.from_pydict(veh_dict)
    sd = fsim.SimDrive(veh, cyc, None)
    try_walk(sd, f"`sd_prep`, te_amb: {te_amb}, te_init: {te_init}")
