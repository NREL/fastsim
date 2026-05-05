"""Dummy paratransit minivan thermal demo, showcasing how to set initial temperatures for thermal components and visualize results."""

# %%
# Imports
import time

import seaborn as sns

import fastsim as fsim
from fastsim.demos.plot_utils import (
    plot_conv_fc_energy,
    plot_conv_fc_pwr,
    plot_conv_temperatures,
    plot_road_loads,
)

sns.set_theme()

CELCIUS_TO_KELVIN = 273.15


# %%
SHOW_PLOTS = True
SAVE_FIGS = False


def set_initial_temperature(veh: fsim.Vehicle, temp_celcius: float) -> fsim.Vehicle:
    """Set temperature for all thermal components in the vehicle."""
    temp_kelvin = temp_celcius + CELCIUS_TO_KELVIN
    veh_dict = veh.to_pydict()
    veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_kelvin
    pt_type = "Conv" if "Conv" in veh_dict["pt_type"] else None
    assert (
        pt_type == "Conv"
    ), "this utility function is currently only set up for conventional vehicles"
    # veh_dict["pt_type"]["HEV"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    #     "temperature_kelvin"
    # ] = temp_kelvin
    veh_dict["pt_type"]["Conv"]["fc"]["thrml"]["FuelConverterThermal"]["state"][
        "temperature_kelvin"
    ] = temp_kelvin
    return fsim.Vehicle.from_pydict(veh_dict)


def set_hvac_target_temperature(veh: fsim.Vehicle, temp_celcius: float) -> fsim.Vehicle:
    temp_kelvin = temp_celcius + CELCIUS_TO_KELVIN
    veh_dict = veh.to_pydict()
    veh_dict["hvac"]["LumpedCabin"]["te_set_kelvin"] = temp_kelvin
    return fsim.Vehicle.from_pydict(veh_dict)


def set_ambient_temperature(cyc: fsim.Cycle, temp_celcius: float) -> fsim.Cycle:
    temp_kelvin = temp_celcius + CELCIUS_TO_KELVIN
    cyc_dict = cyc.to_pydict()
    cyc_dict["temp_amb_air_kelvin"] = [temp_kelvin] * len(cyc_dict["time_seconds"])
    return fsim.Cycle.from_pydict(cyc_dict)


# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%
# Temperature at which to initialize thermal components
initial_temperature_c = 19
hvac_target_temperature_c = 22
ambient_temperature_c = 29

# Load vehicle from resource file & set initial temperature
veh = fsim.Vehicle.from_resource(
    "dummy_minivan_thrml.yaml",
)
veh = set_initial_temperature(veh, initial_temperature_c)
veh = set_hvac_target_temperature(veh, hvac_target_temperature_c)

# Save interval: record every Nth time step (default is 1, meaning every time step is saved)
# Defaults to 1 if left unspecified
veh.set_save_interval(1)

# Load vehicle from resource file & set initial temperature
# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")
cyc = set_ambient_temperature(cyc, ambient_temperature_c)

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(
    f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s"
)

# %%
df = sd.to_dataframe()
sd_dict = sd.to_pydict(flatten=True)
# # Visualize results
fig_fc_pwr, ax_fc_pwr = plot_conv_fc_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_fc_energy, ax_fc_energy = plot_conv_fc_energy(
    df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS
)
fig_temps, ax_temps = plot_conv_temperatures(
    df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS
)
fig, ax = plot_road_loads(df, veh, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
