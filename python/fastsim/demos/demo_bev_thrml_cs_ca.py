# %%
import os
import time

import seaborn as sns

import fastsim as fsim

from plot_utils import (  # type: ignore[import-not-found]
    plot_bev_hvac_pwr,
    plot_bev_res_energy,
    plot_bev_res_pwr,
    plot_road_loads,
    plot_bev_temperatures,
)

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
veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV.yaml")

veh_dict = veh.to_pydict()
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_amb_and_init
veh_dict["pt_type"]["BatteryElectricVehicle"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    "temperature_kelvin"
] = temp_amb_and_init
veh = fsim.Vehicle.from_pydict(veh_dict)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# load cycle from file
cyc_dict = fsim.Cycle.from_resource("udds.csv").to_pydict()
cyc_dict["temp_amb_air_kelvin"] = [temp_amb_and_init] * len(cyc_dict["time_seconds"])
cyc = fsim.Cycle.from_pydict(cyc_dict)

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)

# simulation start time
t0 = time.perf_counter()
# run simulation
sd.walk()
# simulation end time
t1 = time.perf_counter()
t_fsim3_si1 = t1 - t0
print(f"fastsim-3 `sd.walk()` elapsed time with `save_interval` of 1:\n{t_fsim3_si1:.2e} s")

# %%
df = sd.to_dataframe(allow_partial=True)
sd_dict = sd.to_pydict(flatten=True)

# # Visualize results
fig_res_pwr, ax_res_pwr = plot_bev_res_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_res_energy, ax_res_energy = plot_bev_res_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_temps, ax_temps = plot_bev_temperatures(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_hvac, ax_hvac = plot_bev_hvac_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig, ax = plot_road_loads(df, veh, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

assert sd_dict["veh.cabin.LumpedCabin.state.temperature_kelvin"] > 273.15 + 15.0

# %%

# %%
# example for how to use set_default_pwr_interp() method for veh.res
res = fsim.ReversibleEnergyStorage.from_pydict(
    sd.to_pydict()["veh"]["pt_type"]["BatteryElectricVehicle"]["res"],
)
res.set_default_pwr_interp()

# %%
