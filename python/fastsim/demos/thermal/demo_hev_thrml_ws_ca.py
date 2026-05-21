"""
---
execute:
  skip: true
---

# HEV Thermal Demo: Warm Start, Cold Ambient

This demo simulates a hybrid electric vehicle with thermal modeling under
warm start and cold ambient conditions, where the cabin and battery begin
warm, the engine begins hot, and the surrounding air is cold.
"""

# %%
import os

import seaborn as sns

import fastsim as fsim
from fastsim.demos.plot_utils import (
    plot_hev_fc_energy,
    plot_hev_fc_pwr,
    plot_hev_res_energy,
    plot_hev_res_pwr,
    plot_hev_temperatures,
    plot_road_loads,
)

# %%
sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

"""
## Setup and Simulation

Load a thermal HEV, set warm initial temperatures for the cabin and
battery, a hot initial temperature for the engine, and a cold ambient
temperature, then run the simulation.
"""

# %%
celsius_to_kelvin = 273.15
temp_amb = -6.7 + celsius_to_kelvin
temp_init_bat_and_cab = 22.0 + celsius_to_kelvin
temp_init_eng = 70.0 + celsius_to_kelvin

# load 2021 Hyundai Sonata HEV with thermal model
veh_dict = fsim.Vehicle.from_resource(
    "2021_Hyundai_Sonata_Hybrid_Blue_thrml.yaml",
).to_pydict()
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_init_bat_and_cab
veh_dict["pt_type"]["HEV"]["res"]["thrml"]["RESLumpedThermal"]["state"]["temperature_kelvin"] = (
    temp_init_bat_and_cab
)
veh_dict["pt_type"]["HEV"]["fc"]["thrml"]["FuelConverterThermal"]["state"]["temperature_kelvin"] = (
    temp_init_eng
)
veh = fsim.Vehicle.from_pydict(veh_dict)

veh.set_save_interval(1)

# %%
# load cycle and set ambient temperature
cyc_dict = fsim.Cycle.from_resource("udds.csv").to_pydict()
cyc_dict["temp_amb_air_kelvin"] = [temp_amb] * len(cyc_dict["time_seconds"])
cyc = fsim.Cycle.from_pydict(cyc_dict)

# %%
sd = fsim.SimDrive(veh, cyc)
sd.walk()

df = sd.to_dataframe()
sd_dict = sd.to_pydict(flatten=True)

"""
## Visualize Results

Fuel converter power and energy, battery power and energy, component
temperatures, and road loads under warm start, cold ambient conditions.
"""

# %%
fig_fc_pwr, ax_fc_pwr = plot_hev_fc_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_fc_energy, ax_fc_energy = plot_hev_fc_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_res_pwr, ax_res_pwr = plot_hev_res_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_res_energy, ax_res_energy = plot_hev_res_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_temps, ax_temps = plot_hev_temperatures(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig, ax = plot_road_loads(df, veh, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
