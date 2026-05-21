"""
---
execute:
  skip: true
---

# BEV Thermal Demo: Cold Start, Cold Ambient

This demo simulates a battery electric vehicle with thermal modeling under
cold start and cold ambient conditions, where the cabin and battery begin
at the same temperature as the surrounding air.
"""

# %%
import os
import sys
from pathlib import Path

import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastsim as fsim
from plot_utils import (
    plot_bev_hvac_pwr,
    plot_bev_res_energy,
    plot_bev_res_pwr,
    plot_bev_temperatures,
    plot_road_loads,
)

# %%
sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

"""
## Setup and Simulation

Load a thermal BEV, set cold initial temperatures for the cabin and
battery to match the cold ambient, and run the simulation.
"""

# %%
celsius_to_kelvin = 273.15
temp_amb_and_init = -6.7 + celsius_to_kelvin

# load 2020 Chevrolet Bolt BEV with thermal model
veh = fsim.Vehicle.from_resource("2020 Chevrolet Bolt EV thrml.yaml")

veh_dict = veh.to_pydict()
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_amb_and_init
veh_dict["pt_type"]["BEV"]["res"]["thrml"]["RESLumpedThermal"]["state"]["temperature_kelvin"] = (
    temp_amb_and_init
)
veh = fsim.Vehicle.from_pydict(veh_dict)

veh.set_save_interval(1)

# %%
# load cycle and set ambient temperature
cyc_dict = fsim.Cycle.from_resource("udds.csv").to_pydict()
cyc_dict["temp_amb_air_kelvin"] = [temp_amb_and_init] * len(cyc_dict["time_seconds"])
cyc = fsim.Cycle.from_pydict(cyc_dict)

# %%
sd = fsim.SimDrive(veh, cyc)
sd.walk()

df = sd.to_dataframe()
sd_dict = sd.to_pydict(flatten=True)

"""
## Visualize Results

Battery power, energy, component temperatures, HVAC power demand, and road
loads under cold start, cold ambient conditions.
"""

# %%
fig_res_pwr, ax_res_pwr = plot_bev_res_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_res_energy, ax_res_energy = plot_bev_res_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_temps, ax_temps = plot_bev_temperatures(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig_hvac, ax_hvac = plot_bev_hvac_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%
fig, ax = plot_road_loads(df, veh, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
