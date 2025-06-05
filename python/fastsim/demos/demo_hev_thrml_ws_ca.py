"""HEV thermal demo with warm start and cold ambient conditions."""

# %%
import os
import time

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

sns.set_theme()


# if environment var `SHOW_PLOTS=false` is set, no plots are shown
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"
# if environment var `SAVE_FIGS=true` is set, save plots
SAVE_FIGS = os.environ.get("SAVE_FIGS", "false").lower() == "true"

celsius_to_kelvin = 273.15
temp_amb = -6.7 + celsius_to_kelvin
temp_init_bat_and_cab = 22.0 + celsius_to_kelvin
temp_init_eng = 70.0 + celsius_to_kelvin
# `fastsim3` -- load vehicle and cycle, build simulation, and run
# %%

# load 2021 Hyundai Sonata HEV from file
veh_dict = fsim.Vehicle.from_file(
    fsim.package_root()
    / "../../cal_and_val/thermal/f3-vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml",
).to_pydict()
veh_dict["cabin"]["LumpedCabin"]["state"]["temperature_kelvin"] = temp_init_bat_and_cab
veh_dict["pt_type"]["HybridElectricVehicle"]["res"]["thrml"]["RESLumpedThermal"]["state"][
    "temperature_kelvin"
] = temp_init_bat_and_cab
veh_dict["pt_type"]["HybridElectricVehicle"]["fc"]["thrml"]["FuelConverterThermal"]["state"][
    "temperature_kelvin"
] = temp_init_eng
veh = fsim.Vehicle.from_pydict(veh_dict)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
veh.set_save_interval(1)

# load cycle from file
cyc_dict = fsim.Cycle.from_resource("udds.csv").to_pydict()
cyc_dict["temp_amb_air_kelvin"] = [temp_amb] * len(cyc_dict["time_seconds"])
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
fig_fc_pwr, ax_fc_pwr = plot_hev_fc_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_fc_energy, ax_fc_energy = plot_hev_fc_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_res_pwr, ax_res_pwr = plot_hev_res_pwr(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_res_energy, ax_res_energy = plot_hev_res_energy(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig_temps, ax_temps = plot_hev_temperatures(df, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)
fig, ax = plot_road_loads(df, veh, save_figs=SAVE_FIGS, show_plots=SHOW_PLOTS)

# %%

# %%
# example for how to use set_default_pwr_interp() method for veh.res
res = fsim.ReversibleEnergyStorage.from_pydict(
    sd.to_pydict()["veh"]["pt_type"]["HybridElectricVehicle"]["res"],
)
res.set_default_pwr_interp()

# %%
