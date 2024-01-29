"""
Vehicle Import Demonstration
This module demonstrates the vehicle import API
"""
# %%
# Preamble: Basic imports
import os, pathlib

import fastsim.fastsimrust as fsr
import fastsim.utils as utils

#for testing demo files, false when running automatic tests
SHOW_PLOTS = utils.show_plots()

# %%
# Setup some directories
THIS_DIR = pathlib.Path(__file__).parent.absolute()
OUTPUT_DIR = pathlib.Path(THIS_DIR) / "test_output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# List available options for the given year / make / model
make = "Toyota"
model = "Corolla"
year = "2022"

# NOTE: two additional optional arguments can be passed to get_options_for_year_make_model.
# They are the cache_url which is the URL for a file download service to retrieve cache data by year
# and also a data directory where that cache data will be stored.
# If not provided, they default to NREL's cache URL and the OS specific config data directory for this application.
# Also note that, due to interop requirements, these must be passed in as python strings. For example, a
# Python pathlib.Path object will be rejected.

options = fsr.get_options_for_year_make_model(year, make, model)
if SHOW_PLOTS:
    for opt in options:
        print(f"{opt.id}: {opt.transmission}")

# %%
# Get the data for the given option
data = options[1]
if SHOW_PLOTS:
    print(
        f"{data.year} {data.make} {data.model}: {data.comb_mpg_fuel1} mpg ({data.city_mpg_fuel1} CITY / {data.highway_mpg_fuel1} HWY)"
    )

# %%
# Import the vehicle
other_inputs = fsr.OtherVehicleInputs(
    vehicle_width_in=68.0,
    vehicle_height_in=58.0,
    fuel_tank_gal=12.0,
    ess_max_kwh=0.0,
    mc_max_kw=0.0,
    ess_max_kw=0.0,
    fc_max_kw=None,
)  # None -> calculate from EPA data

# NOTE: two additional optional arguments can be passed to vehicle_import_by_id_and_year.
# They are the cache_url which is the URL for a file download service to retrieve cache data by year
# and also a data directory where that cache data will be stored.
# If not provided, they default to NREL's cache URL and the OS specific config data directory for this application.
# Also note that, due to interop requirements, these must be passed in as python strings. For example, a
# Python pathlib.Path object will be rejected.

rv = fsr.vehicle_import_by_id_and_year(data.id, int(year), other_inputs)

fsr.export_vehicle_to_file(rv, str(OUTPUT_DIR / "demo-vehicle.yaml"))

# %%
# Alternative API for importing all vehicles at once
# This API will import all matching configurations for
# the given year, make, and model.

# NOTE: two additional optional arguments can be passed to import_all_vehicles.
# They are the cache_url which is the URL for a file download service to retrieve cache data by year
# and also a data directory where that cache data will be stored.
# If not provided, they default to NREL's cache URL and the OS specific config data directory for this application.
# Also note that, due to interop requirements, these must be passed in as python strings. For example, a
# Python pathlib.Path object will be rejected.

vehs = fsr.import_all_vehicles(int(year), make, model, other_inputs)
if SHOW_PLOTS:
    for v in vehs:
        print(f"Imported {v.scenario_name}")