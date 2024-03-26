"""
Vehicle Import Demonstration
This module demonstrates the vehicle import API
"""
# %%
from fastsim import fastsimrust

REQUIRED_FEATURE = "vehicle-import"
if __name__ == "__main__" and REQUIRED_FEATURE not in fastsimrust.enabled_features():
    raise NotImplementedError(
        f'Feature "{REQUIRED_FEATURE}" is required to run this demo'
    )

# %%
# Preamble: Basic imports
import os, pathlib

import fastsim.fastsimrust as fsr
import fastsim.utils as utils

import tempfile

# for testing demo files, false when running automatic tests
SHOW_PLOTS = utils.show_plots()
SAVE_OUTPUT = SHOW_PLOTS

# %%
if SHOW_PLOTS:
    # Setup some directories
    THIS_DIR = pathlib.Path(__file__).parent.absolute()
    # If the current directory is the fastsim installation directory, then the
    # output directory should be temporary directory
    if "site-packages/fastsim" in str(pathlib.Path(THIS_DIR)):
        OUTPUT_DIR_FULL = tempfile.TemporaryDirectory()
        OUTPUT_DIR = OUTPUT_DIR_FULL.name
        is_temp_dir = True
    # If the current directory is not the fastsim installation directory, find or
    # create "demo_output" directory to save outputs
    else:
        OUTPUT_DIR = pathlib.Path(THIS_DIR) / "demo_output"
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        is_temp_dir = False

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

if SAVE_OUTPUT:
    rv.to_file(OUTPUT_DIR / "demo-vehicle.yaml")

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