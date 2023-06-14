"""
Vehicle Import Demonstration
"""
# %%
# Preamble: Basic imports
import os, pathlib

import fastsim.fastsimrust as fsr
from fastsim.demos.utils import maybe_str_to_bool, DEMO_TEST_ENV_VAR

RAN_SUCCESSFULLY = False
IS_INTERACTIVE = maybe_str_to_bool(os.getenv(DEMO_TEST_ENV_VAR))

# %%
# Setup some directories
THIS_DIR = pathlib.Path(__file__).parent.absolute()
RESOURCE_DIR = (THIS_DIR / ".." / "resources" / "epa_vehdb").resolve()
assert RESOURCE_DIR.exists()
OUTPUT_DIR = pathlib.Path(THIS_DIR) / "test_output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# List available options for the given year / make / model
make = "Toyota"
model = "Corolla"
year = "2022"

options = fsr.get_fuel_economy_gov_options_for_year_make_model(year, make, model)
if IS_INTERACTIVE:
    for opt in options:
        print(f"{opt.id}: {opt.transmission}")

# %%
# Get the data for the given option
opt = options[1]
data = fsr.get_fuel_economy_gov_data_by_option_id(opt.id)
if IS_INTERACTIVE:
    print(f"{data.year} {data.make} {data.model}: {data.comb_mpg_fuel1} mpg ({data.city_mpg_fuel1} CITY / {data.highway_mpg_fuel1} HWY)")

# %%
# Import the vehicle
other_inputs = fsr.OtherVehicleInputs(
    vehicle_width_in=68.0,
    vehicle_height_in=58.0,
    fuel_tank_gal=12.0,
    ess_max_kwh=0.0,
    fc_max_kw=None, # None -> calculate from EPA data
    mc_max_kw=0.0,
    ess_max_kw=0.0)
rv = fsr.vehicle_import_from_id(opt.id, other_inputs, str(RESOURCE_DIR))

fsr.export_vehicle_to_file(rv, str(OUTPUT_DIR / "demo-vehicle.yaml"))

# %%
# Used for automated testing
RAN_SUCCESSFULLY = True
