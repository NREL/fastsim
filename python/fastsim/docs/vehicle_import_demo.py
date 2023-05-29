"""
Vehicle Import Demonstration
"""
import os, pathlib

import fastsimrust as fsr

make = "Toyota"
model = "Corolla"
year = "2022"
options = fsr.get_fuel_economy_gov_options_for_year_make_model(year, make, model)
for opt in options:
    print(f"{opt.id}: {opt.transmission}")
opt = options[1]
data = fsr.get_fuel_economy_gov_data_by_option_id(opt.id)
print(f"{data.year} {data.make} {data.model}: {data.comb_mpg_fuel1} mpg ({data.city_mpg_fuel1} CITY / {data.highway_mpg_fuel1} HWY)")

other_inputs = fsr.OtherVehicleInputs(
    vehicle_width_in=68.0,
    vehicle_height_in=58.0,
    fuel_tank_gal=12.0,
    ess_max_kwh=0.0,
    fc_max_kw=None, # None -> calculate from EPA data
    mc_max_kw=0.0,
    ess_max_kw=0.0)
rv = fsr.vehicle_import_from_id(opt.id, other_inputs)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
output_dir = pathlib.Path(THIS_DIR) / "test_output"
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
fsr.export_vehicle_to_file(rv, str(output_dir / "demo-vehicle.yaml"))
print("DONE!")
