"""
Vehicle Import Demonstration
"""
import fastsimrust as fsr

make = "toyota"
model = "camry"
year = "2016"
options = fsr.get_fuel_economy_gov_options_for_year_make_model(year, make, model)
for opt in options:
    print(f"{opt.id}: {opt.transmission}")
data = fsr.get_fuel_economy_gov_data_by_option_id(options[0].id)
print(f"{data.year} {data.make} {data.model}: {data.comb_mpg_fuel1} mpg ({data.city_mpg_fuel1} CITY / {data.highway_mpg_fuel1} HWY)")


