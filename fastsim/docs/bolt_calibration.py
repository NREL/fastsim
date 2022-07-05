# %%
import fastsim

# %%
target = 28  # kWh/100mi

# %%
veh = fastsim.vehicle.Vehicle(17)
print(veh.Scenario_name)

# %%
label_results = fastsim.simdrivelabel.get_label_fe(veh)
kWh_per_100mi = label_results["adjCombKwhPerMile"] * 100
error = (kWh_per_100mi - target)/target * 100
print(f"Target: {target} kWh/100mi")
print(f"Result: {kWh_per_100mi:.3f} kWh/100mi")
print(f"{error:+.2f}%")
# %%
