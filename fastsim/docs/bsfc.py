# %%
import numpy as np
import pandas as pd
import fastsim as fsim
from matplotlib import pyplot as plt

# %%
points = [
    [4000, 1.2, 700],
    [4000, 1.4, 650],
    [4000, 1.6, 600],
    [4000, 1.8, 550],
    [4000, 2.2, 500],
    [4000, 2.5, 450],
    [4000, 2.7, 400],
    [4000, 3.0, 345],
    [4000, 5.6, 300],
    [4000, 8.2, 300],

    [5000, 1.5, 750],
    [5000, 1.6, 700],
    [5000, 1.8, 650],
    [5000, 1.9, 600],
    [5000, 2.2, 550],
    [5000, 2.3, 500],
    [5000, 2.6, 450],
    [5000, 2.8, 400],
    [5000, 3.4, 350],
    [5000, 6.4, 300],
    [5000, 8.2, 300],
    [5000, 9.1, 350],

    [6000, 1.0, 750],
    [6000, 1.4, 700],
    [6000, 1.5, 650],
    [6000, 1.6, 600],
    [6000, 1.8, 550],
    [6000, 1.9, 500],
    [6000, 2.0, 450],
    [6000, 2.5, 400],
    [6000, 3.7, 350],
    [6000, 6.8, 300],
    [6000, 8.0, 300],
    [6000, 8.8, 350],

    [7000, 1.3, 750],
    [7000, 1.5, 700],
    [7000, 1.6, 650],
    [7000, 1.7, 600],
    [7000, 1.8, 550],
    [7000, 1.9, 500],
    [7000, 2.1, 450],
    [7000, 2.8, 400],
    [7000, 3.8, 350],
    [7000, 7.8, 350],

    [8000, 1.2, 750],
    [8000, 1.4, 700],
    [8000, 1.5, 650],
    [8000, 1.7, 600],
    [8000, 1.8, 550],
    [8000, 2.1, 500],
    [8000, 2.5, 450],
    [8000, 3.0, 400],
    [8000, 4.2, 350],
    [8000, 7.1, 350],
    [8000, 7.3, 400],
]
df = pd.DataFrame(points, columns=["RPM", "MEP (bar)", "BSFC (g/kWh)"])

LHV = 0.0117785551  # kWh/g, gasoline LHV from https://afdc.energy.gov/fuels/properties, rho = 750 kg/m3

def calc_power(row):
    # Calculate power using engine speed and mean effective pressure
    # https://en.wikipedia.org/wiki/Mean_effective_pressure#Derivation
    # MEP = P * n_c / (V_d * N)
    N = row["RPM"] / 60 # rev/s
    MEP = row["MEP (bar)"] * 1e5  # Pa
    return MEP*N*0.000125/2 / 1e3  # kW

def calc_efficiency(row):
    # Calculate efficiency using BSFC and gasoline LHV
    # https://en.wikipedia.org/wiki/Brake-specific_fuel_consumption#The_relationship_between_BSFC_numbers_and_efficiency
    BSFC = row["BSFC (g/kWh)"]
    return 1/(BSFC*LHV)

df["Power (kW)"] = df.apply(calc_power, axis=1)
df["Efficiency"] = df.apply(calc_efficiency, axis=1)
df["Pct Peak Power"] = df["Power (kW)"] / 6.5
df

# %%
df2 = df.sort_values("Pct Peak Power")
fig, ax = plt.subplots(sharey=True)
# Data points
ax.scatter(df2["Pct Peak Power"], df2["Efficiency"]*100, label="From motorcycle data")
# FASTSim engine map
ax.plot(fsim.vehicle.ref_veh.fc_perc_out_array, fsim.vehicle.ref_veh.fc_eff_array*100, label="FASTSim fc_eff_array", color="C1")
# Polynomial fit
poly = np.poly1d(np.polyfit(df2["Pct Peak Power"], df2["Efficiency"], deg=2))
poly_x = fsim.vehicle.ref_veh.fc_perc_out_array
poly_y = poly(poly_x)
ax.plot(poly_x, poly_y*100, label="Polynomial fit", color="C2")

ax.set_xlabel("Percent Output Power")
ax.set_ylabel("Efficiency (%)")
ax.legend()

# %%
fc_eff_map = poly(fsim.params.fc_pwr_out_perc)
fc_eff_map

# %%
