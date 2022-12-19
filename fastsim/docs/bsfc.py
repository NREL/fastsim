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
    [4000, 8.2, 250],
    [4000, 8.7, 300],

    # [8000, 7.7, 0],
    # [7000, 8.9, 0],
    # [9000, 6.3, 0],
    # [6000, 9.3, 0],
    # [6500, 9.1, 0],
    # [5500, 9.2, 0]
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
# df["Pct Peak Power"] = df["Power (kW)"] / max(df["Power (kW)"])
df["Pct Peak Power"] = df["Power (kW)"] / 6.5
df

# %%
df2 = df.sort_values("Pct Peak Power")
fig, ax = plt.subplots(sharey=True)
ax.plot(df2["Pct Peak Power"], df2["Efficiency"]*100, label="From motorcycle data")
ax.plot(fsim.vehicle.ref_veh.fc_perc_out_array, fsim.vehicle.ref_veh.fc_eff_array*100, label="FASTSim fc_eff_array")
ax.set_xlabel("Percent Output Power")
ax.set_ylabel("Efficiency (%)")
ax.legend()

# %%
