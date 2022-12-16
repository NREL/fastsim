# %%
import fastsim as fsim

# %%
gCO2__USgal = 8887  # 8,887 grams of CO2/gallon of gasoline  https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references
mi__km = 0.621371
L__USgal = 3.785411784

# %% MEASURED
gCO2__km = 36.9  # using EPA result for SYM Symba 100
USgal__km = gCO2__km * 1/gCO2__USgal
measured_L__100km = USgal__km * L__USgal * 100
print(f"MEASURED:  {measured_L__100km:.3f} L/100km ({gCO2__km:.3f} gCO2/km)")

# %% SIMULATED
cyc = fsim.cycle.Cycle.from_file("ftpmc1b.csv")
veh = fsim.vehicle.Vehicle.from_file("2020_Hero_Splendor+_100cc_2W.csv")
sd = fsim.simdrive.SimDrive(cyc, veh)
sd.sim_drive()
simulated_L__100km = 1/sd.mpgge * L__USgal * mi__km * 100
simulated_gCO2__km = simulated_L__100km / L__USgal / 100 * gCO2__USgal
print(f"SIMULATED: {simulated_L__100km:.3f} L/100km ({simulated_gCO2__km:.3f} gCO2/km)")
error = (simulated_L__100km - measured_L__100km)/measured_L__100km * 100
print(f"ERROR: {error:.3f}%")
# %%
