# %% [markdown]
# # WLTP Cycle 3B calibration notebook

# %% [markdown]
# Measured values

# %%
# Measured FE (L/100km)
meas_fe_combined = 3.8
meas_fe_low = 2.9
meas_fe_med = 3
meas_fe_high = 3.5
meas_fe_extrahigh = 5

# Measured CO2 emissions (g/km)
meas_CO2_combined = 87
meas_CO2_low = 66
meas_CO2_med = 68
meas_CO2_high = 79
meas_CO2_extrahigh = 113

# %% [markdown]
# Constants

# %%
L_PER_GAL = 3.78541
WILLANS_FACTOR = 724  # gCO2/MJ
E10_HEAT_VALUE = 8.64  # kWh/L

# %% [markdown]
# Function definitions

# %%
def mpgge_to_litersPer100km(mpgge):
    return (1 / mpgge) * (1 / fastsim.params.M_PER_MI) * 1000 * 100 * L_PER_GAL

# %% [markdown]
# Code

# %%
import fastsim
from matplotlib import pyplot as plt

# %%
# Load vehicle
veh = fastsim.vehicle.Vehicle.from_file("2022_TOYOTA_Yaris_Hybrid_Mid.csv")

# %%
# Load WLTC cycle phases
cyc_low = fastsim.cycle.Cycle.from_file("wltc_class3_low3.csv")
cyc_med = fastsim.cycle.Cycle.from_file("wltc_class3_med3b.csv")
cyc_high = fastsim.cycle.Cycle.from_file("wltc_class3_high3b.csv")
cyc_extrahigh = fastsim.cycle.Cycle.from_file("wltc_class3_extra_high3.csv")

# %%
# Generate combined cycle
cyc_combined_dict = fastsim.cycle.concat(
    [
        cyc_low.get_cyc_dict(),
        cyc_med.get_cyc_dict(),
        cyc_high.get_cyc_dict(),
        cyc_extrahigh.get_cyc_dict(),
    ]
)
cyc = fastsim.cycle.Cycle.from_dict(cyc_combined_dict)

# %%
# Simulate combined cycle
sim = fastsim.simdrive.SimDrive(cyc, veh)
sim.sim_drive()

# %%
# Calculate FE
dist_miles_combined = sim.dist_mi.sum()
energy_combined = sim.fs_kwh_out_ach.sum()
fe_mpgge_combined = dist_miles_combined / (energy_combined/sim.props.kwh_per_gge)
fe_litersPer100km_combined = mpgge_to_litersPer100km(fe_mpgge_combined)
print(f"{fe_litersPer100km_combined:.2f} L/100km")
print(f"{abs(fe_litersPer100km_combined - meas_fe_combined)/meas_fe_combined * 100:.2f}%")

# %%
i0 = len(cyc_low.time_s)
i1 = i0 + len(cyc_med.time_s)-1
i2 = i1 + len(cyc_high.time_s)-1

low = slice(None, i0)
medium = slice(i0-1, i1)
high = slice(i1-1, i2)
extrahigh = slice(i2-1, None)

# %% [markdown]
# WLTC 3 Low

# %%
# Calculate FE
dist_miles_low = sim.dist_mi[low].sum()
energy_low = sim.fs_kwh_out_ach[low].sum()
fe_mpgge_low = dist_miles_low / (energy_low/sim.props.kwh_per_gge)
fe_litersPer100km_low = mpgge_to_litersPer100km(fe_mpgge_low)
print(f"{fe_litersPer100km_low:.2f} L/100km")
print(f"{abs(fe_litersPer100km_low - meas_fe_low)/meas_fe_low * 100:.2f}%")

# %%
# Calculate ΔSOC (negative is decrease)
dSOC_low = sim.soc[i0-1] - sim.soc[0]
print(f"{dSOC_low:+.5f}")

# %%
# Calculate ESS energy expended (positive is decrease)
dE_low = -dSOC_low * veh.ess_max_kwh * 1000
print(f"{dE_low:.2f} Wh")

# %%
# Calculate distance in km
dist_km_low = dist_miles_low * fastsim.params.M_PER_MI / 1000
print(f"{dist_km_low:.2f} km")

# %%
# Calculate CO2 emissions (gCO2/km)
dM_low = 0.0036 * dE_low * 1/veh.alt_eff * WILLANS_FACTOR * 1/dist_km_low
print(f"{dM_low:.2f} gCO2/km")

# %%
# Calculate equivalent fuel consumption (L)
fe_adjustment_low = dE_low/1000 * 1/veh.alt_eff * 1/E10_HEAT_VALUE * 100/dist_km_low
print(f"{fe_adjustment_low:+.5f} L/100km")

# Calculate adjusted FE
fe_litersPer100km_low_adj = fe_litersPer100km_low + fe_adjustment_low
print(f"{fe_litersPer100km_low_adj:.2f} L/100km")
print(f"{abs(fe_litersPer100km_low_adj - meas_fe_low)/meas_fe_low * 100:.2f}%")

# %% [markdown]
# WLTC 3B Medium

# %%
# Calculate FE
dist_miles_med = sim.dist_mi[medium].sum()
energy_med = sim.fs_kwh_out_ach[medium].sum()
fe_mpgge_med = dist_miles_med / (energy_med/sim.props.kwh_per_gge)
fe_litersPer100km_med = mpgge_to_litersPer100km(fe_mpgge_med)
print(f"{fe_litersPer100km_med:.2f} L/100km")
print(f"{abs(fe_litersPer100km_med - meas_fe_med)/meas_fe_med * 100:.2f}%")

# %%
# Calculate ΔSOC (negative is decrease)
dSOC_med = sim.soc[i1-1] - sim.soc[i0-1]
print(f"{dSOC_med:+.5f}")

# %%
# Calculate ESS energy expended (positive is decrease)
dE_med = -dSOC_med * veh.ess_max_kwh * 1000
print(f"{dE_med:.2f} Wh")

# %%
# Calculate distance in km
dist_km_med = dist_miles_med * fastsim.params.M_PER_MI / 1000
print(f"{dist_km_med:.2f} km")

# %%
# Calculate CO2 emissions (gCO2/km)
dM_med = 0.0036 * dE_med * 1/veh.alt_eff * WILLANS_FACTOR * 1/dist_km_med
print(f"{dM_med:.2f} gCO2/km")

# %%
# Calculate equivalent fuel consumption (L)
fe_adjustment_med = dE_med/1000 * 1/veh.alt_eff * 1/E10_HEAT_VALUE * 100/dist_km_med
print(f"{fe_adjustment_med:+.5f} L/100km")

# Calculate adjusted FE
fe_litersPer100km_med_adj = fe_litersPer100km_med + fe_adjustment_med
print(f"{fe_litersPer100km_med_adj:.2f} L/100km")
print(f"{abs(fe_litersPer100km_med_adj - meas_fe_med)/meas_fe_med * 100:.2f}%")

# %% [markdown]
# WLTC 3B High

# %%
# Calculate FE
dist_miles_high = sim.dist_mi[high].sum()
energy_high = sim.fs_kwh_out_ach[high].sum()
fe_mpgge_high = dist_miles_high / (energy_high/sim.props.kwh_per_gge)
fe_litersPer100km_high = mpgge_to_litersPer100km(fe_mpgge_high)
print(f"{fe_litersPer100km_high:.2f} L/100km")
print(f"{abs(fe_litersPer100km_high - meas_fe_high)/meas_fe_high * 100:.2f}%")

# %%
# Calculate ΔSOC (negative is decrease)
dSOC_high = sim.soc[i2-1] - sim.soc[i1-1]
print(f"{dSOC_high:+.5f}")

# %%
# Calculate ESS energy expended (positive is decrease)
dE_high = -dSOC_high * veh.ess_max_kwh * 1000
print(f"{dE_high:.2f} Wh")

# %%
# Calculate distance in km
dist_km_high = dist_miles_high * fastsim.params.M_PER_MI / 1000
print(f"{dist_km_high:.2f} km")

# %%
# Calculate CO2 emissions (gCO2/km)
dM_high = 0.0036 * dE_high * 1/veh.alt_eff * WILLANS_FACTOR * 1/dist_km_high
print(f"{dM_high:.2f} gCO2/km")

# %%
# Calculate equivalent fuel consumption (L)
fe_adjustment_high = dE_high/1000 * 1/veh.alt_eff * 1/E10_HEAT_VALUE * 100/dist_km_high
print(f"{fe_adjustment_high:+.5f} L/100km")

# Calculate adjusted FE
fe_litersPer100km_high_adj = fe_litersPer100km_high + fe_adjustment_high
print(f"{fe_litersPer100km_high_adj:.2f} L/100km")
print(f"{abs(fe_litersPer100km_high_adj - meas_fe_high)/meas_fe_high * 100:.2f}%")

# %% [markdown]
# WLTC 3 Extra-High

# %%
# Calculate FE
dist_miles_extrahigh = sim.dist_mi[extrahigh].sum()
energy_extrahigh = sim.fs_kwh_out_ach[extrahigh].sum()
fe_mpgge_extrahigh = dist_miles_extrahigh / (energy_extrahigh/sim.props.kwh_per_gge)
fe_litersPer100km_extrahigh = mpgge_to_litersPer100km(fe_mpgge_extrahigh)
print(f"{fe_litersPer100km_extrahigh:.2f} L/100km")
print(f"{abs(fe_litersPer100km_extrahigh - meas_fe_extrahigh)/meas_fe_extrahigh * 100:.2f}%")

# %%
# Calculate ΔSOC (negative is decrease)
dSOC_extrahigh = sim.soc[-1] - sim.soc[i2-1]
print(f"{dSOC_extrahigh:+.5f}")

# %%
# Calculate ESS energy expended (positive is decrease)
dE_extrahigh = -dSOC_extrahigh * veh.ess_max_kwh * 1000
print(f"{dE_extrahigh:.2f} Wh")

# %%
# Calculate distance in km
dist_km_extrahigh = dist_miles_extrahigh * fastsim.params.M_PER_MI / 1000
print(f"{dist_km_extrahigh:.2f} km")

# %%
# Calculate CO2 emissions (gCO2/km)
dM_extrahigh = 0.0036 * dE_extrahigh * 1/veh.alt_eff * WILLANS_FACTOR * 1/dist_km_extrahigh
print(f"{dM_extrahigh:.2f} gCO2/km")

# %%
# Calculate equivalent fuel consumption (L/100km)
fe_adjustment_extrahigh = dE_extrahigh/1000 * 1/veh.alt_eff * 1/E10_HEAT_VALUE * 100/dist_km_extrahigh
print(f"{fe_adjustment_extrahigh:+.5f} L/100km")

# Calculate adjusted FE
fe_litersPer100km_extrahigh_adj = fe_litersPer100km_extrahigh + fe_adjustment_extrahigh
print(f"{fe_litersPer100km_extrahigh_adj:.2f} L/100km")
print(f"{abs(fe_litersPer100km_extrahigh_adj - meas_fe_extrahigh)/meas_fe_extrahigh * 100:.2f}%")

# %%
print("LOW")
print(f"  Target: {meas_fe_low} L/100km")
print(f"  Before: {fe_litersPer100km_low:.2f} L/100km ({(fe_litersPer100km_low - meas_fe_low)/meas_fe_low * 100:+.2f}%)")
print(f"  After:  {fe_litersPer100km_low_adj:.2f} L/100km ({(fe_litersPer100km_low_adj - meas_fe_low)/meas_fe_low * 100:+.2f}%)")

print("MEDIUM")
print(f"  Target: {meas_fe_med} L/100km")
print(f"  Before: {fe_litersPer100km_med:.2f} L/100km ({(fe_litersPer100km_med - meas_fe_med)/meas_fe_med * 100:+.2f}%)")
print(f"  After:  {fe_litersPer100km_med_adj:.2f} L/100km ({(fe_litersPer100km_med_adj - meas_fe_med)/meas_fe_med * 100:+.2f}%)")

print("HIGH")
print(f"  Target: {meas_fe_high} L/100km")
print(f"  Before: {fe_litersPer100km_high:.2f} L/100km ({(fe_litersPer100km_high - meas_fe_high)/meas_fe_high * 100:+.2f}%)")
print(f"  After:  {fe_litersPer100km_high_adj:.2f} L/100km ({(fe_litersPer100km_high_adj - meas_fe_high)/meas_fe_high * 100:+.2f}%)")

print("EXTRA-HIGH")
print(f"  Target: {meas_fe_extrahigh} L/100km")
print(f"  Before: {fe_litersPer100km_extrahigh:.2f} L/100km ({(fe_litersPer100km_extrahigh - meas_fe_extrahigh)/meas_fe_extrahigh * 100:+.2f}%)")
print(f"  After:  {fe_litersPer100km_extrahigh_adj:.2f} L/100km ({(fe_litersPer100km_extrahigh_adj - meas_fe_extrahigh)/meas_fe_extrahigh * 100:+.2f}%)")

print("COMBINED")
print(f"  Target: {meas_fe_combined} L/100km")
print(f"  Before: {fe_litersPer100km_combined:.2f} L/100km ({(fe_litersPer100km_combined - meas_fe_combined)/meas_fe_combined * 100:+.2f}%)")

# %%
fig1 = plt.figure(figsize=(8,8))

x = cyc.time_s[low]

fig1.suptitle("WLTC Low, Toyota Yaris Hybrid Mid", fontweight="bold", size="x-large")

y1 = sim.fs_kw_out_ach[low]
ax1 = fig1.add_subplot(311)
ax1.plot(x, y1, color="C0")
ax1.set_xlabel("Time (sec)")
ax1.set_ylabel("Power (kW)")
ax1.set_title("sim.fs_kw_out_ach", fontweight="bold")

y2 = sim.soc[low] * 100
ax2 = fig1.add_subplot(312, sharex=ax1)
ax2.plot(x, y2, color="C1")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("SOC (%)")
ax2.set_title("sim.soc", fontweight="bold")

y3 = cyc.mph[low]
ax3 = fig1.add_subplot(313, sharex=ax1)
ax3.plot(x, y3, color="C2")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Velocity (mph)")
ax3.set_title("cyc.mph", fontweight="bold")

plt.tight_layout()
# plt.savefig("C:/Users/kcarow/Downloads/low.jpg")

# %%
fig2 = plt.figure(figsize=(8,8))

x = cyc.time_s[medium]

fig2.suptitle("WLTC Medium, Toyota Yaris Hybrid Mid", fontweight="bold", size="x-large")

y1 = sim.fs_kw_out_ach[medium]
ax1 = fig2.add_subplot(311)
ax1.plot(x, y1, color="C0")
ax1.set_xlabel("Time (sec)")
ax1.set_ylabel("Power (kW)")
ax1.set_title("sim.fs_kw_out_ach", fontweight="bold")

y2 = sim.soc[medium] * 100
ax2 = fig2.add_subplot(312, sharex=ax1)
ax2.plot(x, y2, color="C1")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("SOC (%)")
ax2.set_title("sim.soc", fontweight="bold")

y3 = cyc.mph[medium]
ax3 = fig2.add_subplot(313, sharex=ax1)
ax3.plot(x, y3, color="C2")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Velocity (mph)")
ax3.set_title("cyc.mph", fontweight="bold")

plt.tight_layout()
# plt.savefig("C:/Users/kcarow/Downloads/medium.jpg")

# %%
fig3 = plt.figure(figsize=(8,8))

x = cyc.time_s[high]

fig3.suptitle("WLTC High, Toyota Yaris Hybrid Mid", fontweight="bold", size="x-large")

y1 = sim.fs_kw_out_ach[high]
ax1 = fig3.add_subplot(311)
ax1.plot(x, y1, color="C0")
ax1.set_xlabel("Time (sec)")
ax1.set_ylabel("Power (kW)")
ax1.set_title("sim.fs_kw_out_ach", fontweight="bold")

y2 = sim.soc[high] * 100
ax2 = fig3.add_subplot(312, sharex=ax1)
ax2.plot(x, y2, color="C1")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("SOC (%)")
ax2.set_title("sim.soc", fontweight="bold")

y3 = cyc.mph[high]
ax3 = fig3.add_subplot(313, sharex=ax1)
ax3.plot(x, y3, color="C2")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Velocity (mph)")
ax3.set_title("cyc.mph", fontweight="bold")

plt.tight_layout()
# plt.savefig("C:/Users/kcarow/Downloads/high.jpg")

# %%
fig4 = plt.figure(figsize=(8,8))

x = cyc.time_s[extrahigh]

fig4.suptitle("WLTC Extra-High, Toyota Yaris Hybrid Mid", fontweight="bold", size="x-large")

y1 = sim.fs_kw_out_ach[extrahigh]
ax1 = fig4.add_subplot(311)
ax1.plot(x, y1, color="C0")
ax1.set_xlabel("Time (sec)")
ax1.set_ylabel("Power (kW)")
ax1.set_title("sim.fs_kw_out_ach", fontweight="bold")

y2 = sim.soc[extrahigh] * 100
ax2 = fig4.add_subplot(312, sharex=ax1)
ax2.plot(x, y2, color="C1")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("SOC (%)")
ax2.set_title("sim.soc", fontweight="bold")

y3 = cyc.mph[extrahigh]
ax3 = fig4.add_subplot(313, sharex=ax1)
ax3.plot(x, y3, color="C2")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Velocity (mph)")
ax3.set_title("cyc.mph", fontweight="bold")

plt.tight_layout()
# plt.savefig("C:/Users/kcarow/Downloads/extra-high.jpg")

# %%
fig5 = plt.figure(figsize=(8,8))

x = cyc.time_s

fig5.suptitle("WLTC Combined, Toyota Yaris Hybrid Mid", fontweight="bold", size="x-large")

y1 = sim.fs_kw_out_ach
ax1 = fig5.add_subplot(311)
ax1.plot(x, y1, color="C0")
ax1.set_xlabel("Time (sec)")
ax1.set_ylabel("Power (kW)")
ax1.set_title("sim.fs_kw_out_ach", fontweight="bold")
ax1.axvspan(0, i0, color="green", alpha=0.1)
ax1.axvspan(i0-1, i1, color="yellow", alpha=0.1)
ax1.axvspan(i1-1, i2, color="salmon", alpha=0.1)
ax1.axvspan(i2-1, max(x), color="red", alpha=0.1)

y2 = sim.soc * 100
ax2 = fig5.add_subplot(312, sharex=ax1)
ax2.plot(x, y2, color="C1")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("SOC (%)")
ax2.set_title("sim.soc", fontweight="bold")
ax2.axvspan(0, i0, color="green", alpha=0.1)
ax2.axvspan(i0-1, i1, color="yellow", alpha=0.1)
ax2.axvspan(i1-1, i2, color="salmon", alpha=0.1)
ax2.axvspan(i2-1, max(x), color="red", alpha=0.1)

y3 = cyc.mph
ax3 = fig5.add_subplot(313, sharex=ax1)
ax3.plot(x, y3, color="C2")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Velocity (mph)")
ax3.set_title("cyc.mph", fontweight="bold")
ax3.axvspan(0, i0, color="green", alpha=0.1)
ax3.axvspan(i0-1, i1, color="yellow", alpha=0.1)
ax3.axvspan(i1-1, i2, color="salmon", alpha=0.1)
ax3.axvspan(i2-1, max(x), color="red", alpha=0.1)

plt.tight_layout()
# plt.savefig("C:/Users/kcarow/Downloads/combined.jpg")


