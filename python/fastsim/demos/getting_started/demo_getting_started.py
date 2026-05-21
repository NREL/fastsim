"""
# Getting Started with FASTSim

In this demo you'll learn about the FASTSim core workflow:
load a vehicle, load a drive cycle, and then run a simulation, followed by
inspecting the results.
"""

# %%
import matplotlib.pyplot as plt
import fastsim as fsim

"""
## Setting Up a Simulation

Every FASTSim simulation needs a vehicle, a drive cycle, and a `SimDrive`
that ties them together.
"""

"""
`Vehicle.from_resource` loads one of the vehicle YAML files bundled with
FASTSim. Here we're loading a 2012 Ford Fusion, a conventional ICE vehicle.
Other bundled vehicles include the 2016 Toyota Prius (HEV), 2016 Nissan Leaf
(BEV), 2020 Chevrolet Bolt (BEV), 2021 Hyundai Sonata Hybrid (HEV),
2022 Renault Zoe (BEV), and 2022 Tesla Model 3 (BEV). You can also load your
own vehicle definitions from a YAML file with `Vehicle.from_file`.

`save_interval` controls how often the vehicle saves its internal state to
history. The bundled vehicles default to saving every time step, which is what
we want here for plotting. If you don't need per-step data, you can set a
larger interval to save less frequently, or pass `None` to disable history
recording entirely.
"""

# %%
veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
veh.set_save_interval(1)

"""
`Cycle.from_resource` loads a drive cycle the same way. We're using UDDS,
the EPA Urban Dynamometer Driving Schedule, which represents city driving
conditions. FASTSim also bundles HWFET for highway driving. Like vehicles,
you can load custom cycles from CSV files with `Cycle.from_file`.
"""

# %%
cyc = fsim.Cycle.from_resource("udds.csv")

"""
`SimDrive` is the simulation runner. It takes a vehicle and a cycle and
computes the vehicle's powertrain response at each time step. Calling
`walk()` executes the simulation to completion.
"""

# %%
sd = fsim.SimDrive(veh, cyc)
sd.walk()

"""
## Looking at Results

There are two main ways to get data out of a completed simulation.
`to_dataframe()` returns a Polars DataFrame by default (pass `pandas=True`
for pandas) with one row per saved time step, useful for plotting time series.
`to_pydict(flatten=True)` serializes the full simulation state into a flat
dictionary with dot-separated keys, handy for pulling out specific values
like total fuel consumed.
"""

# %%
df = sd.to_dataframe(pandas=True)
sd_dict = sd.to_pydict(flatten=True)

print(f"Total fuel energy: {sd_dict['veh.pt_type.Conv.fc.state.energy_fuel_joules'] / 1e6:.2f} MJ")
print(f"Number of time steps: {len(df)}")
print(f"\nFirst 10 columns (of {len(df.columns)}):")
print(df.columns.tolist()[:10])

"""
This plot compares the target speed from the drive cycle against what the
vehicle actually achieved. For a properly parameterized vehicle on UDDS,
these should overlap almost exactly.
"""

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["cyc.time_seconds"], df["cyc.speed_meters_per_second"], label="Target", alpha=0.7)
ax.plot(df["cyc.time_seconds"], df["veh.history.speed_ach_meters_per_second"], label="Achieved", linestyle="--")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Speed [m/s]")
ax.set_title("UDDS Drive Cycle: Target vs. Achieved Speed")
ax.legend()
plt.tight_layout()
plt.show()

"""
Here we plot the fuel converter's total output power (propulsion + auxiliary)
over time. You can see when the engine is active and how hard it's working.
During vehicle stops the engine still runs to supply auxiliary loads, so
power doesn't drop to zero.
"""

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    df["cyc.time_seconds"],
    (df["veh.pt_type.Conv.fc.history.pwr_prop_watts"] + df["veh.pt_type.Conv.fc.history.pwr_aux_watts"]) / 1e3,
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("FC Power [kW]")
ax.set_title("Fuel Converter Output Power")
plt.tight_layout()
plt.show()

"""
Cumulative fuel energy shows the total fuel consumed up to each point in the
cycle. The slope tells you the instantaneous rate of consumption: steeper
during acceleration, shallower during idle (where the engine still consumes
fuel to overcome internal friction and supply auxiliary loads).
"""

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    df["cyc.time_seconds"],
    df["veh.pt_type.Conv.fc.history.energy_fuel_joules"] / 1e6,
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Cumulative Fuel Energy [MJ]")
ax.set_title("Cumulative Fuel Consumption")
plt.tight_layout()
plt.show()
