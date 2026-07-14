"""
# Getting Started

This is an interactive demonstration of FASTSim. Developed by the National
Laboratory of the Rockies (NLR), FASTSim provides extremely fast and accurate
estimates of powertrain performance and fuel (or electricity) consumption for
a wide variety of vehicle types. This page walks through the core workflow:
loading a vehicle, loading a drive cycle, running a simulation, and working
with the results.

## Key Concepts

A FASTSim simulation is built on three main components:

- `fastsim.Vehicle`: An object defining the vehicle's physical specifications,
  including the powertrain configuration, mass, aerodynamic drag, component efficiencies, and more.
- `fastsim.Cycle`: A drive cycle, essentially a velocity vs. time profile
  (e.g. EPA regulatory cycles such as UDDS/HWFET, or custom telematics-derived data).
  Drive cycles define how the vehicle moves and the conditions under which it operates.
- `fastsim.SimDrive`: The solver that combines a `Vehicle` and a `Cycle`,
  calculating the flow of power and energy consumption at every time step.

The line below imports FASTSim in Python:
"""

# %%
import fastsim

"""
## Loading a Vehicle

`Vehicle.from_resource` loads one of the vehicle models bundled with FASTSim.
The full list of bundled vehicles can be printed:
"""

# %%
fastsim.Vehicle.list_resources()

"""
This example uses the conventional 2012 Ford Fusion. After loading, a few
key parameters can be read from the vehicle dictionary:
"""

# %%
veh = fastsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")

veh_dict = veh.to_pydict()
print(f"Vehicle loaded: {veh_dict['name']}")
print(f"Vehicle mass: {veh_dict['mass_kilograms']:.1f} kg")
print(f"Drag coefficient: {veh_dict['chassis']['drag_coef']}")

"""
`save_interval` controls how often the vehicle records its internal state to
history vectors. A value of 1 records every time step, which is what we want
for plotting results.
"""

# %%
veh.set_save_interval(1)

"""
## Loading a Drive Cycle

`Cycle.from_resource` loads a default drive cycle in the FASTSim package
called the Urban Dynamometer Driving Schedule, or "UDDS". The UDDS is one of
several
[EPA regulatory cycles](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules)
used to test vehicle fuel economy. UDDS is also referred to as the "city"
test. Below the cycle is plotted as target vehicle speed vs. time.

For a deeper look at drive cycles, including custom cycles and editing cycle
data, see [What Is a Drive Cycle?](demo_drive_cycle.ipynb).
"""

# %%
cyc = fastsim.Cycle.from_resource("udds.csv")
fig = cyc.plot()

"""
## Running the Simulation

`SimDrive` combines a vehicle and a cycle, and `walk()` runs the simulation
from start to finish.
"""

# %%
sd = fastsim.SimDrive(veh, cyc)
sd.walk()

"""
## Inspecting Results

The `SimDrive` object contains the inputs, and after the `walk()` method is
called, which runs the simulation, it also contains the resulting time-series
data from the simulated vehicle over the provided drive cycle. You can
explore these data to understand exactly how the vehicle is performing.
`to_dataframe()` returns the results as a dataframe:
"""

# %%
df = sd.to_dataframe(pandas=True)
print(f"{len(df)} time steps, {len(df.columns)} columns. A few examples:")
print(df.columns.tolist()[:5])

"""
A common visualization is achieved speed vs. time. This attribute is called
"achieved speed" because it is possible that a vehicle is not able to meet
the provided drive cycle. The two lines should overlap almost exactly for a
vehicle with enough power to follow the trace.
"""

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["cyc.time_seconds"],
    y=df["cyc.speed_meters_per_second"],
    name="Target",
    line={"dash": "dash"},
))
fig.add_trace(go.Scatter(
    x=df["cyc.time_seconds"],
    y=df["veh.history.speed_ach_meters_per_second"],
    name="Achieved",
))
fig.update_layout(xaxis_title="Time [s]", yaxis_title="Speed [m/s]")
fig.show()

"""
Fuel power flowing into the engine shows when the vehicle is working hardest
over the cycle:
"""

# %%
import plotly.express as px

fig = px.line(
    x=df["cyc.time_seconds"],
    y=df["veh.pt_type.Conv.fc.history.pwr_fuel_watts"] / 1e3,
)
fig.update_layout(xaxis_title="Time [s]", yaxis_title="Fuel Power [kW]")
fig.show()

"""
## Calculating Fuel Economy

FASTSim reports energy in SI units, so fuel economy is calculated from
cumulative fuel energy and distance. This example converts fuel energy to
gallons of gasoline equivalent using the conventional 33.7 kWh per gallon.
"""

# %%
KWH_PER_GGE = 33.7
METERS_PER_MILE = 1609.34


def mpg_from_sim(sd) -> float:
    """Compute miles per gallon gasoline equivalent from a solved SimDrive."""
    sd_dict = sd.to_pydict(flatten=True)
    fuel_kwh = sd_dict["veh.pt_type.Conv.fc.state.energy_fuel_joules"] / 3.6e6
    miles = sd_dict["veh.state.dist_meters"] / METERS_PER_MILE
    return miles / (fuel_kwh / KWH_PER_GGE)


mpg = mpg_from_sim(sd)
print(f"Fuel economy over UDDS: {mpg:.1f} mpg")

"""
One common question is: **"Why is the MPG from FASTSim higher than what I see
on a car's window sticker?"** FASTSim simulations, by default, provide "raw"
or "unadjusted" fuel economy. This is equivalent to what a vehicle achieves
on a chassis dynamometer in a laboratory setting under controlled conditions,
which is how vehicles are actually tested to measure fuel economy. The EPA
applies a set of "downward adjustments" (often around 10-30%) to laboratory
results to better reflect real-world driving for the window sticker. See
[Comparing Simulations to Label Fuel Economy](../demo_notebooks/demo_label_fe.ipynb)
for how FASTSim reproduces window sticker values.

## Modifying Vehicle Parameters

One common use of FASTSim is to explore various vehicle designs and
configurations and the subsequent impacts on fuel consumption. Let's start
with a simple example that modifies the vehicle mass.

We will:
1. Create a "Heavy" version of the vehicle.
2. Re-run the simulation.
3. Compare the results.

Vehicle parameters can be edited by converting the vehicle to a dictionary,
changing values, and converting back.
"""

# %%
veh_dict_heavy = veh.to_pydict()
veh_dict_heavy["mass_kilograms"] += 800.0
veh_heavy = fastsim.Vehicle.from_pydict(veh_dict_heavy)
veh_heavy.set_save_interval(1)

sd_heavy = fastsim.SimDrive(veh_heavy, cyc)
sd_heavy.walk()

mpg_heavy = mpg_from_sim(sd_heavy)
print(f"Original mass: {veh_dict['mass_kilograms']:.0f} kg -> {mpg:.1f} mpg")
print(f"Heavy mass:    {veh_dict_heavy['mass_kilograms']:.0f} kg -> {mpg_heavy:.1f} mpg")
print(f"Fuel economy change: {(mpg_heavy - mpg) / mpg * 100:.1f}%")

"""
Let's visualize the difference in power demand.
"""

# %%
df_heavy = sd_heavy.to_dataframe(pandas=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["cyc.time_seconds"],
    y=df["veh.pt_type.Conv.fc.history.pwr_fuel_watts"] / 1e3,
    name="Original",
))
fig.add_trace(go.Scatter(
    x=df_heavy["cyc.time_seconds"],
    y=df_heavy["veh.pt_type.Conv.fc.history.pwr_fuel_watts"] / 1e3,
    name="With 800 kg payload",
))
fig.update_layout(xaxis_title="Time [s]", yaxis_title="Fuel Power [kW]")
fig.show()

"""
## Where to Go Next

- [What Is a Drive Cycle?](demo_drive_cycle.ipynb) covers loading, building,
  and editing drive cycles, including road grade.
- [Vehicles in FASTSim](../content/vehicle.md) describes the vehicle model
  hierarchy in more depth.
- [What is a SimDrive Object?](../content/simdrive.md) explains the
  simulation object and its parameters.
"""
