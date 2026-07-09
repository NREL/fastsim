"""
# What Is a Drive Cycle?

A drive cycle is time series data that describes how a vehicle is driven.
At minimum, it includes speed over time, but can also include
road grade, ambient air temperature, or other time-varying quantities.

FASTSim simulates a vehicle model over each time step of the drive cycle
to compute the vehicle's response, including speed, acceleration, and power demand.

"""

"""
## Loading a Drive Cycle from Resources

This example uses HWFET (Highway Fuel Economy Test), a regulatory drive
cycle used to evaluate highway fuel economy.

For more information on HWFET and other regulatory drive cycles, see:  
https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
"""
# %%
import fastsim
cyc = fastsim.Cycle.from_resource("hwfet.csv")

"""
A full list of drive cycles available in FASTSim's resources can be printed:
"""

# %%
fastsim.Cycle.list_resources()

"""
## Visualizing a Drive Cycle

FASTSim has convenience functions for visualizing drive cycles.
"""
# %%
# Default: x=`time_seconds`, y=`speed_meters_per_second`
fig = cyc.plot()

# Try also:
# - cyc.plot(x="dist_meters")
# - cyc.plot(y="grade")
# - cyc.plot(x="dist_meters", y="grade")

# %%
"""
In FASTSim, drive cycles represent all data that vary over time.

The following are inputs to FASTSim drive cycles:
- Time
  - `time_seconds`
- Vehicle speed
  - `speed_meters_per_second`
- Road grade
  - `grade`
- Ambient air temperature
  - `temp_amb_air_kelvin`
  - Only affects thermal vehicle models

FASTSim automatically derives the following from a drive cycle:
- Distance
  - `dist_meters`
  - Accumulated from vehicle speed
- Elevation
  - `elev_meters`
  - Accumulated from grade and distance
  - Initial elevation
    - `init_elev_meters` defaults to 121.92 m (400 ft)
"""

"""
## Defining Custom Drive Cycles

Drive cycles can be loaded from a variety of file types:
- `.csv` CSV files (like the above example)
- `.json` JSON files
- `.msgpack` MessagePack files
- `.toml` TOML files
- `.yaml` YAML files

Here is a small example of a custom drive cycle file:

`custom_cycle.csv`
```csv
time_seconds,speed_meters_per_second,grade
0,0,0
1,0,0
2,0,0
3,0,0
4,0,0
5,0.5,0
6,0.75,0
7,1,0
8,1.25,0
9,1.5,0
10,1.75,0
11,2,0
12,3,0
13,4,0
14,6,0
15,8,0
16,10,0
17,12,0
18,14,0
19,16,0
20,14,0
21,12,0
22,10,0
23,8,0
24,6,0
25,4,0
26,2,0
27,0,0
28,0,0
29,0,0
30,0,0
```

To load this, you can use `fastsim.Cycle.from_file`:

```python
cyc_custom = fastsim.Cycle.from_file("custom_cycle.csv")
```

"""

"""
## Accessing Drive Cycle Fields at Runtime

Drive cycle fields can be accessed at runtime by converting the `Cycle` object to a Python dictionary.
Each key corresponds to a field name, and many values are lists of data points over time.

"""

# %%
# Convert a Cycle to a Python dictionary
cyc_dict = cyc.to_pydict()
print(cyc_dict.keys())
print(cyc_dict)

# %%
# Iterate over the time, speed, and grade fields and print them
# Limit to 10 values
for time, speed, grade in list(zip(cyc_dict["time_seconds"], cyc_dict["speed_meters_per_second"], cyc_dict["grade"]))[:10]:
    print(f"Time [s]: {time}, Speed [m/s]: {speed}, Grade: {grade}")
print("...")

"""
## Editing Drive Cycle Fields at Runtime
"""

# %%
# Modify the speed field (double the speed) in the drive cycle dictionary
cyc_dict["speed_meters_per_second"] = [s * 2 for s in cyc_dict["speed_meters_per_second"]]
print("Updated speed [m/s]: ", cyc_dict["speed_meters_per_second"][:10])

# Modify the ambient temperature to be 22 °C
cyc_dict["temp_amb_air_kelvin"] = [22 + 273.15] * len(cyc_dict["temp_amb_air_kelvin"])
print("Updated ambient temperature [K]: ", cyc_dict["temp_amb_air_kelvin"][:10])
"""
After making changes to the cycle dictionary, be sure to convert it back to a FASTSim `Cycle` before using it in simulation.
"""

# %%
# Convert the cycle dictionary back into a FASTSim Cycle
cyc = fastsim.Cycle.from_pydict(cyc_dict)

# %%
