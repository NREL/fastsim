# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # FASTSim Demonstration
# |
# ![fastsim icon](fastsim-icon-web-131x172.jpg)
#
# Developed by NREL, the Future Automotive Systems Technology Simulator (FASTSim) evaluates the impact of technology improvements on efficiency, performance, cost, and battery life in conventional vehicles, hybrid electric vehicles (HEVs), plug-in hybrid electric vehicles (PHEVs), and all-electric vehicles (EVs).
#
# FASTSim answers questions such as:
# - Which battery sizes are most cost effective for a PHEV or EV?
# - At what battery prices do PHEVs and EVs become cost effective?
# - On average, how much fuel does a PHEV with a 30-mile electric range save?
# - How much fuel savings does an HEV provide for a given drive cycle?
# - How do lifetime costs and petroleum use compare for conventional vehicles, HEVs, PHEVs, and EVs?
#
# FASTSim was originally implemented in Microsoft Excel. The pythonic implementation of FASTSim, demonstrated here, captures the drive cycle energy consumption simulation component of the software. The python version of FASTSim is more convenient than the Excel version when very high computational speed is desired, such as for simulating a large batch of drive cycles.

# %%
import sys
import os
from pathlib import Path
from fastsimrust import abc_to_drag_coeffs
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
# import seaborn as sns
# sns.set(font_scale=2, style='whitegrid')

if not __name__ == "__main__":
    get_ipython().run_line_magic('matplotlib', 'inline')


# local modules
import fastsim as fsim
# importlib.reload(simdrive)
# importlib.reload(cycle)

#%%

v0 = fsim.vehicle.Vehicle.from_vehdb(10, to_rust=False)
v1 = fsim.vehicle.Vehicle.from_vehdb(10, to_rust=False).to_rust()
v2 = fsim.vehicle.Vehicle.from_vehdb(10, to_rust=True) # should not have derived params
v3 = fsim.vehicle.Vehicle.from_vehdb(10, to_rust=True).to_rust()

# %% [markdown]
# ## Individual Drive Cycle
# ### Load Drive Cycle
# 
# Default (UDDS, US06, HWFET) cycles can be loaded from the ```../cycles``` directory, or custom cycles can be specified in the same format. The expected format is a dictionary with the following keys: 
# 
# ```['cycGrade', 'mps', 'time_s', 'road_type']```
# - cycGrade = Road grade [%/100]
# - mps = Vehicle speed [meters per second]
# - time_s = Relative time in the cycles [seconds]
# - road_type = Indicator as to whether or not there is a wireless charging capability from the road to vehicle
# 
# There is no limit to the length of a drive cycle that can be provided as an input to FASTSim.

# %%
t0 = time.time()
cyc = fsim.cycle.Cycle.from_file("udds")
t1 = time.time()
print(f'Time to load cycle: {t1 - t0:.2e} s')

# %% [markdown]
# ### Load Powertrain Model
# 
# A vehicle database in CSV format is required to be in the working directory where FASTSim is running (i.e. the same directory as this notebook). The "get_veh" function selects the appropriate vehicle attributes from the database and contructs the powertrain model (engine efficiency map, etc.). An integer value corresponds to each vehicle in the database. To add a new vehicle, simply populate a new row to the vehicle database CSV.

# %%
t0 = time.time()
veh = fsim.vehicle.Vehicle.from_vehdb(10)
print(f'Time to load vehicle: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Run FASTSim

# %%

# instantiate and run classic version 
t0 = time.time()
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.sim_drive() 
t_py = time.time() - t0
print(f'Time to simulate: {t_py:.2e} s')

rc = cyc.to_rust()
rv = veh.to_rust()
t0 = time.time()
sdr = fsim.simdrive.RustSimDrive(rc, rv)
sdr.sim_drive() 
t_rust = time.time() - t0
print(f'Time to simulate in rust: {t_rust:.2e} s')

print(f"Rust provides a {t_py/t_rust:.5g}x speedup")


# %% [markdown]
# ### Results

# %%
fig, ax = plt.subplots(2, 1, figsize=(9, 5))
ax[0].plot(cyc.time_s, sim_drive.fc_kw_in_ach, label='py')
ax[0].plot(cyc.time_s, sdr.fc_kw_in_ach, linestyle='--', label='rust')
ax[0].legend()
ax[0].set_ylabel('Engine Input\nPower [kW]')

ax[1].plot(cyc.time_s, sim_drive.mph_ach)
ax[1].set_xlabel('Cycle Time [s]')
ax[1].set_ylabel('Speed [MPH]')

plt.show()

# %%
fig, ax = plt.subplots(2, 1, figsize=(9, 5))
ax[0].plot(cyc.time_s, sdr.fc_kw_in_ach - sim_drive.fc_kw_in_ach)
ax[0].set_ylabel('Engine Input\nPower Error [kW]')

ax[1].plot(cyc.time_s, sim_drive.mph_ach)
ax[1].set_xlabel('Cycle Time [s]')
ax[1].set_ylabel('Speed [MPH]')

plt.show()

# %% [markdown]
# ## Running sim_drive_step() with modified auxInKw
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020  
# ### Overriding at each time step

# %%
## Running sim_drive_step() with modified auxInKw
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020

t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9)
cyc = fsim.cycle.Cycle.from_file('udds')
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.init_for_step(init_soc=0.7935)

while sim_drive.i < len(cyc.time_s):
    sim_drive.aux_in_kw[sim_drive.i] = sim_drive.i / cyc.time_s[-1] * 10 
    # above could be a function of some internal sim_drive state
    sim_drive.sim_drive_step()

plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()
print(f'Time to simulate: {time.time() - t0:.2e} s')

# %%
## Running sim_drive_step() with modified auxInKw using Rust
# Note that the aux load array **must** be set as a whole. We currently
# cannot set just an index of an array via the Python bindings to Rust at this time

t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9).to_rust()
cyc = fsim.cycle.Cycle.from_file('udds').to_rust()
sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
sim_drive.init_for_step(init_soc=0.7935)

while sim_drive.i < len(cyc.time_s):
    # NOTE: we need to copy out and in the entire array to work with the Rust version
    # that is, we can't set just a specific element of an array in rust via python bindings at this time
    aux_in_kw = sim_drive.aux_in_kw.tolist()
    aux_in_kw[sim_drive.i] = sim_drive.i / cyc.time_s[-1] * 10 
    sim_drive.aux_in_kw = aux_in_kw
    # above could be a function of some internal sim_drive state
    sim_drive.sim_drive_step()

plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()
print(f'Time to simulate: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Overriding using a constant value

# %%
## Running sim_drive_step() with modified auxInKw
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020

t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9)
cyc = fsim.cycle.Cycle.from_file('udds')
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
auxInKwConst = 12
sim_drive.sim_drive(None, np.ones(len(cyc.time_s))*auxInKwConst)

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')


# %%
## Running sim_drive_step() with modified auxInKw using Rust
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020
t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9).to_rust()
cyc = fsim.cycle.Cycle.from_file('udds').to_rust()
sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
auxInKwConst = 12
sim_drive.sim_drive(None, np.ones(len(cyc.time_s))*auxInKwConst)

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')

# # %% [markdown]
# ### Overriding using a time trace

# %%
## Running sim_drive_step() with modified auxInKw
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020

t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9)
cyc = fsim.cycle.Cycle.from_file('udds')
sim_drive = fsim.simdrive.SimDrive(cyc, veh)

# by assigning the value directly (this is faster than using positional args)
sim_drive.init_for_step(
    0.5,
    aux_in_kw_override=cyc.time_s / cyc.time_s[-1] * 10
)
while sim_drive.i < len(sim_drive.cyc.time_s):
    sim_drive.sim_drive_step()

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')

# %%
## Running sim_drive_step() with modified auxInKw using Rust
# Note that auxInKw is the only variable setup to be externally modified as of 1 July 2020

t0 = time.time()

veh = fsim.vehicle.Vehicle.from_vehdb(9).to_rust()
cyc = fsim.cycle.Cycle.from_file('udds').to_rust()
sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)

# by assigning the value directly (this is faster than using positional args)
sim_drive.init_for_step(
    0.5,
    aux_in_kw_override=np.array(cyc.time_s) / cyc.time_s[-1] * 10
)
while sim_drive.i < len(sim_drive.cyc.time_s):
    sim_drive.sim_drive_step()

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')


# %%
# by assigning positional arguments
# may require recompile if these arguments have not been passed,
# but this is the fastest approach after compilation

veh = fsim.vehicle.Vehicle.from_vehdb(9)
cyc = fsim.cycle.Cycle.from_file('udds')

t0 = time.time()

sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.sim_drive(None, cyc.time_s / cyc.time_s[-1] * 10)

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')

# %%
# by assigning positional arguments (using Rust)
# may require recompile if these arguments have not been passed,
# but this is the fastest approach after compilation

veh = fsim.vehicle.Vehicle.from_vehdb(9).to_rust()
cyc = fsim.cycle.Cycle.from_file('udds').to_rust()

t0 = time.time()

sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
sim_drive.sim_drive(None, np.array(cyc.time_s) / cyc.time_s[-1] * 10)

plt.figure()
plt.plot(cyc.time_s, sim_drive.fc_kw_out_ach, label='FC out')
plt.plot(cyc.time_s, sim_drive.ess_kw_out_ach, label='ESS out')
plt.xlabel('Time [s]')
plt.ylabel('Power [kW]')
plt.legend()
plt.show()

print(f'Time to simulate: {time.time() - t0:.2e} s')

# %% [markdown]
# ## Batch Drive Cycles - TSDC Drive Cycles
# 
# FASTSim's most significant advantage over other powertrain simulation tools comes from the ability 
# to simulate many drive cycles quickly. The same three steps described above (load cycle, load model, run FASTSim) 
# will be used here, however, the demonstration highlights how quickly FASTSim runs over __2,225 miles of driving__ 
# data for 22 vehicles.  Running on a single core, the 241 drive cycles take roughly 25 seconds to run. Each drive 
# cycle requires a fraction of a second of computational time. 
# 
# The drive cycles simulated are from a subset of Chicago Regional Household Travel Inventory housed in the the 
# Transportation Secure Data Center ([TSDC](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-cleansed-data.html)). 
# Cycles within the TSDC are publicly available for download and easily integrate with FASTSim. You may contact the 
# [TSDC](tsdc@nrel.gov) for general questions on the data center, or [Venu Garikapati](venu.garikapati@nrel.gov) for 
# partnership-related inquiries. 
# 
# ### Load Cycles
# Iterate through the drive cycles directory structure and load the cycles into one pandas dataframe. If memory is an issue, 
# this processing can be broken into smaller chunks. The points table must have trip identifiers appended to run FASTSim on 
# individual trips. The trips are identified and labeled using the start and end timestamps in the "trips.csv" summary tables 
# in each of the vehicle directories downloadable from the TSDC.

# %%
t0 = time.time()
data_path = Path(fsim.simdrive.__file__).parent / 'resources/cycles/cmap_subset/'  # path to drive cycles

drive_cycs_df = pd.DataFrame()
trips_df = pd.DataFrame()

veh_dirs = os.listdir(data_path)
veh_dirs = [dn for dn in veh_dirs if not dn.startswith('.')]

unique_tripno = 0
for subdir in veh_dirs:
    sampno = int(subdir.split('_')[0])
    vehno = int(subdir.split('_')[1])
    
    dc_csvs = os.listdir(data_path / subdir)
    dc_csvs = [fn for fn in dc_csvs if not fn.endswith('trips.csv')]
    
    df_i = pd.read_csv(data_path / subdir / 'trips.csv', index_col=False)
    trips_df = pd.concat([trips_df,df_i],ignore_index=True)
    #trips_df = trips_df.append(df_i, ignore_index=True)
    
    veh_pnts_df = pd.DataFrame()
    
    for j in dc_csvs:
        df_j = pd.read_csv(data_path / subdir / j, index_col=False)
        veh_pnts_df = pd.concat([veh_pnts_df, df_j],ignore_index=True)
        #veh_pnts_df = veh_pnts_df.append(df_j, ignore_index=True)
        
    for k in range(len(df_i)):
        start_ts = df_i.start_ts.iloc[k]
        end_ts = df_i.end_ts.iloc[k]
        tripK_df = veh_pnts_df.loc[
            (veh_pnts_df['timestamp'] >= start_ts) & (veh_pnts_df['timestamp'] <= end_ts)].copy()
        tripK_df['nrel_trip_id'] = [unique_tripno] * len(tripK_df)
        unique_tripno += 1
        tripK_df['sampno'] = [sampno] * len(tripK_df)
        tripK_df['vehno'] = [vehno] * len(tripK_df)
        drive_cycs_df = pd.concat([drive_cycs_df, tripK_df],ignore_index=True)
        #drive_cycs_df = drive_cycs_df.append(tripK_df, ignore_index=True)
t1 = time.time()
print(f'Time to load cycles: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Load Model, Run FASTSim
# Includes example of how to load cycle from dict

# %%
veh = fsim.vehicle.Vehicle.from_vehdb(1)  # load vehicle model
output = {}

results_df = pd.DataFrame()
t_start = time.time()
for trp in list(drive_cycs_df.nrel_trip_id.unique()):
    pnts = drive_cycs_df[drive_cycs_df['nrel_trip_id'] == trp].copy()
    pnts['time_local'] = pd.to_datetime(pnts['timestamp'])

    cyc = {}
    cyc['cycGrade'] = np.zeros(len(pnts))
    cyc['mps'] = np.array(
        pnts['speed_mph'] / fsim.params.MPH_PER_MPS)  # MPH to MPS conversion
    cyc['time_s'] = np.array(
        np.cumsum(
            (pnts['time_local'] -
             pnts['time_local'].shift()).fillna(pd.Timedelta(seconds=0)).astype('timedelta64[s]')
        )
    )
    cyc['road_type'] = np.zeros(len(pnts))
    # example of loading cycle from dict
    cyc = fsim.cycle.Cycle.from_dict(cyc)
    
    sim_drive = fsim.simdrive.SimDrive(cyc, veh)
    sim_drive.sim_drive()

    output['nrel_trip_id'] = trp
    output['distance_mi'] = sum(sim_drive.dist_mi)
    duration_sec = sim_drive.cyc.time_s[-1] - sim_drive.cyc.time_s[0]
    output['avg_speed_mph'] = sum(
        sim_drive.dist_mi) / (duration_sec / 3600.0)
    #results_df = results_df.append(output, ignore_index=True)
    results_df = pd.concat([results_df,pd.DataFrame(output,index=[0])],ignore_index=True)
    output['mpgge'] = sim_drive.mpgge
    
t_end = time.time()

# results_df = results_df.astype(float)

print(f'Simulations Complete. Total runtime = {t_end - t_start:.2f} s')
print('     Average time per cycle = {:.2f} s'.format((
    t_end - t_start) / len(drive_cycs_df.nrel_trip_id.unique())))

# %%
# ... and the Rust version
veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()  # load vehicle model
output = {}

rust_results_df = pd.DataFrame()
t_start = time.time()
for trp in list(drive_cycs_df.nrel_trip_id.unique()):
    pnts = drive_cycs_df[drive_cycs_df['nrel_trip_id'] == trp].copy()
    pnts['time_local'] = pd.to_datetime(pnts['timestamp'])

    cyc = {}
    cyc['cycGrade'] = np.zeros(len(pnts))
    cyc['mps'] = np.array(
        pnts['speed_mph'] / fsim.params.MPH_PER_MPS)  # MPH to MPS conversion
    cyc['time_s'] = np.array(
        np.cumsum(
            (pnts['time_local'] -
             pnts['time_local'].shift()).fillna(pd.Timedelta(seconds=0)).astype('timedelta64[s]')
        )
    )
    cyc['road_type'] = np.zeros(len(pnts))
    # example of loading cycle from dict
    cyc = fsim.cycle.Cycle.from_dict(cyc).to_rust()
    
    sim_drive = fsim.simdrive.RustSimDrive(cyc, veh)
    sim_drive.sim_drive()

    output['nrel_trip_id'] = trp
    output['distance_mi'] = sum(sim_drive.dist_mi)
    duration_sec = sim_drive.cyc.time_s[-1] - sim_drive.cyc.time_s[0]
    output['avg_speed_mph'] = sum(
        sim_drive.dist_mi) / (duration_sec / 3600.0)
    rust_results_df = pd.concat([results_df, pd.DataFrame(output,index=[0])],  ignore_index=True)
    #rust_results_df = results_df.append(output, ignore_index=True)
    output['mpgge'] = sim_drive.mpgge
    
t_end = time.time()

# results_df = results_df.astype(float)

print(f'Simulations Complete. Total runtime = {t_end - t_start:.2f} s')
print('     Average time per cycle = {:.2f} s'.format((
    t_end - t_start) / len(drive_cycs_df.nrel_trip_id.unique())))

# %% [markdown]
# ### Results
# 
# In this demo, the batch results from all 494 drive cycles were output to a 
# Pandas Dataframe to simplify post-processing. Any python data structure or 
# output file format can be used to save batch results. For simplicity, time 
# series data was not stored, but it could certainly be included in batch processing.
# In order to plot the data, a handful of results are filtered out either because 
# they are much longer than we are interested in, or there was some GPS issue in 
# data acquisition that led to an unrealistically high cycle average speed.

# %%
df_fltr = results_df[(results_df['distance_mi'] < 1000)
                     & (results_df['distance_mi'] > 0) &
                     (results_df['avg_speed_mph'] < 100)]


# %%
plt.figure()
df_fltr.mpgge.hist(bins=20, rwidth=.9)
plt.xlabel('Miles per Gallon')
plt.ylabel('Number of Cycles')
plt.show()


# %%
df_fltr.plot(
    x='avg_speed_mph',
    y='mpgge',
    kind='scatter',
    s=df_fltr['distance_mi'] * 5,
    alpha=0.3)

# Configure legend and axes
l1 = plt.scatter([], [], s=5, edgecolors='none', color='xkcd:bluish')
l2 = plt.scatter([], [], s=50, edgecolors='none', color='xkcd:bluish')
l3 = plt.scatter([], [], s=250, edgecolors='none', color='xkcd:bluish')

labels = ["1 Mile", "10 Miles", "50 Miles"]

plt.legend(
    [l1, l2, l3],
    labels,
    title='Cycle Distance',
    frameon=True,
    fontsize=12,
    scatterpoints=1)
plt.xlabel('Average Cycle Speed [MPH]')
plt.ylabel('Fuel Economy [MPG]')
plt.show()

# %% [markdown]
# # Cycle manipulation tools
# %% [markdown]
# ## Micro-trip

# %%
# load vehicle
t0 = time.time()
veh = fsim.vehicle.Vehicle.from_vehdb(9)
# veh = veh
print(f'Time to load vehicle: {time.time() - t0:.2e} s')


# %%
# generate micro-trip 
t0 = time.time()
cyc = fsim.cycle.Cycle.from_file("udds")
microtrips = fsim.cycle.to_microtrips(cyc.get_cyc_dict())
cyc = fsim.cycle.Cycle.from_dict(microtrips[1])
print(f'Time to load cycle: {time.time() - t0:.2e} s')


# %%
# simulate
t0 = time.time()
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.sim_drive()
# sim_drive = fsim.simdrive.SimDriveClassic(cyc, veh)
# sim_drive.sim_drive()
print(f'Time to simulate: {time.time() - t0:.2e} s')

t0 = time.time()
sim_drive_post = fsim.simdrive.SimDrivePost(sim_drive)
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(f'Time to post process: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Results

# %%

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = ax.plot(cyc.time_s, sim_drive.fc_kw_in_ach, label='kW')

ax2 = ax.twinx()
speed_line = ax2.plot(cyc.time_s, sim_drive.mph_ach, color='xkcd:pale red', label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# %% [markdown]
# ## Concat cycles/trips
# Includes examples of loading vehicle from standalone file and loading non-standard 
# cycle from file

# %%
# load vehicle
t0 = time.time()
# load from standalone vehicle file
veh = fsim.vehicle.Vehicle.from_file('2012_Ford_Fusion.csv') # load vehicle using name
print(f'Time to load veicle: {time.time() - t0:.2e} s')


# %%
# generate concatenated trip
t0 = time.time()
# load from cycle file path
cyc1 = fsim.cycle.Cycle.from_file(
    str(Path(fsim.simdrive.__file__).parent / 'resources/cycles/udds.csv'))
cyc2 = fsim.cycle.Cycle.from_file("us06")
cyc_combo = fsim.cycle.concat([cyc1.get_cyc_dict(), cyc2.get_cyc_dict()])
cyc_combo = fsim.cycle.Cycle.from_dict(cyc_combo)
print(f'Time to load cycles: {time.time() - t0:.2e} s')


# %%
# simulate
t0 = time.time()
sim_drive = fsim.simdrive.SimDrive(cyc_combo, veh)
sim_drive.sim_drive()
# sim_drive = fsim.simdrive.SimDriveClassic(cyc, veh)
# sim_drive.sim_drive()
print(f'Time to simulate: {time.time() - t0:.2e} s')

t0 = time.time()
sim_drive_post = fsim.simdrive.SimDrivePost(sim_drive)
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(f'Time to post process: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Results

# %%
fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = ax.plot(sim_drive.cyc.time_s, sim_drive.fc_kw_in_ach, label='kW')

ax2 = ax.twinx()
speed_line = ax2.plot(sim_drive.cyc.time_s, sim_drive.mph_ach, color='xkcd:pale red', label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# %% [markdown]
# ## Cycle comparison

# %%
# generate concatenated trip
t0 = time.time()
cyc1 = fsim.cycle.Cycle.from_file("udds")
cyc2 = fsim.cycle.Cycle.from_file("us06")
print('Cycle 1 and 2 equal?')
print(fsim.cycle.equals(cyc1.get_cyc_dict(), cyc2.get_cyc_dict()))
cyc1 = fsim.cycle.Cycle.from_file("udds")
cyc2 = fsim.cycle.Cycle.from_file("udds")
print('Cycle 1 and 2 equal?')
print(fsim.cycle.equals(cyc1.get_cyc_dict(), cyc2.get_cyc_dict()))
cyc2dict = cyc2.get_cyc_dict()
cyc2dict['extra key'] = None
print('Cycle 1 and 2 equal?')
print(fsim.cycle.equals(cyc1.get_cyc_dict(), cyc2dict))
print(f'Time to load and compare cycles: {time.time() - t0:.2e} s')

# %% [markdown]
# ## Resample

# %%
t0 = time.time()
cyc = fsim.cycle.Cycle.from_file("udds")
cyc10Hz = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cyc.get_cyc_dict(), new_dt=0.1))
cyc10s = fsim.cycle.Cycle.from_dict(fsim.cycle.resample(cyc.get_cyc_dict(), new_dt=10))

plt.plot(cyc10Hz.time_s, cyc10Hz.mph, marker=',')
plt.plot(cyc10s.time_s, cyc10s.mph, marker=',')
plt.xlabel('Cycle Time [s]')
plt.ylabel('Vehicle Speed [mph]')
plt.show()
print(f'Time to load and resample: {time.time() - t0:.2e} s')

# %% [markdown]
# ## Concat cycles of different time steps and resample
# This is useful if you have test data with either a variable or overly high sample rate.  

# %%
# load vehicle
t0 = time.time()
# load vehicle using explicit path
veh = fsim.vehicle.Vehicle.from_file(Path(fsim.simdrive.__file__).parent / 
                      'resources/vehdb/2012_Ford_Fusion.csv')
print(f'Time to load vehicle: {time.time() - t0:.2e} s')


# %%
# generate concatenated trip
t0 = time.time()
cyc_udds = fsim.cycle.Cycle.from_file("udds")
# Generate cycle with 0.1 s time steps
cyc_udds_10Hz = fsim.cycle.Cycle.from_dict(
    fsim.cycle.resample(cyc_udds.get_cyc_dict(), new_dt=0.1))
cyc_us06 = fsim.cycle.Cycle.from_file("us06")
cyc_combo = fsim.cycle.concat([cyc_udds_10Hz.get_cyc_dict(), cyc_us06.get_cyc_dict()])
cyc_combo = fsim.cycle.resample(cyc_combo, new_dt=1)
cyc_combo = fsim.cycle.Cycle.from_dict(cyc_combo)
print(f'Time to load and concatenate cycles: {time.time() - t0:.2e} s')


# %%
# simulate
t0 = time.time()
sim_drive = fsim.simdrive.SimDrive(cyc_combo, veh)
sim_drive.sim_drive()
# sim_drive = fsim.simdrive.SimDriveClassic(cyc, veh)
# sim_drive.sim_drive()
print(f'Time to simulate: {time.time() - t0:.2e} s')

t0 = time.time()
sim_drive_post = fsim.simdrive.SimDrivePost(sim_drive)
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(f'Time to post process: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Results

# %%
fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = ax.plot(sim_drive.cyc.time_s, sim_drive.fc_kw_in_ach, label='kW')

ax2 = ax.twinx()
speed_line = ax2.plot(sim_drive.cyc.time_s, sim_drive.mph_ach, color='xkcd:pale red', label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# %% [markdown]
# ## Clip by times

# %%
# load vehicle
t0 = time.time()
veh = fsim.vehicle.Vehicle.from_vehdb(1)
# veh = veh
print(f'Time to load vehicle: {time.time() - t0:.2e} s')


# %%
# generate micro-trip 
t0 = time.time()
cyc = fsim.cycle.Cycle.from_file("udds")
cyc = fsim.cycle.clip_by_times(cyc.get_cyc_dict(), t_end=300)
cyc = fsim.cycle.Cycle.from_dict(cyc)
print(f'Time to load and clip cycle: {time.time() - t0:.2e} s')


# %%
# simulate
t0 = time.time()
sim_drive = fsim.simdrive.SimDrive(cyc, veh)
sim_drive.sim_drive()
# sim_drive = fsim.simdrive.SimDriveClassic(cyc, veh)
# sim_drive.sim_drive()
print(f'Time to simulate: {time.time() - t0:.2e} s')

t0 = time.time()
sim_drive_post = fsim.simdrive.SimDrivePost(sim_drive)
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(f'Time to post process: {time.time() - t0:.2e} s')

# %% [markdown]
# ### Results

# %%
fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = ax.plot(sim_drive.cyc.time_s, sim_drive.fc_kw_in_ach, label='kW')

ax2 = ax.twinx()
speed_line = ax2.plot(sim_drive.cyc.time_s, sim_drive.mph_ach ,color='xkcd:pale red', label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# %% [markdown]
# ### Test Coefficients Calculation
# 
# Test drag and wheel rolling resistance calculation from coastdown test values.

# %%
test_veh = fsim.vehicle.Vehicle.from_vehdb(5, to_rust=True).to_rust()
(drag_coef, wheel_rr_coef) = abc_to_drag_coeffs(test_veh, 25.91, 0.1943, 0.01796, simdrive_optimize=True)
print(f'Drag Coefficient: {drag_coef}')
print(f'Wheel Rolling Resistance Coefficient: {wheel_rr_coef}')
# %%
