# # FASTSim Demonstration
# 
# ![fastsim icon](icon_fastsim.jpg)
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
import sys
import os
from pathlib import Path
# allow it to find simdrive module
fsimpath=str(Path(os.getcwd()).parents[0])
if fsimpath not in sys.path:
    sys.path.append(fsimpath)
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import importlib
# import seaborn as sns
# sns.set(font_scale=2, style='whitegrid')

# local modules
from fastsim import simdrive, vehicle, cycle
# importlib.reload(simdrive)
# importlib.reload(cycle)

print('')
print('*'*30)
print('For a better experience, open demo.ipynb in a jupyter notebook.\n')

# ## Individual Drive Cycle
# ### Load Drive Cycle
# 
# Default (UDDS, US06, HWFET) cycles can be loaded from the ```../cycles``` directory, or custom cycles can be specified in the same format. The expected format is a dictionary with the following keys: 
# 
# ```['cycGrade', 'cycMps', 'cycSecs', 'cycRoadType']```
# - cycGrade = Road grade [%/100]
# - cycMps = Vehicle speed [meters per second]
# - cycSecs = Relative time in the cycles [seconds]
# - cycRoadType = Indicator as to whether or not there is a wireless charging capability from the road to vehicle
# 
# There is no limit to the length of a drive cycle that can be provided as an input to FASTSim.

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc_jit = cyc.get_numba_cyc()
print(time.time() - t0)

# ### Load Powertrain Model
# 
# A vehicle database in CSV format is required to be in the working directory where FASTSim is running (i.e. the same directory as this notebook). The "get_veh" function selects the appropriate vehicle attributes from the database and contructs the powertrain model (engine efficiency map, etc.). An integer value corresponds to each vehicle in the database. To add a new vehicle, simply populate a new row to the vehicle database CSV.

t0 = time.time()
veh = vehicle.Vehicle(1)
veh_jit = veh.get_numba_veh()
print(time.time() - t0)

# ### Run FASTSim
# 
# The "sim_drive" function takes the drive cycle and vehicle models defined above as inputs. The output is a dictionary of time series and scalar values described the simulation results. Typically of interest is the "gge" key, which is an array of time series energy consumption data at each time step in the drive cycle. Additionally, to add a result from the simulator to the output dictionary, a user can modify the sim_drive_sub function source code to include the desired field.
# 
# If running FASTSim in batch over many drive cycles, the output from "sim_drive" can be written to files or database for batch post-processing. 

t0 = time.time()
sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
sim_drive.sim_drive(-1)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
# sim_drive.sim_drive()
sim_drive.set_post_scalars()
print(time.time() - t0)

t0 = time.time()
sim_drive_post = simdrive.SimDrivePost(sim_drive)
output = sim_drive_post.get_output()
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(time.time() - t0)

# ### Results

df = pd.DataFrame.from_dict(output)[['soc','fcKwInAch']]
df['speed'] = cyc.cycMps * 2.23694  # Convert mps to mph

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = df.fcKwInAch.plot(ax=ax, label='kW')

ax2 = ax.twinx()
speed_line = df.speed.plot(color='xkcd:pale red', ax=ax2, label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')

# ## Batch Drive Cycles - TSDC Drive Cycles
# 
# FASTSim's most significant advantage over other powertrain simulation tools comes from the ability to simulate many drive cycles quickly. The same three steps described above (load cycle, load model, run FASTSim) will be used here, however, the demonstration highlights how quickly FASTSim runs over __2,225 miles of driving__ data for 22 vehicles.  Running on a single core, the 241 drive cycles take roughly 25 seconds to run. Each drive cycle requires a fraction of a second of computational time. 
# 
# The drive cycles simulated are from a subset of Chicago Regional Household Travel Inventory housed in the the Transportation Secure Data Center ([TSDC](https://www.nrel.gov/transportation/secure-transportation-data/tsdc-cleansed-data.html)). Cycles within the TSDC are publicly available for download and easily integrate with FASTSim. You may contact the [TSDC](tsdc@nrel.gov) for general questions on the data center, or [Venu Garikapati](venu.garikapati@nrel.gov) for partnership-related inquiries. 
# 
# ### Load Cycles
# Iterate through the drive cycles directory structure and load the cycles into one pandas dataframe. If memory is an issue, this processing can be broken into smaller chunks. The points table must have trip identifiers appended to run FASTSim on individual trips. The trips are identified and labeled using the start and end timestamps in the "trips.csv" summary tables in each of the vehicle directories downloadable from the TSDC.

t0 = time.time()
data_path = '../cycles/cmap_subset/'  # path to drive cycles

drive_cycs_df = pd.DataFrame()
trips_df = pd.DataFrame()

veh_dirs = os.listdir(data_path)
veh_dirs = [dn for dn in veh_dirs if not dn.startswith('.')]

unique_tripno = 0
for i in veh_dirs:
    sampno = int(i.split('_')[0])
    vehno = int(i.split('_')[1])
    
    dc_csvs = os.listdir(data_path+i)
    dc_csvs = [fn for fn in dc_csvs if not fn.endswith('trips.csv')]
    
    df_i = pd.read_csv(data_path+i+'/trips.csv', index_col=False)
    trips_df = trips_df.append(df_i, ignore_index=True)
    
    veh_pnts_df = pd.DataFrame()
    
    for j in dc_csvs:
        df_j = pd.read_csv(data_path+i+'/'+j, index_col=False)
        veh_pnts_df = veh_pnts_df.append(df_j, ignore_index=True)
        
    for k in range(len(df_i)):
        start_ts = df_i.start_ts.iloc[k]
        end_ts = df_i.end_ts.iloc[k]
        tripK_df = veh_pnts_df.loc[(veh_pnts_df['timestamp']>=start_ts) &                         (veh_pnts_df['timestamp']<=end_ts)]
        tripK_df['nrel_trip_id'] = [unique_tripno]*len(tripK_df)
        unique_tripno += 1
        tripK_df['sampno'] = [sampno]*len(tripK_df)
        tripK_df['vehno'] = [vehno]*len(tripK_df)
        drive_cycs_df = drive_cycs_df.append(tripK_df, ignore_index=True)
t1 = time.time()
print('Elapsed time = ' + str(round(t1 - t0, 3)))

# ### Load Model, Run FASTSim

veh = vehicle.Vehicle(1).get_numba_veh()  # load vehicle model
output_dict = {}

results_df = pd.DataFrame()
t_start = time.time()
for trp in list(drive_cycs_df.nrel_trip_id.unique()):
    pnts = drive_cycs_df[drive_cycs_df['nrel_trip_id'] == trp]
    pnts['time_local'] = pd.to_datetime(pnts['timestamp'])

    cyc = {}
    cyc['cycGrade'] = np.zeros(len(pnts))
    cyc['cycMps'] = np.array(
        pnts['speed_mph'] * 0.44704)  # MPH to MPS conversion
    cyc['cycSecs'] = np.array(
        np.cumsum(
            (pnts['time_local'] -
             pnts['time_local'].shift()).fillna(pd.Timedelta(seconds=0)).astype('timedelta64[s]')))
    cyc['cycRoadType'] = np.zeros(len(pnts))
    cyc = simdrive.Cycle(cyc_dict=cyc).get_numba_cyc()
    
    sim_drive = simdrive.SimDriveJit(cyc, veh)
    sim_drive.sim_drive(-1)
    sim_drive_post = simdrive.SimDrivePost(sim_drive)
    output = sim_drive_post.get_output()
    
    del output['soc'], output['fcKwInAch'], output['fcKwOutAch'],    output['fsKwhOutAch']

    output['nrel_trip_id'] = trp
    results_df = results_df.append(output, ignore_index=True)
    
t_end = time.time()

# results_df = results_df.astype(float)

print('Run Complete. Total runtime = %1.2fs' % (t_end - t_start))
print('     Average time per cycle = %1.2fs' % ((
    t_end - t_start) / len(drive_cycs_df.nrel_trip_id.unique())))

# ### Results
# 
# In this demo, the batch results from all 494 drive cycles were output to a Pandas Dataframe to simplify post-processing. Any python data structure or output file format can be used to save batch results. For simplicity, time series data was not stored, but it could certainly be included in batch processing.
# 
# In order to plot the data, a handful of results are filtered out either because they are much longer than we are interested in, or there was some GPS issue in data acquisition that led to an unrealistically high cycle average speed.

df_fltr = results_df[(results_df['distance_mi'] < 1000)
                     & (results_df['distance_mi'] > 0) &
                     (results_df['avg_speed_mph'] < 100)]

plt.figure()
df_fltr.mpgge.hist(bins=20, rwidth=.9)
plt.xlabel('Miles per Gallon')
plt.ylabel('Number of Cycles')
plt.show()

df_fltr.plot(
    x='avg_speed_mph',
    y='mpgge',
    kind='scatter',
    s=df_fltr['distance_mi'] * 5,
    alpha=0.3)

# Configure legend and axes
l1 = plt.scatter([], [], s=5, edgecolors='none', c='xkcd:bluish')
l2 = plt.scatter([], [], s=50, edgecolors='none', c='xkcd:bluish')
l3 = plt.scatter([], [], s=250, edgecolors='none', c='xkcd:bluish')

labels = ["1 Mile", "10 Miles", "50 Miles"]

leg = plt.legend(
    [l1, l2, l3],
    labels,
    title='Cycle Distance',
    frameon=True,
    fontsize=12,
    scatterpoints=1)
plt.xlabel('Average Cycle Speed [MPH]')
plt.ylabel('Fuel Economy [MPG]')
plt.show()

# # Cycle manipulation tools

# ## Micro-trip

# load vehicle
t0 = time.time()
veh = vehicle.Vehicle(1)
# veh_jit = veh.get_numba_veh()
print(time.time() - t0)

# generate micro-trip 
t0 = time.time()
cyc = cycle.Cycle("udds")
microtrips = cycle.to_microtrips(cyc.get_cyc_dict())
cyc.set_from_dict(microtrips[1])
cyc_jit = cyc.get_numba_cyc()
print(time.time() - t0)

# simulate
t0 = time.time()
sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
sim_drive.sim_drive(-1)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
# sim_drive.sim_drive()
sim_drive.set_post_scalars()
print(time.time() - t0)

t0 = time.time()
sim_drive_post = simdrive.SimDrivePost(sim_drive)
output = sim_drive_post.get_output()
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(time.time() - t0)

# ### Results

df = pd.DataFrame.from_dict(output)[['soc','fcKwInAch']]
df['speed'] = cyc.cycMps * 2.23694  # Convert mps to mph

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = df.fcKwInAch.plot(ax=ax, label='kW')

ax2 = ax.twinx()
speed_line = df.speed.plot(color='xkcd:pale red', ax=ax2, label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# ## Concat cycles/trips

# load vehicle
t0 = time.time()
veh = vehicle.Vehicle(1)
veh_jit = veh.get_numba_veh()
print(time.time() - t0)

# generate concatenated trip
t0 = time.time()
cyc1 = cycle.Cycle("udds")
cyc2 = cycle.Cycle("us06")
cyc_combo = cycle.concat([cyc1.get_cyc_dict(), cyc2.get_cyc_dict()])
cyc_combo = cycle.Cycle(cyc_dict=cyc_combo)
cyc_combo_jit = cyc_combo.get_numba_cyc()
print(time.time() - t0)

# simulate
t0 = time.time()
sim_drive = simdrive.SimDriveJit(cyc_combo_jit, veh_jit)
sim_drive.sim_drive(-1)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
# sim_drive.sim_drive()
sim_drive.set_post_scalars()
print(time.time() - t0)

t0 = time.time()
sim_drive_post = simdrive.SimDrivePost(sim_drive)
output = sim_drive_post.get_output()
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(time.time() - t0)

# ### Results

df = pd.DataFrame.from_dict(output)[['soc','fcKwInAch']]
df['speed'] = cyc_combo.cycMps * 2.23694  # Convert mps to mph

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = df.fcKwInAch.plot(ax=ax, label='kW')

ax2 = ax.twinx()
speed_line = df.speed.plot(color='xkcd:pale red', ax=ax2, label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# ## Cycle comparison

# generate concatenated trip
t0 = time.time()
cyc1 = cycle.Cycle("udds")
cyc2 = cycle.Cycle("us06")
print('Cycle 1 and 2 equal?')
print(cycle.equals(cyc1.get_cyc_dict(), cyc2.get_cyc_dict()))
cyc1 = cycle.Cycle("udds")
cyc2 = cycle.Cycle("udds")
print('Cycle 1 and 2 equal?')
print(cycle.equals(cyc1.get_cyc_dict(), cyc2.get_cyc_dict()))
print(time.time() - t0)

# ## Resample

t0 = time.time()
cyc = cycle.Cycle("udds")
cyc10Hz = cycle.Cycle(cyc_dict=cycle.resample(cyc.get_cyc_dict(), new_dt=0.1))
cyc10s = cycle.Cycle(cyc_dict=cycle.resample(cyc.get_cyc_dict(), new_dt=10))

plt.plot(cyc10Hz.cycSecs, cyc10Hz.cycMph, marker=',')
plt.plot(cyc10s.cycSecs, cyc10s.cycMph, marker=',')
plt.xlabel('Cycle Time [s]')
plt.ylabel('Vehicle Speed [mph]')
print(time.time() - t0)
plt.show()

# ## Concat cycles of different time steps and resample
# This is useful if you have test data with either a variable or overly high sample rate.  

# load vehicle
t0 = time.time()
veh = vehicle.Vehicle(1)
veh_jit = veh.get_numba_veh()
print(time.time() - t0)

# generate concatenated trip
t0 = time.time()
cyc1 = cycle.Cycle("udds")
# Generate cycle with 0.1 s time steps
cyc1 = cycle.Cycle(cyc_dict=cycle.resample(cyc1.get_cyc_dict(), new_dt=0.1))
cyc2 = cycle.Cycle("us06")
cyc_combo = cycle.concat([cyc1.get_cyc_dict(), cyc2.get_cyc_dict()])
cyc_combo = cycle.resample(cyc_combo, new_dt=1)
cyc_combo = cycle.Cycle(cyc_dict=cyc_combo)
cyc_combo_jit = cyc_combo.get_numba_cyc()
print(time.time() - t0)

# simulate
t0 = time.time()
sim_drive = simdrive.SimDriveJit(cyc_combo_jit, veh_jit)
sim_drive.sim_drive(-1)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
# sim_drive.sim_drive()
sim_drive.set_post_scalars()
print(time.time() - t0)

t0 = time.time()
sim_drive_post = simdrive.SimDrivePost(sim_drive)
output = sim_drive_post.get_output()
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(time.time() - t0)

# ### Results

df = pd.DataFrame.from_dict(output)[['soc','fcKwInAch']]
df['speed'] = cyc_combo.cycMps * 2.23694  # Convert mps to mph

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = df.fcKwInAch.plot(ax=ax, label='kW')

ax2 = ax.twinx()
speed_line = df.speed.plot(color='xkcd:pale red', ax=ax2, label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

# ## Clip by times

# load vehicle
t0 = time.time()
veh = vehicle.Vehicle(1)
# veh_jit = veh.get_numba_veh()
print(time.time() - t0)

# generate micro-trip 
t0 = time.time()
cyc = cycle.Cycle("udds")
cyc = cycle.clip_by_times(cyc.get_cyc_dict(), t_end=300)
cyc = cycle.Cycle(cyc_dict=cyc)
cyc_jit = cyc.get_numba_cyc()
print(time.time() - t0)

# simulate
t0 = time.time()
sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
sim_drive.sim_drive(-1)
# sim_drive = simdrive.SimDriveClassic(cyc_jit, veh_jit)
# sim_drive.sim_drive()
sim_drive.set_post_scalars()
print(time.time() - t0)

t0 = time.time()
sim_drive_post = simdrive.SimDrivePost(sim_drive)
output = sim_drive_post.get_output()
sim_drive_post.set_battery_wear()
diag = sim_drive_post.get_diagnostics()
print(time.time() - t0)

# ### Results

df = pd.DataFrame.from_dict(output)[['soc','fcKwInAch']]
df['speed'] = cyc.cycMps * 2.23694  # Convert mps to mph

fig, ax = plt.subplots(figsize=(9, 5))
kwh_line = df.fcKwInAch.plot(ax=ax, label='kW')

ax2 = ax.twinx()
speed_line = df.speed.plot(color='xkcd:pale red', ax=ax2, label='Speed')

ax.set_xlabel('Cycle Time [s]', weight='bold')
ax.set_ylabel('Engine Input Power [kW]', weight='bold', color='xkcd:bluish')
ax.tick_params('y', colors='xkcd:bluish')

ax2.set_ylabel('Speed [MPH]', weight='bold', color='xkcd:pale red')
ax2.grid(False)
ax2.tick_params('y', colors='xkcd:pale red')
plt.show()

