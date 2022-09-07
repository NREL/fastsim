# %% [markdown]
# # Connected Autonomous Vehicles Demo
#
# A demonstration of FASTSim functionality for simulating
# connected autonomous vehicles (CAV) and various vehicle
# dynamics (i.e., driving cycle manipulation) techniques.
# %%
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fastsim as fsim
from fastsim.tests.test_coasting import make_coasting_plot
from fastsim.rustext import RUST_AVAILABLE

RAN_SUCCESSFULLY = False

def maybe_str_to_bool(x, default=True):
    """
    Turn values of None or string to bool
    - x: str | None, some parameter, a string or None
    - default: Bool, the default if x is None or blank
    RETURN: True or False
    """
    if x is None:
        return default
    if x is True or x is False:
        return x
    try:
        lower_cased = x.lower().strip()
        if lower_cased == 'false':
            return False
        if lower_cased == 'true':
            return True
        return default
    except:
        return default

IS_INTERACTIVE = maybe_str_to_bool(os.getenv('FASTSIM_DEMO_IS_INTERACTIVE'))


# %% [markdown]
# ## Create a Vehicle and Cycle
# 
# We're going to use a conventional vehicle with
# the UDDS cycle.
# %%
if RUST_AVAILABLE:
    cyc = fsim.cycle.Cycle.from_file('udds').to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
    sd = fsim.simdrive.RustSimDrive(cyc, veh)
    sd.sim_drive()

    base_mpg = sd.mpgge
    if IS_INTERACTIVE:
        print(f"Base fuel economy over UDDS: {sd.mpgge} mpg")
        make_coasting_plot(sd.cyc0, sd.cyc, do_show=True)

# %% [markdown]
# ## Eco-Coasting
# %%
if RUST_AVAILABLE:
    cyc = fsim.cycle.Cycle.from_file('udds').to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
    sd = fsim.simdrive.RustSimDrive(cyc, veh)
    sd.sim_params = fsim.auxiliaries.set_nested_values(sd.sim_params,
        coast_allow=True,
        coast_allow_passing=False,
        coast_start_speed_m_per_s=-1.0
    )
    sd.sim_drive()

    coast_mpg = sd.mpgge
    if IS_INTERACTIVE:
        print(f"Coast fuel economy over UDDS: {sd.mpgge} mpg")
        pct_savings = ((1.0/base_mpg) - (1.0/coast_mpg)) * 100.0 / ((1.0/base_mpg))
        print(f"Percent Savings: {pct_savings} %")
        make_coasting_plot(sd.cyc0, sd.cyc, do_show=True)

# %% [markdown]
# # Car Following at Average Speed
#
# Here we set up an "automated cruise control" for a system
# that drives the average speed of the cycle.
cyc_udds = fsim.cycle.Cycle.from_file('udds').get_cyc_dict()
cyc_stop = fsim.cycle.resample(
    fsim.cycle.make_cycle([0.0, 200.0], [0.0, 0.0]),
    new_dt=1.0,
)
if RUST_AVAILABLE:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    ).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
    sd = fsim.simdrive.RustSimDrive(cyc, veh)
else:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    )
    veh = fsim.vehicle.Vehicle.from_vehdb(1)
    sd = fsim.simdrive.SimDrive(cyc, veh)
sd.sim_params = fsim.auxiliaries.set_nested_values(sd.sim_params,
    idm_allow=True,
    idm_accel_m_per_s2=1.0,
    idm_decel_m_per_s2=-2.5,
    idm_dt_headway_s=2.0,
    idm_minimum_gap_m=0.0,
    idm_v_desired_m_per_s=np.average(np.array(cyc.mps))
)
sd.sim_drive()

cruise_mpg = sd.mpgge
if IS_INTERACTIVE:
    print(f"Cruise fuel economy over UDDS: {sd.mpgge} mpg")
    pct_savings = ((1.0/base_mpg) - (1.0/cruise_mpg)) * 100.0 / ((1.0/base_mpg))
    print(f"Percent Savings: {pct_savings} %")
    make_coasting_plot(sd.cyc0, sd.cyc, do_show=True)

# %% [markdown]
# # Eco-Cruising at Multiple Average Speeds
#
# Here we set up a more realistic version that drives
# the average speed of the microtrip assuming the vehicle
# is able to get the average speed per microtrip from some
# external source.
cyc_udds = fsim.cycle.Cycle.from_file('udds').get_cyc_dict()
cyc_stop = fsim.cycle.resample(
    fsim.cycle.make_cycle([0.0, 200.0], [0.0, 0.0]),
    new_dt=1.0,
)
if RUST_AVAILABLE:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    ).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
    sd = fsim.simdrive.RustSimDrive(cyc, veh)
else:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    )
    veh = fsim.vehicle.Vehicle.from_vehdb(1)
    sd = fsim.simdrive.SimDrive(cyc, veh)
dist_and_avg_speeds = []
microtrips = fsim.cycle.to_microtrips(cyc.get_cyc_dict())
dist_at_start_of_microtrip_m = 0.0
for mt in microtrips:
    mt_cyc = fsim.cycle.Cycle.from_dict(mt)
    if RUST_AVAILABLE:
        mt_cyc = mt_cyc.to_rust()
    mt_dist_m = sum(mt_cyc.dist_m)
    mt_time_s = mt_cyc.time_s[-1]
    mt_avg_spd_m_per_s = mt_dist_m / mt_time_s if mt_time_s > 0.0 else 0.0
    if IS_INTERACTIVE:
        print(f"mt num points      : {len(mt_cyc.time_s)}")
        print(f"mt dist (m)        : {mt_dist_m}")
        print(f"mt time (s)        : {mt_time_s}")
        print(f"mt avg speed (m/s) : {mt_avg_spd_m_per_s}")
    dist_and_avg_speeds.append(
        (dist_at_start_of_microtrip_m, mt_avg_spd_m_per_s)
    )
    dist_at_start_of_microtrip_m += mt_dist_m
if IS_INTERACTIVE:
    print(f"Found speeds for {len(dist_and_avg_speeds)} microtrips")
sd.sim_params = fsim.auxiliaries.set_nested_values(sd.sim_params,
    idm_allow=True,
    idm_accel_m_per_s2=0.5,
    idm_decel_m_per_s2=-2.5,
    idm_dt_headway_s=2.0,
    idm_minimum_gap_m=10.0,
    idm_v_desired_m_per_s=dist_and_avg_speeds[0][1]
)
sd.init_for_step(init_soc=veh.max_soc)
current_mt_idx = 0
dist_traveled_m = 0.0
while sd.i < len(cyc.time_s):
    idx = max(0, sd.i - 1)
    dist_traveled_m += sd.cyc.dist_m[idx]
    lead_veh_speed_m_per_s = sd.cyc0.mps[idx]
    if current_mt_idx < len(dist_and_avg_speeds):
        mt_start_dist_m, mt_avg_spd_m_per_s = dist_and_avg_speeds[current_mt_idx]
        if dist_traveled_m >= mt_start_dist_m:
            sd.sim_params = fsim.auxiliaries.set_nested_values(sd.sim_params,
                idm_v_desired_m_per_s=mt_avg_spd_m_per_s
            )
            if IS_INTERACTIVE:
                print(f"... setting idm_v_desired_m_per_s = {sd.sim_params.idm_v_desired_m_per_s}")
            current_mt_idx += 1
    sd.sim_drive_step()
sd.set_post_scalars()

cruise_mpg = sd.mpgge
if IS_INTERACTIVE:
    print(f"Cruise fuel economy over UDDS: {sd.mpgge} mpg")
    pct_savings = ((1.0/base_mpg) - (1.0/cruise_mpg)) * 100.0 / ((1.0/base_mpg))
    print(f"Percent Savings: {pct_savings} %")
    make_coasting_plot(sd.cyc0, sd.cyc, do_show=True)

# %% [markdown]
# # Eco-Cruise and Eco-Approach running at the same time
#
# Here, we run an Eco-Cruise and Eco-Approach at the same time.
cyc_udds = fsim.cycle.Cycle.from_file('udds').get_cyc_dict()
cyc_stop = fsim.cycle.resample(
    fsim.cycle.make_cycle([0.0, 400.0], [0.0, 0.0]),
    new_dt=1.0,
)
if RUST_AVAILABLE:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    ).to_rust()
    veh = fsim.vehicle.Vehicle.from_vehdb(1).to_rust()
    sd = fsim.simdrive.RustSimDrive(cyc, veh)
else:
    cyc = fsim.cycle.Cycle.from_dict(
        fsim.cycle.concat([cyc_udds, cyc_stop])
    )
    veh = fsim.vehicle.Vehicle.from_vehdb(1)
    sd = fsim.simdrive.SimDrive(cyc, veh)
params = sd.sim_params
params.reset_orphaned()
params.coast_allow = True
params.coast_allow_passing = False
params.coast_start_speed_m_per_s = -1.0
params.idm_allow = True
params.idm_accel_m_per_s2 = 0.5
params.idm_decel_m_per_s2 = -2.5
params.idm_dt_headway_s = 2.0
params.idm_minimum_gap_m = 10.0
params.idm_v_desired_m_per_s = 15.0 # np.sum(cyc_udds['mps']) / cyc_udds['time_s'][-1]
sd.sim_params = params
sd.sim_drive()

if IS_INTERACTIVE:
    eco_mpg = sd.mpgge
    print(f"Cruise and Coast fuel economy over UDDS: {sd.mpgge} mpg")
    pct_savings = ((1.0/base_mpg) - (1.0/eco_mpg)) * 100.0 / ((1.0/base_mpg))
    print(f"Percent Savings: {pct_savings} %")
    make_coasting_plot(sd.cyc0, sd.cyc, do_show=True)


# %%
# The flag below lets us know if this module ran successfully without error
RAN_SUCCESSFULLY = True
# %%
