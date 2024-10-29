# %%
import fastsim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

sns.set_style("whitegrid")
np.seterr(invalid="ignore")

# %%
cyc = fastsim.Cycle.from_resource("udds.csv")

# %%
vehs = [fastsim.Vehicle.from_file(filepath) for filepath in glob.glob("f3_vehicles/*.yaml")]

convs = [veh for veh in vehs if veh.veh_type() == "Conv"]
bevs = [veh for veh in vehs if veh.veh_type() == "BEV"]
# hevs = [veh for veh in vehs if veh.veh_type() == "HEV"]

# %%
# Simulate (FASTSim 3)
conv_sds_f3 = [fastsim.SimDrive(veh, cyc) for veh in convs]
bev_sds_f3 = [fastsim.SimDrive(veh, cyc) for veh in bevs]
# hev_sds_f3 = [fastsim.SimDrive(veh, cyc) for veh in hevs]

# for sd in conv_sds + bev_sds + hev_sds
for sd in conv_sds_f3 + bev_sds_f3:
    sd.walk()

# %%
# Simulate (FASTSim 2)
conv_sds_f2 = [sd.to_fastsim2() for sd in conv_sds_f3]
bev_sds_f2 = [sd.to_fastsim2() for sd in bev_sds_f3]
# hev_sds_f2 = [sd.to_fastsim2() for sd in hev_sds_f3]

# for sd in conv_sds_f2 + bev_sds_f2 + hev_sds_f2
for sd in conv_sds_f2 + bev_sds_f2:
    sd.sim_drive()

# %%
def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

def pct_error(actual, expected):
    return 100 * (actual - expected) / expected

def plot(sds_f3: list[fastsim.SimDrive], sds_f2):
    assert len(sds_f3) == len(sds_f2)
    fig, ax = plt.subplots()

    names = [sd.veh.name for sd in sds_f3]
    pt_types = [sd.veh.veh_type() for sd in sds_f3]
    assert len(set(pt_types)) == 1
    pt_type = pt_types[0]

    if pt_type == "Conv":
        f3 = [(np.array(sd.veh.fc.history.energy_propulsion_joules) +
            np.array(sd.veh.fc.history.energy_aux_joules)) / 1e6 for sd in sds_f3]
        f2 = [np.array(sd.fc_cumu_mj_out_ach.tolist()) for sd in sds_f2]
    elif pt_type == "BEV":
        f3 = [np.array(sd.veh.res.history.energy_out_electrical_joules) / 1e3 for sd in sds_f3]
        f2 = [np.cumsum(sd.ess_kw_out_ach.tolist() * np.diff(sd.cyc.time_s.tolist(), prepend=0)) for sd in sds_f2]

    pct_errs = [pct_error(f3_ys, f2_ys) for f3_ys, f2_ys in zip(f3, f2)]
    max_pct_errs = [np.nanmax(pct_err) for pct_err in pct_errs]

    rmses = [rmse(f3_ys, f2_ys) for f3_ys, f2_ys in zip(f3, f2)]

    ax.scatter(names, max_pct_errs)
    ax.tick_params(axis="x", labelrotation=90)
    

# plot(bev_sds_f3, bev_sds_f2)
plot(conv_sds_f3, conv_sds_f2)

# %%
