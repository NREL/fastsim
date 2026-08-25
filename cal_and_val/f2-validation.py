# Script for validation against fastsim-2

# %%
from pathlib import Path
import fastsim

# %%
cyc = fastsim.Cycle.from_resource("udds.csv")

# %%
# Load FASTSim 3 vehicles
F3_VEH_DIR = Path("./f3-vehicles")
skip_files = [
    # "Toyota Hilux Double Cab 4WD.yaml",
    # "Class 4 Truck (Isuzu NPR HD).yaml",
    # "2020 Hero Splendor+ 100cc.yaml",
]
filepaths = [filepath for filepath in F3_VEH_DIR.iterdir() if filepath.name not in skip_files]
f3_vehs = [fastsim.Vehicle.from_file(filepath) for filepath in filepaths]
f3_convs = [v for v in f3_vehs if v.veh_type() == "Conv"]
f3_bevs = [v for v in f3_vehs if v.veh_type() == "BEV"]
f3_hevs = [v for v in f3_vehs if v.veh_type() == "HEV"]

# %%
# Create FASTSim 3 simdrives
f3_sds = [fastsim.SimDrive(veh, cyc) for veh in f3_vehs]
f3_conv_sds = [fastsim.SimDrive(veh, cyc) for veh in f3_convs]
f3_bev_sds = [fastsim.SimDrive(veh, cyc) for veh in f3_bevs]
f3_hev_sds = [fastsim.SimDrive(veh, cyc) for veh in f3_hevs]
f3_all_sds = f3_conv_sds + f3_bev_sds # + f3_hev_sds

# %%
# Convert to FASTSim 2 simdrives
f2_sds = [sd.to_fastsim2() for sd in f3_sds]
f2_conv_sds = [sd.to_fastsim2() for sd in f3_conv_sds]
f2_bev_sds = [sd.to_fastsim2() for sd in f3_bev_sds]
f2_hev_sds = [sd.to_fastsim2() for sd in f3_hev_sds]
f2_all_sds = f2_conv_sds + f2_bev_sds # + f2_hev_sds

# %%
# Simulate FASTSim 3
for sd in f3_all_sds:
    # sd_dict = sd.to_dict(flatten=False)
    # sd_dict["sim_params"]["trace_miss_opts"] = "Allow"
    # sd = fastsim.SimDrive.from_dict(sd_dict)

    try:
        sd.run()
    except Exception as e:
        print("FAILURE:", sd.veh.name, e)

# %%
# Simulate FASTSim 2
for sd in f2_all_sds:
    print(sd.veh.scenario_name)
    sd.sim_drive()

# %%
