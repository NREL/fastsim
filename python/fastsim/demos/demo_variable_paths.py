"""
Script demonstrating how to use variable_path_list() and history_path_list()
demos to find the paths to variables within fastsim classes.
"""
import fastsim as fsim
print("TODO: This test needs to be fixed.")

# # load 2012 Ford Fusion from file
# veh = fsim.Vehicle.from_file(
#     str(fsim.package_root() / "../../tests/assets/2016_TOYOTA_Prius_Two.yaml")
# )

# # Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
# fsim.set_param_from_path(veh, "save_interval", 1)

# # load cycle from file
# cyc = fsim.Cycle.from_resource("udds.csv")

# # instantiate `SimDrive` simulation object
# sd = fsim.SimDrive(veh, cyc)

# # print out all subpaths for variables in SimDrive
# print("List of variable paths for SimDrive:\n", "\n".join(sd.variable_path_list()))
# print("\n")

# # print out all subpaths for history variables in SimDrive
# print("List of history variable paths for SimDrive:\n", "\n".join(sd.history_path_list()))
