"""
Script demonstrating how to use variable_path_list() and history_path_list()
demos to find the paths to variables within fastsim classes.
"""
import os
import fastsim as fsim
import polars as pl

# load 2012 Ford Fusion from file
veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../tests/assets/2016_TOYOTA_Prius_Two.yaml")
)

# Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
fsim.set_param_from_path(veh, "save_interval", 1)

# load cycle from file
cyc = fsim.Cycle.from_resource("udds.csv")

# instantiate `SimDrive` simulation object
sd = fsim.SimDrive(veh, cyc)
sd.walk()

# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get("ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = fsim.resources_root() / "demos/demo_variable_paths/"


# print out all subpaths for variables in SimDrive
print("List of variable paths for SimDrive:" + "\n".join(sd.variable_path_list()))
print("\n")
if ENABLE_ASSERTS:
    with open(ref_dir / "variable_path_list_expected.txt", 'r') as f:
        variable_path_list_expected = [line.strip() for line in f.readlines()]
        assert variable_path_list_expected == sd.variable_path_list()
if ENABLE_REF_OVERRIDE:
    with open(ref_dir / "variable_path_list_expected.txt", 'w') as f:
        for line in sd.variable_path_list():
            f.write(line + "\n")

# print out all subpaths for history variables in SimDrive
print("List of history variable paths for SimDrive:" +  "\n".join(sd.history_path_list()))
print("\n")

# print results as dataframe
print("Results as dataframe:\n", sd.to_dataframe(), sep="")
if ENABLE_ASSERTS:
    to_dataframe_expected = pl.scan_csv(ref_dir / "to_dataframe_expected.csv")
if ENABLE_REF_OVERRIDE:
    df:pl.DataFrame = sd.to_dataframe().lazy().collect()
    df.write_csv(ref_dir / "to_dataframe_expected.csv")

