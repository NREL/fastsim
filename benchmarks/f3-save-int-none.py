import fastsim as fsim
from memory_profiler import profile

@profile
def build_and_run_sim_drive():
    veh = fsim.Vehicle.from_file(
        # TODO: figure out why `str` is needed here
        str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
    )
    veh.save_interval = None
    cyc = fsim.Cycle.from_resource("cycles/udds.csv")
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()

if __name__ == "__main__":
    build_and_run_sim_drive()

# `python -m memory_profiler f3-save-int-none.py` outputs:
# Filename: f3-save-int-none.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
     # 4     60.6 MiB     60.6 MiB           1   @profile
     # 5                                         def build_and_run_sim_drive():
     # 6     61.2 MiB      0.6 MiB           2       veh = fsim.Vehicle.from_file(
     # 7     60.6 MiB      0.0 MiB           1           str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
     # 8                                             )
     # 9     61.2 MiB      0.0 MiB           1       veh.save_interval = None
    # 10     61.4 MiB      0.2 MiB           1       cyc = fsim.Cycle.from_resource("cycles/udds.csv")
    # 11     61.5 MiB      0.1 MiB           1       sd = fsim.SimDrive(veh, cyc)
    # 12     61.5 MiB      0.0 MiB           1       sd.walk()
