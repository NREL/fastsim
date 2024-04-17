from fastsim import fastsimrust as fsr
import fastsim as fsim
from memory_profiler import profile

# Note that you'll need a `fastsim-2` environment active for this to work

@profile
def build_and_run_sim_drive():
    veh = fsr.RustVehicle.from_file(
        str(fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml")
    )
    cyc = fsr.RustCycle.from_resource("cycles/udds.csv")
    sd = fsr.RustSimDrive(cyc, veh)
    sd.sim_drive()

if __name__ == "__main__":
    build_and_run_sim_drive()

# `python -m memory_profiler f2.py` outputs:
# Filename: f2.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
     # 5    163.2 MiB    163.2 MiB           1   @profile
     # 6                                         def build_and_run_sim_drive():
     # 7    164.0 MiB      0.8 MiB           2       veh = fsr.RustVehicle.from_file(
     # 8                                                 # TODO: figure out why `str` is needed here
     # 9    163.2 MiB      0.0 MiB           1           str(fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml")
    # 10                                             )
    # 11    164.1 MiB      0.1 MiB           1       cyc = fsr.RustCycle.from_resource("cycles/udds.csv")
    # 12    165.1 MiB      1.0 MiB           1       sd = fsr.RustSimDrive(cyc, veh)
    # 13    165.3 MiB      0.2 MiB           1       sd.sim_drive()
