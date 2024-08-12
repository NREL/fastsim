#!../../fastsim/fastsim-venv/bin/python
# above path assumes `fastsim` repo with `fastsim-2`
# checked out and built is in parallel with this repo
from fastsim import fastsimrust as fsr
import fastsim as fsim
from memory_profiler import profile

# Note that you'll need a `fastsim-2` environment active for this to work

@profile(precision=3)
def build_and_run_sim_drive():
    veh = fsr.RustVehicle.from_file(
        str(fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml")
    )
    cyc = fsr.RustCycle.from_resource("udds.csv")
    sd = fsr.RustSimDrive(cyc, veh)
    sd.sim_drive()

if __name__ == "__main__":
    build_and_run_sim_drive()

# `python -m memory_profiler f2.py` outputs:
# Filename: /Users/cbaker2/Documents/GitHub/fastsim-3/benchmarks/./f2.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
    # 10  163.438 MiB  163.438 MiB           1   @profile(precision=3)
    # 11                                         def build_and_run_sim_drive():
    # 12  164.234 MiB    0.797 MiB           2       veh = fsr.RustVehicle.from_file(
    # 13  163.438 MiB    0.000 MiB           1           str(fsim.package_root() / "resources/vehdb/2012_Ford_Fusion.yaml")
    # 14                                             )
    # 15  164.375 MiB    0.141 MiB           1       cyc = fsr.RustCycle.from_resource("udds.csv")
    # 16  165.453 MiB    1.078 MiB           1       sd = fsr.RustSimDrive(cyc, veh)
    # 17  165.656 MiB    0.203 MiB           1       sd.sim_drive()
