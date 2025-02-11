#!../fastsim-3-venv/bin/python
import fastsim as fsim
from memory_profiler import profile

@profile(precision=3)
def build_and_run_sim_drive():
    veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    veh.set_save_interval(None)
    cyc = fsim.Cycle.from_resource("udds.csv")
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()

if __name__ == "__main__":
    build_and_run_sim_drive()

# `python -m memory_profiler f3-save-int-none.py` outputs:
# Filename: /Users/cbaker2/Documents/GitHub/fastsim-3/benchmarks/./f3-save-int-none.py
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#      5   61.203 MiB   61.203 MiB           1   @profile(precision=3)
#      6                                         def build_and_run_sim_drive():
#      7   61.766 MiB    0.562 MiB           2       veh = fsim.Vehicle.from_file(
#      8                                                 
#      9   61.203 MiB    0.000 MiB           1           fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml"
#     10                                             )
#     11   61.766 MiB    0.000 MiB           1       veh.save_interval = None
#     12   61.938 MiB    0.172 MiB           1       cyc = fsim.Cycle.from_resource("udds.csv")
#     13   62.031 MiB    0.094 MiB           1       sd = fsim.SimDrive(veh, cyc)
#     14   62.031 MiB    0.000 MiB           1       sd.walk()
