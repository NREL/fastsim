# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastsim as fsim

sns.set()

veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../rust/tests/assets/2012_Ford_Fusion.yaml")
)
cyc = fsim.Cycle.from_resource("cycles/udds.csv")
sd = fsim.SimDrive(veh, cyc)

sd.walk()
