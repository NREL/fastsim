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

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(
    # TODO: figure out why the slice is needed
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.cyc.time_seconds.tolist()[1:]),
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.veh.fc.history.pwr_out_watts.tolist()) / 1e3,
    label="FC out",
)
ax[0].set_ylabel("Power [kW]")
ax[0].legend()

ax[-1].plot(
    # TODO: figure out why the slice is needed
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.cyc.time_seconds.tolist()[1:]),
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.veh.history.speed_ach_meters_per_second.tolist()),
    label="achieved",
)
ax[-1].plot(
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.cyc.time_seconds.tolist()),
    # TODO: figure out how to make the `tolist` unnecessary
    np.array(sd.cyc.speed_meters_per_second.tolist()),
    label="prescribed",
)
ax[-1].legend()
ax[-1].set_xlabel("Time [s]")
ax[-1].set_ylabel("Speed [m/s]")
# %%
