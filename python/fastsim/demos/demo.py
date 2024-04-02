# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import fastsim as fsim

sns.set_theme()

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "true").lower() == "true"     

veh = fsim.Vehicle.from_file(
    str(fsim.package_root() / "../../rust/tests/assets/2012_Ford_Fusion.yaml")
)
veh.save_interval = 1
cyc = fsim.Cycle.from_resource("cycles/udds.csv")
sd = fsim.SimDrive(veh, cyc)

t0 = time.perf_counter()
sd.walk()
t1 = time.perf_counter()

sd2 = sd.to_fastsim2()

print(f"`sd.walk()` elapsed time: {t1-t0:.2e} s")

# TODO: provide `to_fastsim2` returning fastsim-2 `RustSimdrive` with `to_json` method
# TODO: use above to create dictionary for comparison

if SHOW_PLOTS:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 5))

    # TODO: propagate all the below TODO comments everyhwere else in this file
    ax[0].plot(
        # TODO: figure out why the slice is needed
        # TODO: figure out how to make the `tolist` unnecessary
        np.array(sd.cyc.time_seconds.tolist()[1::veh.save_interval]),
        # TODO: figure out how to make the `tolist` unnecessary
        np.array(sd.veh.fc.history.pwr_out_watts.tolist()) / 1e3,
        label="FC out",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[1].plot(
        # TODO: figure out why the slice is needed
        # TODO: figure out how to make the `tolist` unnecessary
        np.array(sd.cyc.time_seconds.tolist()[1::veh.save_interval]),
        # TODO: figure out how to make the `tolist` unnecessary
        np.array(sd.veh.fc.history.pwr_out_watts.tolist()) / \
        np.array(sd.veh.fc.history.pwr_fuel_watts.tolist()),
        label='FC',
    )
    ax[1].legend()
    ax[1].set_ylabel("Efficiency")


    ax[-1].plot(
        np.array(sd.cyc.time_seconds.tolist()[1::veh.save_interval]),
        np.array(sd.veh.history.speed_ach_meters_per_second.tolist()),
        label="achieved",
    )
    ax[-1].plot(
        np.array(sd.cyc.time_seconds.tolist()),
        np.array(sd.cyc.speed_meters_per_second.tolist()),
        label="prescribed",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Speed [m/s]")
    plt.show()
