# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
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
print(f"fastsim-3 `sd.walk()` elapsed time: {t1-t0:.2e} s")

sd2 = sd.to_fastsim2()
t0 = time.perf_counter()
sd2.sim_drive()
t1 = time.perf_counter()
print(f"fastsim-2 `sd.walk()` elapsed time: {t1-t0:.2e} s")

if SHOW_PLOTS:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

    ax[0].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.fc.history.pwr_out_watts) / 1e3,
        label="f3: FC out",
    )
    ax[0].plot(
        np.array(sd2.cyc.time_s.tolist())[::veh.save_interval],
        np.array(sd2.fc_kw_out_ach.tolist()),
        label="f2: FC out",
    )
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()

    ax[-1].plot(
        np.array(sd.cyc.time_seconds)[::veh.save_interval],
        np.array(sd.veh.history.speed_ach_meters_per_second),
        label="achieved",
    )
    ax[-1].plot(
        np.array(sd.cyc.time_seconds),
        np.array(sd.cyc.speed_meters_per_second),
        label="prescribed",
    )
    ax[-1].legend()
    ax[-1].set_xlabel("Time [s]")
    ax[-1].set_ylabel("Speed [m/s]")
    plt.show()
