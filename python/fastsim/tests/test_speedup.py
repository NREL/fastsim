import time
import numpy as np
import fastsim as fsim
import os

n_iters = 5
run_tests = os.environ.get("RUN_SPEEDUP", "false").lower() == "true"

if run_tests:

    def test_hev_speedup():
        # minimum allowed f3 / f2 speed ratio
        min_speed_ratio_si_none = 2
        min_speed_ratio_si_1 = 1.4

        # load 2016 Toyota Prius Two from file
        veh = fsim.Vehicle.from_resource("2016_TOYOTA_Prius_Two.yaml")

        # Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
        veh.set_save_interval(1)

        # load cycle from file
        cyc = fsim.Cycle.from_resource("udds.csv")

        # instantiate `SimDrive` simulation object
        sd0 = fsim.SimDrive(veh, cyc)

        t_fsim2_list = []
        t_fsim3_list = []
        t_fsim3_no_save_list = []

        for i in range(n_iters):
            sd = sd0.copy()
            # simulation start time
            t0 = time.perf_counter()
            sd.walk()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim3_list.append(t1 - t0)

            sd_no_save = sd0.copy()
            sd_no_save.set_save_interval(None)
            # simulation start time
            t0 = time.perf_counter()
            sd_no_save.walk()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim3_no_save_list.append(t1 - t0)

            sd2 = sd0.to_fastsim2()
            # simulation start time
            t0 = time.perf_counter()
            sd2.sim_drive()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim2_list.append(t1 - t0)

        t_fsim2_mean = np.mean(t_fsim2_list)
        t_fsim3_mean = np.mean(t_fsim3_list)
        t_fsim3_no_save_mean = np.mean(t_fsim3_no_save_list)
        t_fsim2_median = np.median(t_fsim2_list)
        t_fsim3_median = np.median(t_fsim3_list)
        t_fsim3_no_save_median = np.median(t_fsim3_no_save_list)

        assert t_fsim2_mean / t_fsim3_mean > min_speed_ratio_si_1, (
            f"`min_speed_ratio_si_1`: {min_speed_ratio_si_1:.3G}, mean achieved ratio: {(t_fsim2_mean / t_fsim3_mean):.3G}"
        )
        assert t_fsim2_median / t_fsim3_median > min_speed_ratio_si_1, (
            f"`min_speed_ratio_si_1`: {min_speed_ratio_si_1:.3G}, median achieved ratio: {(t_fsim2_median / t_fsim3_median):.3G}"
        )

        assert t_fsim2_mean / t_fsim3_no_save_mean > min_speed_ratio_si_none, (
            f"`min_speed_ratio_si_none`: {min_speed_ratio_si_none:.3G}, mean achieved ratio: {(t_fsim2_mean / t_fsim3_no_save_mean):.3G}"
        )
        assert t_fsim2_median / t_fsim3_no_save_median > min_speed_ratio_si_none, (
            f"`min_speed_ratio_si_none`: {min_speed_ratio_si_none:.3G}, median achieved ratio: {(t_fsim2_median / t_fsim3_no_save_median):.3G}"
        )

    def test_conv_speedup():
        # minimum allowed f3 / f2 speed ratio
        # there is some wiggle room on these but we're trying to get 10x speedup
        # relative to fastsim-2 with `save_interval` of `None`
        min_speed_ratio_si_none = 3.6
        min_speed_ratio_si_1 = 2.0

        # load 2016 Toyota Prius Two from file
        veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")

        # Set `save_interval` at vehicle level -- cascades to all sub-components with time-varying states
        veh.set_save_interval(1)

        # load cycle from file
        cyc = fsim.Cycle.from_resource("udds.csv")

        # instantiate `SimDrive` simulation object
        sd0 = fsim.SimDrive(veh, cyc)

        t_fsim2_list = []
        t_fsim3_list = []
        t_fsim3_no_save_list = []

        for i in range(n_iters):
            sd = sd0.copy()
            # simulation start time
            t0 = time.perf_counter()
            sd.walk()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim3_list.append(t1 - t0)

            sd_no_save = sd0.copy()
            sd_no_save.set_save_interval(None)
            # simulation start time
            t0 = time.perf_counter()
            sd_no_save.walk()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim3_no_save_list.append(t1 - t0)

            sd2 = sd0.to_fastsim2()
            # simulation start time
            t0 = time.perf_counter()
            sd2.sim_drive()
            # simulation end time
            t1 = time.perf_counter()
            if i > 1:
                t_fsim2_list.append(t1 - t0)

        t_fsim2_mean = np.mean(t_fsim2_list)
        t_fsim3_mean = np.mean(t_fsim3_list)
        t_fsim3_no_save_mean = np.mean(t_fsim3_no_save_list)
        t_fsim2_median = np.median(t_fsim2_list)
        t_fsim3_median = np.median(t_fsim3_list)
        t_fsim3_no_save_median = np.median(t_fsim3_no_save_list)

        assert t_fsim2_mean / t_fsim3_mean > min_speed_ratio_si_1, (
            f"`min_speed_ratio_si_1`: {min_speed_ratio_si_1:.3G}, achieved ratio: {(t_fsim2_mean / t_fsim3_mean):.3G}"
        )
        assert t_fsim2_median / t_fsim3_median > min_speed_ratio_si_1, (
            f"`min_speed_ratio_si_1`: {min_speed_ratio_si_1:.3G}, achieved ratio: {(t_fsim2_median / t_fsim3_median):.3G}"
        )

        assert t_fsim2_mean / t_fsim3_no_save_mean > min_speed_ratio_si_none, (
            f"`min_speed_ratio_si_none`: {min_speed_ratio_si_none:.3G}, achieved ratio: {(t_fsim2_mean / t_fsim3_no_save_mean):.3G}"
        )
        assert t_fsim2_median / t_fsim3_no_save_median > min_speed_ratio_si_none, (
            f"`min_speed_ratio_si_none`: {min_speed_ratio_si_none:.3G}, achieved ratio: {(t_fsim2_median / t_fsim3_no_save_median):.3G}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
