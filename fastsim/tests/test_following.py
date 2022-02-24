"""
Tests that check the drive cycle modification functionality.
"""
import unittest

import numpy as np

import fastsim


DO_PLOTS = True


class TestFollowing(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_that_we_have_a_gap_between_us_and_the_lead_vehicle(self):
        "A positive gap should exist between us and the lead vehicle"
        initial_gap_m = 5.0
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 45.0, 55.0, 150.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        )
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        trapz = fastsim.cycle.Cycle(cyc_dict=trapz)
        veh = fastsim.vehicle.Vehicle(5, verbose=False)
        sd = fastsim.simdrive.SimDriveClassic(trapz, veh)
        sd.sim_params.follow_allow = True
        sd.sim_params.follow_initial_gap_m = initial_gap_m
        sd.sim_params.verbose = False
        self.assertTrue(sd.sim_params.follow_allow)
        sd.sim_drive()
        self.assertEqual(initial_gap_m, sd.sim_params.follow_initial_gap_m)
        self.assertTrue(sd.sim_params.follow_allow)
        gaps_m = sd.gap_to_lead_vehicle_m
        if DO_PLOTS:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(sd.cyc0.cycSecs, gaps_m, 'k.')
            ax.set_xlabel('Elapsed Time (s)')
            ax.set_ylabel('Gap (m)')
            fig.tight_layout()
            save_file = "test_that_we_have_a_gap_between_us_and_the_lead_vehicle__0.png"
            fig.savefig(save_file, dpi=300)
            plt.close()
        self.assertTrue((gaps_m == 5.0).all())

    
