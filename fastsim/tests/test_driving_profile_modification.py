"""
Tests that check the drive cycle modification functionality.
"""
import unittest

import fastsim


MPH_TO_MPS = 1.0 / fastsim.params.mphPerMps


class TestDrivingProfileModification(unittest.TestCase):
    def setUp(self) -> None:
        # create a trapezoidal trip shape
        trapz = fastsim.cycle.make_cycle(
            [0.0, 10.0, 20.0, 30.0],
            [0.0, 40.0 * MPH_TO_MPS, 40.0 * MPH_TO_MPS, 0.0])
        trapz = fastsim.cycle.resample(trapz, new_dt=1.0)
        self.trapz = fastsim.cycle.Cycle(cyc_dict=trapz)
        self.veh = fastsim.vehicle.Vehicle(5)
        self.sim_drive = fastsim.simdrive.SimDriveClassic(self.trapz, self.veh)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_that_eco_approach_engages(self):
        "Test the standard interface to Eco-Approach for 'free coasting'"
        coast_start_mph = 39.99
        self.assertFalse(self.sim_drive.impose_coast.any(), "All impose_coast starts out False")
        while self.sim_drive.i < len(self.trapz.cycSecs):
            i = self.sim_drive.i
            prev_i = max(0, i-1)
            self.sim_drive.impose_coast[i] = self.sim_drive.impose_coast[prev_i] or self.sim_drive.mphAch[prev_i] >= coast_start_mph
            self.sim_drive.sim_drive_step()
        self.assertFalse(self.sim_drive.impose_coast[0])
        self.assertTrue(self.sim_drive.impose_coast[-1])
        self.assertTrue(self.sim_drive.impose_coast.any(), "Coast should have been imposed over some portion of the cycle")