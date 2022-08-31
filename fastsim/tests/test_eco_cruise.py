"""
Test the eco-cruise feature in FASTSim
"""
import unittest

import fastsim as fs


class TestEcoCruise(unittest.TestCase):
    def percent_distance_error(self, sd: fs.simdrive.SimDrive) -> float:
        d0 = sd.cyc0.dist_m.sum()
        d = sd.cyc.dist_m.sum()
        return 100.0 * (d - d0) / d0
    
    def cycles_differ(self, sd: fs.simdrive.SimDrive) -> bool:
        if (len(sd.cyc0.time_s) != len(sd.cyc.time_s)):
            return True
        for idx in range(len(sd.cyc0.time_s)):
            if sd.cyc0.time_s[idx] != sd.cyc.time_s[idx]:
                return True
            if abs(sd.cyc0.mps[idx] - sd.cyc.mps[idx]) > 0.01:
                return True
            if abs(sd.cyc0.grade[idx] - sd.cyc.grade[idx]) > 0.01:
                return True
        return False

    def test_that_eco_cruise_interface_works(self):
        cyc = fs.cycle.Cycle.from_file('udds')
        veh = fs.vehicle.Vehicle.from_vehdb(1)
        sd = fs.simdrive.SimDrive(cyc, veh)
        sd.activate_eco_cruise()
        sd.sim_drive()
        self.assertTrue(self.cycles_differ(sd), "Cycles should differ when running with eco-cruise")
        self.assertTrue(abs(self.percent_distance_error(sd)) < 1.0, "Error in distance shouldn't be high; less than 1%")
