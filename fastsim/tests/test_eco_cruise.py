"""
Test the eco-cruise feature in FASTSim
"""
import unittest

import fastsim as fs


class TestEcoCruise(unittest.TestCase):
    def test_that_eco_cruise_interface_works(self):
        cyc = fs.cycle.Cycle.from_file('udds')
        veh = fs.vehicle.Vehicle.from_vehdb(1)
        sd = fs.simdrive.SimDrive(cyc, veh)
        params = sd.sim_params
        params.eco_cruise_allow = True
        params.eco_cruise_blend_factor = 0.0
        sd.sim_params = params
        sd.sim_drive()
