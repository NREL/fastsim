"""Test getting resource lists via Rust API"""
import unittest

import fastsim as fsim
from fastsim import cycle, vehicle

class TestListResources(unittest.TestCase):
    def test_list_resources_for_cycle(self):
        "check if list_resources works for RustCycle"
        c = cycle.Cycle.from_dict({
            "cycSecs": [0.0, 1.0],
            "cycMps": [0.0, 0.0]})
        rc = c.to_rust()
        resources = rc.list_resources()
        self.assertTrue(len(resources) > 0)

    def test_list_resources_for_vehicles(self):
        "check if list_resources works for RustVehicle"
        rv = vehicle.Vehicle.from_vehdb(1).to_rust()
        resources = rv.list_resources()
        self.assertTrue(len(resources) == 1)
