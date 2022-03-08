"""Test suite for cycle instantiation and manipulation."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import vehicle


class TestVehicle(unittest.TestCase):
    def test_equal(self):
        """Verify that a copied VehicleJit and identically instantiated Vehicle are equal."""
        
        print(f"Running {type(self)}.test_equal.")
        veh = vehicle.Vehicle.from_vehdb(1, verbose=False)
        veh_copy = vehicle.copy_vehicle(veh)
        self.assertTrue(vehicle.veh_equal(veh, veh_copy))

    def test_properties(self):
        """Verify that some of the property variables are working as expected."""

        print(f"Running {type(self)}.test_properties.")
        veh = vehicle.Vehicle.from_vehdb(10, verbose=False)
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
        veh.mc_peak_eff = 0.85
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
        veh.mc_peak_eff += 0.05
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))
        veh.mc_full_eff_array *= 1.05
        veh.mc_eff_array *= 1.05
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_eff_array))
        self.assertEqual(veh.mc_peak_eff, np.max(veh.mc_full_eff_array))

    def test_set_dependents(self):
        veh = vehicle.Vehicle.from_vehdb(1)
        veh.set_dependents()

    def test_file_overrides(self):
        veh = vehicle.Vehicle.from_file('test_overrides')
        self.assertAlmostEqual(veh.mc_peak_eff, 0.2, 3)
        self.assertAlmostEqual(veh.fc_peak_eff, 0.9, 3)
        with self.assertRaises(AssertionError):
            vehicle.Vehicle.from_file('fail_overrides')

if __name__ == '__main__':
    from fastsim import vehicle 
    veh = vehicle.Vehicle.from_vehdb(1)