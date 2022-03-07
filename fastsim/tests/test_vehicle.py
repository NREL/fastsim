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
        veh = vehicle.Vehicle(1, verbose=False)
        veh_copy = vehicle.copy_vehicle(veh)
        self.assertTrue(vehicle.veh_equal(veh, veh_copy))

    def test_properties(self):
        """Verify that some of the property variables are working as expected."""

        print(f"Running {type(self)}.test_properties.")
        veh = vehicle.Vehicle(10, verbose=False)
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcPeakEff = 0.85
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcPeakEff += 0.05
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))
        veh.mcFullEffArray *= 1.05
        veh.mcEffArray *= 1.05
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcEffArray))
        self.assertEqual(veh.mcPeakEff, np.max(veh.mcFullEffArray))

    def test_set_dependents(self):
        veh = vehicle.Vehicle(1)
        veh.set_dependents()

    def test_file_overrides(self):
        veh = vehicle.Vehicle('test_overrides')
        self.assertAlmostEqual(veh.mcPeakEff, 0.2, 3)
        self.assertAlmostEqual(veh.fcPeakEff, 0.9, 3)
        with self.assertRaises(AssertionError):
            vehicle.Vehicle('fail_overrides')

if __name__ == '__main__':
    from fastsim import vehicle 
    veh = vehicle.Vehicle(1)