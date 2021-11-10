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
        veh_jit = veh.get_numba_veh()
        veh_jit_copy = vehicle.copy_vehicle(veh_jit)
        self.assertTrue(vehicle.veh_equal(veh_jit, veh_jit_copy))

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

if __name__ == '__main__':
    from fastsim import vehicle 
    veh = vehicle.Vehicle(1).get_numba_veh()