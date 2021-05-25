"""Test suite for cycle instantiation and manipulation."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from fastsim import vehicle


class TestVehicle(unittest.TestCase):
    def test_equal(self):
        """Verify that a copied VehicleJit and identical Vehicle 
        are equal"""

        veh = vehicle.Vehicle(1)
        veh_jit = veh.get_numba_veh()
        veh_jit_copy = vehicle.copy_vehicle(veh_jit)
        self.assertTrue(vehicle.veh_equal(veh_jit, veh_jit_copy))
